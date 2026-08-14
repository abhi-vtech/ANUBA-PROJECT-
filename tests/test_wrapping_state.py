"""
tests/test_wrapping_state.py
─────────────────────────────
Unit tests for src.wrapping_state.WrappingStateMachine.

State machine rules (time-based dwell):
  ACTIVE  → CLOSING  when wrapping continuously overlaps hotdog for >= 3 s
  CLOSING → DONE     when hotdog ID disappears after CLOSING
  lost track (ACTIVE disappear without CLOSING) → no DONE event
  done_ids: retired IDs are permanently ignored (no revival)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from dataclasses import dataclass
from src.wrapping_state import (
    WrappingStateMachine,
    _boxes_intersect,
    _hotdog_coverage_ratio,
    STATE_ACTIVE,
    STATE_CLOSING,
    STATE_DONE,
)

# ── Stub Detection ─────────────────────────────────────────────────────────────

@dataclass
class _Det:
    track_id: int
    bbox: tuple
    class_name: str = "hot-dog"
    confidence: float = 0.9


def _hd(tid, x1, y1, x2, y2):
    return _Det(track_id=tid, bbox=(x1, y1, x2, y2), class_name="hot-dog")

def _wr(x1, y1, x2, y2):
    return _Det(track_id=-1, bbox=(x1, y1, x2, y2), class_name="wrapping")


# ── Convenience: simulate N frames of time ────────────────────────────────────

def _feed(sm, n_frames, frame_start, time_start, fps,
          hotdogs, wrappings):
    """
    Feed `n_frames` frames to the state machine.
    Returns all events collected across those frames.
    """
    events = []
    for i in range(n_frames):
        fi = frame_start + i
        ti = time_start + i / fps
        ev = sm.update(fi, ti, hotdogs, wrappings)
        events.extend(ev)
    return events


# ── Geometry helpers ───────────────────────────────────────────────────────────

class TestGeometry:
    def test_intersect_true(self):
        assert _boxes_intersect((0, 0, 100, 100), (50, 50, 150, 150))

    def test_intersect_false_no_overlap(self):
        assert not _boxes_intersect((0, 0, 50, 50), (100, 100, 200, 200))

    def test_intersect_edge_touching_not_overlap(self):
        # Boxes that only touch at the edge are NOT overlapping (< not <=)
        assert not _boxes_intersect((0, 0, 50, 50), (50, 0, 100, 50))

    def test_coverage_full(self):
        assert _hotdog_coverage_ratio((10, 10, 110, 110), (0, 0, 200, 200)) == pytest.approx(1.0)

    def test_coverage_none(self):
        assert _hotdog_coverage_ratio((0, 0, 50, 50), (100, 100, 200, 200)) == pytest.approx(0.0)


# ── Core state-machine tests ───────────────────────────────────────────────────

class TestWrappingStateMachine:
    FPS = 30.0
    DWELL = 3.0   # default dwell threshold

    def _sm(self, dwell=3.0, gap=1.0, min_closing_frames=100, done_delay_s=0.0, min_coverage_ratio=0.25):
        return WrappingStateMachine(
            wrapping_dwell_s=dwell,
            wrapping_gap_reset_s=gap,
            min_closing_frames=min_closing_frames,
            done_delay_s=done_delay_s,
            min_coverage_ratio=min_coverage_ratio,
        )

    # ── No transition without wrapping ────────────────────────────────────────

    def test_no_closing_without_wrapping(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        # Run 5 seconds without any wrapping
        events = _feed(sm, int(5 * self.FPS), 0, 0.0, self.FPS, hd, [])
        assert events == []
        assert sm.get_state(1) == STATE_ACTIVE

    # ── ACTIVE → CLOSING requires dwell ───────────────────────────────────────

    def test_no_closing_before_dwell(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]   # full overlap
        # 2.9 s — just under threshold
        events = _feed(sm, int(2.9 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        assert not any(e["event"] == "closing" for e in events)
        assert sm.get_state(1) == STATE_ACTIVE

    def test_closing_after_dwell(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        # 3.1 s — just over threshold
        events = _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        closing = [e for e in events if e["event"] == "closing"]
        assert len(closing) == 1
        assert closing[0]["hotdog_tid"] == 1
        assert sm.get_state(1) == STATE_CLOSING

    def test_closing_event_has_dwell_info(self):
        sm = self._sm(dwell=3.0)
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        events = _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        ev = next(e for e in events if e["event"] == "closing")
        assert ev["wrapping_dwell_s"] >= 3.0

    # ── CLOSING is a one-way latch ─────────────────────────────────────────────

    def test_closing_fires_only_once(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        # Run 6 s (double the threshold) — CLOSING should fire exactly once
        events = _feed(sm, int(6 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        closing = [e for e in events if e["event"] == "closing"]
        assert len(closing) == 1

    # ── Dwell timer resets on long gap ────────────────────────────────────────

    def test_dwell_resets_on_long_gap(self):
        sm = self._sm(dwell=3.0, gap=1.0)
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]

        # 2 s with wrapping
        _feed(sm, int(2.0 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        assert sm.get_state(1) == STATE_ACTIVE

        # 2 s gap (> gap_reset_s of 1.0) — should reset dwell
        _feed(sm, int(2.0 * self.FPS), 60, 2.0, self.FPS, hd, [])
        # Timer was reset

        # Another 2 s with wrapping — still under threshold, no CLOSING
        events = _feed(sm, int(2.0 * self.FPS), 120, 4.0, self.FPS, hd, wr)
        assert not any(e["event"] == "closing" for e in events)
        assert sm.get_state(1) == STATE_ACTIVE

    def test_brief_gap_does_not_reset_dwell(self):
        """A gap shorter than gap_reset_s should NOT reset the dwell timer."""
        sm = self._sm(dwell=3.0, gap=1.0)
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]

        # 2 s with wrapping
        _feed(sm, int(2.0 * self.FPS), 0, 0.0, self.FPS, hd, wr)

        # 0.5 s gap (< gap_reset_s) — timer should survive
        _feed(sm, int(0.5 * self.FPS), 60, 2.0, self.FPS, hd, [])

        # 1.5 s more wrapping — total dwell > 3 s → CLOSING
        events = _feed(sm, int(1.5 * self.FPS), 75, 2.5, self.FPS, hd, wr)
        assert any(e["event"] == "closing" for e in events)

    # ── CLOSING → DONE ────────────────────────────────────────────────────────

    def test_done_after_closing_and_disappear(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]

        # Reach CLOSING
        _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        assert sm.get_state(1) == STATE_CLOSING

        # Hotdog disappears at frame 200 (well past 100 frames from closing_frame 0)
        events = sm.update(200, 10.0, [], [])
        done = [e for e in events if e["event"] == "done"]
        assert len(done) == 1
        assert done[0]["hotdog_tid"] == 1
        assert sm.get_state(1) == STATE_DONE
        assert 1 in sm.done_ids

    def test_min_closing_frames_duration(self):
        """CLOSING state must remain active for at least 100 frames before DONE."""
        sm = self._sm(dwell=0.4, min_closing_frames=100, done_delay_s=0.0)
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]

        # Trigger CLOSING at frame 15
        _feed(sm, 15, 0, 0.0, self.FPS, hd, wr)
        assert sm.get_state(1) == STATE_CLOSING

        # Hotdog disappears at frame 35 (20 frames after CLOSING trigger)
        events = sm.update(35, 1.2, [], [])
        assert events == []                      # No DONE event yet!
        assert sm.get_state(1) == STATE_CLOSING  # Stays CLOSING for full 100-frame window

        # At frame 115 (100 frames after closing_frame 15) -> DONE
        events = sm.update(115, 3.8, [], [])
        assert len(events) == 1
        assert events[0]["event"] == "done"
        assert sm.get_state(1) == STATE_DONE

    def test_done_delay_after_disappearance(self):
        """DONE state must wait for done_delay_s after hotdog becomes invisible."""
        sm = self._sm(dwell=0.4, min_closing_frames=0, done_delay_s=3.0)
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]

        # Trigger CLOSING at frame 15 (t=0.5s)
        _feed(sm, 15, 0, 0.0, self.FPS, hd, wr)
        assert sm.get_state(1) == STATE_CLOSING

        # Hotdog disappears at frame 16 (t=0.53s)
        events = sm.update(16, 0.53, [], [])
        assert events == []                      # No DONE event immediately!
        assert sm.get_state(1) == STATE_CLOSING  # Delay period active

        # 2.0 seconds after disappearance (t=2.53s) -> still CLOSING
        events = sm.update(76, 2.53, [], [])
        assert events == []
        assert sm.get_state(1) == STATE_CLOSING

        # 3.1 seconds after disappearance (t=3.63s) -> DONE
        events = sm.update(109, 3.63, [], [])
        assert len(events) == 1
        assert events[0]["event"] == "done"
        assert sm.get_state(1) == STATE_DONE

    def test_done_event_carries_closing_metadata(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        events = sm.update(200, 10.0, [], [])
        done_ev = next(e for e in events if e["event"] == "done")
        assert done_ev["closing_frame"] is not None
        assert done_ev["closing_time"] is not None
        assert done_ev["frame"] == 200
        assert done_ev["timestamp"] == pytest.approx(10.0)

    # ── Lost track (ACTIVE disappear) is NOT marked DONE ─────────────────────

    def test_active_disappear_is_not_done(self):
        sm = self._sm()
        hd = [_hd(2, 0, 0, 100, 100)]
        # Hotdog seen for 1 s with no wrapping, then disappears
        _feed(sm, int(1.0 * self.FPS), 0, 0.0, self.FPS, hd, [])
        events = sm.update(50, 5.0, [], [])   # disappear while ACTIVE
        assert not any(e["event"] == "done" for e in events)
        assert sm.get_state(2) == STATE_ACTIVE   # still ACTIVE (lost track)
        assert 2 not in sm.done_ids

    # ── No-revival rule ───────────────────────────────────────────────────────

    def test_done_id_ignored_if_tracker_reuses(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        sm.update(200, 10.0, [], [])   # → DONE
        assert sm.get_state(1) == STATE_DONE

        # Tracker recycles ID 1 — must be silently ignored
        events = sm.update(201, 11.0, hd, wr)
        assert events == []
        assert sm.get_state(1) == STATE_DONE

    # ── Multiple independent hotdog IDs ───────────────────────────────────────

    def test_two_hotdogs_tracked_independently(self):
        sm = self._sm()
        hd1 = _hd(1, 0,   0, 100, 100)
        hd2 = _hd(2, 200, 0, 300, 100)
        # Wrapping only overlaps hotdog 1
        wr1 = _wr(0, 0, 100, 100)

        # 3.1 s — hotdog 1 reaches CLOSING
        events = _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, [hd1, hd2], [wr1])
        closing = [e for e in events if e["event"] == "closing"]
        assert 1 in {e["hotdog_tid"] for e in closing}
        assert 2 not in {e["hotdog_tid"] for e in closing}
        assert sm.get_state(1) == STATE_CLOSING
        assert sm.get_state(2) == STATE_ACTIVE

        # Hotdog 1 disappears → DONE; hotdog 2 stays ACTIVE
        events = sm.update(200, 10.0, [hd2], [])
        done = [e for e in events if e["event"] == "done"]
        assert 1 in {e["hotdog_tid"] for e in done}
        assert 2 not in {e["hotdog_tid"] for e in done}
        assert sm.get_state(1) == STATE_DONE
        assert sm.get_state(2) == STATE_ACTIVE

    # ── Configurable dwell threshold ──────────────────────────────────────────

    def test_custom_dwell_threshold(self):
        sm = self._sm(dwell=5.0)   # 5 s threshold
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        # 3 s — NOT enough for 5 s threshold
        events = _feed(sm, int(3.0 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        assert not any(e["event"] == "closing" for e in events)
        # 5.1 s total — enough
        events = _feed(sm, int(2.1 * self.FPS), 90, 3.0, self.FPS, hd, wr)
        assert any(e["event"] == "closing" for e in events)

    # ── Partial overlap is sufficient ─────────────────────────────────────────

    def test_partial_overlap_starts_dwell(self):
        """Any intersection (not just high coverage) should start the timer."""
        sm = self._sm(min_coverage_ratio=0.0)
        hd = [_hd(1, 0, 0, 100, 100)]
        # Wrapping only overlaps 10x10 pixels (10% coverage)
        wr = [_wr(90, 90, 200, 200)]
        events = _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        assert any(e["event"] == "closing" for e in events)

    def test_non_overlapping_wrapping_does_not_start_dwell(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(200, 200, 300, 300)]   # completely separate
        events = _feed(sm, int(4.0 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        assert not any(e["event"] == "closing" for e in events)

    # ── get_dwell_elapsed ─────────────────────────────────────────────────────

    def test_get_dwell_elapsed(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        _feed(sm, int(2.0 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        elapsed = sm.get_dwell_elapsed(1, 2.0)
        # dwell_start is set at t=0.0; get_dwell_elapsed(1, 2.0) = 2.0 - 0.0
        assert elapsed >= 1.9   # at least ~2 s elapsed

    # ── reset / full_reset ────────────────────────────────────────────────────

    def test_reset_clears_states_keeps_done_ids(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        sm.update(200, 10.0, [], [])    # → DONE
        assert 1 in sm.done_ids
        sm.reset()
        assert sm._states == {}
        assert 1 in sm.done_ids   # preserved

    def test_full_reset_clears_everything(self):
        sm = self._sm()
        hd = [_hd(1, 0, 0, 100, 100)]
        wr = [_wr(0, 0, 100, 100)]
        _feed(sm, int(3.1 * self.FPS), 0, 0.0, self.FPS, hd, wr)
        sm.update(200, 10.0, [], [])
        sm.full_reset()
        assert sm._states == {}
        assert sm.done_ids == set()
