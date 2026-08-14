"""
tests/test_wrap_station_occlusion.py
─────────────────────────────────────
Wrap-station occlusion bug regression tests.

Scenarios covered:
  1. test_occlusion_id_continuity
     Hotdog enters wrap zone, is fully occluded (no detection), re-appears.
     Assert single monotonic ID throughout.

  2. test_done_track_never_reactivates
     Advance a track to DONE via done_ids; feed a detection at the same
     position the next frame.  Assert no reactivation in HotdogTracker or
     WrappingStateMachine.

  3. test_stalled_watchdog_merge
     Hotdog coasts → retires (timeout) → new YOLO detection appears at same
     position inside wrap zone. Assert stall watchdog auto-merges to old ID.

  4. test_wrap_zone_track_immediate_registration
     Track spawned inside assembly ROI polygon. Assert WSM _states has an
     entry for it after register_wrap_zone_track() is called.

  5. test_dwell_transfer_on_merge
     Partial dwell accumulated on track A; merge_wrap_zone_fragment() +
     transfer_dwell_state() fires; assert track B inherits elapsed dwell and
     reaches CLOSING sooner.

  6. test_fragmented_ids_none_reach_done_then_fix
     Regression: 4 successive IDs in the wrap zone accumulate <dwell
     individually.  With the watchdog + dwell-transfer fix active, assert that
     exactly one DONE event fires for the underlying physical hotdog.

  7. test_appearance_reid_in_wrap_zone
     Histogram-based re-ID: track coasts, histogram stored; next detection
     returns with zero IoU and out of spatial_lock_radius but identical
     colour; assert it re-links to the original ID.
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from typing import Optional, Tuple

from src.hotdog_tracker import (
    HotdogTracker,
    _compute_bgr_histogram,
    _bhattacharyya_distance,
)
from src.wrapping_state import (
    WrappingStateMachine,
    STATE_ACTIVE,
    STATE_CLOSING,
    STATE_DONE,
)
from src.schemas import Detection


# ── Helpers ────────────────────────────────────────────────────────────────────

# Assembly zone polygon in PIXEL space (matching the bbox coordinate space).
# Derived from zones.json zone_06 "assembly" normalised coords × (1920, 1080).
# IMPORTANT: wrap_zone_poly must be in the same space as bboxes (pixel space).
FRAME_W, FRAME_H = 1920, 1080
ASSEMBLY_POLY_PX = [
    (int(0.89375 * FRAME_W), int(0.47037 * FRAME_H)),   # (1716, 508)
    (int(0.89635 * FRAME_W), int(0.66667 * FRAME_H)),   # (1721, 720)
    (int(0.45521 * FRAME_W), int(0.64167 * FRAME_H)),   # (874,  693)
    (int(0.46146 * FRAME_W), int(0.44722 * FRAME_H)),   # (886,  483)
]

# A centroid that is comfortably inside the assembly polygon (pixel space).
# The polygon spans roughly x=[874, 1721], y=[483, 720].
# Choose a point well inside: cx=1300, cy=600.
WRAP_CX_PX = 1300
WRAP_CY_PX = 600

# Bbox centred on the wrap zone pixel position
WRAP_BBOX: Tuple[int, int, int, int] = (
    WRAP_CX_PX - 40, WRAP_CY_PX - 20,
    WRAP_CX_PX + 40, WRAP_CY_PX + 20,
)


def _det(bbox: Tuple, class_name: str = "hot-dog", track_id: int = 1) -> Detection:
    return Detection(track_id=track_id, bbox=bbox, class_name=class_name, confidence=0.9)


def _wr(bbox: Tuple) -> Detection:
    """Wrapping detection."""
    return Detection(track_id=-1, bbox=bbox, class_name="wrapping", confidence=0.9)


def _make_tracker(
    orphan_timeout_s: float = 10.0,
    wrap_orphan_timeout_s: float = 4.0,
    wrap_stall_watchdog_s: float = 5.0,
    appearance_reid_enabled: bool = False,
    **kwargs,
) -> HotdogTracker:
    """Create a HotdogTracker wired to the assembly zone polygon (pixel space)."""
    return HotdogTracker(
        orphan_timeout_s=orphan_timeout_s,
        spatial_lock_radius=300.0,
        iou_threshold=0.30,
        wrap_zone_poly=ASSEMBLY_POLY_PX,
        wrap_spatial_lock_radius=350.0,
        wrap_orphan_timeout_s=wrap_orphan_timeout_s,
        wrap_stall_watchdog_s=wrap_stall_watchdog_s,
        appearance_reid_enabled=appearance_reid_enabled,
        **kwargs,
    )


def _make_wsm(dwell: float = 0.4, gap: float = 2.0) -> WrappingStateMachine:
    return WrappingStateMachine(
        wrapping_dwell_s=dwell,
        wrapping_gap_reset_s=gap,
        min_closing_frames=0,
        done_delay_s=0.0,
    )


# ── Test 1: ID continuity through full occlusion ───────────────────────────────

class TestOcclusionIdContinuity:
    def test_occlusion_id_continuity(self):
        """
        Hotdog detected in wrap zone -> hand occludes it (no detection for 3 s) ->
        hotdog re-appears at same position.  Expect the single original track ID
        to survive throughout (no new ID spawned).
        """
        tracker = _make_tracker(
            orphan_timeout_s=10.0,
            wrap_orphan_timeout_s=4.0,
        )

        # Frame 0: initial detection
        d0 = _det(WRAP_BBOX, track_id=1)
        tracker.update([d0], current_time=0.0)
        log = tracker.get_hotdog_log()
        assert len(log) == 1
        original_tid = next(iter(log))

        # 1–3 s: no detection (hand occludes) — within 4 s wrap timeout
        tracker.update([], current_time=1.0)
        tracker.update([], current_time=2.0)
        tracker.update([], current_time=3.0)

        log = tracker.get_hotdog_log()
        assert original_tid in log
        assert log[original_tid]["is_coasting"] is True

        # Re-appears at same position with different YOLO track_id
        d1 = _det(WRAP_BBOX, track_id=5)
        tracker.update([d1], current_time=3.1)

        log = tracker.get_hotdog_log()
        active = [tid for tid, r in log.items() if r["active"]]
        assert len(active) == 1, f"Expected 1 active track, got {active}"
        assert active[0] == original_tid, (
            f"Expected original tid={original_tid}, got {active[0]}"
        )


# ── Test 2: DONE track never reactivates ──────────────────────────────────────

class TestDoneTrackNeverReactivates:
    def test_done_track_never_reactivates_in_tracker(self):
        """
        Mark track as DONE via done_ids; a detection at the same position must
        NOT reactivate it.
        """
        tracker = _make_tracker()

        d = _det(WRAP_BBOX, track_id=1)
        tracker.update([d], current_time=0.0)
        log = tracker.get_hotdog_log()
        tid = next(iter(log))

        tracker.update([], current_time=0.1, done_ids={tid})
        tracker.update([_det(WRAP_BBOX, track_id=2)], current_time=0.2, done_ids={tid})

        log = tracker.get_hotdog_log()
        assert log[tid]["retired"] is True
        assert tid in tracker._permanent_done_ids

    def test_done_track_trail_disappears_immediately(self):
        """
        When a track ID is marked DONE, its trail buffer in main must be purged
        immediately so tracking lines disappear instantly.
        """
        from src.main import _update_trail_buffer, _TRAIL_BUFFER, _TRAIL_KEY, draw_annotations

        _TRAIL_BUFFER.clear()
        _TRAIL_KEY.clear()

        det = _det(WRAP_BBOX, track_id=10)
        draw_annotations._done_ids = set()
        draw_annotations._detector_id_map = {}
        _update_trail_buffer([det], current_time=1.0)
        assert 10 in _TRAIL_BUFFER

        # Mark track 10 as DONE
        draw_annotations._done_ids = {10}
        _update_trail_buffer([det], current_time=1.1)

        # Trail buffer must be purged immediately
        assert 10 not in _TRAIL_BUFFER

    def test_done_track_never_reactivates_in_wsm(self):
        """
        WSM: once a tid is in done_ids, subsequent update() calls with that
        tid in hotdog_detections must be silently ignored.
        """
        sm = _make_wsm()

        hd = _det(WRAP_BBOX, track_id=1)
        wr = _wr(WRAP_BBOX)

        for i in range(20):
            sm.update(i, i / 30.0, [hd], [wr])
        assert sm.get_state(1) == STATE_CLOSING

        sm.update(100, 10.0, [], [])
        assert sm.get_state(1) == STATE_DONE
        assert 1 in sm.done_ids

        events = sm.update(101, 11.0, [hd], [wr])
        assert sm.get_state(1) == STATE_DONE
        assert not any(e["event"] == "closing" for e in events)


# ── Test 3: Stalled-track watchdog auto-merge ─────────────────────────────────

class TestStalledWatchdogMerge:
    def test_stalled_watchdog_auto_merges_in_wrap_zone(self):
        """
        Hotdog in wrap zone retires; new detection at same position within
        watchdog window is merged (no new ID spawned).
        """
        tracker = _make_tracker(
            wrap_orphan_timeout_s=0.2,
            wrap_stall_watchdog_s=5.0,
        )

        d0 = _det(WRAP_BBOX, track_id=1)
        tracker.update([d0], current_time=0.0)
        log = tracker.get_hotdog_log()
        original_tid = next(iter(log))

        # Let it retire
        tracker.update([], current_time=0.5)
        log = tracker.get_hotdog_log()
        assert log[original_tid]["retired"] is True

        # New YOLO track within watchdog window
        d1 = _det(WRAP_BBOX, track_id=99)
        tracker.update([d1], current_time=1.0)

        log = tracker.get_hotdog_log()
        summary = tracker.get_summary()

        assert log[original_tid]["active"] is True
        assert log[original_tid]["retired"] is False
        assert len(log) == 1, f"Expected 1 track, got {list(log.keys())}"
        assert summary["regression_metrics"]["wrap_zone_merges"] >= 1


# ── Test 4: Immediate wrap-zone registration in WSM ───────────────────────────

class TestWrapZoneImmediateRegistration:
    def test_register_wrap_zone_track_initialises_state(self):
        sm = _make_wsm()
        assert 42 not in sm._states

        sm.register_wrap_zone_track(42, current_time=0.0)

        assert 42 in sm._states
        state = sm._states[42]
        assert state.hotdog_tid == 42
        assert state.state == STATE_ACTIVE
        assert state.wrapping_dwell_start is None

    def test_register_is_noop_if_already_registered(self):
        sm = _make_wsm()
        sm.register_wrap_zone_track(5, current_time=0.0)
        sm._states[5].wrapping_dwell_start = 99.0

        sm.register_wrap_zone_track(5, current_time=1.0)
        assert sm._states[5].wrapping_dwell_start == pytest.approx(99.0)

    def test_register_noop_for_done_tid(self):
        sm = _make_wsm()
        sm.done_ids.add(7)
        sm.register_wrap_zone_track(7, current_time=0.0)
        assert 7 not in sm._states


# ── Test 5: Dwell transfer on merge ───────────────────────────────────────────

class TestDwellTransferOnMerge:
    def test_dwell_transferred_and_closing_fires_sooner(self):
        """
        Track A accumulates 1.5 s dwell (threshold=2.0 s). After transfer to B,
        B needs only 0.5 s more to reach CLOSING.
        """
        tracker = _make_tracker()
        sm = WrappingStateMachine(
            wrapping_dwell_s=2.0,
            wrapping_gap_reset_s=5.0,
            min_closing_frames=0,
            done_delay_s=0.0,
        )
        FPS = 30.0

        hd_a = _det(WRAP_BBOX, track_id=1)
        tracker.update([hd_a], current_time=0.0)
        log = tracker.get_hotdog_log()
        tid_a = next(iter(log))

        wr = _wr(WRAP_BBOX)
        hd_a_det = _det(WRAP_BBOX, track_id=tid_a)
        for i in range(int(1.5 * FPS)):
            sm.update(i, i / FPS, [hd_a_det], [wr])
        assert sm.get_state(tid_a) == STATE_ACTIVE
        assert sm.get_dwell_elapsed(tid_a, 1.5) >= 1.4

        # Simulate new track B
        hd_b = _det(WRAP_BBOX, track_id=100)
        tracker.update([hd_b], current_time=1.6)
        log = tracker.get_hotdog_log()
        tid_b = max(log.keys())

        sm.register_wrap_zone_track(tid_b, current_time=1.6)
        sm.transfer_dwell_state(from_tid=tid_a, to_tid=tid_b)

        assert sm._states[tid_b].wrapping_dwell_start == pytest.approx(
            sm._states[tid_a].wrapping_dwell_start
        )

        hd_b_det = _det(WRAP_BBOX, track_id=tid_b)
        events = []
        for i in range(int(0.6 * FPS) + 1):
            t = 1.6 + i / FPS
            ev = sm.update(int(1.5 * FPS) + i, t, [hd_b_det], [wr])
            events.extend(ev)

        closing_events = [e for e in events if e["event"] == "closing"]
        assert len(closing_events) >= 1, "Expected CLOSING after dwell transfer"
        assert closing_events[0]["hotdog_tid"] == tid_b


# ── Test 6: Regression — fragmented IDs in wrap zone ─────────────────────────

class TestFragmentedIdsWrapZone:
    def test_watchdog_prevents_fragmentation(self):
        """
        Regression for Issue #2: successive track IDs in wrap zone (hotdog_4→5→6→7).
        With the stall watchdog, the first ID accumulates full dwell → one DONE.
        """
        tracker = _make_tracker(
            wrap_orphan_timeout_s=0.1,
            wrap_stall_watchdog_s=10.0,
        )
        sm = WrappingStateMachine(
            wrapping_dwell_s=0.5,
            wrapping_gap_reset_s=10.0,
            min_closing_frames=0,
            done_delay_s=0.0,
        )

        FPS = 30.0
        wr = _wr(WRAP_BBOX)
        all_events = []
        t = 0.0
        frame = 0

        # Spawn first detection
        d = _det(WRAP_BBOX, track_id=1)
        tracker.update([d], current_time=t)
        log = tracker.get_hotdog_log()
        tid_1 = next(iter(log))
        sm.register_wrap_zone_track(tid_1, current_time=t)

        # Accumulate 0.2 s dwell
        for _ in range(int(0.2 * FPS)):
            t += 1 / FPS
            frame += 1
            tracker.update([_det(WRAP_BBOX, track_id=1)], current_time=t)
            ev = sm.update(frame, t, [_det(WRAP_BBOX, track_id=tid_1)], [wr])
            all_events.extend(ev)

        # Occlusion — retire tid_1
        t += 0.15
        frame += 5
        tracker.update([], current_time=t)

        # New YOLO ID appears (fragmentation scenario)
        t += 0.05
        frame += 2
        tracker.update([_det(WRAP_BBOX, track_id=2)], current_time=t)
        log = tracker.get_hotdog_log()

        assert tid_1 in log, "tid_1 should be active after watchdog merge"
        assert log[tid_1]["active"] is True
        n_active = sum(1 for r in log.values() if r["active"])
        assert n_active == 1, f"Expected 1 active track after merge, got {n_active}"

        # Accumulate remaining dwell on tid_1
        for _ in range(int(0.4 * FPS) + 2):
            t += 1 / FPS
            frame += 1
            tracker.update([_det(WRAP_BBOX, track_id=tid_1)], current_time=t)
            ev = sm.update(frame, t, [_det(WRAP_BBOX, track_id=tid_1)], [wr])
            all_events.extend(ev)

        # Disappear -> DONE
        t += 1.0
        frame += 30
        ev = sm.update(frame, t, [], [])
        all_events.extend(ev)

        done_events = [e for e in all_events if e["event"] == "done"]
        closing_events = [e for e in all_events if e["event"] == "closing"]
        assert len(closing_events) >= 1, "Expected at least one CLOSING event"
        assert len(done_events) == 1, f"Expected exactly 1 DONE, got {done_events}"
        assert done_events[0]["hotdog_tid"] == tid_1


# ── Test 7: Hand-transit in-flight occlusion transfer re-ID ──────────────────

class TestHandTransitReId:
    def test_hand_transit_reid_after_occlusion_and_relocation(self):
        """
        Hotdog is touched by a hand at Station A -> picked up and occluded during
        transit -> placed at distant Station B (> 400 px away). Even if ingredients
        change and spatial lock fails, hand-transit Re-ID links it back to the original ID.
        """
        tracker = _make_tracker(appearance_reid_enabled=True)

        hand_a = Detection(track_id=99, bbox=(WRAP_CX_PX - 20, WRAP_CY_PX - 20, WRAP_CX_PX + 20, WRAP_CY_PX + 20), class_name="hand", confidence=0.9)
        d0 = _det(WRAP_BBOX, track_id=1)
        # Frame 0: Hand touches hotdog at Station A
        tracker.update([d0, hand_a], current_time=0.0)
        log = tracker.get_hotdog_log()
        assert len(log) == 1
        original_tid = next(iter(log))
        assert tracker._records[original_tid].was_hand_carried is True

        # Frame 1: Hand in transit (hotdog occluded)
        hand_mid = Detection(track_id=99, bbox=(WRAP_CX_PX + 200, WRAP_CY_PX, WRAP_CX_PX + 240, WRAP_CY_PX + 40), class_name="hand", confidence=0.9)
        tracker.update([hand_mid], current_time=0.1)
        assert tracker._records[original_tid].is_coasting is True

        # Frame 2: Placed at Station B far away (> 400 px)
        far_bbox = (WRAP_CX_PX + 420, WRAP_CY_PX, WRAP_CX_PX + 500, WRAP_CY_PX + 40)
        hand_b = Detection(track_id=99, bbox=(WRAP_CX_PX + 440, WRAP_CY_PX - 10, WRAP_CX_PX + 480, WRAP_CY_PX + 30), class_name="hand", confidence=0.9)
        d2 = _det(far_bbox, track_id=2)
        tracker.update([d2, hand_b], current_time=0.2)

        log = tracker.get_hotdog_log()
        active = [tid for tid, r in log.items() if r["active"]]
        assert len(active) == 1
        assert active[0] == original_tid, (
            f"Hand-transit re-ID should link back to tid={original_tid}, got {active}"
        )

    def test_hand_transit_restores_correct_carried_id_not_previous_station_id(self):
        """
        Scenario:
        - Hotdog #4 was assembled earlier at Station B (chili area) and retired there.
        - Hotdog #5 is at Station A (prep table), picked up by a hand and occluded during transit.
        - Hand places the carried hotdog at Station B (where #4 was previously retired).
        - Assert that the detection is restored as Hotdog #5 (NOT Hotdog #4).
        """
        tracker = _make_tracker(appearance_reid_enabled=True)

        station_a_bbox = (300, 300, 380, 340)
        station_b_bbox = (800, 300, 880, 340)  # Chili area

        # 1. Hotdog #4 was at Station B at t=0.0
        d_hd4 = _det(station_b_bbox, track_id=4)
        tracker.update([d_hd4], current_time=0.0)
        log = tracker.get_hotdog_log()
        tid_4 = next(iter(log))

        # Hotdog #4 is retired at Station B (absence for 15s)
        tracker.update([], current_time=15.0)
        log = tracker.get_hotdog_log()
        assert log[tid_4]["retired"] is True

        # 2. Hotdog #5 appears at Station A at t=20.0
        d_hd5 = _det(station_a_bbox, track_id=5)
        tracker.update([d_hd5], current_time=20.0)
        log = tracker.get_hotdog_log()
        active = [tid for tid, r in log.items() if r["active"]]
        assert len(active) == 1
        tid_5 = active[0]
        assert tid_5 != tid_4

        # 3. Hand picks up Hotdog #5 at Station A at t=25.0
        hand_pickup = Detection(track_id=99, bbox=(320, 290, 360, 330), class_name="hand", confidence=0.9)
        tracker.update([_det(station_a_bbox, track_id=5), hand_pickup], current_time=25.0)

        # 4. In-flight transit: Hotdog #5 is occluded while hand moves to Station B (t=26.0)
        hand_transit = Detection(track_id=99, bbox=(550, 290, 590, 330), class_name="hand", confidence=0.9)
        tracker.update([hand_transit], current_time=26.0)
        assert tracker._records[tid_5].is_coasting is True

        # 5. Hand places Hotdog #5 at Station B (t=27.0)
    def test_exact_hand_lost_id_recovered_in_assembly_area_no_hijack(self):
        """
        Scenario:
        - Hotdog #4 was on the assembly table and retired.
        - Hotdog #5 completed.
        - Hotdog #6 is being handled at prep table, hand touches it and it gets lost during hand occlusion.
        - Hand transitions to the assembly table and places Hotdog #6 down.
        - Assert: Only Hotdog #6 is recovered! Hotdog #4 must NEVER be recovered.
        """
        tracker = _make_tracker(appearance_reid_enabled=True)

        prep_bbox = (200, 300, 280, 340)
        assembly_bbox = (WRAP_CX_PX - 30, WRAP_CY_PX - 15, WRAP_CX_PX + 30, WRAP_CY_PX + 15)

        # 1. Hotdog #4 on assembly table at t=0.0 and retires at t=15.0
        tracker.update([_det(assembly_bbox, track_id=4)], current_time=0.0)
        tracker.update([], current_time=15.0)
        assert tracker._retired_records[1].retired is True  # Hotdog #4 is tid=1

        # 2. Hotdog #6 at prep table at t=20.0
        tracker.update([_det(prep_bbox, track_id=6)], current_time=20.0)
        log = tracker.get_hotdog_log()
        tid_6 = [tid for tid, r in log.items() if r["active"]][0]
        assert tid_6 == 2  # next monotonic ID is 2 (representing hotdog 6)

        # 3. Hand touches Hotdog #6 at t=25.0
        hand_prep = Detection(track_id=99, bbox=(220, 290, 260, 330), class_name="hand", confidence=0.9)
        tracker.update([_det(prep_bbox, track_id=6), hand_prep], current_time=25.0)

        # 4. Hotdog #6 is occluded by hand during transit (lost while touching hand)
        hand_transit = Detection(track_id=99, bbox=(500, 300, 540, 340), class_name="hand", confidence=0.9)
        tracker.update([hand_transit], current_time=26.0)
        assert 99 in tracker._last_lost_by_hand and tracker._last_lost_by_hand[99]["tid"] == tid_6

        # 5. Hand places the carried hotdog on the assembly table at t=27.0
        hand_assembly = Detection(track_id=99, bbox=(WRAP_CX_PX - 10, WRAP_CY_PX - 20, WRAP_CX_PX + 30, WRAP_CY_PX + 20), class_name="hand", confidence=0.9)
        d_assembly = _det(assembly_bbox, track_id=60)
        tracker.update([d_assembly, hand_assembly], current_time=27.0)

        # 6. Verify: Active ID on assembly table is tid_6 (Hotdog #6), NOT Hotdog #4!
        log = tracker.get_hotdog_log()
        active = [tid for tid, r in log.items() if r["active"]]
        assert len(active) == 1
        assert active[0] == tid_6, f"Expected active ID={tid_6} (Hotdog #6), but got {active[0]}"
        assert log[1]["retired"] is True  # Hotdog #4 stayed retired

    def test_two_hands_simultaneous_different_lost_tracks(self):
        """
        Two hands active simultaneously:
        - Hand A (track_id=101) carries Track 1 (orig at 200, 300)
        - Hand B (track_id=102) carries Track 2 (orig at 600, 300)
        Both go occluded / in transit.
        Hand A places a hotdog at (400, 300) while in contact with Hand A.
        Even if (400, 300) was closer to Track 2, Hand A MUST recover Track 1 (its own carried hotdog)!
        """
        tracker = _make_tracker(appearance_reid_enabled=True)

        hd_1 = _det((200, 300, 260, 340), track_id=1)
        hd_2 = _det((600, 300, 660, 340), track_id=2)
        hand_a = Detection(track_id=101, bbox=(210, 290, 250, 330), class_name="hand", confidence=0.9)
        hand_b = Detection(track_id=102, bbox=(610, 290, 650, 330), class_name="hand", confidence=0.9)

        # Frame 0: Both tracks picked up by their respective hands
        tracker.update([hd_1, hd_2, hand_a, hand_b], current_time=0.0)
        tid_1 = tracker._detector_id_map[1]
        tid_2 = tracker._detector_id_map[2]

        # Frame 1: Both hotdogs occluded during transit
        hand_a_trans = Detection(track_id=101, bbox=(300, 290, 340, 330), class_name="hand", confidence=0.9)
        hand_b_trans = Detection(track_id=102, bbox=(550, 290, 590, 330), class_name="hand", confidence=0.9)
        tracker.update([hand_a_trans, hand_b_trans], current_time=0.1)

        assert 101 in tracker._last_lost_by_hand and tracker._last_lost_by_hand[101]["tid"] == tid_1
        assert 102 in tracker._last_lost_by_hand and tracker._last_lost_by_hand[102]["tid"] == tid_2

        # Frame 2: Hand A places a hotdog at (400, 300)
        placed_hd = _det((400, 300, 460, 340), track_id=50)
        hand_a_place = Detection(track_id=101, bbox=(410, 290, 450, 330), class_name="hand", confidence=0.9)
        tracker.update([placed_hd, hand_a_place, hand_b_trans], current_time=0.2)

        # Hand A placed hotdog must be tid_1, NOT tid_2
        log = tracker.get_hotdog_log()
        assert tracker._records[tid_1].is_coasting is False
        assert tracker._records[tid_1].bbox == (400, 300, 460, 340)
        assert tracker._records[tid_2].is_coasting is True
        assert 101 not in tracker._last_lost_by_hand  # consumed
        assert 102 in tracker._last_lost_by_hand      # still pending

    def test_hand_lost_track_exceeding_plausible_displacement_fails(self):
        """
        A hand-lost track reappears at an impossible distance in 1 frame (> max_plausible_displacement).
        Assert it is NOT force-matched via Pass 3 (does not teleport).
        """
        tracker = _make_tracker(appearance_reid_enabled=True, max_hand_speed_px_per_frame=30.0, base_hand_displacement_px=50.0)

        # Frame 0: Hotdog touching hand at (100, 100)
        hd = _det((100, 100, 160, 140), track_id=1)
        hand = Detection(track_id=101, bbox=(110, 90, 150, 130), class_name="hand", confidence=0.9)
        tracker.update([hd, hand], current_time=0.0)
        tid = tracker._detector_id_map[1]

        # Frame 1: Hotdog occluded by hand
        tracker.update([hand], current_time=0.033)
        assert 101 in tracker._last_lost_by_hand

        # Frame 2: (1 frame elapsed) Hand detection appears at (900, 900) - distance > 1100px!
        # Max allowed for 1 frame is 50 + 1 * 30 = 80px.
        teleport_hd = _det((900, 900, 960, 940), track_id=99)
        teleport_hand = Detection(track_id=101, bbox=(910, 890, 950, 930), class_name="hand", confidence=0.9)
        tracker.update([teleport_hd, teleport_hand], current_time=0.066)

        # Original track should remain coasting (not teleported!)
        assert tracker._records[tid].is_coasting is True
        # The detection at (900, 900) spawned as a new ID instead of teleporting tid 1
        new_active = [k for k, r in tracker._records.items() if not r.is_coasting]
        assert len(new_active) == 1
        assert new_active[0] != tid

    def test_static_rack_hotdog_matches_during_unrelated_hand_transit(self):
        """
        Static hotdog at rack (150, 300) blinks / goes briefly missing while Hand B is in transit
        with another carried hotdog at (700, 300).
        Assert the rack hotdog is correctly rematched via Pass 2 local station lock
        and does NOT spawn a new ID (fixing #4/#5/#6 churn).
        """
        tracker = _make_tracker(appearance_reid_enabled=True)

        rack_bbox = (150, 300, 210, 340)
        carried_bbox = (700, 300, 760, 340)

        # Frame 0: Static hotdog at rack, and carried hotdog with Hand B
        hd_rack = _det(rack_bbox, track_id=1)
        hd_carried = _det(carried_bbox, track_id=2)
        hand_b = Detection(track_id=202, bbox=(710, 290, 750, 330), class_name="hand", confidence=0.9)
        tracker.update([hd_rack, hd_carried, hand_b], current_time=0.0)

        tid_rack = tracker._detector_id_map[1]
        tid_carried = tracker._detector_id_map[2]

        # Frame 1: Both occluded (rack blinks, hand B moves)
        hand_b_mov = Detection(track_id=202, bbox=(750, 290, 790, 330), class_name="hand", confidence=0.9)
        tracker.update([hand_b_mov], current_time=0.1)
        assert tracker._records[tid_rack].is_coasting is True

        # Frame 2: Rack hotdog reappears at exact same spot (150, 300) without hand
        # Hand B is still in transit elsewhere
        hd_rack_reappear = _det(rack_bbox, track_id=55)  # new detector ID on reappear
        tracker.update([hd_rack_reappear, hand_b_mov], current_time=0.2)

        # Rack hotdog must be recovered as tid_rack (NO new ID spawned!)
        log = tracker.get_hotdog_log()
        assert tracker._records[tid_rack].is_coasting is False
        assert tracker._records[tid_rack].bbox == rack_bbox
        assert len(log) == 2  # exactly 2 tracks total, no churn!


# ── Histogram utility tests ────────────────────────────────────────────────────

class TestHistogramUtils:
    def test_identical_histograms_distance_zero(self):
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        bbox = (10, 10, 90, 90)
        h1 = _compute_bgr_histogram(frame, bbox)
        h2 = _compute_bgr_histogram(frame, bbox)
        assert _bhattacharyya_distance(h1, h2) == pytest.approx(0.0, abs=1e-4)

    def test_none_histogram_returns_max_distance(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        h = _compute_bgr_histogram(frame, (10, 10, 90, 90))
        assert _bhattacharyya_distance(None, h) == pytest.approx(1.0)
        assert _bhattacharyya_distance(h, None) == pytest.approx(1.0)

    def test_different_colours_have_large_distance(self):
        frame_red = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_red[:, :, 2] = 255
        frame_blue = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_blue[:, :, 0] = 255
        bbox = (0, 0, 100, 100)
        h_red = _compute_bgr_histogram(frame_red, bbox)
        h_blue = _compute_bgr_histogram(frame_blue, bbox)
        dist = _bhattacharyya_distance(h_red, h_blue)
        assert dist > 0.5, f"Expected high distance for different colours, got {dist}"

    def test_empty_bbox_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        h = _compute_bgr_histogram(frame, (50, 50, 50, 50))
        assert h is None
