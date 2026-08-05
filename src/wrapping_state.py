"""
wrapping_state.py
─────────────────
Order-completion state machine based on the "wrapping" YOLO detection class.

Each tracked hotdog ID moves through three states:

    ACTIVE  →  CLOSING  (wrapping has been detected overlapping this hotdog
                         continuously for >= WRAPPING_DWELL_S seconds)
    CLOSING →  DONE     (hotdog ID disappears from detections entirely after
                         having passed through CLOSING)

IDs that reach DONE are permanently retired for the rest of the run (no
revival / no recycling).  A hotdog that disappears while still ACTIVE is
treated as a lost/occluded track — it is NOT marked DONE.

Trigger logic (ACTIVE → CLOSING)
─────────────────────────────────
• Any wrapping bbox that intersects the hotdog bbox (even partially) starts /
  maintains the dwell timer for that hotdog.
• If no wrapping bbox intersects for more than WRAPPING_GAP_RESET_S seconds,
  the timer resets (handles brief detection gaps at the model level).
• Once the timer reaches WRAPPING_DWELL_S (default 3.0 s), the hotdog
  transitions to CLOSING — "order about to complete".

Trigger logic (CLOSING → DONE)
────────────────────────────────
• Once CLOSING, the machine watches for the hotdog track_id to disappear from
  detections (fully wrapped and removed from view).
• Disappearance after CLOSING → DONE — "order completed".

Integration
───────────
This module is intentionally self-contained.  It does not import from or
modify any existing src.* module.

Call ``WrappingStateMachine.update()`` once per frame immediately after
``HotdogTracker.update()`` in src/main.py (one new function call — unchanged
from before).

Returned events can be forwarded to the dashboard or KDS at the caller's
discretion; the machine also logs every transition at INFO level.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# ── Public state constants ─────────────────────────────────────────────────────

STATE_ACTIVE  = "active"
STATE_CLOSING = "closing"
STATE_DONE    = "done"

# ── Module logger ──────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ── Geometry helper ────────────────────────────────────────────────────────────

def _boxes_intersect(
    bbox_a: Tuple[int, int, int, int],
    bbox_b: Tuple[int, int, int, int],
) -> bool:
    """Return True if two bboxes overlap at all (any intersection)."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _hotdog_coverage_ratio(
    hotdog_bbox: Tuple[int, int, int, int],
    wrapping_bbox: Tuple[int, int, int, int],
) -> float:
    """
    Return the fraction of the *hotdog* bbox area covered by the wrapping bbox.
    Kept for introspection / debugging; not used in the core state transitions.
    """
    hx1, hy1, hx2, hy2 = hotdog_bbox
    wx1, wy1, wx2, wy2 = wrapping_bbox
    hotdog_area = max(0, hx2 - hx1) * max(0, hy2 - hy1)
    if hotdog_area <= 0:
        return 0.0
    inter_w = max(0, min(hx2, wx2) - max(hx1, wx1))
    inter_h = max(0, min(hy2, wy2) - max(hy1, wy1))
    return float(inter_w * inter_h) / float(hotdog_area)


# ── Per-ID state record ────────────────────────────────────────────────────────

@dataclass
class _HotdogWrapState:
    """Tracks wrapping lifecycle for one hotdog track_id."""

    hotdog_tid: int

    # ── Public state ──────────────────────────────────────────────────────────
    state: str = STATE_ACTIVE           # "active" | "closing" | "done"
    closing_frame: Optional[int] = None  # frame when CLOSING was entered
    done_frame:    Optional[int] = None  # frame when DONE was entered
    closing_time:  Optional[float] = None
    done_time:     Optional[float] = None

    # ── Dwell timer (ACTIVE only) ─────────────────────────────────────────────
    # wrapping_dwell_start: timestamp when wrapping first started overlapping
    # wrapping_last_seen:   timestamp of most recent frame with wrapping overlap
    # Both are None until wrapping is first detected near this hotdog.
    wrapping_dwell_start: Optional[float] = None
    wrapping_last_seen:   Optional[float] = None

    # ── Disappearance tracking (CLOSING → DONE delay) ────────────────────────
    disappear_time:  Optional[float] = None  # time when hotdog first disappeared
    disappear_frame: Optional[int] = None    # frame when hotdog first disappeared

    last_bbox: Optional[Tuple[int, int, int, int]] = None


# ── Main state machine ─────────────────────────────────────────────────────────

class WrappingStateMachine:
    """
    Per-run ACTIVE → CLOSING → DONE state machine for hotdog wrapping.

    ACTIVE → CLOSING
    ────────────────
    Triggered when a wrapping detection has been overlapping this hotdog for
    ``wrapping_dwell_s`` seconds (default 0.4 s / 400 ms).
    "Continuously" allows brief model-level detection gaps up to
    ``wrapping_gap_reset_s`` seconds (default 1.0 s) without resetting the
    timer.

    CLOSING → DONE
    ──────────────
    Triggered when the hotdog track_id is absent from detections for a delay of
    ``done_delay_s`` seconds (default 3.0 s) after becoming invisible.
    """

    def __init__(
        self,
        wrapping_dwell_s: float = 0.4,
        wrapping_gap_reset_s: float = 1.0,
        min_closing_frames: int = 30,
        done_delay_s: float = 1.0,
    ) -> None:
        """
        Parameters
        ──────────
        wrapping_dwell_s      : seconds wrapping must be present before CLOSING
                                (default 0.4 s / 400 ms)
        wrapping_gap_reset_s  : gap longer than this resets the dwell timer
                                (default 1.0 s — handles brief detector misses)
        min_closing_frames    : minimum frames CLOSING state remains active
                                before transitioning to DONE (default 30 frames / ~1s)
        done_delay_s          : seconds hotdog must remain invisible after CLOSING
                                before transitioning to DONE (default 1.0 s delay)
        """
        self.wrapping_dwell_s     = wrapping_dwell_s
        self.wrapping_gap_reset_s = wrapping_gap_reset_s
        self.min_closing_frames   = min_closing_frames
        self.done_delay_s         = done_delay_s

        # Permanent retirement set — persists for the full run.
        self.done_ids: Set[int] = set()

        # Active state records keyed by track_id.
        self._states: Dict[int, _HotdogWrapState] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def update(
        self,
        frame_idx: int,
        current_time: float,
        hotdog_detections: list,     # List[Detection] class_name == "hot-dog"
        wrapping_detections: list,   # List[Detection] class_name == "wrapping"
        video_ending_soon: bool = False,
    ) -> List[dict]:
        """
        Process one frame.  Call once per frame after HotdogTracker.update().

        Returns a list of transition-event dicts fired this frame (may be []):
            {
                "event":               "closing" | "done",
                "hotdog_tid":          int,
                "frame":               int,
                "timestamp":           float,
                # closing events also include:
                "wrapping_dwell_s":    float,   # how long wrapping was present
                # done events also include:
                "closing_frame":       int,
                "closing_time":        float,
            }
        """
        events: List[dict] = []

        # ── Build this-frame hotdog index (skip retired IDs immediately) ───────
        active_tids: Set[int] = set()
        tid_to_bbox: Dict[int, Tuple[int, int, int, int]] = {}

        for det in hotdog_detections:
            tid = det.track_id
            if tid is None or tid < 0:
                continue
            if tid in self.done_ids:        # no-revival guard
                continue
            active_tids.add(tid)
            tid_to_bbox[tid] = det.bbox

        # ── Collect wrapping bboxes this frame ─────────────────────────────────
        wrapping_bboxes: List[Tuple[int, int, int, int]] = [
            d.bbox for d in wrapping_detections
        ]

        # ── Step 1: Process all visible hotdogs ────────────────────────────────
        for tid in active_tids:
            bbox = tid_to_bbox[tid]

            # Lazily initialise state record for new IDs
            if tid not in self._states:
                self._states[tid] = _HotdogWrapState(hotdog_tid=tid)

            rec = self._states[tid]
            rec.last_bbox = bbox

            # CLOSING / DONE states need no wrapping-overlap processing here
            if rec.state != STATE_ACTIVE:
                continue

            # ── Check whether any wrapping bbox overlaps this hotdog ──────────
            wrapping_overlaps = any(
                _boxes_intersect(bbox, wb) for wb in wrapping_bboxes
            )

            if wrapping_overlaps:
                if rec.wrapping_dwell_start is None:
                    # First frame wrapping is seen near this hotdog
                    rec.wrapping_dwell_start = current_time
                    logger.debug(
                        "[WRAPPING] hotdog tid=%d  wrapping dwell started  t=%.2fs",
                        tid, current_time,
                    )
                rec.wrapping_last_seen = current_time

                # ── Check if dwell threshold has been reached ─────────────────
                dwell_elapsed = current_time - rec.wrapping_dwell_start
                if dwell_elapsed >= self.wrapping_dwell_s:
                    rec.state         = STATE_CLOSING
                    rec.closing_frame = frame_idx
                    rec.closing_time  = current_time

                    logger.info(
                        "[WRAPPING] hotdog tid=%d  ACTIVE → CLOSING  "
                        "frame=%d  t=%.2fs  "
                        "(wrapping present for %.1fs >= %.1fs threshold)",
                        tid, frame_idx, current_time,
                        dwell_elapsed, self.wrapping_dwell_s,
                    )
                    events.append({
                        "event":            "closing",
                        "hotdog_tid":       tid,
                        "frame":            frame_idx,
                        "timestamp":        current_time,
                        "wrapping_dwell_s": round(dwell_elapsed, 2),
                    })

            else:
                # No wrapping overlap this frame — check if gap is too long
                if (
                    rec.wrapping_last_seen is not None
                    and (current_time - rec.wrapping_last_seen)
                        > self.wrapping_gap_reset_s
                ):
                    logger.debug(
                        "[WRAPPING] hotdog tid=%d  dwell timer RESET  "
                        "gap=%.1fs > %.1fs",
                        tid,
                        current_time - rec.wrapping_last_seen,
                        self.wrapping_gap_reset_s,
                    )
                    rec.wrapping_dwell_start = None
                    rec.wrapping_last_seen   = None

        # ── Step 2: CLOSING → DONE transitions ────────────────────────────────
        # A CLOSING hotdog transitions to DONE when its track_id is absent from
        # detections continuously for done_delay_s (default 3.0s delay after becoming invisible).
        for tid, rec in list(self._states.items()):
            if rec.state != STATE_CLOSING:
                continue
            if tid in active_tids:
                # Hotdog is visible again — reset disappearance timer
                rec.disappear_time  = None
                rec.disappear_frame = None
                continue

            # Record time when hotdog first disappeared
            if rec.disappear_time is None:
                rec.disappear_time  = current_time
                rec.disappear_frame = frame_idx

            disappear_frames = (frame_idx - rec.disappear_frame) if rec.disappear_frame is not None else 0

            # Fast-path for video ending (remaining_s <= 0.8s):
            # When video is ending soon and wrapping/hotdog has disappeared for at least 5 frames,
            # transition to DONE IMMEDIATELY without waiting for normal delay.
            if video_ending_soon:
                if disappear_frames < 5:
                    continue
                rec.state      = STATE_DONE
                rec.done_frame = frame_idx
                rec.done_time  = current_time
                self.done_ids.add(tid)
                logger.info(
                    "[WRAPPING] hotdog tid=%d  CLOSING → DONE (video ending immediate, disappeared %d frames)  "
                    "frame=%d  t=%.2fs",
                    tid, disappear_frames, frame_idx, current_time,
                )
                events.append({
                    "event":         "done",
                    "hotdog_tid":    tid,
                    "frame":         frame_idx,
                    "timestamp":     current_time,
                    "closing_frame": rec.closing_frame,
                    "closing_time":  rec.closing_time,
                })
                continue

            # Verify disappearance delay threshold (e.g. 1.0s delay after becoming not visible)
            invisible_duration_s = current_time - rec.disappear_time
            if invisible_duration_s < self.done_delay_s:
                continue

            # Verify minimum closing frames threshold (30 frames)
            closing_duration = (
                (frame_idx - rec.closing_frame) if rec.closing_frame is not None else 30
            )
            if closing_duration < self.min_closing_frames:
                continue

            # Hotdog was CLOSING and has been invisible for >= done_delay_s -> DONE
            rec.state      = STATE_DONE
            rec.done_frame = frame_idx
            rec.done_time  = current_time
            self.done_ids.add(tid)

            logger.info(
                "[WRAPPING] hotdog tid=%d  CLOSING → DONE  "
                "frame=%d  t=%.2fs  "
                "(closing started frame=%s  t=%ss)",
                tid, frame_idx, current_time,
                rec.closing_frame, rec.closing_time,
            )
            events.append({
                "event":         "done",
                "hotdog_tid":    tid,
                "frame":         frame_idx,
                "timestamp":     current_time,
                "closing_frame": rec.closing_frame,
                "closing_time":  rec.closing_time,
            })

        return events

    # ── Introspection helpers ──────────────────────────────────────────────────

    def get_state(self, hotdog_tid: int) -> Optional[str]:
        """Return current state string for a given hotdog track_id, or None."""
        if hotdog_tid in self.done_ids:
            return STATE_DONE
        rec = self._states.get(hotdog_tid)
        return rec.state if rec else None

    def get_dwell_elapsed(self, hotdog_tid: int, current_time: float) -> float:
        """
        Return how many seconds wrapping has been detected on this hotdog
        (dwell timer elapsed), or 0.0 if not started / already past CLOSING.
        """
        rec = self._states.get(hotdog_tid)
        if rec is None or rec.wrapping_dwell_start is None:
            return 0.0
        return max(0.0, current_time - rec.wrapping_dwell_start)

    def get_all_states(self) -> Dict[int, dict]:
        """
        Return a snapshot of all known hotdog wrapping states.
        Useful for dashboard / debug output.  Dict key is track_id.
        """
        result: Dict[int, dict] = {}
        for tid, rec in self._states.items():
            result[tid] = {
                "hotdog_tid":           rec.hotdog_tid,
                "state":                rec.state,
                "wrapping_dwell_start": rec.wrapping_dwell_start,
                "wrapping_last_seen":   rec.wrapping_last_seen,
                "closing_frame":        rec.closing_frame,
                "closing_time":         rec.closing_time,
                "done_frame":           rec.done_frame,
                "done_time":            rec.done_time,
                "last_bbox":            rec.last_bbox,
            }
        return result

    def reset(self) -> None:
        """
        Clear per-ID state (call when switching videos / restarting a run).
        ``done_ids`` is intentionally preserved — call ``full_reset()`` to
        clear it too.
        """
        self._states.clear()

    def full_reset(self) -> None:
        """Clear all state including the done_ids retirement set."""
        self._states.clear()
        self.done_ids.clear()
