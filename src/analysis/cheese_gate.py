"""Cheese pre-gate: confirm a slice the moment it leaves the cheese region.

Workers on this line reach for cheese *first*.  A C/C (chili cheese) dog starts
with a slice out of a cheese well, and that routinely happens before the KDS
ticket has been confirmed: ``TicketStabilityTracker`` still needs its three
agreeing OCR reads and its three ``*** Paid ***`` reads, and the group then has
to reach the head of the FIFO.  Until all of that lands,
``OrderStateMachine.on_action`` drops every event (``current_ticket is None``),
so the one ingredient the order turns on is the one ingredient never recorded.

This module watches the cheese wells independently of ticket state:

    fingertip dwells in a cheese well    -> ARMED, tagged with that well
    hand carries it out of cheese_region -> a CheeseTake is emitted

Leaving the *region* is the confirmation, not leaving the *well*.  All three
cheese wells sit inside one ``cheese_region`` zone, so a hand drifting from the
swiss well to the yellow-slice well has not carried anything anywhere -- while
the bin-exit shortcut in ``temporal.py`` (``_REQUIRE_HOTDOG_RETURN = False``)
fires a place event on exactly that move.  Region exit also does not require the
hand to reach a hotdog, which matters here: the cheese is fetched before there
is a hotdog on the board to reach, so the pending pick would otherwise expire
on ``_PICK_TTL_DISCRETE_S`` and be downgraded to a hover.

Takes confirmed while no order is in progress are buffered and replayed onto the
ticket once it is confirmed.  The well the slice actually came from is carried
through untouched, so a swiss slice on a ticket that asked for yellow sliced
stays visible as swiss and is judged a wrong ingredient rather than quietly
satisfying the cheese requirement.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.core.naming import normalize_item_name

logger = logging.getLogger(__name__)

#: Canonical keys for every well that dispenses cheese.  A ticket is treated as
#: "asks for cheese" when any of these appears in its required counts.
CHEESE_KEYS: frozenset = frozenset(
    {"yellow_cheese", "swiss_cheese", "grated_yellow_cheese"}
)

#: Consecutive frames the fingertip must sit in a cheese well before the hand
#: counts as holding a slice.  LOOSENED 2 -> 1: workers reach for cheese fast
#: (see the module docstring) and a quick touch-and-go in the well was
#: arming on frame 2 or never at all if the dwell was that brief, which is
#: the same missed-slice complaint hotdog_conf_threshold was lowered for.
_MIN_WELL_FRAMES: int = 1

#: Consecutive frames the fingertip must be outside the region before the carry
#: is believed.  Two frames rather than one so a bbox that jitters across the
#: region edge does not emit a take the hand never made.
_MIN_OUT_FRAMES: int = 2

#: An arm that never leaves the region is dropped after this long -- the hand
#: was working the well, not fetching a slice for an order.
_ARM_TTL_S: float = 30.0

#: A track not seen for this long is forgotten, so a recycled tracker id never
#: inherits the previous hand's arm.
_TRACK_TTL_S: float = 5.0


@dataclass
class CheeseTake:
    """One slice confirmed out of a cheese well and clear of the region."""

    item: str        # canonical key, e.g. "swiss_cheese"
    well: str        # the zone name it came from, e.g. "swiss cheese"
    track_id: int
    timestamp: float
    armed_at: float

    @property
    def transit_s(self) -> float:
        """How long the slice spent between the well and the region edge."""
        return self.timestamp - self.armed_at


@dataclass
class _ArmedHand:
    well: str
    item: str
    armed_at: float
    out_frames: int = 0


@dataclass
class _TrackState:
    last_seen: float = 0.0
    well_id: Optional[str] = None
    well_frames: int = 0
    arm: Optional[_ArmedHand] = None


def _point_in(polygon: Iterable[Tuple[float, float]], nx: float, ny: float) -> bool:
    poly = np.array(polygon, dtype=np.float32)
    return cv2.pointPolygonTest(poly, (nx, ny), False) >= 0


def _fingertip(bbox) -> Tuple[float, float]:
    """The point ZoneManager uses for hands: 75% down the hand box.

    Copied deliberately rather than imported so well membership here agrees
    exactly with the bin a pick would have been attributed to elsewhere.
    """
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y1 + (y2 - y1) * 0.75


class CheesePreGate:
    """Emits a :class:`CheeseTake` per slice carried out of the cheese region."""

    def __init__(
        self,
        zones,
        lookback_s: float = 120.0,
        min_well_frames: int = _MIN_WELL_FRAMES,
        enabled: bool = True,
    ):
        self.lookback_s = float(lookback_s)
        self.min_well_frames = int(min_well_frames)

        regions = zones.get_zones_by_type("cheese_region")
        self.region = regions[0] if regions else None
        self.wells = [
            z
            for z in zones.get_all()
            if z.zone_type == "bin" and normalize_item_name(z.name) in CHEESE_KEYS
        ]

        self.enabled = bool(enabled) and self.region is not None and bool(self.wells)
        if not enabled:
            logger.info("Cheese pre-gate disabled by configuration")
        elif self.region is None:
            logger.warning(
                "No 'cheese_region' zone in zones.json -- the cheese pre-gate is "
                "off, so a slice fetched before its ticket is confirmed is still "
                "lost.  Draw the region with scripts/edit_rois.py to enable it."
            )
        elif not self.wells:
            logger.warning(
                "No cheese bins found in zones.json -- the cheese pre-gate is off."
            )
        else:
            logger.info(
                "Cheese pre-gate armed on %d well(s): %s  (region=%s, lookback=%.0fs)",
                len(self.wells),
                ", ".join(z.name for z in self.wells),
                self.region.name,
                self.lookback_s,
            )

        self._tracks: Dict[int, _TrackState] = {}
        self._buffer: deque = deque()
        self.emitted = 0
        self.discarded = 0

    # ------------------------------------------------------------------ update

    def update(self, hand_detections, frame_width: int, frame_height: int, now: float):
        """Advance every hand one frame; return the takes confirmed on this one.

        Takes are also buffered, so a caller with no order in progress can ignore
        the return value and replay :meth:`drain` when the ticket confirms.
        """
        if not self.enabled:
            return []

        takes: List[CheeseTake] = []
        for det in hand_detections or ():
            cx, cy = _fingertip(det.bbox)
            nx, ny = cx / float(frame_width), cy / float(frame_height)

            state = self._tracks.get(det.track_id)
            if state is None:
                state = _TrackState()
                self._tracks[det.track_id] = state
            state.last_seen = now

            well = next(
                (z for z in self.wells if _point_in(z.polygon, nx, ny)), None
            )
            if well is not None:
                if state.well_id == well.id:
                    state.well_frames += 1
                else:
                    state.well_id, state.well_frames = well.id, 1
                if state.well_frames >= self.min_well_frames and state.arm is None:
                    state.arm = _ArmedHand(
                        well=well.name,
                        item=normalize_item_name(well.name),
                        armed_at=now,
                    )
                    logger.debug(
                        "Cheese armed: track=%d well=%r", det.track_id, well.name
                    )
            else:
                state.well_id, state.well_frames = None, 0

            arm = state.arm
            if arm is None:
                continue

            if now - arm.armed_at > _ARM_TTL_S:
                logger.debug(
                    "Cheese arm expired without leaving the region "
                    "(track=%d well=%r, %.1fs)",
                    det.track_id, arm.well, now - arm.armed_at,
                )
                state.arm = None
                continue

            if _point_in(self.region.polygon, nx, ny):
                arm.out_frames = 0
                continue

            arm.out_frames += 1
            if arm.out_frames < _MIN_OUT_FRAMES:
                continue

            take = CheeseTake(
                item=arm.item,
                well=arm.well,
                track_id=det.track_id,
                timestamp=now,
                armed_at=arm.armed_at,
            )
            state.arm = None
            self.emitted += 1
            takes.append(take)
            self._buffer.append(take)
            logger.info(
                "Cheese taken: %s (well=%r) left the cheese region after %.1fs "
                "[track=%d]",
                take.item, take.well, take.transit_s, det.track_id,
            )

        self._prune(now)
        return takes

    def _prune(self, now: float) -> None:
        for tid, state in list(self._tracks.items()):
            if now - state.last_seen > _TRACK_TTL_S:
                del self._tracks[tid]
        while self._buffer and (now - self._buffer[0].timestamp) > self.lookback_s:
            self._buffer.popleft()
            self.discarded += 1

    # ------------------------------------------------------------------ buffer

    def drain(self, now: float) -> List[CheeseTake]:
        """Take everything still inside the lookback window and clear it."""
        self._prune(now)
        drained = list(self._buffer)
        self._buffer.clear()
        return drained

    def forget(self, take: CheeseTake) -> None:
        """Drop one take from the buffer -- it was applied live."""
        try:
            self._buffer.remove(take)
        except ValueError:
            pass

    def reset(self) -> None:
        self._tracks.clear()
        self._buffer.clear()
