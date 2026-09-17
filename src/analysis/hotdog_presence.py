"""Count the hotdogs on the bench for one ticket, without tracking identity.

Why this exists
---------------
The hotdog count used to come out of `HotdogTracker`, which maintains Kalman
filters, colour-histogram re-ID, occlusion buffers and hand-transit re-ID to
decide *which* hotdog each detection is.  The verdict never needed that: the
check is `observed >= required`, and ingredients are credited to the ticket as a
whole (`BatchOrderValidator.on_place_event` takes an ingredient name, not a
hotdog id).  All that identity work fed one threshold test.

It also got the number wrong in the one direction that matters.  Ids were drawn
from a pool `1..expected_hotdogs` shared across the WHOLE RUN, so once N hotdogs
had ever been made the pool was spent and every later detection was dropped
before it became a record: a full hour produced four records, and hotdogs
plainly visible on the feed showed as 0/N on the board.

So the count is taken from presence instead.  A hotdog on the assembly bench is
one hotdog for as long as it stays there; it does not need a name.

The model
---------
* Each detection joins the nearest open SLOT within `match_px`, or opens a new
  one.
* A slot survives a gap of up to `grace_s`.  Hands cover a dog constantly during
  assembly, and a dog that disappears for a second and comes back in the same
  place is the same dog -- not a second one.
* A slot must be seen for `min_age_s` before it counts, which is what keeps a
  sub-second misdetection from becoming a hotdog.
* A slot closed by a WRAPPING event cannot be re-joined.  Wrapping is the end of
  a hotdog's life on the bench, so the next detection in that spot is the next
  dog however fast it appears -- which is what a timer alone cannot decide.
* The count is the high-water mark of slots opened during the ticket. It is a
  high-water mark because the live number falls as dogs are carried away, and an
  order that HAD its three dogs does not stop having had them.

`reset()` at every ticket boundary: this counts one ticket, and nothing from a
finished order carries into the next.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class _Slot:
    cx: float
    cy: float
    first_seen: float
    last_seen: float
    counted: bool = False
    closed: bool = False          # wrapped and gone; never re-joined


@dataclass
class HotdogPresence:
    """Per-ticket hotdog count taken from occupancy of the bench."""

    match_px: float = 140.0
    grace_s: float = 2.0
    min_age_s: float = 0.25

    _slots: List[_Slot] = field(default_factory=list, repr=False)
    _high_water: int = 0
    _ticket: Optional[str] = None

    # -- lifecycle -------------------------------------------------------

    def reset(self, ticket_id: Optional[str] = None) -> None:
        self._slots.clear()
        self._high_water = 0
        self._ticket = ticket_id

    # -- per frame -------------------------------------------------------

    def update(self, boxes: Sequence[Tuple[float, float, float, float]],
               now: float) -> int:
        """Feed this frame's hotdog boxes. Returns the ticket's count so far."""
        for slot in self._slots:
            if not slot.closed and (now - slot.last_seen) > self.grace_s:
                slot.closed = True

        for box in boxes:
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            best, best_d = None, self.match_px
            for slot in self._slots:
                if slot.closed:
                    continue
                d = ((slot.cx - cx) ** 2 + (slot.cy - cy) ** 2) ** 0.5
                if d < best_d:
                    best, best_d = slot, d

            if best is None:
                self._slots.append(_Slot(cx=cx, cy=cy, first_seen=now, last_seen=now))
            else:
                # Follow the dog rather than pinning the slot where it started:
                # it gets nudged along the bench while it is being built.
                best.cx, best.cy = cx, cy
                best.last_seen = now

        for slot in self._slots:
            if not slot.counted and (slot.last_seen - slot.first_seen) >= self.min_age_s:
                slot.counted = True

        live = sum(1 for s in self._slots if s.counted)
        if live > self._high_water:
            self._high_water = live
            logger.info("ticket %s: hotdog count now %d (presence)",
                        self._ticket, self._high_water)
        return self._high_water

    def close_at(self, x: float, y: float) -> bool:
        """A hotdog was wrapped here: retire its slot.

        The slot cannot be re-joined afterwards, so a dog built in the same spot
        immediately afterwards is counted separately. Without this the grace
        window would have to tell succession from occlusion, and no single
        duration does both.
        """
        best, best_d = None, self.match_px
        for slot in self._slots:
            if slot.closed:
                continue
            d = ((slot.cx - x) ** 2 + (slot.cy - y) ** 2) ** 0.5
            if d < best_d:
                best, best_d = slot, d
        if best is None:
            return False
        best.closed = True
        return True

    # -- reporting -------------------------------------------------------

    @property
    def count(self) -> int:
        return self._high_water

    @property
    def live(self) -> int:
        """Slots on the bench right now -- for display, not for the verdict."""
        return sum(1 for s in self._slots if s.counted and not s.closed)
