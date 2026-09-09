"""Temporal aggregation of per-frame OCR observations (section 21).

OCR must never be an independent decision per frame.  Every snapshot produced by
:class:`~src.kds.ticket_parser.TicketParser` is fed here, and this module votes
across a rolling window before anything downstream is allowed to act.  That is
what prevents:

* duplicate tickets (the same card creating an order on 100 consecutive frames),
* OCR character fluctuation in the ticket ID,
* payment-state flicker (one bad frame reading ``Paid``),
* a temporary total OCR failure being mistaken for the ticket disappearing.

The unit of tracking is the card **slot** from
:class:`~src.kds.ticket_detector.TicketCardDetector`, not the OCR text, so a
ticket keeps its identity across frames where nothing could be read at all.
"""

from __future__ import annotations

import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from src.kds.schemas import (
    HotdogGroup,
    KdsEvent,
    KdsEventType,
    PaymentStatus,
    TicketSnapshot,
)

logger = logging.getLogger(__name__)


@dataclass
class TrackedTicket:
    """Aggregated state for one card slot."""

    slot_id: int
    first_seen: float
    last_seen: float
    # Rolling observation windows.
    id_votes: Deque[str] = field(default_factory=deque)
    payment_votes: Deque[PaymentStatus] = field(default_factory=deque)
    content_votes: Deque[str] = field(default_factory=deque)
    snapshots: Deque[TicketSnapshot] = field(default_factory=deque)

    confirmed_id: str = ""
    confirmed_payment: PaymentStatus = PaymentStatus.UNKNOWN
    paid_observations: int = 0
    content_counts: Counter = field(default_factory=Counter)

    # Set once, when the ticket is created.  Guarantees exactly one creation
    # per ticket even though the card is re-read on every frame.
    created: bool = False
    created_at: Optional[float] = None
    frozen_hotdogs: List[HotdogGroup] = field(default_factory=list)
    announced_unpaid: bool = False
    last_readable_snapshot: Optional[TicketSnapshot] = None

    @property
    def age(self) -> float:
        return self.last_seen - self.first_seen


class TicketStabilityTracker:
    """Votes over time and emits confirmed ticket events."""

    def __init__(self, config: Optional[dict] = None):
        config = config or {}
        stability = (config.get("stability", {}) or {}) if isinstance(config, dict) else {}
        self.window_size = int(stability.get("window_size", 15))
        self.id_confirm = int(stability.get("id_confirm_observations", 3))
        self.paid_confirm = int(stability.get("paid_confirm_observations", 3))
        self.content_confirm = int(stability.get("content_confirm_observations", 2))
        # Whether a ticket must read PAID before it enters the system.  Off by
        # default: the KDS shows the order as soon as it is rung up, and waiting
        # for the payment line to be read costs the start of the order -- the
        # kitchen has usually begun by then.  Content settling is the trigger
        # instead, and payment is recorded for information only.
        self.require_paid = bool(stability.get("require_paid", False))
        self.disappear_grace_s = float(stability.get("disappear_grace_s", 3.0))
        self.unreadable_counts_as_present = bool(
            stability.get("unreadable_counts_as_present", True)
        )

        self._tracked: Dict[int, TrackedTicket] = {}
        # ticket_id -> slot_id, so a card that moves position keeps its ticket.
        self._id_to_slot: Dict[str, int] = {}
        # Every ticket ID ever created, so a re-appearing card is never
        # duplicated into a second order.
        self._created_ids: set = set()

    # ------------------------------------------------------------------ update

    def observe(self, slot_id: int, snapshot: TicketSnapshot) -> List[KdsEvent]:
        """Record one observation of one card slot; return any confirmed events."""
        events: List[KdsEvent] = []
        now = snapshot.timestamp

        tracked = self._tracked.get(slot_id)
        if tracked is None:
            tracked = TrackedTicket(slot_id=slot_id, first_seen=now, last_seen=now)
            self._tracked[slot_id] = tracked
        tracked.last_seen = now

        # An unreadable card still proves the ticket is present; it just adds
        # no votes.  This is exactly the temporary-OCR-failure case.
        if not snapshot.readable:
            return events

        tracked.last_readable_snapshot = snapshot
        _push(tracked.snapshots, snapshot, self.window_size)

        if snapshot.ticket_id:
            _push(tracked.id_votes, snapshot.ticket_id, self.window_size)
        _push(tracked.payment_votes, snapshot.payment, self.window_size)

        # --- ticket identity -------------------------------------------------
        previous_id = tracked.confirmed_id
        winner, votes = _majority(tracked.id_votes)
        if winner and votes >= self.id_confirm and winner != tracked.confirmed_id:
            if not tracked.confirmed_id:
                tracked.confirmed_id = winner
                self._id_to_slot[winner] = slot_id
                events.append(
                    KdsEvent(KdsEventType.TICKET_SEEN, winner, now, snapshot)
                )
                logger.info("Ticket %s confirmed on slot %d", winner, slot_id)
            else:
                # The ID changed under a confirmed ticket.  That means the slot
                # is now showing a different order (the old one was bumped and
                # the grid re-flowed), so retire the old identity cleanly.
                logger.info(
                    "Slot %d changed ticket %s -> %s",
                    slot_id,
                    tracked.confirmed_id,
                    winner,
                )
                events.append(
                    KdsEvent(
                        KdsEventType.TICKET_DISAPPEARED,
                        tracked.confirmed_id,
                        now,
                        detail="slot reassigned to " + winner,
                    )
                )
                self._id_to_slot.pop(tracked.confirmed_id, None)
                self._tracked[slot_id] = TrackedTicket(
                    slot_id=slot_id, first_seen=now, last_seen=now
                )
                return events

        ticket_id = tracked.confirmed_id
        if not ticket_id:
            return events

        # --- content ---------------------------------------------------------
        signature = snapshot.content_signature()
        if signature:
            tracked.content_counts[signature] += 1
            _push(tracked.content_votes, signature, self.window_size)

        if snapshot.unknown_shortcuts:
            for text in snapshot.unknown_shortcuts:
                events.append(
                    KdsEvent(
                        KdsEventType.UNKNOWN_SHORTCUT,
                        ticket_id,
                        now,
                        snapshot,
                        detail=text,
                    )
                )

        # --- payment ---------------------------------------------------------
        # RULE 1: only a ticket confirmed PAID over several observations enters
        # the system.  UNKNOWN is treated exactly like NOT PAID.
        if snapshot.payment is PaymentStatus.PAID:
            tracked.paid_observations += 1
        elif snapshot.payment is PaymentStatus.NOT_PAID:
            # A confident NOT PAID reading undoes accumulated PAID votes only
            # while the ticket has not yet been created; once created, the
            # ticket stays created (payment cannot un-happen).
            if not tracked.created and self.require_paid:
                tracked.paid_observations = max(0, tracked.paid_observations - 1)
                if not tracked.announced_unpaid:
                    tracked.announced_unpaid = True
                    events.append(
                        KdsEvent(
                            KdsEventType.TICKET_UNPAID, ticket_id, now, snapshot
                        )
                    )

        ready = (
            tracked.paid_observations >= self.paid_confirm
            if self.require_paid
            else True
        )
        if not tracked.created and ready:
            if ticket_id in self._created_ids:
                # Already created earlier in this session; never create twice.
                tracked.created = True
            else:
                hotdogs = self._frozen_content(tracked)
                if hotdogs is None:
                    # Payment confirmed but the order content has not settled
                    # yet -- wait rather than create a half-read ticket.
                    return events
                tracked.created = True
                tracked.created_at = now
                tracked.frozen_hotdogs = hotdogs
                tracked.confirmed_payment = snapshot.payment
                self._created_ids.add(ticket_id)
                paid_snapshot = _snapshot_with_content(snapshot, hotdogs)
                events.append(
                    KdsEvent(KdsEventType.TICKET_PAID, ticket_id, now, paid_snapshot)
                )
                logger.info(
                    "Ticket %s entered the system (payment=%s, %d paid observation(s))",
                    ticket_id,
                    snapshot.payment.value,
                    tracked.paid_observations,
                )
        elif tracked.created:
            # Content may still be refined while the ticket is live (a new item
            # added at the counter), but the ticket is never re-created.
            hotdogs = self._frozen_content(tracked)
            if hotdogs is not None and _signature_of(hotdogs) != _signature_of(
                tracked.frozen_hotdogs
            ):
                tracked.frozen_hotdogs = hotdogs
                events.append(
                    KdsEvent(
                        KdsEventType.TICKET_UPDATED,
                        ticket_id,
                        now,
                        _snapshot_with_content(snapshot, hotdogs),
                    )
                )
        return events

    def _frozen_content(self, tracked: TrackedTicket) -> Optional[List[HotdogGroup]]:
        """The order content, once the same reading has won often enough."""
        signature, votes = _majority(tracked.content_votes)
        if signature is None or votes < self.content_confirm:
            return None
        for snapshot in reversed(tracked.snapshots):
            if snapshot.content_signature() == signature:
                return [_copy_group(g) for g in snapshot.hotdogs]
        return None

    # ------------------------------------------------------------- disappearance

    def sweep(self, now: float, present_slots: Optional[set] = None) -> List[KdsEvent]:
        """Emit TICKET_DISAPPEARED for slots absent beyond the grace period.

        The grace period is what makes a ticket survive frames where the card
        was momentarily missed; only a sustained absence counts as the ticket
        leaving the KDS (section 11).
        """
        events: List[KdsEvent] = []
        present_slots = present_slots if present_slots is not None else set()
        for slot_id, tracked in list(self._tracked.items()):
            if slot_id in present_slots:
                continue
            if now - tracked.last_seen < self.disappear_grace_s:
                continue
            if tracked.confirmed_id:
                events.append(
                    KdsEvent(
                        KdsEventType.TICKET_DISAPPEARED,
                        tracked.confirmed_id,
                        now,
                        detail="absent for %.1fs" % (now - tracked.last_seen),
                    )
                )
                logger.info("Ticket %s disappeared from the KDS", tracked.confirmed_id)
                self._id_to_slot.pop(tracked.confirmed_id, None)
            self._tracked.pop(slot_id, None)
        return events

    # ------------------------------------------------------------------ access

    def tracked_for_ticket(self, ticket_id: str) -> Optional[TrackedTicket]:
        slot_id = self._id_to_slot.get(ticket_id)
        return self._tracked.get(slot_id) if slot_id is not None else None

    def slot_for_ticket(self, ticket_id: str) -> Optional[int]:
        return self._id_to_slot.get(ticket_id)

    def ticket_for_slot(self, slot_id: int) -> str:
        tracked = self._tracked.get(slot_id)
        return tracked.confirmed_id if tracked else ""

    def reset(self) -> None:
        self._tracked.clear()
        self._id_to_slot.clear()
        self._created_ids.clear()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _push(window: Deque, value, max_size: int) -> None:
    window.append(value)
    while len(window) > max_size:
        window.popleft()


def _majority(window: Deque):
    """Most common value in the window and its count."""
    if not window:
        return None, 0
    counts = Counter(window)
    value, votes = counts.most_common(1)[0]
    return value, votes


def _copy_group(group: HotdogGroup) -> HotdogGroup:
    """Deep-ish copy so live detection counters never mutate a snapshot."""
    from dataclasses import replace

    return replace(
        group,
        addons=[replace(a) for a in group.addons],
        ingredients=list(group.ingredients),
        matched_track_ids=[],
        detected_count=0,
    )


def _signature_of(groups: List[HotdogGroup]) -> str:
    parts = []
    for group in sorted(groups, key=lambda g: (g.item, g.shortcut)):
        addons = ",".join(sorted(a.key + "x" + str(a.quantity) for a in group.addons))
        parts.append("{0}x{1}[{2}]".format(group.item, group.quantity, addons))
    return "|".join(parts)


def _snapshot_with_content(
    snapshot: TicketSnapshot, hotdogs: List[HotdogGroup]
) -> TicketSnapshot:
    from dataclasses import replace

    return replace(snapshot, hotdogs=hotdogs)
