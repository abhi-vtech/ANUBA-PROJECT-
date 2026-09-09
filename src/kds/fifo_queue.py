"""FIFO order groups, lifecycle and final validation (sections 4-6, 9, 11-18).

One paid ticket becomes exactly one :class:`~src.kds.schemas.OrderGroup`.
Groups are queued strictly by the timestamp at which payment was confirmed --
never by where the card sits on the screen (RULE 5).

Two rules dominate the design:

RULE 6 (no early failure)
    While a ticket is still active, missing items mean *waiting*, not wrong.
    Nothing here can produce a WRONG verdict except :meth:`_finalize`, and that
    only runs on a confirmed order-end signal.

RULE 8 (final validation)
    An order is judged exactly once, when the ticket DISAPPEARS from the KDS --
    that is the moment the order is genuinely over, because the ticket has been
    bumped.  At that instant the production evidence is frozen so a late
    detection cannot rewrite history.  (Anything still open at shutdown is
    finalised too, so nothing is silently dropped.)

    Deliberately NOT a trigger: the card turning pink.  On the live KDS the
    card body turns pink simply because the order is overdue, so a verdict
    there would judge an order that is still being made.

A note on what "detected" can honestly mean:
    the production model has a single ``hot-dog`` class -- there is no
    per-variant hotdog class and no chili/cheese/onion classes.  Quantity is
    therefore measured directly and reliably, while the *type* of each hotdog is
    only ever **inferred** from the ingredients the zone pipeline attributed to
    it.  Every type verdict is reported with ``type_match_confident=False`` and
    an explanatory note, and by default a type mismatch alone does not fail an
    order (``strict_type_matching``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from src.kds.schemas import (
    ALLOWED_TRANSITIONS,
    EndReason,
    FailureCategory,
    HotdogGroup,
    LifecycleState,
    OrderGroup,
    TicketSnapshot,
    ValidationResult,
)

logger = logging.getLogger(__name__)


def _content_signature(hotdogs) -> tuple:
    """Comparable summary of an item list, add-ons included."""
    return tuple(
        sorted(
            (h.item, h.quantity, tuple(sorted(a.key for a in h.addons)))
            for h in hotdogs
        )
    )


def _merge_hotdogs(existing, incoming):
    """Fold a fresh KDS reading into what the ticket already asks for.

    Additive on purpose.  A later reading is another observation of the same
    ticket, not a replacement for it: OCR drops a line often enough that a
    straight overwrite would quietly delete a requirement the kitchen still has
    to make, and a vanished requirement is invisible -- it just stops being
    checked.  So a quantity may rise and add-ons may appear, but nothing is
    removed by a reading that failed to see it.

    Items genuinely voided at the counter therefore stay on the checklist.
    That is the deliberate trade: a stale requirement is visible and can be
    judged, a silently dropped one cannot.
    """
    by_item = {}
    order = []
    for group in list(existing) + list(incoming):
        current = by_item.get(group.item)
        if current is None:
            by_item[group.item] = replace(group, addons=list(group.addons))
            order.append(group.item)
            continue
        # Same item seen again: keep the larger requirement.
        if group.quantity > current.quantity:
            current.quantity = group.quantity
        seen = {a.key for a in current.addons}
        for addon in group.addons:
            if addon.key not in seen:
                current.addons.append(addon)
                seen.add(addon.key)
    return [by_item[item] for item in order]


class TicketManager:
    """Owns every order group, the FIFO queue and final validation."""

    def __init__(
        self,
        on_event: Optional[Callable[[str, dict], None]] = None,
        strict_type_matching: bool = False,
        require_paid: bool = False,
        strict_no_extras: bool = False,
    ):
        self._groups: Dict[str, OrderGroup] = {}
        # Ticket IDs in strict paid-order.  Terminal groups are removed.
        self._queue: List[str] = []
        self._completed: List[OrderGroup] = []
        self._warnings: List[OrderGroup] = []
        self._sequence = 0
        self._on_event = on_event
        # Hotdogs detected while the active ticket was already satisfied.  They
        # are handed to the next ticket when it activates (see _activate_next).
        self._pending_detections: List[tuple] = []
        # Hotdogs detected with no ticket at all to attribute them to.
        self._surplus_detections = 0
        self.strict_type_matching = strict_type_matching
        # Whether payment gates entry into the system.  The stability layer has
        # the same switch; both must agree or a ticket is admitted upstream and
        # then silently dropped here.
        self.require_paid = require_paid
        self.strict_no_extras = strict_no_extras

    # ------------------------------------------------------------------ events

    def _emit(self, kind: str, payload: dict) -> None:
        if self._on_event is not None:
            try:
                self._on_event(kind, payload)
            except Exception:  # pragma: no cover - listener must never break us
                logger.exception("KDS event listener failed for %s", kind)

    def update_content(self, ticket_id: str, hotdogs, now: float) -> bool:
        """Apply a KDS content change to a ticket already in the system.

        A ticket can gain or lose items after it first appears -- an item added
        at the counter, or a line that was misread on the first pass and has
        since settled.  The order keeps its identity and its accumulated
        detections; only the requirement changes.  Refused once evidence is
        frozen, because by then the order has been judged.
        """
        group = self.get(ticket_id)
        if group is None or group.evidence_frozen:
            return False

        before = group.expected_total
        merged = _merge_hotdogs(group.hotdogs, hotdogs)
        if _content_signature(merged) == _content_signature(group.hotdogs):
            return False
        group.hotdogs = merged
        logger.info(
            "Ticket %s content updated: expected %d -> %d hotdog(s)",
            ticket_id,
            before,
            group.expected_total,
        )
        self._emit("updated", {"ticket_id": ticket_id, "group": group.to_dict()})
        return True

    def _transition(self, group: OrderGroup, state: LifecycleState, now: float) -> bool:
        """Apply a lifecycle transition, refusing illegal ones."""
        if state is group.state:
            return True
        allowed = ALLOWED_TRANSITIONS.get(group.state, ())
        if state not in allowed:
            logger.warning(
                "Refused illegal transition %s -> %s for ticket %s",
                group.state.value,
                state.value,
                group.ticket_id,
            )
            return False
        group.state = state
        group.state_history.append((now, state))
        self._emit(
            "state",
            {"ticket_id": group.ticket_id, "state": state.value, "timestamp": now},
        )
        return True

    # ------------------------------------------------------------ ticket intake

    def create_from_paid_ticket(
        self, snapshot: TicketSnapshot, now: Optional[float] = None
    ) -> Optional[OrderGroup]:
        """Create the order group for a ticket entering the system (section 4).

        Payment gates this only when ``require_paid`` is set; by default a
        ticket enters as soon as its content settles.

        Returns ``None`` if the ticket already exists -- the single most
        important guard against duplicate creation across frames.
        """
        now = time.monotonic() if now is None else now
        ticket_id = snapshot.ticket_id
        if not ticket_id:
            logger.warning("Refusing to create a ticket with no ID")
            return None
        if ticket_id in self._groups:
            return None
        if self.require_paid and not snapshot.is_paid:
            logger.warning("Refusing to create unpaid ticket %s", ticket_id)
            return None

        known = [g for g in snapshot.hotdogs if not g.is_unknown]
        unknown = [g for g in snapshot.hotdogs if g.is_unknown]
        if not known and not unknown:
            logger.info(
                "Ticket %s is paid but holds no recognisable hotdog content; "
                "not creating an order group",
                ticket_id,
            )
            return None

        self._sequence += 1
        group = OrderGroup(
            ticket_id=ticket_id,
            internal_id="OG-%04d" % self._sequence,
            created_at=now,
            paid_at=now,
            hotdogs=list(snapshot.hotdogs),
            state=LifecycleState.PAID,
            order_type=snapshot.order_type,
            last_seen_on_kds=now,
        )
        group.state_history.append((now, LifecycleState.PAID))
        self._groups[ticket_id] = group
        self._emit("created", {"ticket_id": ticket_id, "group": group.to_dict()})
        logger.info(
            "Created order group %s for ticket %s (%d hotdogs expected)",
            group.internal_id,
            ticket_id,
            group.expected_total,
        )

        self._enqueue(group, now)
        return group

    def _enqueue(self, group: OrderGroup, now: float) -> None:
        """Append to the FIFO queue, ordered by confirmed-paid time (RULE 5)."""
        if not self._transition(group, LifecycleState.QUEUED, now):
            return
        self._queue.append(group.ticket_id)
        # Stable sort by paid time; screen position is deliberately ignored.
        self._queue.sort(key=lambda tid: self._groups[tid].paid_at)
        self._emit("queued", {"ticket_id": group.ticket_id, "position": self.position_of(group.ticket_id)})
        self._activate_next(now)

    def _activate_next(self, now: float) -> Optional[OrderGroup]:
        """Promote the head of the queue to ACTIVE."""
        active = self.active_group
        if active is not None:
            return active
        for ticket_id in self._queue:
            group = self._groups[ticket_id]
            if group.state is LifecycleState.QUEUED:
                if self._transition(group, LifecycleState.ACTIVE, now):
                    group.activated_at = now
                    self._emit("activated", {"ticket_id": ticket_id})
                    logger.info("Ticket %s is now ACTIVE", ticket_id)
                    self._drain_pending(group, now)
                    return group
        return None

    def _drain_pending(self, group: OrderGroup, now: float) -> None:
        """Give a newly activated ticket the hotdogs made while it waited.

        Production runs ahead of the queue: a worker can start the next order
        before the current ticket is bumped.  Those detections were buffered
        rather than discarded, so replay them now -- only as many as this
        ticket actually needs.
        """
        if not self._pending_detections:
            return
        needed = group.expected_total
        replayed = 0
        keep: List[tuple] = []
        for item, track_id, confidence, seen_at in self._pending_detections:
            if replayed < needed:
                self.record_detection(
                    item=item,
                    track_id=track_id,
                    confidence=confidence,
                    now=now,
                    ticket_id=group.ticket_id,
                )
                replayed += 1
            else:
                keep.append((item, track_id, confidence, seen_at))
        self._pending_detections = keep
        if replayed:
            logger.info(
                "Replayed %d buffered hotdog detection(s) onto ticket %s",
                replayed,
                group.ticket_id,
            )

    # --------------------------------------------------------- production input

    def record_detection(
        self,
        item: str,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        now: Optional[float] = None,
        ticket_id: Optional[str] = None,
    ) -> Optional[OrderGroup]:
        """Attribute one confirmed physical hotdog to a ticket (section 9).

        ``item`` is the resolved physical item name when the type could be
        inferred, or the bare detection class when it could not.  Attribution
        goes to the active FIFO ticket unless one is named explicitly.

        The caller is responsible for temporal confirmation -- this must never
        be called from a single-frame detection (section 8).
        """
        now = time.monotonic() if now is None else now
        group = self._groups.get(ticket_id) if ticket_id else self.active_group
        if group is None or group.is_terminal or group.evidence_frozen:
            self._surplus_detections += 1
            return None

        # A ticket that already has everything it asked for does not absorb the
        # rest of the shift.  Once satisfied, further hotdogs belong to whatever
        # comes next in the FIFO -- so they are held, not attributed here.
        # Without this, a long production video piles every hotdog onto the one
        # active ticket and the final verdict is meaningless.
        if ticket_id is None and group.detected_total >= group.expected_total > 0:
            self._pending_detections.append((item, track_id, confidence, now))
            del self._pending_detections[: max(0, len(self._pending_detections) - 64)]
            return None

        if track_id is not None:
            if track_id in group.detected_track_ids:
                return group
            group.detected_track_ids.append(track_id)
        group.detected_counts[item] = group.detected_counts.get(item, 0) + 1

        matched = self._attribute_to_hotdog(group, item, track_id, confidence)
        if group.state is LifecycleState.ACTIVE:
            self._transition(group, LifecycleState.IN_PROGRESS, now)

        self._emit(
            "detection",
            {
                "ticket_id": group.ticket_id,
                "item": item,
                "track_id": track_id,
                "matched_group": matched.item if matched else None,
                "detected_total": group.detected_total,
                "expected_total": group.expected_total,
            },
        )

        # RULE 6: reaching the expected count is progress, not a verdict.  The
        # order still waits for a real end signal before it is validated.
        if group.detected_total >= group.expected_total and group.expected_total > 0:
            self._transition(group, LifecycleState.WAITING_FOR_COMPLETION, now)
        return group

    @staticmethod
    def _attribute_to_hotdog(
        group: OrderGroup,
        item: str,
        track_id: Optional[int],
        confidence: float,
    ) -> Optional[HotdogGroup]:
        """Assign a detection to the best-fitting expected hotdog line."""
        # Prefer an exact type match that still needs units.
        for hotdog in group.hotdogs:
            if hotdog.is_unknown:
                continue
            if hotdog.item == item and hotdog.remaining > 0:
                hotdog.detected_count += 1
                hotdog.detection_confidence = max(hotdog.detection_confidence, confidence)
                if track_id is not None:
                    hotdog.matched_track_ids.append(track_id)
                return hotdog
        # Otherwise fill any line that still needs units.  The physical model
        # cannot tell the variants apart, so an unattributed hotdog counts
        # towards the order; the type verdict is reported separately.
        for hotdog in group.hotdogs:
            if hotdog.is_unknown:
                continue
            if hotdog.remaining > 0:
                hotdog.detected_count += 1
                hotdog.detection_confidence = max(hotdog.detection_confidence, confidence)
                if track_id is not None:
                    hotdog.matched_track_ids.append(track_id)
                return hotdog
        return None

    def note_kds_presence(self, ticket_id: str, now: Optional[float] = None) -> None:
        group = self._groups.get(ticket_id)
        if group is not None:
            group.last_seen_on_kds = time.monotonic() if now is None else now

    # ------------------------------------------------------------- end signals

    def on_ticket_disappeared(
        self, ticket_id: str, now: Optional[float] = None
    ) -> Optional[ValidationResult]:
        """Finalise a ticket that left the KDS -- the ONLY verdict trigger.

        A disappeared ticket is never left waiting indefinitely: it is either
        already finished, or it gets one final validation from whatever
        production evidence exists, and is then removed from monitoring.
        """
        now = time.monotonic() if now is None else now
        group = self._groups.get(ticket_id)
        if group is None:
            return None
        group.disappeared_at = now
        if group.is_terminal:
            self._remove_from_queue(ticket_id)
            self._activate_next(now)
            return group.result
        logger.info("Ticket %s disappeared -> final validation", ticket_id)
        return self._finalize(group, EndReason.TICKET_DISAPPEARED, now)

    def finalize_all(self, now: Optional[float] = None) -> List[ValidationResult]:
        """Finalise everything still open (shutdown / end of video)."""
        now = time.monotonic() if now is None else now
        results = []
        for ticket_id in list(self._queue):
            group = self._groups.get(ticket_id)
            if group is not None and not group.is_terminal:
                result = self._finalize(group, EndReason.SHUTDOWN, now)
                if result is not None:
                    results.append(result)
        return results

    # -------------------------------------------------------------- validation

    def _finalize(
        self, group: OrderGroup, reason: EndReason, now: float
    ) -> Optional[ValidationResult]:
        if group.is_terminal:
            return group.result
        # Freeze evidence first: a detection arriving after the end signal must
        # not change the verdict (section 14).
        group.evidence_frozen = True
        if not self._transition(group, LifecycleState.FINAL_VALIDATION, now):
            # Force the state so a ticket can never be stuck un-finalised.
            group.state = LifecycleState.FINAL_VALIDATION
            group.state_history.append((now, group.state))

        result = self.validate(group, reason, now)
        group.result = result
        group.finalized_at = now

        self._transition(
            group,
            LifecycleState.COMPLETED if result.correct else LifecycleState.WRONG,
            now,
        )
        if result.correct:
            self._completed.append(group)
            logger.info("ORDER COMPLETED - ticket %s CORRECT", group.ticket_id)
        else:
            self._warnings.append(group)
            logger.warning("WRONG ORDER - ticket %s: %s", group.ticket_id, result.message)

        self._emit(
            "finalized",
            {
                "ticket_id": group.ticket_id,
                "correct": result.correct,
                "result": result.to_dict(),
                "group": group.to_dict(),
            },
        )
        self._remove_from_queue(group.ticket_id)
        self._activate_next(now)
        return result

    def validate(
        self, group: OrderGroup, reason: EndReason, now: float
    ) -> ValidationResult:
        """Compare expected against detected and explain any difference."""
        expected = group.expected_counts()
        detected = dict(group.detected_counts)
        expected_total = sum(expected.values())
        detected_total = sum(detected.values())

        categories: List[FailureCategory] = []
        notes: List[str] = []
        missing: Dict[str, int] = {}
        extra: Dict[str, int] = {}

        # Quantity is the reliable signal, so it is judged on totals rather
        # than per-type: the model cannot tell one hotdog variant from another.
        if detected_total < expected_total:
            shortfall = expected_total - detected_total
            for hotdog in group.hotdogs:
                if hotdog.is_unknown or hotdog.remaining <= 0:
                    continue
                take = min(shortfall, hotdog.remaining)
                if take > 0:
                    missing[hotdog.item] = missing.get(hotdog.item, 0) + take
                    shortfall -= take
                if shortfall <= 0:
                    break
            if shortfall > 0:
                missing["hot-dog"] = missing.get("hot-dog", 0) + shortfall
            categories.append(FailureCategory.MISSING_ITEM)
            if detected_total > 0:
                categories.append(FailureCategory.INCOMPLETE_ORDER)
        elif detected_total > expected_total:
            surplus = detected_total - expected_total
            extra["hot-dog"] = surplus
            categories.append(FailureCategory.UNEXPECTED_ITEM)
            categories.append(FailureCategory.WRONG_QUANTITY)

        # Type check -- inferred only, never authoritative.
        type_mismatch = False
        for item, count in expected.items():
            got = detected.get(item, 0)
            if got < count and detected_total >= expected_total:
                type_mismatch = True
                notes.append(
                    "expected %d x %s but the production video could not confirm "
                    "that many of that type" % (count, item)
                )
        for item, count in detected.items():
            if item not in expected and item != "hot-dog":
                extra[item] = extra.get(item, 0) + count
                categories.append(FailureCategory.WRONG_ITEM_TYPE)
                type_mismatch = True
        if type_mismatch and FailureCategory.WRONG_ITEM_TYPE not in categories:
            if self.strict_type_matching:
                categories.append(FailureCategory.WRONG_ITEM_TYPE)

        if group.unknown_items():
            categories.append(FailureCategory.UNKNOWN_ITEM)
            notes.append(
                "unreadable shortcut(s) on the ticket: "
                + ", ".join(group.unknown_items())
            )

        if not self.strict_no_extras:
            categories = [c for c in categories if c is not FailureCategory.UNEXPECTED_ITEM]

        correct = not categories
        result = ValidationResult(
            correct=correct,
            end_reason=reason,
            expected=expected,
            detected=detected,
            missing=missing,
            extra=extra,
            categories=_dedupe(categories),
            type_match_confident=False,
            type_notes=notes,
            timestamp=now,
        )
        result.message = self._explain(group, result)
        return result

    @staticmethod
    def _explain(group: OrderGroup, result: ValidationResult) -> str:
        """Human-readable explanation (sections 12, 16)."""
        if result.correct:
            return "Ticket %s: CORRECT - %d/%d hotdogs verified (%s)" % (
                group.ticket_id,
                sum(result.detected.values()),
                sum(result.expected.values()),
                result.end_reason.value,
            )
        parts = []
        if result.missing:
            parts.append(
                "Missing " + ", ".join("%s x%d" % (k, v) for k, v in result.missing.items())
            )
        if result.extra:
            parts.append(
                "Unexpected " + ", ".join("%s x%d" % (k, v) for k, v in result.extra.items())
            )
        for note in result.type_notes:
            parts.append(note)
        if not parts:
            parts.append("order did not match the ticket")
        return "Ticket %s: WRONG - %s (%s)" % (
            group.ticket_id,
            "; ".join(parts),
            result.end_reason.value,
        )

    # ------------------------------------------------------------------ queries

    def _remove_from_queue(self, ticket_id: str) -> None:
        if ticket_id in self._queue:
            self._queue.remove(ticket_id)

    @property
    def active_group(self) -> Optional[OrderGroup]:
        for ticket_id in self._queue:
            group = self._groups.get(ticket_id)
            if group is not None and group.state in (
                LifecycleState.ACTIVE,
                LifecycleState.IN_PROGRESS,
                LifecycleState.WAITING_FOR_COMPLETION,
            ):
                return group
        return None

    @property
    def queue(self) -> List[OrderGroup]:
        """Every queued group, in strict FIFO order."""
        return [self._groups[t] for t in self._queue if t in self._groups]

    @property
    def completed(self) -> List[OrderGroup]:
        return list(self._completed)

    @property
    def warnings(self) -> List[OrderGroup]:
        return list(self._warnings)

    def get(self, ticket_id: str) -> Optional[OrderGroup]:
        return self._groups.get(ticket_id)

    def exists(self, ticket_id: str) -> bool:
        return ticket_id in self._groups

    def position_of(self, ticket_id: str) -> int:
        return self._queue.index(ticket_id) if ticket_id in self._queue else -1

    def dashboard_state(self) -> dict:
        """Payload for the live dashboard (section 19)."""
        labels = ["ACTIVE", "NEXT"]
        queue = []
        for index, group in enumerate(self.queue):
            label = labels[index] if index < len(labels) else "WAITING"
            entry = group.to_dict()
            entry["queue_label"] = label
            entry["queue_position"] = index
            queue.append(entry)
        active = self.active_group
        return {
            "active": active.to_dict() if active else None,
            "queue": queue,
            "completed": [g.to_dict() for g in self._completed[-20:]],
            "warnings": [g.to_dict() for g in self._warnings[-20:]],
            "counts": {
                "queued": len(self._queue),
                "completed": len(self._completed),
                "wrong": len(self._warnings),
                "buffered_detections": len(self._pending_detections),
                "unassigned_detections": self._surplus_detections,
            },
        }

    def progress_lines(self) -> List[str]:
        """``HOTDOG_A  1 / 2`` style progress for the active ticket."""
        group = self.active_group
        if group is None:
            return []
        return [
            "%-28s %d / %d" % (h.display or h.item, h.detected_count, h.quantity)
            for h in group.hotdogs
            if not h.is_unknown
        ]

    def reset(self) -> None:
        self._groups.clear()
        self._queue.clear()
        self._completed.clear()
        self._warnings.clear()
        self._sequence = 0
        self._pending_detections.clear()
        self._surplus_detections = 0


def _dedupe(items: Iterable) -> List:
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen
