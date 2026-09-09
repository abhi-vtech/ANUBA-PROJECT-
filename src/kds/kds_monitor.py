"""Per-frame KDS orchestration.

Wires the pieces together for one KDS frame::

    frame -> TicketCardDetector -> TicketParser -> TicketStabilityTracker
                                \\-> BlinkDetector          |
                                                            v
                                                      TicketManager
                                                      (FIFO + validation)

Nothing here decides anything by itself: card detection supplies presence,
parsing supplies content, the stability tracker supplies confirmation, and the
manager owns the lifecycle.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from src.kds.colors import ColorBands, load_visual_config
from src.kds.fifo_queue import TicketManager
from src.kds.ocr_engine import OcrEngine, build_ocr_engine
from src.kds.overlay import annotate_kds_frame
from src.kds.schemas import KdsEvent, KdsEventType, TicketSnapshot
from src.kds.shortcut_map import ShortcutMapper
from src.kds.stability import TicketStabilityTracker
from src.kds.ticket_detector import TicketCardDetector
from src.kds.ticket_parser import TicketParser
from src.kds.timeline import EventTimeline

logger = logging.getLogger(__name__)


class KdsMonitor:
    """Reads KDS frames and keeps :class:`TicketManager` up to date."""

    def __init__(
        self,
        ocr: Optional[OcrEngine] = None,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
        manager: Optional[TicketManager] = None,
        timeline: Optional[EventTimeline] = None,
        mapper: Optional[ShortcutMapper] = None,
    ):
        self.config = config if config is not None else load_visual_config(config_path)
        self.bands = ColorBands(self.config)
        self.ocr = ocr if ocr is not None else build_ocr_engine(self.config)
        self.mapper = mapper or ShortcutMapper()
        self.detector = TicketCardDetector(self.bands)
        self.parser = TicketParser(self.ocr, bands=self.bands, mapper=self.mapper)
        self.stability = TicketStabilityTracker(self.config)
        self.timeline = timeline or EventTimeline()
        self.manager = manager or TicketManager()

        ocr_cfg = self.config.get("ocr", {}) or {}
        self.frame_stride = max(1, int(ocr_cfg.get("frame_stride", 3)))
        sync_cfg = self.config.get("sync", {}) or {}
        self.kds_offset_s = float(sync_cfg.get("kds_offset_s", 0.0))

        # One source of truth for the payment gate: the manager refuses unpaid
        # tickets independently, so it must be told what the stability layer
        # decided or a ticket is admitted upstream and dropped downstream.
        self.manager.require_paid = self.stability.require_paid

        self._frame_index = 0
        self._last_snapshots: Dict[str, TicketSnapshot] = {}
        self._events: List[KdsEvent] = []
        # Set by the caller to receive the annotated KDS frame each frame; used
        # to feed the dashboard's live KDS video panel.
        self.on_annotated_frame: Optional[Callable[[np.ndarray], None]] = None
        self.annotate: bool = True
        self.last_annotated_frame: Optional[np.ndarray] = None
        # The most recently judged order.  An order is judged when its ticket
        # leaves the screen, so by then there is no card left to annotate --
        # the verdict is shown as a banner on the feed instead.
        self.last_verdict: Optional[dict] = None
        # Number of ticket cards seen on the most recent frame.  src/main.py
        # idles the production detector whenever the KDS screen is empty, and
        # "empty" means no card at all -- not merely no *paid* ticket.
        self.last_card_count: int = 0

    # ------------------------------------------------------------------ update

    def process_frame(self, frame: np.ndarray, timestamp: float) -> List[KdsEvent]:
        """Process one KDS frame; returns the events confirmed on this frame."""
        self._frame_index += 1
        now = timestamp + self.kds_offset_s
        events: List[KdsEvent] = []

        cards = self.detector.detect(frame, now)
        present_slots = {card.slot_id for card in cards}
        self.last_card_count = len(cards)

        # OCR is expensive and temporal aggregation makes it unnecessary on
        # every frame.  Card *presence* is still checked every frame, because
        # presence is what decides when an order ends.
        run_ocr = (self._frame_index % self.frame_stride) == 0

        for card in cards:
            ticket_id = self.stability.ticket_for_slot(card.slot_id)

            if not run_ocr:
                continue

            snapshot = self.parser.parse(
                card.image,
                timestamp=now,
                frame_index=self._frame_index,
                bbox=card.bbox,
                fallback_ticket_id=ticket_id or None,
            )
            events.extend(self.stability.observe(card.slot_id, snapshot))

            resolved = self.stability.ticket_for_slot(card.slot_id)
            if resolved:
                self.detector.bind_ticket(card.slot_id, resolved)
                self._last_snapshots[resolved] = snapshot
                self.manager.note_kds_presence(resolved, now)

        events.extend(self.stability.sweep(now, present_slots))
        self.detector.prune(now, self.stability.disappear_grace_s * 4)

        for event in events:
            self._apply(event, now)
        self._events.extend(events)

        # Publish the annotated screen last, so the overlay shows the state
        # this frame produced rather than the previous one.
        if self.annotate:
            try:
                annotated = annotate_kds_frame(
                    frame, cards, self.manager, self.stability, self.last_verdict
                )
                self.last_annotated_frame = annotated
                if self.on_annotated_frame is not None:
                    self.on_annotated_frame(annotated)
            except Exception:
                logger.exception("KDS overlay rendering failed")
        return events

    # ------------------------------------------------------------------ routing

    def _apply(self, event: KdsEvent, now: float) -> None:
        """Turn a confirmed KDS event into a lifecycle action."""
        ticket_id = event.ticket_id

        if event.type is KdsEventType.TICKET_SEEN:
            self.timeline.add("ticket_detected", "Ticket %s detected" % ticket_id, ticket_id, now)

        elif event.type is KdsEventType.TICKET_UNPAID:
            # RULE 1: seen but not paid -> not in FIFO, no group, no validation.
            self.timeline.add(
                "payment", "Payment = NOT PAID for %s (ignored)" % ticket_id, ticket_id, now
            )

        elif event.type is KdsEventType.TICKET_PAID:
            # This event means "the ticket has entered the system".  Whether
            # payment gates that is config (stability.require_paid), so report
            # the payment actually read rather than assuming PAID.
            payment = (
                event.snapshot.payment.value
                if event.snapshot is not None
                else "UNKNOWN"
            )
            self.timeline.add(
                "payment", "Payment = %s for %s" % (payment, ticket_id), ticket_id, now
            )
            if event.snapshot is not None:
                group = self.manager.create_from_paid_ticket(event.snapshot, now)
                if group is not None:
                    self.timeline.add(
                        "ticket_created",
                        "Ticket %s created (%s) - %d hotdog(s), added to FIFO"
                        % (ticket_id, group.internal_id, group.expected_total),
                        ticket_id,
                        now,
                    )

        elif event.type is KdsEventType.TICKET_UPDATED:
            if event.snapshot is not None and self.manager.update_content(
                ticket_id, event.snapshot.hotdogs, now
            ):
                self.timeline.add(
                    "ticket_updated", "Ticket %s content updated" % ticket_id, ticket_id, now
                )

        elif event.type is KdsEventType.TICKET_DISAPPEARED:
            self.timeline.add(
                "disappeared",
                "Ticket %s disappeared from the KDS (%s)" % (ticket_id, event.detail),
                ticket_id,
                now,
            )
            result = self.manager.on_ticket_disappeared(ticket_id, now)
            self._log_result(ticket_id, result, now)

        elif event.type is KdsEventType.UNKNOWN_SHORTCUT:
            self.timeline.add(
                "unknown_shortcut",
                "UNKNOWN SHORTCUT on ticket %s: %r (not mapped to any item)"
                % (ticket_id, event.detail),
                ticket_id,
                now,
            )

    def _log_result(self, ticket_id: str, result, now: float) -> None:
        if result is None:
            return
        self.last_verdict = {
            "ticket_id": ticket_id,
            "correct": result.correct,
            "detail": self._verdict_detail(result),
            "message": result.message,
            "shown_at": time.monotonic(),
        }
        self.timeline.add(
            "validation",
            result.message,
            ticket_id,
            now,
            correct=result.correct,
            categories=[c.value for c in result.categories],
        )
        following = self.manager.active_group
        if following is not None:
            self.timeline.add(
                "activated",
                "Ticket %s becomes ACTIVE" % following.ticket_id,
                following.ticket_id,
                now,
            )

    @staticmethod
    def _verdict_detail(result) -> str:
        """One short line naming what was expected and what was missing."""
        expected = sum(result.expected.values())
        detected = sum(result.detected.values())
        detail = "%d of %d hotdogs verified" % (detected, expected)
        if result.missing:
            detail += "   missing " + ", ".join(
                "%s x%d" % (name, count) for name, count in result.missing.items()
            )
        if result.extra:
            detail += "   extra " + ", ".join(
                "%s x%d" % (name, count) for name, count in result.extra.items()
            )
        return detail

    # ------------------------------------------------------------------ access

    def finalize_all(self, now: Optional[float] = None) -> None:
        for result in self.manager.finalize_all(now):
            self.timeline.add("validation", result.message, "", now, correct=result.correct)

    def snapshot_for(self, ticket_id: str) -> Optional[TicketSnapshot]:
        return self._last_snapshots.get(ticket_id)

    def dashboard_state(self) -> dict:
        state = self.manager.dashboard_state()
        state["timeline"] = self.timeline.recent(40)
        state["unknown_shortcuts"] = self.mapper.unknown_report()
        return state

    def reset(self) -> None:
        self.detector.reset()
        self.stability.reset()
        self.manager.reset()
        self._frame_index = 0
        self._last_snapshots.clear()
        self._events.clear()
        self.last_verdict = None
