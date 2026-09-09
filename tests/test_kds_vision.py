"""Tests for the KDS ticket reader, FIFO queue and order validation.

Covers the ten scenarios required by the specification (section 26).  Every
test drives the real pipeline -- parser, stability tracker, FIFO manager --
using a ``StubOcrEngine`` and synthetic card images, so no OCR model download
and no video file is needed.

An order is judged when its ticket DISAPPEARS from the KDS.  The card turning
pink is an overdue indicator only and never produces a verdict; see
TestVerdictOnlyOnDisappearance.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.kds.colors import ColorBands, load_visual_config
from src.kds.fifo_queue import TicketManager
from src.kds.ocr_engine import OcrBox, StubOcrEngine
from src.kds.schemas import (
    EndReason,
    FailureCategory,
    LifecycleState,
    PaymentStatus,
    RowColor,
    UNKNOWN_SHORTCUT,
)
from src.kds.shortcut_map import ShortcutMapper
from src.kds.stability import TicketStabilityTracker
from src.kds.ticket_parser import TicketParser

# ---------------------------------------------------------------------------
# Synthetic card rendering
#
# A card is drawn as coloured horizontal bars, one per row, in the exact BGR
# values the KDS uses.  The parser then classifies them through the real HSV
# bands from config/kds_visual.yaml, so colour handling is genuinely exercised.
# ---------------------------------------------------------------------------

CARD_W, ROW_H = 260, 40

BGR = {
    "white": (235, 240, 235),
    "yellow": (30, 225, 235),
    "grey": (170, 170, 170),
    "orange": (30, 120, 235),
    "cyan": (235, 235, 60),
    "blue": (200, 60, 40),
    "magenta": (210, 40, 210),
    "green_bar": (10, 10, 10),
    "pink": (200, 190, 240),
}


def make_card(rows, body="white"):
    """Render a card image plus the OCR boxes that match its rows.

    ``rows`` is a list of ``(text, colour_name)``.  Returns
    ``(image, [OcrBox, ...])``.
    """
    height = ROW_H * len(rows)
    image = np.zeros((height, CARD_W, 3), np.uint8)
    image[:, :] = BGR[body]
    boxes = []
    for index, (text, colour) in enumerate(rows):
        y1 = index * ROW_H
        y2 = y1 + ROW_H
        # "white" means "no highlight bar", i.e. the card body shows through --
        # which is how an overdue (pink) or warning (yellow) card renders.
        if colour == "white":
            colour = body
        if colour != body:
            image[y1:y2, :] = BGR[colour]
        if text:
            boxes.append(
                OcrBox(text=text, bbox=(12, y1 + 8, CARD_W - 12, y2 - 8), confidence=0.95)
            )
    return image, boxes


def make_parser(boxes_or_fn):
    """A TicketParser whose OCR always returns the given boxes.

    The stub receives the *upscaled* image, so boxes are rescaled to match.
    """
    config = load_visual_config()
    bands = ColorBands(config)

    def callback(image):
        boxes = boxes_or_fn() if callable(boxes_or_fn) else boxes_or_fn
        scale = image.shape[0] / float(_current_card_height[0])
        return [
            OcrBox(
                b.text,
                (
                    int(b.bbox[0] * scale),
                    int(b.bbox[1] * scale),
                    int(b.bbox[2] * scale),
                    int(b.bbox[3] * scale),
                ),
                b.confidence,
            )
            for b in boxes
        ]

    return TicketParser(StubOcrEngine(callback=callback), bands=bands)


_current_card_height = [ROW_H * 6]


def parse_card(rows, body="white"):
    image, boxes = make_card(rows, body)
    _current_card_height[0] = image.shape[0]
    parser = make_parser(boxes)
    return parser.parse(image, timestamp=0.0)


PAID_TICKET_ROWS = [
    ("CHK 1050", "white"),
    ("Dine In", "white"),
    ("2 ORG CHL", "yellow"),
    ("1 AB C/C", "yellow"),
    ("*** Paid *** 18.40", "green_bar"),
]

UNPAID_TICKET_ROWS = [
    ("CHK 1025", "white"),
    ("Dine In", "white"),
    ("1 ORG CHL", "yellow"),
    ("Subtotal 6.20", "green_bar"),
]


# ===========================================================================
# Test 1 -- unpaid ticket is ignored
# ===========================================================================

class TestUnpaidTicket:
    def test_unpaid_ticket_parses_as_not_paid(self):
        snapshot = parse_card(UNPAID_TICKET_ROWS)
        assert snapshot.ticket_id == "CHK 1025"
        assert snapshot.payment is PaymentStatus.NOT_PAID
        assert snapshot.is_paid is False

    def test_unpaid_ticket_never_enters_fifo(self):
        """RULE 1: an unpaid ticket creates no group and no queue entry."""
        snapshot = parse_card(UNPAID_TICKET_ROWS)
        manager = TicketManager()
        assert manager.create_from_paid_ticket(snapshot, now=1.0) is None
        assert manager.queue == []
        assert manager.active_group is None
        assert manager.exists("CHK 1025") is False

    def test_unpaid_ticket_is_not_validated(self):
        snapshot = parse_card(UNPAID_TICKET_ROWS)
        manager = TicketManager()
        manager.create_from_paid_ticket(snapshot, now=1.0)
        assert manager.on_ticket_disappeared("CHK 1025", now=2.0) is None
        assert manager.completed == []
        assert manager.warnings == []

    def test_stability_ignores_unpaid_however_many_frames(self):
        image, boxes = make_card(UNPAID_TICKET_ROWS)
        _current_card_height[0] = image.shape[0]
        parser = make_parser(boxes)
        tracker = TicketStabilityTracker(load_visual_config())
        manager = TicketManager()
        for frame in range(40):
            snapshot = parser.parse(image, timestamp=float(frame))
            for event in tracker.observe(1, snapshot):
                if event.snapshot is not None and event.snapshot.is_paid:
                    manager.create_from_paid_ticket(event.snapshot, now=float(frame))
        assert manager.queue == []


# ===========================================================================
# Test 2 -- paid ticket is created and queued
# ===========================================================================

class TestPaidTicket:
    def test_paid_ticket_parses(self):
        snapshot = parse_card(PAID_TICKET_ROWS)
        assert snapshot.ticket_id == "CHK 1050"
        assert snapshot.payment is PaymentStatus.PAID
        assert snapshot.total_hotdogs == 3

    def test_paid_ticket_creates_group_and_queues(self):
        snapshot = parse_card(PAID_TICKET_ROWS)
        manager = TicketManager()
        group = manager.create_from_paid_ticket(snapshot, now=10.0)
        assert group is not None
        assert group.ticket_id == "CHK 1050"
        assert group.internal_id == "OG-0001"
        assert group.paid_at == 10.0
        assert group.expected_total == 3
        # Queued and immediately promoted to the head of an empty queue.
        assert [g.ticket_id for g in manager.queue] == ["CHK 1050"]
        assert manager.active_group is group
        assert group.state is LifecycleState.ACTIVE

    def test_expected_counts_group_by_item(self):
        snapshot = parse_card(PAID_TICKET_ROWS)
        manager = TicketManager()
        group = manager.create_from_paid_ticket(snapshot, now=10.0)
        assert group.expected_counts() == {
            "ORIGINAL_CHILI_DOG": 2,
            "ALL_BEEF_CHILI_CHEESE_DOG": 1,
        }

    def test_becomes_ticket_for_existing_order_state_machine(self):
        """The group adapts to the Ticket shape the existing pipeline expects."""
        snapshot = parse_card(PAID_TICKET_ROWS)
        manager = TicketManager()
        group = manager.create_from_paid_ticket(snapshot, now=10.0)
        ticket = group.to_ticket()
        assert ticket.ticket_id == "CHK 1050"
        assert ticket.total_hotdogs == 3
        assert len(ticket.hotdog_specs) == 3
        assert "chilli" in ticket.hotdog_specs["hotdog1"]


# ===========================================================================
# Test 3 -- the same ticket over many frames creates exactly one order
# ===========================================================================

class TestDuplicateSuppression:
    def test_one_ticket_from_one_hundred_frames(self):
        image, boxes = make_card(PAID_TICKET_ROWS)
        _current_card_height[0] = image.shape[0]
        parser = make_parser(boxes)
        tracker = TicketStabilityTracker(load_visual_config())
        manager = TicketManager()

        created = 0
        for frame in range(100):
            snapshot = parser.parse(image, timestamp=float(frame))
            for event in tracker.observe(1, snapshot):
                if event.snapshot is not None and event.snapshot.is_paid:
                    if manager.create_from_paid_ticket(event.snapshot, now=float(frame)):
                        created += 1

        assert created == 1
        assert len(manager.queue) == 1
        assert manager.get("CHK 1050").internal_id == "OG-0001"

    def test_manager_refuses_a_second_create_directly(self):
        snapshot = parse_card(PAID_TICKET_ROWS)
        manager = TicketManager()
        assert manager.create_from_paid_ticket(snapshot, now=1.0) is not None
        assert manager.create_from_paid_ticket(snapshot, now=2.0) is None
        assert len(manager.queue) == 1

    def test_unpaid_to_paid_transition_creates_once(self):
        """Section 3: a ticket that becomes paid on screen is created once."""
        state = {"paid": False}
        rows_unpaid = UNPAID_TICKET_ROWS
        rows_paid = [
            ("CHK 1025", "white"),
            ("Dine In", "white"),
            ("1 ORG CHL", "yellow"),
            ("*** Paid *** 6.20", "green_bar"),
        ]
        tracker = TicketStabilityTracker(load_visual_config())
        manager = TicketManager()
        created = 0
        for frame in range(60):
            state["paid"] = frame >= 20
            rows = rows_paid if state["paid"] else rows_unpaid
            snapshot = parse_card(rows)
            for event in tracker.observe(1, snapshot):
                if event.snapshot is not None and event.snapshot.is_paid:
                    if manager.create_from_paid_ticket(event.snapshot, now=float(frame)):
                        created += 1
        assert created == 1
        assert manager.exists("CHK 1025")


# ===========================================================================
# Test 4 -- hotdog / add-on parent-child relationship
# ===========================================================================

class TestAddOnParenting:
    ROWS = [
        ("CHK 1050", "white"),
        ("Dine In", "white"),
        ("2 ORG CHL", "yellow"),
        ("Cheese", "grey"),
        ("Onion", "grey"),
        ("1 AB C/C", "yellow"),
        ("Jalapeno", "grey"),
        ("*** Paid *** 18.40", "green_bar"),
    ]

    def test_addons_attach_to_the_preceding_yellow_bar(self):
        snapshot = parse_card(self.ROWS)
        assert len(snapshot.hotdogs) == 2
        first, second = snapshot.hotdogs
        assert first.item == "ORIGINAL_CHILI_DOG"
        assert first.quantity == 2
        assert first.addon_names() == ["Cheese", "Onion"]
        assert second.item == "ALL_BEEF_CHILI_CHEESE_DOG"
        assert second.addon_names() == ["Jalapeno"]

    def test_addons_are_not_independent_order_items(self):
        """Section 10: an add-on is never promoted to a hotdog."""
        snapshot = parse_card(self.ROWS)
        items = [h.item for h in snapshot.hotdogs]
        assert "Cheese" not in items and "Onion" not in items
        # Two shortcut bars, quantities 2 and 1 -> three hotdogs, not five.
        assert snapshot.total_hotdogs == 3

    def test_orange_rows_are_addons_too(self):
        rows = [
            ("CHK 1051", "white"),
            ("2 ORG C/C", "yellow"),
            ("2 Onion", "orange"),
            ("*** Paid *** 9.00", "green_bar"),
        ]
        snapshot = parse_card(rows)
        assert len(snapshot.hotdogs) == 1
        addon = snapshot.hotdogs[0].addons[0]
        assert addon.display == "Onion"
        assert addon.quantity == 2

    def test_addon_under_a_non_hotdog_product_is_not_attached(self):
        """A modifier under a burger must not land on the hotdog above it."""
        rows = [
            ("CHK 1052", "white"),
            ("1 AB C/C", "yellow"),
            ("1 Dlx Ch Brg", "magenta"),
            ("No Must", "grey"),
            ("*** Paid *** 12.00", "green_bar"),
        ]
        snapshot = parse_card(rows)
        assert len(snapshot.hotdogs) == 1
        assert snapshot.hotdogs[0].addons == []

    def test_orphan_addon_without_a_parent_is_dropped(self):
        rows = [
            ("CHK 1053", "white"),
            ("Onion", "grey"),
            ("*** Paid *** 3.00", "green_bar"),
        ]
        snapshot = parse_card(rows)
        assert snapshot.hotdogs == []

    def test_addons_survive_into_the_order_group(self):
        snapshot = parse_card(self.ROWS)
        manager = TicketManager()
        group = manager.create_from_paid_ticket(snapshot, now=1.0)
        pairs = group.all_addons()
        assert ("ORIGINAL_CHILI_DOG", "Cheese") in [(i, a.display) for i, a in pairs]
        assert ("ALL_BEEF_CHILI_CHEESE_DOG", "Jalapeno") in [
            (i, a.display) for i, a in pairs
        ]

    def test_addons_are_never_marked_detected_without_evidence(self):
        """Section 10: no production evidence => no add-on completion."""
        snapshot = parse_card(self.ROWS)
        manager = TicketManager()
        group = manager.create_from_paid_ticket(snapshot, now=1.0)
        for _ in range(3):
            manager.record_detection("ORIGINAL_CHILI_DOG", now=2.0)
        for _, addon in group.all_addons():
            assert addon.detected is False
            assert addon.detected_count == 0

    def test_negation_addon_removes_the_ingredient_from_the_recipe(self):
        rows = [
            ("CHK 1054", "white"),
            ("1 ORG CHL", "yellow"),
            ("No Onion", "grey"),
            ("*** Paid *** 4.00", "green_bar"),
        ]
        snapshot = parse_card(rows)
        addon = snapshot.hotdogs[0].addons[0]
        assert addon.negation is True


# ===========================================================================
# Test 5 -- correct completion when the ticket disappears
# ===========================================================================

class TestCorrectCompletion:
    """An order is judged when its ticket leaves the KDS -- and only then."""

    def _paid_group(self, manager):
        return manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)

    def test_all_items_detected_then_disappearance_is_correct(self):
        manager = TicketManager()
        group = self._paid_group(manager)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=2.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=2, now=3.0)
        manager.record_detection("ALL_BEEF_CHILI_CHEESE_DOG", track_id=3, now=4.0)
        assert group.state is LifecycleState.WAITING_FOR_COMPLETION

        result = manager.on_ticket_disappeared("CHK 1050", now=5.0)
        assert result.correct is True
        assert result.end_reason is EndReason.TICKET_DISAPPEARED
        assert result.missing == {}
        assert group.state is LifecycleState.COMPLETED
        assert "CORRECT" in result.message

    def test_completed_ticket_leaves_the_queue_and_next_activates(self):
        manager = TicketManager()
        self._paid_group(manager)
        second = manager.create_from_paid_ticket(
            parse_card(
                [
                    ("CHK 1051", "white"),
                    ("1 ORG CHL", "yellow"),
                    ("*** Paid *** 5.00", "green_bar"),
                ]
            ),
            now=2.0,
        )
        for track in range(3):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=3.0)
        manager.on_ticket_disappeared("CHK 1050", now=4.0)

        assert [g.ticket_id for g in manager.queue] == ["CHK 1051"]
        assert manager.active_group is second
        assert len(manager.completed) == 1

    def test_evidence_is_frozen_at_final_validation(self):
        """A detection arriving after the verdict cannot rewrite history."""
        manager = TicketManager()
        group = self._paid_group(manager)
        for track in range(3):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=2.0)
        manager.on_ticket_disappeared("CHK 1050", now=3.0)
        before = dict(group.detected_counts)
        assert manager.record_detection("ORIGINAL_CHILI_DOG", track_id=99, now=4.0) is None
        assert group.detected_counts == before


# ===========================================================================
# Test 6 -- missing item when the ticket disappears is WRONG
# ===========================================================================

class TestMissingItem:
    def test_missing_hotdog_is_reported(self):
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=2.0)
        manager.record_detection("ALL_BEEF_CHILI_CHEESE_DOG", track_id=2, now=3.0)

        result = manager.on_ticket_disappeared("CHK 1050", now=4.0)
        assert result.correct is False
        assert result.missing == {"ORIGINAL_CHILI_DOG": 1}
        assert FailureCategory.MISSING_ITEM in result.categories
        assert FailureCategory.INCOMPLETE_ORDER in result.categories
        assert "Missing" in result.message
        assert manager.get("CHK 1050").state is LifecycleState.WRONG
        assert len(manager.warnings) == 1

    def test_nothing_detected_at_all(self):
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        result = manager.on_ticket_disappeared("CHK 1050", now=2.0)
        assert result.correct is False
        assert sum(result.missing.values()) == 3
        assert FailureCategory.MISSING_ITEM in result.categories

    def test_message_names_the_ticket_and_the_shortfall(self):
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=2.0)
        result = manager.on_ticket_disappeared("CHK 1050", now=3.0)
        assert "CHK 1050" in result.message
        assert "WRONG" in result.message


# ===========================================================================
# Test 7 -- ticket disappears before a successful match
# ===========================================================================

class TestTicketDisappearance:
    def test_disappearance_triggers_final_validation_then_removal(self):
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=2.0)

        result = manager.on_ticket_disappeared("CHK 1050", now=5.0)
        assert result is not None
        assert result.end_reason is EndReason.TICKET_DISAPPEARED
        assert result.correct is False
        assert sum(result.missing.values()) == 2
        # RULE 9: removed from active monitoring.
        assert manager.queue == []
        assert manager.active_group is None
        assert manager.get("CHK 1050").state is LifecycleState.WRONG

    def test_disappearance_after_success_finalises_normally(self):
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        for track in range(3):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=2.0)
        manager.on_ticket_disappeared("CHK 1050", now=3.0)
        assert manager.get("CHK 1050").state is LifecycleState.COMPLETED

        result = manager.on_ticket_disappeared("CHK 1050", now=4.0)
        assert result.correct is True
        assert manager.queue == []

    def test_grace_period_before_disappearance_is_declared(self):
        config = load_visual_config()
        tracker = TicketStabilityTracker(config)
        image, boxes = make_card(PAID_TICKET_ROWS)
        _current_card_height[0] = image.shape[0]
        parser = make_parser(boxes)
        for frame in range(10):
            tracker.observe(1, parser.parse(image, timestamp=float(frame)))
        # Absent, but only briefly -- still present as far as the system knows.
        assert tracker.sweep(now=10.5, present_slots=set()) == []
        events = tracker.sweep(now=20.0, present_slots=set())
        assert [e.ticket_id for e in events] == ["CHK 1050"]

    def test_disappeared_ticket_is_not_waited_on_forever(self):
        """Section 12: the system stops monitoring an order that no longer exists."""
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        manager.on_ticket_disappeared("CHK 1050", now=2.0)
        assert manager.record_detection("ORIGINAL_CHILI_DOG", track_id=9, now=3.0) is None


# ===========================================================================
# Test 8 -- multiple tickets are processed in FIFO order
# ===========================================================================

class TestFifoOrdering:
    def _make(self, manager, number, paid_at):
        rows = [
            ("CHK %d" % number, "white"),
            ("1 ORG CHL", "yellow"),
            ("*** Paid *** 5.00", "green_bar"),
        ]
        return manager.create_from_paid_ticket(parse_card(rows), now=paid_at)

    def test_queue_follows_paid_timestamps(self):
        manager = TicketManager()
        self._make(manager, 1001, 10.0)
        self._make(manager, 1002, 12.0)
        self._make(manager, 1003, 14.0)
        assert [g.ticket_id for g in manager.queue] == ["CHK 1001", "CHK 1002", "CHK 1003"]
        assert manager.active_group.ticket_id == "CHK 1001"

    def test_out_of_order_creation_still_queues_by_paid_time(self):
        """RULE 5: paid time decides, never screen position."""
        manager = TicketManager()
        self._make(manager, 1003, 14.0)
        self._make(manager, 1001, 10.0)
        self._make(manager, 1002, 12.0)
        assert [g.ticket_id for g in manager.queue] == ["CHK 1001", "CHK 1002", "CHK 1003"]

    def test_tickets_activate_one_after_another(self):
        manager = TicketManager()
        self._make(manager, 1001, 10.0)
        self._make(manager, 1002, 12.0)
        self._make(manager, 1003, 14.0)

        order = []
        for step in range(3):
            active = manager.active_group
            order.append(active.ticket_id)
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=step, now=20.0 + step)
            manager.on_ticket_disappeared(active.ticket_id, now=21.0 + step)
        assert order == ["CHK 1001", "CHK 1002", "CHK 1003"]
        assert len(manager.completed) == 3
        assert manager.queue == []

    def test_dashboard_labels_active_next_waiting(self):
        manager = TicketManager()
        self._make(manager, 1050, 10.0)
        self._make(manager, 1051, 11.0)
        self._make(manager, 1052, 12.0)
        self._make(manager, 1053, 13.0)
        labels = [entry["queue_label"] for entry in manager.dashboard_state()["queue"]]
        assert labels == ["ACTIVE", "NEXT", "WAITING", "WAITING"]

    def test_detections_go_to_the_active_ticket_only(self):
        manager = TicketManager()
        first = self._make(manager, 1001, 10.0)
        second = self._make(manager, 1002, 12.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=13.0)
        assert first.detected_total == 1
        assert second.detected_total == 0


# ===========================================================================
# Test 9 -- unknown shortcut
# ===========================================================================

class TestUnknownShortcut:
    ROWS = [
        ("CHK 1099", "white"),
        ("1 ZZZ TOP", "yellow"),
        ("*** Paid *** 5.00", "green_bar"),
    ]

    def test_unknown_shortcut_is_reported_not_mapped(self):
        snapshot = parse_card(self.ROWS)
        assert snapshot.unknown_shortcuts == ["1 ZZZ TOP"]
        assert len(snapshot.hotdogs) == 1
        assert snapshot.hotdogs[0].item == UNKNOWN_SHORTCUT
        assert snapshot.hotdogs[0].is_unknown is True

    def test_unknown_shortcut_is_excluded_from_expectations(self):
        manager = TicketManager()
        group = manager.create_from_paid_ticket(parse_card(self.ROWS), now=1.0)
        assert group.expected_counts() == {}
        assert group.unknown_items() == ["1 ZZZ TOP"]

    def test_unknown_shortcut_fails_validation_with_its_own_category(self):
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(self.ROWS), now=1.0)
        result = manager.on_ticket_disappeared("CHK 1099", now=2.0)
        assert result.correct is False
        assert FailureCategory.UNKNOWN_ITEM in result.categories
        assert "unreadable shortcut" in result.message

    def test_mapper_refuses_to_force_fit_a_neighbour(self):
        mapper = ShortcutMapper()
        match = mapper.resolve_shortcut("1 ZZZ TOP")
        assert match.known is False
        assert match.definition is None
        assert "ZZZ TOP" in mapper.unknown_report()

    def test_known_non_hotdog_lines_are_not_reported_as_unknown(self):
        rows = [
            ("CHK 1098", "white"),
            ("1 ORG CHL", "yellow"),
            ("5 CRN DOG", "white"),
            ("1 SM COKE", "cyan"),
            ("*** Paid *** 5.00", "green_bar"),
        ]
        snapshot = parse_card(rows)
        assert snapshot.unknown_shortcuts == []
        assert len(snapshot.hotdogs) == 1

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("1 Org C/C", "ORIGINAL_CHILI_CHEESE_DOG"),
            ("2 ORG CHL", "ORIGINAL_CHILI_DOG"),
            ("1 AB C/C", "ALL_BEEF_CHILI_CHEESE_DOG"),
            ("1 AB Must", "ALL_BEEF_MUSTARD_DOG"),
            ("2 Org Chicgo", "ORIGINAL_CHICAGO_DOG"),
            ("1 AB Kraut", "ALL_BEEF_KRAUT_DOG"),
            ("1ORG C/C", "ORIGINAL_CHILI_CHEESE_DOG"),
        ],
    )
    def test_every_configured_shortcut_resolves(self, text, expected):
        assert ShortcutMapper().resolve_shortcut(text).definition.item == expected


# ===========================================================================
# Test 10 -- temporary OCR failure
# ===========================================================================

class TestTemporaryOcrFailure:
    def test_identity_survives_frames_with_no_ocr_output(self):
        config = load_visual_config()
        bands = ColorBands(config)
        image, boxes = make_card(PAID_TICKET_ROWS)
        _current_card_height[0] = image.shape[0]

        state = {"blind": False}

        def callback(img):
            if state["blind"]:
                return []
            scale = img.shape[0] / float(image.shape[0])
            return [
                OcrBox(
                    b.text,
                    (
                        int(b.bbox[0] * scale),
                        int(b.bbox[1] * scale),
                        int(b.bbox[2] * scale),
                        int(b.bbox[3] * scale),
                    ),
                    b.confidence,
                )
                for b in boxes
            ]

        parser = TicketParser(StubOcrEngine(callback=callback), bands=bands)
        tracker = TicketStabilityTracker(config)
        manager = TicketManager()

        created = 0
        for frame in range(60):
            # OCR goes blind for 15 consecutive frames mid-ticket.
            state["blind"] = 20 <= frame < 35
            snapshot = parser.parse(
                image,
                timestamp=float(frame) * 0.1,
                fallback_ticket_id=tracker.ticket_for_slot(1) or None,
            )
            if state["blind"]:
                assert snapshot.readable is False
            for event in tracker.observe(1, snapshot):
                if event.snapshot is not None and event.snapshot.is_paid:
                    if manager.create_from_paid_ticket(event.snapshot, now=float(frame)):
                        created += 1
            # The slot is still present, so no disappearance is declared.
            assert tracker.sweep(float(frame) * 0.1, present_slots={1}) == []

        assert created == 1
        assert tracker.ticket_for_slot(1) == "CHK 1050"
        assert len(manager.queue) == 1

    def test_unreadable_snapshot_adds_no_votes(self):
        config = load_visual_config()
        tracker = TicketStabilityTracker(config)
        blind_parser = TicketParser(StubOcrEngine(callback=lambda img: []),
                                    bands=ColorBands(config))
        image, _ = make_card(PAID_TICKET_ROWS)
        for frame in range(30):
            snapshot = blind_parser.parse(image, timestamp=float(frame))
            assert tracker.observe(1, snapshot) == []
        assert tracker.ticket_for_slot(1) == ""


# ===========================================================================
# Cross-cutting: no early failure (RULE 6, section 17)
# ===========================================================================

class TestNoEarlyFailure:
    def test_partial_detection_does_not_fail_while_active(self):
        manager = TicketManager()
        group = manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=2.0)

        assert group.state is LifecycleState.IN_PROGRESS
        assert group.result is None
        assert manager.warnings == []
        assert manager.completed == []
        assert group.is_terminal is False

    def test_progress_is_reported_as_a_ratio_not_a_verdict(self):
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=2.0)
        lines = manager.progress_lines()
        assert any("1 / 2" in line for line in lines)
        assert any("0 / 1" in line for line in lines)

    def test_reaching_the_count_waits_rather_than_completing(self):
        manager = TicketManager()
        group = manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        for track in range(3):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=2.0)
        assert group.state is LifecycleState.WAITING_FOR_COMPLETION
        assert group.result is None


class TestLifecycleIntegrity:
    def test_illegal_transitions_are_refused(self):
        manager = TicketManager()
        group = manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        assert manager._transition(group, LifecycleState.NEW, 2.0) is False
        assert group.state is LifecycleState.ACTIVE

    def test_a_ticket_is_validated_only_once(self):
        """Re-reporting a disappearance returns the cached verdict, not a new one."""
        manager = TicketManager()
        group = manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        first = manager.on_ticket_disappeared("CHK 1050", now=2.0)
        second = manager.on_ticket_disappeared("CHK 1050", now=3.0)
        assert first is not None
        assert second is first, "must not re-judge an order that is already decided"
        assert group.finalized_at == 2.0, "the verdict keeps its original timestamp"
        assert len(manager.warnings) == 1

    def test_state_history_records_the_whole_lifecycle(self):
        manager = TicketManager()
        group = manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        for track in range(3):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=2.0)
        manager.on_ticket_disappeared("CHK 1050", now=3.0)
        states = [state for _, state in group.state_history]
        assert states[0] is LifecycleState.PAID
        assert LifecycleState.QUEUED in states
        assert LifecycleState.ACTIVE in states
        assert LifecycleState.IN_PROGRESS in states
        assert LifecycleState.FINAL_VALIDATION in states
        assert states[-1] is LifecycleState.COMPLETED

    def test_type_verdicts_are_never_claimed_as_confident(self):
        """The model has no hotdog-type class, so type is inference only."""
        manager = TicketManager()
        manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        for track in range(3):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=2.0)
        result = manager.on_ticket_disappeared("CHK 1050", now=3.0)
        assert result.type_match_confident is False

    def test_duplicate_track_ids_are_counted_once(self):
        manager = TicketManager()
        group = manager.create_from_paid_ticket(parse_card(PAID_TICKET_ROWS), now=1.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=7, now=2.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=7, now=2.5)
        assert group.detected_total == 1


class TestColorSemantics:
    def test_each_bar_colour_maps_to_its_role(self):
        rows = [
            ("1 ORG CHL", "yellow"),
            ("Onion", "grey"),
            ("2 Onion", "orange"),
            ("1 SM COKE", "cyan"),
            ("1 TENDER 2PK", "blue"),
            ("1 Dlx Ch Brg", "magenta"),
        ]
        image, boxes = make_card(rows)
        bands = ColorBands(load_visual_config())
        expected = [
            RowColor.YELLOW,
            RowColor.GREY,
            RowColor.ORANGE,
            RowColor.CYAN,
            RowColor.BLUE,
            RowColor.MAGENTA,
        ]
        for box, want in zip(boxes, expected):
            got, _ = bands.classify_row(image, box.bbox)
            assert got is want, "%r classified as %s, expected %s" % (
                box.text,
                got,
                want,
            )

    def test_pink_card_body_is_measured_as_a_fraction(self):
        image, _ = make_card(PAID_TICKET_ROWS, body="pink")
        bands = ColorBands(load_visual_config())
        assert bands.pink_fraction(image) > 0.3
        assert bands.is_pink_on(image) is True

    def test_white_card_body_is_not_pink(self):
        image, _ = make_card(PAID_TICKET_ROWS, body="white")
        bands = ColorBands(load_visual_config())
        assert bands.is_pink_on(image) is False


class TestSurplusDetectionBuffering:
    """A satisfied ticket must not absorb the rest of the shift.

    Production runs ahead of the KDS queue, so hotdogs made after the active
    ticket already has everything it asked for belong to the NEXT ticket.
    Without this, a long production video piles every hotdog onto one ticket
    and the final verdict is meaningless.
    """

    def _ticket(self, manager, number, qty, paid_at):
        rows = [
            ("CHK %d" % number, "white"),
            ("%d ORG CHL" % qty, "yellow"),
            ("*** Paid *** 5.00", "green_bar"),
        ]
        return manager.create_from_paid_ticket(parse_card(rows), now=paid_at)

    def test_satisfied_ticket_stops_absorbing_detections(self):
        manager = TicketManager()
        group = self._ticket(manager, 1001, 2, 10.0)
        for track in range(2):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=11.0)
        assert group.detected_total == 2

        # Six more hotdogs come off the line with no next ticket queued yet.
        for track in range(2, 8):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=12.0)
        assert group.detected_total == 2, "surplus must not inflate a satisfied ticket"
        assert manager.dashboard_state()["counts"]["buffered_detections"] == 6

    def test_buffered_detections_replay_onto_the_next_ticket(self):
        manager = TicketManager()
        first = self._ticket(manager, 1001, 1, 10.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=11.0)
        # Worker starts the next order before the current ticket is bumped.
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=2, now=12.0)
        assert first.detected_total == 1

        second = self._ticket(manager, 1002, 1, 13.0)
        assert second.state is LifecycleState.QUEUED
        manager.on_ticket_disappeared("CHK 1001", now=14.0)

        assert first.result.correct is True
        assert manager.active_group is second
        assert second.detected_total == 1, "the buffered hotdog belongs to #1002"

    def test_replay_never_overfills_the_next_ticket(self):
        manager = TicketManager()
        self._ticket(manager, 1001, 1, 10.0)
        for track in range(1, 8):
            manager.record_detection("ORIGINAL_CHILI_DOG", track_id=track, now=11.0)
        second = self._ticket(manager, 1002, 2, 12.0)
        manager.on_ticket_disappeared("CHK 1001", now=13.0)
        assert second.detected_total == 2

    def test_detections_with_no_ticket_are_counted_not_lost(self):
        manager = TicketManager()
        for track in range(4):
            assert manager.record_detection("hot-dog", track_id=track, now=1.0) is None
        assert manager.dashboard_state()["counts"]["unassigned_detections"] == 4

    def test_explicit_ticket_id_still_overrides_the_buffer(self):
        """Replay and any deliberate attribution bypass the satisfied check."""
        manager = TicketManager()
        group = self._ticket(manager, 1001, 1, 10.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=11.0)
        manager.record_detection(
            "ORIGINAL_CHILI_DOG", track_id=2, now=12.0, ticket_id="CHK 1001"
        )
        assert group.detected_total == 2


class TestVerdictOnlyOnDisappearance:
    """The card turning pink must never produce a verdict.

    On the live KDS the card body escalates white -> yellow -> pink purely with
    age, so a pink card means the order is running LATE, not that it is done.
    The order is judged only when the ticket leaves the screen.
    """

    def test_no_blink_detector_remains_in_the_pipeline(self):
        import importlib

        import src.kds.kds_monitor as monitor_module

        assert not hasattr(monitor_module, "BlinkDetector")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("src.kds.blink_detector")

    def test_manager_has_no_blink_entry_point(self):
        manager = TicketManager()
        assert not hasattr(manager, "on_blink_confirmed")

    def test_end_reasons_are_disappearance_or_shutdown_only(self):
        assert {r.value for r in EndReason} == {"TICKET_DISAPPEARED", "SHUTDOWN"}

    def test_a_pink_card_still_parses_normally(self):
        """An overdue card is still read; it is just not treated as finished."""
        snapshot = parse_card(PAID_TICKET_ROWS, body="pink")
        assert snapshot.ticket_id == "CHK 1050"
        assert snapshot.payment is PaymentStatus.PAID
        assert snapshot.pink_fraction > 0.3
        assert snapshot.total_hotdogs == 3

    def test_an_overdue_ticket_is_not_finalised_while_present(self):
        manager = TicketManager()
        group = manager.create_from_paid_ticket(
            parse_card(PAID_TICKET_ROWS, body="pink"), now=1.0
        )
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=2.0)
        # However long it sits there overdue, no verdict is produced.
        for t in range(3, 60):
            manager.note_kds_presence("CHK 1050", now=float(t))
        assert group.result is None
        assert group.is_terminal is False
        assert manager.completed == [] and manager.warnings == []

        result = manager.on_ticket_disappeared("CHK 1050", now=99.0)
        assert result is not None
        assert result.end_reason is EndReason.TICKET_DISAPPEARED


class TestFailureRecorder:
    """Every ticket is recorded; only the failures are kept.

    Whether an order is wrong is known only when its ticket disappears, long
    after the interesting footage happened -- so recording starts at ticket
    creation and the clip is deleted if the order turns out CORRECT.
    """

    def _frame(self, w=320, h=180):
        return np.full((h, w, 3), 40, np.uint8)

    def _recorder(self, tmp_path):
        from src.kds.failure_recorder import FailureRecorder

        return FailureRecorder(output_dir=str(tmp_path / "failures"), fps=10.0)

    def test_correct_order_clip_is_deleted(self, tmp_path):
        rec = self._recorder(tmp_path)
        rec.start("CHK 1", self._frame())
        for _ in range(10):
            rec.write(self._frame())
        assert rec.active == 1
        assert rec.finish("CHK 1", correct=True) is None
        assert rec.active == 0
        assert rec.discarded == 1 and rec.kept == 0
        assert list((tmp_path / "failures").glob("*.mp4")) == []

    def test_wrong_order_clip_is_kept_with_a_sidecar(self, tmp_path):
        rec = self._recorder(tmp_path)
        rec.start("CHK 2", self._frame())
        for _ in range(10):
            rec.write(self._frame())
        path = rec.finish(
            "CHK 2",
            correct=False,
            detail="Missing ORIGINAL_CHILI_DOG x1",
            result={"missing": {"ORIGINAL_CHILI_DOG": 1}},
        )
        assert path is not None and path.exists() and path.stat().st_size > 0
        assert rec.kept == 1 and rec.discarded == 0

        import json

        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta["ticket_id"] == "CHK 2"
        assert meta["frames"] == 10
        assert "Missing" in meta["detail"]
        assert meta["result"]["missing"] == {"ORIGINAL_CHILI_DOG": 1}

    def test_several_tickets_record_at_once(self, tmp_path):
        rec = self._recorder(tmp_path)
        rec.start("CHK 1", self._frame())
        rec.start("CHK 2", self._frame())
        for _ in range(6):
            rec.write(self._frame())
        rec.finish("CHK 1", correct=True)
        rec.finish("CHK 2", correct=False)
        assert rec.kept == 1 and rec.discarded == 1
        assert len(list((tmp_path / "failures").glob("*.mp4"))) == 1

    def test_frames_of_a_different_size_are_resized_not_dropped(self, tmp_path):
        rec = self._recorder(tmp_path)
        rec.start("CHK 3", self._frame(320, 180))
        rec.write(self._frame(640, 360))
        rec.write(self._frame(320, 180))
        path = rec.finish("CHK 3", correct=False)
        assert path is not None and path.exists()

    def test_unfinished_clips_are_kept_on_shutdown(self, tmp_path):
        """An order that never got a verdict is unverified, so keep the footage."""
        rec = self._recorder(tmp_path)
        rec.start("CHK 4", self._frame())
        rec.write(self._frame())
        rec.close()
        assert rec.active == 0
        assert len(list((tmp_path / "failures").glob("*.mp4"))) == 1

    def test_disabled_recorder_writes_nothing(self, tmp_path):
        from src.kds.failure_recorder import FailureRecorder

        rec = FailureRecorder(output_dir=str(tmp_path / "off"), enabled=False)
        rec.start("CHK 5", self._frame())
        rec.write(self._frame())
        assert rec.active == 0
        assert rec.finish("CHK 5", correct=False) is None

    def test_finishing_an_unknown_ticket_is_harmless(self, tmp_path):
        rec = self._recorder(tmp_path)
        assert rec.finish("nope", correct=False) is None


class TestDashboardComposition:
    def test_both_feeds_are_placed_side_by_side(self):
        from src.kds.failure_recorder import compose_dashboard_frame

        production = np.zeros((720, 1280, 3), np.uint8)
        kds = np.zeros((1024, 1280, 3), np.uint8)
        out = compose_dashboard_frame(production, kds, status="TICKET CHK 1", target_height=360)
        assert out is not None
        # Two panels scaled to the same height, plus the status strip.
        assert out.shape[0] == 360 + 34
        assert out.shape[1] == 640 + 450  # 1280*360/720 and 1280*360/1024

    def test_one_missing_feed_still_composes(self):
        from src.kds.failure_recorder import compose_dashboard_frame

        out = compose_dashboard_frame(np.zeros((720, 1280, 3), np.uint8), None, target_height=360)
        assert out is not None and out.shape[1] == 640

    def test_no_feeds_returns_none(self):
        from src.kds.failure_recorder import compose_dashboard_frame

        assert compose_dashboard_frame(None, None) is None


class TestIdleGate:
    """`has_screen_content` is what src/main.py gates detection on.

    Detection runs while anything is on the KDS screen and stops entirely when
    the screen is blank; the paid-ticket gate below is the second half of that
    condition, which keeps the detector running after a card scrolls away but
    before its order is finalised.
    """

    def _paid(self, manager, number, qty=1, at=1.0):
        rows = [
            ("CHK %d" % number, "white"),
            ("%d ORG CHL" % qty, "yellow"),
            ("*** Paid *** 5.00", "green_bar"),
        ]
        return manager.create_from_paid_ticket(parse_card(rows), now=at)

    def test_no_tickets_means_idle(self):
        manager = TicketManager()
        assert manager.active_group is None

    def test_a_paid_ticket_leaves_idle(self):
        manager = TicketManager()
        self._paid(manager, 1001)
        assert manager.active_group is not None

    def test_queued_tickets_still_count_as_busy(self):
        manager = TicketManager()
        self._paid(manager, 1001, at=1.0)
        self._paid(manager, 1002, at=2.0)
        manager.record_detection("ORIGINAL_CHILI_DOG", track_id=1, now=3.0)
        manager.on_ticket_disappeared("CHK 1001", now=4.0)
        # #1002 is promoted, so the pipeline must not go idle.
        assert manager.active_group is not None
        assert manager.active_group.ticket_id == "CHK 1002"

    def test_idle_again_once_every_ticket_is_finalised(self):
        manager = TicketManager()
        self._paid(manager, 1001)
        manager.on_ticket_disappeared("CHK 1001", now=5.0)
        assert manager.active_group is None

    # ---------------------------------------------------------- screen gate

    class _FakeMonitor:
        def __init__(self, cards):
            self.last_card_count = cards

    def _client(self, cards, active):
        """A KDSVideoClient stand-in exercising the real property."""
        from src.kds.kds_video_client import KDSVideoClient

        client = KDSVideoClient.__new__(KDSVideoClient)
        client.monitor = self._FakeMonitor(cards)
        manager = TicketManager()
        if active:
            self._paid(manager, 1001)
        client.manager = manager
        return client

    def test_blank_screen_is_idle(self):
        assert self._client(cards=0, active=False).has_screen_content is False

    def test_an_unpaid_card_already_wakes_detection(self):
        """The gate is card presence, not payment.

        Production regularly starts before the ticket is confirmed paid, so
        waiting for PAID would miss the first hotdogs of the order.
        """
        assert self._client(cards=1, active=False).has_screen_content is True

    def test_a_live_order_keeps_detection_on_with_no_card_visible(self):
        assert self._client(cards=0, active=True).has_screen_content is True


class TestRecorderStorageLayout:
    def test_in_progress_clips_live_inside_the_output_dir(self, tmp_path):
        """Keeping a failure is a rename, and a rename cannot cross drives.

        %TEMP% is regularly on C: while output/ is on D:, so staging clips in
        the system temp dir loses every failure recording at the final move.
        """
        from src.kds.failure_recorder import FailureRecorder

        out = tmp_path / "failures"
        rec = FailureRecorder(output_dir=str(out), fps=10.0)
        assert rec._tmp_dir.parent == rec.output_dir
        assert rec._tmp_dir.exists()

    def test_pending_dir_is_emptied_as_clips_resolve(self, tmp_path):
        from src.kds.failure_recorder import FailureRecorder

        out = tmp_path / "failures"
        rec = FailureRecorder(output_dir=str(out), fps=10.0)
        frame = np.full((120, 160, 3), 40, np.uint8)
        rec.start("CHK A", frame)
        rec.start("CHK B", frame)
        rec.write(frame)
        assert len(list(rec._tmp_dir.glob("*.mp4"))) == 2
        rec.finish("CHK A", correct=True)
        rec.finish("CHK B", correct=False)
        assert list(rec._tmp_dir.glob("*.mp4")) == []
        assert len(list(out.glob("*.mp4"))) == 1
