"""Data structures for KDS ticket parsing and order-group lifecycle.

These are deliberately separate from ``src/schemas.py``: that module describes
the *production* side (Detection, Action, Order) and is consumed by the
existing pipeline.  This module describes the *KDS* side.  The bridge between
them is :meth:`OrderGroup.to_ticket`, which produces a plain
``src.schemas.Ticket`` so ``OrderStateMachine`` keeps working untouched.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.schemas import LineItem, Ticket

BBox = Tuple[int, int, int, int]

# Sentinel used everywhere a yellow bar held text we do not recognise.  It is
# never mapped onto a real item (RULE 4).
UNKNOWN_SHORTCUT = "UNKNOWN_SHORTCUT"


# ---------------------------------------------------------------------------
# Colour roles
# ---------------------------------------------------------------------------

class RowColor(str, Enum):
    """Background colour role of one text row inside a ticket card."""

    YELLOW = "yellow"          # hotdog shortcut bar (RULE 2)
    GREY = "grey"              # add-on belonging to the bar above it (RULE 3)
    ORANGE = "orange"          # add-on rendered orange instead of grey
    BLUE = "blue"              # non-hotdog product bar (tenders) -- ignored
    MAGENTA = "magenta"        # non-hotdog product bar (burger) -- ignored
    CYAN = "cyan"              # drink line -- ignored
    PAID_GREEN = "paid_green"  # the subtotal / paid bar
    PINK = "pink"              # pale pink CARD BODY (overdue, informational)
    PLAIN = "plain"            # unhighlighted line (fries, corn dog, header)


#: Row colours that mean "this line is an add-on of the item above it".
ADDON_ROLES = (RowColor.GREY, RowColor.ORANGE)

#: Row colours that mean "this line is a product we deliberately ignore".
IGNORED_ITEM_ROLES = (RowColor.CYAN, RowColor.BLUE, RowColor.MAGENTA)


class PaymentStatus(str, Enum):
    PAID = "PAID"
    NOT_PAID = "NOT_PAID"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Parsed ticket content
# ---------------------------------------------------------------------------

@dataclass
class TicketLine:
    """One OCR'd row of a ticket card, with its colour role resolved."""

    text: str
    color: RowColor
    bbox: BBox
    confidence: float = 0.0
    quantity: int = 1
    # Text with the leading quantity stripped, e.g. "2 ORG CHL" -> "ORG CHL".
    body: str = ""

    @property
    def is_hotdog_bar(self) -> bool:
        return self.color is RowColor.YELLOW

    @property
    def is_addon(self) -> bool:
        return self.color in ADDON_ROLES


@dataclass
class AddOn:
    """A grey line attached to its parent hotdog (RULE 3).

    ``detected`` stays False unless the production pipeline supplies real
    evidence.  We never fabricate add-on completion.
    """

    key: str
    display: str
    quantity: int = 1
    ingredient: Optional[str] = None
    negation: bool = False
    raw_text: str = ""
    detected: bool = False
    detected_count: int = 0

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "display": self.display,
            "quantity": self.quantity,
            "ingredient": self.ingredient,
            "negation": self.negation,
            "raw_text": self.raw_text,
            "detected": self.detected,
            "detected_count": self.detected_count,
        }


@dataclass
class HotdogGroup:
    """One yellow shortcut bar plus every grey add-on beneath it.

    This is the parent-child structure the whole architecture preserves: an
    add-on is never promoted to an independent order item (RULE 3, section 10).
    """

    shortcut: str            # canonical shortcut text, e.g. "ORG C/C"
    item: str                # physical item, e.g. "ORIGINAL_CHILI_CHEESE_DOG"
    display: str = ""
    quantity: int = 1
    addons: List[AddOn] = field(default_factory=list)
    ingredients: List[str] = field(default_factory=list)
    raw_text: str = ""
    known: bool = True       # False => UNKNOWN_SHORTCUT

    # Populated during live monitoring / final validation.
    detected_count: int = 0
    matched_track_ids: List[int] = field(default_factory=list)
    detection_confidence: float = 0.0

    @property
    def is_unknown(self) -> bool:
        return not self.known or self.item == UNKNOWN_SHORTCUT

    @property
    def remaining(self) -> int:
        return max(0, self.quantity - self.detected_count)

    def addon_names(self) -> List[str]:
        return [a.display for a in self.addons]

    def to_dict(self) -> dict:
        return {
            "shortcut": self.shortcut,
            "item": self.item,
            "display": self.display,
            "quantity": self.quantity,
            "detected_count": self.detected_count,
            "remaining": self.remaining,
            "known": self.known,
            "ingredients": list(self.ingredients),
            "addons": [a.to_dict() for a in self.addons],
            "matched_track_ids": list(self.matched_track_ids),
            "detection_confidence": round(self.detection_confidence, 3),
            "raw_text": self.raw_text,
        }


@dataclass
class TicketSnapshot:
    """What one ticket card looked like on ONE observation.

    Snapshots are the raw per-observation input to temporal aggregation; they
    are never acted on individually (section 21).
    """

    ticket_id: str
    payment: PaymentStatus
    hotdogs: List[HotdogGroup] = field(default_factory=list)
    other_lines: List[TicketLine] = field(default_factory=list)
    unknown_shortcuts: List[str] = field(default_factory=list)
    bbox: Optional[BBox] = None
    order_type: str = ""     # "Dine In" / "Drive Thru"
    total_text: str = ""     # raw text of the subtotal/paid bar
    pink_fraction: float = 0.0
    timestamp: float = 0.0
    frame_index: int = -1
    readable: bool = True

    @property
    def is_paid(self) -> bool:
        return self.payment is PaymentStatus.PAID

    @property
    def total_hotdogs(self) -> int:
        return sum(h.quantity for h in self.hotdogs if not h.is_unknown)

    def content_signature(self) -> str:
        """Stable key describing the ticket's order content.

        Used by the temporal aggregator to vote on content across frames.
        """
        parts = []
        for h in sorted(self.hotdogs, key=lambda g: (g.item, g.shortcut)):
            addons = ",".join(sorted(a.key + "x" + str(a.quantity) for a in h.addons))
            parts.append("{0}x{1}[{2}]".format(h.item, h.quantity, addons))
        return "|".join(parts)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class LifecycleState(str, Enum):
    """Full ticket lifecycle (section 6 / section 18)."""

    NEW = "NEW"
    PAID = "PAID"
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    IN_PROGRESS = "IN_PROGRESS"
    WAITING_FOR_COMPLETION = "WAITING_FOR_COMPLETION"
    FINAL_VALIDATION = "FINAL_VALIDATION"
    COMPLETED = "COMPLETED"
    WRONG = "WRONG"


# Legal transitions.  TicketManager refuses anything not listed here, so an
# order can never skip payment confirmation or be validated twice.
ALLOWED_TRANSITIONS: Dict[LifecycleState, Tuple[LifecycleState, ...]] = {
    LifecycleState.NEW: (LifecycleState.PAID,),
    LifecycleState.PAID: (LifecycleState.QUEUED,),
    LifecycleState.QUEUED: (LifecycleState.ACTIVE, LifecycleState.FINAL_VALIDATION),
    LifecycleState.ACTIVE: (
        LifecycleState.IN_PROGRESS,
        LifecycleState.WAITING_FOR_COMPLETION,
        LifecycleState.FINAL_VALIDATION,
    ),
    LifecycleState.IN_PROGRESS: (
        LifecycleState.WAITING_FOR_COMPLETION,
        LifecycleState.FINAL_VALIDATION,
    ),
    LifecycleState.WAITING_FOR_COMPLETION: (
        LifecycleState.IN_PROGRESS,
        LifecycleState.FINAL_VALIDATION,
    ),
    LifecycleState.FINAL_VALIDATION: (LifecycleState.COMPLETED, LifecycleState.WRONG),
    LifecycleState.COMPLETED: (),
    LifecycleState.WRONG: (),
}


class FailureCategory(str, Enum):
    MISSING_ITEM = "MISSING_ITEM"
    WRONG_QUANTITY = "WRONG_QUANTITY"
    UNEXPECTED_ITEM = "UNEXPECTED_ITEM"
    WRONG_ITEM_TYPE = "WRONG_ITEM_TYPE"
    INCOMPLETE_ORDER = "INCOMPLETE_ORDER"
    UNKNOWN_ITEM = "UNKNOWN_ITEM"


class EndReason(str, Enum):
    """Why final validation ran.

    An order is judged when its ticket leaves the KDS.  SHUTDOWN covers
    anything still open when the pipeline stops, so no order is left unjudged.
    """

    TICKET_DISAPPEARED = "TICKET_DISAPPEARED"
    SHUTDOWN = "SHUTDOWN"


@dataclass
class ValidationResult:
    correct: bool
    end_reason: EndReason
    expected: Dict[str, int] = field(default_factory=dict)
    detected: Dict[str, int] = field(default_factory=dict)
    missing: Dict[str, int] = field(default_factory=dict)
    extra: Dict[str, int] = field(default_factory=dict)
    categories: List[FailureCategory] = field(default_factory=list)
    message: str = ""
    # Type verdicts are inferred from ingredients, not detected directly, so
    # they always carry this flag (see fifo_queue.py).
    type_match_confident: bool = False
    type_notes: List[str] = field(default_factory=list)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "correct": self.correct,
            "end_reason": self.end_reason.value,
            "expected": dict(self.expected),
            "detected": dict(self.detected),
            "missing": dict(self.missing),
            "extra": dict(self.extra),
            "categories": [c.value for c in self.categories],
            "message": self.message,
            "type_match_confident": self.type_match_confident,
            "type_notes": list(self.type_notes),
            "timestamp": self.timestamp,
        }


@dataclass
class OrderGroup:
    """One paid KDS ticket, for its entire lifecycle (section 6)."""

    ticket_id: str
    internal_id: str
    created_at: float
    paid_at: float
    hotdogs: List[HotdogGroup] = field(default_factory=list)
    state: LifecycleState = LifecycleState.PAID
    order_type: str = ""

    # Live production evidence.
    detected_track_ids: List[int] = field(default_factory=list)
    detected_counts: Dict[str, int] = field(default_factory=dict)
    evidence_frozen: bool = False

    # Bookkeeping.
    last_seen_on_kds: float = 0.0
    disappeared_at: Optional[float] = None
    activated_at: Optional[float] = None
    finalized_at: Optional[float] = None
    result: Optional[ValidationResult] = None
    state_history: List[Tuple[float, LifecycleState]] = field(default_factory=list)

    # ---------------------------------------------------------------- helpers

    @property
    def expected_total(self) -> int:
        return sum(h.quantity for h in self.hotdogs if not h.is_unknown)

    @property
    def detected_total(self) -> int:
        return sum(self.detected_counts.values())

    @property
    def is_terminal(self) -> bool:
        return self.state in (LifecycleState.COMPLETED, LifecycleState.WRONG)

    def expected_counts(self) -> Dict[str, int]:
        """item -> expected quantity, ignoring unknown shortcuts."""
        out: Dict[str, int] = {}
        for h in self.hotdogs:
            if h.is_unknown:
                continue
            out[h.item] = out.get(h.item, 0) + h.quantity
        return out

    def unknown_items(self) -> List[str]:
        return [h.raw_text or h.shortcut for h in self.hotdogs if h.is_unknown]

    def all_addons(self) -> List[Tuple[str, AddOn]]:
        """(parent item, add-on) pairs, preserving the parent-child link."""
        return [(h.item, a) for h in self.hotdogs for a in h.addons]

    def to_ticket(self) -> Ticket:
        """Adapt to the existing ``src.schemas.Ticket``.

        Consumed by ``OrderStateMachine`` / ``BatchOrderValidator``.  Each
        expected hotdog becomes one ``hotdog<N>`` spec whose ingredients are
        the shortcut's configured ingredients plus its add-ons, so the existing
        recipe validator sees a shape it already understands.
        """
        hotdog_specs: Dict[str, Dict[str, int]] = {}
        line_items: List[LineItem] = []
        index = 0
        for group in self.hotdogs:
            if group.is_unknown:
                continue
            items: Dict[str, int] = {}
            for ing in group.ingredients:
                items[ing] = items.get(ing, 0) + 1
            for addon in group.addons:
                if addon.negation and addon.ingredient:
                    items.pop(addon.ingredient, None)
                elif addon.ingredient:
                    items[addon.ingredient] = items.get(addon.ingredient, 0) + 1
            for _ in range(max(1, group.quantity)):
                index += 1
                hotdog_specs["hotdog" + str(index)] = dict(items)
            line_items.append(
                LineItem(variant=group.item, count=group.quantity, items=dict(items))
            )
        shortcut = ", ".join(
            str(h.quantity) + " " + h.shortcut for h in self.hotdogs if not h.is_unknown
        )
        return Ticket(
            ticket_id=self.ticket_id,
            shortcut=shortcut,
            total_hotdogs=max(1, index),
            line_items=line_items,
            hotdog_specs=hotdog_specs,
            # Both fields describe the same hotdogs here: hotdog_specs is the
            # per-unit expansion, line_items the per-variant summary used for
            # display.  Flagged so the counter uses one, not both.
            specs_cover_line_items=True,
        )

    def to_dict(self) -> dict:
        return {
            "ticket_id": self.ticket_id,
            "internal_id": self.internal_id,
            "state": self.state.value,
            "order_type": self.order_type,
            "created_at": self.created_at,
            "paid_at": self.paid_at,
            "activated_at": self.activated_at,
            "finalized_at": self.finalized_at,
            "expected_total": self.expected_total,
            "detected_total": self.detected_total,
            "hotdogs": [h.to_dict() for h in self.hotdogs],
            "detected_counts": dict(self.detected_counts),
            "detected_track_ids": list(self.detected_track_ids),
            "unknown_items": self.unknown_items(),
            "disappeared_at": self.disappeared_at,
            "result": self.result.to_dict() if self.result else None,
        }


# ---------------------------------------------------------------------------
# Events emitted by the KDS monitor
# ---------------------------------------------------------------------------

class KdsEventType(str, Enum):
    TICKET_SEEN = "TICKET_SEEN"
    TICKET_UNPAID = "TICKET_UNPAID"
    TICKET_PAID = "TICKET_PAID"
    TICKET_UPDATED = "TICKET_UPDATED"
    TICKET_DISAPPEARED = "TICKET_DISAPPEARED"
    UNKNOWN_SHORTCUT = "UNKNOWN_SHORTCUT"


@dataclass
class KdsEvent:
    type: KdsEventType
    ticket_id: str
    timestamp: float = field(default_factory=time.monotonic)
    snapshot: Optional[TicketSnapshot] = None
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "ticket_id": self.ticket_id,
            "timestamp": self.timestamp,
            "detail": self.detail,
        }
