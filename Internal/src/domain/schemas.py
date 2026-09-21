from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple


class HandState(str, Enum):
    IDLE = "idle"
    IDLE_IN_ZONE = "idle_in_zone"
    PENDING_PICK = "pending_pick"
    TRANSIT_PENDING = "transit_pending"
    CARRYING = "carrying"
    CARRYING_IN_ASSEMBLY = "carrying_in_assembly"


class OrderStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"


HAND_CLASS: str = "hand"
SAUCE_CLASSES: Set[str] = {"ketchup_sauce", "yellow_mustard_sauce"}
DISH_CLASSES: Set[str] = {"hot-dog", "burger_bun", "french_fries"}


# ---------------------------------------------------------------------------
# Ingredient naming
#
# The same ingredient is spelled differently depending on where it comes from:
# the KDS shortcut config calls it "ketchup_sauce", the detector class is
# "ketchup_sauce", and the pick pipeline records it as "ketchup".  Anything that
# compares a REQUIRED ingredient against a DETECTED one has to put both through
# here first, or the comparison silently never matches -- which is exactly how
# the dashboard checklist came to show ketchup stuck at 0 while mustard ticked
# up, mustard being one of the names that happens to survive unchanged.
#
# Lives here, in the module both the order pipeline and the KDS side already
# import, so there is one vocabulary rather than one per caller.
# ---------------------------------------------------------------------------
def canonical_ingredient(name: str) -> str:
    """Helper to normalize names using canonical mappings (handles casing, underscores, spaces, aliases)."""
    if not name:
        return ""

    # Initial cleaning: lower case, strip, replace multiple spaces/underscores
    cleaned = name.lower().replace("_", " ").strip()

    # Direct canonical names lookup
    canonical_map = {
        "yellow mustard sauce": "yellow_mustard_sauce",
        "yellow_mustard_sauce": "yellow_mustard_sauce",
        "yellow mustard": "yellow_mustard_sauce",
        "yellow_mustard": "yellow_mustard_sauce",

        "pickles (rounds)": "pickle_rounds",
        "pickle rounds": "pickle_rounds",
        "pickles rounds": "pickle_rounds",
        "pickle round": "pickle_rounds",

        "pickles (spears)": "pickle_spears",
        "pickle spears": "pickle_spears",
        "pickles spears": "pickle_spears",
        "pickle spear": "pickle_spears",
        "pickel swears": "pickle_spears",
        "pickle swears": "pickle_spears",

        "diced onions": "diced_onions",
        "diced onion": "diced_onions",
        "onions": "onions",
        "onion": "onions",

        "yellow cheese": "yellow_cheese",
        "yellow cheese (sliced)": "yellow_cheese",
        "yellow cheese sliced": "yellow_cheese",

        "grated yellow cheese": "grated_yellow_cheese",
        "chilli grated yellow cheese": "grated_yellow_cheese",

        "tomato": "tomato",
        "tomatoes": "tomato",

        "swiss cheese": "swiss_cheese",
        "relish": "relish",
        "ketchup sauce": "ketchup",
        "ketchup": "ketchup",

        "sport (wax) peppers": "sport_peppers",
        "sport peppers": "sport_peppers",
        "sport wax peppers": "sport_peppers",
        "wax peppers": "sport_peppers",
    }

    # If it matches a key in the map, return the mapped value
    if cleaned in canonical_map:
        return canonical_map[cleaned]

    # Fallback to standard word cleaning
    if "(" in cleaned:
        cleaned = cleaned.split("(")[0].strip()
    words = cleaned.split()
    cleaned_words = []
    for w in words:
        if w.endswith("s") and w != "swiss":
            w = w[:-1]
        cleaned_words.append(w)
    return "_".join(cleaned_words)


@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float
    polygon: Optional[List[Tuple[float, float]]] = None


@dataclass
class Zone:
    id: str
    name: str
    zone_type: str
    polygon: List[Tuple[int, int]]
    color: str = "#3b82f6"


@dataclass
class FlowSignal:
    """Optical flow co-motion analysis result for a (hand, zone) pair."""

    mean_hand_flow: float = 0.0
    mean_zone_flow: float = 0.0
    direction_similarity: float = 0.0
    magnitude_ratio: float = 0.0
    is_contact: bool = False
    hand_flow_vector: Tuple[float, float] = (0.0, 0.0)
    zone_flow_vector: Tuple[float, float] = (0.0, 0.0)
    features_hand: int = 0
    features_zone: int = 0


@dataclass
class Action:
    track_id: int
    zone_id: str
    zone_name: str
    action_type: str  # "pickup", "pick", "place", or "hover"
    timestamp: float
    duration_ms: float = 0.0
    from_zone: Optional[str] = None
    resolved_hotdog_tid: Optional[int] = None  # YOLO track_id of the target hotdog (sauce attribution)


@dataclass
class LineItem:
    variant: str
    count: int
    items: Dict[str, int]


@dataclass
class Ticket:
    ticket_id: str
    shortcut: str = ""
    total_hotdogs: int = 1
    line_items: List[LineItem] = field(default_factory=list)
    hotdog_specs: Dict[str, Dict[str, int]] = field(default_factory=dict)
    expected_items: List[str] = field(default_factory=list)
    # True when hotdog_specs already expands every hotdog in line_items, so the
    # two describe the same order and counting both double-counts it.  KDS
    # tickets set this; mock tickets carry specs only and leave it False.
    specs_cover_line_items: bool = False



@dataclass
class Order:
    ticket_id: str
    shortcut: str = ""
    expected_items: List[str] = field(default_factory=list)
    remaining_counts: Dict[str, int] = field(default_factory=dict)
    picked_counts: Dict[str, int] = field(default_factory=dict)
    # What the ticket asks for, fixed at the requirement and never decremented.
    # remaining_counts cannot serve this purpose: picked_counts is incremented
    # from five different code paths and only one of them decrements remaining,
    # so "picked + remaining" grows as the order is made.
    required_counts: Dict[str, int] = field(default_factory=dict)
    hotdog_count: int = 1
    # How many hotdogs were actually SEEN being made, uncapped by the ticket.
    # picked_counts["hot-dog"] cannot serve this purpose: it is clamped to
    # required_counts so the checklist settles at 3/3 rather than flickering,
    # which also makes an EXTRA hotdog structurally invisible there. This is
    # the unclamped high-water mark, for telemetry and journeys -- the
    # dashboard deliberately does not read it.
    observed_hotdogs: int = 0
    passed: bool = False
    missing_items: List[str] = field(default_factory=list)
    extra_items: List[str] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    applied_sauces: List[str] = field(default_factory=list)
    wrong_items: List[str] = field(default_factory=list)
    validation_message: str = ""
    added_items_details: List[dict] = field(default_factory=list)
    dashboard_slots: List[dict] = field(default_factory=list)
    ending_soon: bool = False



@dataclass
class Stats:
    total_orders: int = 0
    passed_orders: int = 0
    failed_orders: int = 0

    @property
    def accuracy_pct(self) -> float:
        if self.total_orders == 0:
            return 0.0
        return round(self.passed_orders / self.total_orders * 100, 2)

    @property
    def error_rate_pct(self) -> float:
        if self.total_orders == 0:
            return 0.0
        return round(self.failed_orders / self.total_orders * 100, 2)
