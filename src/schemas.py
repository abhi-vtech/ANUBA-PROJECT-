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


@dataclass
class Detection:
    track_id: int
    bbox: Tuple[int, int, int, int]
    class_name: str
    confidence: float


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
class Ticket:
    ticket_id: str
    expected_items: List[str]


@dataclass
class Order:
    ticket_id: str
    expected_items: List[str] = field(default_factory=list)
    remaining_counts: Dict[str, int] = field(default_factory=dict)
    picked_counts: Dict[str, int] = field(default_factory=dict)
    passed: bool = False
    missing_items: List[str] = field(default_factory=list)
    extra_items: List[str] = field(default_factory=list)
    status: OrderStatus = OrderStatus.PENDING
    applied_sauces: List[str] = field(default_factory=list)
    wrong_items: List[str] = field(default_factory=list)
    validation_message: str = ""
    added_items_details: List[dict] = field(default_factory=list)
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
