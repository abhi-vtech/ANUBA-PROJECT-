from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CartState(str, Enum):
    """Defined cart state machine states."""
    EMPTY = "EMPTY"
    BUILDING = "BUILDING"
    READY = "READY"
    RESET = "RESET"


class EventType(str, Enum):
    """Event types emitted by ROI and motion filter outputs."""
    CONTAINER_PLACED = "CONTAINER_PLACED"
    INGREDIENT_ADDED = "INGREDIENT_ADDED"
    MOTION_STOPPED = "MOTION_STOPPED"
    PREP_COMPLETE = "PREP_COMPLETE"
    CONTAINER_REMOVED = "CONTAINER_REMOVED"
    FORCE_RESET = "FORCE_RESET"
    HOTDOG_DETECTED = "HOTDOG_DETECTED"
    ENTERED_ASSEMBLY = "ENTERED_ASSEMBLY"


@dataclass
class IngredientItem:
    """Represents a single ingredient addition event, allowing duplicate items."""
    name: str
    added_at: float = field(default_factory=time.time)
    confidence: float = 1.0
    roi_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CartEvent:
    """Event payload consumed from ROI + motion filter outputs."""
    event_type: EventType
    ingredient_name: Optional[str] = None
    roi_id: Optional[str] = None
    container_id: Optional[str] = None
    confidence: float = 1.0
    motion_score: float = 0.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CartStateMachine:
    """
    Event-driven Cart State Machine.
    
    Automated State Flow:
        EMPTY    -> Initial clean state.
        READY    -> Triggered when hotdog is detected anywhere in camera feed.
        BUILDING -> Triggered when hotdog enters assembly ROI / ingredients added.
        RESET    -> Triggered when exit logic is confirmed (hand crosses exit line).
    """

    def __init__(
        self,
        container_id: Optional[str] = None,
        persistence_path: Optional[str] = None,
        auto_ready_threshold: Optional[int] = None,
        load_from_disk: bool = False,
    ):
        self.state: CartState = CartState.EMPTY
        self.container_id: Optional[str] = container_id
        self.ingredients: List[IngredientItem] = []
        self.persistence_path: Optional[str] = persistence_path
        self.auto_ready_threshold: Optional[int] = auto_ready_threshold
        
        self.created_at: float = time.time()
        self.updated_at: float = time.time()
        self.transition_history: List[Dict[str, Any]] = []

        if load_from_disk and persistence_path and Path(persistence_path).exists():
            self.load_from_disk()

    # ── State Transition Helpers ───────────────────────────────────────────

    def _set_state(self, new_state: CartState, reason: str = "") -> None:
        old_state = self.state
        if old_state == new_state:
            return

        self.state = new_state
        self.updated_at = time.time()
        entry = {
            "from": old_state.value,
            "to": new_state.value,
            "timestamp": self.updated_at,
            "reason": reason,
        }
        self.transition_history.append(entry)
        logger.info(f"[CartStateMachine] Transition: {old_state.value} -> {new_state.value} ({reason})")

        if self.persistence_path:
            self.persist_to_disk()

    # ── Event Consumption Architecture ─────────────────────────────────────

    def process_event(self, event: CartEvent) -> CartState:
        """
        Consumes event payloads from ROI + Motion filter outputs and updates state machine.
        """
        if event.container_id and not self.container_id:
            self.container_id = event.container_id

        # 1. Container removal / Exit logic confirmation (highest priority reset mechanism)
        if event.event_type in (EventType.CONTAINER_REMOVED, EventType.FORCE_RESET):
            self._handle_container_removed(reason=f"Exit logic confirmed ({event.roi_id or 'Exit_Line'})")
            return self.state

        # 2. Automated feed / assembly ROI state transitions
        if event.event_type == EventType.HOTDOG_DETECTED:
            if self.state == CartState.EMPTY:
                self._set_state(CartState.READY, reason="Hotdog detected in video feed")
            return self.state

        if event.event_type == EventType.ENTERED_ASSEMBLY:
            if self.state in (CartState.EMPTY, CartState.READY):
                self._set_state(CartState.BUILDING, reason="Hotdog entered assembly ROI")
            return self.state

        # 3. State-specific ingredient handling
        if self.state in (CartState.EMPTY, CartState.READY):
            if event.event_type == EventType.CONTAINER_PLACED:
                self._set_state(CartState.EMPTY, reason=f"Container {event.container_id} placed in ROI")
            elif event.event_type == EventType.INGREDIENT_ADDED and event.ingredient_name:
                self.add_ingredient(
                    name=event.ingredient_name,
                    confidence=event.confidence,
                    roi_id=event.roi_id,
                    metadata=event.metadata,
                )
                self._set_state(CartState.BUILDING, reason=f"Ingredient added: {event.ingredient_name}")

        elif self.state == CartState.BUILDING:
            if event.event_type == EventType.INGREDIENT_ADDED and event.ingredient_name:
                self.add_ingredient(
                    name=event.ingredient_name,
                    confidence=event.confidence,
                    roi_id=event.roi_id,
                    metadata=event.metadata,
                )
                if self.auto_ready_threshold and len(self.ingredients) >= self.auto_ready_threshold:
                    self._set_state(CartState.READY, reason=f"Reached auto-ready item count threshold ({self.auto_ready_threshold})")
            elif event.event_type == EventType.PREP_COMPLETE:
                self._set_state(CartState.READY, reason="Preparation complete event received")

        elif self.state == CartState.READY:
            if event.event_type == EventType.INGREDIENT_ADDED and event.ingredient_name:
                # Additional ingredients added after ready state return to building
                self.add_ingredient(
                    name=event.ingredient_name,
                    confidence=event.confidence,
                    roi_id=event.roi_id,
                    metadata=event.metadata,
                )
                self._set_state(CartState.BUILDING, reason=f"Additional ingredient added to ready cart: {event.ingredient_name}")
            elif event.event_type == EventType.PREP_COMPLETE:
                pass  # Already ready

        elif self.state == CartState.RESET:
            # Transition out of RESET back to EMPTY when clean
            self._set_state(CartState.EMPTY, reason="Cart reset sequence finalized")
            if event.event_type == EventType.INGREDIENT_ADDED and event.ingredient_name:
                self.add_ingredient(
                    name=event.ingredient_name,
                    confidence=event.confidence,
                    roi_id=event.roi_id,
                    metadata=event.metadata,
                )
                self._set_state(CartState.BUILDING, reason=f"Ingredient added after reset: {event.ingredient_name}")

        return self.state

    # ── Array-Based Ingredient Tracking ────────────────────────────────────

    def add_ingredient(
        self,
        name: str,
        confidence: float = 1.0,
        roi_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IngredientItem:
        """Appends ingredient to the array, supporting duplicates/multiples of the same item."""
        item = IngredientItem(
            name=name,
            added_at=time.time(),
            confidence=confidence,
            roi_id=roi_id,
            metadata=metadata or {},
        )
        self.ingredients.append(item)
        self.updated_at = time.time()
        logger.info(f"[CartStateMachine] Added ingredient #{len(self.ingredients)}: '{name}' (total={len(self.ingredients)})")
        
        if self.state == CartState.EMPTY:
            self._set_state(CartState.BUILDING, reason=f"First ingredient added: {name}")

        if self.persistence_path:
            self.persist_to_disk()
            
        return item

    def get_ingredient_names(self) -> List[str]:
        """Returns ordered list of all added ingredient names (preserving duplicates)."""
        return [item.name for item in self.ingredients]

    def get_ingredient_counts(self) -> Dict[str, int]:
        """Returns summary counts of ingredients."""
        counts: Dict[str, int] = {}
        for item in self.ingredients:
            counts[item.name] = counts.get(item.name, 0) + 1
        return counts

    # ── Reset Mechanism (Tray Removal) ─────────────────────────────────────

    def _handle_container_removed(self, reason: str = "Tray/container removal detected") -> None:
        """Executes reset mechanism tied to tray/container removal."""
        self._set_state(CartState.RESET, reason=reason)
        # Wipe ingredient cart contents
        self.ingredients.clear()
        self.container_id = None
        # Auto-complete reset cycle to return to EMPTY
        self._set_state(CartState.EMPTY, reason="Cart state cleared and restored to EMPTY after removal")

    def reset(self, reason: str = "Manual reset") -> None:
        """Public reset handle."""
        self._handle_container_removed(reason=reason)

    def mark_ready(self, reason: str = "Manual mark ready") -> None:
        """Explicitly transition cart to READY state."""
        if self.state in (CartState.BUILDING, CartState.EMPTY):
            self._set_state(CartState.READY, reason=reason)

    # ── Cart Persistence Across Multi-step Prep ─────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Serializes cart state machine to dictionary format."""
        return {
            "state": self.state.value,
            "container_id": self.container_id,
            "ingredients": [asdict(item) for item in self.ingredients],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "transition_history": self.transition_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], persistence_path: Optional[str] = None) -> CartStateMachine:
        """Deserializes cart state machine from dictionary format."""
        cart = cls(
            container_id=data.get("container_id"),
            persistence_path=persistence_path,
            load_from_disk=False,
        )
        cart.state = CartState(data.get("state", CartState.EMPTY.value))
        cart.created_at = data.get("created_at", time.time())
        cart.updated_at = data.get("updated_at", time.time())
        cart.transition_history = data.get("transition_history", [])

        ingredients_raw = data.get("ingredients", [])
        cart.ingredients = [
            IngredientItem(
                name=item["name"],
                added_at=item.get("added_at", time.time()),
                confidence=item.get("confidence", 1.0),
                roi_id=item.get("roi_id"),
                metadata=item.get("metadata", {}),
            )
            for item in ingredients_raw
        ]
        return cart

    def persist_to_disk(self) -> None:
        """Saves current state machine to file for multi-step persistence."""
        if not self.persistence_path:
            return
        p = Path(self.persistence_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_from_disk(self) -> None:
        """Restores state machine state from file."""
        if not self.persistence_path:
            return
        p = Path(self.persistence_path)
        if not p.exists():
            return
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        loaded = CartStateMachine.from_dict(data, persistence_path=self.persistence_path)
        self.state = loaded.state
        self.container_id = loaded.container_id
        self.ingredients = loaded.ingredients
        self.created_at = loaded.created_at
        self.updated_at = loaded.updated_at
        self.transition_history = loaded.transition_history
