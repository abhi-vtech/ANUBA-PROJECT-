import json
import tempfile
from pathlib import Path
import pytest

from src.cart_state_machine import (
    CartEvent,
    CartState,
    CartStateMachine,
    EventType,
    IngredientItem,
)


def test_initial_state_is_empty():
    cart = CartStateMachine()
    assert cart.state == CartState.EMPTY
    assert len(cart.ingredients) == 0
    assert cart.get_ingredient_names() == []


def test_state_transitions_empty_to_building_to_ready():
    cart = CartStateMachine(container_id="tray_01")
    
    # 1. Event: First ingredient added
    e1 = CartEvent(
        event_type=EventType.INGREDIENT_ADDED,
        ingredient_name="hotdog_bun",
        roi_id="roi_assembly",
    )
    cart.process_event(e1)
    assert cart.state == CartState.BUILDING
    assert cart.get_ingredient_names() == ["hotdog_bun"]

    # 2. Event: Second ingredient added
    e2 = CartEvent(
        event_type=EventType.INGREDIENT_ADDED,
        ingredient_name="sausage",
        roi_id="roi_assembly",
    )
    cart.process_event(e2)
    assert cart.state == CartState.BUILDING
    assert cart.get_ingredient_names() == ["hotdog_bun", "sausage"]

    # 3. Event: Prep complete
    e3 = CartEvent(event_type=EventType.PREP_COMPLETE)
    cart.process_event(e3)
    assert cart.state == CartState.READY


def test_array_based_ingredient_tracking_multiples():
    cart = CartStateMachine()
    # Add multiple of the same ingredient
    cart.add_ingredient("mustard")
    cart.add_ingredient("mustard")
    cart.add_ingredient("relish")
    cart.add_ingredient("mustard")

    names = cart.get_ingredient_names()
    assert names == ["mustard", "mustard", "relish", "mustard"]
    assert len(names) == 4

    counts = cart.get_ingredient_counts()
    assert counts == {"mustard": 3, "relish": 1}


def test_reset_mechanism_on_container_removal():
    cart = CartStateMachine(container_id="tray_01")
    cart.process_event(CartEvent(event_type=EventType.INGREDIENT_ADDED, ingredient_name="sausage"))
    cart.process_event(CartEvent(event_type=EventType.INGREDIENT_ADDED, ingredient_name="mustard"))
    assert cart.state == CartState.BUILDING
    assert len(cart.ingredients) == 2

    # Container removal event triggers RESET state -> wipes ingredients -> restores to EMPTY
    removal_event = CartEvent(
        event_type=EventType.CONTAINER_REMOVED,
        roi_id="roi_assembly",
    )
    cart.process_event(removal_event)
    
    assert cart.state == CartState.EMPTY
    assert len(cart.ingredients) == 0
    assert cart.container_id is None

    # Check transition history recorded RESET
    hist_to_states = [h["to"] for h in cart.transition_history]
    assert "RESET" in hist_to_states
    assert "EMPTY" in hist_to_states


def test_cart_persistence_across_multistep_prep():
    with tempfile.TemporaryDirectory() as tmp_dir:
        persistence_path = str(Path(tmp_dir) / "cart_state.json")
        
        # Step 1: Prep phase 1
        cart1 = CartStateMachine(container_id="tray_99", persistence_path=persistence_path)
        cart1.add_ingredient("bun")
        cart1.add_ingredient("sausage")
        assert cart1.state == CartState.BUILDING
        
        # Verify saved to disk
        assert Path(persistence_path).exists()

        # Step 2: Load state into new instance
        cart2 = CartStateMachine(persistence_path=persistence_path, load_from_disk=True)
        assert cart2.state == CartState.BUILDING
        assert cart2.container_id == "tray_99"
        assert cart2.get_ingredient_names() == ["bun", "sausage"]

        # Step 3: Add sauce & mark ready
        cart2.process_event(CartEvent(event_type=EventType.INGREDIENT_ADDED, ingredient_name="ketchup"))
        cart2.process_event(CartEvent(event_type=EventType.PREP_COMPLETE))
        assert cart2.state == CartState.READY

        # Reload again
        cart3 = CartStateMachine(persistence_path=persistence_path, load_from_disk=True)
        assert cart3.state == CartState.READY
        assert cart3.get_ingredient_names() == ["bun", "sausage", "ketchup"]


def test_auto_ready_threshold():
    cart = CartStateMachine(auto_ready_threshold=3)
    cart.process_event(CartEvent(event_type=EventType.INGREDIENT_ADDED, ingredient_name="bun"))
    assert cart.state == CartState.BUILDING
    cart.process_event(CartEvent(event_type=EventType.INGREDIENT_ADDED, ingredient_name="sausage"))
    assert cart.state == CartState.BUILDING
    cart.process_event(CartEvent(event_type=EventType.INGREDIENT_ADDED, ingredient_name="cheese"))
    assert cart.state == CartState.READY
