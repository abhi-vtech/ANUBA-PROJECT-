"""
Unit tests for granular vs. discrete ingredient counting.

Coverage:
  A. Multiple granular pinches within SERVING_GAP_S → counted once
  B. Granular pinch after gap > SERVING_GAP_S → counted again
  C. Discrete-item debounce unchanged at DISCRETE_DEBOUNCE_S (1.8s)
  D. Widened _hand_near_hotdog pad for granular ingredients
  E. Longer PendingPick expiry for granular ingredients
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock

from src.schemas import Action, OrderStatus
from src.state_machine import OrderStateMachine, SERVING_GAP_S, DISCRETE_DEBOUNCE_S
from src.ingredient_config import is_granular, GRANULAR_INGREDIENTS
from src.temporal import (
    _hand_near_hotdog,
    _boxes_overlap,
    _PICK_TTL_DISCRETE_S,
    _PICK_TTL_GRANULAR_S,
    _HOTDOG_PAD_DISCRETE_PX,
    _HOTDOG_PAD_GRANULAR_PX,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_action(ingredient: str, timestamp: float, action_type: str = "place") -> Action:
    return Action(
        track_id=1,
        zone_id="zone_test",
        zone_name=ingredient,
        action_type=action_type,
        timestamp=timestamp,
        duration_ms=100.0,
    )


def _sm_with_ticket(required_ingredients: list) -> OrderStateMachine:
    """Build an OrderStateMachine with a mocked ticket containing required_ingredients."""
    sm = OrderStateMachine()

    # Build a flat required_counts so the KDS filter passes
    mock_validator = MagicMock()
    mock_validator.required_counts = {ing: 1 for ing in required_ingredients}
    sm.batch_validator = mock_validator

    # Set up a minimal in-progress order
    from src.schemas import Order
    sm.current_order = Order(
        ticket_id="TEST-001",
        expected_items=required_ingredients,
        picked_counts={},
        status=OrderStatus.IN_PROGRESS,
    )
    sm.current_ticket = MagicMock()
    return sm


# ─── A. Multiple granular pinches within SERVING_GAP_S → 1 count ──────────────

class TestGranularServingGap:

    @pytest.mark.parametrize("ingredient", list(GRANULAR_INGREDIENTS))
    def test_multiple_pinches_within_gap_count_once(self, ingredient):
        sm = _sm_with_ticket([ingredient])
        t0 = 100.0

        # First pinch — should increment to 1
        sm.on_action(_make_action(ingredient, t0))
        assert sm.current_order.picked_counts.get(ingredient, 0) == 1, (
            f"First pinch of '{ingredient}' should count as 1"
        )

        # Second pinch 1 second later (within SERVING_GAP_S) — must NOT increment
        sm.on_action(_make_action(ingredient, t0 + 1.0))
        assert sm.current_order.picked_counts.get(ingredient, 0) == 1, (
            f"Second pinch of '{ingredient}' within {SERVING_GAP_S}s should not add a count"
        )

        # Third pinch 3 seconds later (still within SERVING_GAP_S) — must NOT increment
        sm.on_action(_make_action(ingredient, t0 + 3.0))
        assert sm.current_order.picked_counts.get(ingredient, 0) == 1, (
            f"Third pinch of '{ingredient}' within {SERVING_GAP_S}s should not add a count"
        )

    # ─── B. Granular pinch after gap > SERVING_GAP_S → counted again ──────────

    @pytest.mark.parametrize("ingredient", list(GRANULAR_INGREDIENTS))
    def test_pinch_after_serving_gap_increments_again(self, ingredient):
        sm = _sm_with_ticket([ingredient])
        t0 = 100.0

        sm.on_action(_make_action(ingredient, t0))
        assert sm.current_order.picked_counts.get(ingredient, 0) == 1

        # Second serving: after SERVING_GAP_S + small buffer
        t1 = t0 + SERVING_GAP_S + 0.5
        sm.on_action(_make_action(ingredient, t1))
        assert sm.current_order.picked_counts.get(ingredient, 0) == 2, (
            f"Pinch of '{ingredient}' after {SERVING_GAP_S}s gap should count as a new serving"
        )


# ─── C. Discrete debounce unchanged at DISCRETE_DEBOUNCE_S ───────────────────

class TestDiscreteDebounce:

    DISCRETE_ITEMS = [
        "chilli",
        "yellow_mustard_sauce",
        "pickle_spears",
        "sport_peppers",
        "tomato",
        "ketchup",
    ]

    @pytest.mark.parametrize("ingredient", DISCRETE_ITEMS)
    def test_rapid_duplicate_within_debounce_ignored(self, ingredient):
        sm = _sm_with_ticket([ingredient])
        t0 = 100.0

        sm.on_action(_make_action(ingredient, t0))
        count_after_first = sm.current_order.picked_counts.get(ingredient, 0)
        assert count_after_first == 1

        # Rapid duplicate within DISCRETE_DEBOUNCE_S — should be ignored
        sm.on_action(_make_action(ingredient, t0 + DISCRETE_DEBOUNCE_S * 0.5))
        assert sm.current_order.picked_counts.get(ingredient, 0) == 1, (
            f"Rapid duplicate of '{ingredient}' within {DISCRETE_DEBOUNCE_S}s should be ignored"
        )

    @pytest.mark.parametrize("ingredient", DISCRETE_ITEMS)
    def test_second_action_after_debounce_increments(self, ingredient):
        sm = _sm_with_ticket([ingredient])
        t0 = 100.0

        sm.on_action(_make_action(ingredient, t0))
        sm.on_action(_make_action(ingredient, t0 + DISCRETE_DEBOUNCE_S + 0.1))
        assert sm.current_order.picked_counts.get(ingredient, 0) == 2, (
            f"Second action of '{ingredient}' after debounce window should increment"
        )


# ─── D. Granular ingredients use wider _hand_near_hotdog pad ──────────────────

class TestHandNearHotdogPad:

    def _make_det(self, class_name: str, bbox) -> MagicMock:
        d = MagicMock()
        d.class_name = class_name
        d.bbox = bbox
        return d

    def test_granular_wider_pad_reaches_hotdog(self):
        # Hand bbox just outside discrete pad (65px gap) but within granular pad (110px)
        hand_bbox = (0, 0, 50, 50)        # hand: x1=0,y1=0,x2=50,y2=50
        hotdog_bbox = (115, 0, 200, 50)   # hotdog starts 65px from hand right edge

        detections = [self._make_det("hot-dog", hotdog_bbox)]

        # Discrete pad (60px): gap 65px > pad → should NOT be near
        assert not _hand_near_hotdog(hand_bbox, detections, ingredient="chilli"), \
            "Discrete ingredient should NOT reach hotdog at 65px gap"

        # Granular pad (110px): gap 65px < pad → should be near
        assert _hand_near_hotdog(hand_bbox, detections, ingredient="onions"), \
            "Granular ingredient should reach hotdog at 65px gap"

    def test_discrete_pad_constant_value(self):
        assert _HOTDOG_PAD_DISCRETE_PX == 60

    def test_granular_pad_constant_value(self):
        assert _HOTDOG_PAD_GRANULAR_PX == 110


# ─── E. Granular PendingPick expiry is longer ─────────────────────────────────

class TestPendingPickExpiry:

    def test_granular_ttl_longer_than_discrete(self):
        assert _PICK_TTL_GRANULAR_S > _PICK_TTL_DISCRETE_S, (
            f"Granular TTL ({_PICK_TTL_GRANULAR_S}s) must be > discrete TTL ({_PICK_TTL_DISCRETE_S}s)"
        )

    def test_constant_values(self):
        assert _PICK_TTL_DISCRETE_S == 3.0
        assert _PICK_TTL_GRANULAR_S == 6.0


# ─── F. is_granular() classification ─────────────────────────────────────────

class TestIsGranular:

    def test_granular_ingredients_classified_correctly(self):
        for ing in ["onions", "relish", "grated_yellow_cheese"]:
            assert is_granular(ing), f"'{ing}' should be classified as granular"

    def test_discrete_ingredients_not_granular(self):
        for ing in ["chilli", "yellow_mustard_sauce", "pickle_spears", "sport_peppers",
                    "tomato", "hot-dog", "ketchup", "swiss_cheese"]:
            assert not is_granular(ing), f"'{ing}' should NOT be classified as granular"
