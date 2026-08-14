"""Tests for KDS client and order state machine."""

import json
import os
import time

from src.kds_client import DynamicKDSClient, MockKDSClient
from src.schemas import Action, OrderStatus, Ticket
from src.state_machine import OrderStateMachine


# ---------------------------------------------------------------------------
# MockKDSClient tests
# ---------------------------------------------------------------------------


class TestMockKDSClient:
    def test_sequential_emission(self, tmp_path):
        p = tmp_path / "kds.json"
        p.write_text(
            json.dumps(
                [
                    {"ticket_id": "T001", "expected_items": ["zone 1", "zone 2"]},
                    {"ticket_id": "T002", "expected_items": ["zone 3"]},
                ]
            )
        )
        client = MockKDSClient(str(p), poll_interval=0)
        t1 = client.get_next_ticket()
        assert t1 is not None
        assert t1.ticket_id == "T001"
        t2 = client.get_next_ticket()
        assert t2 is not None
        assert t2.ticket_id == "T002"
        t3 = client.get_next_ticket()
        assert t3 is None

    def test_flat_list_backward_compat(self, tmp_path):
        p = tmp_path / "kds.json"
        p.write_text(json.dumps([{"ticket_id": "T001", "expected_items": ["zone 1"]}]))
        client = MockKDSClient(str(p), poll_interval=0)
        t = client.get_next_ticket()
        assert t.ticket_id == "T001"
        assert t.expected_items == ["zone 1"]

    def test_poll_interval(self, tmp_path):
        p = tmp_path / "kds.json"
        p.write_text(json.dumps([{"ticket_id": "T001", "expected_items": ["x"]}]))
        client = MockKDSClient(str(p), poll_interval=10)
        t1 = client.get_next_ticket()
        assert t1 is not None
        t2 = client.get_next_ticket()
        assert t2 is None  # too soon

    def test_hot_reload(self, tmp_path):
        p = tmp_path / "kds.json"
        p.write_text(json.dumps([{"ticket_id": "T001", "expected_items": ["x"]}]))
        client = MockKDSClient(str(p), poll_interval=0)
        t1 = client.get_next_ticket()
        assert t1.ticket_id == "T001"
        # Modify file
        p.write_text(
            json.dumps(
                [
                    {"ticket_id": "T001", "expected_items": ["x"]},
                    {"ticket_id": "T002", "expected_items": ["y"]},
                ]
            )
        )
        # Force a distinct mtime so the hot-reload triggers deterministically.
        # On filesystems with coarse mtime granularity two rapid writes can share
        # an mtime, which would otherwise make this test flaky.
        bumped = p.stat().st_mtime + 10
        os.utime(p, (bumped, bumped))
        t2 = client.get_next_ticket()
        assert t2.ticket_id == "T001"  # reloaded, starts from index 0 again
        t3 = client.get_next_ticket()
        assert t3.ticket_id == "T002"

    def test_mark_completed_and_abandoned(self, tmp_path):
        p = tmp_path / "kds.json"
        p.write_text(json.dumps([]))
        client = MockKDSClient(str(p), poll_interval=0)
        client.mark_completed("T001")
        client.mark_abandoned("T002")
        assert "T001" in client._completed
        assert "T002" in client._abandoned

    def test_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.json"
        client = MockKDSClient(str(p), poll_interval=0)
        assert client.get_next_ticket() is None


# ---------------------------------------------------------------------------
# DynamicKDSClient tests
# ---------------------------------------------------------------------------


class TestDynamicKDSClient:
    def test_generates_tickets(self):
        client = DynamicKDSClient(
            zone_names=["zone 1", "zone 2", "zone 3"],
            min_items=1,
            max_items=3,
            interval_range=(0, 0),
            seed=42,
        )
        t = client.get_next_ticket()
        assert t is not None
        assert t.ticket_id.startswith("ORD-")
        assert 1 <= len(t.expected_items) <= 3
        assert all(item in ["zone 1", "zone 2", "zone 3"] for item in t.expected_items)

    def test_max_tickets_limit(self):
        client = DynamicKDSClient(
            zone_names=["a", "b"],
            min_items=1,
            max_items=1,
            interval_range=(0, 0),
            max_tickets=2,
            seed=1,
        )
        t1 = client.get_next_ticket()
        assert t1 is not None
        t2 = client.get_next_ticket()
        assert t2 is not None
        t3 = client.get_next_ticket()
        assert t3 is None  # max reached

    def test_seed_reproducibility(self):
        zones = ["a", "b", "c", "d", "e"]
        c1 = DynamicKDSClient(
            zone_names=zones,
            min_items=2,
            max_items=4,
            interval_range=(0, 0),
            seed=99,
        )
        c2 = DynamicKDSClient(
            zone_names=zones,
            min_items=2,
            max_items=4,
            interval_range=(0, 0),
            seed=99,
        )
        t1 = c1.get_next_ticket()
        t2 = c2.get_next_ticket()
        assert t1.ticket_id == t2.ticket_id
        assert t1.expected_items == t2.expected_items

    def test_interval_enforcement(self):
        client = DynamicKDSClient(
            zone_names=["x"],
            min_items=1,
            max_items=1,
            interval_range=(100, 100),
            seed=1,
        )
        t1 = client.get_next_ticket()
        assert t1 is not None
        t2 = client.get_next_ticket()
        assert t2 is None  # interval not elapsed


# ---------------------------------------------------------------------------
# OrderStateMachine tests
# ---------------------------------------------------------------------------


class TestOrderStateMachine:
    def _make_sm(self):
        return OrderStateMachine()

    def test_pick_logs_carried(self):
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1", "zone 2"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pick",
                timestamp=time.monotonic(),
            )
        )
        log = sm.get_validation_log()
        assert log[0]["status"] == "CARRIED"
        # Pick does NOT mark item as matched
        assert sm.get_current_order().remaining_counts["zone 1"] == 1

    def test_pickup_match(self):
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1", "zone 2"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        order = sm.get_current_order()
        assert order.remaining_counts["zone 1"] == 0
        assert order.picked_counts["zone 1"] == 1

    def test_pickup_extra(self):
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z2",
                zone_name="zone 2",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        order = sm.get_current_order()
        assert order.remaining_counts["zone 2"] == -1

    def test_order_completion_via_finalize(self):
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        # Pickup does NOT auto-finalize
        assert sm.get_current_order().status == OrderStatus.IN_PROGRESS
        assert sm.get_current_order().remaining_counts["zone 1"] == 0
        # Explicit finalize completes the order
        sm.finalize_current_order()
        assert sm.stats.total_orders == 1
        assert sm.stats.passed_orders == 1

    def test_new_ticket_rejected_while_in_progress(self):
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1", "zone 2"]))
        # Pickup one item, order not complete
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        assert sm.get_current_order().status == OrderStatus.IN_PROGRESS

        # New ticket while T1 is in progress — should be rejected
        sm.on_kds_ticket(Ticket(ticket_id="T2", expected_items=["zone 3"]))
        assert sm.stats.total_orders == 0
        assert sm.get_current_order().ticket_id == "T1"
        assert sm.get_current_order().remaining_counts["zone 1"] == 0

    def test_finalize_then_accept_new_ticket(self):
        """After finalizing, the state machine accepts a new ticket."""
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        sm.finalize_current_order()
        assert sm.current_ticket is None
        assert sm.get_current_order().ticket_id == ""

        # Now a new ticket should be accepted
        sm.on_kds_ticket(Ticket(ticket_id="T2", expected_items=["zone 2"]))
        assert sm.get_current_order().ticket_id == "T2"
        assert sm.get_current_order().remaining_counts == {"zone 2": 1}

    def test_no_action_without_ticket(self):
        sm = self._make_sm()
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pick",
                timestamp=time.monotonic(),
            )
        )
        assert sm.get_current_order().ticket_id == ""

    def test_hover_and_pickup_logged(self):
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1", "zone 2"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="hover",
                timestamp=time.monotonic(),
            )
        )
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        log = sm.get_validation_log()
        assert log[0]["status"] == "HOVER"
        assert log[1]["status"] == "MATCH"

    def test_order_status_lifecycle(self):
        sm = self._make_sm()
        assert sm.get_current_order().status == OrderStatus.PENDING
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        assert sm.get_current_order().status == OrderStatus.IN_PROGRESS

    def test_kds_client_feedback(self, tmp_path):
        p = tmp_path / "kds.json"
        p.write_text(json.dumps([{"ticket_id": "T1", "expected_items": ["zone 1"]}]))
        client = MockKDSClient(str(p), poll_interval=0)
        sm = OrderStateMachine()
        sm.set_kds_client(client)

        ticket = client.get_next_ticket()
        sm.on_kds_ticket(ticket)
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        # Pickup doesn't auto-finalize, so client isn't notified yet
        assert "T1" not in client._completed
        # Explicit finalize triggers KDS notification
        sm.finalize_current_order()
        assert "T1" in client._completed

    def test_order_persistence(self, tmp_path):
        history_file = tmp_path / "orders.jsonl"
        sm = OrderStateMachine(history_path=str(history_file))
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        # Pickup doesn't auto-finalize, so no history written yet
        assert not history_file.exists()
        sm.finalize_current_order()
        assert history_file.exists()
        lines = history_file.read_text().strip().split("\n")
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["ticket_id"] == "T1"
        assert record["status"] == "completed"
        assert record["passed"] is True

    def test_save_history(self, tmp_path):
        history_file = tmp_path / "orders.json"
        sm = OrderStateMachine(history_path=str(history_file))
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        sm.finalize_current_order()
        sm.save_history()
        assert history_file.exists()
        data = json.loads(history_file.read_text())
        assert len(data) == 1
        assert data[0]["ticket_id"] == "T1"

    def test_place_logs_placed_no_fulfillment(self):
        """place only logs PLACED, does not decrement remaining_counts."""
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="place",
                timestamp=time.monotonic(),
            )
        )
        log = sm.get_validation_log()
        assert log[0]["status"] == "PLACED"
        # remaining_counts unchanged — place does not fulfill
        assert sm.get_current_order().remaining_counts["zone 1"] == 1

    def test_pickup_then_place(self):
        """pickup confirms the item; place just logs PLACED; order stays in progress until finalized."""
        sm = self._make_sm()
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=time.monotonic(),
            )
        )
        # Pickup does NOT auto-finalize
        assert sm.get_current_order().status == OrderStatus.IN_PROGRESS
        assert sm.get_current_order().remaining_counts["zone 1"] == 0
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="place",
                timestamp=time.monotonic(),
            )
        )
        # Place does NOT finalize either
        assert sm.get_current_order().status == OrderStatus.IN_PROGRESS
        # Explicit finalize completes the order
        sm.finalize_current_order()
        assert sm.stats.total_orders == 1
        assert sm.stats.passed_orders == 1

    def test_advanced_validation(self):
        """Verify new calculate_validation logic works for missing, wrong, and extra items."""
        sm = self._make_sm()
        # Expect: 1 'zone 1'
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["zone 1"]))
        
        # Pick: 2 'zone 1' (1 expected, 1 extra) and 1 'zone 2' (not expected, wrong)
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=100.0,
            )
        )
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="zone 1",
                action_type="pickup",
                timestamp=101.0,
            )
        )
        sm.on_action(
            Action(
                track_id=1,
                zone_id="z2",
                zone_name="zone 2",
                action_type="pickup",
                timestamp=102.0,
            )
        )

        # Finalize the order to populate missing items
        order = sm.finalize_current_order()
        
        assert order.passed is False
        assert "zone 1" in order.extra_items
        assert "zone 2" in order.wrong_items
        # Since 'zone 1' was expected and picked (twice), it is NOT missing.
        assert len(order.missing_items) == 0
        
        # Check validation message
        assert order.validation_message == "Wrong + Extra Ingredients"
        
        # Check added_items_details
        details = order.added_items_details
        assert len(details) == 2
        
        # Details should have:
        # 1. 'zone 1' with status='extra', quantity=2
        # 2. 'zone 2' with status='wrong', quantity=1
        zone1_detail = next(d for d in details if d["name"] == "zone 1")
        zone2_detail = next(d for d in details if d["name"] == "zone 2")
        
        assert zone1_detail["status"] == "extra"
        assert zone1_detail["quantity"] == 2
        assert zone1_detail["timestamp"] == 101.0
        
        assert zone2_detail["status"] == "wrong"
        assert zone2_detail["quantity"] == 1
        assert zone2_detail["timestamp"] == 102.0

    def test_canonical_normalization(self):
        """Verify _normalize correctly maps aliases and checks display name retention."""
        sm = self._make_sm()
        # Expect: "pickel swears" (KDS typo)
        sm.on_kds_ticket(Ticket(ticket_id="T1", expected_items=["pickel swears"]))
        
        # Pick: "pickles (spears)" (actual zone label)
        matched_key = sm.on_action(
            Action(
                track_id=1,
                zone_id="z1",
                zone_name="pickles (spears)",
                action_type="pickup",
                timestamp=100.0,
            )
        )
        # Should match and return the expected KDS item key name
        assert matched_key == "pickel swears"
        
        # Verify original display name is preserved in expected items list
        order = sm.get_current_order()
        assert order.expected_items[0] == "pickel swears"
        
        # Verify validation matches it correctly on completion
        sm.finalize_current_order()
        assert order.passed is True
        assert len(order.missing_items) == 0
        assert len(order.wrong_items) == 0



