"""Cheese pre-gate rules, exercised without a model, a GPU or a video."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.cheese_gate import CHEESE_KEYS, CheesePreGate, CheeseTake
from src.analysis.state_machine import OrderStateMachine
from src.domain.schemas import OrderStatus, Ticket
from src.domain.zones import ZoneManager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZONES = os.path.join(ROOT, "config", "zones.json")

W, H = 1920, 1080


class _Hand:
    """Minimal stand-in for a hand Detection."""

    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox


def hand_at(nx, ny, track_id=1):
    """A hand box whose 75%-down fingertip lands on normalised (nx, ny)."""
    cx, cy, height = nx * W, ny * H, 60
    top = cy - 0.75 * height
    return _Hand(track_id, (int(cx - 30), int(top), int(cx + 30), int(top + height)))


def centre(zone):
    xs = [p[0] for p in zone.polygon]
    ys = [p[1] for p in zone.polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)


#: Somewhere well clear of the cheese region.
OUTSIDE = (0.95, 0.90)


def cc_ticket(ticket_id="CHK424"):
    """An ORG C/C: chilli + yellow cheese (sliced), per config/kds_shortcuts.yaml."""
    return Ticket(
        ticket_id=ticket_id, shortcut="ORG C/C", total_hotdogs=1,
        hotdog_specs={"hotdog1": ["chilli", "yellow cheese (sliced)"]},
        specs_cover_line_items=True,
    )


def plain_ticket(ticket_id="CHK999"):
    return Ticket(
        ticket_id=ticket_id, shortcut="ORG", total_hotdogs=1,
        hotdog_specs={"hotdog1": ["ketchup"]}, specs_cover_line_items=True,
    )


def take(item, well, t=50.0):
    return CheeseTake(item=item, well=well, track_id=1, timestamp=t, armed_at=t - 0.2)


class TestCheesePreGate(unittest.TestCase):
    def setUp(self):
        self.zones = ZoneManager(ZONES)
        self.gate = CheesePreGate(self.zones, lookback_s=120.0)
        self.swiss = next(z for z in self.gate.wells if z.name == "swiss cheese")
        self.yellow = next(
            z for z in self.gate.wells if z.name == "yellow cheese (sliced)"
        )

    def run_path(self, path, gate=None, t0=100.0, step=0.05):
        gate = gate or self.gate
        takes = []
        for i, (nx, ny) in enumerate(path):
            takes += gate.update([hand_at(nx, ny)], W, H, t0 + i * step)
        return takes

    def test_every_cheese_well_is_watched(self):
        self.assertTrue(self.gate.enabled)
        self.assertEqual(
            {z.name for z in self.gate.wells},
            {"swiss cheese", "yellow cheese (sliced)", "grated yellow cheese"},
        )

    def test_slice_carried_out_of_the_region_is_confirmed(self):
        takes = self.run_path([centre(self.swiss)] * 4 + [OUTSIDE] * 4)
        self.assertEqual([t.item for t in takes], ["swiss_cheese"])
        self.assertEqual(takes[0].well, "swiss cheese")

    def test_the_well_is_reported_not_the_one_the_ticket_wanted(self):
        takes = self.run_path([centre(self.yellow)] * 4 + [OUTSIDE] * 4)
        self.assertEqual([t.item for t in takes], ["yellow_cheese"])

    def test_moving_between_two_cheese_wells_takes_nothing(self):
        # The bin-exit shortcut in temporal.py fires a place event on this move;
        # leaving the region rather than the well is what makes it a non-event.
        takes = self.run_path(
            [centre(self.swiss)] * 4 + [centre(self.yellow)] * 8
        )
        self.assertEqual(takes, [])

    def test_a_touch_too_brief_to_grab_takes_nothing(self):
        takes = self.run_path([centre(self.swiss)] + [OUTSIDE] * 4)
        self.assertEqual(takes, [])

    def test_jitter_across_the_region_edge_takes_nothing(self):
        takes = self.run_path(
            [centre(self.swiss)] * 4 + [OUTSIDE] + [centre(self.swiss)] * 4
        )
        self.assertEqual(takes, [])

    def test_a_hand_that_never_leaves_the_region_expires(self):
        takes = self.run_path([centre(self.swiss)] * 40, step=1.0)
        self.assertEqual(takes, [])

    def test_takes_are_buffered_then_drained_once(self):
        self.run_path([centre(self.swiss)] * 4 + [OUTSIDE] * 4)
        self.assertEqual([t.item for t in self.gate.drain(110.0)], ["swiss_cheese"])
        self.assertEqual(self.gate.drain(110.0), [])

    def test_the_buffer_forgets_past_the_lookback_window(self):
        self.run_path([centre(self.swiss)] * 4 + [OUTSIDE] * 4)
        self.assertEqual(self.gate.drain(100.0 + 500.0), [])

    def test_a_disabled_gate_emits_nothing(self):
        gate = CheesePreGate(self.zones, enabled=False)
        self.assertFalse(gate.enabled)
        self.assertEqual(
            self.run_path([centre(self.swiss)] * 4 + [OUTSIDE] * 4, gate=gate), []
        )


class TestCheeseReachesTheOrder(unittest.TestCase):
    def machine(self, ticket):
        sm = OrderStateMachine()
        sm.cheese_gate_owns_cheese = True
        sm.on_kds_ticket(ticket)
        return sm

    def test_a_cc_ticket_requires_cheese(self):
        sm = self.machine(cc_ticket())
        self.assertTrue(sm.requires_cheese())
        self.assertIn("yellow_cheese", sm.batch_validator.required_counts)

    def test_cheese_fetched_before_confirmation_is_replayed_onto_the_ticket(self):
        sm = self.machine(cc_ticket("CHK001"))
        applied = sm.apply_cheese_takes(
            [take("yellow_cheese", "yellow cheese (sliced)")], pre_confirmation=True
        )
        self.assertEqual(applied, 1)
        self.assertEqual(sm.current_order.picked_counts.get("yellow_cheese"), 1)
        self.assertNotIn("yellow_cheese", sm.current_order.missing_items)

    def test_the_wrong_cheese_fails_the_order(self):
        sm = self.machine(cc_ticket("CHK002"))
        sm.apply_cheese_takes([take("swiss_cheese", "swiss cheese")], pre_confirmation=True)
        order = sm.current_order
        self.assertFalse(order.passed)
        self.assertIn("swiss_cheese", order.wrong_items)
        # The requirement is still outstanding: swiss does not satisfy yellow.
        self.assertIn("yellow_cheese", order.missing_items)

    def test_the_wrong_cheese_taken_live_is_not_dropped(self):
        # on_action's required_counts gate would have discarded this, leaving a
        # wrong-cheese order indistinguishable from one with no cheese at all.
        sm = self.machine(cc_ticket("CHK003"))
        self.assertEqual(sm.apply_cheese_takes([take("swiss_cheese", "swiss cheese")]), 1)
        self.assertIn("swiss_cheese", sm.current_order.wrong_items)

    def test_buffered_cheese_is_discarded_when_the_ticket_wants_none(self):
        # Nothing ties a slice taken with an empty board to this ticket rather
        # than the one before it, so it must not fail a correct order.
        sm = self.machine(plain_ticket("CHK004"))
        applied = sm.apply_cheese_takes(
            [take("swiss_cheese", "swiss cheese")], pre_confirmation=True
        )
        self.assertEqual(applied, 0)
        self.assertEqual(sm.current_order.picked_counts, {})
        self.assertTrue(sm.current_order.wrong_items == [])

    def test_cheese_taken_live_on_a_no_cheese_ticket_still_counts_as_wrong(self):
        sm = self.machine(plain_ticket("CHK005"))
        self.assertEqual(sm.apply_cheese_takes([take("swiss_cheese", "swiss cheese")]), 1)
        self.assertIn("swiss_cheese", sm.current_order.wrong_items)

    def test_nothing_is_applied_with_no_order_in_progress(self):
        sm = OrderStateMachine()
        sm.cheese_gate_owns_cheese = True
        self.assertEqual(sm.apply_cheese_takes([take("swiss_cheese", "swiss cheese")]), 0)

    def test_one_slice_is_not_counted_twice(self):
        sm = self.machine(cc_ticket("CHK007"))
        applied = sm.apply_cheese_takes(
            [
                take("yellow_cheese", "yellow cheese (sliced)", 50.0),
                take("yellow_cheese", "yellow cheese (sliced)", 50.5),
            ],
            pre_confirmation=True,
        )
        self.assertEqual(applied, 1)
        self.assertEqual(sm.current_order.picked_counts.get("yellow_cheese"), 1)

    def test_every_cheese_well_is_a_known_key(self):
        zones = ZoneManager(ZONES)
        from src.core.naming import normalize_item_name

        wells = {
            normalize_item_name(z.name)
            for z in zones.get_all()
            if z.zone_type == "bin" and "cheese" in z.name.lower()
        }
        self.assertEqual(wells, set(CHEESE_KEYS))


if __name__ == "__main__":
    unittest.main()


class TestRequirementIsFrozenAtCreation(unittest.TestCase):
    """A ticket's requirement is set once, at creation, and never rewritten.

    CHK 248 is the shape that motivated this: two separate `1 ORG C/C` lines,
    the second carrying a `2 Onion` add-on, plus one `1 ORG CHL`.  Three
    hotdogs, expressed as three lines rather than a quantity of two.
    """

    def chk248_lines(self):
        from src.kds.schemas import AddOn, HotdogGroup

        def cc(addons=None):
            return HotdogGroup(
                shortcut="ORG C/C", item="ORIGINAL_CHILI_CHEESE_DOG", quantity=1,
                ingredients=["chilli", "yellow cheese (sliced)"], addons=addons or [],
            )

        onion = AddOn(key="ONION", display="Onion", ingredient="onions",
                      quantity=2, negation=False)
        return [
            cc(),
            cc([onion]),
            HotdogGroup(shortcut="ORG CHL", item="ORIGINAL_CHILI_DOG", quantity=1,
                        ingredients=["chilli", "onions"]),
        ]

    def group(self):
        from src.kds.fifo_queue import TicketManager
        from src.kds.schemas import PaymentStatus, TicketSnapshot

        manager = TicketManager()
        snapshot = TicketSnapshot(
            ticket_id="CHK 248", timestamp=0.0, frame_index=0,
            payment=PaymentStatus.PAID, order_type="Dine In",
            hotdogs=self.chk248_lines(),
        )
        return manager, manager.create_from_paid_ticket(snapshot, now=0.0)

    def test_duplicate_item_lines_are_kept_separate(self):
        _, group = self.group()
        self.assertIsNotNone(group)
        self.assertEqual(len(group.hotdogs), 3)
        self.assertEqual(group.expected_total, 3)

    def test_the_addon_stays_on_its_own_hotdog(self):
        _, group = self.group()
        with_addon = [h for h in group.hotdogs if h.addons]
        self.assertEqual(len(with_addon), 1)
        self.assertEqual([a.display for a in with_addon[0].addons], ["Onion"])
        self.assertEqual(with_addon[0].addons[0].quantity, 2)

    def test_the_merge_helper_is_gone(self):
        import src.kds.fifo_queue as fifo

        for name in ("_merge_hotdogs", "_content_signature"):
            self.assertFalse(hasattr(fifo, name), name + " should have been removed")

    def test_nothing_can_rewrite_the_requirement(self):
        from src.kds.fifo_queue import TicketManager

        manager, group = self.group()
        self.assertFalse(hasattr(manager, "update_content"))
        # A later reading cannot reach the group by any public route, so the
        # requirement it entered with is the one it is judged against.
        self.assertEqual(group.expected_total, 3)

    def test_the_downstream_update_path_is_gone(self):
        from src.analysis.state_machine import OrderStateMachine
        from src.kds.kds_video_client import KDSVideoClient

        self.assertFalse(hasattr(OrderStateMachine, "update_kds_ticket"))
        self.assertFalse(hasattr(KDSVideoClient, "get_ticket_updates"))
