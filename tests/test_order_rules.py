"""Validation rules, exercised without a model, a GPU or a video."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.contract import Frame, TrackedObject
from src.core.engine import AnalysisEngine, EngineConfig, Zone
from src.core.order_rules import ExtraKind, ExtrasPolicy, Failure, OrderValidator
from src.core.ticket_spec import TicketSpec

TICKET = {
    "ticket_id": "kds_3",
    "shortcut": "2 Org Chicgo",
    "total_hotdogs": 2,
    "hotdog1": ["yellow mustard sauce", "relish", "onions"],
    "hotdog2": ["yellow mustard sauce", "relish", "onions"],
}


def spec():
    return TicketSpec.from_json(TICKET)


def validator(policy=None):
    return OrderValidator(spec(), policy or ExtrasPolicy(advisory=False))


class TestTicketSpec(unittest.TestCase):
    def test_groups_are_preserved(self):
        s = spec()
        self.assertEqual(len(s.groups), 2)
        self.assertEqual(s.required_hotdogs, 2)
        self.assertEqual({g.group_id for g in s.groups}, {"hotdog1", "hotdog2"})

    def test_flattening_multiplies_across_groups(self):
        # Each of two hotdogs needs one relish -> two relish in total.
        self.assertEqual(spec().required_counts()["relish"], 2)

    def test_line_item_quantity_multiplies(self):
        s = TicketSpec.from_json({
            "ticket_id": "t1", "total_hotdogs": 3,
            "line_items": [{"variant": "chicago", "count": 3, "items": {"relish": 2}}],
        })
        self.assertEqual(s.required_hotdogs, 3)
        self.assertEqual(s.required_counts()["relish"], 6)

    def test_negation_is_parsed(self):
        s = TicketSpec.from_json({
            "ticket_id": "t2", "total_hotdogs": 1,
            "hotdog1": ["yellow mustard sauce", "NO ONIONS"],
        })
        self.assertEqual(s.forbidden_counts(), {"onions": 1})
        self.assertNotIn("onions", s.required_counts())

    def test_unknown_items_are_flagged_not_dropped_silently(self):
        s = TicketSpec.from_json(
            {"ticket_id": "t3", "total_hotdogs": 1, "hotdog1": ["relish", "caviar"]},
            known_items={"relish", "onions"},
        )
        self.assertEqual(s.unverifiable, ["caviar"])
        self.assertEqual(list(s.required_counts()), ["relish"])

    def test_undetailed_hotdogs_still_required(self):
        s = TicketSpec.from_json(
            {"ticket_id": "t4", "total_hotdogs": 3, "hotdog1": ["relish"]})
        self.assertEqual(s.required_hotdogs, 3)


class TestCheck1Hotdogs(unittest.TestCase):
    def test_missing_hotdog_fails(self):
        v = validator()
        v.observe_hotdog(1)
        for item in ("yellow_mustard_sauce", "relish", "onions"):
            for _ in range(2):
                v.observe_place(item)
        out = v.validate(final=True)
        self.assertFalse(out.correct)
        self.assertFalse(out.checks["hotdogs"])
        self.assertIn(Failure.MISSING_HOTDOG, out.failures)

    def test_both_hotdogs_present_passes_check_1(self):
        v = validator()
        v.observe_hotdog(1)
        v.observe_hotdog(2)
        self.assertTrue(v.validate().checks["hotdogs"])

    def test_track_ids_are_deduplicated(self):
        v = validator()
        for _ in range(50):
            v.observe_hotdog(7)
        self.assertEqual(v.validate().observed_hotdogs, 1)


class TestCheck2RequiredItems(unittest.TestCase):
    def test_missing_item_fails_with_shortfall(self):
        v = validator()
        v.observe_hotdog(1); v.observe_hotdog(2)
        for _ in range(2):
            v.observe_place("yellow_mustard_sauce")
            v.observe_place("relish")
        out = v.validate(final=True)
        self.assertFalse(out.correct)
        self.assertEqual(out.missing, {"onions": 2})
        self.assertIn(Failure.MISSING_ITEM, out.failures)

    def test_complete_order_passes(self):
        v = validator()
        v.observe_hotdog(1); v.observe_hotdog(2)
        for item in ("yellow_mustard_sauce", "relish", "onions"):
            for _ in range(2):
                v.observe_place(item)
        out = v.validate(final=True)
        self.assertTrue(out.correct, out.message)
        self.assertEqual(out.failures, [])

    def test_sauce_shortfall_is_forgiven_at_final_only(self):
        v = validator()
        v.observe_hotdog(1); v.observe_hotdog(2)
        v.observe_place("yellow_mustard_sauce")  # 1 of 2
        for item in ("relish", "onions"):
            for _ in range(2):
                v.observe_place(item)
        self.assertIn("yellow_mustard_sauce", v.validate(final=False).missing)
        self.assertTrue(v.validate(final=True).correct)


class TestCheck3Extras(unittest.TestCase):
    """The capability the current pipeline cannot provide."""

    def _complete(self, v):
        v.observe_hotdog(1); v.observe_hotdog(2)
        for item in ("yellow_mustard_sauce", "relish", "onions"):
            for _ in range(2):
                v.observe_place(item)

    def test_unexpected_well_is_reported(self):
        v = validator()
        self._complete(v)
        v.observe_place("chilli")
        out = v.validate(final=True)
        kinds = {e.item: e.kind for e in out.extras}
        self.assertEqual(kinds.get("chilli"), ExtraKind.UNEXPECTED)

    def test_unexpected_well_fails_when_enforced(self):
        v = validator(ExtrasPolicy(advisory=False))
        self._complete(v)
        v.observe_place("chilli")
        out = v.validate(final=True)
        self.assertFalse(out.correct)
        self.assertIn(Failure.UNEXPECTED_ITEM, out.failures)

    def test_unexpected_well_only_warns_when_advisory(self):
        v = validator(ExtrasPolicy(advisory=True))
        self._complete(v)
        v.observe_place("chilli")
        out = v.validate(final=True)
        self.assertTrue(out.correct)
        self.assertTrue(out.extras)
        self.assertIn("not on ticket", out.message)

    def test_forbidden_item_fails_even_when_advisory(self):
        s = TicketSpec.from_json({
            "ticket_id": "t5", "total_hotdogs": 1,
            "hotdog1": ["yellow mustard sauce", "NO ONIONS"],
        })
        v = OrderValidator(s, ExtrasPolicy(advisory=True))
        v.observe_hotdog(1)
        v.observe_place("yellow_mustard_sauce")
        v.observe_place("onions")
        out = v.validate(final=True)
        self.assertFalse(out.correct)
        self.assertIn(Failure.FORBIDDEN_ITEM, out.failures)

    def test_over_count_respects_tolerance(self):
        v = validator(ExtrasPolicy(advisory=False, over_tolerance=1))
        self._complete(v)
        v.observe_place("relish")  # 3 of 2, within tolerance
        self.assertTrue(v.validate(final=True).correct)
        v.observe_place("relish")  # 4 of 2, past tolerance
        out = v.validate(final=True)
        self.assertFalse(out.correct)
        self.assertIn(Failure.WRONG_QUANTITY, out.failures)

    def test_ignored_wells_never_count_as_extra(self):
        v = validator(ExtrasPolicy(advisory=False, ignore=frozenset({"napkin"})))
        self._complete(v)
        v.observe_place("napkin")
        self.assertTrue(v.validate(final=True).correct)


class TestEngineAcrossRuntimes(unittest.TestCase):
    """Same engine, same frames -> same verdict, whatever produced them."""

    ZONES = [
        Zone("z1", "relish", "bin", [(0.0, 0.0), (0.2, 0.0), (0.2, 0.2), (0.0, 0.2)]),
        Zone("z2", "onions", "bin", [(0.3, 0.0), (0.5, 0.0), (0.5, 0.2), (0.3, 0.2)]),
        Zone("z3", "chilli", "bin", [(0.6, 0.0), (0.8, 0.0), (0.8, 0.2), (0.6, 0.2)]),
    ]

    def _frames(self, wells, fps):
        """Hand dwells in each well long enough to count, at a given frame rate."""
        frames, index = [], 0
        centers = {"relish": (128, 72), "onions": (512, 72), "chilli": (896, 72)}
        for well in wells:
            cx, cy = centers[well]
            for _ in range(int(fps * 2)):  # 2 seconds in the well
                frames.append(Frame(
                    index=index, t=index / fps, width=1280, height=720,
                    objects=[
                        TrackedObject(1, "hand", (cx - 10, cy - 10, cx + 10, cy + 10)),
                        TrackedObject(9, "hot-dog", (600, 400, 700, 450)),
                    ],
                ))
                index += 1
            for _ in range(int(fps * 8)):  # 8 seconds away, clears every gap
                frames.append(Frame(
                    index=index, t=index / fps, width=1280, height=720,
                    objects=[TrackedObject(1, "hand", (20, 600, 40, 620)),
                             TrackedObject(9, "hot-dog", (600, 400, 700, 450))],
                ))
                index += 1
        return frames

    def _engine(self):
        s = TicketSpec.from_json({
            "ticket_id": "eng", "total_hotdogs": 1, "hotdog1": ["relish", "onions"]})
        return AnalysisEngine(self.ZONES, spec=s, config=EngineConfig(dwell_s=0.5),
                              policy=ExtrasPolicy(advisory=False))

    def test_correct_order(self):
        v = self._engine().run(self._frames(["relish", "onions"], 30)).finish()
        self.assertTrue(v.correct, v.message)

    def test_extra_well_is_caught_by_the_engine(self):
        v = self._engine().run(self._frames(["relish", "onions", "chilli"], 30)).finish()
        self.assertFalse(v.correct)
        self.assertEqual([e.item for e in v.extras], ["chilli"])

    def test_missing_well_is_caught(self):
        v = self._engine().run(self._frames(["relish"], 30)).finish()
        self.assertFalse(v.correct)
        self.assertEqual(v.missing, {"onions": 1})

    def test_frame_rate_does_not_change_the_verdict(self):
        """A 12 fps runtime and a 45 fps runtime must agree on the same footage."""
        slow = self._engine().run(self._frames(["relish", "onions", "chilli"], 12)).finish()
        fast = self._engine().run(self._frames(["relish", "onions", "chilli"], 45)).finish()
        self.assertEqual(slow.to_dict()["extras"], fast.to_dict()["extras"])
        self.assertEqual(slow.to_dict()["missing"], fast.to_dict()["missing"])
        self.assertEqual(slow.correct, fast.correct)

    def test_replay_source_reproduces_the_live_verdict(self):
        from src.core.runner import _frame_to_json
        from src.core.sources.replay_source import ReplaySource

        frames = self._frames(["relish", "onions", "chilli"], 30)
        live = self._engine().run(frames).finish()

        path = os.path.join(tempfile.mkdtemp(), "frames.jsonl")
        with open(path, "w") as fh:
            for f in frames:
                fh.write(json.dumps(_frame_to_json(f)) + "\n")
        replayed = self._engine().run(ReplaySource(path)).finish()

        self.assertEqual(live.to_dict(), replayed.to_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
