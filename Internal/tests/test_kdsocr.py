"""Tests for the kds-ocr integration.

They drive the real mapping/emission/journey code with hand-written kds-ocr
records, so no video, OCR model or child process is needed.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.state_machine import OrderStateMachine
from src.domain.schemas import OrderStatus
from src.kdsocr.emissions import EmissionTailer, RecipeEmission
from src.kdsocr.journey import HOTDOG, JourneyLog, PLACE, REQUIREMENT
from src.kdsocr.mapping import IngredientMapper, as_count


def record(ref="CHK-1", reason="appeared", status="ok", hot_dogs=None, **kw):
    rec = {
        "order_ref": ref,
        "emit_reason": reason,
        "status": status,
        "channel": "drive_thru",
        "total_dogs": sum(h.get("qty", 1) for h in (hot_dogs or [])),
        "hot_dogs": hot_dogs or [],
        "totals": [],
    }
    rec.update(kw)
    return rec


def dog(code="ORG C/C", qty=1, ingredients=None, modifiers=None):
    return {"code": code, "qty": qty, "name": code,
            "ingredients": ingredients or [], "modifiers": modifiers or []}


class TestCounts(unittest.TestCase):
    """Requirement: every quantity we act on is a count, never a weight."""

    def test_weight_units_collapse_to_one(self):
        for unit in ("oz", "OZ", "pinch", "g", "ml"):
            self.assertEqual(as_count(4, unit), 1, unit)

    def test_countable_units_keep_their_number(self):
        self.assertEqual(as_count(2, "each"), 2)
        self.assertEqual(as_count(3, "slice"), 3)
        self.assertEqual(as_count(2, "leaf"), 2)

    def test_never_zero_or_negative(self):
        self.assertEqual(as_count(0, "each"), 1)
        self.assertEqual(as_count(-5, "each"), 1)

    def test_junk_is_one_not_a_crash(self):
        self.assertEqual(as_count("lots", None), 1)
        self.assertEqual(as_count(None, None), 1)
        self.assertEqual(as_count(float("inf"), "each"), 1)

    def test_fractions_round(self):
        self.assertEqual(as_count(1.4, "each"), 1)
        self.assertEqual(as_count(1.6, "each"), 2)


class TestMapping(unittest.TestCase):
    def setUp(self):
        self.m = IngredientMapper()

    def test_operations_manual_names_reach_our_zone_names(self):
        self.assertEqual(self.m.resolve("chili").name, "chilli")
        self.assertEqual(self.m.resolve("american cheese").name, "yellow_cheese")
        self.assertEqual(self.m.resolve("shredded cheddar").name, "grated_yellow_cheese")
        self.assertEqual(self.m.resolve("pickle spear").name, "pickle_spears")

    def test_mustard_reaches_the_detector_class(self):
        self.assertEqual(self.m.resolve("mustard").name, "yellow_mustard_sauce")

    def test_known_unobservable_is_unverifiable_not_ok(self):
        r = self.m.resolve("bacon")
        self.assertFalse(r.checkable)
        self.assertEqual(r.status, "unverifiable")
        self.assertTrue(r.reportable)

    def test_base_components_are_dropped_silently(self):
        # The bun and the sausage are in every hot dog: they are covered by
        # the hotdog COUNT, not by an ingredient check. They must not become a
        # requirement AND must not be reported as "not checkable" either.
        for name in ("bun", "hot dog", "all beef dog", "polish dog", "veggie dog"):
            r = self.m.resolve(name)
            self.assertFalse(r.checkable, name)
            self.assertFalse(r.reportable, name)
            self.assertEqual(r.status, "base", name)

    def test_unheard_of_ingredient_is_unknown_never_guessed(self):
        r = self.m.resolve("wasabi aioli")
        self.assertEqual(r.status, "unknown")
        self.assertIsNone(r.name)

    def test_resolve_all_separates_checkable_base_and_reportable(self):
        counts, skipped = self.m.resolve_all([
            {"ingredient": "chili", "qty": 1},
            {"ingredient": "bun", "qty": 1},        # base: silent
            {"ingredient": "all beef dog", "qty": 1},   # base: silent
            {"ingredient": "bacon", "qty": 1},      # unverifiable: reported
            {"ingredient": "mustard", "qty": 1},
        ])
        self.assertEqual(counts, {"chilli": 1, "yellow_mustard_sauce": 1})
        self.assertEqual([s.source for s in skipped], ["bacon"])


class TestEmission(unittest.TestCase):
    def setUp(self):
        self.m = IngredientMapper()

    def test_line_quantity_is_divided_back_out(self):
        # kds-ocr has already multiplied by the line qty; LineItem.items is
        # per-hotdog and gets multiplied again, so it must be divided here.
        em = RecipeEmission.from_record(record(hot_dogs=[
            dog("PLSH KRAUT", 2, [{"ingredient": "sauerkraut", "qty": 2},
                                  {"ingredient": "mustard", "qty": 2}])
        ]), self.m)
        g = em.groups[0]
        self.assertTrue(g["grouped"])
        self.assertEqual(g["qty"], 2)
        self.assertEqual(g["counts"], {"sauerkraut": 1, "yellow_mustard_sauce": 1})

    def test_requirement_survives_the_round_trip(self):
        em = RecipeEmission.from_record(record(hot_dogs=[
            dog("PLSH KRAUT", 2, [{"ingredient": "sauerkraut", "qty": 2}])
        ]), self.m)
        ticket = em.to_ticket()
        from src.analysis.batch_validator import BatchOrderValidator
        req = BatchOrderValidator(ticket).required_counts
        self.assertEqual(req["sauerkraut"], 2)   # 1 per dog x 2 dogs
        self.assertEqual(req["hot-dog"], 2)

    def test_indivisible_line_keeps_the_total_rather_than_rounding(self):
        em = RecipeEmission.from_record(record(hot_dogs=[
            dog("ODD", 2, [{"ingredient": "chili", "qty": 3}])
        ]), self.m)
        g = em.groups[0]
        self.assertFalse(g["grouped"])
        self.assertEqual(g["qty"], 1)
        self.assertEqual(g["counts"], {"chilli": 3})

    def test_voided_is_not_buildable(self):
        em = RecipeEmission.from_record(
            record(status="voided", hot_dogs=[dog()]), self.m)
        self.assertFalse(em.buildable)

    def test_blocked_is_not_buildable(self):
        em = RecipeEmission.from_record(
            record(status="blocked", hot_dogs=[dog()]), self.m)
        self.assertFalse(em.buildable)

    def test_signature_ignores_noise_but_catches_item_change(self):
        a = RecipeEmission.from_record(record(hot_dogs=[dog("ORG C/C", 1)],
                                              amount=5.0), self.m)
        b = RecipeEmission.from_record(record(hot_dogs=[dog("ORG C/C", 1)],
                                              amount=9.0, elapsed_s=300), self.m)
        self.assertEqual(a.signature, b.signature)
        c = RecipeEmission.from_record(record(hot_dogs=[dog("ORG C/C", 2)]), self.m)
        self.assertNotEqual(a.signature, c.signature)

    def test_unusable_records_are_skipped_not_fatal(self):
        self.assertIsNone(RecipeEmission.from_record({}, self.m))
        self.assertIsNone(RecipeEmission.from_record({"order_ref": ""}, self.m))
        self.assertIsNone(RecipeEmission.from_record("nonsense", self.m))

    def test_not_checkable_is_carried_not_dropped(self):
        em = RecipeEmission.from_record(record(hot_dogs=[
            dog("ORG C/C", 1, [{"ingredient": "chili", "qty": 1},
                               {"ingredient": "bacon", "qty": 1}])
        ]), self.m)
        self.assertIn("bacon", em.not_checkable)

    def test_the_checklist_is_toppings_plus_the_hotdog_count(self):
        """The bun and the dog never reach the checklist; the COUNT does."""
        from src.analysis.batch_validator import BatchOrderValidator
        em = RecipeEmission.from_record(record(total_dogs=3, hot_dogs=[
            dog("AB C/C", 1, [{"ingredient": "shredded cheddar", "qty": 1},
                              {"ingredient": "chili", "qty": 1},
                              {"ingredient": "all beef dog", "qty": 1},
                              {"ingredient": "bun", "qty": 1}]),
            dog("PLSH KRAUT", 2, [{"ingredient": "mustard", "qty": 2},
                                  {"ingredient": "sauerkraut", "qty": 2},
                                  {"ingredient": "polish dog", "qty": 2},
                                  {"ingredient": "bun", "qty": 2}]),
        ]), self.m)
        self.assertEqual(em.not_checkable, [])
        req = BatchOrderValidator(em.to_ticket()).required_counts
        self.assertEqual(req, {
            "grated_yellow_cheese": 1,
            "chilli": 1,
            "yellow_mustard_sauce": 2,
            "sauerkraut": 2,
            "hot-dog": 3,          # the required hotdogs
        })
        for absent in ("bun", "all_beef_dog", "polish_dog", "polish_hot_dog"):
            self.assertNotIn(absent, req)

    def test_a_dog_with_no_toppings_still_requires_the_hotdogs(self):
        """A plain dog has only base components -- the count is all we check."""
        from src.analysis.batch_validator import BatchOrderValidator
        em = RecipeEmission.from_record(record(total_dogs=2, hot_dogs=[
            dog("ORG PLAN", 2, [{"ingredient": "hot dog", "qty": 2},
                                {"ingredient": "bun", "qty": 2}]),
        ]), self.m)
        req = BatchOrderValidator(em.to_ticket()).required_counts
        self.assertEqual(req, {"hot-dog": 2})


class TestTailer(unittest.TestCase):
    def test_partial_line_is_held_until_its_newline(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "recipes.jsonl")
            open(path, "w").close()
            t = EmissionTailer(path)
            with open(path, "a") as fh:
                fh.write(json.dumps({"order_ref": "A"}) + "\n")
                fh.write('{"order_ref": "B"')          # mid-write
                fh.flush()
                self.assertEqual([r["order_ref"] for r in t.poll()], ["A"])
                fh.write("}\n")
                fh.flush()
                self.assertEqual([r["order_ref"] for r in t.poll()], ["B"])
            self.assertEqual(t.bad_lines, 0)
            t.close()

    def test_bad_line_is_counted_and_the_rest_still_read(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.jsonl")
            with open(path, "w") as fh:
                fh.write("{not json\n" + json.dumps({"order_ref": "C"}) + "\n")
            t = EmissionTailer(path)
            self.assertEqual([r["order_ref"] for r in t.poll()], ["C"])
            self.assertEqual(t.bad_lines, 1)
            t.close()

    def test_missing_file_is_not_an_error(self):
        t = EmissionTailer("/nonexistent/nope.jsonl")
        self.assertEqual(list(t.poll()), [])


class TestJourney(unittest.TestCase):
    def test_steps_are_ordered_and_a_verdict_closes_the_ticket(self):
        log = JourneyLog(path=None)
        j = log.start("CHK-9", "drive_thru", t=0.0)
        j.add(REQUIREMENT, 0.0, counts={"chilli": 1}, reason="appeared")
        j.add(PLACE, 5.0, item="chilli")
        log.finish("CHK-9", correct=True, message="ok", t=10.0)
        self.assertNotIn("CHK-9", log.open)
        self.assertEqual(log.recent[0].correct, True)
        self.assertEqual([s.kind for s in log.recent[0].steps],
                         [REQUIREMENT, PLACE, "verdict"])

    def test_requirement_now_returns_the_latest(self):
        log = JourneyLog(path=None)
        j = log.start("CHK-1", t=0.0)
        j.add(REQUIREMENT, 0.0, counts={"chilli": 1})
        j.add(REQUIREMENT, 5.0, counts={"chilli": 2, "onions": 1})
        self.assertEqual(j.requirement_now(), {"chilli": 2, "onions": 1})

    def test_observed_counts_repeated_places(self):
        log = JourneyLog(path=None)
        j = log.start("CHK-2", t=0.0)
        for t in (1.0, 2.0, 3.0):
            j.add(PLACE, t, item="onions")
        self.assertEqual(j.observed_now(), {"onions": 3})


    def test_duration_uses_one_clock_not_two(self):
        # The pipeline stamps steps with the video's MEDIA time. Seeding
        # opened_at from a wall clock and comparing it against media time
        # clamped every duration to zero.
        t = {"v": 0.0}
        log = JourneyLog(path=None, clock=lambda: t["v"])
        t["v"] = 1.0
        j = log.start("CHK-50")
        t["v"] = 2.2
        j.add(PLACE, 2.2, item="chilli")
        log.finish("CHK-50", correct=False, message="x", t=81.0)
        r = log.recent[0]
        self.assertEqual(r.opened_at, 1.0)
        self.assertEqual(r.verdict_at, 81.0)
        self.assertEqual(r.duration_s, 80.0)

    def test_open_journey_duration_tracks_the_latest_step(self):
        t = {"v": 10.0}
        log = JourneyLog(path=None, clock=lambda: t["v"])
        j = log.start("CHK-51")
        j.add(PLACE, 40.0, item="chilli")
        self.assertEqual(j.duration_s, 30.0)

    def test_a_broken_clock_does_not_end_the_run(self):
        def boom():
            raise RuntimeError("no clock")
        log = JourneyLog(path=None, clock=boom)
        j = log.start("CHK-52")          # must not raise
        self.assertIsNotNone(j)


    def test_repeated_wrapping_events_collapse_to_distinct_tracks(self):
        # One hotdog fragments into several track ids and fires a wrapping
        # "done" for each. Counting events showed a 3-dog ticket as 6/3 and
        # drove the dashboard progress bar past 100%.
        log = JourneyLog(path=None)
        j = log.start("CHK-60")
        for tid in (1, 2, 1, 1, 8, 2, 2):
            j.add(HOTDOG, 1.0, track_id=tid)
        self.assertEqual(sum(1 for s in j.steps if s.kind == HOTDOG), 7)
        self.assertEqual(j.wrapped_tracks(), 3)

    def test_hotdog_steps_without_a_track_id_are_not_counted(self):
        log = JourneyLog(path=None)
        j = log.start("CHK-61")
        j.add(HOTDOG, 1.0)                 # no track_id
        j.add(HOTDOG, 2.0, track_id=5)
        self.assertEqual(j.wrapped_tracks(), 1)

    def test_wrong_verdicts_are_written_to_disk(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "j.jsonl")
            log = JourneyLog(path=path)
            log.start("CHK-3", t=0.0)
            log.finish("CHK-3", correct=False, message="missing chilli",
                       missing={"chilli": 1}, t=1.0)
            rows = [json.loads(l) for l in open(path)]
            self.assertEqual(rows[0]["ticket_id"], "CHK-3")
            self.assertFalse(rows[0]["correct"])
            self.assertEqual(rows[0]["missing"], {"chilli": 1})


class TestLiveRequirementUpdate(unittest.TestCase):
    """Requirement: a ticket edited on the KDS updates the order in place."""

    def setUp(self):
        self.m = IngredientMapper()
        self.sm = OrderStateMachine()

    def _open(self, hot_dogs):
        em = RecipeEmission.from_record(record(hot_dogs=hot_dogs), self.m)
        self.sm.on_kds_ticket(em.to_ticket())
        return em

    def test_update_applies_the_new_requirement(self):
        self._open([dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])])
        self.assertEqual(self.sm.current_order.required_counts.get("chilli"), 1)
        em2 = RecipeEmission.from_record(record(reason="updated", hot_dogs=[
            dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1},
                               {"ingredient": "onion", "qty": 1}])
        ]), self.m)
        self.assertTrue(self.sm.update_ticket_requirements(em2.to_ticket()))
        self.assertEqual(self.sm.current_order.required_counts.get("onions"), 1)

    def test_update_keeps_what_was_already_observed(self):
        self._open([dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])])
        self.sm.batch_validator.on_place_event("chilli")
        em2 = RecipeEmission.from_record(record(reason="updated", hot_dogs=[
            dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1},
                               {"ingredient": "onion", "qty": 1}])
        ]), self.m)
        self.sm.update_ticket_requirements(em2.to_ticket())
        # The chilli really went on; an edit to the ticket does not undo it.
        self.assertEqual(self.sm.batch_validator.observed_counts.get("chilli"), 1)

    def test_remaining_reflects_progress_after_an_update(self):
        self._open([dog("ORG CHL", 2, [{"ingredient": "chili", "qty": 2}])])
        self.sm.current_order.picked_counts["chilli"] = 1
        em2 = RecipeEmission.from_record(record(reason="updated", hot_dogs=[
            dog("ORG CHL", 2, [{"ingredient": "chili", "qty": 2}])
        ]), self.m)
        self.sm.update_ticket_requirements(em2.to_ticket())
        self.assertEqual(self.sm.current_order.required_counts["chilli"], 2)
        self.assertEqual(self.sm.current_order.remaining_counts["chilli"], 1)

    def test_update_for_a_different_ticket_is_refused(self):
        self._open([dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])])
        other = RecipeEmission.from_record(
            record(ref="CHK-99", hot_dogs=[dog("ORG MUST", 1)]), self.m)
        self.assertFalse(self.sm.update_ticket_requirements(other.to_ticket()))

    def test_update_refused_once_the_order_is_finished(self):
        em = self._open([dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])])
        self.sm.finalize_current_order()
        self.assertFalse(self.sm.update_ticket_requirements(em.to_ticket()))



class TestClient(unittest.TestCase):
    """The FIFO source, live updates and the bump verdict trigger."""

    def setUp(self):
        from src.kdsocr.reader import ReaderConfig
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "recipes.jsonl")
        open(self.path, "w").close()
        self.cfg = ReaderConfig(videos=["dummy.mkv"], recipes_path=self.path,
                                out_dir=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _client(self):
        from src.kdsocr.client import KdsOcrClient
        return KdsOcrClient(self.cfg, journey_path=None, start=False)

    def _emit(self, rec):
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def test_tickets_are_issued_oldest_first(self):
        c = self._client()
        self._emit(record(ref="CHK-1", hot_dogs=[dog("ORG CHL", 1)]))
        self._emit(record(ref="CHK-2", hot_dogs=[dog("ORG MUST", 1)]))
        c.poll()
        self.assertEqual(c.get_next_ticket().ticket_id, "CHK-1")
        self.assertEqual(c.get_next_ticket().ticket_id, "CHK-2")
        self.assertIsNone(c.get_next_ticket())

    def test_voided_ticket_is_never_issued(self):
        c = self._client()
        self._emit(record(ref="CHK-5", status="voided", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        self.assertIsNone(c.get_next_ticket())
        self.assertIn("CHK-5", c.dashboard_state()["skipped"])

    def test_update_is_queued_only_after_the_ticket_was_issued(self):
        c = self._client()
        self._emit(record(ref="CHK-7", hot_dogs=[
            dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])]))
        c.poll()
        # Changed before we ever issued it: the issued ticket already carries
        # the new requirement, so there is nothing to update.
        self._emit(record(ref="CHK-7", reason="updated", hot_dogs=[
            dog("ORG CHL", 2, [{"ingredient": "chili", "qty": 2}])]))
        c.poll()
        self.assertEqual(c.take_updates(), [])
        t = c.get_next_ticket()
        self.assertEqual(t.total_hotdogs, 2)
        # Changed after issuing: now it must be reported.
        self._emit(record(ref="CHK-7", reason="updated", hot_dogs=[
            dog("ORG CHL", 3, [{"ingredient": "chili", "qty": 3}])]))
        c.poll()
        self.assertEqual([u.ticket_id for u in c.take_updates()], ["CHK-7"])

    def test_bump_is_the_verdict_trigger(self):
        c = self._client()
        self._emit(record(ref="CHK-8", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        c.get_next_ticket()
        self.assertEqual(c.take_bumps(), [])
        self._emit(record(ref="CHK-8", reason="bumped", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        self.assertEqual(c.take_bumps(), ["CHK-8"])
        self.assertEqual(c.take_bumps(), [])      # drained

    def test_bump_before_issue_closes_without_a_verdict(self):
        c = self._client()
        self._emit(record(ref="CHK-9", hot_dogs=[dog("ORG CHL", 1)]))
        self._emit(record(ref="CHK-9", reason="bumped", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        # We have no evidence for a ticket the pipeline never opened, so it is
        # not judged -- but it is not silently forgotten either.
        self.assertIsNone(c.get_next_ticket())
        self.assertIn("CHK-9", c.dashboard_state()["skipped"])

    def test_late_emission_cannot_reopen_a_judged_ticket(self):
        c = self._client()
        self._emit(record(ref="CHK-10", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        c.get_next_ticket()
        c.record_verdict("CHK-10", correct=True, message="ok")
        self._emit(record(ref="CHK-10", reason="updated", hot_dogs=[dog("ORG CHL", 5)]))
        c.poll()
        self.assertEqual(c.take_updates(), [])
        self.assertIsNone(c.get_next_ticket())

    def test_journey_records_requirement_then_place_then_verdict(self):
        c = self._client()
        self._emit(record(ref="CHK-11", hot_dogs=[
            dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])]))
        c.poll()
        c.get_next_ticket()
        c.record_place("CHK-11", "chilli", t=3.0)
        c.record_verdict("CHK-11", correct=False, message="wrong",
                         missing={"onions": 1})
        j = c.journeys.recent[0]
        self.assertEqual([s.kind for s in j.steps],
                         [REQUIREMENT, PLACE, "verdict"])
        self.assertFalse(j.correct)
        self.assertEqual(j.missing, {"onions": 1})

    def test_dashboard_state_has_what_the_panel_renders(self):
        c = self._client()
        self._emit(record(ref="CHK-12", hot_dogs=[
            dog("ORG CHL", 2, [{"ingredient": "chili", "qty": 2}])]))
        c.poll()
        st = c.dashboard_state()
        self.assertIn("counts", st)
        self.assertIn("queue", st)
        card = st["queue"][0]
        self.assertEqual(card["ticket_id"], "CHK-12")
        self.assertEqual(card["queue_label"], "QUEUED")
        self.assertEqual(card["expected_total"], 2)
        c.get_next_ticket()
        self.assertEqual(c.dashboard_state()["queue"][0]["queue_label"], "ACTIVE")


    def test_card_progress_is_capped_at_what_the_ticket_asked_for(self):
        c = self._client()
        self._emit(record(ref="CHK-62", total_dogs=2,
                          hot_dogs=[dog("ORG CHL", 2)]))
        c.poll()
        c.get_next_ticket()
        for tid in (1, 2, 3, 4, 5):        # more tracks than dogs ordered
            c.record_hotdog("CHK-62", track_id=tid, t=1.0)
        card = c.dashboard_state()["queue"][0]
        self.assertEqual(card["expected_total"], 2)
        self.assertEqual(card["detected_total"], 2)

    def test_has_screen_content_follows_the_reader_not_a_guess(self):
        c = self._client()
        self.assertFalse(c.has_screen_content)
        self._emit(record(ref="CHK-13", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        self.assertTrue(c.has_screen_content)
        c.get_next_ticket()
        c.record_verdict("CHK-13", correct=True)
        self.assertFalse(c.has_screen_content)




class TestNotOurProblem(unittest.TestCase):
    """A ticket we cannot verify must never be scored as a wrong order."""

    def setUp(self):
        from src.kdsocr.reader import ReaderConfig
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "recipes.jsonl")
        open(self.path, "w").close()
        self.cfg = ReaderConfig(videos=["d.mkv"], recipes_path=self.path,
                                out_dir=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _client(self):
        from src.kdsocr.client import KdsOcrClient
        return KdsOcrClient(self.cfg, journey_path=None, start=False)

    def _emit(self, rec):
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def test_no_hotdog_ticket_is_not_a_problem_and_not_wrong(self):
        c = self._client()
        # A fries-and-a-drink ticket: read, understood, nothing to verify.
        self._emit(record(ref="CHK-20", hot_dogs=[],
                          ticket_items=[{"code": "SML FRY"}, {"code": "SM COKE"}]))
        self._emit(record(ref="CHK-20", reason="bumped", hot_dogs=[],
                          ticket_items=[{"code": "SML FRY"}]))
        c.poll()
        st = c.dashboard_state()
        self.assertIn("CHK-20", st["no_hotdogs"])
        self.assertNotIn("CHK-20", st["skipped"])      # not a problem
        self.assertEqual(st["wrong_total"], 0)         # and NOT a wrong order
        self.assertEqual(st["judged_total"], 0)        # nothing was judged
        # and it raises no warning: there was nothing wrong with this ticket
        self.assertFalse([w for w in st["warnings"] if "CHK-20" in w])

    def test_voided_ticket_is_a_warning_but_still_not_wrong(self):
        c = self._client()
        self._emit(record(ref="CHK-21", status="voided", hot_dogs=[dog("ORG CHL", 1)]))
        self._emit(record(ref="CHK-21", reason="bumped", status="voided",
                          hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        st = c.dashboard_state()
        self.assertIn("CHK-21", st["skipped"])
        self.assertEqual(st["wrong_total"], 0)
        self.assertEqual(st["unverified_total"], 1)
        self.assertTrue(any("CHK-21" in w for w in st["warnings"]))

    def test_unjudged_close_leaves_correct_as_none_not_false(self):
        c = self._client()
        self._emit(record(ref="CHK-22", status="voided", hot_dogs=[dog("ORG CHL", 1)]))
        self._emit(record(ref="CHK-22", reason="bumped", status="voided",
                          hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        j = c.journeys.recent[0]
        self.assertIsNone(j.correct)                   # not False

    def test_a_real_wrong_verdict_still_counts(self):
        c = self._client()
        self._emit(record(ref="CHK-23", hot_dogs=[
            dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])]))
        c.poll()
        c.get_next_ticket()
        c.record_verdict("CHK-23", correct=False, message="missing chilli",
                         missing={"chilli": 1})
        st = c.dashboard_state()
        self.assertEqual(st["wrong_total"], 1)
        self.assertEqual(st["judged_total"], 1)
        self.assertEqual(st["unverified_total"], 0)

    def test_run_end_closes_open_tickets_without_inventing_a_verdict(self):
        c = self._client()
        self._emit(record(ref="CHK-24", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        c.get_next_ticket()
        c.stop()                                       # run ends mid-ticket
        self.assertIsNone(c.journeys.recent[0].correct)
        self.assertEqual(c.dashboard_state()["wrong_total"], 0)




class TestPacing(unittest.TestCase):
    """A recorded KDS feed must not run ahead of the production video."""

    VID = "Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_13_12_to_13_p0007_PDT.mkv"

    def setUp(self):
        from src.kdsocr.reader import ReaderConfig
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "recipes.jsonl")
        open(self.path, "w").close()
        self.cfg = ReaderConfig(videos=[os.path.join(self._tmp.name, self.VID)],
                                recipes_path=self.path, out_dir=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _client(self):
        from src.kdsocr.client import KdsOcrClient
        return KdsOcrClient(self.cfg, journey_path=None, start=False)

    def _emit(self, rec, screen_at):
        rec["screen_at"] = screen_at
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def test_filename_clock_is_understood(self):
        from src.kdsocr.clock import video_start_from_filename
        vs = video_start_from_filename(self.VID)
        self.assertEqual((vs.hour, vs.minute, vs.second), (12, 0, 7))

    def test_ticket_is_held_until_the_production_video_reaches_it(self):
        c = self._client()
        self.assertTrue(c._paced)
        # A ticket 10 minutes into the recording.
        self._emit(record(ref="CHK-30", hot_dogs=[dog("ORG CHL", 1)]),
                   "2026-09-13T12:10:07-07:00")
        c.set_master_time(60.0)          # production feed only 1 min in
        c.poll()
        self.assertIsNone(c.get_next_ticket())
        self.assertEqual(c.dashboard_state()["held_emissions"], 1)

        c.set_master_time(600.0)         # now 10 min in
        c.poll()
        t = c.get_next_ticket()
        self.assertIsNotNone(t)
        self.assertEqual(t.ticket_id, "CHK-30")

    def test_held_emissions_are_applied_in_recording_order(self):
        c = self._client()
        # Written out of order; must be absorbed appeared-then-updated.
        self._emit(record(ref="CHK-31", reason="updated",
                          hot_dogs=[dog("ORG CHL", 2)]),
                   "2026-09-13T12:05:07-07:00")
        self._emit(record(ref="CHK-31", reason="appeared",
                          hot_dogs=[dog("ORG CHL", 1)]),
                   "2026-09-13T12:01:07-07:00")
        c.set_master_time(3600.0)
        c.poll()
        t = c.get_next_ticket()
        # The later `updated` won, which only happens if ordering was applied.
        self.assertEqual(t.total_hotdogs, 2)

    def test_without_a_master_clock_nothing_stalls_forever(self):
        c = self._client()
        self._emit(record(ref="CHK-32", hot_dogs=[dog("ORG CHL", 1)]),
                   "2026-09-13T12:30:07-07:00")
        c.poll()                          # set_master_time never called
        self.assertIsNotNone(c.get_next_ticket())


    def test_production_clock_anchor_removes_the_recording_skew(self):
        from src.kdsocr.client import KdsOcrClient
        from src.kdsocr.clock import video_start_from_filename
        # KDS file starts at 12:00:07, production at 12:00:06.
        prod = video_start_from_filename(
            "Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_13_12_to_13_p0006_PDT.mkv")
        self.assertEqual((prod.hour, prod.minute, prod.second), (12, 0, 6))
        c = KdsOcrClient(self.cfg, journey_path=None, start=False,
                         master_start=prod)
        # A ticket on screen at 12:01:06 is 60s into the PRODUCTION video,
        # not 59s as the KDS file's own clock would say.
        self._emit(record(ref="CHK-34", hot_dogs=[dog("ORG CHL", 1)]),
                   "2026-09-13T12:01:06-07:00")
        c.set_master_time(59.0)
        c.poll()
        self.assertIsNone(c.get_next_ticket())
        c.set_master_time(60.0)
        c.poll()
        self.assertIsNotNone(c.get_next_ticket())

    def test_a_live_run_is_not_paced(self):
        from src.kdsocr.client import KdsOcrClient
        from src.kdsocr.reader import ReaderConfig
        cfg = ReaderConfig(rtsp="rtsp://host/s", recipes_path=self.path,
                           out_dir=self._tmp.name)
        c = KdsOcrClient(cfg, journey_path=None, start=False)
        self.assertFalse(c._paced)
        self._emit(record(ref="CHK-33", hot_dogs=[dog("ORG CHL", 1)]),
                   "2026-09-13T12:59:07-07:00")
        c.poll()
        self.assertIsNotNone(c.get_next_ticket())




class TestRecoveryAndCancellation(unittest.TestCase):
    """Failure paths that would otherwise wedge or lose a ticket."""

    def setUp(self):
        from src.kdsocr.reader import ReaderConfig
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "recipes.jsonl")
        open(self.path, "w").close()
        self.cfg = ReaderConfig(videos=["d.mkv"], recipes_path=self.path,
                                out_dir=self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _client(self):
        from src.kdsocr.client import KdsOcrClient
        return KdsOcrClient(self.cfg, journey_path=None, start=False)

    def _emit(self, rec):
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def test_ticket_unreadable_at_first_is_still_queued_when_it_reads(self):
        c = self._client()
        # First read: kds-ocr could not resolve the items at all.
        self._emit(record(ref="CHK-40", status="blocked", hot_dogs=[]))
        c.poll()
        self.assertIsNone(c.get_next_ticket())
        # A later read succeeds. The ticket must NOT be lost.
        self._emit(record(ref="CHK-40", reason="updated", status="ok",
                          hot_dogs=[dog("ORG CHL", 1,
                                        [{"ingredient": "chili", "qty": 1}])]))
        c.poll()
        t = c.get_next_ticket()
        self.assertIsNotNone(t)
        self.assertEqual(t.ticket_id, "CHK-40")
        # and it no longer counts as a problem
        self.assertNotIn("CHK-40", c.dashboard_state()["skipped"])

    def test_no_hotdog_ticket_that_gains_one_is_queued(self):
        c = self._client()
        self._emit(record(ref="CHK-41", hot_dogs=[],
                          ticket_items=[{"code": "SML FRY"}]))
        c.poll()
        self.assertIsNone(c.get_next_ticket())
        self._emit(record(ref="CHK-41", reason="updated",
                          hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        self.assertEqual(c.get_next_ticket().ticket_id, "CHK-41")

    def test_void_after_issue_cancels_instead_of_wedging(self):
        c = self._client()
        self._emit(record(ref="CHK-42", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        self.assertEqual(c.get_next_ticket().ticket_id, "CHK-42")
        # The crew strikes it while it is being built.
        self._emit(record(ref="CHK-42", reason="updated", status="voided",
                          hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        self.assertEqual(c.take_cancellations(), ["CHK-42"])
        self.assertEqual(c.take_cancellations(), [])       # drained

    def test_losing_every_hotdog_after_issue_cancels(self):
        c = self._client()
        self._emit(record(ref="CHK-43", hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        c.get_next_ticket()
        self._emit(record(ref="CHK-43", reason="updated", hot_dogs=[],
                          ticket_items=[{"code": "SML FRY"}]))
        c.poll()
        self.assertEqual(c.take_cancellations(), ["CHK-43"])

    def test_the_queue_advances_after_a_cancellation(self):
        c = self._client()
        self._emit(record(ref="CHK-44", hot_dogs=[dog("ORG CHL", 1)]))
        self._emit(record(ref="CHK-45", hot_dogs=[dog("ORG MUST", 1)]))
        c.poll()
        self.assertEqual(c.get_next_ticket().ticket_id, "CHK-44")
        self._emit(record(ref="CHK-44", reason="bumped", status="voided",
                          hot_dogs=[dog("ORG CHL", 1)]))
        c.poll()
        c.take_cancellations()
        # CHK-45 must now be reachable, not stuck behind the struck ticket.
        self.assertEqual(c.get_next_ticket().ticket_id, "CHK-45")

    def test_abandon_releases_the_board_without_a_verdict(self):
        from src.analysis.state_machine import OrderStateMachine
        m = IngredientMapper()
        sm = OrderStateMachine()
        em = RecipeEmission.from_record(record(hot_dogs=[
            dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])]), m)
        sm.on_kds_ticket(em.to_ticket())
        sm.abandon_current_order()
        self.assertIsNone(sm.current_ticket)             # board released
        self.assertEqual(sm.history[-1].status, OrderStatus.ABANDONED)
        # An abandoned order was never judged, so it is neither a pass nor a
        # fail and must not move the accuracy numbers at all.
        self.assertEqual(sm.stats.total_orders, 0)
        self.assertEqual(sm.stats.failed_orders, 0)
        self.assertEqual(sm.stats.passed_orders, 0)

    def test_a_judged_order_does_move_the_stats(self):
        from src.analysis.state_machine import OrderStateMachine
        m = IngredientMapper()
        sm = OrderStateMachine()
        em = RecipeEmission.from_record(record(hot_dogs=[
            dog("ORG CHL", 1, [{"ingredient": "chili", "qty": 1}])]), m)
        sm.on_kds_ticket(em.to_ticket())
        sm.finalize_current_order()
        self.assertEqual(sm.stats.total_orders, 1)

    def test_abandon_is_refused_when_nothing_is_open(self):
        from src.analysis.state_machine import OrderStateMachine
        sm = OrderStateMachine()
        self.assertIsNone(sm.abandon_current_order())



if __name__ == "__main__":
    unittest.main()
