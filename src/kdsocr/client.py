"""The KDS side of the pipeline, driven by kds-ocr.

Replaces the old in-process screen reader.  Responsibilities, and no others:

  * run kds-ocr over the KDS screen and follow what it emits   (`poll`)
  * hand the production pipeline one ticket at a time, in the order the
    tickets were paid for                                      (`get_next_ticket`)
  * report that an open ticket's requirement CHANGED, so the order in
    progress can be updated rather than restarted               (`take_updates`)
  * report that a ticket was BUMPED, which is when it gets judged  (`take_bumps`)
  * keep the journey of every ticket for the dashboard          (`dashboard_state`)

The verdict trigger is kds-ocr's `bumped` emission.  Nothing here watches for a
card disappearing: the reader owns that question and answers it explicitly, so
there is no second, disagreeing copy of the rule.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional, Set

from src.domain.schemas import Ticket
from src.kdsocr.emissions import (
    APPEARED, BUMPED, UPDATED, EmissionTailer, RecipeEmission,
)
from src.kdsocr.clock import offset_seconds, video_start_from_filename
from src.kdsocr.journey import ALERT, HOTDOG, PLACE, REQUIREMENT, JourneyLog
from src.kdsocr.mapping import IngredientMapper
from src.kdsocr.reader import KdsOcrReader, ReaderConfig

logger = logging.getLogger(__name__)


class KdsOcrClient:
    """Implements the ticket-source interface `OrderStateMachine` expects."""

    def __init__(
        self,
        config: ReaderConfig,
        mapper: Optional[IngredientMapper] = None,
        journey_path: Optional[str] = "output/ticket_journeys.jsonl",
        start: bool = True,
        ingredient_config: Optional[str] = None,
        master_start=None,
    ):
        self.config = config
        self.mapper = mapper or IngredientMapper(ingredient_config)
        self.reader = KdsOcrReader(config)
        self.tailer = EmissionTailer(config.recipes_path, from_start=True)
        # Set before JourneyLog, whose clock is `self._now`.
        self._master_t: Optional[float] = None
        # The journey is stamped in the SAME clock main.py stamps place and
        # hotdog steps with -- the video's media time -- so a duration is
        # meaningful and the steps sort correctly.
        self.journeys = JourneyLog(journey_path, clock=self._now)

        self._fifo: List[str] = []                       # refs, in arrival order
        self._latest: Dict[str, RecipeEmission] = {}
        self._signature: Dict[str, tuple] = {}
        self._issued: Set[str] = set()
        self._judged: Set[str] = set()
        self._pending_updates: List[str] = []
        self._pending_bumps: List[str] = []
        self._pending_cancels: List[str] = []
        #: Refs that have ever been queued. Kept separately from `_latest`
        #: because a ticket can be seen while still unreadable and only become
        #: buildable on a later read -- "have we seen it" and "have we queued
        #: it" are different questions.
        self._queued: Set[str] = set()
        # Recorded runs are paced against the production video's media clock;
        # a live run needs no pacing because both feeds are the present.
        #
        # The anchor is the PRODUCTION video's start when the caller knows it,
        # not the KDS video's: the two recordings do not begin at the same
        # instant (…p0006_PDT and …p0007_PDT are a second apart), and the
        # master clock we are compared against is media time in the production
        # file.  Anchoring on the KDS file instead would bake that difference
        # in as a permanent skew -- the job the old pipeline's
        # `sync.kds_offset_s` did by hand.
        self._video_start = (video_start_from_filename(config.videos[0])
                             if config.videos else None)
        if config.videos and master_start is not None:
            if self._video_start is not None:
                skew = (self._video_start - master_start).total_seconds()
                if abs(skew) >= 0.5:
                    logger.info(
                        "KDS recording starts %+.1fs from the production "
                        "recording; pacing against the production clock", skew)
            self._video_start = master_start
        self._paced = bool(config.videos) and self._video_start is not None
        self._held: List[tuple] = []                     # (offset_s, emission)
        if config.videos and not self._paced:
            logger.warning(
                "no wall-clock in the KDS filename %r, so emissions cannot be "
                "paced against the production video; they will be applied as "
                "soon as they are read", os.path.basename(str(config.videos[0])))
        self._skipped: Dict[str, str] = {}               # ref -> why it was a problem
        self._no_hotdogs: Dict[str, str] = {}           # ref -> what was on it instead
        self.emissions_seen = 0

        if start:
            self.reader.start()

    # -- reading ---------------------------------------------------------

    def _now(self) -> float:
        """The pipeline's clock: media time once published, else the wall clock."""
        return self._master_t if self._master_t is not None else time.time()

    def set_master_time(self, t: float) -> None:
        """Publish the production video's media time (seconds).

        Only meaningful for a recorded run, where it holds each emission back
        until the production feed actually reaches the minute the ticket was
        on screen.
        """
        self._master_t = float(t)

    def poll(self) -> int:
        """Drain everything kds-ocr has written. Call once per frame; cheap."""
        new = 0
        for record in self.tailer.poll():
            emission = RecipeEmission.from_record(record, self.mapper)
            if emission is None:
                continue
            self.emissions_seen += 1
            new += 1
            if self._paced:
                at = offset_seconds(record.get("screen_at"), self._video_start)
                if at is not None:
                    self._held.append((at, emission))
                    continue
            self._absorb(emission)
        return new + self._release()

    def _release(self) -> int:
        """Absorb held emissions the production video has now caught up to."""
        if not self._held:
            return 0
        if self._master_t is None:
            # Nobody is publishing a clock, so pacing would stall the run
            # forever. Fall through and apply everything.
            due, self._held = self._held, []
        else:
            due = [x for x in self._held if x[0] <= self._master_t]
            if due:
                self._held = [x for x in self._held if x[0] > self._master_t]
        # Order matters: an `updated` must never be applied before its
        # `appeared`, so they go in recording order, not arrival order.
        for _, emission in sorted(due, key=lambda x: x[0]):
            self._absorb(emission)
        return len(due)

    def _absorb(self, em: RecipeEmission) -> None:
        ref = em.order_ref
        if ref in self._judged:
            # A late emission for an order we already judged. Recorded, never
            # allowed to reopen the verdict.
            logger.debug("kds-ocr emission for already-judged %s (%s)", ref, em.reason)
            return

        first_read = ref not in self._latest
        self._latest[ref] = em

        if em.struck:
            # Voided or unreadable: it must never become a build requirement.
            # Still recorded, so a struck ticket is visible rather than absent.
            reason = ("the crew voided this ticket" if em.status == "voided"
                      else "kds-ocr could not read its items (%s)" % em.status)
            if ref not in self._skipped:
                self._skipped[ref] = reason
                logger.info("kds-ocr ticket %s: %s; not queued", ref, reason)
                j = self.journeys.start(ref, em.channel)
                j.add(ALERT, status=em.status, alert=em.alert, detail=reason)
            if ref in self._fifo:
                self._fifo.remove(ref)
            if ref in self._issued and ref not in self._pending_cancels:
                # It was already being built. Without this the order stays
                # open forever -- no bump for it will ever be handed on -- and
                # every later ticket queues up behind it.
                self._pending_cancels.append(ref)
                logger.info("kds-ocr ticket %s was struck mid-build; cancelling", ref)
            if em.is_final:
                self._finish_unbuilt(ref, reason)
            return

        if not em.has_hot_dogs:
            # A fries-and-a-drink ticket. Nothing for the CV system to check,
            # which is a normal outcome and NOT a failure: it is recorded so the
            # board is complete, but it is never queued, never judged, and never
            # counted against accuracy.
            if ref not in self._no_hotdogs:
                self._no_hotdogs[ref] = ", ".join(
                    str(i.get("code") or i.get("name") or "?")
                    for i in em.ticket_items[:6]) or "no hot dogs"
                logger.info("kds-ocr ticket %s has no hot dogs (%s); nothing to verify",
                            ref, self._no_hotdogs[ref])
                j = self.journeys.start(ref, em.channel)
                j.add(ALERT, detail="no hot dogs on this ticket; nothing to verify")
            if ref in self._fifo:
                self._fifo.remove(ref)
            if ref in self._issued and ref not in self._pending_cancels:
                # Every hot dog was edited off a ticket we were already
                # building. There is nothing left to verify.
                self._pending_cancels.append(ref)
                logger.info("kds-ocr ticket %s lost all its hot dogs; cancelling", ref)
            if em.is_final:
                self._close_unjudged(ref, "no hot dogs on this ticket")
            return

        if ref not in self._queued and self._duplicate_misread(ref, em):
            # Not an order. A reference that is one inserted character away
            # from a ticket already on the board AND carries that ticket's
            # exact contents is that ticket, read wrong.
            #
            # Observed 2026-09-16: CHK-2651 and CHK-261 emitted at the same
            # screen_at with identical items, were both queued, both judged and
            # both failed -- one physical order producing two wrong verdicts.
            #
            # Identity is keyed on the reference's trailing digits, which
            # survives a digit being MISREAD but not one being INSERTED:
            # tail("2651") != tail("261"). Both tests are required here,
            # because two customers really can order the same thing at the same
            # time -- finalize() already refuses to merge CHK-486/487 for that
            # reason -- and equally a real ticket can sit one digit away from
            # another. Only the two together mean a misread.
            logger.warning(
                "kds-ocr ref %s ignored: same contents as %s already on the "
                "board and one character longer -- treating as a misread of it",
                ref, self._duplicate_misread(ref, em),
            )
            return

        if ref not in self._queued:
            # First read we could actually build from. A ticket that appeared
            # `blocked` or with no hot dogs and only became readable later
            # still enters the queue here -- keying this off "first sighting"
            # would lose it permanently.
            self._queued.add(ref)
            self._fifo.append(ref)
            self._skipped.pop(ref, None)
            self._no_hotdogs.pop(ref, None)
            j = self.journeys.start(ref, em.channel)
            j.add(REQUIREMENT, counts=dict(em.totals), reason=APPEARED,
                  groups=[{"code": g["code"], "qty": g["qty"]} for g in em.groups],
                  not_checkable=list(em.not_checkable), status=em.status)
            j.not_checkable = list(em.not_checkable)
            self._signature[ref] = em.signature
            logger.info("kds-ocr ticket %s appeared: %d dog(s), %s",
                        ref, em.total_dogs, self._describe(em))

        changed = self._signature.get(ref) != em.signature
        if changed and not first_read:
            self._signature[ref] = em.signature
            j = self.journeys.start(ref, em.channel)
            j.revisions += 1
            j.add(REQUIREMENT, counts=dict(em.totals), reason=UPDATED,
                  groups=[{"code": g["code"], "qty": g["qty"]} for g in em.groups],
                  not_checkable=list(em.not_checkable), status=em.status)
            j.not_checkable = list(em.not_checkable)
            logger.info("kds-ocr ticket %s updated (rev %d): %s",
                        ref, j.revisions, self._describe(em))
            if ref in self._issued and ref not in self._pending_updates:
                self._pending_updates.append(ref)

        if em.alert:
            j = self.journeys.start(ref, em.channel)
            j.add(ALERT, alert=em.alert, status=em.status,
                  detail="; ".join(str(a.get("detail", "")) for a in em.alerts)[:300])

        if em.is_final:
            if ref in self._issued:
                if ref not in self._pending_bumps:
                    self._pending_bumps.append(ref)
                    logger.info("kds-ocr ticket %s bumped; ready to judge", ref)
            else:
                # Bumped before the pipeline ever got to it -- the order was
                # made while we were still on an earlier ticket.  Not judged:
                # we have no evidence for it.
                if ref in self._fifo:
                    self._fifo.remove(ref)
                self._finish_unbuilt(ref, "bumped before the pipeline opened it")

    @staticmethod
    def _describe(em: RecipeEmission) -> str:
        parts = ["%d %s" % (g["qty"], g["code"]) for g in em.groups]
        text = ", ".join(parts) or "no hot dogs"
        if em.not_checkable:
            text += " (not checkable: %s)" % ", ".join(em.not_checkable[:4])
        return text

    def _finish_unbuilt(self, ref: str, why: str) -> None:
        """Close a ticket we could not verify, WITHOUT calling it wrong.

        We have no evidence either way, so a verdict would be invented.  It is
        closed as unjudged and surfaced as a warning instead: an unverified
        order is a gap in coverage, not a failed order, and counting it as
        wrong would understate accuracy and hide real failures.
        """
        self._skipped.setdefault(ref, why)
        self._close_unjudged(ref, why)

    def _close_unjudged(self, ref: str, why: str) -> None:
        self._judged.add(ref)
        self._issued.discard(ref)
        if ref in self._fifo:
            self._fifo.remove(ref)
        j = self.journeys.get(ref)
        if j is not None:
            j.add(ALERT, detail=why)
            self.journeys.finish(ref, correct=None, message="Not verified: %s" % why)
        logger.info("kds-ocr ticket %s closed without a verdict: %s", ref, why)

    # -- the ticket-source interface -------------------------------------

    def get_next_ticket(self) -> Optional[Ticket]:
        """The next unbuilt ticket, oldest first. None when there is nothing."""
        for ref in list(self._fifo):
            if ref in self._issued or ref in self._judged:
                continue
            em = self._latest.get(ref)
            if em is None or not em.buildable:
                continue
            self._issued.add(ref)
            logger.info("issuing kds-ocr ticket %s to the production pipeline", ref)
            return em.to_ticket()
        return None

    def get_all_tickets(self) -> List[Ticket]:
        return [self._latest[r].to_ticket() for r in self._fifo
                if r in self._latest and self._latest[r].buildable]

    def take_updates(self) -> List[Ticket]:
        """Requirement changes for tickets already being built. Drains the queue."""
        out: List[Ticket] = []
        while self._pending_updates:
            ref = self._pending_updates.pop(0)
            em = self._latest.get(ref)
            if em is not None and em.buildable and ref not in self._judged:
                out.append(em.to_ticket())
        return out

    def requeue_update(self, ref: str) -> None:
        """Put back an update the pipeline could not apply yet.

        `take_updates` drains, so an update handed over while the state machine
        is still on the previous ticket -- or has not opened this one yet, since
        the loop takes updates BEFORE it calls get_next_ticket -- used to be
        dropped permanently.  The order then kept whatever requirement its first
        emission carried, and a first emission rests on a single OCR read.

        Seen 2026-09-17 on CHK-251: `appeared` at n_reads=1 said 1 ORG PLAN,
        `bumped` at n_reads=134 said 2 ORG PLAN.  The correction was queued,
        taken, refused because the ticket was not open yet, and lost -- so the
        ticket was judged against the one-read number for the rest of its life.
        """
        if ref not in self._pending_updates and ref not in self._judged:
            self._pending_updates.append(ref)

    def take_cancellations(self) -> List[str]:
        """Tickets being built that must be dropped without a verdict. Drains."""
        out = [r for r in self._pending_cancels if r not in self._judged]
        self._pending_cancels.clear()
        return out

    def take_bumps(self) -> List[str]:
        """Tickets kds-ocr says are finished, so they can be judged. Drains."""
        out = [r for r in self._pending_bumps if r not in self._judged]
        self._pending_bumps.clear()
        return out

    def mark_completed(self, ticket_id: str) -> None:
        self._judged.add(ticket_id)
        self._issued.discard(ticket_id)
        if ticket_id in self._fifo:
            self._fifo.remove(ticket_id)

    def mark_abandoned(self, ticket_id: str) -> None:
        self.mark_completed(ticket_id)

    # -- evidence, for the journey ---------------------------------------

    def record_place(self, ticket_id: str, item: str, t: Optional[float] = None,
                     zone: str = "") -> None:
        j = self.journeys.get(ticket_id)
        if j is not None and item:
            j.add(PLACE, t, item=item, zone=zone)

    def record_hotdog(self, ticket_id: str, track_id: Optional[int] = None,
                      item: str = "", t: Optional[float] = None) -> None:
        j = self.journeys.get(ticket_id)
        if j is not None:
            j.add(HOTDOG, t, track_id=track_id, item=item)

    def record_verdict(self, ticket_id: str, correct: bool, message: str = "",
                       missing: Optional[dict] = None, extras: Optional[list] = None,
                       t: Optional[float] = None) -> None:
        """Close a ticket with a real verdict. The one place a verdict is logged."""
        em = self._latest.get(ticket_id)
        logger.info("TICKET %s: %s -- %s", ticket_id,
                    "CORRECT" if correct else "WRONG", message or "(no detail)")
        self.journeys.finish(
            ticket_id, correct=correct, message=message, missing=missing,
            extras=extras, not_checkable=list(em.not_checkable) if em else None, t=t,
        )
        self._judged.add(ticket_id)
        self._issued.discard(ticket_id)
        if ticket_id in self._fifo:
            self._fifo.remove(ticket_id)

    # -- state -----------------------------------------------------------

    def latest(self, ticket_id: str) -> Optional[RecipeEmission]:
        return self._latest.get(ticket_id)

    @property
    def has_screen_content(self) -> bool:
        """True while any ticket is open on the KDS, by kds-ocr's own reckoning."""
        return any(r not in self._judged for r in self._fifo)

    @staticmethod
    def _one_insertion(longer: str, shorter: str) -> bool:
        """True when `longer` is `shorter` with exactly one extra character."""
        if len(longer) != len(shorter) + 1:
            return False
        return any(longer[:i] + longer[i + 1:] == shorter
                   for i in range(len(longer)))

    def _duplicate_misread(self, ref: str, em) -> Optional[str]:
        """The ref this one is a misread of, or None.

        Requires BOTH tests: the reference is one inserted character longer than
        a ref already queued, and the contents are byte-identical by
        `signature` (codes, quantities, per-group counts and total dogs).
        """
        try:
            sig = em.signature
        except Exception:
            return None
        for known in self._queued:
            if known == ref:
                continue
            if not self._one_insertion(ref, known):
                continue
            if self._signature.get(known) == sig:
                return known
        return None

    def _queue_card(self, ref: str) -> dict:
        """One ticket in the shape the dashboard's KDS panel already renders."""
        em = self._latest.get(ref)
        j = self.journeys.get(ref)
        observed = j.observed_now() if j is not None else {}
        # Distinct tracks, not wrapping events -- and still only an upper
        # bound, because the tracker fragments one physical hotdog across many
        # ids (a 3-dog ticket measured 11 on camA). Capped below so the bar
        # reads as progress rather than claiming a count we cannot support.
        made = j.wrapped_tracks() if j is not None else 0
        expected = em.total_dogs if em else 0
        hotdogs = []
        for g in (em.groups if em else []):
            # Modifiers are shown as add-ons under their dog, which is the
            # shape the ticket itself has.
            addons = []
            for m in g.get("modifiers") or []:
                if not isinstance(m, dict):
                    continue
                addons.append({
                    "key": str(m.get("text") or m.get("ingredient") or ""),
                    "display": str(m.get("text") or m.get("ingredient") or ""),
                    "quantity": int(m.get("qty") or 1),
                    "negation": str(m.get("action") or "").lower() in ("remove", "no"),
                })
            hotdogs.append({
                "quantity": g["qty"],
                "shortcut": g["code"],
                "display": g.get("name") or g["code"],
                "known": True,
                "addons": addons,
                # Per-line attribution is not available: the detector has one
                # `hot-dog` class, so dogs are counted, not assigned to lines.
                "detected_count": min(g["qty"], made) if len(em.groups) == 1 else 0,
            })
        return {
            "ticket_id": ref,
            "queue_label": "ACTIVE" if ref in self._issued else "QUEUED",
            "state": ("in_progress" if ref in self._issued else "queued"),
            "expected_total": expected,
            # Capped: over-reporting progress reads as "this order is done"
            # on a ticket that is not.
            "detected_total": min(made, expected) if expected else made,
            "hotdogs": hotdogs,
            "observed": observed,
            "alert": em.alert if em else None,
            "not_checkable": list(em.not_checkable) if em else [],
        }

    def dashboard_state(self) -> dict:
        state = self.journeys.dashboard_state()
        judged = list(self.journeys.recent)
        wrong = sum(1 for j in judged if j.correct is False)
        live = [r for r in self._fifo if r not in self._judged]
        unjudged = [j for j in self.journeys.recent if j.correct is None]
        warnings = []
        problem = self.health()
        if problem:
            warnings.append(problem)
        # Only genuine problems. A ticket with no hot dogs on it is not one.
        for ref, why in self._skipped.items():
            warnings.append("%s: %s" % (ref, why))
        state.update({
            "source": "kds-ocr",
            "reader_alive": self.reader.alive,
            "emissions": self.emissions_seen,
            "queued": [r for r in live if r not in self._issued],
            "in_progress": sorted(self._issued),
            # `judged_total` counts only orders that actually got a verdict,
            # so accuracy is computed over what we really checked.
            "judged_total": sum(1 for j in judged if j.correct is not None),
            "wrong_total": wrong,
            "unverified_total": len(unjudged),
            "closed_total": len(self._judged),
            "skipped": dict(self._skipped),
            "no_hotdogs": dict(self._no_hotdogs),
            "bad_lines": self.tailer.bad_lines,
            "paced": self._paced,
            "held_emissions": len(self._held),
            # The two keys the existing KDS panel renders.  `counts` also
            # gates the panel: without it the panel decides no reader is
            # configured and falls back to the mock screenshot.
            "queue": [self._queue_card(r) for r in live],
            "counts": {
                "queued": sum(1 for r in live if r not in self._issued),
                "active": len(self._issued),
                "completed": sum(1 for j in judged if j.correct is True),
                "wrong": wrong,
            },
            "warnings": warnings[:6],
        })
        return state

    def health(self) -> Optional[str]:
        """A one-line problem description, or None while everything is fine."""
        if not self.reader.started:
            # Never launched (tests, or a caller feeding the stream itself).
            return None
        if not self.reader.alive:
            code = self.reader.exit_code()
            if code not in (0, None):
                return "kds-ocr exited with code %s -- see %s" % (
                    code, self.config.log_path)
            return "kds-ocr has finished reading the KDS feed"
        if not os.path.exists(self.config.recipes_path):
            return None
        return None

    def stop(self) -> None:
        self.reader.stop()
        # One last read: the child may have emitted a bump as it finalized.
        try:
            self.poll()
        except Exception:
            logger.debug("final kds-ocr poll failed", exc_info=True)
        self.tailer.close()
        # Anything still open never got a bump; close it honestly rather than
        # leaving it unaccounted.
        for ref in list(self.journeys.open):
            self._finish_unbuilt(ref, "run ended while the ticket was still open")
