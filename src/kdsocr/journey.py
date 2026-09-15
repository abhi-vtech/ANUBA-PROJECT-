"""The whole life of one ticket, kept so a WRONG verdict can be explained.

A verdict on its own ("wrong: missing chilli") is not actionable -- the crew
needs to know whether the chilli was never applied, or applied before the
ticket was read, or whether the ticket itself changed halfway through.  So
every input that moved the order is recorded in order:

    requirement   what kds-ocr said the ticket needs (once per emission)
    place         an ingredient we observed going on
    hotdog        a physical hotdog we counted
    alert         something kds-ocr flagged about the read itself
    verdict       the final judgement, with what was missing or extra

The journey is what the dashboard shows and what gets appended to disk; it is
deliberately a flat list of timestamped steps rather than a summary, because
the ordering is the diagnosis.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REQUIREMENT = "requirement"
PLACE = "place"
HOTDOG = "hotdog"
ALERT = "alert"
VERDICT = "verdict"


@dataclass
class Step:
    kind: str
    t: float
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "t": round(self.t, 2), **self.detail}


@dataclass
class TicketJourney:
    ticket_id: str
    opened_at: float
    channel: str = ""
    steps: List[Step] = field(default_factory=list)
    #: True/False once judged; None while open, and also once closed
    #: without a verdict (voided, or never built).
    correct: Optional[bool] = None
    verdict_at: Optional[float] = None
    message: str = ""
    missing: Dict[str, int] = field(default_factory=dict)
    extras: List[dict] = field(default_factory=list)
    not_checkable: List[str] = field(default_factory=list)
    #: How many times kds-ocr changed the requirement while it was open.
    revisions: int = 0
    #: The most recent timestamp seen, in whatever clock the caller is using.
    #: `duration_s` measures against this rather than against `time.time()`:
    #: the pipeline stamps steps with the video's MEDIA time, and mixing that
    #: with a wall clock produced a nonsense duration (a media time of 81
    #: minus a unix timestamp clamps to zero).  One clock, whichever the
    #: caller chose.
    last_t: float = 0.0

    def add(self, kind: str, t: Optional[float] = None, **detail) -> None:
        at = t if t is not None else self.last_t
        self.last_t = max(self.last_t, at)
        self.steps.append(Step(kind, at, detail))

    @property
    def duration_s(self) -> float:
        end = self.verdict_at if self.verdict_at is not None else self.last_t
        return max(0.0, end - self.opened_at)

    def requirement_now(self) -> Dict[str, int]:
        """The most recent requirement, for showing alongside what was seen."""
        for step in reversed(self.steps):
            if step.kind == REQUIREMENT:
                return dict(step.detail.get("counts") or {})
        return {}

    def wrapped_tracks(self) -> int:
        """Distinct hotdog TRACKS seen wrapped -- an upper bound, not a count.

        Deliberately not called "hotdogs made".  Counting wrapping events
        over-reports badly (one dog fires several), and de-duplicating by
        track id is only a partial fix: the tracker fragments a single
        physical hotdog across many ids, so a verified 3-dog ticket measured
        11 distinct tracks on camA.  `src/core/order_rules.py` treats the same
        signal the same way -- EXTRA_HOTDOG is explicitly not a failure on its
        own "because tracker fragmentation inflates this".

        Useful as evidence that wrapping happened and roughly when; useless as
        a quantity.  Anything needing a real count must wait for the tracker to
        stop fragmenting.
        """
        return len({s.detail.get("track_id") for s in self.steps
                    if s.kind == HOTDOG and s.detail.get("track_id") is not None})

    def observed_now(self) -> Dict[str, int]:
        seen: Dict[str, int] = {}
        for step in self.steps:
            if step.kind == PLACE:
                item = step.detail.get("item")
                if item:
                    seen[item] = seen.get(item, 0) + 1
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "channel": self.channel,
            "opened_at": round(self.opened_at, 2),
            "duration_s": round(self.duration_s, 1),
            "revisions": self.revisions,
            "correct": self.correct,
            "message": self.message,
            "missing": dict(self.missing),
            "extras": list(self.extras),
            "not_checkable": list(self.not_checkable),
            "requirement": self.requirement_now(),
            "observed": self.observed_now(),
            # An upper bound inflated by track fragmentation -- named for what
            # it measures, so no reader mistakes it for a hotdog count.
            "wrapped_tracks": self.wrapped_tracks(),
            "steps": [s.to_dict() for s in self.steps],
        }


class JourneyLog:
    """Open journeys, recently judged ones, and an append-only disk record."""

    def __init__(self, path: Optional[str] = None, keep_recent: int = 25,
                 clock=None):
        self.path = path
        self.keep_recent = max(1, keep_recent)
        self.open: Dict[str, TicketJourney] = {}
        self.recent: List[TicketJourney] = []
        #: Supplies "now" in the SAME clock the caller stamps steps with.
        #: The pipeline passes the video's media time; the default wall clock
        #: only applies when nobody supplies one.
        self.clock = clock or time.time

    def now(self) -> float:
        try:
            return float(self.clock())
        except Exception:                     # a caller's clock must not end a run
            logger.debug("journey clock failed; falling back to wall clock",
                         exc_info=True)
            return time.time()

    def start(self, ticket_id: str, channel: str = "", t: Optional[float] = None) -> TicketJourney:
        j = self.open.get(ticket_id)
        if j is None:
            at = t if t is not None else self.now()
            j = TicketJourney(ticket_id=ticket_id, opened_at=at,
                              channel=channel, last_t=at)
            self.open[ticket_id] = j
        return j

    def get(self, ticket_id: str) -> Optional[TicketJourney]:
        return self.open.get(ticket_id)

    def finish(self, ticket_id: str, correct: Optional[bool], message: str = "",
               missing: Optional[dict] = None, extras: Optional[list] = None,
               not_checkable: Optional[list] = None,
               t: Optional[float] = None) -> Optional[TicketJourney]:
        j = self.open.pop(ticket_id, None)
        if j is None:
            return None
        # `None` means closed WITHOUT a verdict -- a voided ticket, or one the
        # pipeline never got to. That is not a wrong order and must never be
        # counted as one.
        j.correct = None if correct is None else bool(correct)
        j.verdict_at = t if t is not None else self.now()
        j.message = message
        j.missing = dict(missing or {})
        j.extras = list(extras or [])
        if not_checkable:
            j.not_checkable = list(not_checkable)
        j.add(VERDICT, j.verdict_at, correct=j.correct, message=message,
              missing=j.missing, extras=j.extras,
              judged=j.correct is not None)
        self.recent.insert(0, j)
        del self.recent[self.keep_recent:]
        self._append(j)
        return j

    def _append(self, j: TicketJourney) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.path, "a") as fh:
                fh.write(json.dumps(j.to_dict(), default=str) + "\n")
        except OSError:
            # A full or read-only disk must not end the run.
            logger.warning("could not append ticket journey to %s", self.path, exc_info=True)

    def dashboard_state(self) -> Dict[str, Any]:
        """Open journeys first, then the last judged ones (wrong ones matter most)."""
        return {
            "open": [j.to_dict() for j in self.open.values()],
            "recent": [j.to_dict() for j in self.recent],
            "wrong_recent": [j.to_dict() for j in self.recent if j.correct is False],
            "unjudged_recent": [j.to_dict() for j in self.recent if j.correct is None],
        }
