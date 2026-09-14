"""Group-aware order validation, with extras.

Three checks, run in order, each able to fail the order on its own:

  1. HOTDOG COUNT   -- are the required hotdogs actually there?
  2. REQUIRED ITEMS -- is every item the ticket asks for present, per group?
  3. EXTRAS         -- was anything added that the ticket did NOT ask for?

Check 3 is the one the current pipeline cannot perform.  `OrderStateMachine`
drops any place event whose well is not in `required_counts` before the
validator ever sees it (src/state_machine.py, "Only count ingredients that are
required by the active KDS ticket"), so an order with chilli added to a plain
dog looks identical to a correct plain dog.  The gate is not needed for its
stated purpose -- suppressing noise from wells nobody touched -- because a
place event only fires on an actual hand-in-well dwell.  Here every well is
observed and the *validator* decides whether an observation was required,
extra, or forbidden.  That is the whole fix: move the decision from the
event-producing edge to the judging centre.

Extras are graded, because not every extra deserves the same response:

    FORBIDDEN  the ticket said NO ONIONS and onions went on   -> always wrong
    UNEXPECTED a well not on the ticket at all                -> wrong unless advisory
    OVER       a required item applied more times than asked  -> quantity issue

`ExtrasPolicy.advisory` exists because a brand-new signal should not be allowed
to fail orders on day one.  Run advisory first, read the `extras` field on real
traffic, then turn it on.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.core.ticket_spec import HOTDOG_LABEL, TicketSpec


class Failure(str, Enum):
    MISSING_ITEM = "MISSING_ITEM"
    MISSING_HOTDOG = "MISSING_HOTDOG"
    EXTRA_HOTDOG = "EXTRA_HOTDOG"
    UNEXPECTED_ITEM = "UNEXPECTED_ITEM"
    FORBIDDEN_ITEM = "FORBIDDEN_ITEM"
    WRONG_QUANTITY = "WRONG_QUANTITY"


class ExtraKind(str, Enum):
    UNEXPECTED = "unexpected"
    FORBIDDEN = "forbidden"
    OVER = "over"


@dataclass
class ExtrasPolicy:
    """How hard to judge things the ticket did not ask for."""

    #: Report extras but never fail the order on them.  Start here.
    advisory: bool = True
    #: An "over" count fails only past this slack.  Sauce passes are bursty and
    #: a second mustard stripe is not a wrong order, so allow one.
    over_tolerance: int = 1
    #: Wells that are free to appear on any order (shared/garnish wells).
    ignore: frozenset = frozenset()
    #: Items whose short-count is forgiven at final validation when at least
    #: one application was seen -- a partial sauce pass is still a sauce pass.
    #: Mirrors the existing forgiveness in BatchOrderValidator.validate().
    forgive_short: frozenset = frozenset({"ketchup", "yellow_mustard_sauce"})


@dataclass
class Extra:
    item: str
    kind: ExtraKind
    observed: int
    required: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {"item": self.item, "kind": self.kind.value,
                "observed": self.observed, "required": self.required}


@dataclass
class Verdict:
    correct: bool
    ticket_id: str
    checks: Dict[str, bool] = field(default_factory=dict)
    required_hotdogs: int = 0
    observed_hotdogs: int = 0
    missing: Dict[str, int] = field(default_factory=dict)
    extras: List[Extra] = field(default_factory=list)
    failures: List[Failure] = field(default_factory=list)
    unverifiable: List[str] = field(default_factory=list)
    message: str = ""
    advisory_extras: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correct": self.correct,
            "ticket_id": self.ticket_id,
            "checks": dict(self.checks),
            "required_hotdogs": self.required_hotdogs,
            "observed_hotdogs": self.observed_hotdogs,
            "missing": dict(self.missing),
            "extras": [e.to_dict() for e in self.extras],
            "failures": [f.value for f in self.failures],
            "unverifiable": list(self.unverifiable),
            "advisory_extras": self.advisory_extras,
            "message": self.message,
        }


class OrderValidator:
    """Accumulates observations for one ticket, then judges it.

    Deliberately has no notion of frames, zones or tracking: it is fed
    `observe_place()` / `observe_hotdog()` by whatever produced those events.
    That is what lets the identical validator sit behind the Ultralytics loop
    and the DeepStream probe.
    """

    def __init__(self, spec: TicketSpec, policy: Optional[ExtrasPolicy] = None):
        self.spec = spec
        self.policy = policy or ExtrasPolicy()
        self.required = spec.required_counts()
        self.forbidden = spec.forbidden_counts()
        self.observed: Dict[str, int] = {}
        self.hotdog_ids: set = set()
        self.log: List[Dict[str, Any]] = []

    # -- observation -----------------------------------------------------

    def observe_place(self, item: str, t: float = 0.0, track_id: Optional[int] = None) -> None:
        """Record an ingredient application. Every well, not just required ones."""
        if not item:
            return
        self.observed[item] = self.observed.get(item, 0) + 1
        self.log.append({"t": round(t, 2), "item": item, "track_id": track_id,
                         "expected": item in self.required})

    def observe_hotdog(self, track_id: int) -> None:
        """Record a distinct hotdog identity. Idempotent per track id."""
        if track_id is not None and track_id >= 0:
            self.hotdog_ids.add(track_id)

    def set_hotdog_count(self, count: int) -> None:
        """For runtimes that report a count rather than identities."""
        self.hotdog_ids = set(range(max(0, int(count))))

    # -- judgement -------------------------------------------------------

    def validate(self, final: bool = False) -> Verdict:
        observed = dict(self.observed)
        if final:
            for item in self.policy.forgive_short:
                need = self.required.get(item, 0)
                got = observed.get(item, 0)
                if 0 < got < need:
                    observed[item] = need

        failures: List[Failure] = []

        # -- check 1: required hotdogs ----------------------------------
        required_hd = self.spec.required_hotdogs
        observed_hd = len(self.hotdog_ids)
        # Hotdogs may also be counted as a placed item by sources that treat
        # the dish as a well; take whichever signal is stronger.
        observed_hd = max(observed_hd, observed.get(HOTDOG_LABEL, 0))
        check_hotdogs = observed_hd >= required_hd
        if not check_hotdogs:
            failures.append(Failure.MISSING_HOTDOG)
        elif required_hd and observed_hd > required_hd:
            # Not a failure on its own: tracker fragmentation inflates this.
            failures.append(Failure.EXTRA_HOTDOG)

        # -- check 2: required items present ----------------------------
        missing: Dict[str, int] = {}
        for item, need in self.required.items():
            if item == HOTDOG_LABEL:
                continue
            got = observed.get(item, 0)
            if got < need:
                missing[item] = need - got
        check_items = not missing
        if missing:
            failures.append(Failure.MISSING_ITEM)

        # -- check 3: extras --------------------------------------------
        extras: List[Extra] = []
        for item, count in sorted(observed.items()):
            if count <= 0 or item == HOTDOG_LABEL or item in self.policy.ignore:
                continue
            if item in self.forbidden:
                extras.append(Extra(item, ExtraKind.FORBIDDEN, count, 0))
                continue
            need = self.required.get(item, 0)
            if need == 0:
                extras.append(Extra(item, ExtraKind.UNEXPECTED, count, 0))
            elif count > need + self.policy.over_tolerance:
                extras.append(Extra(item, ExtraKind.OVER, count, need))

        forbidden_hit = [e for e in extras if e.kind is ExtraKind.FORBIDDEN]
        unexpected = [e for e in extras if e.kind is ExtraKind.UNEXPECTED]
        over = [e for e in extras if e.kind is ExtraKind.OVER]

        if forbidden_hit:
            failures.append(Failure.FORBIDDEN_ITEM)
        if unexpected and not self.policy.advisory:
            failures.append(Failure.UNEXPECTED_ITEM)
        if over and not self.policy.advisory:
            failures.append(Failure.WRONG_QUANTITY)

        # A forbidden item is an explicit instruction violated, so it fails the
        # order even in advisory mode -- advisory covers only wells the ticket
        # never mentioned, where our well vocabulary may simply be incomplete.
        check_extras = not forbidden_hit and (
            self.policy.advisory or not (unexpected or over)
        )

        correct = check_hotdogs and check_items and check_extras
        verdict = Verdict(
            correct=correct,
            ticket_id=self.spec.ticket_id,
            checks={"hotdogs": check_hotdogs, "required_items": check_items, "extras": check_extras},
            required_hotdogs=required_hd,
            observed_hotdogs=observed_hd,
            missing=missing,
            extras=extras,
            failures=failures,
            unverifiable=list(self.spec.unverifiable),
            advisory_extras=self.policy.advisory,
        )
        verdict.message = self._message(verdict)
        return verdict

    def _message(self, v: Verdict) -> str:
        if v.correct and not v.extras:
            return "Order {0}: correct - {1} hotdog(s), all items confirmed.".format(
                v.ticket_id, v.observed_hotdogs)
        parts: List[str] = []
        if not v.checks["hotdogs"]:
            parts.append("expected {0} hotdog(s), saw {1}".format(
                v.required_hotdogs, v.observed_hotdogs))
        if v.missing:
            parts.append("missing: " + ", ".join(
                "{0}x {1}".format(n, i) for i, n in sorted(v.missing.items())))
        for kind, label in ((ExtraKind.FORBIDDEN, "should not be on this order"),
                            (ExtraKind.UNEXPECTED, "not on ticket"),
                            (ExtraKind.OVER, "more than asked")):
            hits = [e for e in v.extras if e.kind is kind]
            if hits:
                parts.append("{0}: {1}".format(label, ", ".join(
                    "{0}x {1}".format(e.observed, e.item) for e in hits)))
        if v.unverifiable:
            parts.append("not checkable: " + ", ".join(v.unverifiable))
        head = "correct, with notes" if v.correct else "issue detected - recheck order"
        return "Order {0}: {1} - {2}.".format(v.ticket_id, head, "; ".join(parts))
