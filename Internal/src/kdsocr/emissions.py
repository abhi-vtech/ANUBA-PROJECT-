"""Read what kds-ocr emits.

kds-ocr streams one JSON object per emission into its ``--recipes-out`` file:
once when a ticket **appears**, again on every real **item change**, and a
final time at the **bump**.  Later records for the same ``order_ref``
supersede earlier ones -- they are not deltas, each is the whole ticket as
currently read.

`RecipeEmission` is that record, validated and converted into our vocabulary.
`EmissionTailer` follows the file the way `tail -f` would, so the reader
process and this process stay decoupled: kds-ocr can crash and restart
without taking the detection loop with it.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from src.domain.schemas import LineItem, Ticket
from src.kdsocr.mapping import IngredientMapper, MappedIngredient, as_count

logger = logging.getLogger(__name__)

#: The three emission points, in lifecycle order.
APPEARED = "appeared"
UPDATED = "updated"
BUMPED = "bumped"

#: kds-ocr statuses that must NOT be turned into a build requirement.
#: `voided` means the crew struck the ticket -- cancel, never verify.
#: `blocked` means it refused to guess the items at all.
NON_BUILDABLE = frozenset({"voided", "blocked"})


@dataclass
class RecipeEmission:
    """One kds-ocr emission, mapped into our names and counts."""

    order_ref: str
    reason: str                              # appeared | updated | bumped
    status: str                              # ok | ok_with_alert | voided | partial | blocked
    channel: str = ""
    total_dogs: int = 0
    on_screen: bool = True
    alert: Optional[str] = None
    alerts: List[dict] = field(default_factory=list)
    #: One entry per hot-dog LINE: {"code", "qty", "counts": {ours: n}, "not_checkable": [...]}
    groups: List[dict] = field(default_factory=list)
    #: Order-level totals in our names, summed across every hot dog.
    totals: Dict[str, int] = field(default_factory=dict)
    #: Ingredients we could not check, deduplicated, in kds-ocr's spelling.
    not_checkable: List[str] = field(default_factory=list)
    #: Everything on the ticket, hot dog or not -- context for the dashboard.
    ticket_items: List[dict] = field(default_factory=list)
    ticket_count: Optional[int] = None
    emitted_at: str = ""
    raw: dict = field(default_factory=dict)

    @property
    def is_final(self) -> bool:
        return self.reason == BUMPED

    @property
    def has_hot_dogs(self) -> bool:
        return bool(self.groups)

    @property
    def struck(self) -> bool:
        """The crew voided it, or kds-ocr refused to read its items."""
        return self.status in NON_BUILDABLE

    @property
    def buildable(self) -> bool:
        """Should this emission drive a build requirement at all?

        Two very different reasons it may not, which callers must tell apart:
        `struck` is a problem, while a ticket with no hot dogs on it is simply
        nothing for us to verify -- see `KdsOcrClient._absorb`.
        """
        return not self.struck and self.has_hot_dogs

    @property
    def signature(self) -> tuple:
        """What must change for this to count as a different requirement.

        Compared instead of the whole record so a re-emission that only moved
        the timer, the amount or the age colour does not churn the order.
        """
        return tuple(sorted(
            (g["code"], g["qty"], tuple(sorted(g["counts"].items())))
            for g in self.groups
        )) + (self.total_dogs,)

    # -- construction ----------------------------------------------------

    @classmethod
    def from_record(cls, rec: dict, mapper: IngredientMapper) -> Optional["RecipeEmission"]:
        """Build from one raw kds-ocr JSON record, or None if unusable."""
        if not isinstance(rec, dict):
            return None
        ref = str(rec.get("order_ref") or "").strip()
        if not ref:
            logger.debug("kds-ocr record with no order_ref, skipped")
            return None

        groups: List[dict] = []
        skipped: Dict[str, MappedIngredient] = {}
        for line in rec.get("hot_dogs") or []:
            if not isinstance(line, dict):
                continue
            qty = max(1, as_count(line.get("qty", 1)))
            counts, not_ok = mapper.resolve_all(line.get("ingredients") or [])
            for m in not_ok:
                skipped.setdefault(m.source, m)
            # kds-ocr has ALREADY multiplied each ingredient by the line
            # quantity, but our LineItem.items is per-hotdog and gets
            # multiplied again by `count`.  Divide back out, and only claim the
            # grouping when it divides exactly -- otherwise keep the line whole
            # so a requirement is never silently rounded away.
            per_dog, exact = _per_dog(counts, qty)
            groups.append({
                "code": str(line.get("code") or line.get("name") or "hot-dog"),
                "name": str(line.get("name") or ""),
                "qty": qty if exact else 1,
                "counts": per_dog,
                "totals": dict(counts),
                "modifiers": list(line.get("modifiers") or []),
                "grouped": exact,
            })

        totals, total_not_ok = mapper.resolve_all(rec.get("totals") or [])
        for m in total_not_ok:
            skipped.setdefault(m.source, m)

        return cls(
            order_ref=ref,
            reason=str(rec.get("emit_reason") or APPEARED),
            status=str(rec.get("status") or "ok"),
            channel=str(rec.get("channel") or ""),
            total_dogs=max(0, as_count(rec.get("total_dogs", 0))) if rec.get("total_dogs") else 0,
            on_screen=bool(rec.get("on_screen", True)),
            alert=rec.get("alert"),
            alerts=list(rec.get("alerts") or []),
            groups=groups,
            totals=totals,
            not_checkable=sorted(skipped),
            ticket_items=list(rec.get("ticket_items") or []),
            ticket_count=rec.get("ticket_count"),
            emitted_at=str(rec.get("emitted_at") or ""),
            raw=rec,
        )

    # -- handing on ------------------------------------------------------

    def to_ticket(self) -> Ticket:
        """The requirement, in the shape `OrderStateMachine` already consumes.

        One kds-ocr hot-dog line becomes one `LineItem`, so the group structure
        survives: "2 ORG C/C" is one line of count 2, not two anonymous dogs.
        """
        line_items = [
            LineItem(variant=g["code"], count=g["qty"], items=dict(g["counts"]))
            for g in self.groups
        ]
        dogs = self.total_dogs or sum(g["qty"] for g in self.groups) or len(self.groups)
        return Ticket(
            ticket_id=self.order_ref,
            shortcut=self.channel,
            total_hotdogs=dogs,
            line_items=line_items,
            hotdog_specs={},
            expected_items=[],
            specs_cover_line_items=False,
        )


def _per_dog(counts: Dict[str, int], qty: int) -> tuple:
    """Undo kds-ocr's line multiplication. Returns (per_dog_counts, exact)."""
    if qty <= 1:
        return dict(counts), True
    per: Dict[str, int] = {}
    for name, total in counts.items():
        if total % qty:
            return dict(counts), False
        per[name] = total // qty
    return per, True


class EmissionTailer:
    """Follows a JSONL file that another process is appending to."""

    def __init__(self, path: str, from_start: bool = True):
        self.path = path
        self._fh = None
        self._inode = None
        self._buf = ""
        self._from_start = from_start
        self.bad_lines = 0

    def _open(self) -> bool:
        try:
            st = os.stat(self.path)
        except OSError:
            return False
        if self._fh is not None and self._inode == st.st_ino:
            return True
        # First open, or the file was replaced -- start over on the new one.
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            logger.info("kds-ocr recipe file replaced, reopening %s", self.path)
        try:
            self._fh = open(self.path, "r")
        except OSError:
            self._fh = None
            return False
        self._inode = st.st_ino
        self._buf = ""
        if not self._from_start:
            self._fh.seek(0, os.SEEK_END)
        return True

    def poll(self) -> Iterator[dict]:
        """Yield every complete JSON line appended since the last call.

        A partial final line is held until its newline arrives, so a record
        caught mid-write is never parsed as truncated JSON.
        """
        if not self._open():
            return
        chunk = self._fh.read()
        if not chunk:
            return
        self._buf += chunk
        *lines, self._buf = self._buf.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                self.bad_lines += 1
                logger.warning("unparseable kds-ocr line (%d so far): %.120s",
                               self.bad_lines, line)

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None
