"""kds-ocr ingredient names and quantities -> ours.

Two jobs, both narrow:

  1. NAME.  kds-ocr is spelled from the Operations Manual; we are spelled from
     the bin zones and the YOLO classes.  `config/kdsocr_ingredients.yaml` is
     the only place the correspondence lives.  An ingredient absent from both
     of its tables is *unknown*, which is reported -- never guessed at and
     never dropped.

  2. QUANTITY.  Every quantity we act on must be a COUNT of observable
     applications.  kds-ocr already collapses its weight units (`oz`,
     `pinch` -> 1) in ``recipes/convert.py``, but we re-assert it here rather
     than trust it: this layer is the boundary, and a float or an `oz` string
     arriving from a future upstream change must not silently become a
     requirement the crew cannot satisfy.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml

from src.domain.paths import resource
from src.domain.schemas import canonical_ingredient

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config/kdsocr_ingredients.yaml"

#: Anything measured rather than counted.  A weight cannot be verified by
#: watching a hand, so it collapses to "one application".
WEIGHT_UNITS = frozenset({"oz", "ounce", "ounces", "pinch", "dash", "g", "gram", "ml"})


class Verdictability(str):
    """Why an ingredient can or cannot be checked."""

    #: A topping we can observe -- it becomes a requirement.
    OK = "ok"
    #: Part of every hot dog by definition (the bun, the sausage itself).
    #: Covered by the hotdog COUNT, so it is dropped silently: not required,
    #: and not reported as "not checkable" either.  Listing "bun" against
    #: every ticket is noise that buries the toppings that do matter.
    BASE = "base"
    #: Something a crew member really can forget but we cannot see. Reported,
    #: never silently passed.
    UNVERIFIABLE = "unverifiable"
    #: In none of the tables. Reported and logged, never guessed at.
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MappedIngredient:
    """One kds-ocr ingredient line, resolved against our vocabulary."""

    source: str                  # the name kds-ocr used, verbatim
    name: Optional[str]          # our canonical name, or None if not checkable
    count: int                   # observable applications required
    status: str                  # Verdictability.*

    @property
    def checkable(self) -> bool:
        return self.status == Verdictability.OK and bool(self.name)

    @property
    def reportable(self) -> bool:
        """Should a verdict mention that this one could not be checked?

        A base component should not: it is covered by the hotdog count, and
        naming it on every ticket buries the toppings that do matter.
        """
        return self.status in (Verdictability.UNVERIFIABLE, Verdictability.UNKNOWN)


def as_count(qty: Any, unit: Optional[str] = None) -> int:
    """Kitchen quantity -> a number of observable applications.

    Requirement: the pipeline only ever reasons in counts.  A weight unit is
    one application regardless of its magnitude (you cannot see 1.5 oz of
    chili go on); anything countable keeps its number, rounded, floored at 1.
    """
    if unit and str(unit).strip().lower() in WEIGHT_UNITS:
        return 1
    try:
        n = float(qty)
    except (TypeError, ValueError):
        return 1
    if not math.isfinite(n):
        return 1
    return max(1, int(round(n)))


class IngredientMapper:
    """Resolves kds-ocr ingredient names against our zones and classes."""

    def __init__(self, config_path: Optional[str] = None):
        path = config_path or resource(DEFAULT_CONFIG)
        with open(path, "r") as fh:
            cfg = yaml.safe_load(fh) or {}
        self._map: Dict[str, str] = {
            self._key(k): canonical_ingredient(str(v))
            for k, v in (cfg.get("map") or {}).items()
        }
        self._base = {self._key(x) for x in (cfg.get("base") or [])}
        self._unverifiable = {self._key(x) for x in (cfg.get("unverifiable") or [])}
        # Reported once per name, not once per ticket: an unknown ingredient on
        # a popular item would otherwise flood the log.
        self._warned: set = set()
        logger.info(
            "kds-ocr ingredient map: %d checkable, %d base (covered by the "
            "hotdog count), %d known-unverifiable",
            len(self._map), len(self._base), len(self._unverifiable),
        )

    @staticmethod
    def _key(name: Any) -> str:
        return " ".join(str(name).strip().lower().split())

    def resolve(self, source: str, qty: Any = 1, unit: Optional[str] = None) -> MappedIngredient:
        key = self._key(source)
        count = as_count(qty, unit)
        if key in self._base:
            # Checked first, so moving a name into `base` always wins over an
            # older entry left behind in `map`.
            return MappedIngredient(source, None, count, Verdictability.BASE)
        if key in self._map:
            return MappedIngredient(source, self._map[key], count, Verdictability.OK)
        if key in self._unverifiable:
            return MappedIngredient(source, None, count, Verdictability.UNVERIFIABLE)
        if key and key not in self._warned:
            self._warned.add(key)
            logger.warning(
                "kds-ocr ingredient %r is in neither table of %s; treated as not "
                "checkable. Add it to `map` if we can observe it.", source, DEFAULT_CONFIG,
            )
        return MappedIngredient(source, None, count, Verdictability.UNKNOWN)

    def resolve_all(self, rows: Iterable[dict]) -> Tuple[Dict[str, int], list]:
        """kds-ocr ``[{"ingredient": ..., "qty": ...}]`` -> (counts, not_checkable).

        `counts` is keyed by our canonical name and is what a requirement is
        built from.  `not_checkable` keeps only the ingredients worth
        REPORTING -- something a crew member could forget but we cannot see --
        so the verdict can say "not checkable: bacon" instead of passing
        silently.  Base components (bun, the sausage) are dropped from both:
        they are covered by the hotdog count.
        """
        counts: Dict[str, int] = {}
        skipped: list = []
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            mapped = self.resolve(
                row.get("ingredient") or row.get("name") or "",
                row.get("qty", 1),
                row.get("unit"),
            )
            if mapped.checkable:
                counts[mapped.name] = counts.get(mapped.name, 0) + mapped.count
            elif mapped.reportable and mapped.source:
                skipped.append(mapped)
        return counts, skipped
