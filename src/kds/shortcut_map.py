"""Shortcut -> physical item mapping, loaded from ``config/kds_shortcuts.yaml``.

RULE 4: only shortcuts present in the config are supported.  Anything else is
returned as an UNKNOWN match, logged by the caller, and never coerced onto a
neighbouring item.  Adding a menu item is a config edit; nothing in this module
or in the tracking/validation logic needs to change.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import yaml

from src.kds.schemas import UNKNOWN_SHORTCUT, AddOn, HotdogGroup
from src.domain.paths import resource

logger = logging.getLogger(__name__)

# Leading quantity, e.g. "2 ORG CHL", "10  ORG CHL", "2x ORG CHL".
_QTY_RE = re.compile(r"^\s*(\d{1,3})\s*[xX]?\s+(.*)$")
# A bare leading quantity glued to the text, e.g. "1ORG C/C".
_QTY_GLUED_RE = re.compile(r"^\s*(\d{1,3})\s*([A-Za-z].*)$")


def normalize(text: str) -> str:
    """Canonical form used for every lookup.

    Upper-cased, punctuation other than ``/`` and ``-`` dropped, whitespace
    collapsed.  ``"1 Org  C/C."`` and ``"ORG C/C"`` normalise to the same key
    once the quantity has been split off.
    """
    if not text:
        return ""
    cleaned = text.upper().replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"[^A-Z0-9/\- ]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_quantity(text: str) -> Tuple[int, str]:
    """Split a leading quantity off a KDS line.

    Returns ``(quantity, remainder)``; quantity defaults to 1 when the line
    carries none.  Handles the KDS's inconsistent spacing (``"1ORG C/C"``).
    """
    if not text:
        return 1, ""
    match = _QTY_RE.match(text)
    if match:
        return max(1, int(match.group(1))), match.group(2).strip()
    match = _QTY_GLUED_RE.match(text)
    if match:
        return max(1, int(match.group(1))), match.group(2).strip()
    return 1, text.strip()


@dataclass
class ShortcutDef:
    key: str
    shortcut: str
    item: str
    display: str
    ingredients: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)


@dataclass
class AddOnDef:
    key: str
    display: str
    ingredient: Optional[str] = None
    negation: bool = False
    aliases: List[str] = field(default_factory=list)


@dataclass
class ShortcutMatch:
    """Result of resolving one yellow-bar line."""

    known: bool
    quantity: int
    raw_text: str
    definition: Optional[ShortcutDef] = None
    score: float = 0.0

    def to_group(self) -> HotdogGroup:
        if not self.known or self.definition is None:
            return HotdogGroup(
                shortcut=normalize(self.raw_text) or UNKNOWN_SHORTCUT,
                item=UNKNOWN_SHORTCUT,
                display=self.raw_text.strip(),
                quantity=self.quantity,
                raw_text=self.raw_text,
                known=False,
            )
        return HotdogGroup(
            shortcut=self.definition.shortcut,
            item=self.definition.item,
            display=self.definition.display,
            quantity=self.quantity,
            ingredients=list(self.definition.ingredients),
            raw_text=self.raw_text,
            known=True,
        )


@dataclass
class AddOnMatch:
    known: bool
    quantity: int
    raw_text: str
    definition: Optional[AddOnDef] = None

    def to_addon(self) -> AddOn:
        if not self.known or self.definition is None:
            label = self.raw_text.strip() or "UNKNOWN"
            return AddOn(
                key="UNKNOWN:" + normalize(label),
                display=label,
                quantity=self.quantity,
                ingredient=None,
                raw_text=self.raw_text,
            )
        return AddOn(
            key=self.definition.key,
            display=self.definition.display,
            quantity=self.quantity,
            ingredient=self.definition.ingredient,
            negation=self.definition.negation,
            raw_text=self.raw_text,
        )


class ShortcutMapper:
    """Loads and queries the shortcut / add-on vocabulary.

    The mapper is intentionally the *only* component that knows menu names.
    ``ticket_parser`` asks it what a line means; ``fifo_queue`` asks it nothing
    at all -- it works purely with the resolved ``HotdogGroup`` objects.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or resource("config/kds_shortcuts.yaml")
        self.detection_class: str = "hot-dog"
        self._shortcuts: Dict[str, ShortcutDef] = {}
        self._addons: Dict[str, AddOnDef] = {}
        # normalised alias -> definition
        self._shortcut_index: Dict[str, ShortcutDef] = {}
        self._addon_index: Dict[str, AddOnDef] = {}
        self._non_hotdog: set = set()
        self.fuzzy_threshold: float = 0.82
        self.enable_fuzzy: bool = True
        self._unknown_seen: Dict[str, int] = {}
        self._load()

    # ------------------------------------------------------------------ load

    def _load(self) -> None:
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
        except FileNotFoundError:
            logger.error("KDS shortcut config not found: %s", self.config_path)
            data = {}

        self.detection_class = data.get("detection_class", "hot-dog")

        for key, raw in (data.get("shortcuts") or {}).items():
            definition = ShortcutDef(
                key=key,
                shortcut=raw.get("shortcut", key),
                item=raw.get("item", key),
                display=raw.get("display", raw.get("shortcut", key)),
                ingredients=list(raw.get("ingredients") or []),
                aliases=list(raw.get("aliases") or []),
            )
            self._shortcuts[key] = definition
            for alias in [definition.shortcut, definition.key] + definition.aliases:
                self._shortcut_index[normalize(alias)] = definition

        for key, raw in (data.get("addons") or {}).items():
            definition = AddOnDef(
                key=key,
                display=raw.get("display", key.title()),
                ingredient=raw.get("ingredient"),
                negation=bool(raw.get("negation", False)),
                aliases=list(raw.get("aliases") or []),
            )
            self._addons[key] = definition
            for alias in [definition.display, definition.key] + definition.aliases:
                self._addon_index[normalize(alias)] = definition

        self._non_hotdog = {normalize(x) for x in (data.get("non_hotdog_items") or [])}

        matching = data.get("matching") or {}
        self.fuzzy_threshold = float(matching.get("fuzzy_threshold", 0.82))
        self.enable_fuzzy = bool(matching.get("enable_fuzzy", True))

        logger.info(
            "Loaded %d hotdog shortcuts and %d add-ons from %s",
            len(self._shortcuts),
            len(self._addons),
            self.config_path,
        )

    # --------------------------------------------------------------- queries

    @property
    def shortcuts(self) -> Dict[str, ShortcutDef]:
        return dict(self._shortcuts)

    @property
    def addons(self) -> Dict[str, AddOnDef]:
        return dict(self._addons)

    def is_non_hotdog_line(self, text: str) -> bool:
        """True for known menu lines we deliberately ignore (fries, drinks).

        Keeps the UNKNOWN_SHORTCUT signal meaningful: an ignored corn dog is
        not the same thing as an unreadable hotdog bar.

        Matching is fuzzy here for the same reason it is fuzzy for shortcuts --
        OCR drift turns "SML FRY" into "SMIL FRY".  Being lenient about a line
        we are going to ignore anyway costs nothing and keeps the unknown log
        clean; being lenient about a hotdog would risk mis-mapping an order,
        which is why that path keeps a stricter threshold.
        """
        _, body = split_quantity(text)
        key = normalize(body)
        if not key:
            return False
        if key in self._non_hotdog:
            return True
        if any(known and known in key for known in self._non_hotdog):
            return True
        if not self.enable_fuzzy:
            return False
        collapsed = key.replace(" ", "")
        for known in self._non_hotdog:
            if not known:
                continue
            if difflib.SequenceMatcher(None, key, known).ratio() >= self.fuzzy_threshold:
                return True
            if collapsed == known.replace(" ", ""):
                return True
        return False

    def item_display(self, item: str) -> str:
        for definition in self._shortcuts.values():
            if definition.item == item:
                return definition.display
        return item

    def ingredients_for_item(self, item: str) -> List[str]:
        for definition in self._shortcuts.values():
            if definition.item == item:
                return list(definition.ingredients)
        return []

    def resolve_shortcut(self, text: str, log_unknown: bool = True) -> ShortcutMatch:
        """Resolve one yellow-bar line to a supported item, or UNKNOWN.

        ``log_unknown=False`` is for speculative probes (e.g. "could these two
        rows be one wrapped line?") where a miss is expected and must not
        pollute the UNKNOWN_SHORTCUT signal.
        """
        quantity, body = split_quantity(text)
        key = normalize(body)
        if not key:
            return ShortcutMatch(known=False, quantity=quantity, raw_text=text)

        definition = self._shortcut_index.get(key)
        if definition is not None:
            return ShortcutMatch(True, quantity, text, definition, 1.0)

        # Tolerate stray trailing tokens the OCR picked up from the bar edge.
        collapsed = key.replace(" ", "")
        for alias, candidate in self._shortcut_index.items():
            if alias.replace(" ", "") == collapsed:
                return ShortcutMatch(True, quantity, text, candidate, 0.99)

        if self.enable_fuzzy:
            best, score = self._fuzzy(key, self._shortcut_index)
            if best is not None and score >= self.fuzzy_threshold:
                return ShortcutMatch(True, quantity, text, best, score)

        if log_unknown:
            self._unknown_seen[key] = self._unknown_seen.get(key, 0) + 1
            if self._unknown_seen[key] == 1:
                logger.warning("UNKNOWN SHORTCUT: %r (normalised %r)", text, key)
        return ShortcutMatch(known=False, quantity=quantity, raw_text=text)

    def resolve_addon(self, text: str, log_unknown: bool = True) -> AddOnMatch:
        """Resolve one grey line to a known add-on, or keep it as free text."""
        quantity, body = split_quantity(text)
        key = normalize(body)
        if not key:
            return AddOnMatch(known=False, quantity=quantity, raw_text=text)

        definition = self._addon_index.get(key)
        if definition is not None:
            return AddOnMatch(True, quantity, text, definition)

        collapsed = key.replace(" ", "")
        for alias, candidate in self._addon_index.items():
            if alias.replace(" ", "") == collapsed:
                return AddOnMatch(True, quantity, text, candidate)

        if self.enable_fuzzy:
            best, score = self._fuzzy(key, self._addon_index)
            if best is not None and score >= self.fuzzy_threshold:
                return AddOnMatch(True, quantity, text, best)

        if log_unknown:
            logger.info("Unrecognised add-on text kept verbatim: %r", text)
        return AddOnMatch(known=False, quantity=quantity, raw_text=text)

    @staticmethod
    def _fuzzy(key: str, index: Dict[str, object]) -> Tuple[Optional[object], float]:
        best_alias = None
        best_score = 0.0
        for alias in index:
            score = difflib.SequenceMatcher(None, key, alias).ratio()
            if score > best_score:
                best_score = score
                best_alias = alias
        if best_alias is None:
            return None, 0.0
        return index[best_alias], best_score

    def unknown_report(self) -> Dict[str, int]:
        """Every unrecognised yellow-bar text seen, with an occurrence count."""
        return dict(self._unknown_seen)


_DEFAULT_MAPPER: Optional[ShortcutMapper] = None


def default_mapper() -> ShortcutMapper:
    """Process-wide mapper, loaded once."""
    global _DEFAULT_MAPPER
    if _DEFAULT_MAPPER is None:
        _DEFAULT_MAPPER = ShortcutMapper()
    return _DEFAULT_MAPPER
