"""Ticket -> group-structured requirement, from JSON or from OCR.

`BatchOrderValidator` flattens a ticket into one `required_counts` dict at
construction time.  That loses which hotdog each item belonged to, so "hotdog 1
got both mustards and hotdog 2 got none" validates identically to "each got
one".  `TicketSpec` keeps the grouping and flattens only where a check
genuinely needs a total.

Two front doors, one output:

    from_json(dict)         config/kds_mock.json, and any ticket POSTed as JSON
    from_snapshot(snapshot) src.kds.schemas.TicketSnapshot, straight off OCR

Both normalise item names through `src.core.naming.normalize_item_name`, so
"Pickles (Rounds)" from OCR and "pickle_rounds" from JSON land on one key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.core.naming import normalize_item_name

HOTDOG_LABEL = "hot-dog"


@dataclass(frozen=True)
class ItemReq:
    """One required (or forbidden) ingredient inside one group."""

    name: str
    qty: int = 1
    #: True for "NO ONIONS" style lines: observing this item is an error.
    negated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "qty": self.qty, "negated": self.negated}


@dataclass
class GroupSpec:
    """One hotdog (or one run of identical hotdogs) and its add-ons.

    `quantity` is how many physical hotdogs this group covers -- a KDS line of
    "2 ORG CHICGO" is one group with quantity 2, and its item quantities are
    per-hotdog, so the flattened requirement multiplies the two.
    """

    group_id: str
    variant: str = ""
    quantity: int = 1
    items: List[ItemReq] = field(default_factory=list)

    @property
    def required_items(self) -> List[ItemReq]:
        return [i for i in self.items if not i.negated]

    @property
    def forbidden_items(self) -> List[ItemReq]:
        return [i for i in self.items if i.negated]

    def flattened(self) -> Dict[str, int]:
        """Item -> total count this group needs across all its hotdogs."""
        out: Dict[str, int] = {}
        for item in self.required_items:
            out[item.name] = out.get(item.name, 0) + item.qty * self.quantity
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "variant": self.variant,
            "quantity": self.quantity,
            "items": [i.to_dict() for i in self.items],
        }


@dataclass
class TicketSpec:
    """A whole ticket, still grouped."""

    ticket_id: str
    groups: List[GroupSpec] = field(default_factory=list)
    shortcut: str = ""
    #: Items the ticket names that we have no well/class for.  Kept so the
    #: validator can say "not checkable" instead of silently passing them.
    unverifiable: List[str] = field(default_factory=list)
    source: str = "json"

    @property
    def required_hotdogs(self) -> int:
        return sum(g.quantity for g in self.groups)

    def required_counts(self) -> Dict[str, int]:
        """Every required item across every group, summed."""
        out: Dict[str, int] = {}
        for group in self.groups:
            for name, qty in group.flattened().items():
                out[name] = out.get(name, 0) + qty
        return out

    def forbidden_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for group in self.groups:
            for item in group.forbidden_items:
                out[item.name] = out.get(item.name, 0) + 1
        return out

    def required_item_names(self) -> set:
        return set(self.required_counts())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.ticket_id,
            "shortcut": self.shortcut,
            "source": self.source,
            "required_hotdogs": self.required_hotdogs,
            "groups": [g.to_dict() for g in self.groups],
            "unverifiable": list(self.unverifiable),
        }

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_json(cls, raw: Dict[str, Any], known_items: Optional[Iterable[str]] = None) -> "TicketSpec":
        """Build from a ticket dict.

        Understands all three shapes already present in config/kds_mock.json
        and src.domain.schemas.Ticket:

            {"hotdog1": ["yellow mustard sauce", ...], "hotdog2": [...]}
            {"hotdog_specs": {"hotdog1": {"relish": 1, ...}}}
            {"line_items": [{"variant": ..., "count": 2, "items": {...}}]}
            {"expected_items": ["relish", "onions"]}          (ungrouped)

        An ungrouped ticket becomes a single group, so every caller downstream
        sees the same structure regardless of which shape came in.
        """
        groups: List[GroupSpec] = []
        known = set(known_items) if known_items is not None else None
        unverifiable: List[str] = []

        def mk_items(src: Any) -> List[ItemReq]:
            items: List[ItemReq] = []
            pairs: List[tuple] = []
            if isinstance(src, dict):
                pairs = list(src.items())
            elif isinstance(src, (list, tuple)):
                counted: Dict[str, int] = {}
                for entry in src:
                    key = str(entry)
                    counted[key] = counted.get(key, 0) + 1
                pairs = list(counted.items())
            for raw_name, qty in pairs:
                negated = False
                text = str(raw_name).strip()
                low = text.lower()
                # KDS renders exclusions as "NO ONIONS" / "W/O RELISH".
                for prefix in ("no ", "w/o ", "without "):
                    if low.startswith(prefix):
                        negated = True
                        text = text[len(prefix):]
                        break
                name = normalize_item_name(text)
                if not name:
                    continue
                if known is not None and name not in known:
                    unverifiable.append(name)
                    continue
                items.append(ItemReq(name=name, qty=max(1, int(qty)), negated=negated))
            return items

        # Shape 1/2: explicit per-hotdog specs.
        specs: Dict[str, Any] = {}
        if isinstance(raw.get("hotdog_specs"), dict):
            specs.update(raw["hotdog_specs"])
        for key, value in raw.items():
            if key.startswith("hotdog") and key not in ("total_hotdogs", "hotdog_specs"):
                if isinstance(value, (list, dict)):
                    specs[key] = value
        for group_id, value in specs.items():
            groups.append(GroupSpec(group_id=group_id, variant=group_id,
                                    quantity=1, items=mk_items(value)))

        # Shape 3: line items, skipping any the specs already expanded.
        for line in raw.get("line_items", []) or []:
            variant = line.get("variant") if isinstance(line, dict) else getattr(line, "variant", "")
            if variant in specs:
                continue
            items_src = line.get("items") if isinstance(line, dict) else getattr(line, "items", {})
            count = int(line.get("count", 1) if isinstance(line, dict) else getattr(line, "count", 1))
            groups.append(GroupSpec(group_id=str(variant), variant=str(variant),
                                    quantity=max(1, count), items=mk_items(items_src)))

        # Shape 4: a flat expected_items list, no grouping available.
        if not groups and raw.get("expected_items"):
            total = int(raw.get("total_hotdogs", 1) or 1)
            groups.append(GroupSpec(group_id="order", variant="", quantity=max(1, total),
                                    items=mk_items(raw["expected_items"])))

        # A ticket may declare more hotdogs than it details add-ons for; the
        # undetailed ones still have to exist physically.
        declared = int(raw.get("total_hotdogs", 0) or 0)
        covered = sum(g.quantity for g in groups)
        if declared > covered:
            groups.append(GroupSpec(group_id="plain", variant="plain",
                                    quantity=declared - covered, items=[]))

        return cls(
            ticket_id=str(raw.get("ticket_id", "unknown")),
            shortcut=str(raw.get("shortcut", "")),
            groups=groups,
            unverifiable=sorted(set(unverifiable)),
            source="json",
        )

    @classmethod
    def from_snapshot(cls, snapshot: Any, known_items: Optional[Iterable[str]] = None) -> "TicketSpec":
        """Build from an OCR `TicketSnapshot` without importing the OCR stack.

        Reads only the attributes `src.kds.schemas.TicketSnapshot` guarantees,
        so this module stays importable on a machine with no cv2.  Unknown
        shortcuts are carried through as groups with no items: we still require
        the hotdog to exist, but we assert nothing about its add-ons (RULE 4).
        """
        known = set(known_items) if known_items is not None else None
        groups: List[GroupSpec] = []
        unverifiable: List[str] = []

        for hotdog in getattr(snapshot, "hotdogs", []) or []:
            items: List[ItemReq] = []
            for addon in getattr(hotdog, "addons", []) or []:
                raw_name = getattr(addon, "ingredient", None) or getattr(addon, "key", "")
                name = normalize_item_name(str(raw_name))
                if not name:
                    continue
                if known is not None and name not in known:
                    unverifiable.append(name)
                    continue
                items.append(ItemReq(
                    name=name,
                    qty=max(1, int(getattr(addon, "quantity", 1) or 1)),
                    negated=bool(getattr(addon, "negation", False)),
                ))
            for raw_name in getattr(hotdog, "ingredients", []) or []:
                name = normalize_item_name(str(raw_name))
                if not name or any(i.name == name for i in items):
                    continue
                if known is not None and name not in known:
                    unverifiable.append(name)
                    continue
                items.append(ItemReq(name=name, qty=1))
            groups.append(GroupSpec(
                group_id=str(getattr(hotdog, "shortcut", "") or getattr(hotdog, "item", "group")),
                variant=str(getattr(hotdog, "item", "")),
                quantity=max(1, int(getattr(hotdog, "quantity", 1) or 1)),
                items=items,
            ))

        return cls(
            ticket_id=str(getattr(snapshot, "ticket_id", "unknown")),
            shortcut=str(getattr(snapshot, "order_type", "")),
            groups=groups,
            unverifiable=sorted(set(unverifiable)),
            source="ocr",
        )
