#!/usr/bin/env python3
"""Print the recipe for every KDS shortcut code.

    .venv/bin/python scripts/dump_recipes.py                  # everything
    .venv/bin/python scripts/dump_recipes.py --category hot_dog
    .venv/bin/python scripts/dump_recipes.py --out output/ITEM_RECIPES.txt
    .venv/bin/python scripts/dump_recipes.py --json output/item_recipes_flat.json

Generated from kds-ocr/reference/, which the project treats as the source of
truth -- item_recipes.json for the layers, item_registry.json for the name,
category and provenance. Generated rather than written by hand so it cannot
drift from the files the pipeline actually validates against.

A code with no documented recipe is listed as such and never guessed: the
registry marks items `confirmed`, `assumption` or `??`, and inventing layers
for an undocumented item would put a requirement on a ticket that no document
supports. Those appear in their own section at the end.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "kds-ocr" / "reference"

# Hot dogs first: they are what the pipeline verifies. The rest is context.
CATEGORY_ORDER = [
    "hot_dog", "burger", "pastrami", "sandwich", "bigfoot",
    "fries", "fryer_item", "dessert", "drink", "combo",
    "modifier", "retail", "misc", "channel", "prefix", "artifact", "unknown",
]

STATUS_MARK = {"confirmed": "", "assumption": "  [assumption]", "??": "  [UNVERIFIED]"}


def load():
    rec = json.loads((REF / "item_recipes.json").read_text())
    reg = json.loads((REF / "item_registry.json").read_text())
    meta = {"recipes": rec.pop("_meta", {}), "registry": reg.pop("_meta", {})}
    return rec, reg, meta


def fmt_qty(qty) -> str:
    """0.125 -> 1/8, 1.0 -> 1 -- the chart's own fractions, not decimals."""
    fractions = {0.125: "1/8", 0.25: "1/4", 0.333: "1/3", 0.5: "1/2",
                 0.666: "2/3", 0.75: "3/4"}
    if isinstance(qty, float):
        for v, s in fractions.items():
            if abs(qty - v) < 0.005:
                return s
        if qty.is_integer():
            return str(int(qty))
        return ("%g" % qty)
    return str(qty)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", help="only this category (e.g. hot_dog)")
    ap.add_argument("--out", help="also write the text to this file")
    ap.add_argument("--json", dest="json_out", help="also write a flat JSON mapping")
    args = ap.parse_args(argv)

    rec, reg, meta = load()
    lines: list[str] = []
    w = lines.append

    w("KDS SHORTCUT RECIPES")
    w("=" * 72)
    w("Generated from kds-ocr/reference/item_recipes.json and item_registry.json.")
    src = meta["recipes"].get("source")
    if src:
        w("")
        w("Source: " + src)
    note = meta["recipes"].get("note")
    if note:
        w("Note:   " + note)
    w("")
    w("Quantities are per ONE of that item. [assumption] / [UNVERIFIED] mark the")
    w("registry's own confidence in the item, not in the recipe.")
    w("")

    by_cat: dict[str, list[str]] = defaultdict(list)
    for code in rec:
        entry = reg.get(code) or {}
        by_cat[entry.get("category", "uncategorised")].append(code)

    order = [c for c in CATEGORY_ORDER if c in by_cat]
    order += sorted(c for c in by_cat if c not in CATEGORY_ORDER)
    if args.category:
        order = [c for c in order if c == args.category]

    total = 0
    for cat in order:
        codes = sorted(by_cat[cat])
        w("")
        w("-" * 72)
        w("%s   (%d)" % (cat.upper().replace("_", " "), len(codes)))
        w("-" * 72)
        for code in codes:
            entry = reg.get(code) or {}
            name = entry.get("name") or "(not in the registry)"
            mark = STATUS_MARK.get(entry.get("status", ""), "")
            w("")
            w("%-14s %s%s" % (code, name, mark))
            for layer in rec[code]:
                try:
                    ing, qty, unit = layer[0], layer[1], layer[2]
                except (IndexError, TypeError):
                    w("                 %s" % (layer,))
                    continue
                w("                 %-22s %5s %s" % (ing, fmt_qty(qty), unit))
            total += 1

    # Codes the registry knows but no document gives layers for.
    if not args.category:
        missing = defaultdict(list)
        for code, entry in reg.items():
            if code not in rec:
                missing[(entry or {}).get("category", "uncategorised")].append(code)
        if missing:
            w("")
            w("=" * 72)
            w("NO DOCUMENTED RECIPE   (%d codes)" % sum(len(v) for v in missing.values()))
            w("=" * 72)
            w("Reported as unexploded by the views, never guessed. Modifiers,")
            w("drinks and retail lines have no layers by nature; anything here")
            w("under a food category is a genuine gap in the reference.")
            for cat in sorted(missing):
                w("")
                w("%s:" % cat)
                w("   " + ", ".join(sorted(missing[cat])))

    w("")
    w("=" * 72)
    w("%d recipe(s) listed." % total)

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print("\n-> %s" % args.out)
    if args.json_out:
        flat = {
            code: {
                "name": (reg.get(code) or {}).get("name"),
                "category": (reg.get(code) or {}).get("category"),
                "status": (reg.get(code) or {}).get("status"),
                "ingredients": [
                    {"ingredient": l[0], "qty": l[1], "unit": l[2]}
                    for l in rec[code] if isinstance(l, list) and len(l) >= 3
                ],
            }
            for code in sorted(rec)
        }
        Path(args.json_out).write_text(json.dumps(flat, indent=2) + "\n")
        print("-> %s" % args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
