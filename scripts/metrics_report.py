"""Measured metrics for a completed run.

    python scripts/metrics_report.py

Reads what the pipeline already writes -- ``output/orders.jsonl`` for order
outcomes and ``output/kds_timeline.jsonl`` for KDS reading -- and reports the
numbers needed to judge a change to ingredient attribution or ticket parsing.

The headline number is the credited/required ratio per ingredient.  An
ingredient credited many times more often than the ticket asks for means the
attribution path is firing repeatedly on one application; the order is then
marked WRONG for an "extra item" the kitchen never added.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _bar(ratio, width=18):
    """A crude visual so an outlier is obvious at a glance."""
    if ratio <= 0:
        return ""
    filled = min(width, max(1, int(round(ratio / 2.0))))
    return "#" * filled


def report_orders(orders):
    print("=" * 68)
    print("ORDERS")
    print("=" * 68)
    if not orders:
        print("  no orders recorded yet")
        return

    passed = sum(1 for o in orders if o.get("passed"))
    total = len(orders)
    print("  total    : %d" % total)
    print("  passed   : %d" % passed)
    print("  failed   : %d" % (total - passed))
    print("  accuracy : %.1f%%" % (100.0 * passed / total))

    reasons = collections.Counter(o.get("result", "?") for o in orders if not o.get("passed"))
    if reasons:
        print("\n  failure reasons")
        for reason, n in reasons.most_common():
            print("    %-46s x%d" % (reason[:46], n))


def report_ingredients(orders):
    print()
    print("=" * 68)
    print("INGREDIENT ATTRIBUTION   (credited vs what the tickets asked for)")
    print("=" * 68)
    required = collections.Counter()
    credited = collections.Counter()
    for order in orders:
        for item in order.get("expected_items", []):
            required[item] += 1
        for item, count in (order.get("picked_counts") or {}).items():
            credited[item] += count

    names = sorted(set(required) | set(credited))
    if not names:
        print("  nothing recorded yet")
        return

    print("  %-24s %8s %8s %7s  %s" % ("ingredient", "required", "credited", "ratio", ""))
    print("  " + "-" * 64)
    over = under = 0
    for name in names:
        req = required.get(name, 0)
        got = credited.get(name, 0)
        if req == 0:
            ratio_text = "  n/a"
            flag = "NOT ON ANY TICKET"
            over += 1 if got else 0
        else:
            ratio = got / float(req)
            ratio_text = "%5.1fx" % ratio
            if ratio > 1.5:
                flag = "OVER  " + _bar(ratio)
                over += 1
            elif ratio < 0.5:
                flag = "UNDER"
                under += 1
            else:
                flag = "ok"
        print("  %-24s %8d %8d %7s  %s" % (name[:24], req, got, ratio_text, flag))

    print()
    print("  over-credited  : %d ingredient(s)" % over)
    print("  under-credited : %d ingredient(s)" % under)
    if over:
        print("\n  An ingredient credited far more often than required means one")
        print("  application is being counted repeatedly.  The order is then failed")
        print("  for an extra item the kitchen never added.")


def report_kds(events):
    print()
    print("=" * 68)
    print("KDS READING")
    print("=" * 68)
    if not events:
        print("  no KDS timeline (mock mode, or the KDS reader did not run)")
        return

    kinds = collections.Counter(e.get("kind", "?") for e in events)
    for kind in ("ticket_detected", "payment", "ticket_created", "ticket_updated",
                 "disappeared", "unknown_shortcut", "verdict"):
        if kind in kinds:
            print("  %-18s %d" % (kind, kinds[kind]))

    unknown = collections.Counter()
    for event in events:
        if event.get("kind") == "unknown_shortcut":
            unknown[event.get("message", "")[:52]] += 1
    if unknown:
        print("\n  unmapped shortcuts (each one fails its whole ticket)")
        for text, n in unknown.most_common(10):
            print("    %-52s x%d" % (text, n))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", default=os.path.join(ROOT, "output", "orders.jsonl"))
    parser.add_argument("--timeline", default=os.path.join(ROOT, "output", "kds_timeline.jsonl"))
    args = parser.parse_args(argv)

    orders = _load_jsonl(args.orders)
    events = _load_jsonl(args.timeline)

    print()
    print("METRICS  --  %s" % os.path.relpath(args.orders, ROOT))
    report_orders(orders)
    report_ingredients(orders)
    report_kds(events)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
