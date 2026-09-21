#!/usr/bin/env python3
"""Per-ticket report: what the ticket asked for, what the system saw, and the tally.

    .venv/bin/python scripts/ticket_report.py
    .venv/bin/python scripts/ticket_report.py --json output/ticket_report.json

Reads output/ticket_journeys.jsonl -- one record per judged ticket, written by
the run itself -- and prints a line per ingredient per ticket:

    REQUIRED   what the KDS ticket asked for
    ADDED      what the pipeline actually observed going on
    STATUS     OK        added >= required
               MISSING   required and never seen   <- the only thing that fails
               EXTRA     seen but never asked for  <- reported, never a fault

The verdict rule is BatchOrderValidator's, not a second opinion invented here:
an order is wrong when a required ingredient never arrived, and nothing added on
top of the requirement fails it.  Accuracy is computed over JUDGED tickets only;
tickets still open when the run ended have no verdict and are counted separately
rather than silently scored as correct.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> list:
    if not path.exists():
        raise SystemExit(f"no journeys at {path} -- has a run finished?")
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _as_counter(v) -> Counter:
    if isinstance(v, dict):
        return Counter({str(a): int(b) for a, b in v.items()})
    if isinstance(v, list):
        out = Counter()
        for x in v:
            if isinstance(x, dict):
                out[str(x.get("item"))] += int(x.get("count", 1) or 1)
            else:
                out[str(x)] += 1
        return out
    return Counter()


def required_and_added(journey: dict) -> tuple:
    """Reconstruct (required, added) from what a journey actually stores.

    A journey records `observed`, `missing` and `extras`, and its `requirement`
    field is frequently empty -- an ORG PLAN ticket, for instance, carries its
    whole requirement in the hotdog count, so there is no ingredient dict to
    write.  So the requirement is rebuilt from the two sides that ARE recorded:

        required = missing            (asked for, never seen)
                 + observed - extras  (asked for, and seen)

    `extras` is by definition "observed but not required", so subtracting it
    from `observed` leaves exactly the observed items that were required.  This
    is a reconstruction, not a second judgement: the verdict still comes from
    the `correct` flag the run wrote.
    """
    observed = _as_counter(journey.get("observed"))
    missing = _as_counter(journey.get("missing"))
    extras = _as_counter(journey.get("extras"))

    declared = _as_counter(journey.get("requirement"))
    if declared:
        return declared, observed

    required = Counter(missing)
    for item, n in observed.items():
        if item not in extras:
            required[item] += n
    return required, observed


def hms(seconds) -> str:
    if seconds is None:
        return "-"
    s = max(0, int(round(float(seconds))))
    return "%02d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


def recording_clock(journeys_path: Path):
    """When the dashboard capture for this run began, as a unix timestamp.

    `opened_at` in a journey is MEDIA time in the source video, which is not
    where the ticket sits in recording.mp4 -- the capture starts when the run
    starts, and a recorded run processes an hour of video over two hours of wall
    clock. So the seek offset has to come from the wall stamps and the capture's
    own start: `opened_wall - capture_start`.

    Read from recording.json beside the journeys (the per-run layout), falling
    back to the run directory's timestamp when the recorder never wrote a
    sidecar -- flagged approximate, because the recorder starts a few seconds
    after the directory is stamped.
    """
    side = journeys_path.parent / "recording.json"
    if side.exists():
        try:
            meta = json.loads(side.read_text())
            return dt.datetime.fromisoformat(meta["started"]).timestamp(), False
        except (ValueError, KeyError):
            pass
    # output/runs/20260917_070247 -> 2026-09-17 07:02:47
    name = journeys_path.parent.name
    try:
        return dt.datetime.strptime(name[-15:], "%Y%m%d_%H%M%S").timestamp(), True
    except ValueError:
        return None, False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journeys", default=str(ROOT / "output" / "ticket_journeys.jsonl"))
    ap.add_argument("--json", dest="json_out", default=None,
                    help="also write the whole report as JSON here")
    ap.add_argument("--lead-in-s", type=float, default=5.0,
                    help="seek this many seconds early so the ticket is already "
                         "on screen when playback lands")
    args = ap.parse_args(argv)

    jpath = Path(args.journeys)
    journeys = load(jpath)
    rec_start, rec_approx = recording_clock(jpath)
    report = []
    judged = wrong = correct = unjudged = 0
    tot_req = tot_added = tot_missing = tot_extra = 0

    for j in journeys:
        ticket = str(j.get("ticket_id") or "unknown")
        required, added = required_and_added(j)
        # Hotdogs are counted separately from ingredients everywhere else in
        # this pipeline, but they are the commonest miss, so show them as a row.
        hd_missing = _as_counter(j.get("missing")).get("hot-dog", 0)
        if hd_missing:
            required["hot-dog"] = required.get("hot-dog", 0) or hd_missing
        correct_flag = j.get("correct")

        rows = []
        for item in sorted(set(required) | set(added)):
            need = required.get(item, 0)
            got = added.get(item, 0)
            if need and got >= need:
                status = "OK"
            elif need and got == 0:
                status = "MISSING"
            elif need and got < need:
                # Presence-tested, same as the validator: seen at all satisfies.
                status = "OK"
            else:
                status = "EXTRA"
            rows.append({"item": item, "required": need, "added": got, "status": status})
            tot_req += need
            tot_added += got
            tot_missing += 1 if status == "MISSING" else 0
            tot_extra += 1 if status == "EXTRA" else 0

        if correct_flag is None:
            unjudged += 1
            verdict = "NOT JUDGED (open when the run ended)"
        else:
            judged += 1
            correct += 1 if correct_flag else 0
            wrong += 0 if correct_flag else 1
            verdict = "CORRECT" if correct_flag else "WRONG"

        # Where this ticket sits in recording.mp4 -- NOT opened_at, which is
        # media time in the source video and unrelated to the capture.
        seek_s = ends_s = None
        if rec_start is not None and j.get("opened_wall"):
            seek_s = max(0.0, float(j["opened_wall"]) - rec_start - args.lead_in_s)
            if j.get("verdict_wall"):
                ends_s = max(0.0, float(j["verdict_wall"]) - rec_start)

        entry = {
            "ticket_id": ticket,
            "verdict": verdict,
            "correct": correct_flag,
            "hotdogs_required": j.get("required_hotdogs"),
            "hotdogs_observed": j.get("observed_hotdogs"),
            # media time in the SOURCE video
            "opened_at": j.get("opened_at"),
            "duration_s": j.get("duration_s"),
            # offsets into recording.mp4, which is what you seek to
            "recording_seek_s": round(seek_s, 1) if seek_s is not None else None,
            "recording_seek": hms(seek_s) if seek_s is not None else None,
            "recording_ends_s": round(ends_s, 1) if ends_s is not None else None,
            "recording_ends": hms(ends_s) if ends_s is not None else None,
            "recording_clock_approx": rec_approx if seek_s is not None else None,
            "message": j.get("message"),
            "items": rows,
        }
        report.append(entry)

        print("=" * 72)
        print(f"TICKET {ticket}    {verdict}")
        if seek_s is not None:
            print(f"  recording.mp4  seek {hms(seek_s)} -> {hms(ends_s)}"
                  f"{'   (clock approximate)' if rec_approx else ''}")
        if j.get("duration_s") is not None:
            print(f"  duration {float(j['duration_s']):.1f}s")
        if entry["hotdogs_required"] is not None:
            print(f"  hotdogs  required {entry['hotdogs_required']}  "
                  f"observed {entry['hotdogs_observed']}")
        if not rows:
            print("  (no ingredient detail recorded for this ticket)")
        else:
            print(f"  {'ITEM':<28}{'REQUIRED':>9}{'ADDED':>7}   STATUS")
            for r in rows:
                print(f"  {r['item']:<28}{r['required']:>9}{r['added']:>7}   {r['status']}")
        if j.get("message"):
            print(f"  -> {j['message']}")

    total = len(journeys)
    acc = (correct / judged * 100.0) if judged else 0.0
    print("=" * 72)
    print("SUMMARY")
    print(f"  tickets seen        {total}")
    print(f"  judged              {judged}   (correct {correct}, wrong {wrong})")
    print(f"  not judged          {unjudged}   (open when the run ended)")
    print(f"  accuracy            {acc:.1f}%   (over judged tickets only)")
    print(f"  required (units)    {tot_req}")
    print(f"  added    (units)    {tot_added}")
    print(f"  missing  (lines)    {tot_missing}   <- the only thing that fails an order")
    print(f"  extra    (lines)    {tot_extra}   <- reported, never a fault")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "summary": {
                "tickets": total, "judged": judged, "correct": correct,
                "wrong": wrong, "not_judged": unjudged,
                "accuracy_pct": round(acc, 2),
                "required_units": tot_req, "added_units": tot_added,
                "missing_lines": tot_missing, "extra_lines": tot_extra,
            },
            "tickets": report,
        }, indent=2))
        print(f"\nJSON -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
