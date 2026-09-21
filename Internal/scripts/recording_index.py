#!/usr/bin/env python3
"""Where each ticket starts inside the dashboard recording.

    .venv/bin/python scripts/recording_index.py
    .venv/bin/python scripts/recording_index.py --recording output/recordings/dashboard_X.mp4

Writes `<recording>_index.json` and prints the same thing as a seek table, so a
ticket can be found in a two-hour capture without scrubbing for it:

    TICKET     VERDICT   SEEK TO   ENDS AT   LENGTH
    CHK-251    WRONG     00:04:12  00:06:41   2m29s

The offsets are positions IN THE VIDEO, already converted from the wall clock.
A journey stores `opened_wall` / `verdict_wall` as unix timestamps and the
recording's sidecar stores when the capture began, so the seek point is
`opened_wall - capture_start`, minus a few seconds of lead-in so the ticket is
already on screen when playback lands.

Same arithmetic `clip_wrong_orders.py` uses to cut, which is why an index entry
and a clip of the same ticket agree. This does not re-encode anything -- it is
the map, not the cuts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAD_IN_S = 5.0


def hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return "%02d:%02d:%02d" % (seconds // 3600, (seconds % 3600) // 60, seconds % 60)


def human(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return "%dm%02ds" % (seconds // 60, seconds % 60) if seconds >= 60 else "%ds" % seconds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", help="default: newest .mp4 in output/recordings")
    ap.add_argument("--journeys", default=str(ROOT / "output" / "ticket_journeys.jsonl"))
    ap.add_argument("--out", help="default: <recording>_index.json")
    args = ap.parse_args(argv)

    rec_dir = ROOT / "output" / "recordings"
    if args.recording:
        rec = Path(args.recording)
    else:
        mp4s = sorted(rec_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not mp4s:
            print("no recording in %s" % rec_dir)
            return 1
        rec = mp4s[-1]

    sidecar = rec.with_suffix(".json")
    if not sidecar.exists():
        print("no sidecar beside %s -- cannot place tickets on the recording's "
              "clock without knowing when the capture started" % rec.name)
        return 1
    meta = json.loads(sidecar.read_text())
    start = dt.datetime.fromisoformat(meta["started"]).timestamp()
    length = float(meta.get("wall_s") or 0.0)

    journeys = []
    jpath = Path(args.journeys)
    if jpath.exists():
        for line in jpath.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    journeys.append(json.loads(line))
                except ValueError:
                    pass

    rows = []
    for j in journeys:
        opened = j.get("opened_wall")
        ended = j.get("verdict_wall")
        if not opened:
            continue
        seek = max(0.0, float(opened) - start - LEAD_IN_S)
        end = (float(ended) - start) if ended else None
        if length and seek >= length:
            continue          # happened after the capture stopped
        correct = j.get("correct")
        rows.append({
            "ticket_id": j.get("ticket_id"),
            "verdict": "CORRECT" if correct is True else ("WRONG" if correct is False else "NOT JUDGED"),
            "correct": correct,
            "seek_s": round(seek, 1),
            "seek": hms(seek),
            "ends_s": round(end, 1) if end is not None else None,
            "ends": hms(end) if end is not None else None,
            "length_s": round(end - seek, 1) if end is not None else None,
            "message": j.get("message"),
        })
    rows.sort(key=lambda r: r["seek_s"])

    out = Path(args.out) if args.out else rec.with_name(rec.stem + "_index.json")
    out.write_text(json.dumps({
        "recording": rec.name,
        "started": meta["started"],
        "wall_s": length,
        "duration": hms(length),
        "lead_in_s": LEAD_IN_S,
        "tickets": rows,
    }, indent=2))

    print("RECORDING  %s   (%s long, starts %s)" % (rec.name, hms(length), meta["started"]))
    print()
    print("%-11s%-12s%-10s%-10s%-8s" % ("TICKET", "VERDICT", "SEEK TO", "ENDS AT", "LENGTH"))
    print("-" * 74)
    for r in rows:
        print("%-11s%-12s%-10s%-10s%-8s" % (
            r["ticket_id"], r["verdict"], r["seek"], r["ends"] or "-",
            human(r["length_s"]) if r["length_s"] else "-"))
    if not rows:
        print("(no ticket carried wall-clock stamps)")
    print()
    print("index -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
