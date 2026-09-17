#!/usr/bin/env python3
"""Move older runs into output/runs/<run_id>/, the layout new runs already use.

    .venv/bin/python scripts/migrate_runs.py --dry-run
    .venv/bin/python scripts/migrate_runs.py

Runs used to scatter their output: the capture in output/recordings/, the clips
in output/wrong_orders/ (renamed by hand between runs so the next run would not
overwrite them), and a single output/ticket_report.json that every run
overwrote.  Telling two runs apart meant reading timestamps out of filenames.

Each run becomes:

    output/runs/<run_id>/
        recording.mp4          the dashboard capture
        recording.json         its sidecar (capture start, wall_s, fps)
        index.json             seek offsets per ticket
        wrong_orders/          one clip per wrong ticket
        ticket_report.json     required vs added, and accuracy
        ticket_journeys.jsonl  per-ticket steps and verdicts

Clips are matched to their capture by the prefix they were named with, so a clip
can only land beside the recording it was cut from.  Files are MOVED, never
copied -- these are hundreds of megabytes each, and a half-migrated tree with
two copies of everything would be worse than the scattering it replaces.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
RUNS = OUT / "runs"

# Journeys/reports were rotated by hand as runs went by; this records which
# belongs to which capture. Anything not listed is left where it is.
KNOWN = {
    "dashboard_20260916_201702":     {"journeys": "ticket_journeys_prev_221441.jsonl"},
    "dashboard_run_20260916_221456": {"journeys": "ticket_journeys_1h_20260916.jsonl"},
    "dashboard_20260917_064726":     {},
    "dashboard_20260917_070247":     {"journeys": "ticket_journeys_10min.jsonl",
                                      "report": "ticket_report_10min.json"},
    "dashboard_20260917_102335":     {"journeys": "ticket_journeys.jsonl",
                                      "report": "ticket_report.json"},
}


def run_id_for(stem: str) -> str:
    """`dashboard_run_20260916_221456` -> `20260916_221456`."""
    parts = stem.split("_")
    return "_".join(parts[-2:]) if len(parts) >= 2 else stem


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    rec_dir = OUT / "recordings"
    if not rec_dir.is_dir():
        print("no output/recordings to migrate")
        return 0

    clip_dirs = [d for d in (OUT / "wrong_orders", OUT / "wrong_orders_1h",
                             OUT / "wrong_orders_prev") if d.is_dir()]

    moves: list[tuple[Path, Path]] = []
    for mp4 in sorted(rec_dir.glob("*.mp4")):
        stem = mp4.stem
        rid = run_id_for(stem)
        dest = RUNS / rid
        moves.append((mp4, dest / "recording.mp4"))

        side = mp4.with_suffix(".json")
        if side.exists():
            moves.append((side, dest / "recording.json"))
        idx = mp4.with_name(stem + "_index.json")
        if idx.exists():
            moves.append((idx, dest / "index.json"))

        # A clip is named "<capture stem>_<ticket>.mp4", so the prefix is the
        # only thing that may decide which run it belongs to.
        for cd in clip_dirs:
            for clip in sorted(cd.glob(stem + "_*.mp4")):
                moves.append((clip, dest / "wrong_orders" / clip.name))

        meta = KNOWN.get(stem, {})
        for key, name in (("journeys", "ticket_journeys.jsonl"),
                          ("report", "ticket_report.json")):
            src_name = meta.get(key)
            if src_name and (OUT / src_name).exists():
                moves.append((OUT / src_name, dest / name))

    if not moves:
        print("nothing to migrate")
        return 0

    by_run: dict[str, list] = {}
    for src, dst in moves:
        by_run.setdefault(dst.parent.name if dst.parent.name != "wrong_orders"
                          else dst.parent.parent.name, []).append((src, dst))

    for rid in sorted(by_run):
        total = sum((s.stat().st_size for s, _ in by_run[rid] if s.exists()), 0)
        print(f"\noutput/runs/{rid}/   ({total/1e6:.0f} MB, {len(by_run[rid])} files)")
        for src, dst in by_run[rid]:
            rel = dst.relative_to(RUNS / rid)
            print(f"    {src.relative_to(OUT)}  ->  {rel}")

    if args.dry_run:
        print("\n--dry-run: nothing moved")
        return 0

    for src, dst in moves:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))

    # Only remove the old clip directories once they are actually empty: a
    # leftover file means a clip was not matched to a run, and silently
    # deleting it would lose the one copy that exists.
    for cd in clip_dirs:
        remaining = list(cd.iterdir()) if cd.is_dir() else []
        if not remaining:
            cd.rmdir()
        else:
            print(f"\nleft in place ({len(remaining)} unmatched): {cd.relative_to(OUT)}")
    if (OUT / "recordings").is_dir() and not list((OUT / "recordings").iterdir()):
        (OUT / "recordings").rmdir()

    print(f"\nmigrated {len(moves)} file(s) into output/runs/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
