#!/usr/bin/env python3
"""Cut the WRONG orders out of a dashboard recording, and keep only those.

    python3 scripts/clip_wrong_orders.py                      # after a run
    python3 scripts/clip_wrong_orders.py --keep-full           # keep the source too

A run records the dashboard continuously, because starting and stopping a
browser per ticket would cost an Xorg and a Firefox launch each time and would
miss the seconds either side.  This trims that one file down afterwards: one
clip per order that was judged WRONG, and the full recording deleted unless
`--keep-full` says otherwise.

Correct orders are not kept.  Neither are orders that were never judged --
voided, or still open when the run ended -- because there is no verdict to
review.

Cutting is a stream copy (`-c copy`), so it is fast and re-encodes nothing.
That means cuts land on keyframes: with the recorder's 2-second keyframe
interval a clip can begin up to ~2 s early, which is padding in the right
direction for reviewing what went wrong.

The join between the journeys and the video depends on which recorder made it.
A dashboard recording is a screen capture in real time, so the WALL clock joins
them: `opened_wall`/`verdict_wall`, against the start time in the recording's
sidecar JSON.  An annotated-feed recording (RECORD_VIDEO) holds one frame per
frame the pipeline processed, so it runs on the video's own MEDIA time and
`opened_at`/`duration_s` join it directly, with no sidecar.  `--timebase` picks;
the default reads the sidecar's presence and is right either way.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Seconds of context kept either side of the order.
PRE_PAD_S = 5.0
POST_PAD_S = 5.0


def log(msg: str) -> None:
    print("[clip] " + msg, flush=True)


def load_journeys(path: Path) -> list:
    if not path.exists():
        log("no journeys at %s" % path)
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            log("skipping an unparseable journey line")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", help="the dashboard .mp4 (default: newest in output/recordings)")
    ap.add_argument("--journeys", default=str(ROOT / "output" / "ticket_journeys.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "output" / "wrong_orders"))
    ap.add_argument("--keep-full", action="store_true",
                    help="keep the full recording as well as the clips")
    ap.add_argument("--pre-pad", type=float, default=PRE_PAD_S)
    ap.add_argument("--post-pad", type=float, default=POST_PAD_S)
    ap.add_argument("--timebase", choices=("auto", "wall", "media"), default="auto",
                    help="which clock the recording runs on; auto picks wall "
                         "when a sidecar is present, media otherwise")
    args = ap.parse_args(argv)

    rec_dir = ROOT / "output" / "recordings"
    if args.recording:
        recording = Path(args.recording)
    else:
        recs = sorted(rec_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime)
        if not recs:
            log("no recording found in %s" % rec_dir)
            return 1
        recording = recs[-1]
    if not recording.exists():
        log("recording not found: %s" % recording)
        return 1

    # Which clock the recording runs on depends on which recorder made it.
    #
    # The dashboard recorder captures a browser window in real time, so the
    # file follows the WALL clock and its sidecar says when it started: the
    # journeys' `opened_wall`/`verdict_wall` are what line up with it.
    #
    # The annotated-feed recorder (RECORD_VIDEO) writes one frame per frame
    # the pipeline processed, idle frames included, so its own time IS the
    # production video's media time -- the clock the journey steps already
    # use.  `opened_at` and `duration_s` place those cuts directly, and no
    # sidecar is needed or written.  This is the path on a box that cannot
    # run the Xorg/Firefox/GStreamer recorder at all.
    sidecar = recording.with_suffix(".json")
    timebase = args.timebase
    if timebase == "auto":
        timebase = "wall" if sidecar.exists() else "media"
    rec_start = 0.0
    rec_len = 0.0
    if timebase == "wall":
        if not sidecar.exists():
            log("no sidecar next to %s -- cannot place the clips on the wall clock"
                % recording.name)
            return 1
        meta = json.loads(sidecar.read_text())
        import datetime as _dt
        rec_start = _dt.datetime.fromisoformat(meta["started"]).timestamp()
        rec_len = float(meta.get("wall_s") or 0.0)
    log("cutting %s on the %s clock" % (recording.name, timebase))

    journeys = load_journeys(Path(args.journeys))
    wrong = [j for j in journeys if j.get("correct") is False]
    skipped = [j for j in journeys if j.get("correct") is None]
    log("%d judged order(s): %d wrong, %d correct, %d never judged"
        % (len(journeys), len(wrong),
           sum(1 for j in journeys if j.get("correct") is True), len(skipped)))
    if not wrong:
        log("nothing went wrong, so there is nothing to keep")
        if not args.keep_full:
            recording.unlink()
            sidecar.unlink(missing_ok=True)   # the media-time path writes none
            log("removed %s" % recording.name)
        return 0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for j in wrong:
        ref = str(j.get("ticket_id") or "unknown").replace("/", "-")
        if timebase == "wall":
            opened = j.get("opened_wall") or 0.0
            ended = j.get("verdict_wall") or 0.0
            if not opened or not ended:
                log("%s has no wall-clock stamps (recorded before they were added);"
                    " skipping" % ref)
                continue
            start = max(0.0, opened - rec_start - args.pre_pad)
            end = ended - rec_start + args.post_pad
        else:
            # Media time: the journey's own clock, so the window is the order
            # itself with no recording offset to subtract.
            opened = float(j.get("opened_at") or 0.0)
            start = max(0.0, opened - args.pre_pad)
            end = opened + float(j.get("duration_s") or 0.0) + args.post_pad
        if rec_len and start >= rec_len:
            log("%s happened after the recording stopped; skipping" % ref)
            continue
        dur = max(1.0, end - start)
        dest = out_dir / ("%s_%s.mp4" % (recording.stem, ref))
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-ss", "%.3f" % start, "-i", str(recording),
               "-t", "%.3f" % dur, "-c", "copy", str(dest)]
        try:
            subprocess.run(cmd, check=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            log("ffmpeg failed for %s: %s" % (ref, exc))
            continue
        size = dest.stat().st_size if dest.exists() else 0
        log("%s -> %s (%.0fs, %.1f MB) %s" % (
            ref, dest.name, dur, size / 1048576.0,
            (j.get("message") or "")[:70]))
        made += 1

    if made and not args.keep_full:
        recording.unlink()
        sidecar.unlink(missing_ok=True)       # the media-time path writes none
        log("removed the full recording; kept %d wrong-order clip(s)" % made)
    elif not made:
        log("no clips were cut, so the full recording is kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
