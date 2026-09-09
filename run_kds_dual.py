"""Run the dual-input pipeline: a KDS screen video plus a production video.

    python run_kds_dual.py                          # uses the defaults below
    python run_kds_dual.py --kds <a.mkv> --production <b.mkv>

Both inputs run on their own capture thread and stamp events with the same
monotonic clock, so this works identically for files and for live RTSP cameras.
A known capture start difference is corrected with ``sync.kds_offset_s`` in
``config/kds_visual.yaml`` (or ``--kds-offset``).

The live dashboard is served at http://localhost:8000.

Every ticket is recorded (production feed beside the KDS feed) from the moment
it is created until its verdict.  The clip is KEPT only if the order turns out
WRONG -- correct orders are recorded and then deleted -- so ``output/failures/``
only ever holds orders worth reviewing.

While no ticket is on the KDS there is nothing to validate, so detection drops
to one frame in N (``--idle-detect-stride``) and the tracking work is skipped.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Default inputs.
#
# Repository convention:
#     videos/KDS_Feed/  -> the KDS screen recording (ticket display)
#     videos/KDS/       -> the matching kitchen camera (food assembly)
#
# The two files below are a genuinely time-aligned pair: both cover
# 2026-09-06 11:00-12:00 PDT at the same store.  Running this script with no
# arguments uses them, so the normal case is just `python run_kds_dual.py`.
#
# If a folder holds exactly one video, that file is used instead -- so dropping
# a new recording in keeps working without editing this file.
# ---------------------------------------------------------------------------

KDS_DIR = ROOT / "videos" / "KDS_Feed"
PRODUCTION_DIR = ROOT / "videos" / "KDS"

DEFAULT_KDS = (
    KDS_DIR / "Wienerschnitzel_Sacramento_CA_95818__kds__2026_09_06_11_to_12_PDT.mkv"
)
DEFAULT_PRODUCTION = (
    PRODUCTION_DIR
    / "Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"
)

VIDEO_SUFFIXES = (".mkv", ".mp4", ".avi", ".mov", ".m4v")


def resolve_default(named: Path, folder: Path, label: str) -> Path | None:
    """Pick the default video for one input.

    Prefers the named file; otherwise falls back to the only video in the
    folder.  Returns ``None`` when neither exists, so the caller can print a
    useful message instead of failing on a missing path.
    """
    if named.exists():
        return named
    if folder.is_dir():
        videos = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
        )
        if len(videos) == 1:
            return videos[0]
        if len(videos) > 1:
            print(
                "note: %s holds %d videos; using %s (pass --%s to choose another)"
                % (folder.name, len(videos), videos[0].name, label)
            )
            return videos[0]
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the KDS + production dual-video order accuracy pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="With no arguments, uses the paired recordings in "
               "videos/KDS_Feed/ and videos/KDS/.",
    )
    parser.add_argument(
        "--kds",
        default=None,
        help="KDS screen video file or RTSP URL (the ticket display). "
             "Defaults to the recording in videos/KDS_Feed/.",
    )
    parser.add_argument(
        "--production",
        default=None,
        help="Production/kitchen video file or RTSP URL (the food assembly area). "
             "Defaults to the recording in videos/KDS/.",
    )
    parser.add_argument(
        "--kds-offset",
        type=float,
        default=None,
        help="Seconds to add to KDS timestamps before comparing with production "
             "events. Positive means the KDS stream started later.",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace file playback at wall-clock speed (as a live camera would).",
    )
    parser.add_argument(
        "--log-level", default="INFO", help="Python log level (default INFO)."
    )
    parser.add_argument(
        "--no-record-failures",
        action="store_true",
        help="Do not record dashboard clips of failed orders. By default every "
             "ticket is recorded and the clip is kept only if the order is WRONG "
             "(correct orders are recorded then deleted).",
    )
    parser.add_argument(
        "--idle-detect-stride",
        type=int,
        default=None,
        metavar="N",
        help="While no ticket is on the KDS, run detection on only 1 frame in N "
             "(0 = pause detection entirely). Default 5.",
    )
    parser.add_argument(
        "--keep-history",
        action="store_true",
        help="carry the previous run's order history into this one "
             "(default: archive it so the dashboard starts empty)",
    )
    parser.add_argument(
        "--exit-on-end",
        action="store_true",
        help="Exit once the production video finishes.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.kds is None:
        resolved = resolve_default(DEFAULT_KDS, KDS_DIR, "kds")
        if resolved is None:
            print(
                "error: no KDS video given and none found in %s\n"
                "       pass --kds <file|rtsp url>" % KDS_DIR,
                file=sys.stderr,
            )
            return 2
        args.kds = str(resolved)

    if args.production is None:
        resolved = resolve_default(DEFAULT_PRODUCTION, PRODUCTION_DIR, "production")
        if resolved is None:
            print(
                "error: no production video given and none found in %s\n"
                "       pass --production <file|rtsp url>" % PRODUCTION_DIR,
                file=sys.stderr,
            )
            return 2
        args.production = str(resolved)

    for label, source in (("KDS", args.kds), ("production", args.production)):
        # RTSP/HTTP sources are passed through untouched.
        if "://" not in str(source) and not Path(source).exists():
            print("error: %s source not found: %s" % (label, source), file=sys.stderr)
            return 2

    if args.kds_offset is not None:
        _write_offset(args.kds_offset)

    env = os.environ.copy()
    env["KDS_MODE"] = "video"
    env["KDS_SOURCE"] = str(args.kds)
    env["VIDEO_SOURCE"] = str(args.production)
    env["LOG_LEVEL"] = args.log_level
    env["REALTIME"] = "true" if args.realtime else "false"
    env["EXIT_ON_END"] = "true" if args.exit_on_end else "false"
    # Each run starts with an empty board.  The previous run's order history is
    # archived rather than deleted, under output/history/.
    env["FRESH_START"] = "0" if args.keep_history else "1"
    if args.no_record_failures:
        env["RECORD_FAILURES"] = "false"
    if args.idle_detect_stride is not None:
        env["IDLE_DETECT_STRIDE"] = str(args.idle_detect_stride)

    print("KDS input        : %s" % args.kds)
    print("Production input : %s" % args.production)
    print("Dashboard        : http://localhost:8000")
    if not args.no_record_failures:
        print("Failure clips    : output/failures/  (kept only for WRONG orders)")
    if args.idle_detect_stride is not None:
        print("Idle detection   : 1 frame in %d while the KDS is empty"
              % args.idle_detect_stride)
    print()

    try:
        return subprocess.call(
            [sys.executable, "-u", "-m", "src.main"], cwd=str(ROOT), env=env
        )
    except KeyboardInterrupt:
        return 130


def _write_offset(offset: float) -> None:
    """Persist --kds-offset into config/kds_visual.yaml."""
    import yaml

    path = ROOT / "config" / "kds_visual.yaml"
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("sync", {})["kds_offset_s"] = float(offset)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    print("sync.kds_offset_s set to %.3fs" % offset)


if __name__ == "__main__":
    raise SystemExit(main())
