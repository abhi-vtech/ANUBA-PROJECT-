"""
run_full_video.py
─────────────────
Runs the main video processing pipeline against a full 1-hour video (or custom video)
and produces a structured hotdog-by-hotdog JSON timeline with:
  • hotdog_id and track_id
  • start timestamp (when hotdog first appeared / started)
  • items added (with ingredient name and timestamp for each addition)
  • end timestamp (when wrapping was detected and completed / marked done)
  • wrapping lifecycle details (dwell start, closing, done)
  • total duration

Usage:
──────
    # Run default 1-hour video (auto-detects 1-hour video in videos/)
    python run_full_video.py

    # Run specific 1-hour video file
    python run_full_video.py --video videos/Wienerschnitzel_Sacramento_CA_95818__camA__2026_07_04_12_to_13_PDT.mkv

    # Specify custom output path
    python run_full_video.py --output output/hotdog_summary.json --timeline output/hotdog_timeline.json

Output:
───────
  • Live dashboard at http://localhost:8000  (while running)
  • output/hotdog_summary.json              (full pipeline summary)
  • output/hotdog_timeline.json             (per-hotdog timeline JSON)
  • Formatted plain-text terminal report
"""

import sys
import os
import subprocess
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DEFAULT_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "hotdog_summary.json")
DEFAULT_TIMELINE_PATH = os.path.join(OUTPUT_DIR, "hotdog_timeline.json")

# Candidate 1-hour video filenames in order of preference
ONE_HOUR_CANDIDATES = [
    
    "Wienerschnitzel_Sacramento_CA_95818__camB__2026_07_04_12_to_13_PDT.mkv",
    
]


def _format_time_str(ts: Optional[float]) -> Optional[str]:
    """Format seconds timestamp to HH:MM:SS."""
    if ts is None:
        return None
    if ts < 86400:
        m, s = divmod(int(ts), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return time.strftime("%H:%M:%S", time.localtime(ts))


def _python_executable() -> str:
    """Prefer workspace virtual environment if present."""
    candidates = []
    if os.name == "nt":
        candidates.append(os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe"))
    else:
        candidates.append(os.path.join(BASE_DIR, ".venv", "bin", "python"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return sys.executable


def find_default_video() -> Optional[str]:
    """Find the best 1-hour video available in videos/ directory."""
    if not os.path.exists(VIDEOS_DIR):
        return None

    # Check preferred candidates first
    for cand in ONE_HOUR_CANDIDATES:
        p = os.path.join(VIDEOS_DIR, cand)
        if os.path.exists(p):
            return p

    # Fallback to any .mkv or .mp4 file in videos/
    for f in Path(VIDEOS_DIR).iterdir():
        if f.suffix.lower() in [".mkv", ".mp4", ".avi"]:
            return str(f)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline on full 1-hour video and export structured hotdog JSON timeline."
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="Path to video file (defaults to 1-hour video in videos/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_SUMMARY_PATH,
        help=f"Path to output hotdog summary JSON (default: {DEFAULT_SUMMARY_PATH})",
    )
    parser.add_argument(
        "--timeline",
        "-t",
        type=str,
        default=DEFAULT_TIMELINE_PATH,
        help=f"Path to output hotdog timeline JSON (default: {DEFAULT_TIMELINE_PATH})",
    )
    args = parser.parse_args()

    # Determine video path
    video_path = args.video
    if not video_path:
        video_path = os.environ.get("VIDEO_SOURCE")

    if not video_path:
        video_path = find_default_video()

    if not video_path or not os.path.exists(video_path):
        # Resolve relative to BASE_DIR if needed
        if video_path and os.path.exists(os.path.join(BASE_DIR, video_path)):
            video_path = os.path.join(BASE_DIR, video_path)
        else:
            print(f"[ERROR] Video file not found: {video_path}")
            print(f"Available videos in {VIDEOS_DIR}:")
            if os.path.exists(VIDEOS_DIR):
                for f in Path(VIDEOS_DIR).iterdir():
                    print(f"  {f.name} ({round(f.stat().st_size / (1024*1024), 1)} MB)")
            sys.exit(1)

    video_path = os.path.abspath(video_path)
    python_executable = _python_executable()

    summary_path = os.path.abspath(args.output)
    timeline_path = os.path.abspath(args.timeline)

    file_size_mb = round(os.path.getsize(video_path) / (1024 * 1024), 1)

    print("=" * 70)
    print("  HOTDOG ACCURACY TRACKER — FULL VIDEO RUN")
    print("=" * 70)
    print(f"  Video Path       : {video_path}")
    print(f"  Video Filename   : {os.path.basename(video_path)} ({file_size_mb} MB)")
    print(f"  Python Runtime   : {python_executable}")
    print(f"  Full Summary Out : {summary_path}")
    print(f"  Timeline JSON Out: {timeline_path}")
    print(f"  Live Dashboard   : http://localhost:8000")
    print("=" * 70)
    print()

    env = os.environ.copy()
    env["VIDEO_SOURCE"] = video_path
    env["EXIT_ON_END"] = "true"
    env["LOG_LEVEL"] = "INFO"

    # Launch main pipeline as subprocess
    process = subprocess.Popen(
        [python_executable, "-u", "-m", "src.main"],
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    total_detections_sum: Dict[str, int] = {}
    final_stats: Dict[str, Any] = {}
    hotdog_summary_snapshot: Dict[str, Any] = {}
    metrics_count = 0
    start_wall_time = time.time()

    print("[INFO] Pipeline started. Processing video frames...")
    print("[INFO] Press Ctrl+C to stop early and export current progress.\n")
    sys.stdout.flush()

    try:
        for line in process.stdout:
            line_s = line.rstrip()
            if not line_s:
                continue

            parsed_as_json = False
            if line_s.startswith("{"):
                try:
                    data = json.loads(line_s)
                    parsed_as_json = True

                    if data.get("event") == "metrics":
                        final_stats = data
                        elapsed = data.get("elapsed_s", 0)
                        fps = data.get("fps", 0)
                        total_ord = data.get("total_orders", 0)
                        metrics_count += 1
                        print(
                            f"  [{time.strftime('%H:%M:%S')}] "
                            f"video_time={_format_time_str(elapsed)} ({elapsed:.0f}s)  "
                            f"fps={fps:.1f}  orders={total_ord}",
                            flush=True,
                        )
                        for det_class, count in data.get("detections", {}).items():
                            total_detections_sum[det_class] = (
                                total_detections_sum.get(det_class, 0) + count
                            )

                    elif data.get("event") == "hotdog_summary":
                        hotdog_summary_snapshot = data.get("hotdog_log", {})
                        total_hd = hotdog_summary_snapshot.get("total_hotdogs", 0)
                        completed_hd = hotdog_summary_snapshot.get("completed_hotdogs", 0)
                        print(
                            f"\n  [{time.strftime('%H:%M:%S')}] [hotdog_summary] "
                            f"{total_hd} total hotdog(s) tracked | {completed_hd} completed/wrapped",
                            flush=True,
                        )
                    else:
                        print(f"  [JSON] {line_s}", flush=True)

                except json.JSONDecodeError:
                    parsed_as_json = False

            if not parsed_as_json:
                print(f"  {line_s}", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — collecting final summary...")
        process.terminate()

    process.wait()

    # ── Fallback: read directly from generated output file or API ───────────
    if not hotdog_summary_snapshot and os.path.exists(DEFAULT_SUMMARY_PATH):
        try:
            with open(DEFAULT_SUMMARY_PATH, "r") as f:
                hotdog_summary_snapshot = json.load(f)
        except Exception:
            pass

    if not hotdog_summary_snapshot:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/api/hotdog_log", timeout=5) as resp:
                hotdog_api = json.loads(resp.read().decode())
                raw = hotdog_api.get("hotdogs", {})
                hotdog_summary_snapshot = {
                    "total_hotdogs": len(raw),
                    "hotdogs": list(raw.values()),
                    "orders": {f"order{i}": rec for i, rec in enumerate(raw.values(), start=1)},
                }
        except Exception as e:
            print(f"[WARN] Could not fetch hotdog log from API ({e}).")

    # ── Extract or Build Structured Hotdogs List ────────────────────────────
    hotdogs_list: List[Dict[str, Any]] = []
    if "hotdogs" in hotdog_summary_snapshot and isinstance(hotdog_summary_snapshot["hotdogs"], list):
        hotdogs_list = hotdog_summary_snapshot["hotdogs"]
    elif "orders" in hotdog_summary_snapshot:
        for key, rec in hotdog_summary_snapshot.get("orders", {}).items():
            start_ts = rec.get("start_time_s", rec.get("first_seen"))
            end_ts = rec.get("end_time_s", rec.get("wrapping_done_time", rec.get("last_seen")))
            dur = rec.get("duration_s", (round(end_ts - start_ts, 2) if start_ts and end_ts else 0.0))
            undergone_wrap = rec.get("undergone_wrapping", rec.get("completed", False))

            items_list = []
            for it in rec.get("items_added", []):
                ts = it.get("timestamp_s", it.get("timestamp", it.get("video_timestamp_s", 0.0)))
                items_list.append({
                    "hotdog_id": rec.get("hotdog_id", key),
                    "item": it.get("item"),
                    "count": it.get("count", 1),
                    "timestamp_s": round(ts, 2),
                    "time_str": it.get("time_str", _format_time_str(ts)),
                    "video_timestamp_s": round(ts, 2),
                })

            hotdogs_list.append({
                "hotdog_id": str(rec.get("hotdog_id", key)),
                "track_id": rec.get("track_id"),
                "order_id": rec.get("order_id"),
                "status": rec.get("status", "done" if undergone_wrap else "in_progress"),
                "undergone_wrapping": undergone_wrap,
                "start_time": {
                    "timestamp_s": round(start_ts, 2) if start_ts is not None else None,
                    "time_str": _format_time_str(start_ts),
                },
                "end_time": {
                    "timestamp_s": round(end_ts, 2) if end_ts is not None else None,
                    "time_str": _format_time_str(end_ts),
                },
                "wrapping": {
                    "undergone_wrapping": undergone_wrap,
                    "started_at_s": rec.get("wrapping_dwell_start"),
                    "started_at_str": _format_time_str(rec.get("wrapping_dwell_start")),
                    "closing_at_s": rec.get("wrapping_closing_time"),
                    "closing_at_str": _format_time_str(rec.get("wrapping_closing_time")),
                    "done_at_s": rec.get("wrapping_done_time"),
                    "done_at_str": _format_time_str(rec.get("wrapping_done_time")),
                },
                "duration_s": dur,
                "items_added": items_list,
                "item_names": rec.get("item_names", [it["item"] for it in items_list]),
                "item_counts": rec.get("item_counts", {}),
                "completed": rec.get("completed", undergone_wrap),
            })

    # Sort hotdogs deterministically by start timestamp and ID
    hotdogs_list.sort(
        key=lambda h: (
            h.get("start_time", {}).get("timestamp_s") or 0.0,
            h.get("track_id") or 0,
        )
    )

    # Collect item timeline across all hotdogs
    item_timeline: List[Dict[str, Any]] = []
    for h in hotdogs_list:
        item_timeline.extend(h.get("items_added", []))
    item_timeline.sort(key=lambda x: x.get("timestamp_s", x.get("timestamp", 0.0)))

    completed_hotdogs_count = sum(
        1 for h in hotdogs_list if h.get("status") == "done" or h.get("undergone_wrapping")
    )

    # ── Build Full Summary Object ───────────────────────────────────────────
    full_summary = {
        "video": os.path.basename(video_path),
        "video_path": video_path,
        "run_duration_s": round(time.time() - start_wall_time, 1),
        "total_hotdogs": len(hotdogs_list),
        "completed_hotdogs": completed_hotdogs_count,
        "in_progress_hotdogs": len(hotdogs_list) - completed_hotdogs_count,
        "total_orders_processed": final_stats.get("total_orders", 0),
        "passed_orders": final_stats.get("passed_orders", 0),
        "failed_orders": final_stats.get("failed_orders", 0),
        "accuracy_pct": final_stats.get("accuracy_pct", 0.0),
        "detections_by_class": total_detections_sum,
        "hotdogs": hotdogs_list,
        "item_timeline": item_timeline,
        "orders": {f"order{i}": h for i, h in enumerate(hotdogs_list, start=1)},
        "regression_metrics": hotdog_summary_snapshot.get("regression_metrics", {}),
    }

    # ── Build Compact Timeline JSON Object ──────────────────────────────────
    timeline_summary = {
        "video": os.path.basename(video_path),
        "total_hotdogs": len(hotdogs_list),
        "completed_hotdogs": completed_hotdogs_count,
        "hotdogs": [
            {
                "hotdog_id": h.get("hotdog_id"),
                "status": h.get("status"),
                "undergone_wrapping": h.get("undergone_wrapping"),
                "start_time": h.get("start_time", {}).get("time_str"),
                "start_timestamp_s": h.get("start_time", {}).get("timestamp_s"),
                "end_time": h.get("end_time", {}).get("time_str"),
                "end_timestamp_s": h.get("end_time", {}).get("timestamp_s"),
                "duration_s": h.get("duration_s"),
                "items_added": [
                    {
                        "item": it.get("item"),
                        "time": it.get("time_str"),
                        "timestamp_s": it.get("timestamp_s"),
                    }
                    for it in h.get("items_added", [])
                ],
                "item_summary": ", ".join(h.get("item_names", [])),
                "wrapping_completed": h.get("wrapping", {}).get("done_at_str")
                or h.get("wrapping", {}).get("closing_at_str"),
            }
            for h in hotdogs_list
        ],
    }

    # ── Save JSON Output Files ──────────────────────────────────────────────
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=2)

    Path(timeline_path).parent.mkdir(parents=True, exist_ok=True)
    with open(timeline_path, "w") as f:
        json.dump(timeline_summary, f, indent=2)

    # ── Print Structured Terminal Report ───────────────────────────────────
    print()
    print("=" * 75)
    print("  HOTDOG ACCURACY & TIMELINE REPORT")
    print("=" * 75)
    print(f"  Video File        : {os.path.basename(video_path)}")
    print(f"  Run Duration      : {full_summary['run_duration_s']}s wall time")
    print(f"  Total Hotdogs     : {full_summary['total_hotdogs']}")
    print(f"  Completed/Wrapped : {full_summary['completed_hotdogs']}")
    print(f"  In Progress       : {full_summary['in_progress_hotdogs']}")
    print(f"  Orders Processed  : {full_summary['total_orders_processed']}")
    print(f"  Passed / Failed   : {full_summary['passed_orders']} / {full_summary['failed_orders']}")
    print(f"  Accuracy          : {full_summary['accuracy_pct']}%")
    print("=" * 75)
    print()

    if hotdogs_list:
        print("  DETAILED HOTDOG TIMELINES:")
        print("  " + "-" * 71)
        for h in hotdogs_list:
            hid = h.get("hotdog_id")
            status = h.get("status", "unknown").upper()
            start_str = h.get("start_time", {}).get("time_str") or "00:00:00"
            end_str = h.get("end_time", {}).get("time_str") or "--:--:--"
            dur = h.get("duration_s", 0.0)
            wrap_done = h.get("wrapping", {}).get("done_at_str")
            wrap_close = h.get("wrapping", {}).get("closing_at_str")
            wrap_str = f"Wrapped at {wrap_done or wrap_close}" if (wrap_done or wrap_close) else "No wrapping"

            print(f"  Hotdog #{hid} [{status}]")
            print(f"    • Started at : {start_str} ({h.get('start_time', {}).get('timestamp_s')}s)")
            print(f"    • Ended at   : {end_str} ({h.get('end_time', {}).get('timestamp_s')}s) | {wrap_str} | Duration: {dur}s")

            items = h.get("items_added", [])
            if items:
                print("    • Ingredients Added:")
                for item_info in items:
                    iname = item_info.get("item")
                    tstr = item_info.get("time_str", "")
                    ts = item_info.get("timestamp_s", 0.0)
                    print(f"        - [{tstr}] {iname} (at {ts:.1f}s)")
            else:
                print("    • Ingredients Added: (none detected)")
            print("  " + "-" * 71)
    else:
        print("  (No hotdogs detected)")

    print()
    print("  Detections Summary:")
    if total_detections_sum:
        for cls, cnt in sorted(total_detections_sum.items()):
            print(f"    - {cls}: {cnt}")
    else:
        print("    (none)")

    print()
    print("=" * 75)
    print(f"  Full JSON Output     -> {summary_path}")
    print(f"  Timeline JSON Output -> {timeline_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
