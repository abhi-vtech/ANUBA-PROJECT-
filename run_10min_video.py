"""
run_10min_video.py
──────────────────
Runs the main pipeline against the 10-minute Wienerschnitzel video
(videos/wienerschnitzel_10m.mkv) and produces a hotdog-by-hotdog JSON
summary at the end.

Usage
─────
    python run_10min_video.py

Output
──────
  • Live dashboard at http://localhost:8000  (while running)
  • output/hotdog_summary.json              (after completion)
  • stdout summary in plain text + JSON

Do NOT modify this file to change core behavior — it is purely an
additive runner script.
"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "wienerschnitzel_10m.mkv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "hotdog_summary.json")


def _python_executable() -> str:
    """Prefer the workspace virtual environment so GPU-enabled deps are used."""
    candidates = []
    if os.name == "nt":
        candidates.append(os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe"))
    else:
        candidates.append(os.path.join(BASE_DIR, ".venv", "bin", "python"))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return sys.executable


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] 10-minute video not found at: {VIDEO_PATH}")
        print("Available videos in videos/:")
        for f in Path(os.path.join(BASE_DIR, "videos")).iterdir():
            print(f"  {f.name}")
        sys.exit(1)

    python_executable = _python_executable()

    print("=" * 60)
    print("  HOTDOG TRACKER — 10-Minute Video Run")
    print("=" * 60)
    print(f"  Video  : {os.path.basename(VIDEO_PATH)}")
    print(f"  Output : {SUMMARY_PATH}")
    print(f"  Python : {python_executable}")
    print(f"  Dashboard will be live at: http://localhost:8000")
    print("=" * 60)
    print()

    env = os.environ.copy()
    env["VIDEO_SOURCE"] = VIDEO_PATH
    env["EXIT_ON_END"] = "true"
    # Ensure INFO-level log events (metrics, hotdog_summary) pass through stdout
    env["LOG_LEVEL"] = "INFO"
    env["MULTI_ID"] = "true"

    # ──────────────────────────────────────────────────────────────────────
    # Launch the main pipeline
    # ──────────────────────────────────────────────────────────────────────
    process = subprocess.Popen(
        [python_executable, "-u", "-m", "src.main"],
        cwd=BASE_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )

    # Counters for the plain-text report
    total_detections_sum: dict = {}
    final_stats: dict = {}
    hotdog_log_snapshot: dict = {}
    metrics_count = 0

    start_time = time.time()

    print("[INFO] Pipeline starting… (dashboard will appear at http://localhost:8000)")
    print("[INFO] Press Ctrl+C to abort.\n")
    sys.stdout.flush()

    CONSOLE_LOG_PATH = os.path.join(OUTPUT_DIR, "console_output.log")
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    console_file = open(CONSOLE_LOG_PATH, "w", encoding="utf-8")

    def _log_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        print(msg, **kwargs)
        console_file.write(msg + "\n")
        console_file.flush()

    try:
        for line in process.stdout:
            line_s = line.rstrip()
            console_file.write(line_s + "\n")
            console_file.flush()

            # Try to parse as JSON metric/event line first
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
                        _log_print(
                            f"  [{time.strftime('%H:%M:%S')}] "
                            f"elapsed={elapsed:.0f}s  fps={fps:.1f}  "
                            f"orders={total_ord}",
                            flush=True,
                        )
                        for det_class, count in data.get("detections", {}).items():
                            total_detections_sum[det_class] = (
                                total_detections_sum.get(det_class, 0) + count
                            )

                    elif data.get("event") == "hotdog_summary":
                        hl = data.get("hotdog_log", {})
                        raw_orders = hl.get("orders", {})
                        hotdog_log_snapshot = raw_orders
                        _log_print(
                            f"  [{time.strftime('%H:%M:%S')}] "
                            f"[hotdog_summary] {hl.get('total_hotdogs', 0)} hotdog(s) captured",
                            flush=True,
                        )
                    else:
                        # Other JSON lines: print them so nothing is hidden
                        _log_print(f"  [JSON] {line_s}", flush=True)

                except json.JSONDecodeError:
                    parsed_as_json = False

            # Non-JSON lines — print them directly so startup errors and
            # plain print() output are always visible
            if not parsed_as_json and line_s:
                _log_print(f"  {line_s}", flush=True)

    except KeyboardInterrupt:
        _log_print("\n[INFO] Interrupted by user — collecting hotdog log snapshot...")
        process.terminate()

    process.wait()

    # ── Fallback: read directly from generated output file or API ───────────
    if not hotdog_log_snapshot and os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r") as f:
                hotdog_log_snapshot = json.load(f)
        except Exception:
            pass

    if not hotdog_log_snapshot:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/api/hotdog_log", timeout=5) as resp:
                hotdog_api = json.loads(resp.read().decode())
                raw = hotdog_api.get("hotdogs", {})
                hotdog_log_snapshot = {
                    "total_hotdogs": len(raw),
                    "hotdogs": list(raw.values()),
                    "orders": {f"order{i}": rec for i, rec in enumerate(raw.values(), start=1)},
                }
        except Exception as e:
            print(f"[WARN] Could not fetch hotdog log from API ({e}).")

    # ── Extract or Build Structured Hotdogs List ────────────────────────────
    def _format_time_str(ts):
        if ts is None:
            return None
        if ts < 86400:
            m, s = divmod(int(ts), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"
        return time.strftime("%H:%M:%S", time.localtime(ts))

    hotdogs_list = []
    if "hotdogs" in hotdog_log_snapshot and isinstance(hotdog_log_snapshot["hotdogs"], list):
        hotdogs_list = hotdog_log_snapshot["hotdogs"]
    elif "orders" in hotdog_log_snapshot:
        for key, rec in hotdog_log_snapshot.get("orders", {}).items():
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

    hotdogs_list.sort(
        key=lambda h: (
            h.get("start_time", {}).get("timestamp_s") or 0.0,
            h.get("track_id") or 0,
        )
    )

    item_timeline = []
    for h in hotdogs_list:
        item_timeline.extend(h.get("items_added", []))
    item_timeline.sort(key=lambda x: x.get("timestamp_s", x.get("timestamp", 0.0)))

    completed_hotdogs_count = sum(
        1 for h in hotdogs_list if h.get("status") == "done" or h.get("undergone_wrapping")
    )

    orders_summary = {f"order{i}": h for i, h in enumerate(hotdogs_list, start=1)}

    full_summary = {
        "video": os.path.basename(VIDEO_PATH),
        "run_duration_s": round(time.time() - start_time, 1),
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
        "orders": orders_summary,
        "regression_metrics": hotdog_log_snapshot.get("regression_metrics", {}),
    }

    timeline_summary = {
        "video": os.path.basename(VIDEO_PATH),
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

    # ──────────────────────────────────────────────────────────────────────
    # Save JSON summary
    # ──────────────────────────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(full_summary, f, indent=2)

    timeline_path = os.path.join(OUTPUT_DIR, "hotdog_timeline.json")
    with open(timeline_path, "w") as f:
        json.dump(timeline_summary, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────
    # Print plain-text report
    # ──────────────────────────────────────────────────────────────────────
    _log_print()
    _log_print("=" * 70)
    _log_print("  VIDEO SUMMARY & HOTDOG TIMELINE REPORT")
    _log_print("=" * 70)
    _log_print(f"  Video             : {os.path.basename(VIDEO_PATH)}")
    _log_print(f"  Run duration      : {full_summary['run_duration_s']}s")
    _log_print(f"  Metrics snapshots : {metrics_count}")
    _log_print(f"  Total orders      : {full_summary['total_orders_processed']}")
    _log_print(f"  Passed orders     : {full_summary['passed_orders']}")
    _log_print(f"  Failed orders     : {full_summary['failed_orders']}")
    _log_print(f"  Accuracy          : {full_summary['accuracy_pct']}%")
    _log_print()
    _log_print(f"  Total Hotdogs     : {full_summary['total_hotdogs']}")
    _log_print(f"  Completed/Wrapped : {full_summary['completed_hotdogs']}")
    _log_print()

    if hotdogs_list:
        _log_print("  Hotdog Timeline & Ingredient Log:")
        _log_print("  " + "-" * 66)
        for h in hotdogs_list:
            hid = h.get("hotdog_id")
            st = h.get("status", "unknown").upper()
            start_str = h.get("start_time", {}).get("time_str") or "00:00:00"
            end_str = h.get("end_time", {}).get("time_str") or "--:--:--"
            dur = h.get("duration_s", 0.0)
            items = h.get("items_added", [])
            item_str = ", ".join(f"[{it.get('time_str')}] {it.get('item')}" for it in items) if items else "(none)"
            _log_print(f"  Hotdog #{hid} [{st}]: {start_str} -> {end_str} ({dur}s) | Items: {item_str}")
    else:
        _log_print("  (No hotdog item associations recorded)")

    _log_print()
    _log_print("  Detections by class:")
    if total_detections_sum:
        for cls, cnt in sorted(total_detections_sum.items()):
            _log_print(f"    - {cls}: {cnt}")
    else:
        _log_print("    (none — no metrics events received)")

    _log_print()
    _log_print(f"  Full JSON saved     -> {SUMMARY_PATH}")
    _log_print(f"  Timeline JSON saved -> {timeline_path}")
    _log_print(f"  Console log saved   -> {CONSOLE_LOG_PATH}")
    _log_print("=" * 70)

    console_file.close()


if __name__ == "__main__":
    main()
