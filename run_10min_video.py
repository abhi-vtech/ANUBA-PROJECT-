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

    try:
        for line in process.stdout:
            line_s = line.rstrip()

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
                        print(
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
                        print(
                            f"  [{time.strftime('%H:%M:%S')}] "
                            f"[hotdog_summary] {hl.get('total_hotdogs', 0)} hotdog(s) captured",
                            flush=True,
                        )
                    else:
                        # Other JSON lines: print them so nothing is hidden
                        print(f"  [JSON] {line_s}", flush=True)

                except json.JSONDecodeError:
                    parsed_as_json = False

            # Non-JSON lines — print them directly so startup errors and
            # plain print() output are always visible
            if not parsed_as_json and line_s:
                print(f"  {line_s}", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user — fetching hotdog log snapshot...")
        process.terminate()

    process.wait()

    # ── Fetch hotdog log (fallback: try API if event wasn't captured) ──────
    if not hotdog_log_snapshot:
        try:
            import urllib.request
            with urllib.request.urlopen("http://localhost:8000/api/hotdog_log", timeout=5) as resp:
                hotdog_api = json.loads(resp.read().decode())
                raw = hotdog_api.get("hotdogs", {})
                for i, (hid, rec) in enumerate(raw.items(), start=1):
                    hotdog_log_snapshot[f"order{i}"] = {
                        "hotdog_id": rec.get("hotdog_id", hid),
                        "track_id": rec.get("track_id"),
                        "item_names": rec.get("item_names", []),
                        "items_added": rec.get("items_added", []),
                        "completed": len(rec.get("items_added", [])) > 0,
                    }
        except Exception as e:
            print(f"[WARN] Could not fetch hotdog log from API ({e}).")

    # ── Build order-style summary ───────────────────────────────────────────
    orders_summary: dict = {}
    item_timeline: list = []
    for key, rec in hotdog_log_snapshot.items():
        if isinstance(rec, dict) and "hotdog_id" in rec:
            orders_summary[key] = rec
            items_added = rec.get("items_added", [])
            for it in items_added:
                if "hotdog_id" not in it:
                    it["hotdog_id"] = rec["hotdog_id"]
            item_timeline.extend(items_added)
        else:
            orders_summary[key] = {"hotdog_id": key, "item_names": []}

    item_timeline.sort(key=lambda x: x.get("timestamp", 0))

    full_summary = {
        "video": os.path.basename(VIDEO_PATH),
        "run_duration_s": round(time.time() - start_time, 1),
        "total_hotdogs": len(orders_summary),
        "total_orders_processed": final_stats.get("total_orders", 0),
        "passed_orders": final_stats.get("passed_orders", 0),
        "failed_orders": final_stats.get("failed_orders", 0),
        "accuracy_pct": final_stats.get("accuracy_pct", 0.0),
        "detections_by_class": total_detections_sum,
        "item_timeline": item_timeline,
        "orders": orders_summary,
    }

    # ──────────────────────────────────────────────────────────────────────
    # Save JSON summary
    # ──────────────────────────────────────────────────────────────────────
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(full_summary, f, indent=2)

    # ──────────────────────────────────────────────────────────────────────
    # Print plain-text report
    # ──────────────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  VIDEO SUMMARY REPORT")
    print("=" * 60)
    print(f"  Video             : {os.path.basename(VIDEO_PATH)}")
    print(f"  Run duration      : {full_summary['run_duration_s']}s")
    print(f"  Metrics snapshots : {metrics_count}")
    print(f"  Total orders      : {full_summary['total_orders_processed']}")
    print(f"  Passed orders     : {full_summary['passed_orders']}")
    print(f"  Failed orders     : {full_summary['failed_orders']}")
    print(f"  Accuracy          : {full_summary['accuracy_pct']}%")
    print()
    print(f"  Hotdogs detected  : {full_summary['total_hotdogs']}")
    print()

    if orders_summary:
        print("  Per-Hotdog Item Log:")
        print("  " + "-" * 40)
        for order_key, rec in orders_summary.items():
            hid = rec.get("hotdog_id", order_key)
            items = rec.get("item_names", [])
            item_counts = rec.get("item_counts", {})
            item_str = ", ".join(items) if items else "(none detected)"
            if item_counts:
                count_str = ", ".join(
                    f"{name}={count}" for name, count in sorted(item_counts.items())
                )
                item_str = f"{item_str} | counts: {count_str}"
            print(f"  {order_key} [{hid}]: {item_str}")
    else:
        print("  (No hotdog item associations recorded)")

    print()
    print("  Detections by class:")
    if total_detections_sum:
        for cls, cnt in sorted(total_detections_sum.items()):
            print(f"    - {cls}: {cnt}")
    else:
        print("    (none — no metrics events received)")

    print()
    print("  JSON Summary (order-style):")
    print("  " + "-" * 40)
    compact = {
        k: {"hotdog_id": v.get("hotdog_id", k), "items": v.get("item_names", [])}
        for k, v in orders_summary.items()
    }
    print("  " + json.dumps(compact, indent=4).replace("\n", "\n  "))

    print()
    print(f"  Full JSON saved -> {SUMMARY_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
