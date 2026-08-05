"""
test_wrapping_detection.py
──────────────────────────
Standalone diagnostic script that runs ONLY wrapping-class detection across
every video in the videos/ folder and reports:

  • Whether "wrapping" was ever detected
  • Frame numbers, timestamps, confidence scores, and bounding boxes for each hit
  • A per-video summary table at the end

This script does NOT use any other pipeline code — it talks directly to
ultralytics YOLO with no tracker, no ByteTrack, no confidence filter other
than its own adjustable threshold.

Usage:
    python test_wrapping_detection.py [--conf 0.1] [--video path/to/video.mp4]

Flags:
    --conf  FLOAT    Detection confidence threshold (default: 0.10 — very low,
                     to catch even weak signals)
    --video PATH     Run on a single video instead of all videos/ files
    --max-frames N   Stop each video after N frames (default: 0 = whole video)
    --every N        Sample every N-th frame (default: 1 = every frame)
    --show           Display annotated frames with cv2.imshow (needs a display)
"""

import argparse
import os
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "videos"
MODEL_PATH = BASE_DIR / "rf_trained" / "weights.pt"

WRAPPING_CLASS = "wrapping"


def parse_args():
    p = argparse.ArgumentParser(description="Wrapping detection diagnostic")
    p.add_argument("--conf", type=float, default=0.10,
                   help="Confidence threshold (default 0.10)")
    p.add_argument("--video", type=str, default=None,
                   help="Path to a single video (default: all videos/ files)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Stop after N frames per video (0 = entire video)")
    p.add_argument("--every", type=int, default=1,
                   help="Sample every N-th frame (default 1 = every frame)")
    p.add_argument("--show", action="store_true",
                   help="Display annotated frames (requires a display)")
    return p.parse_args()


def _fmt_time(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_on_video(model, video_path: Path, conf: float, max_frames: int,
                 every: int, show: bool) -> dict:
    """
    Run detection on one video.  Returns a summary dict.
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path.name}")
        return {"video": video_path.name, "error": "Cannot open"}

    video_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps

    print(f"\n{'─'*60}")
    print(f"  Video  : {video_path.name}")
    print(f"  FPS    : {video_fps:.1f}   Frames: {total_frames}   "
          f"Duration: {_fmt_time(duration_s)}")
    print(f"  Conf   : {conf}   Every: {every}   "
          f"MaxFrames: {max_frames if max_frames else 'all'}")
    print(f"{'─'*60}")

    hits = []           # list of detection dicts
    frame_idx   = 0
    sampled     = 0
    t_start     = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Sample every N-th frame
        if (frame_idx - 1) % every != 0:
            continue

        sampled += 1

        # Stop early if requested
        if max_frames and sampled > max_frames:
            break

        video_ts = frame_idx / video_fps

        # ── Run inference (no tracker — pure detection) ────────────────────
        results = model(frame, verbose=False, conf=conf, device="cpu")

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id   = int(box.cls)
                cls_name = model.names.get(cls_id, str(cls_id))
                if cls_name != WRAPPING_CLASS:
                    continue

                confidence = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                w = x2 - x1
                h = y2 - y1

                hit = {
                    "frame":      frame_idx,
                    "timestamp":  round(video_ts, 2),
                    "time_str":   _fmt_time(video_ts),
                    "confidence": round(confidence, 4),
                    "bbox":       (x1, y1, x2, y2),
                    "width":      w,
                    "height":     h,
                }
                hits.append(hit)

                print(
                    f"  [HIT] frame={frame_idx:6d}  t={_fmt_time(video_ts)}"
                    f"  conf={confidence:.3f}"
                    f"  bbox=({x1},{y1},{x2},{y2})  {w}x{h}px"
                )

                if show:
                    # Draw detection on frame
                    import cv2 as _cv2
                    _cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"wrapping {confidence:.2f}"
                    _cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if show:
            import cv2 as _cv2
            _cv2.putText(frame, f"frame {frame_idx}  t={_fmt_time(video_ts)}",
                         (10, 30), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            _cv2.imshow("Wrapping detection", frame)
            key = _cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        # Progress every 300 sampled frames
        if sampled % 300 == 0:
            elapsed = time.perf_counter() - t_start
            pct = (frame_idx / total_frames * 100) if total_frames else 0
            print(f"  ... frame {frame_idx:6d} / {total_frames}  "
                  f"({pct:.0f}%)  hits so far: {len(hits)}  "
                  f"elapsed: {elapsed:.1f}s")

    cap.release()
    if show:
        import cv2 as _cv2
        _cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t_start
    summary = {
        "video":          video_path.name,
        "total_frames":   total_frames,
        "sampled_frames": sampled,
        "duration_s":     round(duration_s, 1),
        "wrapping_hits":  len(hits),
        "elapsed_s":      round(elapsed, 1),
        "hits":           hits,
    }

    if hits:
        confs = [h["confidence"] for h in hits]
        print(f"\n  ✓  {len(hits)} wrapping detection(s)")
        print(f"     conf: min={min(confs):.3f}  max={max(confs):.3f}  "
              f"avg={sum(confs)/len(confs):.3f}")
        print(f"     first hit: frame {hits[0]['frame']}  t={hits[0]['time_str']}")
        print(f"     last  hit: frame {hits[-1]['frame']}  t={hits[-1]['time_str']}")
    else:
        print(f"\n  ✗  NO wrapping detections found at conf >= {conf}")

    return summary


def main():
    args = parse_args()

    # Force line-buffered stdout so progress is visible immediately even
    # when output is piped or captured by a task runner.
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, line_buffering=True, encoding="utf-8"
    )

    # ── Load model ──────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  WRAPPING CLASS DETECTION DIAGNOSTIC", flush=True)
    print("=" * 60, flush=True)
    print(f"  Model: {MODEL_PATH.name}", flush=True)
    print(f"  Conf threshold: {args.conf}", flush=True)
    print(flush=True)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed.  Run: pip install ultralytics")
        sys.exit(1)

    print("[INFO] Loading model …", flush=True)
    model = YOLO(str(MODEL_PATH))

    # Confirm wrapping class exists
    cls_names = model.names
    print(f"[INFO] Model has {len(cls_names)} classes:", flush=True)
    for idx, name in cls_names.items():
        marker = " ← TARGET" if name == WRAPPING_CLASS else ""
        print(f"       {idx}: {name}{marker}", flush=True)
    print(flush=True)

    if WRAPPING_CLASS not in cls_names.values():
        print(f"[ERROR] '{WRAPPING_CLASS}' class is NOT in this model's weights!")
        sys.exit(1)

    # ── Collect videos ──────────────────────────────────────────────────────
    if args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(
            p for p in VIDEO_DIR.iterdir()
            if p.suffix.lower() in {".mp4", ".mkv", ".avi", ".mov", ".webm"}
        )
        if not videos:
            print(f"[ERROR] No video files found in {VIDEO_DIR}")
            sys.exit(1)

    print(f"[INFO] Will scan {len(videos)} video(s):", flush=True)
    for v in videos:
        size_mb = v.stat().st_size / 1_048_576
        print(f"       {v.name}  ({size_mb:.0f} MB)", flush=True)
    print(flush=True)

    # ── Run detection on each video ─────────────────────────────────────────
    all_summaries = []
    for video_path in videos:
        summary = run_on_video(
            model       = model,
            video_path  = video_path,
            conf        = args.conf,
            max_frames  = args.max_frames,
            every       = args.every,
            show        = args.show,
        )
        all_summaries.append(summary)

    # ── Final summary table ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  OVERALL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Video':<50}  {'Hits':>6}  {'Frames':>7}  {'Time':>6}")
    print(f"  {'-'*50}  {'------':>6}  {'-------':>7}  {'------':>6}")
    total_hits = 0
    for s in all_summaries:
        total_hits += s.get("wrapping_hits", 0)
        err = s.get("error", "")
        hits_str = str(s.get("wrapping_hits", "-")) if not err else "ERROR"
        print(
            f"  {s['video']:<50}  {hits_str:>6}  "
            f"{s.get('sampled_frames', 0):>7}  "
            f"{s.get('elapsed_s', 0):>5.0f}s"
        )
    print(f"  {'─'*50}")
    print(f"  {'TOTAL':50}  {total_hits:>6}")
    print(f"{'='*60}")

    if total_hits == 0:
        print()
        print("  ⚠  DIAGNOSIS: Model never fired the 'wrapping' class.")
        print("     Possible causes:")
        print("     1. Videos do not contain wrapping events.")
        print("     2. Confidence threshold still too high — retry with --conf 0.01")
        print("     3. The model was trained at a different resolution — try")
        print("        resizing frames or check the model's expected imgsz.")
        print("     4. 'wrapping' was present in training but the class is")
        print("        severely under-represented → model has very low recall.")


if __name__ == "__main__":
    main()
