"""
test_wrapping_pipeline.py
─────────────────────────
End-to-end smoke test: runs the REAL Detector (model.track + ByteTrack)
on order1.mp4 and reports every "wrapping" detection that survives the
full confidence + tracker pipeline.

Usage:
    python test_wrapping_pipeline.py [--max-frames 200]
"""

import sys
import os
import argparse
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# Set sys.argv[0] so src/paths.py resolves app_root correctly
sys.argv[0] = str(BASE_DIR / "test_wrapping_pipeline.py")

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True, encoding="utf-8")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-frames", type=int, default=200,
                   help="Stop after N frames (default 200)")
    p.add_argument("--video", type=str, default="videos/order1.mp4")
    args = p.parse_args()

    video_path = BASE_DIR / args.video

    print("=" * 60)
    print("  WRAPPING — FULL PIPELINE SMOKE TEST")
    print("=" * 60)
    print(f"  Video      : {video_path.name}")
    print(f"  Max frames : {args.max_frames}")
    print()

    # ── Load Detector exactly as main.py does ──────────────────────────────
    import yaml
    from src.detector import Detector
    from src.paths import resource

    config = yaml.safe_load(Path(resource("config/model.yaml")).read_text())
    model_path   = config.get("model_path", "rf_trained/weights.pt")
    tracker_type = config.get("tracker_type", "bytetrack")
    confidence   = config.get("confidence_threshold", 0.5)
    wrapping_conf = config.get("wrapping_conf_threshold", 0.15)

    print(f"  Model       : {model_path}")
    print(f"  Global conf : {confidence}")
    print(f"  Wrapping cf : {wrapping_conf}")
    print(f"  Tracker     : {tracker_type}")
    print()
    print("[INFO] Loading Detector (this may take 10-15s)…", flush=True)

    detector = Detector(
        model_path=resource(model_path),
        tracker_type=tracker_type,
        tracker_config=resource("config/tracker.yaml"),
        class_conf_overrides={"wrapping": wrapping_conf},
    )

    print(f"[INFO] Model classes: {list(detector.model.names.values())}")
    print()

    # ── Open video ─────────────────────────────────────────────────────────
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Scanning up to {args.max_frames} frames of {total_frames} total…")
    print()

    frame_idx = 0
    wrapping_hits = []
    all_classes_seen: dict = {}
    t0 = time.perf_counter()

    while frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        ts = frame_idx / fps

        detections = detector.detect(frame, conf_threshold=confidence)

        for det in detections:
            all_classes_seen[det.class_name] = all_classes_seen.get(det.class_name, 0) + 1
            if det.class_name == "wrapping":
                wrapping_hits.append({
                    "frame": frame_idx,
                    "ts":    round(ts, 2),
                    "conf":  round(det.confidence, 3),
                    "bbox":  det.bbox,
                    "tid":   det.track_id,
                })
                x1, y1, x2, y2 = det.bbox
                print(
                    f"  [WRAPPING] frame={frame_idx:5d}  t={ts:6.2f}s"
                    f"  conf={det.confidence:.3f}  tid={det.track_id}"
                    f"  bbox=({x1},{y1},{x2},{y2})",
                    flush=True,
                )

    cap.release()
    elapsed = time.perf_counter() - t0

    print()
    print("=" * 60)
    print(f"  Scanned {frame_idx} frames in {elapsed:.1f}s")
    print()
    print("  All classes detected:")
    for cls, count in sorted(all_classes_seen.items()):
        marker = " ← WRAPPING" if cls == "wrapping" else ""
        print(f"    {cls}: {count}{marker}")
    print()

    if wrapping_hits:
        confs = [h["conf"] for h in wrapping_hits]
        print(f"  ✓ {len(wrapping_hits)} wrapping detection(s)")
        print(f"    conf: min={min(confs):.3f}  max={max(confs):.3f}  avg={sum(confs)/len(confs):.3f}")
        print(f"    first: frame {wrapping_hits[0]['frame']}  t={wrapping_hits[0]['ts']}s")
    else:
        print("  ✗ NO wrapping detected in pipeline mode")
        print("    → Check tracker.yaml track_high_thresh and model.track() conf= arg")

    print("=" * 60)


if __name__ == "__main__":
    main()
