"""Calibrate the KDS colour bands against real footage.

    python scripts/calibrate_kds_colors.py <kds video or image> [--samples 40]

Samples frames, runs the real card detector and row classifier, and reports:

* how many ticket cards were found per frame,
* the measured HSV of every classified row, grouped by the role it was given,
* every row that fell through to PLAIN (the rows a band is missing),
* the light-pink fraction per card over time -- the overdue indicator.

Use it after changing KDS hardware/theme, or when a bar colour is not being
recognised.  Nothing is written; this only reports.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kds.colors import ColorBands, load_visual_config  # noqa: E402
from src.kds.ticket_detector import TicketCardDetector  # noqa: E402


def iter_frames(source: str, samples: int):
    """Yield ``(index, frame)`` evenly spread across a video, or a single image."""
    if Path(source).suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"):
        image = cv2.imread(source)
        if image is None:
            raise SystemExit("could not read image: %s" % source)
        yield 0, image
        return

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit("could not open video: %s" % source)
    # Frame-index seeking is unreliable on some containers, so seek by time.
    duration_ms = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_ms = duration_ms * 1000.0 / fps if fps > 1.5 else duration_ms
    for i in range(samples):
        capture.set(cv2.CAP_PROP_POS_MSEC, total_ms * i / float(max(1, samples)))
        ok, frame = capture.read()
        if ok:
            yield i, frame
    capture.release()


def describe(hsv_pixels: np.ndarray) -> str:
    if hsv_pixels.size == 0:
        return "(no pixels)"
    lo = np.percentile(hsv_pixels, 5, axis=0).astype(int)
    hi = np.percentile(hsv_pixels, 95, axis=0).astype(int)
    med = np.median(hsv_pixels, axis=0).astype(int)
    return "H %3d-%3d (med %3d)  S %3d-%3d (med %3d)  V %3d-%3d (med %3d)" % (
        lo[0], hi[0], med[0], lo[1], hi[1], med[1], lo[2], hi[2], med[2]
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="KDS video file or a still image of the screen")
    parser.add_argument("--samples", type=int, default=30, help="frames to sample")
    parser.add_argument(
        "--rows", type=int, default=14, help="max rows to sample per card"
    )
    args = parser.parse_args(argv)

    config = load_visual_config()
    bands = ColorBands(config)
    detector = TicketCardDetector(bands)

    by_role = defaultdict(list)
    plain_rows = []
    pink_series = []
    card_counts = []

    for index, frame in iter_frames(args.source, args.samples):
        cards = detector.find_card_boxes(frame)
        card_counts.append(len(cards))
        for (x1, y1, x2, y2) in cards:
            card = frame[y1:y2, x1:x2]
            if card.size == 0:
                continue
            pink_series.append((index, round(bands.pink_fraction(card), 3)))
            height = card.shape[0]
            step = max(1, height // args.rows)
            for top in range(0, height - step, step):
                strip = card[top:top + step]
                role, fractions = bands.classify_patch(strip)
                hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV).reshape(-1, 3)
                by_role[role.value].append(hsv)
                if role.value == "plain":
                    best = max(fractions.items(), key=lambda kv: kv[1]) if fractions else ("-", 0)
                    plain_rows.append((index, top, best[0], round(best[1], 3)))

    print("=" * 78)
    print("KDS COLOUR CALIBRATION - %s" % args.source)
    print("=" * 78)
    print("frames sampled      : %d" % len(card_counts))
    if card_counts:
        print("cards per frame     : min %d  max %d  mean %.1f"
              % (min(card_counts), max(card_counts), sum(card_counts) / len(card_counts)))
    if not any(card_counts):
        print("\n!! No ticket cards detected.  Check card_detection.* in "
              "config/kds_visual.yaml (screen_v_max, min_area_fraction).")

    print("\nMeasured HSV per classified role")
    print("-" * 78)
    for role, chunks in sorted(by_role.items()):
        pixels = np.concatenate(chunks) if chunks else np.empty((0, 3))
        print("  %-11s n=%-7d %s" % (role, len(chunks), describe(pixels)))

    if plain_rows:
        print("\nRows that matched NO band (closest band shown)")
        print("-" * 78)
        for frame_index, top, band, fraction in plain_rows[:25]:
            print("  frame %-4d y=%-5d closest=%-11s fraction=%.3f"
                  % (frame_index, top, band, fraction))
        if len(plain_rows) > 25:
            print("  ... %d more" % (len(plain_rows) - 25))
        print("\n  Rows here are usually the card body itself (expected) or a bar "
              "colour with no band yet (add one under `colors:`).")

    if pink_series:
        values = [v for _, v in pink_series]
        threshold = bands.bands["light_pink"].min_fraction
        above = sum(1 for v in values if v >= threshold)
        print("\nLight-pink fraction per card observation")
        print("-" * 78)
        print("  min %.3f  max %.3f  mean %.3f   (threshold %.2f)"
              % (min(values), max(values), sum(values) / len(values), threshold))
        print("  observations above threshold: %d / %d" % (above, len(values)))
        print("\n  Pink marks an OVERDUE card, nothing more.  Orders are judged "
              "when the ticket disappears from the screen, never on colour, so "
              "a busy screen reading mostly pink is expected and harmless.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
