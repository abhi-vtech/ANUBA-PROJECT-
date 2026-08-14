#!/usr/bin/env python3
"""
run_exit_line_detector.py
─────────────────────────
Interactive OpenCV Exit Line Detector Script.

Features:
1. Interactive Drawing Mode (`--draw`): Click & drag on video frame to position exit tripwire line.
2. Exit Detection Execution:
   - Detects worker hand crossing exit line.
   - Declares all wrapped hotdogs up to that point as EXITED / DISPATCHED.
   - Resets Cart State Machine for the next order.
   - Displays vibrant visual overlay with crossing alerts.

Usage:
  # Draw exit line interactively
  python run_exit_line_detector.py --video videos/test.mp4 --draw

  # Run exit detection pipeline
  python run_exit_line_detector.py --video videos/test.mp4
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.cart_state_machine import CartEvent, CartStateMachine, EventType
from src.detector import Detector
from src.exit_detector import ExitDetector
from src.hotdog_tracker import HotdogTracker
from src.paths import resource
from src.wrapping_state import WrappingStateMachine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_exit_line_detector")

# Interactive mouse callback state
_drawing_p1 = None
_drawing_p2 = None
_is_drawing = False


def mouse_callback(event, x, y, flags, param):
    global _drawing_p1, _drawing_p2, _is_drawing
    frame_w, frame_h = param["width"], param["height"]
    norm_x, norm_y = x / frame_w, y / frame_h

    if event == cv2.EVENT_LBUTTONDOWN:
        _drawing_p1 = (norm_x, norm_y)
        _drawing_p2 = (norm_x, norm_y)
        _is_drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and _is_drawing:
        _drawing_p2 = (norm_x, norm_y)
    elif event == cv2.EVENT_LBUTTONUP:
        _drawing_p2 = (norm_x, norm_y)
        _is_drawing = False
        logger.info(f"Drawn Line Endpoints: p1={_drawing_p1}, p2={_drawing_p2}")


def run_draw_mode(video_path: str, config_path: str):
    """Interactive GUI mode to position exit line using mouse dragging."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to read first video frame")
        cap.release()
        return

    h, w = frame.shape[:2]
    win_name = "Draw Exit Line (Click & Drag, Press 'S' to Save, 'Q' to Quit)"
    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, mouse_callback, {"width": w, "height": h})

    detector = ExitDetector(config_path)
    global _drawing_p1, _drawing_p2
    _drawing_p1 = detector.config.p1
    _drawing_p2 = detector.config.p2

    logger.info("Interactive Drawing Instructions:")
    logger.info("  - Click and drag left mouse button to draw exit line")
    logger.info("  - Press 'S' to save line configuration to disk")
    logger.info("  - Press 'Q' or ESC to exit")

    while True:
        display_frame = frame.copy()
        if _drawing_p1 and _drawing_p2:
            p1_px = (int(_drawing_p1[0] * w), int(_drawing_p1[1] * h))
            p2_px = (int(_drawing_p2[0] * w), int(_drawing_p2[1] * h))
            cv2.line(display_frame, p1_px, p2_px, (0, 255, 255), 3)
            cv2.circle(display_frame, p1_px, 6, (0, 0, 255), -1)
            cv2.circle(display_frame, p2_px, 6, (0, 255, 0), -1)

        cv2.putText(
            display_frame,
            "Click & Drag Line | 'S': Save | 'Q': Quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow(win_name, display_frame)
        key = cv2.waitKey(30) & 0xFF
        if key in (27, ord("q")):
            break
        elif key == ord("s"):
            detector.config.p1 = _drawing_p1
            detector.config.p2 = _drawing_p2
            detector.save_config(config_path)
            cv2.putText(
                display_frame,
                "CONFIG SAVED!",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow(win_name, display_frame)
            cv2.waitKey(1000)
            break

    cap.release()
    cv2.destroyAllWindows()


def run_pipeline(video_path: str, config_path: str, display: bool = True):
    """Runs exit detection pipeline on video stream."""
    exit_detector = ExitDetector(config_path)
    cart_machine = CartStateMachine(container_id="Assembly Tray #1")
    yolo_detector = Detector(model_path=resource("rf_trained/weights.pt"))
    hotdog_tracker = HotdogTracker()
    wrap_machine = WrappingStateMachine()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return

    logger.info(f"Starting Exit Line Detection Pipeline on {video_path}")
    logger.info(f"Loaded Exit Line: p1={exit_detector.config.p1}, p2={exit_detector.config.p2}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_idx = 0
    exited_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        video_time_s = frame_idx / fps
        h, w = frame.shape[:2]

        # 1. YOLO Detections
        detections = yolo_detector.detect(frame)
        hand_bboxes = [d.bbox for d in detections if d.class_name == "hand"]

        # 2. Update Hotdog & Wrapping State
        hotdog_tracks = hotdog_tracker.update(detections, video_time_s, frame.shape)
        wrapping_events = wrap_machine.update(detections, hotdog_tracks, video_time_s)

        # Collect hotdogs that are wrapped / closing / done up to this point
        wrapped_ids = list(wrap_machine.done_ids)
        if hasattr(wrap_machine, "_states"):
            for h_id, wrap_state in wrap_machine._states.items():
                if getattr(wrap_state, "state", None) in ("closing", "done") and h_id not in wrapped_ids:
                    wrapped_ids.append(h_id)

        # If wrap_machine has no wrapped IDs yet, fallback to active hotdog tracks
        if not wrapped_ids and len(hotdog_tracks) > 0:
            wrapped_ids = [t.track_id for t in hotdog_tracks]

        # 3. Check Exit Line Crossing
        exit_event = exit_detector.check_crossing(
            hand_bboxes=hand_bboxes,
            wrapped_hotdog_ids=wrapped_ids,
            frame_width=w,
            frame_height=h,
        )

        if exit_event:
            logger.info(f"🔥 [EXIT TRIGGERED] Hotdogs Exited: {exit_event.exited_hotdog_ids}")
            exited_log.append(exit_event)
            # Reset Cart State Machine for next order prep
            cart_machine.process_event(
                CartEvent(
                    event_type=EventType.CONTAINER_REMOVED,
                    roi_id="Exit_Line_ROI",
                    metadata={"exited_hotdog_ids": exit_event.exited_hotdog_ids},
                )
            )

        # 4. Visualization Overlay
        if display:
            frame = exit_detector.draw_overlay(frame)

            # Draw Hand Bboxes
            for hb in hand_bboxes:
                cv2.rectangle(frame, (hb[0], hb[1]), (hb[2], hb[3]), (0, 255, 0), 2)

            # Draw Header Status
            cv2.rectangle(frame, (10, 10), (450, 60), (17, 24, 39), -1)
            cv2.putText(
                frame,
                f"Cart State: {cart_machine.state.value} | Exited: {len(exited_log)} orders",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Exit Line Hotdog Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Pipeline Finished. Total Exit Events Recorded: {len(exited_log)}")


def main():
    parser = argparse.ArgumentParser(description="Interactive Exit Line Detector")
    parser.add_argument("--video", type=str, default="videos/hotdog_sample.mp4", help="Path to video file")
    parser.add_argument("--config", type=str, default="config/exit_line.json", help="Path to exit line json config")
    parser.add_argument("--draw", action="store_true", help="Launch interactive mouse GUI to position exit line")
    parser.add_argument("--no-display", action="store_true", help="Run in headless mode without GUI window")

    args = parser.parse_args()

    # Find sample video if default doesn't exist
    if not Path(args.video).exists():
        video_dir = Path(resource("videos"))
        if video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            if videos:
                args.video = str(videos[0])

    if args.draw:
        run_draw_mode(args.video, args.config)
    else:
        run_pipeline(args.video, args.config, display=not args.no_display)


if __name__ == "__main__":
    main()
