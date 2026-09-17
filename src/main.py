import faulthandler
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import shutil
import time
from pathlib import Path

# `kill -USR1 <pid>` prints every thread's stack to stderr without stopping the
# run.  This pipeline stalls in ways that produce no log line at all -- a capture
# thread waiting on a decoder, the KDS reader held behind the master clock -- and
# without this the only way to see where is to kill it and guess.
if hasattr(signal, "SIGUSR1"):
    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (RuntimeError, ValueError):  # no stderr to write to, or unsupported
        pass


class _StallWatchdog:
    """Dump every thread's stack when the main loop stops advancing.

    This pipeline has stalled twice in ways that produced no log line at all --
    a blocked decoder seek, and a clip writer released under another thread --
    and a silent process is indistinguishable from a slow one until hours have
    been lost.  The watchdog turns that into a stack trace in the run log, at
    the moment it happens, without needing anyone to be watching.

    It only reports; it never kills the run.
    """

    def __init__(self, timeout_s: float = 180.0):
        self.timeout_s = float(timeout_s)
        self._beat = time.monotonic()
        self._reported = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="stall-watchdog", daemon=True
        )

    def start(self) -> "_StallWatchdog":
        self._thread.start()
        return self

    def beat(self) -> None:
        """Called once per processed frame: the loop is alive."""
        self._beat = time.monotonic()
        self._reported = False

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(10.0):
            idle = time.monotonic() - self._beat
            if idle < self.timeout_s or self._reported:
                continue
            self._reported = True   # once per stall, not every 10s
            logging.getLogger(__name__).error(
                "STALL: the main loop has not advanced a frame in %.0fs. "
                "Thread stacks follow.", idle,
            )
            try:
                faulthandler.dump_traceback(all_threads=True)
            except Exception:
                logging.getLogger(__name__).exception("could not dump stacks")

import cv2
import numpy as np
import uvicorn
import yaml

from src.video.capture import VideoCaptureThread
from src import dashboard
from src.ui.dashboard import add_event, app, update_frame_data, cart_machine
from src.inference.detector import Detector
from src.analysis.flow import OpticalFlowAnalyzer
from src.domain.paths import resource
from src.domain.schemas import (
    HAND_CLASS, SAUCE_CLASSES, Action, Detection, OrderStatus,
    canonical_ingredient,
)
from src.analysis.state_machine import OrderStateMachine
from src.analysis.hotdog_tracker import (
    HotdogTracker,
    _shrink_hand_bbox,
    _hand_working_point,
)
# ── Wrapping-state order-completion module (additive — do not remove) ──────────
from src.analysis.wrapping_state import WrappingStateMachine
from src.analysis.temporal import TemporalTracker
# ── Cheese pre-gate (additive — do not remove) ────────────────────────────────
from src.analysis.cheese_gate import CHEESE_KEYS, CheesePreGate
from src.domain.zones import ZoneManager
from src.video.video_recorder import recorder_from_env

metrics_logger = logging.getLogger("src.metrics")
logger = logging.getLogger(__name__)


def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


def _boxes_overlap(bbox_a, bbox_b) -> bool:
    """Return True when two bounding boxes (x1,y1,x2,y2) intersect."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _poly_centroid(det):
    """Return (cx, cy) using the segmentation polygon centroid when available,
    falling back to the bbox midpoint."""
    poly = getattr(det, "polygon", None)
    if poly is not None and len(poly) >= 3:
        arr = np.array(poly, dtype=np.float32)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())
    x1, y1, x2, y2 = det.bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _poly_or_box_overlap(det_a, det_b) -> bool:
    """True if polygon contours intersect (pixel-accurate when both have masks),
    else falls back to bbox overlap."""
    poly_a = getattr(det_a, "polygon", None)
    poly_b = getattr(det_b, "polygon", None)
    if (
        poly_a is not None and len(poly_a) >= 3
        and poly_b is not None and len(poly_b) >= 3
    ):
        try:
            ax1, ay1, ax2, ay2 = det_a.bbox
            bx1, by1, bx2, by2 = det_b.bbox
            x_min = max(0, min(ax1, bx1))
            y_min = max(0, min(ay1, by1))
            x_max = max(ax2, bx2)
            y_max = max(ay2, by2)
            W = max(x_max - x_min + 2, 2)
            H = max(y_max - y_min + 2, 2)
            pts_a = np.array(
                [(int(p[0]) - x_min, int(p[1]) - y_min) for p in poly_a], dtype=np.int32
            )
            pts_b = np.array(
                [(int(p[0]) - x_min, int(p[1]) - y_min) for p in poly_b], dtype=np.int32
            )
            m_a = np.zeros((H, W), dtype=np.uint8)
            m_b = np.zeros((H, W), dtype=np.uint8)
            cv2.fillPoly(m_a, [pts_a], 1)
            cv2.fillPoly(m_b, [pts_b], 1)
            return bool(np.any(m_a & m_b))
        except Exception:
            pass
    return _boxes_overlap(det_a.bbox, det_b.bbox)


# Per-class bounding box colors (BGR)
CLASS_COLORS = {
    "hand": (0, 255, 0),                 # Green
    "hot-dog": (0, 165, 255),            # Orange
    "ketchup_sauce": (0, 0, 255),        # Red
    "yellow_mustard_sauce": (0, 255, 255), # Yellow
    "burger_bun": (255, 200, 0),         # Light Blue
    "french_fries": (255, 255, 0),       # Cyan
    "diced_onions": (255, 255, 255),     # White
    "grated_yellow_cheese": (0, 200, 255), # Gold
    "pickle_spears": (0, 128, 0),        # Dark Green
    "pickle_rounds": (144, 238, 144),    # Light Green
    "sport_peppers": (0, 255, 127),      # Spring Green
    "chilli": (0, 69, 139),              # Dark Brown/Red
    "tomato": (71, 99, 255),             # Tomato Red
    "relish": (0, 100, 0),               # Deep Green
    "knife": (192, 192, 192),            # Silver/Gray
    "wrapping": (200, 200, 200),         # Light Gray
}
DEFAULT_BBOX_COLOR = (0, 255, 255)


def _wrap_text(text, font, scale, thickness, max_width):
    """Split text into lines that fit within max_width pixels."""
    if not text:
        return [text]
    words = text.split(" ")
    lines = []
    current = words[0]
    for word in words[1:]:
        test = f"{current} {word}"
        if cv2.getTextSize(test, font, scale, thickness)[0][0] <= max_width:
            current = test
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


import random
import time as _time_module
from collections import deque
from typing import Tuple

_TRACK_COLORS: dict = {}

# Standalone trail buffer: track_id -> canonical trail key (stable)
# Completely independent of HotdogTracker — driven by raw YOLO track_id each frame.
#
# _TRAIL_KEY maps every seen track_id -> a stable "canonical key".
# When a new track_id appears near a recently-lost trail, it inherits that
# canonical key (and therefore that trail + color), so the line continues.
_TRAIL_BUFFER: dict = {}      # canonical_key -> deque[dict]
_TRAIL_LAST_SEEN: dict = {}   # canonical_key -> float (video time of last detection)
_TRAIL_LAST_POS: dict = {}    # canonical_key -> (cx, cy)  last known centroid
_TRAIL_KEY: dict = {}         # track_id -> canonical_key
_TRAIL_MAXLEN = 1200          # ~50 seconds at 24 fps (persists whole assembly)
_TRAIL_FADE_S = 10.0          # seconds to hold & fade trail after disappearance
_TRAIL_DEADBAND_PX = 2        # skip jitter < 2px
_TRAIL_LINK_DIST_PX = 400     # max centroid distance to inherit an existing trail (allows hand transfer across frame)
_TRAIL_LINK_TIME_S = 8.0      # inherit trails lost within last 8 seconds (covers hand occlusion & transfer dwell)


def _hsv_to_bgr(h_deg: float, s: float = 0.95, v: float = 1.0) -> Tuple[int, int, int]:
    """Convert HSV color (Hue: 0..360, Saturation: 0..1, Value: 0..1) to OpenCV BGR."""
    hsv_pixel = np.uint8([[[int((h_deg % 360.0) / 2.0), int(s * 255), int(v * 255)]]])
    bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


def _get_track_color(track_key) -> Tuple[int, int, int]:
    key_str = str(track_key)
    if key_str not in _TRACK_COLORS:
        rng = random.Random(key_str)
        hue = rng.uniform(0.0, 360.0)
        _TRACK_COLORS[key_str] = _hsv_to_bgr(hue, s=0.95, v=1.0)
    return _TRACK_COLORS[key_str]


def _find_nearby_canonical(cx: float, cy: float, current_time: float, exclude_keys: set):
    """Return the canonical key of the nearest recently-lost trail within linking range, or None."""
    best_key = None
    best_dist2 = _TRAIL_LINK_DIST_PX ** 2
    for ckey, pos in _TRAIL_LAST_POS.items():
        if ckey in exclude_keys:
            continue
        last_t = _TRAIL_LAST_SEEN.get(ckey, 0.0)
        if (current_time - last_t) > _TRAIL_LINK_TIME_S:
            continue  # Too old to link
        dx = cx - pos[0]
        dy = cy - pos[1]
        d2 = dx * dx + dy * dy
        if d2 < best_dist2:
            best_dist2 = d2
            best_key = ckey
    return best_key


def _update_trail_buffer(detections, current_time: float):
    """Feed this frame's hotdog detections into _TRAIL_BUFFER with trail continuity."""
    done_ids = getattr(draw_annotations, "_done_ids", set())
    detector_id_map = getattr(draw_annotations, "_detector_id_map", {})

    def _is_done(tid):
        if not done_ids:
            return False
        if tid in done_ids:
            return True
        mono = detector_id_map.get(tid)
        if mono is not None and mono in done_ids:
            return True
        ckey = _TRAIL_KEY.get(tid)
        if ckey is not None and (ckey in done_ids or detector_id_map.get(ckey) in done_ids):
            return True
        return False

    # Immediately purge any canonical trail keys associated with DONE hotdogs
    stale_done_keys = set()
    for ckey in list(_TRAIL_BUFFER.keys()):
        if _is_done(ckey):
            stale_done_keys.add(ckey)
    for tid, ckey in list(_TRAIL_KEY.items()):
        if _is_done(tid):
            stale_done_keys.add(ckey)

    for ckey in stale_done_keys:
        _TRAIL_BUFFER.pop(ckey, None)
        _TRAIL_LAST_SEEN.pop(ckey, None)
        _TRAIL_LAST_POS.pop(ckey, None)

    for tid in [t for t, k in list(_TRAIL_KEY.items()) if k in stale_done_keys or _is_done(t)]:
        _TRAIL_KEY.pop(tid, None)

    # Collect canonical keys active THIS frame so two detections can't both inherit the same trail
    active_canonical_this_frame: set = set()

    for det in detections:
        if det.class_name != "hot-dog" or det.track_id is None:
            continue
        raw_tid = det.track_id
        # Map raw detector track_id to monotonic hotdog ID if known
        tid = detector_id_map.get(raw_tid, raw_tid)
        if _is_done(tid) or _is_done(raw_tid):
            continue  # Do not record trail points for completed/DONE hotdogs

        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Resolve canonical key for this track_id
        if tid not in _TRAIL_KEY:
            # New track_id: try to inherit a nearby trail that was recently lost
            inherited = _find_nearby_canonical(cx, cy, current_time, active_canonical_this_frame)
            if inherited is not None and not _is_done(inherited):
                # Continue that trail under the new track_id
                _TRAIL_KEY[tid] = inherited
            else:
                # Brand new location — start a fresh trail keyed by first track_id seen there
                _TRAIL_KEY[tid] = tid
                _TRAIL_BUFFER[tid] = deque(maxlen=_TRAIL_MAXLEN)

        ckey = _TRAIL_KEY[tid]
        if _is_done(ckey):
            continue

        active_canonical_this_frame.add(ckey)

        # Ensure buffer exists (edge case: inherited key might not exist yet)
        if ckey not in _TRAIL_BUFFER:
            _TRAIL_BUFFER[ckey] = deque(maxlen=_TRAIL_MAXLEN)

        buf = _TRAIL_BUFFER[ckey]
        # Deadband: skip tiny jitter movements
        if buf:
            last = buf[-1]
            dx, dy = cx - last["x"], cy - last["y"]
            if (dx * dx + dy * dy) < _TRAIL_DEADBAND_PX ** 2:
                _TRAIL_LAST_SEEN[ckey] = current_time
                _TRAIL_LAST_POS[ckey] = (cx, cy)
                continue

        buf.append({"x": cx, "y": cy, "t": current_time})
        _TRAIL_LAST_SEEN[ckey] = current_time
        _TRAIL_LAST_POS[ckey] = (cx, cy)

    # Purge canonical trails gone > _TRAIL_FADE_S seconds or marked DONE
    stale_keys = [
        ckey for ckey, t in _TRAIL_LAST_SEEN.items()
        if (current_time - t) > _TRAIL_FADE_S or _is_done(ckey)
    ]
    for ckey in stale_keys:
        _TRAIL_BUFFER.pop(ckey, None)
        _TRAIL_LAST_SEEN.pop(ckey, None)
        _TRAIL_LAST_POS.pop(ckey, None)
    # Clean up track_id -> key mappings for stale canonical keys
    stale_set = set(stale_keys)
    for tid in [t for t, k in list(_TRAIL_KEY.items()) if k in stale_set or _is_done(t)]:
        _TRAIL_KEY.pop(tid, None)

def _draw_dashed_rect(img, pt1, pt2, color, thickness=1, style='dotted'):
    x1, y1 = pt1
    x2, y2 = pt2
    points = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1))
    ]
    dash_len = 8 if style == 'dashed' else 4
    for p1, p2 in points:
        dist = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
        dashes = int(dist / dash_len)
        for i in range(dashes):
            if i % 2 == 0:
                start_x = int(p1[0] + (p2[0] - p1[0]) * i / dashes)
                start_y = int(p1[1] + (p2[1] - p1[1]) * i / dashes)
                end_x = int(p1[0] + (p2[0] - p1[0]) * (i + 1) / dashes)
                end_y = int(p1[1] + (p2[1] - p1[1]) * (i + 1) / dashes)
                cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

def draw_annotations(frame, detections, zones, current_order):
    h, w = frame.shape[:2]
    
    # (System ROI drawing removed per user request)───────────────────────────────────────────────────────────────────────
    
    # Only the ROIs THIS ticket needs.  Drawing all twenty bins labelled every
    # frame buried the two or three the crew actually has to touch, and a zone
    # the ticket never mentions is one the pipeline already refuses to credit
    # (see the required_counts gate in OrderStateMachine.on_action).  Non-bin
    # zones -- assembly, the sauce vessel, the cheese region -- are always
    # drawn: they are the workspace, not ingredients.
    _required = set((getattr(current_order, "required_counts", None) or {}))
    for zone in zones.get_all():
        if zone.zone_type == "bin" and _required:
            if canonical_ingredient(zone.name) not in _required:
                continue
        poly = [(int(p[0] * w), int(p[1] * h)) for p in zone.polygon]
        color = hex_to_bgr(zone.color)
        cv2.polylines(frame, [np.array(poly)], True, color, 2)
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        zone_w = max(xs) - min(xs)
        cx = sum(xs) // len(xs)
        cy = sum(ys) // len(ys)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thickness = 2
        max_text_w = max(zone_w - 8, 20)
        lines = _wrap_text(zone.name, font, scale, thickness, max_text_w)
        line_h = cv2.getTextSize("Ay", font, scale, thickness)[0][1]
        pad = 6
        total_h = len(lines) * (line_h + pad)
        start_y = cy - total_h // 2 + line_h
        for i, line in enumerate(lines):
            tw = cv2.getTextSize(line, font, scale, thickness)[0][0]
            lx = cx - tw // 2
            ly = start_y + i * (line_h + pad)
            cv2.rectangle(
                frame, (lx - 2, ly - line_h - 2), (lx + tw + 2, ly + 4), (0, 0, 0), -1
            )
            cv2.putText(frame, line, (lx, ly), font, scale, color, thickness)

    hotdog_log = getattr(draw_annotations, "_hotdog_log_ref", {})

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        color = CLASS_COLORS.get(det.class_name, DEFAULT_BBOX_COLOR)
        poly = getattr(det, "polygon", None)
        has_poly = poly is not None and len(poly) >= 3

        if det.class_name in ("hot-dog", "wrapped"):
            label_text = "Hotdog"

            # ── Draw filled segmentation mask + orange contour (no bounding box) ──
            seg_color   = (0, 140, 255)   # Vibrant orange (BGR)
            fill_alpha  = 0.30

            if has_poly:
                pts = np.array(poly, dtype=np.int32)
                # Semi-transparent fill
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], seg_color)
                cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
                # Solid contour
                cv2.polylines(frame, [pts], isClosed=True, color=seg_color, thickness=2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.60
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            lbl_y1 = max(0, y1 - text_h - 8)
            lbl_y2 = y1
            cv2.rectangle(frame, (x1, lbl_y1), (x1 + text_w + 10, lbl_y2), (0, 100, 220), -1)
            cv2.rectangle(frame, (x1, lbl_y1), (x1 + text_w + 10, lbl_y2), seg_color, 1)

            cv2.putText(
                frame,
                label_text,
                (x1 + 5, lbl_y2 - 3),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )
        else:
            # ── All other classes: filled mask + colored contour ──────────────
            fill_alpha = 0.25
            if has_poly:
                pts = np.array(poly, dtype=np.int32)
                overlay = frame.copy()
                cv2.fillPoly(overlay, [pts], color)
                cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
                cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)

            label_text = det.class_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.50
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)

            lbl_y1 = max(0, y1 - text_h - 10)
            lbl_y2 = max(text_h + 10, y1)
            cv2.rectangle(frame, (x1, lbl_y1), (x1 + text_w + 10, lbl_y2), (0, 0, 0), -1)
            cv2.rectangle(frame, (x1, lbl_y1), (x1 + text_w + 10, lbl_y2), color, 1)

            cv2.putText(
                frame,
                label_text,
                (x1 + 5, lbl_y2 - 5),
                font,
                font_scale,
                color,
                thickness,
            )

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 5, color, -1)

    # Removed bottom status bar for added ingredients as requested
    return frame


# ── Exit-line tripwire module (additive — do not remove) ─────────────────────
from src.analysis.exit_detector import ExitDetector  # noqa: E402

_main_exit_detector = ExitDetector(resource("config/exit_line.json"))

def _draw_exit_line_overlays(
    frame: np.ndarray,
    hand_detections: list,
    wrapping_sm,
    hotdog_tracker,
) -> np.ndarray:
    """
    Draw exit tripwire line and check hand crossing for outgoing wrapped hotdogs.
    Called in main loop so exit line appears in all video runs and live dashboard.
    """
    h, w = frame.shape[:2]
    cfg_path = Path(resource("config/exit_line.json"))
    if cfg_path.exists():
        try:
            mtime = cfg_path.stat().st_mtime
            if getattr(_draw_exit_line_overlays, "_last_mtime", 0) != mtime:
                _draw_exit_line_overlays._last_mtime = mtime
                _main_exit_detector.load_config(str(cfg_path))
        except Exception:
            pass

    hand_bboxes = [d.bbox for d in hand_detections]
    wrapped_ids = list(wrapping_sm.done_ids)
    if hasattr(wrapping_sm, "_states"):
        for h_id, wrap_state in wrapping_sm._states.items():
            if getattr(wrap_state, "state", None) in ("closing", "done") and h_id not in wrapped_ids:
                wrapped_ids.append(h_id)

    # Monotonic mapping
    wrapped_mono_ids = []
    for wid in wrapped_ids:
        _mono = hotdog_tracker._detector_id_map.get(wid)
        if _mono is not None and _mono not in wrapped_mono_ids:
            wrapped_mono_ids.append(_mono)

    # The hotdogs themselves, so the exit can be decided on the thing that is
    # actually leaving rather than on a hand that happened to reach across.
    _hd_boxes = []
    for det in detections:
        if det.class_name not in ("hot-dog", "wrapped") or det.track_id is None:
            continue
        _m = hotdog_tracker._detector_id_map.get(det.track_id, det.track_id)
        if _m is not None:
            _hd_boxes.append((_m, tuple(int(v) for v in det.bbox)))

    evt = _main_exit_detector.check_crossing(
        hand_bboxes=hand_bboxes,
        wrapped_hotdog_ids=wrapped_mono_ids,
        frame_width=w,
        frame_height=h,
        hotdog_boxes=_hd_boxes,
    )

    if evt:
        logger.info(f"🔥 [EXIT LINE] Outgoing Hotdog Detected! Exited: {evt.exited_hotdog_ids}")
        from src.ui.dashboard import record_hotdog_exit
        record_hotdog_exit(evt.exited_hotdog_ids)
        add_event("hotdog_exited", zone="Exit_Line_ROI", item="hotdog_exited")

    # The crossing check above always runs -- it feeds the outgoing counters and
    # the analysis panel.  Only the DRAWING is optional: the tripwire line and
    # its "outgoing" banner sit on top of the detection feed without telling you
    # anything the dashboard does not already show.  SHOW_EXIT_OVERLAY=1 brings
    # them back.
    if not _SHOW_EXIT_OVERLAY:
        return frame
    return _main_exit_detector.draw_overlay(frame)


# Draw the exit tripwire and its outgoing banner on the detection feed?
# Off by default: the crossing is still detected and counted, it is just not
# painted over the camera view.
_SHOW_EXIT_OVERLAY = os.environ.get("SHOW_EXIT_OVERLAY", "0").lower() in (
    "1", "true", "yes",
)


# ── Wrapping-state on-screen overlays (additive — do not remove) ───────────────
from src.analysis.wrapping_state import STATE_CLOSING, STATE_DONE  # noqa: E402





def _env(key, default=None, cast=None):
    val = os.environ.get(key)
    if val is None:
        return default
    if cast is not None:
        return cast(val)
    return val


def main():
    log_level = _env("LOG_LEVEL", "WARNING")
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )
    logging.getLogger("src.analysis.temporal").setLevel(logging.DEBUG)
    logging.getLogger("src.metrics").setLevel(logging.INFO)
    # The ticket lifecycle -- appeared, changed, bumped, CORRECT/WRONG -- is
    # what an operator watches a run for, so it is visible at the default
    # WARNING level rather than needing LOG_LEVEL=INFO.
    logging.getLogger("src.kdsocr").setLevel(logging.INFO)

    log_path = Path("output/yolo_detections.log")
    # output/ is gitignored, so a fresh clone has no such directory and the
    # open() below raises FileNotFoundError before the pipeline ever starts.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        try:
            log_path.unlink()
        except OSError:
            pass
    with open(log_path, "w") as f:
        f.write("=== YOLO Detections Log ===\n")

    # EXIT_ON_END: when set to "true", exit cleanly after the first video ends
    # instead of looping through the video playlist.  Used by run_10min_video.py.
    exit_on_end = str(_env("EXIT_ON_END", "false")).lower() in ("1", "true", "yes")

    config = yaml.safe_load(Path(resource("config/model.yaml")).read_text())

    source = _env("VIDEO_SOURCE") or config["source"]
    model_path = _env("MODEL_PATH") or config.get("model_path", "yolov8n.pt")
    model_type = _env("MODEL_TYPE") or config.get("model_type", "yolo")
    tracker_type = _env("TRACKER_TYPE") or config.get("tracker_type", "botsort")
    confidence = _env(
        "CONFIDENCE_THRESHOLD", config.get("confidence_threshold", 0.5), float
    )
    pick_dwell = _env("PICK_DWELL_MS", config.get("pick_dwell_ms", 800), int)
    place_dwell = _env("PLACE_DWELL_MS", config.get("place_dwell_ms", 500), int)
    frame_w = _env("FRAME_WIDTH", config.get("frame_width"), int)
    frame_h = _env("FRAME_HEIGHT", config.get("frame_height"), int)
    fps = _env("FPS", config.get("fps"), int)
    realtime = str(_env("REALTIME", config.get("realtime", "false"))).lower() in (
        "1",
        "true",
        "yes",
    )
    # kds-ocr is the ONLY reader of the KDS screen. The default used to be
    # "mock", so a missing config silently swapped the real ticket source for a
    # synthetic one -- the KDS screen has one system reading it, and this is it.
    kds_mode = _env("KDS_MODE", config.get("kds_mode", "kdsocr"))

    # Window of the recording to process.  START_AT_S seeks BOTH feeds to the
    # same media offset before the first frame is decoded, so the two videos
    # keep describing the same minutes; RUN_FOR_S stops the run that many
    # seconds of video later.  0 means "from the beginning" / "to the end".
    start_at_s = max(0.0, _env("START_AT_S", config.get("start_at_s", 0.0), float))
    run_for_s = max(0.0, _env("RUN_FOR_S", config.get("run_for_s", 0.0), float))
    stop_at_s = start_at_s + run_for_s if run_for_s > 0 else None

    metrics_interval = _env("LOG_METRICS_INTERVAL", 5, int)

    metrics_logger.info(
        json.dumps(
            {
                "event": "pipeline_start",
                "source": source,
                "model_path": model_path,
                "model_type": model_type,
                "tracker_type": tracker_type,
                "confidence_threshold": confidence,
                "pick_dwell_ms": pick_dwell,
                "place_dwell_ms": place_dwell,
                "frame_width": frame_w,
                "frame_height": frame_h,
                "fps": fps,
                "kds_mode": kds_mode,
            }
        )
    )

    # ── Dashboard first ────────────────────────────────────────────────────
    # Started BEFORE the model loads.  Loading the weights takes several
    # seconds, and while it ran there was no server at all, so the browser
    # sat on a connection error that looks identical to a broken dashboard.
    # Binding first means the page is up immediately and can say what it is
    # waiting for.  Every endpoint already tolerates the pipeline objects
    # not existing yet.
    dashboard_port = int(_env("DASHBOARD_PORT", 8000))

    # Fail loudly if the port is taken.  uvicorn only *logs* a bind error and
    # the pipeline would carry on processing video with no dashboard, while the
    # browser keeps showing the stale page served by the older process -- which
    # looks exactly like a broken feed rather than a second instance.
    _probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # SO_REUSEADDR, matching what uvicorn itself does when it binds.  Without
    # it a socket left in TIME_WAIT by the PREVIOUS run reads as "in use" for
    # about a minute, so relaunching promptly after a run refused to start and
    # blamed a second instance that was not there.
    _probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        _probe.bind(("0.0.0.0", dashboard_port))
    except OSError:
        raise SystemExit(
            "Dashboard port %d is already in use.\n"
            "Another run of this pipeline is probably still going: whatever you\n"
            "see at http://localhost:%d would be that older process, not this one.\n"
            "Stop it first, or start this one with DASHBOARD_PORT=<other port>."
            % (dashboard_port, dashboard_port)
        )
    finally:
        _probe.close()

    dashboard_thread = threading.Thread(
        target=lambda: uvicorn.run(
            app, host="0.0.0.0", port=dashboard_port, log_level="warning"
        ),
        daemon=True,
    )
    dashboard_thread.start()
    # Give uvicorn time to bind and start serving before we process video frames.
    # Without this pause the entire short video can finish before the server is ready.
    time.sleep(2.0)
    logger.info("Dashboard serving on http://localhost:%d", dashboard_port)

    # Per-class gates, shared by whichever backend runs.
    _class_conf = {
        "wrapping": float(_env("WRAPPING_CONF_THRESHOLD",
                               config.get("wrapping_conf_threshold", 0.15))),
        "hot-dog": float(_env("HOTDOG_CONF_THRESHOLD",
                              config.get("hotdog_conf_threshold", 0.55))),
        "knife": 0.80,
    }

    # DEEPSTREAM_DETECTIONS=<file.jsonl> replays detections produced by
    # DeepStream inside its container instead of running the model here.  Only
    # the detector changes: the KDS reader, the state machines and the
    # dashboard are identical either way, which is the point of keeping the
    # swap at this one seam.
    _ds_path = str(_env("DEEPSTREAM_DETECTIONS", "")).strip()
    if _ds_path:
        from src.inference.deepstream_replay import DeepStreamReplayDetector

        _labels = [
            l.strip() for l in
            Path(_env("DEEPSTREAM_LABELS", "deepstream_test/labels.txt")).read_text().splitlines()
            if l.strip()
        ]
        detector = DeepStreamReplayDetector(
            _ds_path, labels=_labels, class_conf_overrides=_class_conf,
            max_skew_s=float(_env("DEEPSTREAM_MAX_SKEW_S", 0.05, float)),
        )
        logger.info("Inference backend: DeepStream replay (%s), %d classes",
                    _ds_path, len(_labels))
    else:
        detector = Detector(
            model_path,
            secondary_model_path=config.get("secondary_model_path"),
            prompt_classes=config.get("prompt_classes"),
            model_type=model_type,
            tracker_type=tracker_type,
            tracker_config=resource("config/tracker.yaml"),
            # The same per-class gates the DeepStream branch uses, so the two
            # backends cannot drift apart.
            class_conf_overrides=_class_conf,
            # Jetson GPU knobs.  half -> FP16 (auto-disabled if CUDA is absent);
            # imgsz -> inference resolution.  Both overridable per-run with the
            # HALF / IMGSZ env vars, or persistently in config/model.yaml.
            half=str(_env("HALF", config.get("half", True))).lower()
            in ("1", "true", "yes"),
            imgsz=_env("IMGSZ", config.get("imgsz"), int),
            )
        logger.info("Inference backend: %s (%s)", model_type, model_path)
    zones = ZoneManager(resource("config/zones.json"))

    tracker_config = yaml.safe_load(Path(resource("config/tracker.yaml")).read_text())
    flow_config = tracker_config.get("optical_flow", {})
    flow_enabled_env = _env("OPTICAL_FLOW_ENABLED")
    if flow_enabled_env is not None:
        flow_config["enabled"] = flow_enabled_env.lower() in ("true", "1", "yes")

    flow_analyzer = None
    if flow_config.get("enabled", False):
        flow_analyzer = OpticalFlowAnalyzer(
            method=flow_config.get("method", "sparse_lk"),
            max_corners=flow_config.get("max_corners", 100),
            quality_level=flow_config.get("quality_level", 0.01),
            min_distance=flow_config.get("min_distance", 7),
            direction_threshold=flow_config.get("direction_threshold", 0.5),
            magnitude_ratio_threshold=flow_config.get("magnitude_ratio_threshold", 0.3),
            flow_motion_threshold=flow_config.get("flow_motion_threshold", 2.0),
            lk_window_size=flow_config.get("lk_window_size", 21),
            lk_max_level=flow_config.get("lk_max_level", 3),
        )

    temporal = TemporalTracker(
        pick_dwell_ms=pick_dwell,
        place_dwell_ms=place_dwell,
        transition_timeout_ms=config.get("transition_timeout_ms", 2000),
        carry_timeout_ms=config.get("carry_timeout_ms", 5000),
        max_speed=config.get("max_speed", 2.0),
        orphan_timeout_s=config.get("orphan_timeout_s", 5.0),
        dedup_timeout_s=config.get("dedup_timeout_s", 0.3),
        dedup_distance=config.get("dedup_distance", 0.05),
        co_motion_dwell_ms=flow_config.get("co_motion_dwell_ms", 200),
        flow_contact_threshold=flow_config.get("flow_contact_threshold", 2),
    )
    trail_cfg = tracker_config.get("trajectory_trail", {})
    cfg_track_buffer = float(_env("HOTDOG_TRACK_BUFFER", tracker_config.get("track_buffer", 300)))
    cfg_match_thresh = float(_env("HOTDOG_MATCH_THRESH", tracker_config.get("match_thresh", 0.5)))

    # Convert track_buffer (frames) to orphan_timeout_s (seconds at ~30 FPS)
    orphan_timeout_s = cfg_track_buffer / float(fps if fps else 30.0)

    # ── Hotdog tracker (additive, does not modify core pipeline) ──────────

    # Build wrap-zone polygon in PIXEL space for HotdogTracker.
    # zones.json stores normalised [0-1] coordinates; HotdogTracker uses pixel
    # bboxes, so we must scale by (frame_w, frame_h).
    # _point_in_wrap_zone() was always returning False before this because
    # wrap_zone_poly was never passed — meaning the 20 s coast buffer, 450 px
    # spatial lock, and stall watchdog were completely inactive.
    _wrap_zone_poly_px = None
    _assembly_zones = zones.get_zones_by_type("assembly")
    if _assembly_zones:
        _az = _assembly_zones[0]  # use first (only) assembly zone
        _wrap_zone_poly_px = [
            (px * frame_w, py * frame_h)
            for px, py in _az.polygon
        ]
        logger.info(
            "[INIT] Wrap-zone polygon wired: %d vertices  zone=%r  "
            "frame=%dx%d",
            len(_wrap_zone_poly_px), _az.name, frame_w, frame_h,
        )
    else:
        logger.warning(
            "[INIT] No assembly zone found in zones.json — "
            "wrap-zone overrides (20 s coast, 450 px lock) will NOT activate."
        )

    # Read wrap_station overrides from tracker.yaml
    _wrap_cfg = tracker_config.get("wrap_station", {})
    _wrap_spatial_lock  = float(_wrap_cfg.get("spatial_lock_radius",   450.0))
    _wrap_orphan_s      = float(_wrap_cfg.get("occlusion_buffer_s",     20.0))
    _wrap_stall_s       = float(_wrap_cfg.get("stall_watchdog_window_s", 20.0))
    _wrap_reid_enabled  = bool(_wrap_cfg.get("hand_transit_reid_enabled", _wrap_cfg.get("appearance_reid_enabled", True)))
    _wrap_reid_gap_s    = float(_wrap_cfg.get("hand_transit_max_gap_s", 60.0))
    _wrap_hand_speed    = float(_wrap_cfg.get("max_hand_speed_px_per_frame", 55.0))
    _wrap_hand_timeout  = int(_wrap_cfg.get("hand_lost_timeout_frames", 180))
    _wrap_hand_window   = int(_wrap_cfg.get("hand_assoc_frame_window", 5))
    _wrap_hand_disp     = float(_wrap_cfg.get("base_hand_displacement_px", 300.0))

    hotdog_tracker = HotdogTracker(
        proximity_pad=int(_env("HOTDOG_PROXIMITY_PAD", 30)),
        orphan_timeout_s=float(_env("HOTDOG_ORPHAN_TIMEOUT", orphan_timeout_s)),
        spatial_lock_radius=float(_env("HOTDOG_SPATIAL_LOCK_RADIUS", 300.0)),
        item_dwell_s=float(_env("HOTDOG_ITEM_DWELL_S",
                                config.get("hotdog_item_dwell_s", 0.5))),
        min_sauce_aspect=float(_env("HOTDOG_MIN_SAUCE_ASPECT",
                                    config.get("hotdog_min_sauce_aspect_ratio", 1.2))),
        min_sauce_height=int(_env("HOTDOG_MIN_SAUCE_HEIGHT_PX",
                                   config.get("hotdog_min_sauce_height_px", 40))),
        min_sauce_confidence=float(_env("HOTDOG_MIN_SAUCE_CONFIDENCE",
                                        config.get("hotdog_min_sauce_confidence", 0.60))),
        max_sauce_instances=int(_env("HOTDOG_MAX_SAUCE_INSTANCES",
                                     config.get("hotdog_max_sauce_instances", 1))),
        require_hand_proximity=bool(_env("HOTDOG_REQUIRE_HAND_PROXIMITY",
                                         config.get("hotdog_require_hand_proximity", True))),
        iou_threshold=cfg_match_thresh,
        trail_maxlen=int(_env("TRAIL_MAXLEN", trail_cfg.get("maxlen", 60))),
        trail_smooth_alpha=float(_env("TRAIL_SMOOTH_ALPHA", trail_cfg.get("smooth_alpha", 0.35))),
        trail_anchor=_env("TRAIL_ANCHOR", trail_cfg.get("anchor", "bottom_center")),
        retain_lost_trails=str(_env("RETAIN_LOST_TRAILS", trail_cfg.get("retain_lost_trails", True))).lower() in ("true", "1", "yes"),
        # ── Wrap-zone ROI overrides ──────────────────────────────────────────
        wrap_zone_poly=_wrap_zone_poly_px,           # pixel-space polygon
        wrap_spatial_lock_radius=_wrap_spatial_lock, # 450 px inside wrap zone
        wrap_orphan_timeout_s=_wrap_orphan_s,        # 20 s coast in wrap zone
        wrap_stall_watchdog_s=_wrap_stall_s,         # 20 s merge window
        hand_transit_reid_enabled=_wrap_reid_enabled,
        hand_transit_max_gap_s=_wrap_reid_gap_s,
        max_hand_speed_px_per_frame=_wrap_hand_speed,
        hand_lost_timeout_frames=_wrap_hand_timeout,
        hand_assoc_frame_window=_wrap_hand_window,
        base_hand_displacement_px=_wrap_hand_disp,
    )

    # Stop once this many tickets have been judged (0 = run the whole video).
    # A ticket is judged when it leaves the KDS, so this samples whole orders
    # from the front of a recording rather than a fixed number of minutes.
    max_tickets = int(_env("MAX_TICKETS", config.get("max_tickets", 0)))
    _tickets_judged = []                      # one entry per verdict, for the log
    _ticket_limit_reached = threading.Event()  # set on the KDS reader thread

    # Tickets the KDS has judged, handed from the KDS reader thread to the main
    # loop.  The order state machine owns the checklist panels and must let go
    # of a ticket once the KDS has finished with it, or the dashboard keeps
    # showing the finished order's items while the next ticket is already up.
    _kds_judged_tickets: list = []
    _kds_judged_lock = threading.Lock()
    # The KDS screen is read by the kds-ocr project running as a child
    # process (see src/kdsocr/).
    kdsocr = None
    kds = None
    kds_preview = None
    if kds_mode in ("kdsocr", "kds-ocr", "video"):
        from src.kdsocr import KdsOcrClient, ReaderConfig

        kds_source = os.getenv("KDS_SOURCE") or config.get("kds_source")
        if not kds_source:
            raise SystemExit(
                "kds_mode is %r but no KDS source was given. "
                "Set KDS_SOURCE=<path-or-rtsp> or kds_source: in config/model.yaml"
                % kds_mode
            )
        _is_rtsp = str(kds_source).lower().startswith(("rtsp://", "rtsps://"))
        _kds_replay = str(_env("KDS_RECIPES", "") or "").strip()

        # Anchor the KDS pacing on THIS video's wall-clock start, so the 1 s
        # difference between the two recordings does not become a standing
        # skew between the ticket and the food.
        from src.kdsocr.clock import video_start_from_filename

        _master_start = None
        _can_pace = False
        if not _is_rtsp:
            if isinstance(source, str):
                _master_start = video_start_from_filename(os.path.basename(source))
                if _master_start is None:
                    logger.info(
                        "no wall-clock in the production filename %r; pacing "
                        "the KDS feed on its own clock instead",
                        os.path.basename(source))
            # Can we hold each emission back ourselves until this video reaches
            # the minute the ticket was on screen?  If so the child may read as
            # fast as it likes and we release on schedule, which decouples the
            # two rates entirely.  `--realtime` is only the fallback for when
            # we cannot pace: its 1x replay would starve a production loop
            # running faster than real time.
            _can_pace = (
                _master_start is not None
                or video_start_from_filename(os.path.basename(str(kds_source)))
                is not None
            )
        reader_cfg = ReaderConfig(
            repo=_env("KDSOCR_REPO", config.get("kdsocr_repo", "kds-ocr")),
            videos=[] if _is_rtsp else [str(kds_source)],
            rtsp=str(kds_source) if _is_rtsp else "",
            live_seconds=float(_env("KDSOCR_LIVE_SECONDS",
                                    config.get("kdsocr_live_seconds", 0)) or 0),
            out_dir=config.get("kdsocr_out", "output/kdsocr"),
            # KDS_RECIPES=<path> replays an existing kds-ocr recipe stream
            # instead of reading the screen again. Same source either way --
            # kds-ocr's own JSON -- but a replay is identical on every run, so
            # a vision change can be measured against a KDS side that did not
            # move underneath it.
            recipes_path=(_kds_replay
                          or config.get("kdsocr_recipes", "output/kdsocr/recipes.jsonl")),
            replay=bool(_kds_replay),
            gpu=str(_env("KDSOCR_GPU", config.get("kdsocr_gpu", True))).lower()
                not in ("false", "0", "no"),
            realtime=bool(realtime) or (not _is_rtsp and not _can_pace),
            # Both feeds must be cut at the same offset or they no longer
            # describe the same minutes.
            start_at=0.0 if _is_rtsp else float(start_at_s or 0.0),
            log_path=config.get("kdsocr_log", "output/kdsocr/reader.log"),
        )
        kdsocr = KdsOcrClient(
            reader_cfg,
            journey_path=config.get("ticket_journeys", "output/ticket_journeys.jsonl"),
            ingredient_config=config.get("kdsocr_ingredients"),
            master_start=_master_start,
        )
        kds = kdsocr

        # A picture of the KDS screen for the dashboard panel.  kds-ocr reads
        # the feed in its own process, so without this the panel has ticket
        # state but no image.  Purely a VIEW -- nothing is parsed from it and
        # no decision depends on it; the tickets come from the emissions.
        if str(_env("KDS_PREVIEW", config.get("kds_preview", True))).lower() \
                not in ("false", "0", "no"):
            from src.kdsocr.preview import KdsPreview

            _kds_start = video_start_from_filename(os.path.basename(str(kds_source)))
            # The KDS media time matching production media time 0: the two
            # recordings do not necessarily start at the same instant.
            _offset = 0.0
            if _master_start is not None and _kds_start is not None:
                _offset = (_master_start - _kds_start).total_seconds()
            kds_preview = KdsPreview(
                str(kds_source),
                on_frame=dashboard.update_kds_frame,
                fps=float(_env("KDS_PREVIEW_FPS", config.get("kds_preview_fps", 3.0))),
                live=_is_rtsp,
                start_at_s=0.0 if _is_rtsp else float(start_at_s or 0.0),
                offset_s=_offset,
            )
            kds_preview.start()
        logger.info("kds-ocr reading the KDS screen from %s",
                    "<rtsp>" if _is_rtsp else kds_source)
    history_path = config.get("kds_history")

    # A fresh run starts with an empty board.  Order history is rehydrated from
    # this file at construction, so without archiving it first the dashboard
    # opens showing the previous run's completed orders, its stats and its
    # checklist -- which reads as though this run had already done the work.
    if _env("FRESH_START", "0") == "1" and history_path:
        _hist = Path(history_path)
        if _hist.exists() and _hist.stat().st_size > 0:
            _archive = _hist.parent / "history"
            _archive.mkdir(parents=True, exist_ok=True)
            _stamp = time.strftime("%Y%m%d_%H%M%S")
            _moved = _archive / ("%s_%s%s" % (_hist.stem, _stamp, _hist.suffix))
            try:
                shutil.move(str(_hist), str(_moved))
                logger.info("Fresh start: previous history archived to %s", _moved)
            except OSError as exc:
                logger.warning("Could not archive %s: %s", _hist, exc)

    state_machine = OrderStateMachine(
        history_path=history_path,
    )
    state_machine.set_kds_client(kds)

    # ── Ensure clean, fresh startup (do not load old values) ────────────────
    cart_machine.reset("System start clean session")

    # Run ONLY the video specified in config/model.yaml (or VIDEO_SOURCE env var)
    video_playlist = [source]
    current_video_idx = 0

    # Hardware video decode (nvv4l2decoder) unless INGEST=opencv or the source
    # has no hardware chain; see src/gst_capture.py.
    from src.video.gst_capture import open_capture

    capture = open_capture(
        source,
        target_width=frame_w,
        target_height=frame_h,
        fps=fps,
        realtime=realtime,
        ingest=str(_env("INGEST", config.get("ingest", "gstreamer"))),
        start_at_s=start_at_s,
    )
    capture.start()
    logger.info("Video input: %s via %s", source, getattr(capture, "backend", "opencv"))

    # `dashboard` is already imported at module scope; re-importing it here
    # made it a function-local name, so any earlier use in main() -- the KDS
    # preview callback, for one -- raised UnboundLocalError.
    dashboard.state_machine = state_machine

    # Feed analysis and system stats for the dashboard's Analysis window.
    import atexit

    from src.analysis.feed_analysis import FeedAnalyzer
    from src.system_monitor import SystemMonitor

    analyzer = FeedAnalyzer(
        zones=zones,
        zones_path=resource("config/zones.json"),
        exit_config=resource("config/exit_line.json"),
        lifecycle=config.get("lifecycle"),
    )
    system_monitor = SystemMonitor().start()
    analysis_path = Path(resource("output/feed_analysis.json"))
    atexit.register(system_monitor.stop)
    atexit.register(lambda: analyzer.write_json(analysis_path))
    video_duration_s = 0.0
    if capture._is_file_source and capture.cap.get(cv2.CAP_PROP_FPS):
        video_duration_s = capture.cap.get(cv2.CAP_PROP_FRAME_COUNT) / capture.cap.get(cv2.CAP_PROP_FPS)
    if detector.backend == "onnx":
        model_label = {"tensorrt": "ONNX · TensorRT FP16", "cuda": "ONNX · CUDA",
                       "cpu": "ONNX · CPU"}.get(detector.onnx_provider, "ONNX")
    elif detector.backend == "tensorrt":
        model_label = "TensorRT engine"
    else:
        model_label = "PyTorch FP16" if detector.half else "PyTorch"
    decoder_label = {"nvv4l2decoder": "hardware decode", "opencv": "CPU decode"}.get(
        getattr(capture, "backend", "opencv"), "CPU decode")
    last_analysis_publish = 0.0
    last_analysis_write = time.time()

    prev_gray = None
    # Optional recording of the annotated feed (RECORD_VIDEO=<path>).
    recorder = recorder_from_env(fps, source)

    # Optional continuous recording of the DASHBOARD view (RECORD_DASHBOARD):
    # RECORD_DASHBOARD records the DASHBOARD ITSELF, via
    # scripts/record_dashboard.py: a hidden Xorg display, Firefox in kiosk mode
    # on the page, captured with GStreamer straight into the Jetson's hardware
    # H.264 encoder.  The file is the browser window, not a second rendering of
    # it that could drift from what the page shows.  Separate from
    # RECORD_VIDEO, which records the annotated production feed alone.
    dashboard_recorder = None
    _dash_record_path = _env("RECORD_DASHBOARD", "")
    if str(_dash_record_path).strip():
        from src.video.video_recorder import default_recording_path

        if str(_dash_record_path).strip().lower() in ("1", "true", "yes", "on"):
            _dash_record_path = str(
                default_recording_path("dashboard")
            ).replace(".mkv", "_dashboard.mkv")
        _dash_record_path = str(_dash_record_path)
        # Two recorders, one contract.  The Jetson's drives a hidden Xorg and
        # Firefox through GStreamer; the Windows one drives the installed
        # Chrome through Playwright.  Both take the same flags and leave the
        # same sidecar, so everything downstream -- and the clipping in
        # particular -- is the same either way.
        _rec_script = ("scripts/record_dashboard_win.py" if sys.platform == "win32"
                       else "scripts/record_dashboard.py")
        _rec_cmd = [
            sys.executable, str(resource(_rec_script)),
            "--url", "http://127.0.0.1:%d/" % int(os.environ.get("DASHBOARD_PORT", "8000")),
            "--out", _dash_record_path,
            "--size", "%dx%d" % (int(_env("RECORD_DASHBOARD_WIDTH", 1920, int)),
                                 int(_env("RECORD_DASHBOARD_HEIGHT", 1080, int))),
            # A safety net: if this process dies without running its shutdown,
            # the recorder still stops rather than running on forever.
            "--while-pid", str(os.getpid()),
        ]
        try:
            # CREATE_NEW_PROCESS_GROUP is what makes CTRL_BREAK_EVENT
            # deliverable to the recorder at shutdown; without it the call is
            # rejected and the only way to stop it would be a kill, which is
            # exactly what loses the video.
            _rec_kwargs = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                           if sys.platform == "win32"
                           else {"start_new_session": True})
            dashboard_recorder = subprocess.Popen(_rec_cmd, **_rec_kwargs)
            logger.info("Recording the dashboard window to %s (pid %d)",
                        _dash_record_path, dashboard_recorder.pid)
        except OSError:
            logger.warning("could not start the dashboard recorder", exc_info=True)
            dashboard_recorder = None
    frame_count = 0
    last_metrics_time = time.time()
    pipeline_start_time = time.time()

    _sauce_frames = {}
    _sauce_applied = {}
    _sauce_last_fired = {}
    SAUCE_COOLDOWN_S = 0.7   # balanced cooldown for fast sauce passes

    # ── Wrapping-state machine (additive — do not remove) ──────────────────────
    wrapping_sm = WrappingStateMachine(
        wrapping_dwell_s=float(_env("WRAPPING_DWELL_S", config.get("wrapping_dwell_s", 0.4))),
        min_closing_frames=int(_env("WRAPPING_MIN_CLOSING_FRAMES", config.get("wrapping_min_closing_frames", 30))),
        done_delay_s=float(_env("WRAPPING_DONE_DELAY_S", config.get("wrapping_done_delay_s", 1.0))),
        min_coverage_ratio=float(_env("WRAPPING_MIN_COVERAGE_RATIO", config.get("wrapping_min_coverage_ratio", 0.25))),
    )
    SAUCE_MIN_FRAMES = 1     # 1 frame accumulation for responsive detection

    # ── Cheese pre-gate (additive — do not remove) ─────────────────────────────
    # Cheese is fetched first on this line, routinely before the KDS ticket has
    # been confirmed and reached the head of the FIFO.  on_action() drops every
    # event while no order is in progress, so without this the one ingredient a
    # C/C turns on is the one never recorded.  The gate watches the cheese wells
    # regardless of ticket state and confirms a slice when the hand carries it
    # out of the cheese_region; takes made with no order live are buffered and
    # replayed when the ticket confirms.
    cheese_gate = CheesePreGate(
        zones,
        # 30s, not 120s: this is how far back a cheese trip can be and still be
        # credited to the ticket that follows it.  Two minutes reached back past
        # whole orders -- CHK-254 inherited a take from 68s before it existed and
        # was failed for a cheese it never touched.  Half a minute covers the
        # real case, a slice fetched while the ticket was still coming up the
        # queue, without reaching into the order before it.
        lookback_s=float(_env("CHEESE_LOOKBACK_S", config.get("cheese_lookback_s", 30.0))),
        enabled=str(_env("CHEESE_GATE", config.get("cheese_gate", True))).lower()
        not in ("false", "0", "no"),
    )
    # While the gate owns cheese, on_action must ignore the well-exit place
    # event or every slice is counted twice.
    state_machine.cheese_gate_owns_cheese = cheese_gate.enabled

    # The hotdog COUNT comes from presence on the bench, not from tracked
    # identities -- see src/analysis/hotdog_presence.py for why. HotdogTracker
    # still runs for the trails, the wrapping states and item attribution.
    from src.analysis.hotdog_presence import HotdogPresence

    hotdog_presence = HotdogPresence(
        match_px=float(_env("HOTDOG_MATCH_PX", config.get("hotdog_match_px", 140.0))),
        grace_s=float(_env("HOTDOG_GRACE_S", config.get("hotdog_grace_s", 2.0))),
        min_age_s=float(_env("HOTDOG_MIN_TRACK_AGE_S",
                             config.get("hotdog_min_track_age_s", 0.25))),
    )

    def _close_ticket_state(ref):
        # Both counters are per ticket: nothing from a finished order may be
        # counted towards the next one.
        hotdog_tracker.close_ticket(ref)
        hotdog_presence.reset(None)

    state_machine._on_ticket_closed = _close_ticket_state

    kds_image_cache = {}

    # Idle-mode tuning: how often to still run YOLO while the KDS is empty.
    # 1 (the default) disables idling altogether -- detection runs on every
    # frame whatever the KDS screen shows.  Set >1 to run 1 frame in N while
    # the screen is blank, or 0 to stop detection completely while it is.
    idle_detect_stride = int(_env("IDLE_DETECT_STRIDE", config.get("idle_detect_stride", 1)))
    idle_enabled = idle_detect_stride != 1
    _idle_state = False
    _idle_frames = 0
    _idle_t0 = 0.0          # start of the current idle stretch, for the heartbeat
    # Media time of the frame being processed.  Seeded here because the ticket
    # poll at the top of the loop runs before the first frame is read, and the
    # cheese replay below needs a clock to prune its buffer against.
    current_time = 0.0

    # Monotonic ids of hotdogs finished while the current mock group is on the
    # board.  Cleared when the group is handed over.
    _mock_made_ids: set = set()

    # Reports a silent stall instead of letting the run look merely slow.
    watchdog = _StallWatchdog(
        timeout_s=float(_env("STALL_TIMEOUT_S", config.get("stall_timeout_s", 180.0)))
    ).start()

    try:
        while True:
            loop_start = time.perf_counter()

            # MAX_TICKETS reached.  Checked here rather than in the listener so
            # the run stops between frames, with the clip for the last ticket
            # already closed by finish().
            if _ticket_limit_reached.is_set():
                logger.info(
                    "Stopping: %d ticket(s) judged (MAX_TICKETS=%d)",
                    len(_tickets_judged), max_tickets,
                )
                if state_machine.current_ticket is not None:
                    state_machine.finalize_current_order()
                break

            # Mock KDS only: no card ever disappears, so nothing would ever end
            # a group and the board would hold the first one for the whole run.
            # A group is done when the hotdogs it asked for have been made --
            # count is the signal the vision side measures directly, whereas
            # most of the named ingredients (chilli, sliced cheese) have no
            # detector class at all and would never be satisfied.
            #
            # NEVER with kds-ocr: there the ticket ends when the reader says it
            # was BUMPED.  Finalizing on hotdog count instead would judge the
            # order the moment the last dog was wrapped -- before the toppings
            # finished going on, and before any late edit to the ticket -- and
            # would silently bypass the bump trigger entirely.
            if kdsocr is None and state_machine.current_ticket is not None:
                _need = getattr(state_machine.current_ticket, "total_hotdogs", 0) or 0
                _made = len(_mock_made_ids)
                if _need > 0 and _made >= _need:
                    logger.info(
                        "Group %s complete (%d/%d hotdogs made); moving to the next group",
                        state_machine.current_ticket.ticket_id, _made, _need,
                    )
                    state_machine.finalize_current_order()
                    _mock_made_ids.clear()

            # ── kds-ocr: new reads, changed tickets, bumped tickets ─────────
            if kdsocr is not None:
                # Pace a recorded KDS feed against this video's media clock, or
                # an hour of tickets arrives before the production feed reaches
                # the food they describe. A live feed ignores this.
                kdsocr.set_master_time(current_time)
                if kds_preview is not None:
                    kds_preview.set_master_time(current_time)
                kdsocr.poll()

                # A ticket edited on the KDS while the food is being made.
                # The order in progress is re-pointed at the new requirement
                # and KEEPS what has already been observed going on.
                for _updated in kdsocr.take_updates():
                    if state_machine.update_ticket_requirements(_updated):
                        logger.info(
                            "Ticket %s changed on the KDS; requirement updated to %s",
                            _updated.ticket_id,
                            state_machine.current_order.required_counts,
                        )
                    else:
                        # Not open yet (this loop takes updates BEFORE
                        # get_next_ticket), or another ticket is being built.
                        # Hold it rather than lose it -- see requeue_update.
                        kdsocr.requeue_update(_updated.ticket_id)

                # A ticket that stopped being verifiable after we opened it
                # (voided, or every hot dog edited off it).  Released without a
                # verdict -- otherwise no bump for it is ever handed on and
                # every later ticket queues up behind it forever.
                for _ref in kdsocr.take_cancellations():
                    _cur = state_machine.current_ticket
                    if _cur is not None and _cur.ticket_id == _ref:
                        state_machine.abandon_current_order()
                        logger.info(
                            "Ticket %s dropped without a verdict: it is no "
                            "longer verifiable", _ref,
                        )

                # THE VERDICT TRIGGER.  kds-ocr tells us the ticket was bumped;
                # we do not re-derive "the card disappeared" anywhere.
                for _ref in kdsocr.take_bumps():
                    _cur = state_machine.current_ticket
                    if _cur is None or _cur.ticket_id != _ref:
                        continue
                    _order = state_machine.finalize_current_order()
                    if _order is None:
                        continue
                    kdsocr.record_verdict(
                        _ref,
                        correct=bool(_order.passed),
                        message=_order.validation_message,
                        # Only what never arrived.  A required ingredient seen
                        # fewer times than the ticket's quantity is still on
                        # the dog, and the observed number counts detections
                        # rather than applications, so a shortfall against it
                        # is not evidence of anything.  Matches the presence
                        # test in BatchOrderValidator.validate.
                        missing={k: v
                                 for k, v in _order.required_counts.items()
                                 if _order.picked_counts.get(k, 0) == 0},
                        extras=[{"item": i} for i in (_order.extra_items or [])],
                        t=current_time,
                    )
                    _tickets_judged.append(_ref)
                    if max_tickets and len(_tickets_judged) >= max_tickets:
                        _ticket_limit_reached.set()

            # A ticket the KDS has judged is over.  Release it so the poll
            # below picks up whatever is next in the queue and the checklist
            # panels follow the board instead of the bumped order.
            if _kds_judged_tickets:
                with _kds_judged_lock:
                    judged_now = list(_kds_judged_tickets)
                    _kds_judged_tickets.clear()
                current = state_machine.current_ticket
                if current is not None and current.ticket_id in judged_now:
                    logger.info(
                        "Ticket %s judged by the KDS; advancing the board to the "
                        "next ticket in the queue", current.ticket_id,
                    )
                    state_machine.finalize_current_order()

            if state_machine.current_ticket is None and kds is not None:
                ticket = kds.get_next_ticket()
                if ticket:
                    state_machine.on_kds_ticket(ticket)
                    # The ticket is confirmed now, but the cheese for it was
                    # probably fetched before it was.  Replay what the gate
                    # buffered while the board was empty.
                    replayed = state_machine.apply_cheese_takes(
                        cheese_gate.drain(current_time), pre_confirmation=True
                    )
                    if replayed:
                        logger.info(
                            "Ticket %s confirmed: %d cheese slice(s) taken before "
                            "confirmation applied to it",
                            ticket.ticket_id, replayed,
                        )

            frame_item = capture.get_frame()
            if frame_item is None:
                # Check if the video just looped (marker was received inside get_frame)
                if capture.consume_loop():
                    temporal.reset()
                    prev_gray = None
                    if state_machine.current_ticket is not None:
                        state_machine.finalize_current_order()
                    if capture._is_file_source:
                        logger.info("Video ended. Exiting.")
                        break
                time.sleep(0.01)
                continue
            frame, current_time = frame_item

            # RUN_FOR_S: stop on media time, not wall time, so the run covers
            # exactly the requested stretch of the recording however fast or
            # slow the detector happened to be.  Open tickets are finalised
            # below rather than dropped, so nothing is silently lost.
            if stop_at_s is not None and current_time >= stop_at_s:
                logger.info(
                    "Reached the end of the requested window: %.1fs -> %.1fs "
                    "(%.1f minutes of video)",
                    start_at_s, stop_at_s, (stop_at_s - start_at_s) / 60.0,
                )
                if state_machine.current_ticket is not None:
                    state_machine.finalize_current_order()
                break

            # Keep the KDS reader in step with this video.  It reads far faster
            # than the detector runs, so without this it drifts minutes ahead.
            # Force finished status 3.5 seconds before the video ends
            if capture._is_file_source:
                total_frames = capture.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame = capture.cap.get(cv2.CAP_PROP_POS_FRAMES)
                video_fps = capture.cap.get(cv2.CAP_PROP_FPS)
                if video_fps > 0 and total_frames > 0 and current_frame > 10:
                    remaining_s = (total_frames - current_frame) / video_fps
                    if remaining_s <= 0.5:
                        if state_machine.current_order:
                            state_machine.current_order.passed = True
                            state_machine.current_order.ending_soon = True

            # ── Idle mode (off by default) ─────────────────────────────────────
            # Disabled unless `idle_detect_stride` is changed from 1, because
            # skipping frames means not seeing production that happens while
            # the KDS is blank.  When it IS enabled, the gate is card presence
            # rather than payment: a card appears well before it is confirmed
            # paid, and production often starts in that window, so the detector
            # must already be running by the time the ticket activates.
            idle_now = (
                idle_enabled
                and kdsocr is not None
                and not kdsocr.has_screen_content
            )
            if idle_now != _idle_state:
                _idle_state = idle_now
                if idle_now:
                    how = (
                        "throttled to 1 frame in %d" % idle_detect_stride
                        if idle_detect_stride > 0
                        else "stopped"
                    )
                    logger.info("Detection %s - KDS screen is empty", how)
                    _idle_t0 = time.perf_counter()
                    _idle_frames = 0
                else:
                    logger.info("Detection resumed - something is on the KDS")
            skip_detection = idle_now and (
                idle_detect_stride <= 0 or (frame_count % idle_detect_stride) != 0
            )

            detect_start = time.perf_counter()
            if skip_detection:
                detections = []
                _idle_frames += 1
                if _idle_frames % 300 == 0:
                    _idle_elapsed = time.perf_counter() - _idle_t0
                    logger.info(
                        "Idle: %d frames skimmed at %.1f fps (KDS still empty)",
                        _idle_frames,
                        _idle_frames / _idle_elapsed if _idle_elapsed > 0 else 0.0,
                    )
            else:
                # The replay backend answers by media time, not by pixels.
                if hasattr(detector, "media_time"):
                    detector.media_time = current_time
                elif getattr(detector, "backend", "") == "deepstream":
                    setattr(detector, "media_time", current_time)
                detections = detector.detect(frame, conf_threshold=confidence)
            detect_ms = (time.perf_counter() - detect_start) * 1000.0

            if skip_detection:
                # Idle fast path.  Everything below this point is tracking and
                # validation work that only means something when an order is
                # live.  It is skipped wholesale rather than fed empty
                # detections -- WrappingStateMachine treats a track vanishing
                # as "done", so empty input would fire bogus completions.
                idle_view = frame.copy()
                cv2.putText(
                    idle_view,
                    "IDLE - KDS screen empty (detection off, full-speed playback)",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (180, 180, 180),
                    1,
                    cv2.LINE_AA,
                )
                if recorder is not None:
                    recorder.write(idle_view)
                update_frame_data(
                    idle_view,
                    state_machine.current_ticket,
                    state_machine.get_current_order(),
                    state_machine.get_stats(),
                    detections={},
                    hotdog_log=hotdog_tracker.get_hotdog_log(),
                )
                if kdsocr is not None:
                    dashboard.set_kds_state(kdsocr.dashboard_state())
                loop_ms = (time.perf_counter() - loop_start) * 1000.0
                frame_count += 1
                continue

            # Log YOLO detections for debugging
            with open("output/yolo_detections.log", "a") as f:
                f.write(f"Frame {frame_count}:\n")
                for d in detections:
                    f.write(f"  {d.class_name} ({d.confidence:.2f}) [ID:{d.track_id}] {d.bbox}\n")

            hand_detections = [d for d in detections if d.class_name == HAND_CLASS]

            h, w = frame.shape[:2]

            # ── Filter sauce detections for display & tracking ────────────────
            # Sauce bottles are only shown on screen (and passed to the hotdog
            # tracker) when they are in a valid position:
            #   • Inside the sauce_vessel zone  (bottle resting — shown but no event)
            #   • Inside the assembly zone       (being used — shown + event)
            #   • Overlapping a food item bbox   (being applied — shown + event)
            # Any other position (ingredient bins, right-side counter) is a
            # false positive → removed from the visible list entirely.
            _vessel_zones_vis = zones.get_zones_by_type("sauce_vessel")
            _food_bboxes_vis  = [
                d.bbox for d in detections
                if d.class_name in {"hot-dog", "burger_bun", "french_fries"}
            ]

            def _sauce_is_visible(det) -> bool:
                """Return True if this sauce detection should be shown/processed."""
                if det.class_name not in SAUCE_CLASSES:
                    return True  # non-sauce detections always visible
                # Use polygon centroid when available (more accurate than bbox midpoint)
                cx, cy = _poly_centroid(det)
                cx_n, cy_n = cx / w, cy / h
                # Allow if inside vessel zone
                if _vessel_zones_vis:
                    for vz in _vessel_zones_vis:
                        if cv2.pointPolygonTest(
                            np.array(vz.polygon, dtype=np.float32),
                            (cx_n, cy_n), False
                        ) >= 0:
                            return True
                # Allow if in assembly zone
                bz = zones.get_zone_for_bbox(
                    det.bbox, w, h, seg_polygon=getattr(det, "polygon", None)
                )
                if bz is not None and bz.zone_type == "assembly":
                    return True
                # Allow if overlapping food (polygon-accurate)
                for fd in detections:
                    if fd.class_name in {"hot-dog", "burger_bun", "french_fries"}:
                        if _poly_or_box_overlap(det, fd):
                            return True
                return False  # suppress

            visible_detections = [d for d in detections if _sauce_is_visible(d)]
            # ─────────────────────────────────────────────────────────────────

            flow_start = time.perf_counter()
            flow_signals = None
            if flow_analyzer is not None and prev_gray is not None:
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                flow_signals = {}
                for det in hand_detections:
                    zone = zones.get_zone_for_bbox(det.bbox, w, h)
                    if zone is not None and zone.zone_type == "bin":
                        # NOT `signal`: that name shadowed the `signal`
                        # MODULE for the whole of main(), so signal.SIGINT
                        # resolved to a FlowSignal and the dashboard recorder
                        # could never be stopped cleanly.
                        flow_signal = flow_analyzer.compute_flow(
                            prev_gray,
                            curr_gray,
                            det.bbox,
                            zone.polygon,
                            w,
                            h,
                        )
                        flow_signals[det.track_id] = flow_signal
                prev_gray = curr_gray
            elif flow_analyzer is not None:
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_ms = (time.perf_counter() - flow_start) * 1000.0

            requires_relish = False
            if state_machine.batch_validator:
                if "relish" in state_machine.batch_validator.required_counts:
                    requires_relish = True

            temporal_start = time.perf_counter()
            actions = temporal.update(
                hand_detections,
                zones,
                w,
                h,
                flow_signals,
                current_time=current_time,
                hotdog_detections=[d for d in detections if d.class_name == "hot-dog"],
                requires_relish=requires_relish,
            )
            temporal_ms = (time.perf_counter() - temporal_start) * 1000.0

            # ── Cheese pre-gate (additive — do not remove) ─────────────────────
            # Runs whatever the ticket state is.  A take confirmed while an order
            # is in progress is applied straight away (and dropped from the
            # buffer so the replay above cannot use it twice); one confirmed with
            # the board empty stays buffered for the ticket that follows.
            cheese_takes = cheese_gate.update(hand_detections, w, h, current_time)
            if cheese_takes and state_machine.current_order.status == OrderStatus.IN_PROGRESS:
                state_machine.apply_cheese_takes(cheese_takes)
                # Forget every take the live order was offered, not just the ones
                # it counted: one dropped by the debounce was a duplicate, and
                # replaying it onto the next ticket would invent a slice.
                for _take in cheese_takes:
                    cheese_gate.forget(_take)

            # ── Sauce detection ───────────────────────────────────────────────
            # Spatial gate — three exclusive zones, no track_id state needed:
            #
            #   Zone A — sauce_vessel  (bottle resting on counter)
            #     → bottle is identified/known here; no sauce event fired.
            #
            #   Zone B — assembly OR bbox overlaps a food item
            #     → bottle is actively being applied → fire sauce action.
            #
            #   Anywhere else (ingredient bins, counter between A and B)
            #     → suppress: this is a false positive or the bottle in transit.
            #
            # Robustness: purely positional, unaffected by tracker ID re-use.

            for key, last_t in list(_sauce_last_fired.items()):
                if (current_time - last_t) >= SAUCE_COOLDOWN_S:
                    _sauce_applied[key] = False



            # Vessel zones (may be empty list if operator hasn't drawn one yet)
            vessel_zones = zones.get_zones_by_type("sauce_vessel")

            # Food classes that qualify as "sauce being applied to food"
            _FOOD_CLASSES = {"hot-dog", "burger_bun", "french_fries"}
            food_bboxes = [d.bbox for d in detections if d.class_name in _FOOD_CLASSES]
            KETCHUP_HOTDOG_PROXIMITY_PX = 140.0

            def _in_assembly(bbox, polygon=None) -> bool:
                zone = zones.get_zone_for_bbox(bbox, w, h, seg_polygon=polygon)
                return zone is not None and zone.zone_type == "assembly"

            def _near_assembly_hotdog(sauce_det) -> bool:
                """Require ketchup and its target hotdog to share assembly space."""
                scx, scy = _poly_centroid(sauce_det)

                # Prefer current visual detections, which are the most precise.
                candidates = [
                    (d.bbox, getattr(d, "polygon", None))
                    for d in detections
                    if d.class_name == "hot-dog"
                ]
                # Retain continuity during a brief hand occlusion of the hotdog.
                candidates.extend(
                    (rec.bbox, None)
                    for rec in hotdog_tracker._records.values()
                    if not rec.retired
                )
                for hotdog_bbox, hotdog_polygon in candidates:
                    if not _in_assembly(hotdog_bbox, hotdog_polygon):
                        continue
                    hcx = (hotdog_bbox[0] + hotdog_bbox[2]) / 2.0
                    hcy = (hotdog_bbox[1] + hotdog_bbox[3]) / 2.0
                    distance = ((scx - hcx) ** 2 + (scy - hcy) ** 2) ** 0.5
                    if _boxes_overlap(sauce_det.bbox, hotdog_bbox) or distance <= KETCHUP_HOTDOG_PROXIMITY_PX:
                        return True
                return False

            # Reset per-class frame counters and session flags for sauce classes absent this frame
            detected_sauce_classes = {
                det.class_name for det in detections if det.class_name in SAUCE_CLASSES
            }
            for sc in list(_sauce_frames.keys()):
                if sc not in detected_sauce_classes:
                    del _sauce_frames[sc]
                    # Clear all (sc, hotdog_tid) entries for this sauce class
                    for k in list(_sauce_applied.keys()):
                        if isinstance(k, tuple) and k[0] == sc:
                            del _sauce_applied[k]


            for det in detections:
                if det.class_name not in SAUCE_CLASSES:
                    continue

                # Normalised centre of the sauce bottle bbox
                cx_n = ((det.bbox[0] + det.bbox[2]) / 2) / w
                cy_n = ((det.bbox[1] + det.bbox[3]) / 2) / h

                # ── Zone A: bottle is resting in the vessel zone ──────────────
                # Suppress — it's just sitting there.  Reset frame counter and session applied flag.
                # Use polygon centroid when available
                bottle_cx, bottle_cy = _poly_centroid(det)
                cx_n = bottle_cx / w
                cy_n = bottle_cy / h

                if vessel_zones:
                    in_vessel = any(
                        cv2.pointPolygonTest(
                            np.array(vz.polygon, dtype=np.float32),
                            (cx_n, cy_n),
                            False,
                        ) >= 0
                        for vz in vessel_zones
                    )
                    if in_vessel:
                        _sauce_frames[det.class_name] = 0
                        # Clear per-hotdog applied flags for this sauce class on vessel return
                        for k in list(_sauce_applied.keys()):
                            if isinstance(k, tuple) and k[0] == det.class_name:
                                del _sauce_applied[k]
                        continue

                # ── Zone C: somewhere on the counter but NOT assembly / food ──
                # Suppress — false positive near ingredient bins, reflections,
                # or bottle in mid-air transit not yet over food.
                bottle_zone = zones.get_zone_for_bbox(
                    det.bbox, w, h, seg_polygon=getattr(det, "polygon", None)
                )
                in_assembly  = (bottle_zone is not None and bottle_zone.zone_type == "assembly")
                # Polygon-accurate overlap check with food objects
                overlaps_food = any(
                    _poly_or_box_overlap(det, fd_det)
                    for fd_det in detections
                    if fd_det.class_name in _FOOD_CLASSES
                )

                if not in_assembly and not overlaps_food:
                    _sauce_frames[det.class_name] = 0
                    for k in list(_sauce_applied.keys()):
                        if isinstance(k, tuple) and k[0] == det.class_name:
                            del _sauce_applied[k]
                    continue

                # Ketchup is valid only while the bottle is in assembly and
                # actually close to an assembly hotdog.  Do not count a bottle
                # merely sitting in the assembly area or passing through it.
                if det.class_name == "ketchup_sauce" and (
                    not in_assembly or not _near_assembly_hotdog(det)
                ):
                    _sauce_frames[det.class_name] = 0
                    continue

                # ── Zone B: bottle in assembly zone or directly over food ─────
                # This is the application event — bottle is being used on food.
                _sauce_frames[det.class_name] = (
                    _sauce_frames.get(det.class_name, 0) + 1
                )

                # Frame accumulation check is still per sauce class (detector stability)
                if (
                    _sauce_frames[det.class_name] >= SAUCE_MIN_FRAMES
                ):
                    # ── Sauce attribution via Hand-as-Bridge ─────────────────────
                    # Find the hand nearest to the sauce bottle (using the
                    # shrunk palm/fingertip region so we only match the hand
                    # actively holding the bottle, not a bystander hand).
                    # From that hand's working point, resolve the specific
                    # hotdog track_id being sauced — preventing blind
                    # fire_count attribution across multiple hotdogs.
                    bottle_wpt = _hand_working_point(det.bbox)

                    nearest_sauce_hand = None
                    nearest_sauce_hand_dist = float('inf')
                    for hand_det in hand_detections:
                        shx1, shy1, shx2, shy2 = _shrink_hand_bbox(hand_det.bbox)
                        hpt_x = (shx1 + shx2) / 2.0
                        hpt_y = float(shy2)
                        d = ((bottle_wpt[0] - hpt_x) ** 2 + (bottle_wpt[1] - hpt_y) ** 2) ** 0.5
                        if d < 250 and d < nearest_sauce_hand_dist:
                            nearest_sauce_hand_dist = d
                            nearest_sauce_hand = hand_det

                    resolved_hotdog_tids: list = []
                    if nearest_sauce_hand is not None:
                        # Hand confirmed near bottle — route sauce to ALL hotdogs
                        # sitting side-by-side or held in hand (bottle bbox overlap
                        # or centroid within 250px).
                        # Includes active tracker records so hotdog 2 is resolved
                        # even when occluded by the worker's hand holding it.
                        bottle_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                        bottle_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                        
                        PROXIMITY_THRESHOLD_PX = (
                            KETCHUP_HOTDOG_PROXIMITY_PX
                            if det.class_name == "ketchup_sauce"
                            else 250.0
                        )
                        nearby_hotdog_tids = []
                        closest_tid = None
                        closest_dist = float('inf')

                        # Combine raw YOLO detections + active tracker records (for hand occlusion)
                        hotdog_targets = []
                        seen_tids = set()
                        for hd in detections:
                            if hd.class_name == "hot-dog":
                                hotdog_targets.append((hd.track_id, hd.bbox))
                                seen_tids.add(hd.track_id)
                        for tid, rec in hotdog_tracker._records.items():
                            if not rec.retired and tid not in seen_tids:
                                hotdog_targets.append((tid, rec.bbox))
                        
                        for htid, hbbox in hotdog_targets:
                            rcx = (hbbox[0] + hbbox[2]) / 2.0
                            rcy = (hbbox[1] + hbbox[3]) / 2.0
                            dist = ((bottle_cx - rcx) ** 2 + (bottle_cy - rcy) ** 2) ** 0.5
                            overlaps = _boxes_overlap(det.bbox, hbbox)
                            
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_tid = htid
                                
                            if overlaps or dist <= PROXIMITY_THRESHOLD_PX:
                                nearby_hotdog_tids.append(htid)
                                
                        if nearby_hotdog_tids:
                            resolved_hotdog_tids = nearby_hotdog_tids
                        elif (
                            closest_tid is not None
                            and closest_dist <= PROXIMITY_THRESHOLD_PX
                        ):
                            resolved_hotdog_tids = [closest_tid]

                    if not resolved_hotdog_tids:
                        # No hand visible — fall back to polygon-overlap logic
                        hotdogs_under_bottle = [
                            d for d in detections
                            if d.class_name == "hot-dog"
                            and _poly_or_box_overlap(det, d)
                        ]
                        if hotdogs_under_bottle:
                            # Safely map to exactly the specific hotdogs under the sauce
                            resolved_hotdog_tids = [d.track_id for d in hotdogs_under_bottle]


                    for resolved_tid in resolved_hotdog_tids:
                        # Per-hotdog cooldown: each (sauce_class, hotdog_tid) pair
                        # tracks its own cooldown independently so two hotdogs in a
                        # batch can each receive sauce without blocking each other.
                        sauce_key = (det.class_name, resolved_tid)
                        elapsed_for_hd = current_time - _sauce_last_fired.get(sauce_key, 0.0)
                        if elapsed_for_hd >= SAUCE_COOLDOWN_S and not _sauce_applied.get(sauce_key, False):
                            actions.append(
                                Action(
                                    track_id=det.track_id,
                                    zone_id=bottle_zone.id if bottle_zone else "food_overlap",
                                    zone_name=det.class_name,
                                    action_type="sauce",
                                    timestamp=current_time,
                                    resolved_hotdog_tid=resolved_tid,
                                )
                            )
                            _sauce_applied[sauce_key] = True
                            _sauce_last_fired[sauce_key] = current_time
                            logger.debug(
                                "Sauce '%s' fired for hotdog tid=%s via hand-bridge",
                                det.class_name, resolved_tid,
                            )



            # ── Wire pick/place/sauce actions → hotdog tracker (event-driven) ───
            # TemporalTracker knows WHAT was picked and WHICH hand fired.
            # Use the hand's working-point (bottom-centre of shrunk bbox) to find
            # the nearest active hotdog ID and commit the item/sauce directly.
            for action in actions:
                if action.action_type in ("pick", "place", "pickup", "sauce"):
                    # For sauce with a pre-resolved hotdog tid, resolve monotonic id
                    # and commit directly without spatial re-search.
                    if (
                        action.action_type == "sauce"
                        and action.resolved_hotdog_tid is not None
                    ):
                        raw_tid = action.resolved_hotdog_tid
                        # If already a valid monotonic tid in _records, use directly
                        if raw_tid in hotdog_tracker._records or raw_tid in hotdog_tracker._retired_records:
                            mono_tid = raw_tid
                        else:
                            mono_tid = hotdog_tracker._detector_id_map.get(raw_tid, raw_tid)

                        committed_tid = hotdog_tracker.force_commit_item(
                            hand_working_pt=(0.0, 0.0),  # unused when target_tid set
                            item_class=action.zone_name,
                            now=current_time,
                            max_radius=300,
                            is_sauce=True,
                            target_tid=mono_tid,
                        )
                        if committed_tid is not None:
                            logger.debug(
                                "[sauce→hotdog] direct '%s' → hotdog #%d (resolved_tid=%s, mono_tid=%s)",
                                action.zone_name, committed_tid,
                                action.resolved_hotdog_tid, mono_tid,
                            )
                        continue  # skip spatial fallback below

                    firing_hand = next(
                        (h for h in hand_detections if h.track_id == action.track_id),
                        hand_detections[0] if hand_detections else None,
                    )
                    if firing_hand is not None:
                        hwpt = _hand_working_point(firing_hand.bbox)
                        committed_tid = hotdog_tracker.force_commit_item(
                            hand_working_pt=hwpt,
                            item_class=action.zone_name,
                            now=current_time,
                            max_radius=600,
                            is_sauce=(action.action_type == "sauce"),
                        )
                        if committed_tid is not None:
                            logger.debug(
                                "[action→hotdog] %s '%s' → hotdog #%d",
                                action.action_type, action.zone_name, committed_tid,
                            )

            # ── State machine + dashboard events ─────────────────────────────────
            for action in actions:
                state_machine.on_action(action)

                # Record the evidence against the open ticket, so a WRONG
                # verdict can say WHEN an ingredient went on (or that it never
                # did) rather than only that it was missing.
                if kdsocr is not None and state_machine.current_ticket is not None:
                    if action.action_type in ("place", "sauce"):
                        # Only the ROIs this ticket asks for.  The journey is
                        # the evidence behind a verdict, and a bin the ticket
                        # never mentioned is not evidence about this order --
                        # the validator already refuses to credit it, so
                        # recording it only padded the journey with items
                        # nobody ordered.
                        _item = canonical_ingredient(action.zone_name)
                        _req = state_machine.current_order.required_counts or {}
                        # While the pre-gate owns cheese, on_action drops these
                        # bin events and credits the region exit instead -- so
                        # recording them here put a place in the journey that
                        # nothing counted, and the evidence contradicted the
                        # verdict: a ticket could show a cheese place and still
                        # report that same cheese missing.  apply_cheese_takes
                        # records the take that WAS counted.
                        _owned = (state_machine.cheese_gate_owns_cheese
                                  and _item in CHEESE_KEYS)
                        if _item in _req and not _owned:
                            kdsocr.record_place(
                                state_machine.current_ticket.ticket_id,
                                _item, t=current_time, zone=action.zone_name,
                            )

                if action.action_type == "pickup":
                    add_event(
                        "pickup",
                        zone=action.zone_id,
                        item=action.zone_name,
                        duration=action.duration_ms / 1000,
                    )
                elif action.action_type == "pick":
                    add_event(
                        "pick",
                        zone=action.zone_id,
                        item=action.zone_name,
                        duration=action.duration_ms / 1000,
                    )
                elif action.action_type == "place":
                    add_event(
                        "place",
                        zone=action.zone_id,
                        item=action.zone_name,
                        duration=action.duration_ms / 1000,
                    )
                elif action.action_type == "hover":
                    add_event(
                        "hover",
                        zone=action.zone_id,
                        item=action.zone_name,
                        duration=action.duration_ms / 1000,
                    )
                elif action.action_type == "sauce":
                    add_event(
                        "sauce",
                        item=action.zone_name,
                    )


            # Translate wrapping done_ids (YOLO/monotonic track_ids) → monotonic hotdog IDs
            _wrapping_done_mono = set(wrapping_sm.done_ids) | set(getattr(hotdog_tracker, "_permanent_done_ids", set()))
            for _yolo_tid in list(wrapping_sm.done_ids):
                _mono = hotdog_tracker._detector_id_map.get(_yolo_tid)
                if _mono is not None:
                    _wrapping_done_mono.add(_mono)

            # ── Hotdog tracker update (ByteTrack + Monotonic IDs + POS Fusion) ──
            active_ticket_id = state_machine.current_ticket.ticket_id if state_machine.current_ticket else None
            # Ketchup is attributed above through the strict assembly +
            # target-hotdog gate.  Do not feed raw ketchup detections into
            # HotdogTracker as well: its generic item association can otherwise
            # create a second, ungated ketchup event for a nearby/stale track.
            # Mustard continues through the normal tracker path.
            tracker_detections = [
                det for det in detections if det.class_name != "ketchup_sauce"
            ]

            expected_hotdogs = 1
            if state_machine.current_ticket:
                expected_hotdogs = state_machine.current_ticket.total_hotdogs

            hotdog_tracker.update(
                tracker_detections,
                current_time=current_time,
                frame=frame,
                active_ticket_id=active_ticket_id,
                done_ids=_wrapping_done_mono,
                expected_hotdogs=expected_hotdogs,
            )

            is_ending_soon = False
            if capture._is_file_source:
                total_frames = capture.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                current_frame = capture.cap.get(cv2.CAP_PROP_POS_FRAMES)
                video_fps = capture.cap.get(cv2.CAP_PROP_FPS)
                if video_fps > 0 and total_frames > 0 and current_frame > 10:
                    remaining_s = (total_frames - current_frame) / video_fps
                    if remaining_s <= 0.8:
                        is_ending_soon = True

            # ── Wrapping-state update (additive — do not remove) ───────────────
            _hotdog_dets_mono = []
            for d in detections:
                if d.class_name == "hot-dog":
                    _mono = hotdog_tracker._detector_id_map.get(d.track_id, d.track_id)
                    _hotdog_dets_mono.append(
                        Detection(
                            track_id=_mono,
                            bbox=d.bbox,
                            class_name=d.class_name,
                            confidence=d.confidence,
                        )
                    )

            _wrapping_events = wrapping_sm.update(
                frame_idx=frame_count,
                current_time=current_time,
                hotdog_detections=_hotdog_dets_mono,
                wrapping_detections=[d for d in detections if d.class_name in ("wrapping", "wrapper", "reg_clamshell")],
                wrapped_detections=[d for d in detections if d.class_name == "wrapped"],
                video_ending_soon=is_ending_soon,
            )
            # Print state transitions to terminal immediately for visibility
            for _ev in _wrapping_events:
                _ev_type = _ev.get("event")
                _yolo_tid = _ev.get("hotdog_tid")
                _mono = hotdog_tracker._detector_id_map.get(_yolo_tid, _yolo_tid)
                hotdog_tracker.record_wrapping_event(
                    event_type=_ev_type,
                    track_id=_mono,
                    timestamp=_ev.get("timestamp", current_time),
                    frame_idx=_ev.get("frame", frame_count),
                    closing_time=_ev.get("closing_time"),
                )
                if _ev_type == "closing":
                    if state_machine.current_order:
                        state_machine.current_order.ending_soon = True
                    add_event("hover", zone="wrapping", item="about_to_complete", duration=0.4)
                elif _ev_type == "done":
                    if state_machine.current_order:
                        state_machine.current_order.ending_soon = False
                    if _mono is not None:
                        hotdog_tracker._permanent_done_ids.add(_mono)
                    add_event("place", zone="assembly", item="wrapped_hotdog", duration=1.0)
                    # Feed the KDS FIFO one confirmed physical hotdog.  A
                    # wrapping "done" is already temporally confirmed by
                    # WrappingStateMachine, so this is never a single-frame
                    # detection (section 8).  The type is inferred from the
                    # ingredients the zone pipeline attributed to this track.
                    # Mock KDS has no ticket manager to feed, so count the
                    # hotdogs this group has actually had MADE here.  A
                    # wrapping "done" is the same evidence the KDS path uses;
                    # picked_counts["hot-dog"] is not -- that counts hotdogs
                    # seen, and two being visible satisfied a whole group
                    # fourteen seconds into the video.
                    # Wrapping ends this hotdog's life on the bench. Retire its
                    # presence slot so a dog built in the same spot right after
                    # is counted as a new one -- a timer alone cannot tell
                    # succession from occlusion, and this event can.
                    _wrec = hotdog_tracker._records.get(_mono) or \
                            hotdog_tracker._retired_records.get(_mono)
                    _wbox = getattr(_wrec, "bbox", None) if _wrec else None
                    if _wbox:
                        hotdog_presence.close_at((_wbox[0] + _wbox[2]) / 2.0,
                                                 (_wbox[1] + _wbox[3]) / 2.0)

                    if kdsocr is None and _mono is not None:
                        _mock_made_ids.add(_mono)
                    # Same evidence, recorded against the kds-ocr ticket: a
                    # finished hotdog is a step in its journey, so a WRONG
                    # verdict can show how many were actually made and when.
                    if (kdsocr is not None and _mono is not None
                            and state_machine.current_ticket is not None):
                        kdsocr.record_hotdog(
                            state_machine.current_ticket.ticket_id,
                            track_id=_mono, t=current_time,
                        )


            # POS / KDS Fusion Mismatch Validation (throttled alert)
            if state_machine.current_ticket:
                t_id = state_machine.current_ticket.ticket_id
                exp_count = state_machine.current_ticket.total_hotdogs
                # Count = hotdogs present on the bench for THIS ticket. Any
                # detection either joins a slot or opens one, so a hotdog that
                # is visible on the feed is counted -- which is the whole
                # failure this replaces: the tracker drew ids from a run-wide
                # pool of 1..expected and silently dropped every detection once
                # that pool was spent, so plainly visible dogs read as 0/N.
                if hotdog_presence._ticket != t_id:
                    hotdog_presence.reset(t_id)
                presence_count = hotdog_presence.update(
                    [d.bbox for d in detections if d.class_name == "hot-dog"],
                    current_time,
                )

                h_log = hotdog_tracker.get_hotdog_log()
                # Two separate faults used to live in this list comprehension.
                #
                # FLICKER: a record counted from its very first frame, so an
                # object mistaken for a hotdog for a fraction of a second became
                # a hotdog -- and since the count below only ever moves upward,
                # that phantom never went away.  Requiring the identity to have
                # survived `hotdog_min_track_age_s` discards sub-second noise.
                # It is an AGE test, not a continuous-visibility test, because
                # hands cover the dog constantly during assembly -- the same
                # reason wrap_station carries a 20 s occlusion buffer -- and a
                # "must stay visible" rule would starve real hotdogs instead.
                #
                # MISSED: eligibility was `order_id == t_id or (active and
                # order_id is None)`.  A hotdog still physically on the bench
                # whose track had already been bound to an earlier ticket
                # matched neither arm, so it was detected and drawn on the feed
                # yet stayed permanently invisible to this ticket's checklist.
                # A dog present NOW is being made now; the `min(_req_hd, ...)`
                # cap below still prevents crediting more than the ticket asked.
                _min_age = float(_env("HOTDOG_MIN_TRACK_AGE_S",
                                      config.get("hotdog_min_track_age_s", 0.6)))

                def _settled(rec):
                    """True once this identity has existed for `_min_age`.

                    `first_seen` is stamped with the same `current_time` the
                    tracker is driven by (media time for a file), so this
                    threshold means the same thing regardless of how fast the
                    pipeline happens to be running.
                    """
                    first = rec.get("first_seen")
                    if first is None:
                        return False
                    return (current_time - float(first)) >= _min_age

                # Mine, or physically present and not yet claimed by anyone.
                #
                # An earlier attempt here counted ANY active record, to rescue
                # hotdogs that were plainly on the feed yet bound to a previous
                # ticket and therefore invisible to this one.  That widened the
                # leak instead of closing it: the previous order's dogs could be
                # credited to this one.  The boundary is the right place to fix
                # it -- HotdogTracker.close_ticket() retires a ticket's records
                # when it closes, so nothing stays bound to a finished order and
                # this narrower rule no longer starves.
                active_tracks = [
                    rec for rec in h_log.values()
                    if _settled(rec)
                    and (rec.get("order_id") == t_id
                         or (rec.get("order_id") is None and rec.get("active")))
                ]
                # Presence is the count; the tracked-identity number is kept
                # only as a floor, so a dog the tracker did see is never lost
                # if presence somehow missed it. Neither can pull the other
                # down -- both are per ticket and both only rise within one.
                vision_count = max(presence_count, len(active_tracks))
                # No over-count alert. `active_tracks` is inflated by track
                # fragmentation -- one physical hotdog becomes several ids --
                # so this fired on correct orders and meant nothing.

                if state_machine.current_order:
                    # Arrival, not an exact count, and never allowed to fall.
                    # The raw number is "hotdog tracks active right now", which
                    # drops again as tracks end, so assigning it directly made
                    # the checklist show 3/3 and then 1/3 on an order whose
                    # hotdogs had all been made. Capped at what the ticket
                    # asked for, so it settles there and stops.
                    _req_hd = int(
                        state_machine.current_order.required_counts.get(
                            "hot-dog", exp_count) or exp_count
                    )
                    _seen_hd = int(
                        state_machine.current_order.picked_counts.get("hot-dog", 0)
                    )
                    state_machine.current_order.picked_counts["hot-dog"] = (
                        min(_req_hd, max(_seen_hd, vision_count)) if _req_hd
                        else max(_seen_hd, vision_count)
                    )
                    state_machine.calculate_validation(
                        state_machine.current_order, 
                        is_final=(state_machine.current_order.status.value == "completed")
                    )

            detection_counts = {}
            for det in visible_detections:
                detection_counts[det.class_name] = (
                    detection_counts.get(det.class_name, 0) + 1
                )

            # ── Automated Cart State Machine Transitions ───────────────────────
            from src.analysis.cart_state_machine import CartEvent, EventType

            # Check active hotdogs in tracker (with persistence to prevent flickering)
            has_active_hotdogs = len(hotdog_tracker._records) > 0
            if has_active_hotdogs:
                main._no_hotdogs_start_t = None
                cart_machine.process_event(CartEvent(event_type=EventType.HOTDOG_DETECTED))
                for tid, rec in hotdog_tracker._records.items():
                    z = zones.get_zone_for_bbox(rec.bbox, w, h, is_hand=False)
                    if z and z.zone_type == "assembly":
                        cart_machine.process_event(CartEvent(event_type=EventType.ENTERED_ASSEMBLY))
                        break
            else:
                cart_machine.process_event(CartEvent(event_type=EventType.NO_HOTDOGS))

            draw_annotations._hotdog_log_ref = hotdog_tracker.get_hotdog_log()
            draw_annotations._current_video_time = current_time
            draw_annotations._done_ids = set()
            draw_annotations._detector_id_map = dict(getattr(hotdog_tracker, "_detector_id_map", {}))
            annotated = draw_annotations(
                frame.copy(), visible_detections, zones, state_machine.get_current_order()
            )

            # ── Feed analysis: hotdog lifecycle, sauces, items, outgoing ──────
            # Read-only: nothing here feeds back into tracking, KDS or orders.
            try:
                analyzer.update(
                    frame_idx=frame_count,
                    video_time=current_time,
                    frame_size=(w, h),
                    detections=detections,
                    visible_detections=visible_detections,
                    actions=actions,
                    wrapping_events=_wrapping_events,
                    hotdog_tracker=hotdog_tracker,
                )
                analyzer.draw(annotated)  # exit line, red while a hand touches it
            except Exception:
                logger.exception("Feed analysis failed on frame %d", frame_count)
            _now_wall = time.time()
            _prev_wall = getattr(analyzer, "_prev_frame_wall", None)
            analyzer._prev_frame_wall = _now_wall
            if _prev_wall is not None and _now_wall > _prev_wall:
                _inst = 1.0 / (_now_wall - _prev_wall)
                analyzer._fps_ema = 0.9 * getattr(analyzer, "_fps_ema", _inst) + 0.1 * _inst
            if _now_wall - last_analysis_publish >= 0.25:
                last_analysis_publish = _now_wall
                _elapsed = _now_wall - pipeline_start_time
                _video_s = current_time if capture._is_file_source else 0.0
                _rate = _video_s / _elapsed if _elapsed > 0 else 0.0
                if kdsocr is not None:
                    dashboard.set_kds_state(kdsocr.dashboard_state())
                dashboard.set_analysis(analyzer.snapshot())
                dashboard.set_system({
                    "model": model_label,
                    "decoder": decoder_label,
                    "fps_now": round(getattr(analyzer, "_fps_ema", 0.0), 1),
                    "fps_avg": round(frame_count / _elapsed, 2) if _elapsed > 0 else None,
                    "detect_ms": round(detect_ms, 1),
                    "flow_ms": round(flow_ms, 1),
                    "loop_ms": round(1000.0 / analyzer._fps_ema, 1) if getattr(analyzer, "_fps_ema", 0) else None,
                    "frames": frame_count,
                    "video_s": round(_video_s, 1),
                    "duration_s": round(video_duration_s, 1),
                    "eta_s": round((video_duration_s - _video_s) / _rate) if _rate > 0 and video_duration_s else None,
                    "recorder": getattr(recorder, "encoder", None) if recorder is not None else None,
                    "hw": system_monitor.snapshot(),
                })
            if _now_wall - last_analysis_write >= 30.0:
                last_analysis_write = _now_wall
                try:
                    analyzer.write_json(analysis_path)
                except OSError:
                    logger.debug("Could not write %s", analysis_path, exc_info=True)

            _wrapping_done_mono = set()
            for _yolo_tid in list(wrapping_sm.done_ids):
                _mono = hotdog_tracker._detector_id_map.get(_yolo_tid)
                if _mono is not None:
                    _wrapping_done_mono.add(_mono)

            # ── Overlay KDS Image ──────────────────────────────────────────────
            # Removed KDS overlay on video feed (now displayed in dashboard)
            # ───────────────────────────────────────────────────────────────────

            # Recording (RECORD_VIDEO): exactly the frame the dashboard shows.
            if recorder is not None:
                recorder.write(annotated)
            update_frame_data(
                annotated,
                state_machine.current_ticket,
                state_machine.get_current_order(),
                state_machine.get_stats(),
                detections=detection_counts,
                validation_log=state_machine.get_validation_log(),
                track_states=temporal.get_track_states(),
                hotdog_log=hotdog_tracker.get_hotdog_log(),
                wrapping_done_ids=_wrapping_done_mono,
            )

            # KDS FIFO queue, completed/warning lists and the event timeline.
            # The old in-process KDS reader also composed a production+KDS
            # frame here for per-ticket failure clips. It went with src/kds/,
            # and the dashboard is now recorded as a real browser window
            # (scripts/record_dashboard.py), so nothing composes frames.

            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            frame_count += 1
            watchdog.beat()

            now = time.time()
            if now - last_metrics_time >= metrics_interval:
                elapsed = now - pipeline_start_time
                fps_actual = frame_count / elapsed if elapsed > 0 else 0
                stats = state_machine.get_stats()
                metrics_logger.info(
                    json.dumps(
                        {
                            "event": "metrics",
                            "fps": round(fps_actual, 2),
                            "frame_count": frame_count,
                            "elapsed_s": round(elapsed, 1),
                            "loop_ms": round(loop_ms, 2),
                            "detect_ms": round(detect_ms, 2),
                            "flow_ms": round(flow_ms, 2),
                            "temporal_ms": round(temporal_ms, 2),
                            "detections": detection_counts,
                            "total_orders": stats.total_orders,
                            "passed_orders": stats.passed_orders,
                            "failed_orders": stats.failed_orders,
                            "accuracy_pct": stats.accuracy_pct,
                            "error_rate_pct": stats.error_rate_pct,
                        }
                    )
                )
                last_metrics_time = now
    except KeyboardInterrupt:
        print("[DIAG] Exiting via KeyboardInterrupt", flush=True)
    except Exception as _diag_exc:
        print(f"[DIAG] Exiting via EXCEPTION: {type(_diag_exc).__name__}: {_diag_exc}", flush=True)
        import traceback; traceback.print_exc()
    finally:
        print("[DIAG] In finally block — main loop ended", flush=True)
        # ── Finalise the recording before anything else can fail ─────────
        try:
            if recorder is not None:
                _n = recorder.close()
                print(f"[RECORD] wrote {_n} frames -> {recorder.path}"
                      + (f" (stopped early: {recorder.error})" if recorder.error else ""),
                      flush=True)
        except NameError:
            pass  # the loop failed before the recorder was created
        except Exception as _rec_exc:
            logger.warning("Failed to finalise recording: %s", _rec_exc)
        # ── Sync wrapping states before emitting final summary ────────────
        try:
            hotdog_tracker.sync_wrapping_states(wrapping_sm)
        except Exception:
            pass

        # ── Emit final hotdog summary as a JSON event and write to disk ───
        try:
            summary_payload = hotdog_tracker.get_summary(wrapping_sm=wrapping_sm)
            metrics_logger.info(
                json.dumps({
                    "event": "hotdog_summary",
                    "hotdog_log": summary_payload,
                })
            )
            out_file = Path(resource("output/hotdog_summary.json"))
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, "w") as f:
                json.dump(summary_payload, f, indent=2)
        except Exception as e:
            logger.debug("Failed to write final hotdog summary: %s", e)

        if state_machine.current_order and state_machine.current_order.status == OrderStatus.IN_PROGRESS:
            state_machine.finalize_current_order()

        watchdog.stop()
        if dashboard_recorder is not None:
            try:
                # Ask, never kill: the recorder has to close the GStreamer
                # pipeline and tear down Firefox and the hidden display (or,
                # on Windows, close the browser context, which is what
                # finalises the video at all).  A killed recorder leaves an
                # unplayable file and an orphaned Xorg.
                #
                # Windows is asked with a file, not a signal.  It has no SIGINT
                # to send, and CTRL_BREAK_EVENT is delivered to every process
                # in the recorder's group -- the Playwright driver included.
                # Killing the driver means the browser context can never be
                # closed, and closing it is the only thing that writes the
                # video, so the polite signal destroyed exactly what it was
                # trying to save.
                if sys.platform == "win32":
                    try:
                        Path(_dash_record_path).with_suffix(".stop").touch()
                    except OSError:
                        logger.warning("could not ask the recorder to stop",
                                       exc_info=True)
                else:
                    dashboard_recorder.send_signal(signal.SIGINT)
                # Generous: the Windows recorder re-encodes the whole screencast
                # on the way out, and that file is as long as the run in wall
                # time -- minutes of 1080p, not seconds.  Timing out here kills
                # the recorder mid-conversion and costs the recording, which is
                # far worse than waiting.
                dashboard_recorder.wait(timeout=900)
                logger.info("Dashboard recording saved to %s", _dash_record_path)
            except subprocess.TimeoutExpired:
                logger.warning("dashboard recorder did not finish in time; "
                               "the video may be incomplete")
                dashboard_recorder.kill()
            except Exception:
                # Visible, not debug: a recorder that fails to stop cleanly
                # leaves an unplayable video, and silence made that
                # indistinguishable from success.
                logger.warning("dashboard recorder shutdown failed", exc_info=True)
        if kds_preview is not None:
            try:
                kds_preview.stop()
            except Exception:
                logger.warning("KDS preview shutdown failed", exc_info=True)
        if kdsocr is not None:
            try:
                kdsocr.stop()
            except Exception:
                logger.warning("kds-ocr shutdown failed", exc_info=True)

        # RECORD_WRONG_ONLY (default on): trim the run down to one clip per
        # WRONG order and drop the rest.  Deliberately AFTER kdsocr.stop() --
        # that is what closes and writes the last journeys, so running this
        # before it could miss an order judged in the final seconds.
        #
        # Trimming afterwards rather than recording per ticket: a recorder per
        # ticket would cost an Xorg and a Firefox launch each time, and would
        # miss the seconds either side of the order.
        #
        # Either recorder's file can be trimmed.  The dashboard capture is
        # preferred where it exists, because it shows the verdict as the
        # reviewer would have read it; the annotated feed is what a box
        # without Xorg/Firefox/GStreamer can produce, and it carries the same
        # boxes, masks and zones.  The clip script tells the two apart by the
        # sidecar, so all that is decided here is which file to hand it.
        _trim_target = ""
        if dashboard_recorder is not None:
            _trim_target = _dash_record_path
        else:
            try:
                if recorder is not None:
                    _trim_target = str(recorder.path)
            except NameError:
                pass  # the loop failed before the recorder was created
        if (_trim_target
                and str(_env("RECORD_WRONG_ONLY", True)).lower()
                not in ("false", "0", "no")):
            try:
                subprocess.run(
                    [sys.executable,
                     str(resource("scripts/clip_wrong_orders.py")),
                     "--recording", _trim_target,
                     "--journeys", str(config.get("ticket_journeys",
                                                  "output/ticket_journeys.jsonl"))],
                    check=False, timeout=900)
            except Exception:
                logger.warning("could not trim the recording to the wrong orders",
                               exc_info=True)
        state_machine.save_history()
        capture.release()
        
        # Play alert sound if the order failed
        try:
            if state_machine.stats and state_machine.stats.failed_orders > 0:
                print("Order failed!")
        except Exception as e:
            print("Failed to check order status:", e)
        
        
        # EXIT_ON_END: return instead of parking, so an unattended batch run
        # (scripts/run_full_and_shutdown.py) can tell that the work is finished.
        # Without this the process sits here forever waiting for Ctrl+C and no
        # caller can ever see it complete.
        if exit_on_end:
            print('\nVideo processing complete. Exiting (EXIT_ON_END).')
            return

        # Keep the dashboard running after the video finishes.
        # Use a sleep loop instead of .join() so that Ctrl+C can interrupt it.
        # The dashboard thread is a daemon — it will be killed automatically
        # when the main (foreground) thread exits.
        print('\nVideo processing complete! The dashboard is still running at http://localhost:8000')
        print('Press Ctrl+C to exit.')
        try:
            while dashboard_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print('\nShutting down...')


if __name__ == "__main__":
    main()
