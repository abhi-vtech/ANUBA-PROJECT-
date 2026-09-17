from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExitLineConfig:
    """Configuration for an exit tripwire line or polygon ROI."""
    p1: Tuple[float, float] = (0.8, 0.2)  # (x, y) normalized coordinates (0.0 - 1.0)
    p2: Tuple[float, float] = (0.8, 0.8)
    line_name: str = "Exit_Tripwire_Line"
    polygon: Optional[List[Tuple[float, float]]] = None  # Optional polygon ROI


@dataclass
class ExitEvent:
    """Event emitted when hand or hotdog crosses the exit line."""
    timestamp: float
    exited_hotdog_ids: List[Any]
    hand_bbox: Optional[Tuple[int, int, int, int]] = None
    message: str = ""


def _line_segment_intersection(
    seg1: Tuple[Tuple[float, float], Tuple[float, float]],
    seg2: Tuple[Tuple[float, float], Tuple[float, float]],
) -> bool:
    """Return True if 2D line segment seg1 intersects seg2."""
    (x1, y1), (x2, y2) = seg1
    (x3, y3), (x4, y4) = seg2

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    A, B = (x1, y1), (x2, y2)
    C, D = (x3, y3), (x4, y4)

    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def _bbox_intersects_line(
    bbox: Tuple[int, int, int, int],
    line_p1: Tuple[int, int],
    line_p2: Tuple[int, int],
) -> bool:
    """Return True if bounding box (x1, y1, x2, y2) intersects the line segment (p1, p2)."""
    bx1, by1, bx2, by2 = bbox
    # Check if any of the 4 bbox edge segments intersect line_p1 -> line_p2
    edges = [
        ((bx1, by1), (bx2, by1)),
        ((bx2, by1), (bx2, by2)),
        ((bx2, by2), (bx1, by2)),
        ((bx1, by2), (bx1, by1)),
        ((bx1, by1), (bx2, by2)),  # diagonals for robust containment
        ((bx1, by2), (bx2, by1)),
    ]
    for edge in edges:
        if _line_segment_intersection(edge, (line_p1, line_p2)):
            return True

    # Check if line endpoints land inside bbox
    if bx1 <= line_p1[0] <= bx2 and by1 <= line_p1[1] <= by2:
        return True
    if bx1 <= line_p2[0] <= bx2 and by1 <= line_p2[1] <= by2:
        return True

    return False


class ExitDetector:
    """
    Detects hand or object movement across an exit line / ROI.
    When a hand crosses the line, commits all wrapped hotdogs up to that frame as EXITED.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = ExitLineConfig()
        self.config_path = config_path
        self.last_exit_time: float = 0.0
        self.exit_cooldown_s: float = 2.0  # Debounce consecutive triggers
        self.exited_history: List[ExitEvent] = []
        self.is_active_crossing: bool = False

        if config_path and Path(config_path).exists():
            self.load_config(config_path)

    def load_config(self, path: str) -> None:
        """Load line coordinates from JSON file."""
        p = Path(path)
        if not p.exists():
            return
        data = json.loads(p.read_text())
        p1 = tuple(data.get("p1", [0.8, 0.2]))
        p2 = tuple(data.get("p2", [0.8, 0.8]))
        polygon = data.get("polygon")
        if polygon:
            polygon = [tuple(pt) for pt in polygon]

        self.config = ExitLineConfig(
            p1=(float(p1[0]), float(p1[1])),
            p2=(float(p2[0]), float(p2[1])),
            line_name=data.get("line_name", "Exit_Tripwire_Line"),
            polygon=polygon,
        )
        logger.info(f"[ExitDetector] Loaded config from {path}: p1={self.config.p1}, p2={self.config.p2}")

    def save_config(self, path: str) -> None:
        """Save line coordinates to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "p1": list(self.config.p1),
            "p2": list(self.config.p2),
            "line_name": self.config.line_name,
            "polygon": [list(pt) for pt in self.config.polygon] if self.config.polygon else None,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"[ExitDetector] Saved line config to {path}")

    def get_pixel_coords(self, frame_width: int, frame_height: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Return line endpoints in absolute pixel coordinates."""
        p1_px = (int(self.config.p1[0] * frame_width), int(self.config.p1[1] * frame_height))
        p2_px = (int(self.config.p2[0] * frame_width), int(self.config.p2[1] * frame_height))
        return p1_px, p2_px

    def check_crossing(
        self,
        hand_bboxes: List[Tuple[int, int, int, int]],
        wrapped_hotdog_ids: List[Any],
        frame_width: int = 1920,
        frame_height: int = 1080,
        hotdog_boxes: Optional[List[Tuple[Any, Tuple[int, int, int, int]]]] = None,
    ) -> Optional[ExitEvent]:
        """Emit an exit when a WRAPPED HOTDOG crosses the line.

        It used to be the hand that triggered this, and any hand crossing
        committed EVERY wrapped hotdog at once:

            for hand_bbox in hand_bboxes:            # a hand -- any hand
                if _bbox_intersects_line(...):
                    exited_ids = list(wrapped_hotdog_ids)   # all of them

        A crew member reaching over the line for anything at all -- a wrapper, a
        tray, the next order -- therefore reported the whole board as having
        gone out.  The hand is not the thing leaving; the hotdog is.

        `hotdog_boxes` is [(mono_id, bbox), ...] for hotdogs currently tracked.
        Only an id that is BOTH wrapped and physically over the line exits, and
        only that id -- so one dog going out reports one dog, not the board.

        The hand is still tracked, but only to light up the tripwire overlay
        (`is_active_crossing`); it no longer commits anything.
        """
        now = time.time()
        p1_px, p2_px = self.get_pixel_coords(frame_width, frame_height)

        # Overlay state only. Deliberately does not gate the event below.
        self.is_active_crossing = any(
            _bbox_intersects_line(b, p1_px, p2_px) for b in (hand_bboxes or [])
        )

        if not hotdog_boxes:
            return None

        wrapped = set(wrapped_hotdog_ids or [])
        crossing_ids = [
            hid for hid, box in hotdog_boxes
            if hid in wrapped and _bbox_intersects_line(box, p1_px, p2_px)
        ]
        if not crossing_ids:
            return None

        if now - self.last_exit_time < self.exit_cooldown_s:
            return None

        self.last_exit_time = now
        self.is_active_crossing = True
        event = ExitEvent(
            timestamp=now,
            exited_hotdog_ids=crossing_ids,
            hand_bbox=None,
            message=f"Wrapped hotdog crossed the exit line: {crossing_ids}",
        )
        self.exited_history.append(event)
        logger.info(f"[ExitDetector] {event.message}")
        return event

    def draw_overlay(
        self,
        frame: np.ndarray,
        banner_message: Optional[str] = None,
    ) -> np.ndarray:
        """
        Renders glowing exit tripwire line and crossing alert on frame.
        """
        h, w = frame.shape[:2]
        p1_px, p2_px = self.get_pixel_coords(w, h)

        # Dynamic line color: bright neon cyan/green normally, glowing red when crossed
        color = (0, 0, 255) if self.is_active_crossing else (255, 255, 0)
        glow_color = (0, 0, 180) if self.is_active_crossing else (200, 200, 0)

        # Glow layer
        cv2.line(frame, p1_px, p2_px, glow_color, 8)
        # Core tripwire line
        cv2.line(frame, p1_px, p2_px, color, 3)

        # Draw line endpoints
        cv2.circle(frame, p1_px, 6, (0, 255, 255), -1)
        cv2.circle(frame, p2_px, 6, (0, 255, 255), -1)

        # Label text along line
        mid_x = (p1_px[0] + p2_px[0]) // 2
        mid_y = (p1_px[1] + p2_px[1]) // 2
        status_str = "EXIT LINE [CROSSED]" if self.is_active_crossing else "EXIT LINE"
        cv2.putText(
            frame,
            status_str,
            (mid_x - 40, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
        )

        # Optional crossing banner
        if banner_message or (time.time() - self.last_exit_time < 2.5 and len(self.exited_history) > 0):
            msg = banner_message or self.exited_history[-1].message
            cv2.rectangle(frame, (20, 20), (w - 20, 70), (0, 0, 0), -1)
            cv2.rectangle(frame, (20, 20), (w - 20, 70), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"STATUS: {msg}",
                (35, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        return frame
