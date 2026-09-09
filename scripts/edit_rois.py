"""Adjust the ROI polygons in ``config/zones.json`` on a real camera frame.

    python scripts/edit_rois.py

Edits the zones **one at a time** -- chilli, cheese region, onions, assembly and
the rest -- by dragging the points that are already there.  It is an adjuster,
not a drawing tool: only the zone you are currently on is shown, so nothing
else clutters the frame.  A full redraw is available per zone if a box is so
far off that nudging points is not worth it.

Controls
--------
  drag             move a point (grab within 12 px of it)
  left click       on an edge, inserts a new point there
  right click      on a point, deletes it (a polygon keeps at least 3)
  n / p            next / previous zone
  r                redraw this zone from scratch (click points, ENTER accept, ESC cancel)
  u                undo the last change to this zone
  R                reset this zone back to what was on disk
  o                toggle the other zones on/off as faint context
  [ / ]            step the background frame back / forward
  f                jump to a specific frame number (typed in the terminal)
  s                save every zone to config/zones.json
  q / ESC          quit (asks first if there are unsaved edits)

The polygons are stored normalised 0..1, so an ROI edited on one resolution
stays correct at any other -- which is why the pipeline can run this 1920x1080
recording at 1280x720 without touching this file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO = (
    ROOT
    / "videos"
    / "KDS"
    / "Wienerschnitzel_Sacramento_CA_95818__camA__2026_09_06_11_to_12_PDT.mkv"
)
DEFAULT_ZONES = ROOT / "config" / "zones.json"

WINDOW = "ROI editor"
GRAB_RADIUS = 12          # px, how close a click must be to grab a point
EDGE_RADIUS = 8           # px, how close a click must be to an edge to insert
MAX_W, MAX_H = 1600, 900  # keep the window on screen

ACTIVE = (80, 235, 255)   # BGR, the zone being edited
OTHER = (110, 110, 110)
POINT = (60, 60, 245)
HANDLE = (255, 255, 255)
TEXT = (245, 245, 245)
DARK = (24, 22, 20)


def hex_to_bgr(value: str) -> Tuple[int, int, int]:
    value = (value or "").lstrip("#")
    if len(value) != 6:
        return ACTIVE
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


class RoiEditor:
    def __init__(self, video: Path, zones_path: Path, start_frame: int = 0):
        self.zones_path = zones_path
        with open(zones_path, "r", encoding="utf-8") as handle:
            self.zones = json.load(handle)
        if not isinstance(self.zones, list) or not self.zones:
            raise SystemExit("no zones found in %s" % zones_path)

        # Pristine copy so 'R' can restore a single zone without a reload.
        self.original = json.loads(json.dumps(self.zones))

        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise SystemExit("could not open video: %s" % video)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.frame_index = start_frame
        self.frame = self._read_frame(start_frame)
        if self.frame is None:
            raise SystemExit("could not read a frame from %s" % video)

        self.height, self.width = self.frame.shape[:2]
        self.scale = min(MAX_W / self.width, MAX_H / self.height, 1.0)
        self.view_w = int(self.width * self.scale)
        self.view_h = int(self.height * self.scale)

        self.index = 0                    # which zone is being edited
        self.dragging: Optional[int] = None
        self.hover: Optional[int] = None
        self.show_others = False
        self.dirty = False
        self.history: List[list] = []     # undo stack for the active zone
        self.redraw_points: Optional[List[Tuple[int, int]]] = None
        self.status = ""
        self.status_until = 0.0

    # ------------------------------------------------------------------ frame

    def _read_frame(self, index: int) -> Optional[np.ndarray]:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = self.cap.read()
        if not ok:
            # Long recordings often have unreliable frame-index seeking; fall
            # back to seeking by timestamp before giving up.
            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0, index) * 1000.0 / fps)
            ok, frame = self.cap.read()
        return frame if ok else None

    def step_frame(self, delta: int) -> None:
        target = max(0, self.frame_index + delta)
        frame = self._read_frame(target)
        if frame is None:
            self.notify("no frame at %d" % target)
            return
        self.frame_index = target
        self.frame = frame
        self.notify("frame %d" % target)

    # ------------------------------------------------------------------ zones

    @property
    def zone(self) -> dict:
        return self.zones[self.index]

    def points_px(self, zone: dict) -> List[Tuple[int, int]]:
        """Normalised polygon -> view pixels."""
        return [
            (int(x * self.width * self.scale), int(y * self.height * self.scale))
            for x, y in zone["polygon"]
        ]

    def set_points_px(self, zone: dict, points: List[Tuple[int, int]]) -> None:
        """View pixels -> normalised polygon, clamped to the frame."""
        zone["polygon"] = [
            [
                round(min(max(px / (self.width * self.scale), 0.0), 1.0), 5),
                round(min(max(py / (self.height * self.scale), 0.0), 1.0), 5),
            ]
            for px, py in points
        ]
        self.dirty = True

    def push_history(self) -> None:
        self.history.append(json.loads(json.dumps(self.zone["polygon"])))
        del self.history[:-40]

    def undo(self) -> None:
        if not self.history:
            self.notify("nothing to undo")
            return
        self.zone["polygon"] = self.history.pop()
        self.dirty = True
        self.notify("undo")

    def reset_zone(self) -> None:
        self.push_history()
        self.zone["polygon"] = json.loads(
            json.dumps(self.original[self.index]["polygon"])
        )
        self.dirty = True
        self.notify("reset to saved")

    def goto(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.zones)
        self.history.clear()
        self.dragging = None
        self.redraw_points = None
        self.notify("%s" % self.zone.get("name", self.zone.get("id")))

    # ------------------------------------------------------------------ mouse

    def on_mouse(self, event, x, y, flags, _param) -> None:
        if self.redraw_points is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.redraw_points.append((x, y))
            return

        points = self.points_px(self.zone)

        if event == cv2.EVENT_MOUSEMOVE and self.dragging is None:
            self.hover = _nearest(points, x, y, GRAB_RADIUS)

        elif event == cv2.EVENT_LBUTTONDOWN:
            hit = _nearest(points, x, y, GRAB_RADIUS)
            if hit is not None:
                self.push_history()
                self.dragging = hit
            else:
                edge = _nearest_edge(points, x, y, EDGE_RADIUS)
                if edge is not None:
                    self.push_history()
                    points.insert(edge + 1, (x, y))
                    self.set_points_px(self.zone, points)
                    self.dragging = edge + 1
                    self.notify("point added")

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            points[self.dragging] = (x, y)
            self.set_points_px(self.zone, points)

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            hit = _nearest(points, x, y, GRAB_RADIUS)
            if hit is not None:
                if len(points) <= 3:
                    self.notify("a polygon needs at least 3 points")
                    return
                self.push_history()
                points.pop(hit)
                self.set_points_px(self.zone, points)
                self.notify("point removed")

    # ------------------------------------------------------------------ render

    def notify(self, message: str) -> None:
        self.status = message
        self.status_until = time.time() + 2.5

    def render(self) -> np.ndarray:
        canvas = cv2.resize(self.frame, (self.view_w, self.view_h))

        if self.show_others:
            for i, zone in enumerate(self.zones):
                if i == self.index:
                    continue
                pts = np.array(self.points_px(zone), np.int32)
                if len(pts) >= 2:
                    cv2.polylines(canvas, [pts], True, OTHER, 1, cv2.LINE_AA)

        if self.redraw_points is not None:
            self._draw_redraw(canvas)
        else:
            self._draw_active(canvas)

        self._draw_hud(canvas)
        return canvas

    def _draw_active(self, canvas: np.ndarray) -> None:
        zone = self.zone
        color = hex_to_bgr(zone.get("color", ""))
        points = self.points_px(zone)
        pts = np.array(points, np.int32)

        if len(points) >= 3:
            shade = canvas.copy()
            cv2.fillPoly(shade, [pts], color)
            cv2.addWeighted(shade, 0.18, canvas, 0.82, 0, canvas)
            cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
        elif len(points) >= 2:
            cv2.polylines(canvas, [pts], False, color, 2, cv2.LINE_AA)

        for i, (px, py) in enumerate(points):
            grabbed = (i == self.dragging) or (i == self.hover)
            cv2.circle(canvas, (px, py), 7 if grabbed else 5, HANDLE, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 7 if grabbed else 5, POINT, 2, cv2.LINE_AA)
            cv2.putText(canvas, str(i), (px + 9, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, HANDLE, 1, cv2.LINE_AA)

    def _draw_redraw(self, canvas: np.ndarray) -> None:
        points = self.redraw_points or []
        if len(points) >= 2:
            cv2.polylines(canvas, [np.array(points, np.int32)], False,
                          (60, 200, 255), 2, cv2.LINE_AA)
        for px, py in points:
            cv2.circle(canvas, (px, py), 5, (60, 200, 255), -1, cv2.LINE_AA)

    def _draw_hud(self, canvas: np.ndarray) -> None:
        zone = self.zone
        name = zone.get("name", zone.get("id", "?"))
        lines = [
            "[%d/%d]  %s   (%s, %s)"
            % (self.index + 1, len(self.zones), name,
               zone.get("id", "?"), zone.get("zone_type", "?")),
            "points: %d    frame: %d    %s"
            % (len(zone["polygon"]), self.frame_index,
               "UNSAVED CHANGES" if self.dirty else "saved"),
        ]
        if self.redraw_points is not None:
            lines.append("REDRAW: click points, ENTER to accept, ESC to cancel")
        else:
            lines.append("drag=move  click edge=add  right-click=delete  "
                         "n/p=zone  r=redraw  u=undo  R=reset  o=others  s=save  q=quit")

        pad = 8
        box_h = 20 * len(lines) + pad
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (canvas.shape[1], box_h), DARK, -1)
        cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)
        for i, line in enumerate(lines):
            shade = TEXT if i else hex_to_bgr(zone.get("color", ""))
            cv2.putText(canvas, line, (10, 18 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, shade, 1, cv2.LINE_AA)

        if self.status and time.time() < self.status_until:
            cv2.putText(canvas, self.status, (10, canvas.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 160), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------- save

    def save(self) -> None:
        backup = self.zones_path.with_suffix(".json.bak")
        try:
            shutil.copyfile(self.zones_path, backup)
        except OSError:
            backup = None
        with open(self.zones_path, "w", encoding="utf-8") as handle:
            json.dump(self.zones, handle, indent=2)
            handle.write("\n")
        self.original = json.loads(json.dumps(self.zones))
        self.dirty = False
        self.notify("saved to %s" % self.zones_path.name)
        print("saved %d zones to %s%s"
              % (len(self.zones), self.zones_path,
                 " (backup: %s)" % backup.name if backup else ""))

    # -------------------------------------------------------------------- run

    def run(self) -> int:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)
        print(__doc__.split("Controls")[1] if "Controls" in __doc__ else "")
        print("editing %d zones from %s" % (len(self.zones), self.zones_path))
        self.notify(self.zone.get("name", ""))

        while True:
            cv2.imshow(WINDOW, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                # Window closed with the X button.
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                continue

            if self.redraw_points is not None:
                if key in (13, 10):          # ENTER
                    if len(self.redraw_points) >= 3:
                        self.push_history()
                        self.set_points_px(self.zone, self.redraw_points)
                        self.notify("redrawn with %d points" % len(self.redraw_points))
                        self.redraw_points = None
                    else:
                        self.notify("need at least 3 points")
                elif key == 27:              # ESC
                    self.redraw_points = None
                    self.notify("redraw cancelled")
                elif key == ord("c"):
                    self.redraw_points = []
                continue

            if key in (ord("q"), 27):
                if self.dirty and not self._confirm_discard():
                    continue
                break
            elif key == ord("n"):
                self.goto(1)
            elif key == ord("p"):
                self.goto(-1)
            elif key == ord("r"):
                self.redraw_points = []
                self.notify("redraw: click the new points")
            elif key == ord("u"):
                self.undo()
            elif key == ord("R"):
                self.reset_zone()
            elif key == ord("o"):
                self.show_others = not self.show_others
                self.notify("other zones %s" % ("shown" if self.show_others else "hidden"))
            elif key == ord("s"):
                self.save()
            elif key == ord("["):
                self.step_frame(-300)
            elif key == ord("]"):
                self.step_frame(300)
            elif key == ord("f"):
                self._jump_to_frame()

        cv2.destroyAllWindows()
        self.cap.release()
        return 0

    def _confirm_discard(self) -> bool:
        print("\nYou have unsaved ROI changes.")
        answer = input("Quit without saving? [y/N] ").strip().lower()
        return answer in ("y", "yes")

    def _jump_to_frame(self) -> None:
        try:
            raw = input("frame number (0-%d): " % max(0, self.total_frames - 1))
            target = int(raw.strip())
        except (ValueError, EOFError):
            self.notify("not a number")
            return
        frame = self._read_frame(target)
        if frame is None:
            self.notify("could not read frame %d" % target)
            return
        self.frame_index = target
        self.frame = frame


def _nearest(points, x, y, radius) -> Optional[int]:
    best, best_d = None, radius * radius
    for i, (px, py) in enumerate(points):
        d = (px - x) ** 2 + (py - y) ** 2
        if d <= best_d:
            best, best_d = i, d
    return best


def _nearest_edge(points, x, y, radius) -> Optional[int]:
    """Index of the edge (i -> i+1) the click is on, or None."""
    if len(points) < 2:
        return None
    best, best_d = None, float(radius)
    for i in range(len(points)):
        a = np.array(points[i], float)
        b = np.array(points[(i + 1) % len(points)], float)
        ab = b - a
        length = np.hypot(*ab)
        if length < 1e-6:
            continue
        t = max(0.0, min(1.0, np.dot(np.array([x, y], float) - a, ab) / (length ** 2)))
        dist = np.hypot(*(a + t * ab - np.array([x, y], float)))
        if dist < best_d:
            best, best_d = i, dist
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Adjust config/zones.json ROI polygons on a camera frame.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO),
                        help="video (or image) to draw the ROIs on")
    parser.add_argument("--zones", default=str(DEFAULT_ZONES),
                        help="zones JSON to edit (default config/zones.json)")
    parser.add_argument("--frame", type=int, default=0,
                        help="frame number to start on")
    parser.add_argument("--zone", default=None,
                        help="start on the first zone whose name/id contains this")
    parser.add_argument("--list", action="store_true",
                        help="list the zones and exit")
    args = parser.parse_args(argv)

    zones_path = Path(args.zones)
    if not zones_path.exists():
        print("error: zones file not found: %s" % zones_path, file=sys.stderr)
        return 2

    if args.list:
        with open(zones_path, "r", encoding="utf-8") as handle:
            for i, zone in enumerate(json.load(handle)):
                print("%2d  %-14s %-22s %-14s %d points"
                      % (i + 1, zone.get("id", "?"), zone.get("name", "?"),
                         zone.get("zone_type", "?"), len(zone.get("polygon", []))))
        return 0

    video = Path(args.video)
    if not video.exists():
        print("error: video not found: %s" % video, file=sys.stderr)
        return 2

    editor = RoiEditor(video, zones_path, args.frame)
    if args.zone:
        needle = args.zone.lower()
        for i, zone in enumerate(editor.zones):
            if needle in str(zone.get("name", "")).lower() or needle in str(zone.get("id", "")).lower():
                editor.index = i
                break
        else:
            print("note: no zone matching %r, starting at the first" % args.zone)
    return editor.run()


if __name__ == "__main__":
    raise SystemExit(main())
