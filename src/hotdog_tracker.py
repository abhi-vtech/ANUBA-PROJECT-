"""
hotdog_tracker.py
─────────────────
Pure ByteTrack / OC-SORT motion-and-position tracker with monotonic non-recycling track IDs,
persistent spatial position lock, sticky order attribution, and hand-gated ingredient logging.

Key Principles:
• Pure motion-and-position tracking (2D Kalman Filter + Spatial IoU & Centroid Position Lock).
• Optional appearance (colour-histogram) re-ID fallback for the wrap/assembly zone.
• Monotonic global ID allocation: Track IDs strictly increment and retired IDs are NEVER reused.
• Persistent spatial position lock (300px radius & 10s coasting buffer) ensures hotdogs at a station
  retain their exact ID throughout hand, bottle, and tool occlusions.
• DONE / WRAPPED tracks are permanently blocked from reactivation via _permanent_done_ids.
• Stalled-track watchdog: new spawn in wrap zone that matches a recently retired position
  triggers an automatic merge instead of fragmenting into a new ID.
• Sticky order attribution bound at creation time.
"""

from __future__ import annotations

import logging
import time
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from scipy.optimize import linear_sum_assignment

# Classes that are NOT the hotdog itself (i.e., items that can be "on" it)
ITEM_CLASSES = {
    "ketchup_sauce",
    "yellow_mustard_sauce",
    "relish",
    "onions",
    "diced_onions",
    "tomato",
    "pickle_rounds",
    "pickle_spears",
    "sport_peppers",
    "chilli",
    "chili",
    "yellow_cheese",
    "swiss_cheese",
    "grated_yellow_cheese",
    "burger_bun",
}

ITEM_CLASS_ALIASES = {
    "sport (wax) peppers": "sport_peppers",
    "sport peppers": "sport_peppers",
    "sport wax peppers": "sport_peppers",
    "wax peppers": "sport_peppers",
    "chilli": "chilli",
    "chili": "chilli",
    "pickel swears": "pickle_spears",
    "pickles (spears)": "pickle_spears",
    "pickle spears": "pickle_spears",
    "pickle (spears)": "pickle_spears",
    "pickles": "pickle_spears",
    "pickle": "pickle_spears",
    "pickle_rounds": "pickle_rounds",
    "pickle rounds": "pickle_rounds",
    "sweet relish": "relish",
    "relish": "relish",
    "onions": "onions",
    "diced onions": "onions",
    "diced_onions": "onions",
}

logger = logging.getLogger(__name__)

PROXIMITY_PAD_PX = 80
# Dry ingredients are credited ONLY by a completed well trip: the hand dips
# into an ingredient well and comes back to a hotdog (TemporalTracker's
# trajectory mode -> force_commit_item).  With this True the dwell-based path
# below is limited to sauces, where the bottle itself is the visible evidence.
#
# The dwell path used to credit a dry ingredient whenever its detection simply
# sat near a hotdog for DEFAULT_ITEM_DWELL_S with a hand nearby -- no trip
# required -- so it added ingredients independently of, and often before, the
# trip logic.  Set False to restore that behaviour.
DRY_ITEMS_REQUIRE_WELL_TRIP = False

DEFAULT_ITEM_DWELL_S = 0.5

DEFAULT_ITEM_DWELL_OVERRIDES = {
    "ketchup_sauce":        0.05,
    "yellow_mustard_sauce": 0.05,
    "yellow_cheese":        0.2,
    "pickle_spears":        0.15,
    "pickle_rounds":        0.15,
    "sport_peppers":        0.2,
}

SAUCE_STRICT_OVERLAP_CLASSES: set[str] = {"ketchup_sauce", "yellow_mustard_sauce"}
SAUCE_MAX_COMMITS: int = 2
MAX_CENTROID_FALLBACK_PX: int = 150

REQUIRE_HAND_PROXIMITY: bool = True
HAND_PAD_PX:            int  = 80

SAUCE_SHAPE_CLASSES: set[str] = {"ketchup_sauce", "yellow_mustard_sauce"}
MIN_SAUCE_ASPECT_RATIO: float = 0.8
MIN_SAUCE_HEIGHT_PX:    int   = 30
MAX_SAUCE_INSTANCES:    int   = 1
MIN_SAUCE_CONFIDENCE:  float = 0.55

# ── Hand-as-Bridge attribution constants ──────────────────────────────────────
HAND_BBOX_WIDTH_RATIO:  float = 0.6   # center 60 % of hand width
HAND_BBOX_HEIGHT_RATIO: float = 0.6   # bottom 60 % of hand height

# Radius: shrunk-hand working-point → item centroid (primary attribution path)
HAND_ITEM_RADIUS:   int = 250
# Radius: shrunk-hand working-point → hotdog centroid (selects which hotdog gets item)
HAND_HOTDOG_RADIUS: int = 350
# Direct item→hotdog fallback radius when no hand is visible at all
DIRECT_FALLBACK_PX: int = 250


def _shrink_hand_bbox(
    bbox: Tuple[int, int, int, int],
    width_ratio: float = HAND_BBOX_WIDTH_RATIO,
    height_ratio: float = HAND_BBOX_HEIGHT_RATIO,
) -> Tuple[int, int, int, int]:
    """
    Return a tighter bounding box representing the palm/fingertip region of a
    hand detection.  The raw YOLO hand bbox often covers the full arm+hand area
    and will overlap adjacent hotdogs in tight assembly situations.  Keeping
    only the bottom-center portion ensures the returned bbox touches only the
    hotdog the worker is actively working on.

    width_ratio  – fraction of the original width kept (centred on the bbox x-axis).
    height_ratio – fraction of the original height kept (from the BOTTOM up).
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) / 2.0
    new_x1 = int(cx - (w * width_ratio) / 2.0)
    new_x2 = int(cx + (w * width_ratio) / 2.0)
    new_y1 = int(y2 - h * height_ratio)   # bottom portion only (fingertips)
    new_y2 = y2
    return (new_x1, new_y1, new_x2, new_y2)


def _hand_working_point(
    bbox: Tuple[int, int, int, int],
) -> Tuple[float, float]:
    """
    Return the fingertip proxy point for a hand detection: the bottom-center
    of the shrunk hand bbox.  This is the most precise estimate of where the
    worker's fingers are making contact with food.
    """
    sx1, sy1, sx2, sy2 = _shrink_hand_bbox(bbox)
    return ((sx1 + sx2) / 2.0, float(sy2))


def _compute_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    inter = w * h
    if inter == 0:
        return 0.0
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def _centroid_distance(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    cx1 = (bbox1[0] + bbox1[2]) / 2.0
    cy1 = (bbox1[1] + bbox1[3]) / 2.0
    cx2 = (bbox2[0] + bbox2[2]) / 2.0
    cy2 = (bbox2[1] + bbox2[3]) / 2.0
    return float(((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5)


def _hand_touching_bbox(
    hand_dets: list,
    target_bbox: Tuple[int, int, int, int],
    pad: int = 20,
) -> bool:
    """
    Return True when any hand's SHRUNK (palm/fingertip) bbox overlaps the
    padded target_bbox.  Using the shrunk bbox prevents a large hand detection
    from falsely registering contact with an adjacent hotdog in tight
    multi-hotdog assembly scenarios.
    """
    x1, y1, x2, y2 = target_bbox
    px1, py1, px2, py2 = x1 - pad, y1 - pad, x2 + pad, y2 + pad
    for h in hand_dets:
        # Use SHRUNK hand bbox — palm/fingertip region only
        shx1, shy1, shx2, shy2 = _shrink_hand_bbox(h.bbox)
        if shx2 >= px1 and shx1 <= px2 and shy2 >= py1 and shy1 <= py2:
            return True
    return False


class KalmanBoxFilter:
    """
    2D Kalman Filter for bounding box tracking (ByteTrack / OC-SORT style).
    State vector: [cx, cy, w, h, v_cx, v_cy, v_w, v_h]
    """
    def __init__(self, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))

        self.mean = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.covariance = np.eye(8, dtype=np.float32) * 10.0
        self.covariance[4:, 4:] *= 100.0

        self._F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self._F[i, i + 4] = 1.0

        self._H = np.eye(4, 8, dtype=np.float32)
        self._Q = np.eye(8, dtype=np.float32)
        self._Q[:4, :4] *= 1.0
        self._Q[4:, 4:] *= 10.0
        self._R = np.eye(4, dtype=np.float32) * 1.0

    def predict(self) -> Tuple[int, int, int, int]:
        self.mean = np.dot(self._F, self.mean)
        self.covariance = np.dot(np.dot(self._F, self.covariance), self._F.T) + self._Q
        return self.get_bbox()

    def update(self, bbox: Tuple[int, int, int, int]):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, float(x2 - x1))
        h = max(1.0, float(y2 - y1))
        z = np.array([cx, cy, w, h], dtype=np.float32)

        y = z - np.dot(self._H, self.mean)
        S = np.dot(np.dot(self._H, self.covariance), self._H.T) + self._R
        K = np.dot(np.dot(self.covariance, self._H.T), np.linalg.inv(S))

        self.mean = self.mean + np.dot(K, y)
        self.covariance = self.covariance - np.dot(np.dot(K, self._H), self.covariance)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        cx, cy, w, h = self.mean[:4]
        w = max(1.0, w)
        h = max(1.0, h)
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return (x1, y1, x2, y2)


def format_time_str(ts: Optional[float]) -> Optional[str]:
    """Format seconds timestamp to HH:MM:SS or string representation."""
    if ts is None:
        return None
    if ts < 86400:
        m, s = divmod(int(ts), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return time.strftime("%H:%M:%S", time.localtime(ts))


MAX_HAND_SPEED_PX_PER_FRAME: float = 55.0
HAND_LOST_TIMEOUT_FRAMES: int = 180
HAND_ASSOC_FRAME_WINDOW: int = 5
BASE_HAND_DISPLACEMENT_PX: float = 300.0


@dataclass
class HandTrack:
    """Tracks a single hand continuously across frames."""
    hand_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: float
    last_frame: int
    working_pos: Tuple[float, float]


@dataclass
class HotdogRecord:
    """Tracks one physical hotdog across frames with monotonic ID."""
    hotdog_id: str          # "1", "2", ... (monotonic ID string)
    track_id: int           # monotonic integer ID
    first_seen: float       # Unix timestamp / video timestamp
    last_seen: float        # Unix timestamp / video timestamp
    order_id: Optional[str] = None  # Sticky order attribution
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    items_added: List[Dict] = field(default_factory=list)
    trail: deque = field(default_factory=deque, repr=False)
    kalman: Optional[KalmanBoxFilter] = field(default=None, repr=False)
    retired: bool = False
    is_coasting: bool = False
    last_hand_id: Optional[int] = None  # continuous hand_id of carrying hand
    last_hand_working_pos: Optional[Tuple[float, float]] = None  # (cx, cy) of working hand
    last_hand_contact_time: Optional[float] = None  # Timestamp of last hand contact
    last_hand_contact_frame: int = 0  # Frame index of last hand contact
    was_hand_carried: bool = False  # Set True when touched by hand before/during occlusion
    # BGR colour histogram (retained for backward compatibility)
    bgr_histogram: Optional[np.ndarray] = field(default=None, repr=False)
    _item_counts: Dict[str, int] = field(default_factory=dict, repr=False)
    _seen_items: set[str] = field(default_factory=set, repr=False)
    _committed_items: set = field(default_factory=set, repr=False)
    # Wrapping lifecycle tracking
    wrapping_dwell_start: Optional[float] = None
    wrapping_closing_time: Optional[float] = None
    wrapping_done_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "active"

    def add_trail_point(
        self,
        timestamp: float,
        maxlen: int = 60,
        alpha: float = 0.45,
        anchor: str = "point",
    ) -> None:
        x1, y1, x2, y2 = self.bbox
        if anchor == "top_left":
            raw_x, raw_y = float(x1), float(y1)
        elif anchor == "bottom_center":
            raw_x, raw_y = (x1 + x2) / 2.0, float(y2)
        else:
            raw_x, raw_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        if self.trail:
            last = self.trail[-1]
            dx = abs(raw_x - last["x"])
            dy = abs(raw_y - last["y"])
            if dx < 2.0 and dy < 2.0:
                smoothed_x = last["x"]
                smoothed_y = last["y"]
            else:
                smoothed_x = int(alpha * raw_x + (1.0 - alpha) * last["x"])
                smoothed_y = int(alpha * raw_y + (1.0 - alpha) * last["y"])
        else:
            smoothed_x = int(raw_x)
            smoothed_y = int(raw_y)

        if self.trail.maxlen != maxlen:
            old_items = list(self.trail)
            self.trail = deque(old_items, maxlen=maxlen)

        self.trail.append({
            "x": smoothed_x,
            "y": smoothed_y,
            "timestamp": round(timestamp, 3),
        })

    def distance_to_trail(
        self,
        cx: float,
        cy: float,
        max_time_gap: float = 30.0,
        now: float = 0.0,
    ) -> float:
        """Return minimum Euclidean distance from (cx, cy) to any recent point in this hotdog's trail."""
        min_d = float('inf')
        if self.trail:
            for pt in reversed(self.trail):
                if now > 0 and (now - pt.get("timestamp", now)) > max_time_gap:
                    break
                d = ((cx - pt["x"]) ** 2 + (cy - pt["y"]) ** 2) ** 0.5
                if d < min_d:
                    min_d = d
        if min_d == float('inf'):
            bx = (self.bbox[0] + self.bbox[2]) / 2.0
            by = (self.bbox[1] + self.bbox[3]) / 2.0
            min_d = ((cx - bx) ** 2 + (cy - by) ** 2) ** 0.5
        return float(min_d)

    def record_item(self, class_name: str, timestamp: float) -> bool:
        count = self._item_counts.get(class_name, 0) + 1
        self._item_counts[class_name] = count
        self._seen_items.add(class_name)

        if timestamp < 86400:
            m, s = divmod(int(timestamp), 60)
            h, m = divmod(m, 60)
            t_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            t_str = time.strftime("%H:%M:%S", time.localtime(timestamp))

        self.items_added.append({
            "hotdog_id": self.hotdog_id,
            "item": class_name,
            "count": count,
            "timestamp": round(timestamp, 2),
            "video_timestamp_s": round(timestamp, 2),
            "time_str": t_str,
        })
        return True

    @property
    def item_names(self) -> List[str]:
        return [e["item"] for e in self.items_added]


class HotdogTracker:
    """
    Pure ByteTrack / OC-SORT motion tracker with monotonic non-recycling track IDs,
    persistent spatial position lock, and hand-proximity tie-breaking.
    Call update() once per frame with detections and optional active ticket ID.
    """

    def __init__(
        self,
        proximity_pad: int = PROXIMITY_PAD_PX,
        orphan_timeout_s: float = 10.0,    # 300 frames at 30 FPS
        spatial_lock_radius: float = 300.0, # 300px spatial position lock radius
        item_dwell_s: float = DEFAULT_ITEM_DWELL_S,
        item_dwell_overrides: Optional[Dict[str, float]] = None,
        min_sauce_aspect: float = MIN_SAUCE_ASPECT_RATIO,
        min_sauce_height: int = MIN_SAUCE_HEIGHT_PX,
        min_sauce_confidence: float = MIN_SAUCE_CONFIDENCE,
        max_sauce_instances: int = MAX_SAUCE_INSTANCES,
        require_hand_proximity: bool = REQUIRE_HAND_PROXIMITY,
        hand_pad: int = HAND_PAD_PX,
        iou_threshold: float = 0.30,
        trail_maxlen: int = 60,
        trail_smooth_alpha: float = 0.35,
        trail_anchor: str = "bottom_center",
        retain_lost_trails: bool = True,
        # ── Wrap-zone ROI fixes ──────────────────────────────────────────────
        wrap_zone_poly: Optional[List[Tuple[float, float]]] = None,
        wrap_spatial_lock_radius: float = 350.0,  # wider lock inside wrap zone
        wrap_orphan_timeout_s: float = 4.0,        # longer coast in wrap zone
        wrap_stall_watchdog_s: float = 5.0,        # merge window after retire+spawn
        hand_transit_reid_enabled: bool = True,    # Hand-transition occlusion transfer re-ID
        hand_transit_max_gap_s: float = 60.0,      # Max transit gap (seconds)
        hand_placement_radius: float = 400.0,      # Search radius around active hand placement
        max_hand_speed_px_per_frame: float = MAX_HAND_SPEED_PX_PER_FRAME,
        hand_lost_timeout_frames: int = HAND_LOST_TIMEOUT_FRAMES,
        hand_assoc_frame_window: int = HAND_ASSOC_FRAME_WINDOW,
        base_hand_displacement_px: float = BASE_HAND_DISPLACEMENT_PX,
        appearance_reid_enabled: Optional[bool] = None,  # Backward-compatible alias
        appearance_reid_thresh: Optional[float] = None,  # Backward-compatible alias
    ):
        self._pad = proximity_pad
        self._orphan_timeout = orphan_timeout_s
        self._spatial_lock_radius = spatial_lock_radius
        self._item_dwell_s = item_dwell_s
        self._item_dwell_overrides = dict(DEFAULT_ITEM_DWELL_OVERRIDES)
        if item_dwell_overrides:
            self._item_dwell_overrides.update(item_dwell_overrides)
        self._min_sauce_aspect = min_sauce_aspect
        self._min_sauce_height = min_sauce_height
        self._min_sauce_confidence = min_sauce_confidence
        self._max_sauce_instances = max_sauce_instances
        self._require_hand_proximity = require_hand_proximity
        self._hand_pad = hand_pad
        self._iou_threshold = iou_threshold
        self._trail_maxlen = trail_maxlen
        self._trail_smooth_alpha = trail_smooth_alpha
        self._trail_anchor = trail_anchor
        self._retain_lost_trails = retain_lost_trails

        # ── Wrap-zone ROI configuration ──────────────────────────────────────
        # wrap_zone_poly: normalised [(x,y),...] polygon for the assembly/wrap zone.
        # When None, wrap-zone-specific behaviours apply globally (safe fallback).
        self._wrap_zone_poly: Optional[np.ndarray] = (
            np.array(wrap_zone_poly, dtype=np.float32) if wrap_zone_poly else None
        )
        self._wrap_spatial_lock_radius = wrap_spatial_lock_radius
        self._wrap_orphan_timeout = wrap_orphan_timeout_s
        self._wrap_stall_watchdog_s = wrap_stall_watchdog_s
        self._hand_transit_reid_enabled = (
            appearance_reid_enabled if appearance_reid_enabled is not None else hand_transit_reid_enabled
        )
        self._hand_transit_max_gap_s = hand_transit_max_gap_s
        self._hand_placement_radius = hand_placement_radius
        self._max_hand_speed_px_per_frame = max_hand_speed_px_per_frame
        self._hand_lost_timeout_frames = hand_lost_timeout_frames
        self._hand_assoc_frame_window = hand_assoc_frame_window
        self._base_hand_displacement_px = base_hand_displacement_px

        # Kept for backward compatibility
        self._appearance_reid_enabled = self._hand_transit_reid_enabled
        self._appearance_reid_thresh = appearance_reid_thresh if appearance_reid_thresh is not None else 0.45

        # Global Monotonic ID Counter (strictly increments, never recycled)
        self._next_monotonic_id: int = 1
        self._frame_count: int = 0
        self._next_hand_id: int = 1
        self._active_hands: Dict[int, HandTrack] = {}

        # Active tid -> HotdogRecord
        self._records: Dict[int, HotdogRecord] = {}
        self._retired_records: Dict[int, HotdogRecord] = {}

        # Map detector track_id (if present) -> monotonic integer track ID
        self._detector_id_map: Dict[int, int] = {}

        self._dwell_start: Dict[Tuple[int, str], float] = {}
        self._committed_pairs: Set[Tuple[int, str]] = set()
        self._last_time: Optional[float] = None

        # Fix A: permanent DONE set — survives between frames, never cleared
        # This is belt-and-suspenders on top of the done_ids parameter passed per-frame.
        self._permanent_done_ids: Set[int] = set()

        # Regression Instrumentation Counters
        self._occlusion_events: int = 0
        self._occlusion_rematches: int = 0
        self._neighbor_swaps: int = 0
        self._id_recycled_after_exit: int = 0
        self._wrap_zone_merges: int = 0  # Fix C counter

        # Per-hand lost bookkeeping:
        # hand_id -> {"tid": int, "lost_frame": int, "lost_time": float, "lost_position": Tuple[float, float]}
        self._last_lost_by_hand: Dict[int, Dict[str, Any]] = {}

    def _normalize_item_class_name(self, name: str) -> str:
        name_clean = name.strip().lower()
        return ITEM_CLASS_ALIASES.get(name_clean, name_clean.replace(" ", "_"))

    def _update_hands(self, hand_dets: list, now: float) -> List[Tuple[Any, int]]:
        """
        Track hands across frames assigning a continuous hand_id via detector track_id, IoU, and proximity.
        Returns a list of (hand_det, hand_id) pairs.
        """
        matched_pairs: List[Tuple[Any, int]] = []
        unmatched_hand_dets = list(hand_dets)
        unmatched_hand_ids = set(self._active_hands.keys())

        # 1. First pass: Match by detector track_id if available (> 0)
        for h_det in list(unmatched_hand_dets):
            if hasattr(h_det, 'track_id') and h_det.track_id is not None and h_det.track_id > 0:
                hid = h_det.track_id
                if hid in unmatched_hand_ids:
                    htrack = self._active_hands[hid]
                    htrack.bbox = h_det.bbox
                    htrack.last_seen = now
                    htrack.last_frame = self._frame_count
                    htrack.working_pos = _hand_working_point(h_det.bbox)
                    matched_pairs.append((h_det, hid))
                    unmatched_hand_ids.remove(hid)
                    unmatched_hand_dets.remove(h_det)

        # 2. Second pass: Match remaining hands by IoU >= 0.10 or centroid distance <= 350px
        for h_det in list(unmatched_hand_dets):
            best_hid = None
            best_score = float('inf')

            hx_c = (h_det.bbox[0] + h_det.bbox[2]) / 2.0
            hy_c = (h_det.bbox[1] + h_det.bbox[3]) / 2.0

            for hid in list(unmatched_hand_ids):
                htrack = self._active_hands[hid]
                iou = _compute_iou(h_det.bbox, htrack.bbox)
                bx_c = (htrack.bbox[0] + htrack.bbox[2]) / 2.0
                by_c = (htrack.bbox[1] + htrack.bbox[3]) / 2.0
                dist = ((hx_c - bx_c) ** 2 + (hy_c - by_c) ** 2) ** 0.5

                if iou >= 0.10 or dist <= 350.0:
                    score = (1.0 - iou) * 50.0 + dist
                    if score < best_score:
                        best_score = score
                        best_hid = hid

            if best_hid is not None:
                htrack = self._active_hands[best_hid]
                htrack.bbox = h_det.bbox
                htrack.last_seen = now
                htrack.last_frame = self._frame_count
                htrack.working_pos = _hand_working_point(h_det.bbox)
                matched_pairs.append((h_det, best_hid))
                unmatched_hand_ids.remove(best_hid)
                unmatched_hand_dets.remove(h_det)

        # 3. Third pass: For remaining unmatched hand detections, assign new / detector hand_id
        for h_det in unmatched_hand_dets:
            if hasattr(h_det, 'track_id') and h_det.track_id is not None and h_det.track_id > 0:
                new_hid = h_det.track_id
            else:
                new_hid = self._next_hand_id
                self._next_hand_id += 1

            htrack = HandTrack(
                hand_id=new_hid,
                bbox=h_det.bbox,
                last_seen=now,
                last_frame=self._frame_count,
                working_pos=_hand_working_point(h_det.bbox),
            )
            self._active_hands[new_hid] = htrack
            matched_pairs.append((h_det, new_hid))

        # Age out inactive hands older than 2.0s
        for hid in list(self._active_hands.keys()):
            if (now - self._active_hands[hid].last_seen) > 2.0:
                del self._active_hands[hid]

        return matched_pairs

    def update(
        self,
        detections: list,
        current_time: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        active_ticket_id: Optional[str] = None,
        done_ids: Optional[Set[int]] = None,
        expected_hotdogs: Optional[int] = None,
    ) -> None:
        self._frame_count += 1
        now = current_time if current_time is not None else time.time()
        self._last_time = now

        # Fix A: accumulate into permanent done set so the guard works even if
        # done_ids is not passed every frame (belt-and-suspenders).
        if done_ids:
            self._permanent_done_ids.update(done_ids)

        # Permanently retire wrapped / completed hotdogs (done_ids)
        _effective_done = self._permanent_done_ids
        if done_ids:
            for tid in list(self._records.keys()):
                if tid in _effective_done:
                    rec = self._records.pop(tid)
                    rec.retired = True
                    rec.is_coasting = False
                    rec.trail.clear()
                    self._retired_records[tid] = rec

        # Clean expired entries in _last_lost_by_hand
        expired_hids = [
            hid for hid, entry in self._last_lost_by_hand.items()
            if (self._frame_count - entry["lost_frame"]) > self._hand_lost_timeout_frames
            or (now - entry["lost_time"]) > self._hand_transit_max_gap_s
        ]
        for hid in expired_hids:
            del self._last_lost_by_hand[hid]

        # Extract hotdog and hand detections directly
        MIN_HOTDOG_POLY_AREA_PX = 1500.0
        hotdog_dets = []
        for d in detections:
            if d.class_name == "hot-dog":
                area = 0.0
                if getattr(d, "polygon", None) and len(d.polygon) >= 3:
                    pts = np.array(d.polygon, dtype=np.float32)
                    area = cv2.contourArea(pts)
                if area >= MIN_HOTDOG_POLY_AREA_PX:
                    hotdog_dets.append(d)
                else:
                    logger.debug(f"[FILTER] Rejected hot-dog small chunk with polygon area {area:.1f}px (track_id {d.track_id})")

        hand_dets = [d for d in detections if d.class_name == "hand"]

        # Track hand identities across frames
        hand_pairs = self._update_hands(hand_dets, now)

        item_dets = []
        for det in detections:
            if det.class_name in ("hot-dog", "hand"):
                continue
            norm_name = self._normalize_item_class_name(det.class_name)
            if norm_name in ITEM_CLASSES:
                item_dets.append(det)

        # Update last known hand working position for active records touching a hand.
        for rec in self._records.values():
            for h_det, hid in hand_pairs:
                if _boxes_overlap(h_det.bbox, rec.bbox) or _hand_touching_bbox([h_det], rec.bbox, pad=self._hand_pad):
                    rec.last_hand_id = hid
                    rec.last_hand_working_pos = _hand_working_point(h_det.bbox)
                    rec.last_hand_contact_time = now
                    rec.last_hand_contact_frame = self._frame_count
                    rec.was_hand_carried = True

        # 1. Predict Kalman filter states for active/coasting tracks
        active_tids = list(self._records.keys())
        predicted_bboxes = {}
        for tid, rec in self._records.items():
            if rec.kalman is not None:
                predicted_bboxes[tid] = rec.kalman.predict()
            else:
                predicted_bboxes[tid] = rec.bbox

        matched_records: Dict[int, Tuple[int, int, int, int]] = {}
        unmatched_dets = list(hotdog_dets)
        unmatched_track_ids = set(active_tids)
        self._active_detector_ids = set()

        # Pass 0: Direct ByteTrack track_id matching for known active tracks
        bytetrack_matched_indices = set()
        for i, det in enumerate(unmatched_dets):
            if hasattr(det, 'track_id') and det.track_id is not None and det.track_id != -1:
                det_tid = det.track_id
                if det_tid in self._detector_id_map:
                    mon_tid = self._detector_id_map[det_tid]
                    if mon_tid in unmatched_track_ids:
                        pred_bbox = predicted_bboxes[mon_tid]
                        rec_bbox = self._records[mon_tid].bbox
                        iou = max(_compute_iou(det.bbox, pred_bbox), _compute_iou(det.bbox, rec_bbox))
                        d_spatial = min(_centroid_distance(det.bbox, pred_bbox), _centroid_distance(det.bbox, rec_bbox))
                        time_lost = now - self._records[mon_tid].last_seen
                        max_dist = min(self._spatial_lock_radius, 40.0 + 20.0 * time_lost)
                        
                        if d_spatial <= max_dist and (iou >= 0.10 or d_spatial <= 30.0):
                            matched_records[mon_tid] = det.bbox
                            unmatched_track_ids.remove(mon_tid)
                            bytetrack_matched_indices.add(i)
                            self._detector_id_map[det_tid] = mon_tid
                            self._active_detector_ids.add(det_tid)

        if bytetrack_matched_indices:
            unmatched_dets = [d for idx, d in enumerate(unmatched_dets) if idx not in bytetrack_matched_indices]

        # Pass 1: Bounding Box IoU matching against predicted Kalman positions & confirmed positions
        if unmatched_dets and unmatched_track_ids:
            unmatched_tid_list = list(unmatched_track_ids)
            iou_matrix = np.zeros((len(unmatched_dets), len(unmatched_tid_list)), dtype=np.float32)
            for i, det in enumerate(unmatched_dets):
                for j, tid in enumerate(unmatched_tid_list):
                    iou_matrix[i, j] = max(
                        _compute_iou(det.bbox, predicted_bboxes[tid]),
                        _compute_iou(det.bbox, self._records[tid].bbox),
                    )

            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_det_indices = set()
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= min(0.25, self._iou_threshold):
                    det = unmatched_dets[r]
                    tid = unmatched_tid_list[c]
                    
                    pred_bbox = predicted_bboxes[tid]
                    rec_bbox = self._records[tid].bbox
                    d_spatial = min(_centroid_distance(det.bbox, pred_bbox), _centroid_distance(det.bbox, rec_bbox))
                    time_lost = now - self._records[tid].last_seen
                    max_dist = min(self._spatial_lock_radius, 40.0 + 20.0 * time_lost)
                    
                    if d_spatial > max_dist:
                        continue
                        
                    matched_records[tid] = det.bbox
                    unmatched_track_ids.remove(tid)
                    matched_det_indices.add(r)
                    if hasattr(det, 'track_id') and det.track_id is not None:
                        self._detector_id_map[det.track_id] = tid
                        self._active_detector_ids.add(det.track_id)

            unmatched_dets = [d for idx, d in enumerate(unmatched_dets) if idx not in matched_det_indices]

        # Detector track_id pass for leftover detections (if detector track_id is recognized)
        yolo_matched_dets = []
        for det in list(unmatched_dets):
            if hasattr(det, 'track_id') and det.track_id is not None and det.track_id != -1:
                det_tid = det.track_id
                if det_tid in self._detector_id_map and self._detector_id_map[det_tid] in unmatched_track_ids:
                    mon_tid = self._detector_id_map[det_tid]
                    pred_bbox = predicted_bboxes[mon_tid]
                    rec_bbox = self._records[mon_tid].bbox
                    d_spatial = min(_centroid_distance(det.bbox, pred_bbox), _centroid_distance(det.bbox, rec_bbox))
                    time_lost = now - self._records[mon_tid].last_seen
                    max_dist = min(self._spatial_lock_radius, 40.0 + 20.0 * time_lost)
                    
                    if d_spatial <= max_dist:
                        matched_records[mon_tid] = det.bbox
                        unmatched_track_ids.remove(mon_tid)
                        yolo_matched_dets.append(det)
                        self._active_detector_ids.add(det_tid)

        for d in yolo_matched_dets:
            unmatched_dets.remove(d)

        # Pre-compute current hand working-points (fingertip proxies) for all active hands
        current_hand_pts: List[Tuple[float, float]] = [
            _hand_working_point(h.bbox) for h in hand_dets
        ]

        # ── Pass 2: Local Station Spatial Lock on Active Tracks (STATIONARY LOCK <= 160px) ─
        # Guarantees that stationary hotdogs resting at their stations NEVER jump or swap IDs.
        # Localized matching allows static rack hotdogs to match without interference.
        if unmatched_dets and unmatched_track_ids:
            remaining_unmatched_dets = []
            for det in unmatched_dets:
                best_tid = None
                best_dist = 160.0  # Tight stationary lock radius

                for tid in list(unmatched_track_ids):
                    rec = self._records[tid]
                    pred_bbox = predicted_bboxes[tid]
                    d_spatial = min(
                        _centroid_distance(det.bbox, pred_bbox),
                        _centroid_distance(det.bbox, rec.bbox),
                    )
                    if d_spatial <= best_dist:
                        best_dist = d_spatial
                        best_tid = tid

                if best_tid is not None:
                    matched_records[best_tid] = det.bbox
                    unmatched_track_ids.remove(best_tid)
                    if hasattr(det, 'track_id') and det.track_id is not None:
                        self._detector_id_map[det.track_id] = best_tid
                else:
                    remaining_unmatched_dets.append(det)

            unmatched_dets = remaining_unmatched_dets

        # ── Pass 3: Hand-Transition In-Flight Relocation (IDENTITY & DISPLACEMENT BOUNDED) ───
        # When a worker physically carries a hotdog from Station A to Station B:
        # ONLY the specific track ID lost by the SAME hand is recovered, provided the displacement
        # is physically plausible within max_hand_speed_px_per_frame over elapsed frames.
        if self._hand_transit_reid_enabled and unmatched_dets and hand_pairs and self._last_lost_by_hand:
            still_unmatched_after_hand_transit: list = []

            for det in unmatched_dets:
                det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                det_cy = (det.bbox[1] + det.bbox[3]) / 2.0

                # Find which active hand(s) are touching/near this detection
                matched_hid = None
                for h_det, hid in hand_pairs:
                    if _boxes_overlap(h_det.bbox, det.bbox) or _hand_touching_bbox([h_det], det.bbox, pad=self._hand_pad + 20):
                        matched_hid = hid
                        break
                    # Fingertip check
                    hw_x, hw_y = _hand_working_point(h_det.bbox)
                    if ((det_cx - hw_x) ** 2 + (det_cy - hw_y) ** 2) ** 0.5 <= 100.0:
                        matched_hid = hid
                        break

                recovered_tid: Optional[int] = None

                if matched_hid is not None and matched_hid in self._last_lost_by_hand:
                    entry = self._last_lost_by_hand[matched_hid]
                    cand_tid = entry["tid"]
                    if cand_tid not in self._permanent_done_ids:
                        elapsed_frames = max(1, self._frame_count - entry["lost_frame"])
                        elapsed_time = max(0.0, now - entry["lost_time"])
                        effective_elapsed_frames = max(float(elapsed_frames), elapsed_time * 30.0)
                        lost_x, lost_y = entry["lost_position"]
                        displacement = ((det_cx - lost_x) ** 2 + (det_cy - lost_y) ** 2) ** 0.5
                        max_plausible_displacement = self._base_hand_displacement_px + effective_elapsed_frames * self._max_hand_speed_px_per_frame

                        if displacement <= max_plausible_displacement:
                            recovered_tid = cand_tid
                            del self._last_lost_by_hand[matched_hid]
                            logger.info(
                                "[HAND_TRANSIT_BOUNDED] Hand #%d placed Hotdog #%d at (%.0f, %.0f) "
                                "(displacement=%.1fpx <= max=%.1fpx, frames=%d, time=%.2fs)",
                                matched_hid, recovered_tid, det_cx, det_cy, displacement,
                                max_plausible_displacement, elapsed_frames, elapsed_time,
                            )
                        else:
                            logger.info(
                                "[HAND_TRANSIT_REJECTED] Hand #%d placement at (%.0f, %.0f) rejected: "
                                "displacement=%.1fpx > max=%.1fpx for %d frames",
                                matched_hid, det_cx, det_cy, displacement, max_plausible_displacement, elapsed_frames,
                            )

                if recovered_tid is not None:
                    if recovered_tid in self._retired_records:
                        rec = self._retired_records.pop(recovered_tid)
                        rec.retired = False
                        rec.is_coasting = False
                        rec.last_seen = now
                        rec.bbox = det.bbox
                        rec.kalman = KalmanBoxFilter(det.bbox)
                        rec.was_hand_carried = False
                        rec.last_hand_id = matched_hid
                        rec.last_hand_contact_frame = self._frame_count
                        rec.add_trail_point(
                            timestamp=now,
                            maxlen=self._trail_maxlen,
                            alpha=self._trail_smooth_alpha,
                            anchor=self._trail_anchor,
                        )
                        self._records[recovered_tid] = rec
                    elif recovered_tid in self._records:
                        rec = self._records[recovered_tid]
                        rec.is_coasting = False
                        rec.last_seen = now
                        rec.bbox = det.bbox
                        rec.kalman = KalmanBoxFilter(det.bbox)
                        rec.was_hand_carried = False
                        rec.last_hand_id = matched_hid
                        rec.last_hand_contact_frame = self._frame_count
                        rec.add_trail_point(
                            timestamp=now,
                            maxlen=self._trail_maxlen,
                            alpha=self._trail_smooth_alpha,
                            anchor=self._trail_anchor,
                        )
                        if recovered_tid in unmatched_track_ids:
                            unmatched_track_ids.remove(recovered_tid)

                    matched_records[recovered_tid] = det.bbox
                    if hasattr(det, 'track_id') and det.track_id is not None:
                        self._detector_id_map[det.track_id] = recovered_tid
                        self._active_detector_ids.add(det.track_id)
                else:
                    still_unmatched_after_hand_transit.append(det)

            unmatched_dets = still_unmatched_after_hand_transit

        # ── Pass 4: Wider Spatial Lock for Active Tracks (Radius <= spatial_lock_radius) ───
        if unmatched_dets and unmatched_track_ids:
            remaining_unmatched_dets = []

            for det in unmatched_dets:
                best_tid = None
                best_score = float('inf')

                det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                for tid in list(unmatched_track_ids):
                    rec = self._records[tid]
                    pred_bbox = predicted_bboxes[tid]
                    d_spatial = min(
                        _centroid_distance(det.bbox, pred_bbox),
                        _centroid_distance(det.bbox, rec.bbox),
                        rec.distance_to_trail(det_cx, det_cy, max_time_gap=60.0, now=now),
                    )

                    time_lost = now - rec.last_seen
                    max_dist = min(self._spatial_lock_radius, 40.0 + 20.0 * time_lost)
                    if d_spatial <= max_dist:
                        score = d_spatial
                        if score < best_score:
                            best_score = score
                            best_tid = tid

                if best_tid is not None:
                    matched_records[best_tid] = det.bbox
                    unmatched_track_ids.remove(best_tid)
                    if hasattr(det, 'track_id') and det.track_id is not None:
                        self._detector_id_map[det.track_id] = best_tid
                        self._active_detector_ids.add(det.track_id)
                else:
                    remaining_unmatched_dets.append(det)

            unmatched_dets = remaining_unmatched_dets

        # ── Pass 2.5: Spatial Lock Occlusion Recovery from Static Retired Records ───
        # Recovers static hotdogs occluded in place (e.g. by towels, boxes, condiment bottles)
        HAND_CONTINUITY_WEIGHT: float = 0.4
        HAND_CONTINUITY_MAX_PX: float = 600.0

        still_unmatched_dets = []
        for det in unmatched_dets:
            best_retired_tid = None
            best_combined_score = float('inf')
            best_retired_dist  = float('inf')
            det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
            det_cy = (det.bbox[1] + det.bbox[3]) / 2.0

            for rtid, rrec in list(self._retired_records.items()):
                if rtid in self._permanent_done_ids:
                    continue
                if (now - rrec.last_seen) > self._orphan_timeout:
                    continue

                pred_r_bbox = rrec.kalman.predict() if rrec.kalman else rrec.bbox
                d_spatial = min(
                    _centroid_distance(det.bbox, rrec.bbox),
                    _centroid_distance(det.bbox, pred_r_bbox),
                    rrec.distance_to_trail(det_cx, det_cy, max_time_gap=60.0, now=now),
                )
                time_lost = now - rrec.last_seen
                max_dist = min(self._spatial_lock_radius, 40.0 + 20.0 * time_lost)
                if d_spatial > max_dist:
                    continue

                rrec_cx = (rrec.bbox[0] + rrec.bbox[2]) / 2.0
                rrec_cy = (rrec.bbox[1] + rrec.bbox[3]) / 2.0
                det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                
                # Prevent swapping: if the track was lost inside the assembly (wrap) region,
                # it cannot be recovered by a detection outside the assembly region.
                if self._point_in_wrap_zone(rrec_cx, rrec_cy) and not self._point_in_wrap_zone(det_cx, det_cy):
                    continue

                if rrec.last_hand_working_pos is not None and current_hand_pts:
                    hand_cont_dist = min(
                        (
                            (rrec.last_hand_working_pos[0] - hpt[0]) ** 2
                            + (rrec.last_hand_working_pos[1] - hpt[1]) ** 2
                        ) ** 0.5
                        for hpt in current_hand_pts
                    )
                    hand_cont_dist = min(hand_cont_dist, HAND_CONTINUITY_MAX_PX)
                else:
                    hand_cont_dist = HAND_CONTINUITY_MAX_PX / 2.0

                combined_score = d_spatial + HAND_CONTINUITY_WEIGHT * hand_cont_dist

                if combined_score < best_combined_score:
                    best_combined_score = combined_score
                    best_retired_dist   = d_spatial
                    best_retired_tid    = rtid

            if best_retired_tid is not None:
                recovered_rec = self._retired_records.pop(best_retired_tid)
                recovered_rec.retired = False
                recovered_rec.is_coasting = False
                recovered_rec.last_seen = now
                recovered_rec.bbox = det.bbox
                if recovered_rec.kalman is None:
                    recovered_rec.kalman = KalmanBoxFilter(det.bbox)
                else:
                    recovered_rec.kalman.update(det.bbox)
                recovered_rec.add_trail_point(
                    timestamp=now,
                    maxlen=self._trail_maxlen,
                    alpha=self._trail_smooth_alpha,
                    anchor=self._trail_anchor,
                )
                self._records[best_retired_tid] = recovered_rec
                matched_records[best_retired_tid] = det.bbox
                if hasattr(det, 'track_id') and det.track_id is not None:
                    self._detector_id_map[det.track_id] = best_retired_tid
                    self._active_detector_ids.add(det.track_id)
                self._wrap_zone_merges += 1
                gap_s = now - recovered_rec.last_seen
                logger.info(
                    "[OCCLUSION_RECOVERY] Hotdog #%d recovered (spatial=%.0fpx, "
                    "combined_score=%.1f, gap=%.2fs, hand_guided=%s)",
                    best_retired_tid, best_retired_dist, best_combined_score,
                    gap_s,
                    recovered_rec.last_hand_working_pos is not None and bool(current_hand_pts),
                )
            else:
                still_unmatched_dets.append(det)

        unmatched_dets = still_unmatched_dets

        # 3. Update matched tracks & handle coasting re-matches
        for tid, bbox in matched_records.items():
            rec = self._records[tid]
            if rec.is_coasting:
                self._occlusion_rematches += 1
                rec.is_coasting = False

            rec.last_seen = now
            rec.bbox = bbox
            for h_det, hid in hand_pairs:
                if _boxes_overlap(h_det.bbox, bbox) or _hand_touching_bbox([h_det], bbox, pad=self._hand_pad):
                    rec.last_hand_id = hid
                    rec.last_hand_working_pos = _hand_working_point(h_det.bbox)
                    rec.last_hand_contact_time = now
                    rec.last_hand_contact_frame = self._frame_count
                    rec.was_hand_carried = True
            if rec.kalman is None:
                rec.kalman = KalmanBoxFilter(bbox)
            else:
                rec.kalman.update(bbox)
            rec.add_trail_point(
                timestamp=now,
                maxlen=self._trail_maxlen,
                alpha=self._trail_smooth_alpha,
                anchor=self._trail_anchor,
            )

        # Spawn new monotonic tracks ONLY for detections at genuinely NEW spatial positions.
        # Fix C: Stalled-track watchdog — before spawning, check if a recently-retired record
        # in the wrap zone occupies the same position. If so, merge instead of spawning new ID.
        _new_wrap_zone_tids: List[int] = []  # collect for callers (e.g. WrappingStateMachine)

        # Sort left-to-right to guarantee stable monotonic ID assignment
        unmatched_dets.sort(key=lambda d: d.bbox[0] + d.bbox[2])

        for det in unmatched_dets:
            det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
            det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
            det_in_wrap_zone = self._point_in_wrap_zone(det_cx, det_cy)

            # Fix C: stall-watchdog — ONLY merge if this is the immediate previous track that just blinked
            merged_tid: Optional[int] = None
            if det_in_wrap_zone:
                best_stall_tid: Optional[int] = None
                best_stall_dist = float('inf')
                for rtid, rrec in list(self._retired_records.items()):
                    if rtid in self._permanent_done_ids:
                        continue
                    # Strict guard: Only merge if retired within last 3 seconds (brief blink) and tight distance <= 100px
                    if (now - rrec.last_seen) > 3.0:
                        continue
                    rrec_cx = (rrec.bbox[0] + rrec.bbox[2]) / 2.0
                    rrec_cy = (rrec.bbox[1] + rrec.bbox[3]) / 2.0
                    if not self._point_in_wrap_zone(rrec_cx, rrec_cy):
                        continue
                    d = _centroid_distance(det.bbox, rrec.bbox)
                    if d <= 100.0 and d < best_stall_dist:
                        best_stall_dist = d
                        best_stall_tid = rtid

                if best_stall_tid is not None:
                    # Merge: recover retired record instead of spawning new ID
                    rec = self._retired_records.pop(best_stall_tid)
                    rec.retired = False
                    rec.is_coasting = False
                    rec.last_seen = now
                    rec.bbox = det.bbox
                    if rec.kalman is None:
                        rec.kalman = KalmanBoxFilter(det.bbox)
                    else:
                        rec.kalman.update(det.bbox)
                    if frame is not None:
                        rec.bgr_histogram = _compute_bgr_histogram(frame, det.bbox)
                    rec.add_trail_point(
                        timestamp=now,
                        maxlen=self._trail_maxlen,
                        alpha=self._trail_smooth_alpha,
                        anchor=self._trail_anchor,
                    )
                    self._records[best_stall_tid] = rec
                    matched_records[best_stall_tid] = det.bbox
                    if hasattr(det, 'track_id') and det.track_id is not None:
                        self._detector_id_map[det.track_id] = best_stall_tid
                    self._wrap_zone_merges += 1
                    merged_tid = best_stall_tid
                    logger.warning(
                        "[WRAP_ZONE_MERGE] Stalled-track watchdog: hotdog #%d recovered "
                        "instead of spawning new ID (dist=%.0fpx, gap=%.1fs)",
                        best_stall_tid, best_stall_dist, now - rec.last_seen,
                    )

            if merged_tid is not None:
                continue  # already handled above

            # Normal spawn path (distinct hotdogs get unique monotonic IDs)
            new_tid = None
            if expected_hotdogs is not None:
                used_ids = set()
                for r in self._records.values():
                    if str(r.hotdog_id).isdigit(): used_ids.add(int(r.hotdog_id))
                for r in self._retired_records.values():
                    if getattr(r, 'status', '') == 'done' and str(r.hotdog_id).isdigit():
                        used_ids.add(int(r.hotdog_id))
                        
                for i in range(1, expected_hotdogs + 1):
                    if i not in used_ids:
                        new_tid = i
                        break
                        
                if new_tid is None:
                    continue  # Capped at expected_hotdogs, ignore spurious detection
            else:
                new_tid = self._next_monotonic_id
                self._next_monotonic_id += 1

            if hasattr(det, 'track_id') and det.track_id is not None:
                self._detector_id_map[det.track_id] = new_tid
                self._active_detector_ids.add(det.track_id)

            rec = HotdogRecord(
                hotdog_id=str(new_tid),
                track_id=new_tid,
                first_seen=now,
                last_seen=now,
                order_id=active_ticket_id,
                bbox=det.bbox,
                kalman=KalmanBoxFilter(det.bbox),
                bgr_histogram=_compute_bgr_histogram(frame, det.bbox) if frame is not None else None,
            )
            for h_det, hid in hand_pairs:
                if _boxes_overlap(h_det.bbox, det.bbox) or _hand_touching_bbox([h_det], det.bbox, pad=self._hand_pad):
                    rec.last_hand_id = hid
                    rec.last_hand_working_pos = _hand_working_point(h_det.bbox)
                    rec.last_hand_contact_time = now
                    rec.last_hand_contact_frame = self._frame_count
                    rec.was_hand_carried = True
            rec.add_trail_point(
                timestamp=now,
                maxlen=self._trail_maxlen,
                alpha=self._trail_smooth_alpha,
                anchor=self._trail_anchor,
            )
            self._records[new_tid] = rec
            matched_records[new_tid] = det.bbox
            if det_in_wrap_zone:
                _new_wrap_zone_tids.append(new_tid)

        # 4. Check for coasting / retired tracks (Absence > orphan_timeout_s)
        # Fix D: use extended wrap-zone orphan timeout for tracks inside the zone.
        retired_tids = []
        for tid, rec in self._records.items():
            if tid not in matched_records:
                if not rec.is_coasting:
                    rec.is_coasting = True
                    self._occlusion_events += 1
                    # Record the exact ID that was lost during hand contact
                    was_in_hand = (
                        rec.was_hand_carried or
                        (rec.last_hand_contact_time is not None and abs(now - rec.last_hand_contact_time) <= 2.5)
                    )
                    if was_in_hand and rec.last_hand_id is not None:
                        pos = rec.last_hand_working_pos or ((rec.bbox[0] + rec.bbox[2]) / 2.0, (rec.bbox[1] + rec.bbox[3]) / 2.0)
                        self._last_lost_by_hand[rec.last_hand_id] = {
                            "tid": tid,
                            "lost_frame": self._frame_count,
                            "lost_time": now,
                            "lost_position": pos,
                        }
                        logger.info(
                            "[HAND_OCCLUSION_LOST] Hand #%d lost Hotdog #%d at (%.0f, %.0f) at frame %d, time=%.2fs",
                            rec.last_hand_id, tid, pos[0], pos[1], self._frame_count, now,
                        )
                rec_cx = (rec.bbox[0] + rec.bbox[2]) / 2.0
                rec_cy = (rec.bbox[1] + rec.bbox[3]) / 2.0
                effective_timeout = (
                    self._wrap_orphan_timeout
                    if self._point_in_wrap_zone(rec_cx, rec_cy)
                    else self._orphan_timeout
                )
                if (now - rec.last_seen) > effective_timeout:
                    rec.retired = True
                    retired_tids.append(tid)

        for tid in retired_tids:
            self._retired_records[tid] = self._records.pop(tid)

        # Store newly-spawned wrap-zone track IDs for the caller to register with WSM
        self._last_new_wrap_zone_tids: List[int] = _new_wrap_zone_tids

        # 5. Associate items with active hotdogs — Hand-as-Bridge attribution
        #
        # Attribution chain (for dry ingredients AND sauces):
        #
        #   Step 1 [Hand-as-Bridge PRIMARY]:
        #     item_detected
        #     → find nearest hand's working-point (bottom-center of SHRUNK hand bbox)
        #       within HAND_ITEM_RADIUS of item centroid
        #     → from that hand point, find nearest hotdog within HAND_HOTDOG_RADIUS
        #     → that hotdog exclusively receives the item
        #
        #   Step 2 [Overlap fallback — hand not visible]:
        #     item bbox overlaps padded hotdog bbox → nearest overlapping hotdog wins.
        #     This handles the common case where the item is placed directly ON the
        #     hotdog but the hand is occluded or not detected in that frame.
        #     Applies to BOTH dry ingredients AND sauces.
        #
        #   Step 3 [Centroid fallback — no overlap found]:
        #     Nearest hotdog centroid within DIRECT_FALLBACK_PX.
        #     Dry items only (sauces require overlap or hand bridge).
        #
        # Per-record _committed_items:
        #   Dry ingredients only — locks item class to that hotdog permanently.
        #   Sauces use SAUCE_MAX_COMMITS counter instead (allows multiple applications).
        active_pairs: Set[Tuple[int, str]] = set()
        item_dets = self._deduplicate_sauces(item_dets)

        for item_det in item_dets:
            if not self._is_valid_item_detection(item_det):
                continue
            if getattr(item_det, 'confidence', 1.0) < 0.55:
                continue  # Filter low-confidence noisy detections (e.g. false cheese fires)

            item_class = self._normalize_item_class_name(item_det.class_name)
            is_sauce = item_class in SAUCE_STRICT_OVERLAP_CLASSES

            if DRY_ITEMS_REQUIRE_WELL_TRIP and not is_sauce:
                # Dry ingredients arrive via force_commit_item() once a well
                # trip completes; seeing one near a hotdog is not evidence that
                # it was put ON the hotdog.
                continue

            item_cx = (item_det.bbox[0] + item_det.bbox[2]) / 2.0
            item_cy = (item_det.bbox[1] + item_det.bbox[3]) / 2.0

            # ── Step 1: Hand-as-Bridge (primary) ──────────────────────────────
            nearest_hand_pt: Optional[Tuple[float, float]] = None
            nearest_hand_dist = float('inf')
            for h in hand_dets:
                hpt = _hand_working_point(h.bbox)
                dist = ((item_cx - hpt[0]) ** 2 + (item_cy - hpt[1]) ** 2) ** 0.5
                if dist < HAND_ITEM_RADIUS and dist < nearest_hand_dist:
                    nearest_hand_dist = dist
                    nearest_hand_pt = hpt

            best_tid = None
            if nearest_hand_pt is not None:
                best_dist = float('inf')
                for tid, rec in self._records.items():
                    rx1, ry1, rx2, ry2 = rec.bbox
                    rcx = (rx1 + rx2) / 2.0
                    rcy = (ry1 + ry2) / 2.0
                    d = ((nearest_hand_pt[0] - rcx) ** 2 + (nearest_hand_pt[1] - rcy) ** 2) ** 0.5
                    if d < HAND_HOTDOG_RADIUS and d < best_dist:
                        # For dry ingredients, ensure item is in direct overlap or tight proximity <= 150px
                        item_hd_dist = _centroid_distance(item_det.bbox, rec.bbox)
                        has_overlap = self._overlap_area(item_det.bbox, rec.bbox) > 0
                        if is_sauce or has_overlap or item_hd_dist <= 150:
                            best_dist = d
                            best_tid = tid

            # ── Step 2: Overlap fallback — item bbox overlaps padded hotdog bbox
            if best_tid is None:
                best_overlap = 0.0
                for tid, rec in self._records.items():
                    area = self._overlap_area(item_det.bbox, rec.bbox)
                    if area > best_overlap:
                        best_overlap = area
                        best_tid = tid

            # ── Step 3: Centroid fallback — dry items only, tight radius ──────
            if best_tid is None and not is_sauce and self._records:
                best_dist = float('inf')
                for tid, rec in self._records.items():
                    dist = _centroid_distance(item_det.bbox, rec.bbox)
                    if dist < best_dist and dist <= 150:
                        best_dist = dist
                        best_tid = tid

            if best_tid is not None:
                rec = self._records[best_tid]
                key = (best_tid, item_class)
                active_pairs.add(key)

                if is_sauce:
                    sauce_commits = sum(1 for e in rec.items_added if e["item"] == item_class)
                    if sauce_commits >= SAUCE_MAX_COMMITS:
                        continue

                if key not in self._dwell_start:
                    self._dwell_start[key] = now

                effective_dwell_s = self._item_dwell_overrides.get(item_class, self._item_dwell_s)
                elapsed = now - self._dwell_start[key]

                # Commit guard:
                # • Dry ingredients: locked per-record in _committed_items (permanent session lock).
                # • Sauces: controlled by SAUCE_MAX_COMMITS counter (allows multiple applications
                #   and prevents _committed_items from blocking a legitimate second hotdog or
                #   second-pass application — checked above via sauce_commits guard).
                already_committed = (
                    (not is_sauce) and (item_class in rec._committed_items)
                ) or (
                    is_sauce and key in self._committed_pairs
                )

                if elapsed >= effective_dwell_s and not already_committed:
                    # Use shrunk hand bbox for the gate check too
                    hand_near = (not self._require_hand_proximity) or _hand_touching_bbox(
                        hand_dets, rec.bbox, pad=self._hand_pad
                    )
                    hand_on_bottle = _hand_touching_bbox(hand_dets, item_det.bbox, pad=20) if is_sauce else True

                    if not hand_near or not hand_on_bottle:
                        self._dwell_start[key] = now
                    else:
                        rec.record_item(item_class, now)
                        if not is_sauce:
                            # Lock dry ingredients permanently to this hotdog record
                            rec._committed_items.add(item_class)
                        self._committed_pairs.add(key)  # kept for dwell-timer resets
                        self._dwell_start[key] = now

        # Reset dwell timers for broken contact (committed_pairs is dwell-window-scoped only)
        stale_keys = [k for k in self._dwell_start if k not in active_pairs]
        for k in stale_keys:
            del self._dwell_start[k]
        stale_commit_keys = [k for k in self._committed_pairs if k not in active_pairs]
        for k in stale_commit_keys:
            self._committed_pairs.discard(k)

    def force_commit_item(
        self,
        hand_working_pt: Tuple[float, float],
        item_class: str,
        now: float,
        max_radius: int = 400,
        is_sauce: bool = False,
        target_tid: Optional[int] = None,
    ) -> Optional[int]:
        """
        Directly attribute an item or sauce to the nearest active hotdog
        without requiring the item to be visually detected.

        Called from main.py when a TemporalTracker pick/place/sauce action fires,
        using the hand's working-point (bottom-centre of shrunk hand bbox) to
        resolve which hotdog the worker is standing over.

        If ``target_tid`` is provided (a monotonic hotdog ID pre-resolved by the
        caller), the spatial search is bypassed and the commit goes directly to
        that hotdog.  This is used for sauce attribution where the hand-as-bridge
        logic in main.py already resolved the correct hotdog before creating the
        action — re-resolving via hand working point would select the wrong hotdog
        when two hotdogs are side-by-side.

        Returns the hotdog track-ID that received the commit, or None.
        """
        if not self._records:
            return None

        item_class = self._normalize_item_class_name(item_class)

        if target_tid is not None:
            # Direct commit — bypass spatial search, caller has already resolved.
            if target_tid not in self._records or self._records[target_tid].retired:
                return None
            best_tid = target_tid
            best_dist = 0.0
        else:
            # Find nearest active hotdog within max_radius of the hand's working point
            best_tid = None
            best_dist = float("inf")
            for tid, rec in self._records.items():
                if rec.retired:
                    continue
                rx1, ry1, rx2, ry2 = rec.bbox
                rcx = (rx1 + rx2) / 2.0
                rcy = (ry1 + ry2) / 2.0
                d = ((hand_working_pt[0] - rcx) ** 2 + (hand_working_pt[1] - rcy) ** 2) ** 0.5
                if d < max_radius and d < best_dist:
                    best_dist = d
                    best_tid = tid

        if best_tid is None:
            return None

        rec = self._records[best_tid]
        key = (best_tid, item_class)

        # Guard: dry ingredients are locked per-record; sauces capped at SAUCE_MAX_COMMITS
        if is_sauce:
            sauce_commits = sum(1 for e in rec.items_added if e["item"] == item_class)
            if sauce_commits >= SAUCE_MAX_COMMITS:
                return best_tid  # capped — do not add duplicate count
            # Per-(hotdog, sauce) cooldown window
            if key in self._committed_pairs:
                return best_tid
        else:
            if item_class in rec._committed_items:
                return best_tid  # already on this hotdog

        rec.record_item(item_class, now)
        if not is_sauce:
            rec._committed_items.add(item_class)
        self._committed_pairs.add(key)
        logger.debug(
            "force_commit_item: '%s' → hotdog #%d (dist=%.0fpx, sauce=%s, direct=%s)",
            item_class, best_tid, best_dist, is_sauce, target_tid is not None,
        )
        return best_tid

    def record_wrapping_event(
        self,
        event_type: str,
        track_id: int,
        timestamp: float,
        frame_idx: Optional[int] = None,
        dwell_start: Optional[float] = None,
        closing_time: Optional[float] = None,
    ) -> None:
        """Record wrapping lifecycle transition event on a hotdog record."""
        mono_tid = self._detector_id_map.get(track_id, track_id)
        rec = self._records.get(mono_tid) or self._retired_records.get(mono_tid)
        if rec is None:
            return

        if event_type == "closing":
            if dwell_start is not None:
                rec.wrapping_dwell_start = dwell_start
            elif rec.wrapping_dwell_start is None:
                rec.wrapping_dwell_start = max(0.0, timestamp - 0.4)
            rec.wrapping_closing_time = closing_time or timestamp
            rec.status = "wrapping"
        elif event_type == "done":
            rec.wrapping_done_time = timestamp
            rec.end_time = timestamp
            rec.status = "done"
            rec.retired = True
            self._permanent_done_ids.add(mono_tid)

    def sync_wrapping_states(self, wrapping_sm: Any) -> None:
        """Sync all states from WrappingStateMachine to HotdogRecords."""
        if wrapping_sm is None:
            return
        try:
            all_states = wrapping_sm.get_all_states()
            for yolo_tid, state_dict in all_states.items():
                mono_tid = self._detector_id_map.get(yolo_tid, yolo_tid)
                rec = self._records.get(mono_tid) or self._retired_records.get(mono_tid)
                if rec is None:
                    continue

                if state_dict.get("wrapping_dwell_start") is not None:
                    rec.wrapping_dwell_start = state_dict["wrapping_dwell_start"]
                if state_dict.get("closing_time") is not None:
                    rec.wrapping_closing_time = state_dict["closing_time"]
                if state_dict.get("done_time") is not None:
                    rec.wrapping_done_time = state_dict["done_time"]
                    rec.end_time = state_dict["done_time"]

                st = state_dict.get("state")
                if st == "done" or yolo_tid in getattr(wrapping_sm, "done_ids", set()):
                    rec.status = "done"
                    rec.retired = True
                    self._permanent_done_ids.add(mono_tid)
                elif st == "closing":
                    rec.status = "wrapping"
        except Exception as e:
            logger.debug("sync_wrapping_states exception: %s", e)

    def get_hotdog_log(self) -> Dict:
        now = self._last_time if self._last_time is not None else time.time()
        result = {}
        all_recs = {**self._records, **self._retired_records}
        
        import os
        multi_id = True
        
        if not multi_id and all_recs:
            # Single ID mode: merge all records into ID "1"
            merged_rec = {
                "track_id": 1,
                "hotdog_id": "1",
                "order_id": None,
                "status": "idle",
                "items_added": [],
                "item_names": [],
                "item_counts": {},
                "active": False,
                "retired": False,
                "is_coasting": False,
                "first_seen": None,
                "last_seen": None,
                "start_time_s": None,
                "start_time_str": None,
                "end_time_s": None,
                "end_time_str": None,
                "wrapping_dwell_start": None,
                "wrapping_closing_time": None,
                "wrapping_done_time": None,
                "time_since_disappeared": 0.0,
                "bbox": None,
                "trail": []
            }
            
            for tid, rec in sorted(all_recs.items(), key=lambda x: x[1].first_seen if x[1].first_seen else 0):
                time_since_disappeared = max(0.0, now - rec.last_seen)
                active = (not rec.retired) and (time_since_disappeared <= 1.0)
                
                if merged_rec["first_seen"] is None or (rec.first_seen and rec.first_seen < merged_rec["first_seen"]):
                    merged_rec["first_seen"] = rec.first_seen
                    start_ts = rec.first_seen
                    merged_rec["start_time_s"] = round(start_ts, 2) if start_ts is not None else None
                    merged_rec["start_time_str"] = format_time_str(start_ts)
                
                if merged_rec["last_seen"] is None or (rec.last_seen and rec.last_seen > merged_rec["last_seen"]):
                    merged_rec["last_seen"] = rec.last_seen
                    merged_rec["time_since_disappeared"] = round(time_since_disappeared, 3)
                
                merged_rec["active"] = merged_rec["active"] or active
                merged_rec["retired"] = rec.retired if not active else False
                merged_rec["is_coasting"] = merged_rec["is_coasting"] or rec.is_coasting
                
                if rec.status != "idle":
                    merged_rec["status"] = rec.status
                if rec.order_id:
                    merged_rec["order_id"] = rec.order_id
                
                merged_rec["items_added"].extend(rec.items_added)
                merged_rec["item_names"].extend(rec.item_names)
                for k, v in rec._item_counts.items():
                    merged_rec["item_counts"][k] = merged_rec["item_counts"].get(k, 0) + v
                
                if rec.wrapping_dwell_start: merged_rec["wrapping_dwell_start"] = rec.wrapping_dwell_start
                if rec.wrapping_closing_time: merged_rec["wrapping_closing_time"] = rec.wrapping_closing_time
                if rec.wrapping_done_time:
                    merged_rec["wrapping_done_time"] = rec.wrapping_done_time
                    end_ts = rec.wrapping_done_time
                    merged_rec["end_time_s"] = round(end_ts, 2)
                    merged_rec["end_time_str"] = format_time_str(end_ts)
                
                if active or merged_rec["bbox"] is None:
                    merged_rec["bbox"] = rec.bbox
                merged_rec["trail"].extend(rec.trail)
            
            result[1] = merged_rec
            return result

        for tid, rec in all_recs.items():
            time_since_disappeared = max(0.0, now - rec.last_seen)
            active = (not rec.retired) and (time_since_disappeared <= 1.0)
            start_ts = rec.first_seen
            end_ts = rec.wrapping_done_time or rec.wrapping_closing_time or (rec.last_seen if rec.retired else None)
            result[tid] = {
                "track_id":               tid,
                "hotdog_id":              rec.hotdog_id,
                "order_id":               rec.order_id,
                "status":                 rec.status,
                "items_added":            rec.items_added,
                "item_names":             rec.item_names,
                "item_counts":            dict(rec._item_counts),
                "active":                 active,
                "retired":                rec.retired,
                "is_coasting":            rec.is_coasting,
                "first_seen":             rec.first_seen,
                "last_seen":              rec.last_seen,
                "start_time_s":           round(start_ts, 2) if start_ts is not None else None,
                "start_time_str":         format_time_str(start_ts),
                "end_time_s":             round(end_ts, 2) if end_ts is not None else None,
                "end_time_str":           format_time_str(end_ts),
                "wrapping_dwell_start":   rec.wrapping_dwell_start,
                "wrapping_closing_time":  rec.wrapping_closing_time,
                "wrapping_done_time":     rec.wrapping_done_time,
                "time_since_disappeared": round(time_since_disappeared, 3),
                "bbox":                   rec.bbox,
                "trail":                  list(rec.trail),
            }
        return result

    def get_summary(self, wrapping_sm: Optional[Any] = None) -> Dict:
        if wrapping_sm is not None:
            self.sync_wrapping_states(wrapping_sm)

        orders = {}
        hotdogs_list = []
        item_timeline = []
        all_recs = {**self._records, **self._retired_records}

        import os
        multi_id = True

        if not multi_id and all_recs:
            # Single ID mode: merge all records into a single order1
            merged_rec = {
                "track_id": 1,
                "hotdog_id": "1",
                "order_id": None,
                "status": "idle",
                "items_added": [],
                "item_names": [],
                "item_counts": {},
                "active": False,
                "retired": False,
                "is_coasting": False,
                "first_seen": None,
                "last_seen": None,
                "start_time_s": None,
                "start_time_str": None,
                "end_time_s": None,
                "end_time_str": None,
                "wrapping_dwell_start": None,
                "wrapping_closing_time": None,
                "wrapping_done_time": None,
                "time_since_disappeared": 0.0,
                "bbox": None,
                "trail": []
            }
            
            # Use same merge logic as get_hotdog_log
            now = self._last_time if self._last_time is not None else time.time()
            for tid, rec in sorted(all_recs.items(), key=lambda x: x[1].first_seen if x[1].first_seen else 0):
                time_since_disappeared = max(0.0, now - rec.last_seen)
                active = (not rec.retired) and (time_since_disappeared <= 1.0)
                if merged_rec["first_seen"] is None or (rec.first_seen and rec.first_seen < merged_rec["first_seen"]):
                    merged_rec["first_seen"] = rec.first_seen
                    merged_rec["start_time_s"] = round(rec.first_seen, 2)
                    merged_rec["start_time_str"] = format_time_str(rec.first_seen)
                if merged_rec["last_seen"] is None or (rec.last_seen and rec.last_seen > merged_rec["last_seen"]):
                    merged_rec["last_seen"] = rec.last_seen
                    merged_rec["time_since_disappeared"] = round(time_since_disappeared, 3)
                merged_rec["active"] = merged_rec["active"] or active
                merged_rec["retired"] = rec.retired if not active else False
                merged_rec["is_coasting"] = merged_rec["is_coasting"] or rec.is_coasting
                if rec.status != "idle": merged_rec["status"] = rec.status
                if rec.order_id: merged_rec["order_id"] = rec.order_id
                
                # Merge items safely to avoid duplicates if tracking dropped mid-addition
                for item_event in rec.items_added:
                    # Avoid duplicate items with very close timestamps
                    is_dup = False
                    for existing in merged_rec["items_added"]:
                        if existing["item"] == item_event["item"] and abs(existing["timestamp"] - item_event["timestamp"]) < 1.0:
                            is_dup = True
                            break
                    if not is_dup:
                        merged_rec["items_added"].append(item_event)
                        merged_rec["item_names"].append(item_event["item"])
                        merged_rec["item_counts"][item_event["item"]] = merged_rec["item_counts"].get(item_event["item"], 0) + 1
                        
                        item_timeline.append({
                            "hotdog_id": "1",
                            "item": item_event["item"],
                            "timestamp": item_event["timestamp"],
                            "time_str": item_event.get("time_str", format_time_str(item_event["timestamp"]))
                        })
                
                if rec.wrapping_dwell_start: merged_rec["wrapping_dwell_start"] = rec.wrapping_dwell_start
                if rec.wrapping_closing_time: merged_rec["wrapping_closing_time"] = rec.wrapping_closing_time
                if rec.wrapping_done_time:
                    merged_rec["wrapping_done_time"] = rec.wrapping_done_time
                    merged_rec["end_time_s"] = round(rec.wrapping_done_time, 2)
                    merged_rec["end_time_str"] = format_time_str(rec.wrapping_done_time)
                if active or merged_rec["bbox"] is None:
                    merged_rec["bbox"] = rec.bbox

            orders["order1"] = merged_rec
            hotdogs_list.append(merged_rec)
            
            # Sort timeline
            item_timeline.sort(key=lambda x: x["timestamp"])
            
            return {
                "total_hotdogs": 1,
                "hotdogs": hotdogs_list,
                "orders": orders,
                "timeline": item_timeline
            }

        # Sort hotdogs by first_seen and track_id for deterministic order
        sorted_recs = sorted(all_recs.items(), key=lambda item: (item[1].first_seen, item[0]))

        for i, (tid, rec) in enumerate(sorted_recs, start=1):
            key = f"order{i}"

            start_ts = rec.first_seen
            end_ts = (
                rec.wrapping_done_time
                if rec.wrapping_done_time is not None
                else (rec.wrapping_closing_time if rec.wrapping_closing_time is not None else rec.last_seen)
            )
            duration_s = (
                max(0.0, round(end_ts - start_ts, 2))
                if (start_ts is not None and end_ts is not None)
                else 0.0
            )

            undergone_wrapping = (
                rec.wrapping_done_time is not None
                or rec.wrapping_closing_time is not None
                or rec.wrapping_dwell_start is not None
                or rec.status in ("done", "wrapping")
            )

            status = (
                "done"
                if (rec.status == "done" or rec.wrapping_done_time is not None)
                else (
                    "wrapping"
                    if (rec.status == "wrapping" or rec.wrapping_closing_time is not None)
                    else ("retired" if rec.retired else "in_progress")
                )
            )

            formatted_items = []
            for it in rec.items_added:
                ts = it.get("timestamp", it.get("video_timestamp_s", 0.0))
                formatted_items.append({
                    "hotdog_id": rec.hotdog_id,
                    "item": it.get("item"),
                    "count": it.get("count", 1),
                    "timestamp_s": round(ts, 2),
                    "time_str": it.get("time_str", format_time_str(ts)),
                    "video_timestamp_s": round(ts, 2),
                })

            hotdog_entry = {
                "hotdog_id": rec.hotdog_id,
                "track_id": tid,
                "order_id": rec.order_id,
                "status": status,
                "undergone_wrapping": undergone_wrapping,
                "start_time": {
                    "timestamp_s": round(start_ts, 2) if start_ts is not None else None,
                    "time_str": format_time_str(start_ts),
                },
                "end_time": {
                    "timestamp_s": round(end_ts, 2) if end_ts is not None else None,
                    "time_str": format_time_str(end_ts),
                },
                "wrapping": {
                    "undergone_wrapping": undergone_wrapping,
                    "started_at_s": round(rec.wrapping_dwell_start, 2) if rec.wrapping_dwell_start is not None else None,
                    "started_at_str": format_time_str(rec.wrapping_dwell_start),
                    "closing_at_s": round(rec.wrapping_closing_time, 2) if rec.wrapping_closing_time is not None else None,
                    "closing_at_str": format_time_str(rec.wrapping_closing_time),
                    "done_at_s": round(rec.wrapping_done_time, 2) if rec.wrapping_done_time is not None else None,
                    "done_at_str": format_time_str(rec.wrapping_done_time),
                },
                "duration_s": duration_s,
                "items_added": formatted_items,
                "item_names": rec.item_names,
                "item_counts": dict(rec._item_counts),
                "completed": undergone_wrapping or (len(rec.items_added) > 0 and status == "done"),
            }

            hotdogs_list.append(hotdog_entry)

            # Backwards-compatible orders mapping
            orders[key] = {
                "track_id": tid,
                "hotdog_id": rec.hotdog_id,
                "order_id": rec.order_id,
                "status": status,
                "start_time_s": round(start_ts, 2) if start_ts is not None else None,
                "start_time_str": format_time_str(start_ts),
                "end_time_s": round(end_ts, 2) if end_ts is not None else None,
                "end_time_str": format_time_str(end_ts),
                "duration_s": duration_s,
                "undergone_wrapping": undergone_wrapping,
                "item_names": rec.item_names,
                "item_counts": dict(rec._item_counts),
                "items_added": formatted_items,
                "completed": hotdog_entry["completed"],
            }
            item_timeline.extend(formatted_items)

        item_timeline.sort(key=lambda x: x.get("timestamp_s", 0))
        completed_count = sum(1 for h in hotdogs_list if h["status"] == "done" or h["undergone_wrapping"])

        return {
            "total_hotdogs": len(all_recs),
            "completed_hotdogs": completed_count,
            "hotdogs": hotdogs_list,
            "orders": orders,
            "item_timeline": item_timeline,
            "regression_metrics": {
                "occlusion_events": self._occlusion_events,
                "occlusion_rematches": self._occlusion_rematches,
                "neighbor_swaps": self._neighbor_swaps,
                "id_recycled_after_exit": self._id_recycled_after_exit,
                "wrap_zone_merges": self._wrap_zone_merges,
            }
        }

    def merge_wrap_zone_fragment(
        self,
        old_tid: int,
        new_tid: int,
    ) -> bool:
        """
        Explicitly merge a fragmented wrap-zone track: transfer new_tid's record
        (items, trail, bbox) into old_tid and remove new_tid.

        Returns True if the merge was applied, False if either ID is unknown or
        old_tid is in _permanent_done_ids.

        Callers (e.g. WrappingStateMachine) should also call
        ``wrapping_sm.transfer_dwell_state(old_tid, new_tid)`` after this.
        """
        if old_tid in self._permanent_done_ids:
            logger.warning(
                "[WRAP_ZONE_MERGE] Cannot merge: old_tid=%d is permanently DONE", old_tid
            )
            return False
        old_rec = self._records.get(old_tid) or self._retired_records.get(old_tid)
        new_rec = self._records.get(new_tid)
        if old_rec is None or new_rec is None:
            return False

        # Transfer items from new to old
        old_rec.items_added.extend(new_rec.items_added)
        for cls, cnt in new_rec._item_counts.items():
            old_rec._item_counts[cls] = old_rec._item_counts.get(cls, 0) + cnt
        old_rec._seen_items.update(new_rec._seen_items)
        old_rec._committed_items.update(new_rec._committed_items)

        # Inherit spatial state from new (it has fresher position)
        old_rec.bbox = new_rec.bbox
        old_rec.last_seen = new_rec.last_seen
        old_rec.retired = False
        old_rec.is_coasting = False
        if new_rec.kalman is not None:
            old_rec.kalman = new_rec.kalman
        if new_rec.bgr_histogram is not None:
            old_rec.bgr_histogram = new_rec.bgr_histogram

        # Move old_rec back to active if it was retired
        if old_tid in self._retired_records:
            self._retired_records.pop(old_tid)
        self._records[old_tid] = old_rec

        # Remove new record entirely
        self._records.pop(new_tid, None)
        self._retired_records.pop(new_tid, None)

        # Remap detector_id_map entries pointing at new_tid → old_tid
        for det_id, mon_id in list(self._detector_id_map.items()):
            if mon_id == new_tid:
                self._detector_id_map[det_id] = old_tid

        self._wrap_zone_merges += 1
        logger.info(
            "[WRAP_ZONE_MERGE] Merged fragment hotdog #%d → #%d", new_tid, old_tid
        )
        return True

    def _point_in_wrap_zone(self, cx: float, cy: float) -> bool:
        """Return True if the centroid (cx, cy) is inside the wrap zone polygon.

        The polygon stored in ``self._wrap_zone_poly`` must be in the **same
        coordinate space** as the bounding boxes passed to ``update()``.
        Typically this means pixel space (e.g. ``[[x1, y1], [x2, y2], ...]``).

        Returns False when no polygon is configured.
        """
        if self._wrap_zone_poly is None:
            return False
        pt = (float(cx), float(cy))
        result = cv2.pointPolygonTest(self._wrap_zone_poly, pt, False)
        return result >= 0

    def reset(self) -> None:
        self._records.clear()
        self._retired_records.clear()
        self._detector_id_map.clear()
        self._dwell_start.clear()
        self._committed_pairs.clear()
        self._permanent_done_ids.clear()
        self._occlusion_events = 0
        self._occlusion_rematches = 0
        self._neighbor_swaps = 0
        self._id_recycled_after_exit = 0
        self._wrap_zone_merges = 0

    def _normalize_item_class_name(self, class_name: str) -> str:
        cleaned = class_name.strip().lower()
        if cleaned in ITEM_CLASS_ALIASES:
            return ITEM_CLASS_ALIASES[cleaned]
        if cleaned in ITEM_CLASSES:
            return cleaned
        if "(" in cleaned:
            cleaned = cleaned.split("(", 1)[0].strip()
        cleaned = cleaned.replace(" ", "_")
        if cleaned.endswith("s") and cleaned not in {"swiss"} and cleaned not in ITEM_CLASSES:
            cleaned = cleaned[:-1]
        return cleaned

    def _deduplicate_sauces(self, item_dets: list) -> list:
        non_sauce = [d for d in item_dets if d.class_name not in SAUCE_SHAPE_CLASSES]
        sauce     = [d for d in item_dets if d.class_name in SAUCE_SHAPE_CLASSES]

        sauce = [d for d in sauce if getattr(d, 'confidence', 1.0) >= self._min_sauce_confidence]

        kept_sauce: list = []
        classes_seen: Dict[str, int] = {}
        for det in sorted(sauce, key=lambda d: getattr(d, 'confidence', 1.0), reverse=True):
            count = classes_seen.get(det.class_name, 0)
            if count < self._max_sauce_instances:
                kept_sauce.append(det)
                classes_seen[det.class_name] = count + 1

        return non_sauce + kept_sauce

    def _is_valid_item_detection(self, det) -> bool:
        if det.class_name not in SAUCE_SHAPE_CLASSES:
            return True

        x1, y1, x2, y2 = det.bbox
        w = max(x2 - x1, 1)
        h = max(y2 - y1, 1)
        aspect = h / w

        if h < self._min_sauce_height:
            return False
        if aspect < self._min_sauce_aspect:
            return False
        return True

    def _overlap_area(
        self,
        bbox_item: Tuple[int, int, int, int],
        bbox_hotdog: Tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = bbox_item
        bx1, by1, bx2, by2 = bbox_hotdog
        pad = self._pad
        bx1 -= pad
        by1 -= pad
        bx2 += pad
        by2 += pad

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        w = max(0, inter_x2 - inter_x1)
        h = max(0, inter_y2 - inter_y1)
        return float(w * h)


def _boxes_overlap(bbox_a, bbox_b) -> bool:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


# ── Appearance re-ID helpers ───────────────────────────────────────────────────

def _compute_bgr_histogram(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    bins: int = 32,
) -> Optional[np.ndarray]:
    """
    Compute a normalised 3-channel (B, G, R) histogram over the crop defined by
    *bbox* (x1, y1, x2, y2 in pixel space). Returns None if the crop is empty.
    """
    if frame is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (
        max(0, bbox[0]), max(0, bbox[1]),
        min(w, bbox[2]), min(h, bbox[3]),
    )
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hist = cv2.calcHist(
        [crop], [0, 1, 2], None,
        [bins, bins, bins],
        [0, 256, 0, 256, 0, 256],
    )
    cv2.normalize(hist, hist)
    return hist.flatten().astype(np.float32)


def _bhattacharyya_distance(
    hist_a: Optional[np.ndarray],
    hist_b: Optional[np.ndarray],
) -> float:
    """
    Return the Bhattacharyya distance between two normalised BGR histograms.
    Range [0, 1]; 0 = identical, 1 = completely different.
    Returns 1.0 if either histogram is None.
    """
    if hist_a is None or hist_b is None:
        return 1.0
    # cv2.compareHist expects 2-D single-channel, but we flattened to 1-D.
    # Reshape to column vector for the comparison function.
    a = hist_a.reshape(-1, 1)
    b = hist_b.reshape(-1, 1)
    return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
