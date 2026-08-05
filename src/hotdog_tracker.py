"""
hotdog_tracker.py
─────────────────
Pure ByteTrack / OC-SORT motion-and-position tracker with monotonic non-recycling track IDs,
persistent spatial position lock, sticky order attribution, and hand-gated ingredient logging.

Key Principles:
• Pure motion-and-position tracking (2D Kalman Filter + Spatial IoU & Centroid Position Lock).
• Zero visual appearance crop matching.
• Monotonic global ID allocation: Track IDs strictly increment and retired IDs are NEVER reused.
• Persistent spatial position lock (300px radius & 10s coasting buffer) ensures hotdogs at a station
  retain their exact ID throughout hand, bottle, and tool occlusions.
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
DEFAULT_ITEM_DWELL_S = 0.5

DEFAULT_ITEM_DWELL_OVERRIDES = {
    "ketchup_sauce":        0.5,
    "yellow_mustard_sauce": 0.5,
    "yellow_cheese":        1.5,
    "pickle_spears":        0.15,
    "pickle_rounds":        0.15,
    "sport_peppers":        1.0,
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


@dataclass
class HotdogRecord:
    """Tracks one physical hotdog across frames with monotonic ID."""
    hotdog_id: str          # "1", "2", ... (monotonic ID string)
    track_id: int           # monotonic integer ID
    first_seen: float       # Unix timestamp
    last_seen: float        # Unix timestamp
    order_id: Optional[str] = None  # Sticky order attribution
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    items_added: List[Dict] = field(default_factory=list)
    trail: deque = field(default_factory=deque, repr=False)
    kalman: Optional[KalmanBoxFilter] = field(default=None, repr=False)
    retired: bool = False
    is_coasting: bool = False
    last_hand_working_pos: Optional[Tuple[float, float]] = None  # (cx, cy) of working hand
    _item_counts: Dict[str, int] = field(default_factory=dict, repr=False)
    _seen_items: set[str] = field(default_factory=set, repr=False)
    # Per-record committed items: once an item class is recorded on THIS hotdog
    # it is locked here for the whole session and cannot migrate to a different
    # hotdog if the detection briefly disappears and reappears nearby.
    _committed_items: set = field(default_factory=set, repr=False)

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

        # Global Monotonic ID Counter (strictly increments, never recycled)
        self._next_monotonic_id: int = 1

        # Active tid -> HotdogRecord
        self._records: Dict[int, HotdogRecord] = {}
        self._retired_records: Dict[int, HotdogRecord] = {}

        # Map detector track_id (if present) -> monotonic integer track ID
        self._detector_id_map: Dict[int, int] = {}

        self._dwell_start: Dict[Tuple[int, str], float] = {}
        self._committed_pairs: Set[Tuple[int, str]] = set()
        self._last_time: Optional[float] = None

        # Regression Instrumentation Counters
        self._occlusion_events: int = 0
        self._occlusion_rematches: int = 0
        self._neighbor_swaps: int = 0
        self._id_recycled_after_exit: int = 0

    def _normalize_item_class_name(self, name: str) -> str:
        name_clean = name.strip().lower()
        return ITEM_CLASS_ALIASES.get(name_clean, name_clean.replace(" ", "_"))

    def update(
        self,
        detections: list,
        current_time: Optional[float] = None,
        frame: Optional[np.ndarray] = None,
        active_ticket_id: Optional[str] = None,
    ) -> None:
        now = current_time if current_time is not None else time.time()
        self._last_time = now

        # Extract hotdog and hand detections directly
        hotdog_dets = [d for d in detections if d.class_name == "hot-dog"]
        hand_dets = [d for d in detections if d.class_name == "hand"]

        item_dets = []
        for det in detections:
            if det.class_name in ("hot-dog", "hand"):
                continue
            norm_name = self._normalize_item_class_name(det.class_name)
            if norm_name in ITEM_CLASSES:
                item_dets.append(det)

        # Update last known hand working position for active records touching a hand
        for rec in self._records.values():
            for h in hand_dets:
                hx1, hy1, hx2, hy2 = h.bbox
                if _boxes_overlap(h.bbox, rec.bbox) or _hand_touching_bbox([h], rec.bbox, pad=self._hand_pad):
                    rec.last_hand_working_pos = ((hx1 + hx2) / 2.0, (hy1 + hy2) / 2.0)

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

        # Pass 1: Bounding Box IoU matching against predicted Kalman positions
        if unmatched_dets and unmatched_track_ids:
            unmatched_tid_list = list(unmatched_track_ids)
            iou_matrix = np.zeros((len(unmatched_dets), len(unmatched_tid_list)), dtype=np.float32)
            for i, det in enumerate(unmatched_dets):
                for j, tid in enumerate(unmatched_tid_list):
                    iou_matrix[i, j] = _compute_iou(det.bbox, predicted_bboxes[tid])

            cost_matrix = 1.0 - iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            matched_det_indices = set()
            for r, c in zip(row_ind, col_ind):
                if iou_matrix[r, c] >= self._iou_threshold:
                    det = unmatched_dets[r]
                    tid = unmatched_tid_list[c]
                    matched_records[tid] = det.bbox
                    unmatched_track_ids.remove(tid)
                    matched_det_indices.add(r)
                    if hasattr(det, 'track_id') and det.track_id is not None:
                        self._detector_id_map[det.track_id] = tid

            unmatched_dets = [d for idx, d in enumerate(unmatched_dets) if idx not in matched_det_indices]

        # Detector track_id pass for leftover detections (if detector track_id is recognized)
        yolo_matched_dets = []
        for det in list(unmatched_dets):
            if hasattr(det, 'track_id') and det.track_id is not None and det.track_id != -1:
                det_tid = det.track_id
                if det_tid in self._detector_id_map and self._detector_id_map[det_tid] in unmatched_track_ids:
                    mon_tid = self._detector_id_map[det_tid]
                    matched_records[mon_tid] = det.bbox
                    unmatched_track_ids.remove(mon_tid)
                    yolo_matched_dets.append(det)

        for d in yolo_matched_dets:
            unmatched_dets.remove(d)

        # Pass 2: Position-Based Spatial Lock & Hand-Proximity Tie-Breaker
        # For ALL remaining unmatched detections, match to active/coasting tracks within spatial_lock_radius
        remaining_unmatched_dets = []

        for det in unmatched_dets:
            det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
            det_cy = (det.bbox[1] + det.bbox[3]) / 2.0

            best_tid = None
            best_score = float('inf')

            for tid in list(unmatched_track_ids):
                rec = self._records[tid]
                pred_bbox = predicted_bboxes[tid]
                d_spatial = _centroid_distance(det.bbox, pred_bbox)

                if d_spatial <= self._spatial_lock_radius:
                    d_hand = 0.0
                    if rec.last_hand_working_pos is not None:
                        hx, hy = rec.last_hand_working_pos
                        d_hand = ((det_cx - hx) ** 2 + (det_cy - hy) ** 2) ** 0.5

                    score = d_spatial + (0.3 * d_hand)
                    if score < best_score:
                        best_score = score
                        best_tid = tid

            if best_tid is not None:
                matched_records[best_tid] = det.bbox
                unmatched_track_ids.remove(best_tid)
                if hasattr(det, 'track_id') and det.track_id is not None:
                    self._detector_id_map[det.track_id] = best_tid
            else:
                remaining_unmatched_dets.append(det)

        unmatched_dets = remaining_unmatched_dets

        # 3. Update matched tracks & handle coasting re-matches
        for tid, bbox in matched_records.items():
            rec = self._records[tid]
            if rec.is_coasting:
                self._occlusion_rematches += 1
                rec.is_coasting = False

            rec.last_seen = now
            rec.bbox = bbox
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

        # Spawn new monotonic tracks ONLY for detections at genuinely NEW spatial positions
        for det in unmatched_dets:
            new_tid = self._next_monotonic_id
            self._next_monotonic_id += 1

            if hasattr(det, 'track_id') and det.track_id is not None:
                self._detector_id_map[det.track_id] = new_tid

            rec = HotdogRecord(
                hotdog_id=str(new_tid),
                track_id=new_tid,
                first_seen=now,
                last_seen=now,
                order_id=active_ticket_id,
                bbox=det.bbox,
                kalman=KalmanBoxFilter(det.bbox),
            )
            rec.add_trail_point(
                timestamp=now,
                maxlen=self._trail_maxlen,
                alpha=self._trail_smooth_alpha,
                anchor=self._trail_anchor,
            )
            self._records[new_tid] = rec

        # 4. Check for coasting / retired tracks (Absence > orphan_timeout_s)
        retired_tids = []
        for tid, rec in self._records.items():
            if tid not in matched_records:
                if not rec.is_coasting:
                    rec.is_coasting = True
                    self._occlusion_events += 1
                if (now - rec.last_seen) > self._orphan_timeout:
                    rec.retired = True
                    retired_tids.append(tid)

        for tid in retired_tids:
            self._retired_records[tid] = self._records.pop(tid)

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

            item_class = self._normalize_item_class_name(item_det.class_name)
            is_sauce = item_class in SAUCE_STRICT_OVERLAP_CLASSES

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
                        best_dist = d
                        best_tid = tid

            # ── Step 2: Overlap fallback — item bbox overlaps padded hotdog bbox
            # Used when no hand bridge found (hand occluded, low confidence, etc.).
            # Item placed directly on a hotdog will overlap its bbox naturally.
            # Applies to both dry items and sauces.
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
                    if dist < best_dist and dist <= DIRECT_FALLBACK_PX:
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
    ) -> Optional[int]:
        """
        Directly attribute an item or sauce to the nearest active hotdog
        without requiring the item to be visually detected.

        Called from main.py when a TemporalTracker pick/place/sauce action fires,
        using the hand's working-point (bottom-centre of shrunk hand bbox) to
        resolve which hotdog the worker is standing over.

        Returns the hotdog track-ID that received the commit, or None.
        """
        if not self._records:
            return None

        item_class = self._normalize_item_class_name(item_class)

        # Find nearest active hotdog within max_radius of the hand's working point
        best_tid: Optional[int] = None
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
            "force_commit_item: '%s' → hotdog #%d (dist=%.0fpx, sauce=%s)",
            item_class, best_tid, best_dist, is_sauce,
        )
        return best_tid

    def get_hotdog_log(self) -> Dict:
        now = self._last_time if self._last_time is not None else time.time()
        result = {}
        all_recs = {**self._records, **self._retired_records}
        for tid, rec in all_recs.items():
            time_since_disappeared = max(0.0, now - rec.last_seen)
            active = (not rec.retired) and (time_since_disappeared <= 1.0)
            result[tid] = {
                "track_id":               tid,
                "hotdog_id":              rec.hotdog_id,
                "order_id":               rec.order_id,
                "items_added":            rec.items_added,
                "item_names":             rec.item_names,
                "item_counts":            dict(rec._item_counts),
                "active":                 active,
                "retired":                rec.retired,
                "is_coasting":            rec.is_coasting,
                "first_seen":             rec.first_seen,
                "last_seen":              rec.last_seen,
                "time_since_disappeared": round(time_since_disappeared, 3),
                "bbox":                   rec.bbox,
                "trail":                  list(rec.trail),
            }
        return result

    def get_summary(self) -> Dict:
        orders = {}
        item_timeline = []
        all_recs = {**self._records, **self._retired_records}
        for i, (tid, rec) in enumerate(all_recs.items(), start=1):
            key = f"order{i}"
            orders[key] = {
                "track_id":   tid,
                "hotdog_id":  rec.hotdog_id,
                "order_id":   rec.order_id,
                "item_names": rec.item_names,
                "item_counts": dict(rec._item_counts),
                "items_added": rec.items_added,
                "completed":  len(rec.items_added) > 0,
            }
            item_timeline.extend(rec.items_added)

        item_timeline.sort(key=lambda x: x.get("timestamp", 0))

        return {
            "total_hotdogs": len(all_recs),
            "item_timeline": item_timeline,
            "orders": orders,
            "regression_metrics": {
                "occlusion_events":       self._occlusion_events,
                "occlusion_rematches":    self._occlusion_rematches,
                "neighbor_swaps":         self._neighbor_swaps,
                "id_recycled_after_exit": self._id_recycled_after_exit,
            }
        }

    def reset(self) -> None:
        self._records.clear()
        self._retired_records.clear()
        self._detector_id_map.clear()
        self._dwell_start.clear()
        self._committed_pairs.clear()
        self._occlusion_events = 0
        self._occlusion_rematches = 0
        self._neighbor_swaps = 0
        self._id_recycled_after_exit = 0

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
