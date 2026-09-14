"""Optical flow co-motion analysis for robust pick detection.

Compares flow vectors in hand bounding boxes against flow in ingredient zone
polygons. When both regions move together (correlated direction and magnitude),
that is evidence of hand-object contact (a real pick). When only the hand moves,
that is a hover or transient motion.


Supports three methods:
- sparse_lk: Pyramidal Lucas-Kanade on Shi-Tomasi corners (CPU, ~1-2ms/frame)
- dis_ultrafast: DIS dense flow with ultrafast preset (CPU, ~5-10ms on Jetson)
- dis_fast: DIS dense flow with fast preset (CPU, ~10-20ms on Jetson)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.domain.schemas import FlowSignal

logger = logging.getLogger(__name__)

# Minimum tracked features to consider flow reliable
_MIN_FEATURES = 3


class OpticalFlowAnalyzer:
    def __init__(
        self,
        method: str = "sparse_lk",
        max_corners: int = 100,
        quality_level: float = 0.01,
        min_distance: int = 7,
        direction_threshold: float = 0.5,
        magnitude_ratio_threshold: float = 0.3,
        flow_motion_threshold: float = 2.0,
        lk_window_size: int = 21,
        lk_max_level: int = 3,
    ):
        self.method = method
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.direction_threshold = direction_threshold
        self.magnitude_ratio_threshold = magnitude_ratio_threshold
        self.flow_motion_threshold = flow_motion_threshold
        self.lk_window_size = (lk_window_size, lk_window_size)
        self.lk_max_level = lk_max_level
        self.prev_gray: Optional[np.ndarray] = None

        if method == "dis_ultrafast":
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_ULTRAFAST)
        elif method == "dis_fast":
            self._dis = cv2.DISOpticalFlow_create(cv2.DISOpticalFlow_PRESET_FAST)
        else:
            self._dis = None

    def compute_flow(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        hand_bbox: Tuple[int, int, int, int],
        zone_polygon_norm: List[Tuple[float, float]],
        frame_width: int,
        frame_height: int,
    ) -> FlowSignal:
        """Compute co-motion flow between hand bbox and zone polygon."""
        if self.method == "sparse_lk":
            return self._compute_sparse_lk(
                prev_gray,
                curr_gray,
                hand_bbox,
                zone_polygon_norm,
                frame_width,
                frame_height,
            )
        return self._compute_dis(
            prev_gray,
            curr_gray,
            hand_bbox,
            zone_polygon_norm,
            frame_width,
            frame_height,
        )

    # ------------------------------------------------------------------
    # Sparse Lucas-Kanade
    # ------------------------------------------------------------------

    def _compute_sparse_lk(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        hand_bbox: Tuple[int, int, int, int],
        zone_polygon_norm: List[Tuple[float, float]],
        frame_width: int,
        frame_height: int,
    ) -> FlowSignal:
        h, w = prev_gray.shape[:2]

        hand_mask = self._bbox_to_mask(hand_bbox, h, w)
        zone_mask = self._polygon_to_mask(zone_polygon_norm, h, w)

        # Detect Shi-Tomasi corners in each ROI
        hand_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=hand_mask,
        )
        zone_pts = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=zone_mask,
        )

        n_hand = len(hand_pts) if hand_pts is not None else 0
        n_zone = len(zone_pts) if zone_pts is not None else 0

        if n_hand < _MIN_FEATURES or n_zone < _MIN_FEATURES:
            return FlowSignal(
                features_hand=n_hand,
                features_zone=n_zone,
            )

        # Single LK call with concatenated points (amortises pyramid cost)
        all_pts = np.vstack([hand_pts, zone_pts]).astype(np.float32)
        tracked, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            curr_gray,
            all_pts,
            None,
            winSize=self.lk_window_size,
            maxLevel=self.lk_max_level,
        )

        # Split results back into hand / zone portions
        hand_status = status[:n_hand].flatten()
        zone_status = status[n_hand:].flatten()

        hand_tracked = tracked[:n_hand][hand_status == 1].reshape(-1, 2)
        hand_orig = hand_pts[hand_status == 1].reshape(-1, 2)
        zone_tracked = tracked[n_hand:][zone_status == 1].reshape(-1, 2)
        zone_orig = zone_pts[zone_status == 1].reshape(-1, 2)

        if len(hand_tracked) < _MIN_FEATURES or len(zone_tracked) < _MIN_FEATURES:
            return FlowSignal(
                features_hand=len(hand_tracked),
                features_zone=len(zone_tracked),
            )

        # Mean displacement vectors
        hand_disp = hand_tracked - hand_orig
        zone_disp = zone_tracked - zone_orig

        hand_vec = (
            float(np.mean(hand_disp[:, 0])),
            float(np.mean(hand_disp[:, 1])),
        )
        zone_vec = (
            float(np.mean(zone_disp[:, 0])),
            float(np.mean(zone_disp[:, 1])),
        )
        hand_mag = float(np.mean(np.linalg.norm(hand_disp, axis=1)))
        zone_mag = float(np.mean(np.linalg.norm(zone_disp, axis=1)))

        dir_sim, mag_ratio, is_contact = self._compute_comotion(
            hand_vec,
            zone_vec,
            hand_mag,
            zone_mag,
        )

        return FlowSignal(
            mean_hand_flow=hand_mag,
            mean_zone_flow=zone_mag,
            direction_similarity=dir_sim,
            magnitude_ratio=mag_ratio,
            is_contact=is_contact,
            hand_flow_vector=hand_vec,
            zone_flow_vector=zone_vec,
            features_hand=len(hand_tracked),
            features_zone=len(zone_tracked),
        )

    # ------------------------------------------------------------------
    # DIS dense flow
    # ------------------------------------------------------------------

    def _compute_dis(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        hand_bbox: Tuple[int, int, int, int],
        zone_polygon_norm: List[Tuple[float, float]],
        frame_width: int,
        frame_height: int,
    ) -> FlowSignal:
        if self._dis is None:
            logger.error("DIS optical flow not initialised; method=%s", self.method)
            return FlowSignal()

        flow = self._dis.calc(prev_gray, curr_gray, None)
        h, w = prev_gray.shape[:2]

        hand_mask = self._bbox_to_mask(hand_bbox, h, w)
        zone_mask = self._polygon_to_mask(zone_polygon_norm, h, w)

        hx, hy, hm = self._mean_flow_in_mask(flow, hand_mask)
        zx, zy, zm = self._mean_flow_in_mask(flow, zone_mask)

        hand_vec = (float(hx), float(hy))
        zone_vec = (float(zx), float(zy))

        dir_sim, mag_ratio, is_contact = self._compute_comotion(
            hand_vec,
            zone_vec,
            float(hm),
            float(zm),
        )

        return FlowSignal(
            mean_hand_flow=float(hm),
            mean_zone_flow=float(zm),
            direction_similarity=dir_sim,
            magnitude_ratio=mag_ratio,
            is_contact=is_contact,
            hand_flow_vector=hand_vec,
            zone_flow_vector=zone_vec,
            features_hand=int(np.count_nonzero(hand_mask)),
            features_zone=int(np.count_nonzero(zone_mask)),
        )

    # ------------------------------------------------------------------
    # Co-motion analysis
    # ------------------------------------------------------------------

    def _compute_comotion(
        self,
        hand_vec: Tuple[float, float],
        zone_vec: Tuple[float, float],
        hand_mag: float,
        zone_mag: float,
    ) -> Tuple[float, float, bool]:
        """Yagi et al. co-motion thresholds for hand-object contact."""
        if (
            hand_mag < self.flow_motion_threshold
            or zone_mag < self.flow_motion_threshold
        ):
            return (0.0, 0.0, False)

        dot = hand_vec[0] * zone_vec[0] + hand_vec[1] * zone_vec[1]
        norm_product = hand_mag * zone_mag
        if norm_product < 1e-8:
            return (0.0, 0.0, False)

        direction_similarity = dot / norm_product

        max_mag = max(hand_mag, zone_mag)
        magnitude_ratio = min(hand_mag, zone_mag) / max_mag

        is_contact = (
            direction_similarity > self.direction_threshold
            and magnitude_ratio > self.magnitude_ratio_threshold
        )
        return (direction_similarity, magnitude_ratio, is_contact)

    # ------------------------------------------------------------------
    # Mask helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_to_mask(
        bbox: Tuple[int, int, int, int],
        h: int,
        w: int,
    ) -> np.ndarray:
        """Create a binary mask from a bounding box (x1, y1, x2, y2)."""
        x1, y1, x2, y2 = bbox
        mask = np.zeros((h, w), dtype=np.uint8)
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        mask[y1c:y2c, x1c:x2c] = 255
        return mask

    @staticmethod
    def _polygon_to_mask(
        polygon_norm: List[Tuple[float, float]],
        h: int,
        w: int,
    ) -> np.ndarray:
        """Create a binary mask from a normalised polygon [(x, y), ...]."""
        pts = np.array(
            [[int(px * w), int(py * h)] for px, py in polygon_norm],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    @staticmethod
    def _mean_flow_in_mask(
        flow: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Mean (dx, dy, magnitude) of dense flow where mask is non-zero."""
        if mask.sum() == 0:
            return (0.0, 0.0, 0.0)
        fx = flow[:, :, 0][mask > 0]
        fy = flow[:, :, 1][mask > 0]
        if len(fx) == 0:
            return (0.0, 0.0, 0.0)
        mean_dx = float(np.mean(fx))
        mean_dy = float(np.mean(fy))
        mean_mag = float(np.mean(np.sqrt(fx**2 + fy**2)))
        return (mean_dx, mean_dy, mean_mag)
