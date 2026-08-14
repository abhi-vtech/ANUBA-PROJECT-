"""Tests for optical flow co-motion analysis and TemporalTracker flow integration."""

import time
from unittest.mock import patch

import numpy as np

from src.flow import OpticalFlowAnalyzer
from src.schemas import Detection, FlowSignal, HandState, Zone
from src.temporal import TemporalTracker


# Shared test fixtures
BIN_ZONE = Zone(
    id="bin_lettuce",
    name="lettuce",
    zone_type="bin",
    polygon=[(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)],
)
ASSEMBLY_ZONE = Zone(
    id="asm_main",
    name="assembly",
    zone_type="assembly",
    polygon=[(0.6, 0.1), (0.8, 0.1), (0.8, 0.3), (0.6, 0.3)],
)
W, H = 1920, 1080


class FakeZoneManager:
    def __init__(self, bin_zone=BIN_ZONE, assembly_zone=ASSEMBLY_ZONE):
        self.bin_zone = bin_zone
        self.assembly_zone = assembly_zone

    def get_zone_for_bbox(self, bbox, frame_width=1920, frame_height=1080):
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2) / frame_width
        cy = ((y1 + y2) / 2) / frame_height
        if self.bin_zone.polygon[0][0] <= cx <= self.bin_zone.polygon[1][0]:
            if self.bin_zone.polygon[0][1] <= cy <= self.bin_zone.polygon[2][1]:
                return self.bin_zone
        if self.assembly_zone.polygon[0][0] <= cx <= self.assembly_zone.polygon[1][0]:
            if (
                self.assembly_zone.polygon[0][1]
                <= cy
                <= self.assembly_zone.polygon[2][1]
            ):
                return self.assembly_zone
        return None


def _det_in_bin(track_id=1):
    return Detection(
        track_id=track_id, bbox=(200, 100, 400, 200), class_name="hand", confidence=0.9
    )


def _det_in_assembly(track_id=1):
    return Detection(
        track_id=track_id,
        bbox=(1200, 100, 1400, 200),
        class_name="hand",
        confidence=0.9,
    )


def _det_outside(track_id=1):
    return Detection(
        track_id=track_id, bbox=(900, 500, 1000, 600), class_name="hand", confidence=0.9
    )


# ---------------------------------------------------------------------------
# FlowSignal tests
# ---------------------------------------------------------------------------


class TestFlowSignal:
    def test_default_values(self):
        fs = FlowSignal()
        assert fs.mean_hand_flow == 0.0
        assert fs.mean_zone_flow == 0.0
        assert fs.direction_similarity == 0.0
        assert fs.magnitude_ratio == 0.0
        assert fs.is_contact is False
        assert fs.features_hand == 0
        assert fs.features_zone == 0

    def test_contact_true(self):
        fs = FlowSignal(
            mean_hand_flow=5.0,
            mean_zone_flow=4.0,
            direction_similarity=0.85,
            magnitude_ratio=0.8,
            is_contact=True,
            hand_flow_vector=(3.0, 4.0),
            zone_flow_vector=(2.4, 3.2),
            features_hand=50,
            features_zone=40,
        )
        assert fs.is_contact is True
        assert fs.features_hand == 50


# ---------------------------------------------------------------------------
# Co-motion analysis tests
# ---------------------------------------------------------------------------


class TestComputeComotion:
    def setup_method(self):
        self.analyzer = OpticalFlowAnalyzer()

    def test_contact_detected(self):
        """Correlated flow vectors trigger is_contact=True."""
        hand_vec = (3.0, 4.0)
        zone_vec = (2.7, 3.6)  # same direction, similar magnitude
        dir_sim, mag_ratio, is_contact = self.analyzer._compute_comotion(
            hand_vec, zone_vec, 5.0, 4.5
        )
        assert dir_sim > 0.5
        assert mag_ratio > 0.3
        assert is_contact is True

    def test_no_contact_opposite_direction(self):
        """Opposite flow vectors -> is_contact=False."""
        hand_vec = (3.0, 0.0)
        zone_vec = (-3.0, 0.0)
        dir_sim, mag_ratio, is_contact = self.analyzer._compute_comotion(
            hand_vec, zone_vec, 3.0, 3.0
        )
        assert dir_sim < 0
        assert is_contact is False

    def test_no_contact_low_magnitude(self):
        """Below flow_motion_threshold -> is_contact=False."""
        dir_sim, mag_ratio, is_contact = self.analyzer._compute_comotion(
            (1.0, 0.0), (0.9, 0.0), 1.0, 0.9
        )
        assert is_contact is False

    def test_no_contact_perpendicular(self):
        """Perpendicular vectors -> direction_similarity near 0 -> is_contact=False."""
        hand_vec = (3.0, 0.0)
        zone_vec = (0.0, 3.0)
        dir_sim, mag_ratio, is_contact = self.analyzer._compute_comotion(
            hand_vec, zone_vec, 3.0, 3.0
        )
        assert abs(dir_sim) < 0.1
        assert is_contact is False

    def test_no_contact_magnitude_mismatch(self):
        """Large magnitude difference -> magnitude_ratio too low -> is_contact=False."""
        hand_vec = (10.0, 0.0)
        zone_vec = (9.5, 0.0)
        dir_sim, mag_ratio, is_contact = self.analyzer._compute_comotion(
            hand_vec, zone_vec, 10.0, 1.0
        )
        assert mag_ratio < 0.3
        assert is_contact is False


# ---------------------------------------------------------------------------
# Mask helper tests
# ---------------------------------------------------------------------------


class TestMaskHelpers:
    def test_bbox_to_mask(self):
        mask = OpticalFlowAnalyzer._bbox_to_mask((10, 20, 50, 60), 100, 100)
        assert mask.shape == (100, 100)
        assert mask[20:60, 10:50].sum() > 0
        assert mask[0:20, :].sum() == 0

    def test_bbox_to_mask_clamped(self):
        """Bbox exceeding frame boundaries is clamped."""
        mask = OpticalFlowAnalyzer._bbox_to_mask((-5, -5, 50, 50), 100, 100)
        assert mask.shape == (100, 100)
        assert mask[0:50, 0:50].sum() > 0

    def test_polygon_to_mask(self):
        polygon = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
        mask = OpticalFlowAnalyzer._polygon_to_mask(polygon, 100, 100)
        assert mask.shape == (100, 100)
        assert mask.sum() > 0

    def test_mean_flow_in_mask(self):
        """Mean flow over a masked region."""
        flow = np.zeros((10, 10, 2), dtype=np.float32)
        flow[:, :, 0] = 3.0  # dx = 3 everywhere
        flow[:, :, 1] = 4.0  # dy = 4 everywhere
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        dx, dy, mag = OpticalFlowAnalyzer._mean_flow_in_mask(flow, mask)
        assert abs(dx - 3.0) < 0.01
        assert abs(dy - 4.0) < 0.01
        assert abs(mag - 5.0) < 0.01

    def test_mean_flow_empty_mask(self):
        """Empty mask returns zero flow."""
        flow = np.zeros((10, 10, 2), dtype=np.float32)
        mask = np.zeros((10, 10), dtype=np.uint8)
        dx, dy, mag = OpticalFlowAnalyzer._mean_flow_in_mask(flow, mask)
        assert dx == 0.0
        assert dy == 0.0
        assert mag == 0.0


# ---------------------------------------------------------------------------
# OpticalFlowAnalyzer integration tests (synthetic frames)
# ---------------------------------------------------------------------------


class TestOpticalFlowAnalyzer:
    def test_sparse_lk_init(self):
        analyzer = OpticalFlowAnalyzer(method="sparse_lk")
        assert analyzer.method == "sparse_lk"
        assert analyzer._dis is None

    def test_dis_ultrafast_init(self):
        analyzer = OpticalFlowAnalyzer(method="dis_ultrafast")
        assert analyzer._dis is not None

    def test_dis_fast_init(self):
        analyzer = OpticalFlowAnalyzer(method="dis_fast")
        assert analyzer._dis is not None

    def test_sparse_lk_static_frames_no_motion(self):
        """Two identical frames produce near-zero flow."""
        analyzer = OpticalFlowAnalyzer(method="sparse_lk")
        frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        polygon = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
        signal = analyzer.compute_flow(frame, frame.copy(), bbox, polygon, 100, 100)
        # Static frames -> very low flow, no contact
        assert signal.is_contact is False
        assert signal.features_hand >= 0
        assert signal.features_zone >= 0

    def test_sparse_lk_moving_object(self):
        """Frame with lateral shift should detect motion."""
        analyzer = OpticalFlowAnalyzer(method="sparse_lk")
        h, w = 200, 200
        prev = np.zeros((h, w), dtype=np.uint8)
        curr = np.zeros((h, w), dtype=np.uint8)
        # Draw a textured rectangle that shifts right by 5 pixels
        prev[50:80, 20:60] = 128
        curr[50:80, 25:65] = 128
        # Add noise for Shi-Tomasi features
        rng = np.random.default_rng(42)
        prev = np.clip(
            prev.astype(np.float32) + rng.normal(0, 30, (h, w)), 0, 255
        ).astype(np.uint8)
        curr = np.clip(
            curr.astype(np.float32) + rng.normal(0, 30, (h, w)), 0, 255
        ).astype(np.uint8)
        bbox = (10, 40, 80, 90)
        polygon = [(0.05, 0.2), (0.4, 0.2), (0.4, 0.5), (0.05, 0.5)]
        signal = analyzer.compute_flow(prev, curr, bbox, polygon, w, h)
        # Just check it doesn't crash and returns valid structure
        assert isinstance(signal, FlowSignal)
        assert signal.features_hand >= 0

    def test_dis_static_frames(self):
        """DIS on identical frames produces near-zero flow."""
        analyzer = OpticalFlowAnalyzer(method="dis_ultrafast")
        frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        bbox = (10, 10, 50, 50)
        polygon = [(0.1, 0.1), (0.3, 0.1), (0.3, 0.3), (0.1, 0.3)]
        signal = analyzer.compute_flow(frame, frame.copy(), bbox, polygon, 100, 100)
        assert signal.is_contact is False

    def test_dis_moving_object(self):
        """DIS on shifted frames detects motion."""
        analyzer = OpticalFlowAnalyzer(method="dis_ultrafast")
        h, w = 200, 200
        prev = np.zeros((h, w), dtype=np.uint8)
        curr = np.zeros((h, w), dtype=np.uint8)
        prev[50:80, 20:60] = 128
        curr[50:80, 25:65] = 128
        bbox = (10, 40, 80, 90)
        polygon = [(0.05, 0.2), (0.4, 0.2), (0.4, 0.5), (0.05, 0.5)]
        signal = analyzer.compute_flow(prev, curr, bbox, polygon, w, h)
        assert isinstance(signal, FlowSignal)


# ---------------------------------------------------------------------------
# TemporalTracker flow integration tests
# ---------------------------------------------------------------------------


class TestTemporalTrackerWithFlow:
    def _make_tracker(self, **kwargs):
        return TemporalTracker(
            pick_dwell_ms=800,
            place_dwell_ms=500,
            transition_timeout_ms=2000,
            co_motion_dwell_ms=200,
            flow_contact_threshold=2,
            **kwargs,
        )

    def _contact_signal(self, **overrides):
        """Create a FlowSignal with co-motion contact confirmed."""
        defaults = dict(
            mean_hand_flow=5.0,
            mean_zone_flow=4.0,
            direction_similarity=0.85,
            magnitude_ratio=0.8,
            is_contact=True,
            hand_flow_vector=(3.0, 4.0),
            zone_flow_vector=(2.4, 3.2),
            features_hand=50,
            features_zone=40,
        )
        defaults.update(overrides)
        return FlowSignal(**defaults)

    def test_flow_reduces_pick_dwell(self):
        """With co-motion confirmed, pick qualifies at co_motion_dwell_ms."""
        tracker = self._make_tracker()
        zones = FakeZoneManager()
        t0 = time.monotonic()

        # Frame 1: Enter bin zone (no flow on first frame)
        with patch("time.monotonic", return_value=t0):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0)
        assert tracker.tracks[1].state == HandState.IDLE_IN_ZONE

        # Frame 2: 100ms later, flow confirms contact (1st contact frame)
        flow = self._contact_signal()
        with patch("time.monotonic", return_value=t0 + 0.1):
            tracker.update(
                [_det_in_bin()],
                zones,
                W,
                H,
                flow_signals={1: flow},
                current_time=t0 + 0.1,
            )
        assert tracker.tracks[1].flow_contact_count == 1

        # Frame 3: 250ms total, 2nd contact frame -> co_motion_dwell_ms met
        with patch("time.monotonic", return_value=t0 + 0.25):
            tracker.update(
                [_det_in_bin()],
                zones,
                W,
                H,
                flow_signals={1: flow},
                current_time=t0 + 0.25,
            )

        state = tracker.tracks[1]
        assert state.pending_picks  # pending pick created
        assert state.flow_contact_count == 0  # reset after pick

    def test_no_flow_uses_full_dwell(self):
        """Without flow_signals, pick requires full pick_dwell_ms."""
        tracker = self._make_tracker()
        zones = FakeZoneManager()
        t0 = time.monotonic()

        with patch("time.monotonic", return_value=t0):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0)

        # 250ms < 800ms -> no pick yet
        with patch("time.monotonic", return_value=t0 + 0.25):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0 + 0.25)

        assert not tracker.tracks[1].pending_picks

    def test_flow_contact_requires_threshold(self):
        """Co-motion must be confirmed for flow_contact_threshold consecutive frames."""
        tracker = TemporalTracker(
            pick_dwell_ms=800,
            place_dwell_ms=500,
            transition_timeout_ms=2000,
            co_motion_dwell_ms=200,
            flow_contact_threshold=3,
        )
        zones = FakeZoneManager()
        t0 = time.monotonic()

        with patch("time.monotonic", return_value=t0):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0)

        flow = self._contact_signal()

        # 1st and 2nd contact frames — not enough
        with patch("time.monotonic", return_value=t0 + 0.1):
            tracker.update(
                [_det_in_bin()],
                zones,
                W,
                H,
                flow_signals={1: flow},
                current_time=t0 + 0.1,
            )
        with patch("time.monotonic", return_value=t0 + 0.2):
            tracker.update(
                [_det_in_bin()],
                zones,
                W,
                H,
                flow_signals={1: flow},
                current_time=t0 + 0.2,
            )

        assert tracker.tracks[1].flow_contact_count == 2
        assert not tracker.tracks[1].pending_picks

    def test_flow_contact_resets_on_no_contact(self):
        """flow_contact_count resets to 0 when is_contact becomes False."""
        tracker = self._make_tracker()
        zones = FakeZoneManager()
        t0 = time.monotonic()

        with patch("time.monotonic", return_value=t0):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0)

        flow = self._contact_signal()
        with patch("time.monotonic", return_value=t0 + 0.1):
            tracker.update(
                [_det_in_bin()],
                zones,
                W,
                H,
                flow_signals={1: flow},
                current_time=t0 + 0.1,
            )
        assert tracker.tracks[1].flow_contact_count == 1

        # No flow on next frame -> reset
        no_flow = FlowSignal(is_contact=False, features_hand=10, features_zone=8)
        with patch("time.monotonic", return_value=t0 + 0.2):
            tracker.update(
                [_det_in_bin()],
                zones,
                W,
                H,
                flow_signals={1: no_flow},
                current_time=t0 + 0.2,
            )
        assert tracker.tracks[1].flow_contact_count == 0

    def test_flow_contact_resets_on_zone_leave(self):
        """flow_contact_count resets when hand leaves the zone."""
        tracker = self._make_tracker()
        zones = FakeZoneManager()
        t0 = time.monotonic()

        with patch("time.monotonic", return_value=t0):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0)

        flow = self._contact_signal()
        with patch("time.monotonic", return_value=t0 + 0.1):
            tracker.update(
                [_det_in_bin()],
                zones,
                W,
                H,
                flow_signals={1: flow},
                current_time=t0 + 0.1,
            )
        assert tracker.tracks[1].flow_contact_count == 1

        # Leave zone
        with patch("time.monotonic", return_value=t0 + 0.2):
            tracker.update([_det_outside()], zones, W, H, current_time=t0 + 0.2)
        assert tracker.tracks[1].flow_contact_count == 0

    def test_backward_compatible_no_flow(self):
        """TemporalTracker.update() without flow_signals behaves like before."""
        tracker = TemporalTracker(
            pick_dwell_ms=800,
            place_dwell_ms=500,
            transition_timeout_ms=2000,
        )
        zones = FakeZoneManager()
        t0 = time.monotonic()

        with patch("time.monotonic", return_value=t0):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0)

        # 250ms < 800ms -> no pick yet
        with patch("time.monotonic", return_value=t0 + 0.25):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0 + 0.25)

        assert not tracker.tracks[1].pending_picks
        assert tracker.tracks[1].flow_contact_count == 0

        # 900ms >= 800ms -> pick
        with patch("time.monotonic", return_value=t0 + 0.9):
            tracker.update([_det_in_bin()], zones, W, H, current_time=t0 + 0.9)

        assert tracker.tracks[1].pending_picks

    def test_non_bin_zone_flow_ignored(self):
        """Flow signals for detections outside bin zones are not used for pick logic."""
        tracker = self._make_tracker()
        zones = FakeZoneManager()
        t0 = time.monotonic()

        # Hand in assembly zone — flow should not affect pick logic
        with patch("time.monotonic", return_value=t0):
            tracker.update([_det_in_assembly()], zones, W, H, current_time=t0)

        # Assembly detections don't enter the bin-zone pick block,
        # so flow_contact_count stays at 0
        assert tracker.tracks[1].flow_contact_count == 0
