import json
import tempfile
from pathlib import Path
import pytest

from src.exit_detector import (
    ExitDetector,
    ExitEvent,
    ExitLineConfig,
    _bbox_intersects_line,
    _line_segment_intersection,
)


def test_line_intersection_geometry():
    seg1 = ((0.0, 0.0), (10.0, 10.0))
    seg2 = ((0.0, 10.0), (10.0, 0.0))
    assert _line_segment_intersection(seg1, seg2) is True

    seg3 = ((0.0, 0.0), (2.0, 2.0))
    seg4 = ((5.0, 5.0), (10.0, 10.0))
    assert _line_segment_intersection(seg3, seg4) is False


def test_bbox_intersects_line():
    bbox = (100, 100, 200, 200)
    line_p1 = (50, 150)
    line_p2 = (250, 150)
    assert _bbox_intersects_line(bbox, line_p1, line_p2) is True

    non_crossing_bbox = (300, 300, 400, 400)
    assert _bbox_intersects_line(non_crossing_bbox, line_p1, line_p2) is False


def test_exit_detector_check_crossing():
    detector = ExitDetector()
    detector.config.p1 = (0.5, 0.0)
    detector.config.p2 = (0.5, 1.0)  # Vertical line at x=50% width

    frame_w, frame_h = 1000, 1000
    hand_bbox_crossing = (450, 400, 550, 500)  # Spans x=450 to 550 across x=500
    wrapped_hotdogs = [101, 102]

    # First crossing trigger
    event = detector.check_crossing(
        hand_bboxes=[hand_bbox_crossing],
        wrapped_hotdog_ids=wrapped_hotdogs,
        frame_width=frame_w,
        frame_height=frame_h,
    )
    assert event is not None
    assert event.exited_hotdog_ids == [101, 102]
    assert len(detector.exited_history) == 1

    # Cooldown debounce test (immediate second frame crossing ignored)
    event_debounced = detector.check_crossing(
        hand_bboxes=[hand_bbox_crossing],
        wrapped_hotdog_ids=wrapped_hotdogs,
        frame_width=frame_w,
        frame_height=frame_h,
    )
    assert event_debounced is None


def test_exit_detector_config_save_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        config_path = str(Path(tmp_dir) / "exit_line.json")
        detector1 = ExitDetector()
        detector1.config.p1 = (0.2, 0.3)
        detector1.config.p2 = (0.9, 0.7)
        detector1.save_config(config_path)

        assert Path(config_path).exists()

        detector2 = ExitDetector(config_path=config_path)
        assert detector2.config.p1 == (0.2, 0.3)
        assert detector2.config.p2 == (0.9, 0.7)
