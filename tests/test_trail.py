from src.hotdog_tracker import HotdogRecord, HotdogTracker
from src.schemas import Detection


def test_trail_point_smoothing_and_anchor():
    rec = HotdogRecord(
        hotdog_id="hotdog_1",
        track_id=1,
        first_seen=0.0,
        last_seen=0.0,
        bbox=(100, 100, 200, 200),
    )

    # First point at bottom_center: x = (100+200)/2 = 150, y = 200
    rec.add_trail_point(timestamp=1.0, maxlen=10, alpha=0.5, anchor="bottom_center")
    assert len(rec.trail) == 1
    assert rec.trail[0]["x"] == 150
    assert rec.trail[0]["y"] == 200

    # Second point moved to bbox (110, 110, 210, 220): raw x = 160, raw y = 220
    rec.bbox = (110, 110, 210, 220)
    rec.add_trail_point(timestamp=2.0, maxlen=10, alpha=0.5, anchor="bottom_center")
    assert len(rec.trail) == 2
    # Smoothed: x = int(0.5 * 160 + 0.5 * 150) = 155
    # Smoothed: y = int(0.5 * 220 + 0.5 * 200) = 210
    assert rec.trail[1]["x"] == 155
    assert rec.trail[1]["y"] == 210


def test_trail_maxlen_enforcement():
    tracker = HotdogTracker(trail_maxlen=3, trail_smooth_alpha=1.0)
    det = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)

    for t in range(5):
        det.bbox = (100 + t * 10, 100, 200 + t * 10, 200)
        tracker.update([det], current_time=float(t))

    log = tracker.get_hotdog_log()
    first = next(iter(log.values()))
    trail = first["trail"]

    assert len(trail) == 3
    # Should contain the last 3 timestamps (2.0, 3.0, 4.0)
    timestamps = [p["timestamp"] for p in trail]
    assert timestamps == [2.0, 3.0, 4.0]
