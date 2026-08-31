from src.hotdog_tracker import HotdogTracker
from src.schemas import Detection


def test_sport_peppers_uses_separate_dwell_override():
    tracker = HotdogTracker(item_dwell_s=1.5, item_dwell_overrides={"sport_peppers": 3.0})

    hotdog = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)
    hand = Detection(track_id=2, bbox=(120, 120, 220, 220), class_name="hand", confidence=0.9)
    sport_peppers = Detection(track_id=3, bbox=(120, 120, 220, 220), class_name="sport_peppers", confidence=0.9)

    tracker.update([hotdog], current_time=0.0)
    tracker.update([hotdog, hand, sport_peppers], current_time=0.0)
    tracker.update([hotdog, hand, sport_peppers], current_time=3.1)

    log = tracker.get_hotdog_log()
    first = next(iter(log.values()))
    assert first["item_counts"]["sport_peppers"] == 1


def test_summary_exposes_chilli_item_counts():
    tracker = HotdogTracker(item_dwell_s=0.0)

    hotdog = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)
    hand = Detection(track_id=2, bbox=(120, 120, 220, 220), class_name="hand", confidence=0.9)
    chilli = Detection(track_id=3, bbox=(120, 120, 220, 220), class_name="chilli", confidence=0.9)

    tracker.update([hotdog], current_time=0.0)
    tracker.update([hotdog, hand, chilli], current_time=0.0)
    tracker.update([hotdog, hand, chilli], current_time=0.2)

    summary = tracker.get_summary()
    assert summary["orders"]["order1"]["item_counts"]["chilli"] == 1


def test_sustained_mustard_overlap_counts_once_per_contact_window():
    # Override sauce dwell to 0 so the test focuses purely on the commit/dedup
    # logic rather than waiting for the 2.5 s production dwell threshold.
    tracker = HotdogTracker(
        item_dwell_s=0.0,
        item_dwell_overrides={"yellow_mustard_sauce": 0.0},
    )

    hotdog = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)
    hand = Detection(track_id=2, bbox=(120, 120, 220, 220), class_name="hand", confidence=0.9)
    mustard = Detection(track_id=3, bbox=(120, 120, 150, 220), class_name="yellow_mustard_sauce", confidence=0.9)

    tracker.update([hotdog], current_time=0.0)
    tracker.update([hotdog, hand, mustard], current_time=0.0)
    tracker.update([hotdog, hand, mustard], current_time=0.1)
    tracker.update([hotdog, hand, mustard], current_time=0.2)

    summary = tracker.get_summary()
    assert summary["orders"]["order1"]["item_counts"]["yellow_mustard_sauce"] == 1


def test_monotonic_ids_and_retirement():
    tracker = HotdogTracker(orphan_timeout_s=1.0)
    
    hd1 = Detection(track_id=None, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)
    tracker.update([hd1], current_time=0.0, active_ticket_id="TICKET-001")
    
    log = tracker.get_hotdog_log()
    assert len(log) == 1
    record1 = next(iter(log.values()))
    assert record1["hotdog_id"] == "1"
    assert record1["order_id"] == "TICKET-001"
    
    # Retire hd1 after > 1.0s timeout
    tracker.update([], current_time=1.5)
    log_after_retire = tracker.get_hotdog_log()
    assert log_after_retire[1]["retired"] is True
    
    # Spawn new item at far location -> must receive monotonic ID 2 (never recycling ID 1)
    hd2 = Detection(track_id=None, bbox=(500, 500, 600, 600), class_name="hot-dog", confidence=0.9)
    tracker.update([hd2], current_time=2.0, active_ticket_id="TICKET-002")
    
    log_final = tracker.get_hotdog_log()
    assert len(log_final) == 2
    assert log_final[2]["hotdog_id"] == "2"
    assert log_final[2]["order_id"] == "TICKET-002"


def test_invariant_exact_location_reappearance_gets_new_monotonic_id():
    """
    INVARIANT TEST:
    Item 1 exits. A near-identical item appears shortly after at the EXACT SAME bounding box.
    Under the monotonic retire-on-exit rule, it MUST spawn ID 2 and CANNOT inherit retired ID 1.
    """
    tracker = HotdogTracker(orphan_timeout_s=1.0)
    
    # Item 1 appears at bbox (100, 100, 200, 200)
    item1 = Detection(track_id=None, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.95)
    tracker.update([item1], current_time=0.0)
    log = tracker.get_hotdog_log()
    assert log[1]["hotdog_id"] == "1"
    assert log[1]["retired"] is False

    # Item 1 exits (absent for 1.5 seconds > orphan_timeout_s=1.0) -> ID 1 retired & DONE
    tracker.update([], current_time=1.5, done_ids={1})
    log = tracker.get_hotdog_log()
    assert log[1]["retired"] is True

    # Item 2 appears at the EXACT SAME coordinates (100, 100, 200, 200)
    item2 = Detection(track_id=None, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.95)
    tracker.update([item2], current_time=2.0)
    
    log = tracker.get_hotdog_log()
    assert len(log) == 2
    assert log[1]["retired"] is True
    assert log[2]["hotdog_id"] == "2"
    assert log[2]["retired"] is False
    # Ensure ID 1 is NEVER active or inherited
    assert log[1]["hotdog_id"] != log[2]["hotdog_id"]


def test_bbox_jitter_and_low_iou_maintains_stable_id():
    """
    Test that bounding box aspect ratio shifts and IoU drops (e.g. 0 IoU) do NOT spawn new IDs,
    because spatial centroid proximity fallback keeps the active track stable.
    """
    tracker = HotdogTracker(orphan_timeout_s=5.0)
    
    # Frame 0: Hot dog at (100, 100, 200, 200) -> centroid (150, 150)
    hd = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)
    tracker.update([hd], current_time=0.0)
    
    # Frame 1: Bounding box shifts slightly (110, 110, 210, 210) -> centroid (160, 160)
    hd_shift = Detection(track_id=2, bbox=(110, 110, 210, 210), class_name="hot-dog", confidence=0.9)
    tracker.update([hd_shift], current_time=0.1)
    
    # Frame 2: Low IoU / aspect ratio change (120, 120, 180, 250) -> centroid (150, 185)
    hd_aspect = Detection(track_id=3, bbox=(120, 120, 180, 250), class_name="hot-dog", confidence=0.9)
    tracker.update([hd_aspect], current_time=0.2)
    
    log = tracker.get_hotdog_log()
    # Must maintain EXACTLY 1 active record (ID 1)
    assert len(log) == 1
    assert log[1]["hotdog_id"] == "1"
    assert log[1]["active"] is True


def test_bytetrack_low_conf_pass_during_occlusion():
    """
    Test that low-confidence detections (e.g. conf 0.20) during hand occlusion are matched in Pass 2
    rather than dropped or spawning new track IDs.
    """
    tracker = HotdogTracker(iou_threshold=0.30)
    
    # High confidence detection initial frame
    hd = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.90)
    tracker.update([hd], current_time=0.0)
    
    # Hand occludes hotdog -> low confidence detection (conf 0.20) in next frame
    hand = Detection(track_id=2, bbox=(110, 110, 190, 190), class_name="hand", confidence=0.85)
    hd_occluded = Detection(track_id=3, bbox=(105, 105, 195, 195), class_name="hot-dog", confidence=0.20)
    tracker.update([hd_occluded, hand], current_time=0.1)
    
    log = tracker.get_hotdog_log()
    assert len(log) == 1
    assert log[1]["hotdog_id"] == "1"
    assert log[1]["active"] is True


def test_hand_proximity_tie_breaker_disambiguates_neighbors():
    """
    Test that when two adjacent candidate hotdogs are present, the hand proximity tie-breaker
    prefers the candidate track where the hand was working during occlusion.
    """
    tracker = HotdogTracker(orphan_timeout_s=2.0)
    
    # Two adjacent hotdogs: hd1 at (100, 100, 200, 200), hd2 at (100, 250, 200, 350)
    hd1 = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)
    hd2 = Detection(track_id=2, bbox=(100, 250, 200, 350), class_name="hot-dog", confidence=0.9)
    # Hand working specifically over hd2
    hand = Detection(track_id=3, bbox=(120, 270, 180, 330), class_name="hand", confidence=0.9)
    
    tracker.update([hd1, hd2, hand], current_time=0.0)
    
    # hd2 becomes occluded by hand (not detected for 1 frame)
    tracker.update([hd1, hand], current_time=0.1)
    
    # Detection reappears near hd2 position -> hand tie-breaker matches back to hd2
    hd2_reappear = Detection(track_id=4, bbox=(105, 255, 205, 355), class_name="hot-dog", confidence=0.85)
    tracker.update([hd1, hd2_reappear], current_time=0.2)
    
    log = tracker.get_hotdog_log()
    assert len(log) == 2
    assert log[2]["hotdog_id"] == "2"


def test_hand_carried_hotdog_wins_over_stationary_assembly_candidate():
    """A placement under the carrying hand keeps the lost hotdog's ID."""
    tracker = HotdogTracker(orphan_timeout_s=5.0)

    carried = Detection(track_id=1, bbox=(100, 300, 160, 340), class_name="hot-dog", confidence=0.9)
    stationary = Detection(track_id=2, bbox=(500, 300, 560, 340), class_name="hot-dog", confidence=0.9)
    pickup_hand = Detection(track_id=10, bbox=(110, 290, 150, 330), class_name="hand", confidence=0.9)
    tracker.update([carried, stationary, pickup_hand], current_time=0.0)
    carried_tid = tracker._detector_id_map[1]

    # Both detections blink out, but only the carried hotdog is associated
    # with the hand moving to the assembly station.
    transit_hand = Detection(track_id=10, bbox=(300, 290, 340, 330), class_name="hand", confidence=0.9)
    tracker.update([transit_hand], current_time=0.1)
    assert tracker._last_lost_by_hand[10]["tid"] == carried_tid

    # The placement is exactly where the unrelated stationary track was last
    # seen.  It must recover the carried ID, not steal/relabel the station ID.
    placed = Detection(track_id=20, bbox=(500, 300, 560, 340), class_name="hot-dog", confidence=0.9)
    placed_hand = Detection(track_id=10, bbox=(510, 290, 550, 330), class_name="hand", confidence=0.9)
    tracker.update([placed, placed_hand], current_time=0.2)

    assert tracker._detector_id_map[20] == carried_tid
    assert tracker._records[carried_tid].bbox == placed.bbox


def test_one_hand_cannot_mark_adjacent_hotdogs_as_carried():
    """A broad hand box must keep one carried identity, not overwrite it."""
    tracker = HotdogTracker(orphan_timeout_s=5.0)
    left = Detection(track_id=1, bbox=(100, 300, 160, 340), class_name="hot-dog", confidence=0.9)
    right = Detection(track_id=2, bbox=(180, 300, 240, 340), class_name="hot-dog", confidence=0.9)
    hand = Detection(track_id=10, bbox=(100, 290, 160, 330), class_name="hand", confidence=0.9)

    tracker.update([left, right, hand], current_time=0.0)
    left_tid = tracker._detector_id_map[1]

    # The hand can still be within the padded area of both hotdogs, but only
    # the one nearest its working point may be recorded as carried/lost.
    tracker.update([hand], current_time=0.1)
    assert tracker._last_lost_by_hand[10]["tid"] == left_tid


def test_regression_metrics_reported_in_summary():
    """Verify that occlusion events and regression metrics are exposed in get_summary()."""
    tracker = HotdogTracker(orphan_timeout_s=1.0)
    hd1 = Detection(track_id=1, bbox=(100, 100, 200, 200), class_name="hot-dog", confidence=0.9)
    tracker.update([hd1], current_time=0.0)
    
    # Miss detection -> enters coasting
    tracker.update([], current_time=0.1)
    
    # Re-appear -> occlusion rematch
    tracker.update([hd1], current_time=0.2)
    
    summary = tracker.get_summary()
    assert "regression_metrics" in summary
    metrics = summary["regression_metrics"]
    assert metrics["occlusion_events"] >= 1
    assert metrics["occlusion_rematches"] >= 1
    assert metrics["id_recycled_after_exit"] == 0





