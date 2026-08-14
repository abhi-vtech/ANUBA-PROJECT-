from src.zones import ZoneManager
from src.schemas import Detection


def test_zone_intersection():
    zones = ZoneManager("config/zones.json")
    # zone_01 (pickles (rounds)) polygon center ~(0.523, 0.373) → pixel (1004, 403) at 1920x1080
    det = Detection(
        track_id=1, bbox=(954, 353, 1054, 453), class_name="hand", confidence=0.9
    )
    zone = zones.get_zone_for_bbox(det.bbox)
    assert zone is not None
    assert zone.name == "pickles (rounds)"
