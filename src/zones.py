import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.schemas import Zone


class ZoneManager:
    def __init__(self, config_path: str):
        self.zones: List[Zone] = []
        self._load(config_path)

    def _load(self, path: str):
        data = json.loads(Path(path).read_text())
        for item in data:
            if item["name"] in ("pastrami", "grilled onions"):
                continue
            self.zones.append(
                Zone(
                    id=item["id"],
                    name=item["name"],
                    zone_type=item["zone_type"],
                    polygon=[tuple(p) for p in item["polygon"]],
                    color=item.get("color", "#3b82f6"),
                )
            )

    def get_zone_for_bbox(
        self, bbox, frame_width: int = 1920, frame_height: int = 1080, is_hand: bool = True
    ) -> Optional[Zone]:
        x1, y1, x2, y2 = bbox
        if is_hand:
            # Use fingertip working point (75% down hand height) so fingertips dip into correct bin
            cx = (x1 + x2) // 2
            cy = int(y1 + (y2 - y1) * 0.75)
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Normalize point coordinates
        nx, ny = cx / frame_width, cy / frame_height

        # Pass 1: exact containment — centroid must be strictly inside the polygon.
        # This prevents wide tolerance margins from mis-assigning picks to the
        # wrong adjacent bin (e.g., zone_01 / pickles-rounds swallowing a pick
        # intended for the immediately adjacent zone_02 / grated-yellow-cheese).
        for zone in self.zones:
            poly = np.array(zone.polygon, dtype=np.float32)
            if cv2.pointPolygonTest(poly, (nx, ny), False) >= 0:
                return zone

        # Pass 2: small-tolerance fallback (0.01 normalised units) for cases
        # where the centroid lands just outside a zone boundary due to rounding.
        # 0.01 is safe because adjacent bins are ~0.007 units apart; using a
        # larger value (the old 0.05) caused the wrong bin to match.
        for zone in self.zones:
            poly = np.array(zone.polygon, dtype=np.float32)
            if cv2.pointPolygonTest(poly, (nx, ny), True) >= -0.01:
                return zone

        return None


    def get_all(self) -> List[Zone]:
        return self.zones

    def get_zones_by_type(self, zone_type: str) -> List[Zone]:
        """Return all zones whose zone_type matches the given string."""
        return [z for z in self.zones if z.zone_type == zone_type]
