import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from src.domain.schemas import Zone

#: Zones matched by how much of the HAND overlaps them, not by whether a single
#: point lands inside.  A scoop of onions is a brief dip with the fingertips at
#: the edge of the bin, so the fingertip point often never lands inside the
#: polygon at all and the pick was missed entirely.  Any zone named here is
#: matched when the overlap reaches `TOUCH_OVERLAP_FRACTION`.
_TOUCH_ZONES = ("onions",)

#: How much of the hand box must fall inside the zone to count as touching it.
#: Deliberately low: the ask is that even a light touch counts, so anything
#: from a graze upwards registers rather than requiring a squarely-placed hand.
TOUCH_OVERLAP_FRACTION: float = 0.05

#: `relish` sits directly ON TOP of `onions` -- same x, stacked, touching at
#: about y=0.236 -- so a hand at that boundary could be reaching into either.
#: The rule (from the person who watches this station): it is relish only when
#: the hand is in the UPPER HALF of the onions bin AND actually overlapping
#: relish; anywhere else it is onions.
#:
#: The previous version of this compared the point against a hardcoded
#: y < 0.373.  0.373 lies BELOW the whole onions bin (y 0.239-0.321), so the
#: test was true for every point inside onions and every onion pick was filed
#: as relish -- `onions` could never be returned at all.  The boundary is now
#: read from the polygon instead of hardcoded, so it cannot drift from the
#: calibration again.
_RELISH_OVER_ONIONS = ("onions", "relish")

#: How close the FINGERTIP point must be to the relish bin before the upper
#: half of the onions bin is read as relish, in normalised units (~0.012 is
#: about 13 px of a 1080-tall frame).
#:
#: Measured against the fingertip, NOT the hand box: the box runs from the
#: fingers up past the wrist, so a hand reaching into the top of the onions bin
#: always has its upper edge over relish, and box overlap called almost
#: everything relish.  The fingers are what is in the bin.
RELISH_NEAR_DIST: float = 0.012


def _overlap_fraction(bbox_norm, polygon) -> float:
    """Fraction of the hand box covered by the zone's bounding rectangle.

    The bins are near-rectangular, so the polygon's bounding rect is a good
    enough stand-in and costs a handful of arithmetic per zone -- this runs for
    every hand on every frame, and rasterising the polygon would not.
    """
    x1, y1, x2, y2 = bbox_norm
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    poly = np.array(polygon, dtype=np.float32)
    px1, py1 = float(poly[:, 0].min()), float(poly[:, 1].min())
    px2, py2 = float(poly[:, 0].max()), float(poly[:, 1].max())
    ix = max(0.0, min(x2, px2) - max(x1, px1))
    iy = max(0.0, min(y2, py2) - max(y1, py1))
    return (ix * iy) / area


def _y_mid(polygon) -> float:
    ys = [p[1] for p in polygon]
    return (min(ys) + max(ys)) / 2.0


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

    def _resolve_onion_relish(self, zone: Zone, nx: float, ny: float) -> Zone:
        """Onions or relish, for a hand at the boundary between them.

        Relish sits directly on top of onions, so a point near the seam could
        belong to either.  It is relish only when BOTH hold: the point is in
        the upper half of the onions bin, AND the hand is genuinely over the
        relish bin (not merely brushing its edge).  Everything else in that
        area is onions, which is the common case at this station.
        """
        if zone.name != "onions":
            return zone
        if ny >= _y_mid(zone.polygon):
            return zone                      # lower half: onions, no question
        relish = next((z for z in self.zones if z.name == "relish"), None)
        if relish is None:
            return zone
        poly = np.array(relish.polygon, dtype=np.float32)
        # Signed distance: >= 0 inside relish, negative outside by that much.
        dist = cv2.pointPolygonTest(poly, (nx, ny), True)
        if dist >= -RELISH_NEAR_DIST:
            return relish
        return zone

    def get_zone_for_bbox(
        self,
        bbox,
        frame_width: int = 1920,
        frame_height: int = 1080,
        is_hand: bool = True,
        seg_polygon=None,
    ) -> Optional[Zone]:
        x1, y1, x2, y2 = bbox

        # Priority 1: segmentation polygon centroid (most accurate)
        if seg_polygon is not None and len(seg_polygon) >= 3:
            arr = np.array(seg_polygon, dtype=np.float32)
            cx = float(arr[:, 0].mean())
            cy = float(arr[:, 1].mean())
        elif is_hand:
            # Use fingertip working point (75% down hand height) so fingertips dip into correct bin
            cx = (x1 + x2) // 2
            cy = int(y1 + (y2 - y1) * 0.75)
        else:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Normalize point coordinates
        nx, ny = cx / frame_width, cy / frame_height
        hand_norm = ((x1 / frame_width, y1 / frame_height,
                      x2 / frame_width, y2 / frame_height) if is_hand else None)

        # Pass 1: exact containment — centroid must be strictly inside the polygon.
        # This prevents wide tolerance margins from mis-assigning picks to the
        # wrong adjacent bin (e.g., zone_01 / pickles-rounds swallowing a pick
        # intended for the immediately adjacent zone_02 / grated-yellow-cheese).
        for zone in self.zones:
            poly = np.array(zone.polygon, dtype=np.float32)
            dist = cv2.pointPolygonTest(poly, (nx, ny), True)
            if dist >= 0:
                return self._resolve_onion_relish(zone, nx, ny)

        # Pass 2: small-tolerance fallback (0.01 normalised units) for cases
        # where the centroid lands just outside a zone boundary due to rounding.
        # 0.01 is safe because adjacent bins are ~0.007 units apart; using a
        # larger value (the old 0.05) caused the wrong bin to match.
        for zone in self.zones:
            poly = np.array(zone.polygon, dtype=np.float32)
            dist = cv2.pointPolygonTest(poly, (nx, ny), True)
            if dist >= -0.01:
                return self._resolve_onion_relish(zone, nx, ny)

        # Pass 3: touch, for the zones in _TOUCH_ZONES only.  Both passes above
        # ask whether ONE point is inside the polygon; a quick dip for onions
        # often leaves that point just outside, so the pick never registered.
        # Here it is enough that the hand OVERLAPS the bin at all.
        #
        # Last, so it can never take a pick away from a zone the hand is
        # squarely inside -- an adjacent bin still wins on the point test.
        if is_hand:
            best, best_frac = None, 0.0
            for zone in self.zones:
                if zone.name not in _TOUCH_ZONES:
                    continue
                frac = _overlap_fraction(hand_norm, zone.polygon)
                if frac >= TOUCH_OVERLAP_FRACTION and frac > best_frac:
                    best, best_frac = zone, frac
            if best is not None:
                return self._resolve_onion_relish(best, nx, ny)

        return None


    def get_all(self) -> List[Zone]:
        return self.zones

    def get_zones_by_type(self, zone_type: str) -> List[Zone]:
        """Return all zones whose zone_type matches the given string."""
        return [z for z in self.zones if z.zone_type == zone_type]
