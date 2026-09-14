"""Runtime-neutral analysis: `Frame`s in, `Event`s and `Verdict`s out.

Imports nothing but stdlib and the core contract, so the identical object runs
behind Ultralytics, behind a DeepStream probe, and behind a JSONL replay in
CI.  If this module ever needs cv2, torch or pyservicemaker, the layering has
been broken.

What it does:

  * resolves which well (zone) a hand is in, by polygon containment
  * turns a sustained hand-in-well dwell into ONE place event, debounced
  * counts distinct hotdog identities from the tracker
  * feeds both into `OrderValidator`

What it deliberately does NOT do: decode video, run a model, draw anything.

The debounce is expressed in media-time seconds, never in frames, so a run at
12 fps and a run at 45 fps on the same footage produce the same events.  A
frame-count debounce would silently change behaviour when the runtime got
faster, which is exactly the class of bug that makes two pipelines disagree.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.core.contract import Event, Frame, TrackedObject
from src.core.order_rules import ExtrasPolicy, OrderValidator, Verdict
from src.core.ticket_spec import HOTDOG_LABEL, TicketSpec

HAND_LABEL = "hand"

#: Wells dispensed as several small pinches per serving need a long gap so one
#: serving is not counted several times.  Mirrors src/ingredient_config.py.
GRANULAR_ITEMS = frozenset({"onions", "diced_onions", "relish", "grated_yellow_cheese"})


@dataclass
class Zone:
    """A well or region, as a polygon in normalised (0-1) coordinates."""

    id: str
    name: str
    zone_type: str
    polygon: List[Tuple[float, float]]

    def contains(self, point: Tuple[float, float], width: int, height: int) -> bool:
        x, y = point
        inside = False
        n = len(self.polygon)
        if n < 3:
            return False
        px1, py1 = self.polygon[-1][0] * width, self.polygon[-1][1] * height
        for i in range(n):
            px2, py2 = self.polygon[i][0] * width, self.polygon[i][1] * height
            if ((py1 > y) != (py2 > y)) and (
                x < (px2 - px1) * (y - py1) / (py2 - py1 + 1e-12) + px1
            ):
                inside = not inside
            px1, py1 = px2, py2
        return inside


@dataclass
class EngineConfig:
    """Every threshold, in media-time seconds. No frame counts."""

    dwell_s: float = 0.5
    discrete_debounce_s: float = 1.8
    granular_gap_s: float = 6.0
    onion_gap_s: float = 12.0
    hotdog_label: str = HOTDOG_LABEL
    hand_label: str = HAND_LABEL
    #: Zone types treated as ingredient wells.
    well_types: frozenset = frozenset({"bin", "sauce_vessel", "cheese_region"})

    def gap_for(self, item: str) -> float:
        if "onion" in item:
            return self.onion_gap_s
        if item in GRANULAR_ITEMS:
            return self.granular_gap_s
        return self.discrete_debounce_s


def load_zones(path: str) -> List[Zone]:
    import json

    with open(path) as fh:
        raw = json.load(fh)
    zones: List[Zone] = []
    for z in raw:
        poly = [(float(p[0]), float(p[1])) for p in z.get("polygon") or []]
        if len(poly) < 3:
            continue
        zones.append(Zone(id=z.get("id", ""), name=z.get("name", ""),
                          zone_type=z.get("zone_type", ""), polygon=poly))
    return zones


class AnalysisEngine:
    """Stateful across frames; one instance per order/run."""

    def __init__(
        self,
        zones: Sequence[Zone],
        spec: Optional[TicketSpec] = None,
        config: Optional[EngineConfig] = None,
        policy: Optional[ExtrasPolicy] = None,
        normalize=None,
    ):
        from src.core.naming import normalize_item_name

        self.config = config or EngineConfig()
        self._normalize = normalize or normalize_item_name
        self.wells = [z for z in zones if z.zone_type in self.config.well_types]
        # Well name -> canonical item key, resolved once.
        self.well_item = {z.id: self._normalize(z.name) for z in self.wells}
        self.spec = spec
        self.validator = OrderValidator(spec, policy) if spec else None
        self.events: List[Event] = []
        self.frames_seen = 0

        # hand track id -> (zone id, time it entered)
        self._hand_in: Dict[int, Tuple[str, float]] = {}
        # item -> media time of the last counted application
        self._last_counted: Dict[str, float] = {}

    # -- per frame -------------------------------------------------------

    def process(self, frame: Frame) -> List[Event]:
        """Consume one frame, return the events it produced."""
        produced: List[Event] = []
        self.frames_seen += 1

        for hotdog in frame.by_label(self.config.hotdog_label):
            if hotdog.is_tracked and self.validator is not None:
                before = len(self.validator.hotdog_ids)
                self.validator.observe_hotdog(hotdog.track_id)
                if len(self.validator.hotdog_ids) > before:
                    produced.append(Event(frame.t, "hotdog_seen",
                                          {"track_id": hotdog.track_id}))

        for hand in frame.by_label(self.config.hand_label):
            produced.extend(self._handle_hand(hand, frame))

        self.events.extend(produced)
        return produced

    def _handle_hand(self, hand: TrackedObject, frame: Frame) -> List[Event]:
        out: List[Event] = []
        tid = hand.track_id
        here = self._well_at(hand.center, frame.width, frame.height)

        if here is None:
            self._hand_in.pop(tid, None)
            return out

        entered = self._hand_in.get(tid)
        if entered is None or entered[0] != here.id:
            self._hand_in[tid] = (here.id, frame.t)
            return out

        if frame.t - entered[1] < self.config.dwell_s:
            return out

        item = self.well_item.get(here.id, "")
        if not item:
            return out

        last = self._last_counted.get(item)
        if last is not None and frame.t - last < self.config.gap_for(item):
            return out
        self._last_counted[item] = frame.t

        # Observe EVERY well, required or not.  The old pipeline dropped
        # non-required wells here; that is what made extras undetectable.
        if self.validator is not None:
            self.validator.observe_place(item, t=frame.t, track_id=tid)
        expected = bool(self.validator and item in self.validator.required)
        out.append(Event(frame.t, "place", {
            "item": item, "well": here.name, "track_id": tid, "expected": expected,
        }))
        return out

    def _well_at(self, point, width: int, height: int) -> Optional[Zone]:
        for zone in self.wells:
            if zone.contains(point, width, height):
                return zone
        return None

    # -- run / finish ----------------------------------------------------

    def run(self, frames: Iterable[Frame]) -> "AnalysisEngine":
        for frame in frames:
            self.process(frame)
        return self

    def finish(self) -> Optional[Verdict]:
        if self.validator is None:
            return None
        return self.validator.validate(final=True)

    def summary(self) -> Dict[str, Any]:
        verdict = self.finish()
        return {
            "frames": self.frames_seen,
            "events": [e.to_dict() for e in self.events],
            "verdict": verdict.to_dict() if verdict else None,
        }
