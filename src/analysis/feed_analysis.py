"""Feed analysis for the dashboard's Analysis window.

What it follows, every frame:

* **Hotdogs and their lifecycle** -- on counter -> wrapping -> wrapped -> outgoing
  - wrapping: a wrapper or clamshell covers the hotdog (`wrapping_cover_ratio`)
    for `wrapping_dwell_s`
  - wrapped: a `wrapped` detection over the hotdog, a `closed_reg_clamshell`
    there, or -- for a clamshell -- the hotdog stops being detected while the
    clamshell stays in the same place (`wrapped_same_place_px`) for
    `wrapped_same_place_s`
  - outgoing: a hand touches the exit line (config/exit_line.json) and the
    oldest wrapped hotdog goes out: once per hotdog, one per touch
* **Sauces added**, from the pipeline's existing sauce events, per hotdog
* **Other detected items added** -- buns and fries placed in the assembly zone
* **Detections** per class: live, frames seen, and track ids seen

Ingredients picked from the ROI bins (config/zones.json, zone_type "bin") are
deliberately left out.  This module only reads what the pipeline already
computes; nothing it decides feeds back into tracking, KDS or order accuracy.

Rule thresholds come from the `lifecycle:` section of config/model.yaml.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from src.analysis.exit_detector import ExitDetector, ExitEvent, _bbox_intersects_line

logger = logging.getLogger(__name__)

PACKAGING = {"wrapper", "reg_clamshell", "black_clamshell", "white_clamshell"}
CLAMSHELLS = {"reg_clamshell", "black_clamshell", "white_clamshell"}
ADDED_ITEMS = {"burger_bun", "french_fries"}
STATES = ("on_counter", "wrapping", "wrapped", "outgoing")

DEFAULTS = {
    "wrapping_cover_ratio": 0.5,
    "wrapping_dwell_s": 0.4,
    "wrapped_same_place_s": 2.0,
    "wrapped_same_place_px": 40,
    "outgoing_cooldown_s": 2.0,
    "outgoing_window_s": 120,
    "lost_after_s": 10,
}

# A hotdog track that reappears under a new id this close to where a hotdog was
# lost, this soon, is the same hotdog (the detector's ids fragment).
MERGE_RADIUS_PX = 80
MERGE_GAP_S = 3.0
# An added item must sit in the assembly zone this long, and is attached to a
# hotdog within this distance.
ITEM_DWELL_S = 0.5
ITEM_NEAR_PX = 250
ITEM_REPEAT_PX = 60
ITEM_REPEAT_S = 15.0

_NAMES = {
    "ketchup_sauce": "Ketchup",
    "yellow_mustard_sauce": "Mustard",
    "burger_bun": "Bun",
    "french_fries": "Fries",
    "wrapper": "wrapper",
    "reg_clamshell": "clamshell",
    "black_clamshell": "black clamshell",
    "white_clamshell": "white clamshell",
}


def _pretty(name: Optional[str]) -> str:
    if not name:
        return ""
    return _NAMES.get(name, name.replace("_", " "))


def _center(b) -> Tuple[float, float]:
    return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0


def _dist(a, b) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _coverage(inner, outer) -> float:
    """Share of `inner`'s box area that `outer` covers."""
    area = max(0, inner[2] - inner[0]) * max(0, inner[3] - inner[1])
    if area <= 0:
        return 0.0
    w = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (w * h) / area


def _intersects(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _clock(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    s = int(max(0.0, seconds))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def bin_zone_names(zones_path) -> set:
    """Lower-cased names of the ingredient-bin ROIs in config/zones.json."""
    try:
        data = json.loads(Path(zones_path).read_text())
    except (OSError, ValueError):
        return set()
    zones = data if isinstance(data, list) else data.get("zones", [])
    return {str(z.get("name", "")).strip().lower() for z in zones if z.get("zone_type") == "bin"}


@dataclass
class _Hotdog:
    hid: int
    first_seen: float
    last_seen: float
    bbox: Tuple[int, int, int, int]
    state: str = "on_counter"
    lost: bool = False
    packaging: Optional[str] = None
    wrapped_by: Optional[str] = None
    wrapping_at: Optional[float] = None
    wrapped_at: Optional[float] = None
    outgoing_at: Optional[float] = None
    cover_since: Optional[float] = None
    vanished_at: Optional[float] = None
    clam_ref: Optional[Tuple[float, float]] = None
    clam_seen_at: Optional[float] = None
    sauces: List[dict] = field(default_factory=list)
    items: List[dict] = field(default_factory=list)

    @property
    def last_activity(self) -> float:
        return max(self.last_seen, self.wrapping_at or 0.0, self.wrapped_at or 0.0, self.outgoing_at or 0.0)

    def public(self) -> dict:
        return {
            "id": self.hid,
            "state": self.state,
            "lost": self.lost,
            "packaging": _pretty(self.packaging) or None,
            "wrapped_by": self.wrapped_by,
            "first_seen": _clock(self.first_seen),
            "wrapping_at": _clock(self.wrapping_at),
            "wrapped_at": _clock(self.wrapped_at),
            "outgoing_at": _clock(self.outgoing_at),
            "sauces": list(self.sauces),
            "items": list(self.items),
        }


class FeedAnalyzer:
    def __init__(self, zones=None, zones_path="config/zones.json", exit_config="config/exit_line.json",
                 lifecycle: Optional[dict] = None):
        self.cfg = {**DEFAULTS, **{k: v for k, v in (lifecycle or {}).items() if k in DEFAULTS}}
        self.zones = zones
        self.bin_names = bin_zone_names(zones_path)
        self.exit = ExitDetector(str(exit_config)) if Path(str(exit_config)).exists() else None

        self.hotdogs: Dict[int, _Hotdog] = {}
        self.track_to_hotdog: Dict[int, int] = {}
        self._next_id = 1

        self.events: deque = deque(maxlen=50)
        self.all_events: List[dict] = []
        self.live: Dict[str, int] = {}
        self.class_frames: Dict[str, int] = {}
        self.class_tracks: Dict[str, set] = {}
        self.sauce_counts: Dict[str, int] = {}
        self.item_counts: Dict[str, int] = {}
        self.bin_items_skipped = 0
        self.line_touches = 0
        self.touches_without_order = 0
        self.outgoing_total = 0

        self._touching = False
        self._last_touch_t = -1e9
        self._item_since: Dict[Tuple[str, int], float] = {}
        self._item_done: set = set()
        self._recent_items: deque = deque(maxlen=40)
        self.frame_idx = 0
        self.video_time = 0.0

    # ── Per frame ────────────────────────────────────────────────────────────

    def update(self, *, frame_idx: int, video_time: float, frame_size: Tuple[int, int],
               detections: Iterable, visible_detections: Iterable, actions: Iterable = (),
               wrapping_events: Iterable = (), hotdog_tracker=None) -> None:
        t = float(video_time)
        self.frame_idx, self.video_time = frame_idx, t
        detections = list(detections)

        live: Dict[str, int] = {}
        for d in visible_detections:
            live[d.class_name] = live.get(d.class_name, 0) + 1
            if d.track_id is not None and d.track_id >= 0:
                self.class_tracks.setdefault(d.class_name, set()).add(d.track_id)
        for name in live:
            self.class_frames[name] = self.class_frames.get(name, 0) + 1
        self.live = live

        visible = self._associate([d for d in detections if d.class_name == "hot-dog"], t)
        self._lifecycle(
            visible,
            [d for d in detections if d.class_name in PACKAGING],
            [d for d in detections if d.class_name == "wrapped"],
            [d for d in detections if d.class_name == "closed_reg_clamshell"],
            wrapping_events,
            hotdog_tracker,
            t,
        )
        self._actions(actions, hotdog_tracker, t)
        self._items(detections, frame_size, t)
        self._outgoing([d for d in detections if d.class_name == "hand"], frame_size, t)

    def draw(self, frame):
        """Draw the exit line (red while a hand touches it) and the outgoing banner."""
        if self.exit is not None:
            self.exit.draw_overlay(frame)
        return frame

    # ── Hotdog identity ──────────────────────────────────────────────────────

    def _associate(self, hotdog_dets: List, t: float) -> set:
        visible: set = set()
        for d in hotdog_dets:
            tid = d.track_id if d.track_id is not None else -1
            hid = self.track_to_hotdog.get(tid) if tid >= 0 else None
            if hid is None:
                c = _center(d.bbox)
                best, best_d = None, float(MERGE_RADIUS_PX)
                for hd in self.hotdogs.values():
                    if hd.hid in visible or hd.state in ("wrapped", "outgoing"):
                        continue
                    if t - hd.last_seen > MERGE_GAP_S:
                        continue
                    dd = _dist(c, _center(hd.bbox))
                    if dd <= best_d:
                        best, best_d = hd, dd
                if best is None:
                    best = _Hotdog(hid=self._next_id, first_seen=t, last_seen=t, bbox=tuple(d.bbox))
                    self.hotdogs[best.hid] = best
                    self._next_id += 1
                    self._event(t, "hotdog", best.hid, f"Hotdog #{best.hid} on the counter")
                hid = best.hid
                if tid >= 0:
                    self.track_to_hotdog[tid] = hid
            if hid in visible:
                continue  # a second detection of the same hotdog this frame
            hd = self.hotdogs[hid]
            hd.bbox, hd.last_seen, hd.lost = tuple(d.bbox), t, False
            visible.add(hid)
        return visible

    def _resolve(self, tid, tracker) -> Optional[int]:
        """Analyzer hotdog for a detector track id or a HotdogTracker id."""
        if tid is None:
            return None
        hid = self.track_to_hotdog.get(tid)
        if hid is not None:
            return hid
        rec = None
        if tracker is not None:
            rec = getattr(tracker, "_records", {}).get(tid) or getattr(tracker, "_retired_records", {}).get(tid)
        if rec is None:
            return None
        c = _center(rec.bbox)
        candidates = [hd for hd in self.hotdogs.values() if hd.state != "outgoing"]
        best = min(candidates, key=lambda hd: _dist(c, _center(hd.bbox)), default=None)
        if best is not None and _dist(c, _center(best.bbox)) <= 150:
            return best.hid
        return None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _lifecycle(self, visible, packaging, wrapped_dets, closed_clams, wrapping_events, tracker, t) -> None:
        cfg = self.cfg
        closing = set()
        for ev in wrapping_events or ():
            if ev.get("event") == "closing":
                hid = self._resolve(ev.get("hotdog_tid"), tracker)
                if hid is not None:
                    closing.add(hid)

        for hd in self.hotdogs.values():
            if hd.state in ("wrapped", "outgoing"):
                continue

            # Wrapped: a wrapped hotdog, or a closed clamshell, where it was.
            recent = hd.hid in visible or t - hd.last_seen <= cfg["wrapped_same_place_s"] + 1.0
            if recent:
                hit = next(((d, "wrapped detection", "wrapper") for d in wrapped_dets
                            if _intersects(d.bbox, hd.bbox)), None)
                hit = hit or next(((d, "closed clamshell detected", "reg_clamshell") for d in closed_clams
                                   if _intersects(d.bbox, hd.bbox)), None)
                if hit is not None:
                    self._set_wrapped(hd, t, hit[1], hd.packaging or hit[2])
                    continue

            if hd.hid in visible:
                hd.vanished_at = hd.clam_ref = None
                best_cls, best_cov = None, 0.0
                for p in packaging:
                    cov = _coverage(hd.bbox, p.bbox)
                    if cov > best_cov:
                        best_cls, best_cov = p.class_name, cov
                if hd.state == "on_counter":
                    if best_cov >= cfg["wrapping_cover_ratio"]:
                        hd.cover_since = t if hd.cover_since is None else hd.cover_since
                        if t - hd.cover_since >= cfg["wrapping_dwell_s"]:
                            self._set_wrapping(hd, t, best_cls)
                    else:
                        hd.cover_since = None
                    if hd.state == "on_counter" and hd.hid in closing:
                        self._set_wrapping(hd, t, best_cls or "wrapper")
                continue

            # Not visible this frame.
            if hd.state == "wrapping" and hd.packaging in CLAMSHELLS:
                self._clamshell_closed(hd, packaging, t)
            if hd.state in ("on_counter", "wrapping") and not hd.lost:
                limit = cfg["lost_after_s"] + (cfg["wrapped_same_place_s"] if hd.state == "wrapping" else 0)
                if t - hd.last_seen > limit:
                    hd.lost = True

    def _clamshell_closed(self, hd: _Hotdog, packaging, t: float) -> None:
        """Hotdog gone while its clamshell stays in the same place long enough."""
        cfg = self.cfg
        clams = [p for p in packaging if p.class_name in CLAMSHELLS]
        if hd.vanished_at is None:
            hd.vanished_at = t
        if hd.clam_ref is None:
            last = _center(hd.bbox)
            near = min(clams, key=lambda p: _dist(_center(p.bbox), last), default=None)
            if near is not None and _dist(_center(near.bbox), last) <= max(80.0, cfg["wrapped_same_place_px"] * 2):
                hd.clam_ref, hd.clam_seen_at = _center(near.bbox), t
            elif t - hd.vanished_at > 1.0:
                hd.vanished_at = None  # no clamshell to watch; look again next frame
            return
        if any(_dist(_center(p.bbox), hd.clam_ref) <= cfg["wrapped_same_place_px"] for p in clams):
            hd.clam_seen_at = t
            if t - hd.vanished_at >= cfg["wrapped_same_place_s"]:
                self._set_wrapped(hd, t, "clamshell closed in place", hd.packaging)
        elif t - (hd.clam_seen_at or t) > 0.5:
            hd.vanished_at = hd.clam_ref = None  # the clamshell moved or disappeared

    def _set_wrapping(self, hd: _Hotdog, t: float, packaging: Optional[str]) -> None:
        hd.state, hd.wrapping_at, hd.cover_since = "wrapping", t, None
        hd.packaging = packaging or "wrapper"
        self._event(t, "wrapping", hd.hid, f"Hotdog #{hd.hid} wrapping in a {_pretty(hd.packaging)}")

    def _set_wrapped(self, hd: _Hotdog, t: float, how: str, packaging: Optional[str]) -> None:
        if hd.wrapping_at is None:
            hd.wrapping_at = t
        hd.state, hd.wrapped_at, hd.wrapped_by, hd.lost = "wrapped", t, how, False
        hd.packaging = packaging or hd.packaging or "wrapper"
        self._event(t, "wrapped", hd.hid, f"Hotdog #{hd.hid} wrapped ({how})")

    # ── Sauces and items ─────────────────────────────────────────────────────

    def _actions(self, actions, tracker, t: float) -> None:
        for a in actions or ():
            kind = getattr(a, "action_type", None)
            if kind in ("pick", "place", "pickup"):
                self.bin_items_skipped += 1  # ROI-bin ingredients are not analysed
                continue
            if kind != "sauce":
                continue
            name = str(getattr(a, "zone_name", "") or "sauce")
            if name.strip().lower() in self.bin_names:
                self.bin_items_skipped += 1
                continue
            self.sauce_counts[name] = self.sauce_counts.get(name, 0) + 1
            hid = self._resolve(getattr(a, "resolved_hotdog_tid", None), tracker)
            if hid is not None:
                self.hotdogs[hid].sauces.append({"name": _pretty(name), "at": _clock(t)})
                self._event(t, "sauce", hid, f"{_pretty(name)} on hotdog #{hid}")
            else:
                self._event(t, "sauce", None, f"{_pretty(name)} applied")

    def _items(self, detections, frame_size, t: float) -> None:
        w, h = frame_size
        present = set()
        for d in detections:
            if d.class_name not in ADDED_ITEMS or d.track_id is None or d.track_id < 0:
                continue
            key = (d.class_name, d.track_id)
            present.add(key)
            if key in self._item_done:
                continue
            zone = self.zones.get_zone_for_bbox(d.bbox, w, h) if self.zones is not None else None
            if zone is None or getattr(zone, "zone_type", None) != "assembly":
                self._item_since.pop(key, None)
                continue
            since = self._item_since.setdefault(key, t)
            if t - since < ITEM_DWELL_S:
                continue
            self._item_done.add(key)
            c = _center(d.bbox)
            # The detector's ids fragment: the same item a moment ago is not new.
            if any(name == d.class_name and t - at <= ITEM_REPEAT_S and _dist(c, pos) <= ITEM_REPEAT_PX
                   for name, pos, at in self._recent_items):
                continue
            self._recent_items.append((d.class_name, c, t))
            self.item_counts[d.class_name] = self.item_counts.get(d.class_name, 0) + 1
            candidates = [hd for hd in self.hotdogs.values() if hd.state in ("on_counter", "wrapping") and not hd.lost]
            near = min(candidates, key=lambda hd: _dist(c, _center(hd.bbox)), default=None)
            if near is not None and _dist(c, _center(near.bbox)) <= ITEM_NEAR_PX:
                near.items.append({"name": _pretty(d.class_name), "at": _clock(t)})
                self._event(t, "item", near.hid, f"{_pretty(d.class_name)} with hotdog #{near.hid}")
            else:
                self._event(t, "item", None, f"{_pretty(d.class_name)} placed in assembly")
        for key in [k for k in self._item_since if k not in present]:
            self._item_since.pop(key, None)

    # ── Outgoing ─────────────────────────────────────────────────────────────

    def _outgoing(self, hands, frame_size, t: float) -> None:
        if self.exit is None:
            return
        w, h = frame_size
        p1, p2 = self.exit.get_pixel_coords(w, h)
        touching = any(_bbox_intersects_line(tuple(int(v) for v in d.bbox), p1, p2) for d in hands)
        self.exit.is_active_crossing = touching
        rising = touching and not self._touching
        self._touching = touching
        if not rising or t - self._last_touch_t < self.cfg["outgoing_cooldown_s"]:
            return
        self._last_touch_t = t
        self.line_touches += 1
        ready = [hd for hd in self.hotdogs.values()
                 if hd.state == "wrapped" and hd.wrapped_at is not None
                 and t - hd.wrapped_at <= self.cfg["outgoing_window_s"]]
        if not ready:
            self.touches_without_order += 1
            return
        hd = min(ready, key=lambda x: x.wrapped_at)
        hd.state, hd.outgoing_at = "outgoing", t
        self.outgoing_total += 1
        message = f"Hotdog #{hd.hid} outgoing"
        self._event(t, "outgoing", hd.hid, message)
        self.exit.exited_history.append(ExitEvent(timestamp=time.time(), exited_hotdog_ids=[hd.hid], message=message))
        del self.exit.exited_history[:-20]
        self.exit.last_exit_time = time.time()

    # ── Output ───────────────────────────────────────────────────────────────

    def _event(self, t: float, kind: str, hid: Optional[int], text: str) -> None:
        entry = {"at": _clock(t), "t": round(t, 2), "kind": kind, "hotdog": hid, "text": text}
        self.events.appendleft(entry)
        self.all_events.append(entry)
        logger.info("[ANALYSIS] %s %s", entry["at"], text)

    def snapshot(self) -> dict:
        states = {s: 0 for s in STATES}
        for hd in self.hotdogs.values():
            if hd.lost and hd.state in ("on_counter", "wrapping"):
                continue
            states[hd.state] += 1
        shown = [hd for hd in sorted(self.hotdogs.values(), key=lambda x: x.last_activity, reverse=True)
                 if not (hd.lost and hd.state == "on_counter" and not hd.sauces and not hd.items)]
        return {
            "video_time": _clock(self.video_time),
            "frame": self.frame_idx,
            "states": states,
            "totals": {
                "hotdogs": len(self.hotdogs),
                "wrapping": sum(1 for hd in self.hotdogs.values() if hd.wrapping_at is not None),
                "wrapped": sum(1 for hd in self.hotdogs.values() if hd.wrapped_at is not None),
                "outgoing": self.outgoing_total,
                "sauces": sum(self.sauce_counts.values()),
                "items": sum(self.item_counts.values()),
                "line_touches": self.line_touches,
                "touches_without_order": self.touches_without_order,
                "bin_items_skipped": self.bin_items_skipped,
            },
            "sauces": {_pretty(k): v for k, v in sorted(self.sauce_counts.items())},
            "items": {_pretty(k): v for k, v in sorted(self.item_counts.items())},
            "detections_live": dict(sorted(self.live.items())),
            "detections_frames": dict(sorted(self.class_frames.items(), key=lambda kv: -kv[1])),
            "detections_tracks": {k: len(v) for k, v in sorted(self.class_tracks.items(), key=lambda kv: -len(kv[1]))},
            "hotdogs": [hd.public() for hd in shown[:12]],
            "events": list(self.events),
            "exit_line": {"configured": self.exit is not None, "touching": self._touching},
            "rules": dict(self.cfg),
        }

    def write_json(self, path) -> None:
        out = self.snapshot()
        out["hotdogs"] = [hd.public() for hd in sorted(self.hotdogs.values(), key=lambda x: x.hid)]
        out["events"] = self.all_events
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=2))
        tmp.replace(p)
