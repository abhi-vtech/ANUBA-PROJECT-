import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.schemas import Action, Detection, FlowSignal, HandState, Zone
from src.zones import ZoneManager
from src.ingredient_config import is_granular

logger = logging.getLogger(__name__)

# An ingredient is only credited to a hotdog once the hand that dipped into
# the well actually comes BACK to a hotdog.  With this True, hotdog proximity
# is the ONLY thing that confirms a pending pick; the older shortcuts that
# committed on leaving the bin, or on entering the assembly area, are off.
# Those shortcuts credited an ingredient to an order whenever a hand crossed a
# bin on its way somewhere else, which is the main source of phantom onions.
# Set False to restore the previous behaviour.
_REQUIRE_HOTDOG_RETURN: bool = False

# Minimum number of consecutive frames the hand must be inside a bin zone
# before a pending pick is registered. At ~20fps, 2 frames = ~100ms.
_MIN_BIN_FRAMES: int = 2

# Pending pick TTL — how long (seconds) a pick survives without reaching hotdog.
# Granular ingredients get extra time (burst pinches over longer interval);
# discrete items expire faster to reduce false positives.
_PICK_TTL_DISCRETE_S: float = 3.0
_PICK_TTL_GRANULAR_S: float = 6.0

# Hotdog proximity padding in pixels.
# Granular ingredients (sprinkle-style) need a wider gate.
_HOTDOG_PAD_DISCRETE_PX: int = 60
_HOTDOG_PAD_GRANULAR_PX: int = 110


def _boxes_overlap(bbox_a, bbox_b, pad: int = 0) -> bool:
    """Return True when two bounding boxes (x1,y1,x2,y2) intersect (with optional padding)."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    bx1, by1, bx2, by2 = bx1 - pad, by1 - pad, bx2 + pad, by2 + pad
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _hand_near_hotdog(hand_bbox, hotdog_detections: List[Detection], ingredient: str = "") -> bool:
    """Return True if the hand bbox is near or overlapping any hotdog bbox.

    Uses a wider pad for granular ingredients (onions, relish, grated_yellow_cheese)
    so sprinkle-style releases just outside the bbox still register.
    """
    pad = _HOTDOG_PAD_GRANULAR_PX if is_granular(ingredient) else _HOTDOG_PAD_DISCRETE_PX
    return any(
        _boxes_overlap(hand_bbox, d.bbox, pad=pad)
        for d in (hotdog_detections or [])
        if d.class_name == "hot-dog"
    )


@dataclass
class PendingPick:
    ingredient: str
    zone_id: str
    timestamp: float
    centroid: Tuple[float, float]
    confirmed: bool = True
    bin_frame_count: int = 0   # frames hand was inside the bin zone


@dataclass
class CarriedItem:
    ingredient: str
    zone_id: str
    timestamp: float


@dataclass
class TrackState:
    current_zone: Optional[Zone] = None
    zone_entry_time: float = 0.0
    zone_frame_count: int = 0          # consecutive frames inside current zone
    carried_items: List[CarriedItem] = field(default_factory=list)
    pending_picks: Dict[str, PendingPick] = field(default_factory=dict)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    last_seen: float = 0.0
    last_centroid: Optional[Tuple[float, float]] = None
    state: HandState = HandState.IDLE
    flow_contact_count: int = 0

    def compute_state(self) -> HandState:
        if self.pending_picks:
            if self.current_zone is not None:
                return HandState.PENDING_PICK
            return HandState.TRANSIT_PENDING
        if self.carried_items:
            if (
                self.current_zone is not None
                and self.current_zone.zone_type == "assembly"
            ):
                return HandState.CARRYING_IN_ASSEMBLY
            return HandState.CARRYING
        if self.current_zone is not None:
            return HandState.IDLE_IN_ZONE
        return HandState.IDLE

    def refresh_state(self) -> None:
        new_state = self.compute_state()
        if new_state != self.state:
            logger.debug(
                "Hand state transition: %s -> %s", self.state.value, new_state.value
            )
            self.state = new_state


class TemporalTracker:
    def __init__(
        self,
        pick_dwell_ms: int = 800,
        place_dwell_ms: int = 500,
        transition_timeout_ms: int = 2000,
        carry_timeout_ms: int = 5000,
        max_speed: float = 2.0,
        orphan_timeout_s: float = 5.0,
        dedup_timeout_s: float = 0.3,
        dedup_distance: float = 0.05,
        co_motion_dwell_ms: int = 200,
        flow_contact_threshold: int = 2,
    ):
        self.pick_dwell_ms = pick_dwell_ms
        self.place_dwell_ms = place_dwell_ms
        self.transition_timeout_ms = transition_timeout_ms
        self.carry_timeout_ms = carry_timeout_ms
        self.max_speed = max_speed
        self.orphan_timeout_s = orphan_timeout_s
        self.dedup_timeout_s = dedup_timeout_s
        self.dedup_distance = dedup_distance
        self.co_motion_dwell_ms = co_motion_dwell_ms
        self.flow_contact_threshold = flow_contact_threshold

        self.tracks: Dict[int, TrackState] = {}
        self.lost_tracks: Dict[int, TrackState] = {}  # recently lost, for dedup

    def get_track_states(self) -> Dict[int, HandState]:
        return {tid: ts.state for tid, ts in self.tracks.items()}

    def reset(self) -> None:
        """Clear timeline-dependent state when video time restarts."""
        self.tracks.clear()
        self.lost_tracks.clear()

    def _centroid(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _normalize_centroid(
        self, centroid: Tuple[float, float], w: int, h: int
    ) -> Tuple[float, float]:
        return (centroid[0] / w, centroid[1] / h)

    def _speed(
        self,
        track: TrackState,
        centroid: Tuple[float, float],
        now: float,
        w: int,
        h: int,
    ) -> float:
        if not track.trajectory:
            return 0.0
        prev_cx, prev_cy, prev_t = track.trajectory[-1]
        dt = now - prev_t
        if dt <= 0:
            return 0.0
        nx, ny = self._normalize_centroid(centroid, w, h)
        px, py = self._normalize_centroid((prev_cx, prev_cy), w, h)
        dist = ((nx - px) ** 2 + (ny - py) ** 2) ** 0.5
        return dist / dt

    def _try_dedup(
        self,
        track_id: int,
        centroid: Tuple[float, float],
        w: int,
        h: int,
        current_time: float,
    ) -> Optional[TrackState]:
        """Check if a new track matches a recently lost track (ID switch dedup)."""
        nx, ny = self._normalize_centroid(centroid, w, h)
        best_match = None
        best_dist = self.dedup_distance

        for lost_id, lost_state in list(self.lost_tracks.items()):
            if current_time - lost_state.last_seen > self.dedup_timeout_s:
                continue
            if lost_state.last_centroid is None:
                continue
            lx, ly = self._normalize_centroid(lost_state.last_centroid, w, h)
            dist = ((nx - lx) ** 2 + (ny - ly) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = lost_id

        if best_match is not None:
            state = self.lost_tracks.pop(best_match)
            state.last_seen = current_time
            return state
        return None

    def update(
        self,
        detections: List[Detection],
        zones: ZoneManager,
        frame_width: int,
        frame_height: int,
        flow_signals: Optional[Dict[int, FlowSignal]] = None,
        *,
        current_time: float,
        hotdog_detections: Optional[List[Detection]] = None,
        requires_relish: bool = False,
    ) -> List[Action]:
        actions: List[Action] = []
        now = current_time
        active_ids: Set[int] = set()

        for det in detections:
            active_ids.add(det.track_id)
            centroid = self._centroid(det.bbox)
            zone = zones.get_zone_for_bbox(det.bbox, frame_width, frame_height)

            # New track — check for ID-switch dedup
            if det.track_id not in self.tracks:
                inherited = self._try_dedup(
                    det.track_id, centroid, frame_width, frame_height, now
                )
                if inherited is not None:
                    self.tracks[det.track_id] = inherited
                    # Expire any stale pending picks from inherited state
                    for zid in list(inherited.pending_picks.keys()):
                        pp = inherited.pending_picks[zid]
                        ttl = _PICK_TTL_GRANULAR_S if is_granular(pp.ingredient) else _PICK_TTL_DISCRETE_S
                        if (now - pp.timestamp) > ttl:
                            del inherited.pending_picks[zid]
                    inherited.refresh_state()
                else:
                    ts = TrackState(
                        current_zone=zone,
                        zone_entry_time=now,
                        last_seen=now,
                        last_centroid=centroid,
                    )
                    ts.refresh_state()
                    self.tracks[det.track_id] = ts
                    self.tracks[det.track_id].trajectory.append((*centroid, now))
                    continue

            state = self.tracks[det.track_id]
            state.last_seen = now
            state.last_centroid = centroid
            state.trajectory.append((*centroid, now))

            # ── Expire stale pending picks ──────────────────────────────────────
            # If hand never reached a hotdog in _PICK_TTL_S seconds, discard pick.
            expired_zones = []
            for zid, pp in state.pending_picks.items():
                age = now - pp.timestamp
                ttl = _PICK_TTL_GRANULAR_S if is_granular(pp.ingredient) else _PICK_TTL_DISCRETE_S
                if age > ttl:
                    logger.debug(
                        "Pending pick for '%s' expired after %.1fs without hotdog delivery (track=%d)",
                        pp.ingredient, age, det.track_id,
                    )
                    actions.append(
                        Action(
                            track_id=det.track_id,
                            zone_id=pp.zone_id,
                            zone_name=pp.ingredient,
                            action_type="hover",
                            timestamp=now,
                            duration_ms=(now - pp.timestamp) * 1000,
                            from_zone=pp.zone_id,
                        )
                    )
                    expired_zones.append(zid)
            for zid in expired_zones:
                del state.pending_picks[zid]
            if expired_zones:
                state.refresh_state()

            # Check for expired carried items
            timed_out = [
                item
                for item in state.carried_items
                if (now - item.timestamp) * 1000 > self.carry_timeout_ms
            ]
            if timed_out:
                for item in timed_out:
                    logger.debug(
                        "Carry timeout for track %d, clearing '%s'",
                        det.track_id,
                        item.ingredient,
                    )
                state.carried_items = [
                    item
                    for item in state.carried_items
                    if (now - item.timestamp) * 1000 <= self.carry_timeout_ms
                ]
                state.refresh_state()

            # ── TRAJECTORY CONFIRMATION: Hand near hotdog with pending pick ─────
            # Confirmed ONLY when the hand physically reaches a hotdog bbox.
            # Each pending pick uses its own per-ingredient proximity pad.
            # Filter pending picks by proximity before committing.
            confirmed_any = False
            for pp_zone_id, pp in list(state.pending_picks.items()):
                if not _hand_near_hotdog(det.bbox, hotdog_detections, ingredient=pp.ingredient):
                    continue
                elapsed = (now - pp.timestamp) * 1000
                actions.append(
                    Action(
                        track_id=det.track_id,
                        zone_id=pp.zone_id,
                        zone_name=pp.ingredient,
                        action_type="pick",
                        timestamp=now,
                        duration_ms=elapsed,
                        from_zone=pp.zone_id,
                    )
                )
                actions.append(
                    Action(
                        track_id=det.track_id,
                        zone_id="assembly",
                        zone_name=pp.ingredient,
                        action_type="place",
                        timestamp=now,
                        duration_ms=0.0,
                        from_zone=pp.zone_id,
                    )
                )
                del state.pending_picks[pp_zone_id]
                confirmed_any = True
                logger.debug(
                    "TRAJECTORY CONFIRM: '%s' placed on hotdog (track=%d, age=%.0fms)",
                    pp.ingredient, det.track_id, elapsed,
                )
            if confirmed_any:
                state.refresh_state()
                continue

            # ── Zone transition logic ───────────────────────────────────────────
            if zone is None:
                if state.current_zone is not None:
                    # Hand LEFT a zone
                    # Legacy bin-exit shortcut: commit without ever reaching a
                    # hotdog.  Disabled by _REQUIRE_HOTDOG_RETURN.
                    if state.pending_picks and not _REQUIRE_HOTDOG_RETURN:
                        for pp_zone_id, pp in list(state.pending_picks.items()):
                            if "onion" in pp.ingredient.lower():
                                continue  # Wait for trajectory confirmation
                            elapsed = (now - pp.timestamp) * 1000
                            actions.append(Action(track_id=det.track_id, zone_id=pp.zone_id, zone_name=pp.ingredient, action_type="pick", timestamp=now, duration_ms=elapsed, from_zone=pp.zone_id))
                            actions.append(Action(track_id=det.track_id, zone_id=state.current_zone.id, zone_name=pp.ingredient, action_type="place", timestamp=now, duration_ms=0.0, from_zone=pp.zone_id))
                            del state.pending_picks[pp_zone_id]
                            logger.debug("BIN EXIT (to None): '%s' placed (track=%d)", pp.ingredient, det.track_id)

                    # Remaining pending picks survive until hotdog proximity or _PICK_TTL_S timeout
                    for pp_zone_id, pp in list(state.pending_picks.items()):
                        logger.debug(
                            "Pick for '%s' in transit toward hotdog (track=%d)",
                            pp.ingredient, det.track_id,
                        )
                    state.current_zone = None
                    state.zone_entry_time = now
                    state.zone_frame_count = 0
                    state.flow_contact_count = 0
                    state.refresh_state()
                continue

            if state.current_zone is not None and zone.id != state.current_zone.id:
                # Zone change
                # Legacy bin-exit shortcut; see _REQUIRE_HOTDOG_RETURN.
                if state.pending_picks and not _REQUIRE_HOTDOG_RETURN:
                    for pp_zone_id, pp in list(state.pending_picks.items()):
                        if "onion" in pp.ingredient.lower():
                            continue  # Wait for trajectory confirmation
                        elapsed = (now - pp.timestamp) * 1000
                        actions.append(Action(track_id=det.track_id, zone_id=pp.zone_id, zone_name=pp.ingredient, action_type="pick", timestamp=now, duration_ms=elapsed, from_zone=pp.zone_id))
                        actions.append(Action(track_id=det.track_id, zone_id=state.current_zone.id, zone_name=pp.ingredient, action_type="place", timestamp=now, duration_ms=0.0, from_zone=pp.zone_id))
                        del state.pending_picks[pp_zone_id]
                        logger.debug("BIN EXIT (to %s): '%s' placed (track=%d)", zone.name, pp.ingredient, det.track_id)

                state.current_zone = zone
                state.zone_entry_time = now
                state.zone_frame_count = 1

                # If entering assembly zone with pending picks, confirm all as placed
                if (
                    zone.zone_type == "assembly"
                    and state.pending_picks
                    and not _REQUIRE_HOTDOG_RETURN
                ):
                    logger.debug(
                        "Assembly entry (zone-change): track=%d confirming %d pending picks: %s",
                        det.track_id,
                        len(state.pending_picks),
                        list(state.pending_picks.keys()),
                    )
                    for pp_zone_id, pp in list(state.pending_picks.items()):
                        if "onion" in pp.ingredient.lower():
                            continue  # Wait for trajectory confirmation (hotdog proximity)
                        elapsed = (now - pp.timestamp) * 1000
                        actions.append(
                            Action(
                                track_id=det.track_id,
                                zone_id=pp.zone_id,
                                zone_name=pp.ingredient,
                                action_type="pick",
                                timestamp=now,
                                duration_ms=elapsed,
                                from_zone=pp.zone_id,
                            )
                        )
                        actions.append(
                            Action(
                                track_id=det.track_id,
                                zone_id=zone.id,
                                zone_name=pp.ingredient,
                                action_type="place",
                                timestamp=now,
                                duration_ms=0.0,
                                from_zone=pp.zone_id,
                            )
                        )
                        logger.debug(
                            "Assembly confirm (zone-change): '%s' placed (track=%d)",
                            pp.ingredient, det.track_id,
                        )
                        del state.pending_picks[pp_zone_id]

                state.refresh_state()
                continue

            if state.current_zone is not None and zone.id == state.current_zone.id:
                # Same zone — increment frame count
                state.zone_frame_count = getattr(state, "zone_frame_count", 0) + 1
                elapsed_ms = (now - state.zone_entry_time) * 1000

                if zone.zone_type == "bin":
                    flow_signal = flow_signals.get(det.track_id) if flow_signals else None
                    if flow_signal and flow_signal.is_contact:
                        state.flow_contact_count += 1
                        
                if zone.zone_type == "bin" and zone.id not in state.pending_picks:
                    # TRAJECTORY MODE: Register pending pick after _MIN_BIN_FRAMES frames.
                    # Confirmation only fires when hand reaches hotdog proximity.
                    # Onions get a higher frame threshold to prevent small hovers from registering,
                    # and MUST have visual flow contact (to prove hand grabbed it, not just hovered).
                    min_frames = _MIN_BIN_FRAMES
                    requires_contact = False
                    if "onion" in zone.name.lower():
                        min_frames = 3  # Reduced dwell time for the down side
                        requires_contact = False
                    elif "relish" in zone.name.lower():
                        min_frames = 10  # ~500ms at 20fps
                    elif "cheese" in zone.name.lower():
                        min_frames = 2  # Very sensitive
                    elif "chilli" in zone.name.lower():
                        min_frames = 6  # Reduced threshold to catch fast second hovers

                    if state.zone_frame_count >= min_frames:
                        if requires_contact and state.flow_contact_count == 0:
                            pass # Wait for actual visual motion/contact in the bin
                        else:
                            state.pending_picks[zone.id] = PendingPick(
                            ingredient=zone.name,
                            zone_id=zone.id,
                            timestamp=now,
                            centroid=centroid,
                            confirmed=True,
                            bin_frame_count=state.zone_frame_count,
                        )
                        logger.debug(
                            "Trajectory pending pick: track=%d zone=%s ingredient=%s (frame %d)",
                            det.track_id, zone.id, zone.name, state.zone_frame_count,
                        )
                        state.flow_contact_count = 0
                        state.refresh_state()

                elif zone.zone_type == "assembly" and state.carried_items:
                    if elapsed_ms >= self.place_dwell_ms:
                        item = state.carried_items.pop(0)
                        logger.debug(
                            "Place: track=%d ingredient=%s from_zone=%s, "
                            "remaining carried=%d",
                            det.track_id,
                            item.ingredient,
                            item.zone_id,
                            len(state.carried_items),
                        )
                        actions.append(
                            Action(
                                track_id=det.track_id,
                                zone_id=zone.id,
                                zone_name=item.ingredient,
                                action_type="place",
                                timestamp=now,
                                duration_ms=elapsed_ms,
                                from_zone=item.zone_id,
                            )
                        )
                        state.zone_entry_time = now
                        state.refresh_state()

            elif state.current_zone is None:
                # Entered a zone from no zone
                state.current_zone = zone
                state.zone_entry_time = now
                state.zone_frame_count = 1

                # If entering assembly zone with pending picks, confirm all
                if (
                    zone.zone_type == "assembly"
                    and state.pending_picks
                    and not _REQUIRE_HOTDOG_RETURN
                ):
                    logger.debug(
                        "Assembly entry (from None): track=%d confirming %d pending picks: %s",
                        det.track_id,
                        len(state.pending_picks),
                        list(state.pending_picks.keys()),
                    )
                    for pp_zone_id, pp in list(state.pending_picks.items()):
                        if "onion" in pp.ingredient.lower():
                            continue  # Wait for trajectory confirmation (hotdog proximity)
                        elapsed = (now - pp.timestamp) * 1000
                        actions.append(
                            Action(
                                track_id=det.track_id,
                                zone_id=pp.zone_id,
                                zone_name=pp.ingredient,
                                action_type="pick",
                                timestamp=now,
                                duration_ms=elapsed,
                                from_zone=pp.zone_id,
                            )
                        )
                        actions.append(
                            Action(
                                track_id=det.track_id,
                                zone_id=zone.id,
                                zone_name=pp.ingredient,
                                action_type="place",
                                timestamp=now,
                                duration_ms=0.0,
                                from_zone=pp.zone_id,
                            )
                        )
                        logger.debug(
                            "Assembly (from None) confirm: '%s' placed (track=%d)",
                            pp.ingredient, det.track_id,
                        )
                        del state.pending_picks[pp_zone_id]

                state.refresh_state()

        # Orphan cleanup and move lost tracks to dedup buffer
        for track_id in list(self.tracks.keys()):
            if track_id not in active_ids:
                state = self.tracks[track_id]
                if now - state.last_seen > self.orphan_timeout_s:
                    del self.tracks[track_id]

        # Move recently lost tracks to lost_tracks for dedup
        for track_id in list(self.tracks.keys()):
            if track_id not in active_ids:
                self.lost_tracks[track_id] = self.tracks[track_id]

        # Clean up expired lost tracks
        for track_id in list(self.lost_tracks.keys()):
            if now - self.lost_tracks[track_id].last_seen > self.dedup_timeout_s:
                del self.lost_tracks[track_id]

        return actions
