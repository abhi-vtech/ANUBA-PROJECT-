import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.schemas import Action, Detection, FlowSignal, HandState, Zone
from src.zones import ZoneManager

logger = logging.getLogger(__name__)

# Ingredients applied with a ladle / spoon.
# Their pending picks survive until the hand leaves the zone (zone-exit) or
# enters the assembly zone — whichever comes first.
_LADLE_INGREDIENTS: frozenset = frozenset({
    "sauerkraut",
    "yellow cheese (sliced)", "yellow cheese sliced",
    "pickle swears", "pickel swears", "pickle", "pickle_spears", "pickle spears",
    "pickles (spears)", "pickles (rounds)", "pickle_rounds", "pickles",
    "chilli", "chili",
})

# Items that require the hand to actually reach the assembly zone (or hotdog)
# before the pick is confirmed.  A bin dwell of ASSEMBLY_CONFIRM_DWELL_MS is
# required first, then the hand MUST transition to assembly — if the hand
# leaves the bin without going to assembly the pending pick is silently dropped.
# This eliminates false picks caused by short hovers over ingredient trays.
_ASSEMBLY_CONFIRM_INGREDIENTS: frozenset = frozenset({
    "onions",
    "diced onions",
    "diced_onions",
    "grilled onions",
})
ASSEMBLY_CONFIRM_DWELL_MS: int = 1000  # 1 second minimum bin contact


def _boxes_overlap(bbox_a, bbox_b) -> bool:
    """Return True when two bounding boxes (x1,y1,x2,y2) intersect."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


@dataclass
class PendingPick:
    ingredient: str
    zone_id: str
    timestamp: float
    centroid: Tuple[float, float]
    confirmed: bool = True  # False for items that need assembly-entry to confirm


@dataclass
class CarriedItem:
    ingredient: str
    zone_id: str
    timestamp: float


@dataclass
class TrackState:
    current_zone: Optional[Zone] = None
    zone_entry_time: float = 0.0
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
                    # If we inherited pending_picks, check if any are expired
                    for zid in list(inherited.pending_picks.keys()):
                        pp = inherited.pending_picks[zid]
                        if (now - pp.timestamp) * 1000 > self.transition_timeout_ms:
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

            # ── Expire non-ladle pending picks ─────────────────────────────────
            # Ladle ingredients survive until zone-exit (handled below) so they
            # are excluded from the normal transition timeout.
            expired_zones = []
            for zid, pp in state.pending_picks.items():
                if pp.ingredient.lower() in _LADLE_INGREDIENTS:
                    continue  # ladle picks survive until zone-exit
                if (now - pp.timestamp) * 1000 > self.transition_timeout_ms:
                    logger.debug(
                        "Pending pick for zone '%s' (ingredient '%s') timed out "
                        "after %.0fms for track %d",
                        pp.zone_id,
                        pp.ingredient,
                        (now - pp.timestamp) * 1000,
                        det.track_id,
                    )
                    actions.append(
                        Action(
                            track_id=det.track_id,
                            zone_id=pp.zone_id,
                            zone_name=pp.ingredient,
                            action_type="hover",
                            timestamp=now,
                            duration_ms=self.pick_dwell_ms,
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

            # Speed check for overflight filter
            speed = self._speed(state, centroid, now, frame_width, frame_height)
            is_fast = speed > self.max_speed

            # ── Ladle ingredient: immediate place on hotdog contact ─────────────
            # If the hand has a ladle-ingredient pending pick AND directly overlaps
            # a hotdog bbox, fire pick+place immediately and consume the pending
            # pick.  This is the fastest possible response to the worker adding
            # chilli/sauerkraut/pickle directly to a hotdog on the prep belt.
            # Per-hotdog-ID counting is intentionally omitted here; it will be
            # added back as a separate feature once the core edge-cases are stable.
            overlaps_hotdog = any(
                d.class_name == "hot-dog" and _boxes_overlap(det.bbox, d.bbox)
                for d in detections
            )

            if overlaps_hotdog and state.pending_picks:
                confirmed_any = False
                for pp_zone_id, pp in list(state.pending_picks.items()):
                    if pp.ingredient.lower() not in _LADLE_INGREDIENTS:
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
                        "Ladle ingredient '%s' placed via hotdog-overlap shortcut "
                        "(track=%d)",
                        pp.ingredient, det.track_id,
                    )
                if confirmed_any:
                    state.refresh_state()

            # ── Zone transition logic ───────────────────────────────────────────
            if zone is None:
                if state.current_zone is not None:
                    # Hand LEFT a zone.  For any remaining ladle pending picks,
                    # fire pick+place now — the zone-exit is the signal that the
                    # application is complete (e.g. worker ladles chilli onto a
                    # hotdog that the model didn't detect, then lifts hand away).
                    for pp_zone_id, pp in list(state.pending_picks.items()):
                        if pp.ingredient.lower() in _LADLE_INGREDIENTS:
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
                            logger.debug(
                                "Ladle ingredient '%s' zone-exit place fired (track=%d)",
                                pp.ingredient, det.track_id,
                            )
                            del state.pending_picks[pp_zone_id]
                        elif not pp.confirmed:
                            # Assembly-confirm item (e.g. onions): hand left bin zone.
                            # We DO NOT delete it immediately; we let it survive during transit
                            # across 'None' space. It will timeout naturally via transition_timeout_ms
                            # if it doesn't reach the assembly zone.
                            logger.debug(
                                "Unconfirmed pre-pick for '%s' kept alive during transit (track=%d)",
                                pp.ingredient, det.track_id,
                            )
                    # Left a zone — reset entry time
                    state.current_zone = None
                    state.zone_entry_time = now
                    state.flow_contact_count = 0
                    state.refresh_state()
                continue

            if state.current_zone is not None and zone.id != state.current_zone.id:
                # Zone change — if leaving a bin with pending ladle picks, commit them immediately
                for pp_zone_id, pp in list(state.pending_picks.items()):
                    if pp.ingredient.lower() in _LADLE_INGREDIENTS:
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
                        logger.debug(
                            "Ladle ingredient '%s' committed on zone-change (track=%d)",
                            pp.ingredient, det.track_id,
                        )
                        del state.pending_picks[pp_zone_id]

                state.current_zone = zone
                state.zone_entry_time = now
                state.flow_contact_count = 0

                # If entering assembly zone with pending picks, confirm all
                if zone.zone_type == "assembly" and state.pending_picks:
                    logger.debug(
                        "Assembly entry (zone-change): track=%d confirming %d pending picks: %s",
                        det.track_id,
                        len(state.pending_picks),
                        list(state.pending_picks.keys()),
                    )
                    for pp_zone_id, pp in list(state.pending_picks.items()):
                        elapsed = (now - pp.timestamp) * 1000
                        if not pp.confirmed:
                            # Assembly-confirm item (e.g. onions): hand reached assembly.
                            # NOW fire the pickup action to confirm the pick.
                            logger.debug(
                                "Unconfirmed pre-pick for '%s' CONFIRMED by assembly entry "
                                "(track=%d)",
                                pp.ingredient, det.track_id,
                            )
                            actions.append(
                                Action(
                                    track_id=det.track_id,
                                    zone_id=pp.zone_id,
                                    zone_name=pp.ingredient,
                                    action_type="pickup",
                                    timestamp=now,
                                    duration_ms=elapsed,
                                )
                            )
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
                        if pp.ingredient.lower() in _LADLE_INGREDIENTS:
                            # Ladle ingredients: place immediately on assembly entry
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
                        else:
                            state.carried_items.append(
                                CarriedItem(
                                    ingredient=pp.ingredient,
                                    zone_id=pp.zone_id,
                                    timestamp=now,
                                )
                            )
                    state.pending_picks.clear()

                state.refresh_state()
                continue

            if state.current_zone is not None and zone.id == state.current_zone.id:
                # Same zone — check dwell time
                elapsed_ms = (now - state.zone_entry_time) * 1000

                if zone.zone_type == "bin" and zone.id not in state.pending_picks:
                    # Flow-augmented dwell: reduce threshold when co-motion confirms contact
                    flow_signal = (
                        flow_signals.get(det.track_id) if flow_signals else None
                    )
                    if flow_signal and flow_signal.is_contact:
                        state.flow_contact_count += 1
                    else:
                        state.flow_contact_count = 0

                    if (
                        state.flow_contact_count >= self.flow_contact_threshold
                        and flow_signal is not None
                    ):
                        effective_dwell_ms = self.co_motion_dwell_ms
                    else:
                        effective_dwell_ms = self.pick_dwell_ms

                    if zone.name.lower() == "sauerkraut":
                        effective_dwell_ms = max(effective_dwell_ms, 1000)

                    # Chilli bins (zone_12, zone_13): require 800ms dwell threshold
                    # so casual hand transit or quick pass-throughs do not trigger false picks.
                    if "chili" in zone.name.lower() or "chilli" in zone.name.lower():
                        effective_dwell_ms = max(effective_dwell_ms, 325)

                    # Sports wax pepper bin: cap dwell at 1000 ms max.
                    _zone_lower = zone.name.lower()
                    if (
                        "sport" in _zone_lower
                        or "wax pepper" in _zone_lower
                        or ("wax" in _zone_lower and "pepper" in _zone_lower)
                    ):
                        effective_dwell_ms = max(effective_dwell_ms, 1000)

                    # Yellow cheese (sliced): require 1500ms dwell to prevent false picks from adjacent pickel swears hand overlap.
                    if "yellow cheese" in _zone_lower or "cheese" in _zone_lower:
                        effective_dwell_ms = max(effective_dwell_ms, 1500)

                    # Pickles / pickle spears: reduced to 150 ms for instant & responsive picks.
                    if "pickle" in _zone_lower or "spear" in _zone_lower or "swear" in _zone_lower:
                        effective_dwell_ms = 150

                    # Polish hot dog bin: cap dwell at 1000 ms max.
                    if "polish" in _zone_lower or "hot dog" in _zone_lower:
                        effective_dwell_ms = max(effective_dwell_ms, 1000)

                    # Onions / diced onions: require 1 second bin dwell AND then hand
                    # must reach assembly before pick is confirmed.  The pending pick is
                    # stored as unconfirmed; it will be silently dropped if the hand
                    # leaves the bin without going to assembly.
                    is_assembly_confirm = zone.name.lower() in _ASSEMBLY_CONFIRM_INGREDIENTS or "onion" in zone.name.lower()
                    if is_assembly_confirm:
                        if state.flow_contact_count >= self.flow_contact_threshold:
                            effective_dwell_ms = 400  # Quick scoop with confirmed motion
                        else:
                            effective_dwell_ms = 1500 # High dwell if just hovering/resting

                    if elapsed_ms >= effective_dwell_ms and not is_fast:
                        # For assembly-confirm items: store as unconfirmed (no pickup action yet).
                        # Pickup will only fire when hand reaches assembly.
                        needs_assembly_confirm = is_assembly_confirm
                        state.pending_picks[zone.id] = PendingPick(
                            ingredient=zone.name,
                            zone_id=zone.id,
                            timestamp=now,
                            centroid=centroid,
                            confirmed=not needs_assembly_confirm,
                        )
                        if not needs_assembly_confirm:
                            # Normal items: fire pickup immediately as before
                            actions.append(
                                Action(
                                    track_id=det.track_id,
                                    zone_id=zone.id,
                                    zone_name=zone.name,
                                    action_type="pickup",
                                    timestamp=now,
                                    duration_ms=elapsed_ms,
                                )
                            )
                            logger.debug(
                                "Pick qualified: track=%d zone=%s ingredient=%s, "
                                "pending_picks=%s",
                                det.track_id,
                                zone.id,
                                zone.name,
                                list(state.pending_picks.keys()),
                            )
                        else:
                            # Assembly-confirm items: pick is pre-registered but not yet committed.
                            # It will only count when the hand reaches assembly.
                            logger.debug(
                                "Pre-pick (unconfirmed — awaits assembly): track=%d "
                                "zone=%s ingredient=%s dwell=%.0fms",
                                det.track_id,
                                zone.id,
                                zone.name,
                                elapsed_ms,
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

                # If entering assembly zone with pending picks, confirm all
                if zone.zone_type == "assembly" and state.pending_picks:
                    logger.debug(
                        "Assembly entry (from None): track=%d confirming %d pending picks: %s",
                        det.track_id,
                        len(state.pending_picks),
                        list(state.pending_picks.keys()),
                    )
                    for pp_zone_id, pp in list(state.pending_picks.items()):
                        elapsed = (now - pp.timestamp) * 1000
                        if not pp.confirmed:
                            # Assembly-confirm item (e.g. onions): hand reached assembly.
                            # NOW fire the pickup action to confirm the pick.
                            logger.debug(
                                "Unconfirmed pre-pick for '%s' CONFIRMED by assembly entry "
                                "(track=%d)",
                                pp.ingredient, det.track_id,
                            )
                            actions.append(
                                Action(
                                    track_id=det.track_id,
                                    zone_id=pp.zone_id,
                                    zone_name=pp.ingredient,
                                    action_type="pickup",
                                    timestamp=now,
                                    duration_ms=elapsed,
                                )
                            )
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
                        if pp.ingredient.lower() in _LADLE_INGREDIENTS:
                            # Ladle ingredients: place immediately on assembly entry
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
                        else:
                            state.carried_items.append(
                                CarriedItem(
                                    ingredient=pp.ingredient,
                                    zone_id=pp.zone_id,
                                    timestamp=now,
                                )
                            )
                    state.pending_picks.clear()

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
                # Don't delete from self.tracks yet — orphan_timeout handles that

        # Clean up expired lost tracks
        for track_id in list(self.lost_tracks.keys()):
            if now - self.lost_tracks[track_id].last_seen > self.dedup_timeout_s:
                del self.lost_tracks[track_id]

        return actions
