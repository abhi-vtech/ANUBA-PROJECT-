#!/usr/bin/env python3
"""1-minute DeepStream run of the full production logic: ONNX YOLO11-seg
model, real-time instance masks, IOU tracking, the wrapping -> wrapped ->
outgoing hotdog lifecycle, bin-ingredient exclusion (config/zones.json), and
a tegrastats system sampler -- everything src/feed_analysis.py +
src/system_monitor.py do, ported into a DeepStream pyservicemaker probe.
"""
import json
import math
import re
import subprocess
import threading
import time

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator

import os as _os

WORK = "/work"
CLIP = _os.environ.get("DS_CLIP", f"{WORK}/deepstream_test/clip_1min_start.h264")
CONFIG = f"{WORK}/deepstream_test/config_infer.txt"
TRACKER_CONFIG = "/opt/nvidia/deepstream/deepstream-9.1/samples/configs/deepstream-app/config_tracker_IOU.yml"
ZONES_PATH = f"{WORK}/deepstream_test/zones.json"
RUN_TAG = _os.environ.get("DS_RUN_TAG", "1min_full")
OUT = f"{WORK}/deepstream_test/out_{RUN_TAG}.mkv"
EVENTS_OUT = f"{WORK}/deepstream_test/events_{RUN_TAG}.jsonl"
SYSTEM_OUT = f"{WORK}/deepstream_test/system_{RUN_TAG}.jsonl"
LIVE_PATH = f"{WORK}/deepstream_test/live_{RUN_TAG}.json"
VIDEO_DURATION_S = float(_os.environ.get("DS_VIDEO_DURATION_S", "60"))

CLASSES = ["black_clamshell", "burger_bun", "burger_bun_with_toppings", "closed_reg_clamshell",
           "french_fries", "hand", "hot-dog", "ketchup_sauce", "knife", "reg_clamshell",
           "scoop", "tongs", "white_clamshell", "wrapped", "wrapper", "yellow_mustard_sauce"]
CID = {n: i for i, n in enumerate(CLASSES)}
WRAP_CLASSES = {CID["wrapper"], CID["reg_clamshell"], CID["black_clamshell"], CID["white_clamshell"]}
CLAMSHELL_CLASSES = {CID["reg_clamshell"], CID["black_clamshell"], CID["white_clamshell"]}
CLOSED_CLAMSHELL = CID["closed_reg_clamshell"]
WRAPPED_CLASS = CID["wrapped"]
HOTDOG_CLASS = CID["hot-dog"]
HAND_CLASS = CID["hand"]
SAUCE_CLASSES = {CID["ketchup_sauce"]: "ketchup", CID["yellow_mustard_sauce"]: "mustard"}
ITEM_CLASSES = {CID["burger_bun"]: "burger_bun", CID["french_fries"]: "french_fries",
                CID["burger_bun_with_toppings"]: "burger_bun_with_toppings"}

# lifecycle thresholds -- from config/model.yaml `lifecycle:` block
WRAPPING_COVER_RATIO = 0.5
WRAPPING_DWELL_S = 0.4
WRAPPED_SAME_PLACE_S = 2.0
WRAPPED_SAME_PLACE_PX = 40
OUTGOING_COOLDOWN_S = 2.0
OUTGOING_WINDOW_S = 120
LOST_AFTER_S = 10.0

# exit line -- config/exit_line.json, normalized -> pixels at 1280x720
P1 = (0.694375 * 1280, 0.0877778 * 720)
P2 = (0.24 * 1280, 0.1788889 * 720)


def iou_cover(a, b):
    """Fraction of box a covered by box b."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
    return inter / area_a


def center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def point_seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def point_in_polygon(pt, poly):
    """Ray-casting point-in-polygon test. `poly` is a list of (x, y) pixels."""
    x, y = pt
    inside = False
    n = len(poly)
    x1, y1 = poly[-1]
    for i in range(n):
        x2, y2 = poly[i]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1):
            inside = not inside
        x1, y1 = x2, y2
    return inside


def load_bin_zones(path, frame_w=1280, frame_h=720):
    """config/zones.json -> [(name, [(x, y) pixels...]), ...] for zone_type == 'bin'."""
    zones = []
    try:
        with open(path) as f:
            data = json.load(f)
        for z in data:
            if z.get("zone_type") != "bin":
                continue
            poly = [(px * frame_w, py * frame_h) for px, py in z["polygon"]]
            zones.append((z.get("name", z.get("id", "bin")), poly))
    except Exception as e:
        print(f"WARNING: could not load bin zones from {path}: {e}", flush=True)
    return zones


def in_any_bin(box, bin_zones):
    """True if a detection's box center sits inside any bin-ingredient zone --
    these are supply trays, not something added to a hotdog, and are excluded
    from the lifecycle/item analysis (matches src/feed_analysis.py)."""
    c = center(box)
    for _name, poly in bin_zones:
        if point_in_polygon(c, poly):
            return True
    return False


class SystemSampler:
    """tegrastats-based system monitor, ported from src/system_monitor.py --
    samples RAM/CPU/GPU/temperature/power once a second in a background thread."""

    def __init__(self, out_path, interval_ms=1000):
        self.out_path = out_path
        self.interval_ms = interval_ms
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._proc = None

    def _run(self):
        try:
            self._proc = subprocess.Popen(
                ["tegrastats", "--interval", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
        except FileNotFoundError:
            print("WARNING: tegrastats not found in container -- no system samples", flush=True)
            return
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            sample = parse_tegrastats(line)
            sample["t"] = time.time()
            self.samples.append(sample)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._proc is not None:
            self._proc.terminate()
        if self._thread is not None:
            self._thread.join(timeout=3)
        with open(self.out_path, "w") as f:
            for s in self.samples:
                f.write(json.dumps(s) + "\n")
        return self.samples


def parse_tegrastats(line):
    ram = re.search(r"RAM (\d+)/(\d+)MB", line)
    gpu = re.search(r"GR3D_FREQ (\d+)%", line)
    cpu_cores = re.findall(r"(\d+)%@(\d+)", line)
    tj = re.search(r"tj@([\d.]+)C", line)
    vdd_in = re.search(r"VDD_IN (\d+)mW", line)
    return {
        "ram_used_mb": int(ram.group(1)) if ram else None,
        "ram_total_mb": int(ram.group(2)) if ram else None,
        "gpu_pct": int(gpu.group(1)) if gpu else None,
        "cpu_avg_pct": (sum(int(c[0]) for c in cpu_cores) / len(cpu_cores)) if cpu_cores else None,
        "tj_c": float(tj.group(1)) if tj else None,
        "power_w": (int(vdd_in.group(1)) / 1000.0) if vdd_in else None,
    }


def _clock(seconds):
    if seconds is None:
        return None
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class Hotdog:
    def __init__(self, tid, box, t):
        self.tid = tid
        self.box = box
        self.first_seen = t
        self.last_seen = t
        self.state = "on_counter"
        self.wrap_start = None
        self.wrapped_at = None
        self.outgoing_at = None
        self.packaging = None
        self.sauces = []
        self.items = []
        self.last_clamshell_box = None
        self.clamshell_still_since = None
        self.gone_since = None

    def public(self):
        return {
            "id": self.tid,
            "state": self.state,
            "lost": False,  # lost hotdogs are dropped from the dict entirely, see LOST_AFTER_S
            "packaging": self.packaging,
            "first_seen": _clock(self.first_seen),
            "wrapping_at": _clock(self.wrap_start),
            "wrapped_at": _clock(self.wrapped_at),
            "outgoing_at": _clock(self.outgoing_at),
            "sauces": list(self.sauces),
            "items": list(self.items),
        }


class LifecycleTracker(BatchMetadataOperator):
    def __init__(self, bin_zones=None, video_duration_s=None, live_path=None, publish_every_s=0.25):
        super().__init__()
        self.hotdogs = {}  # tid -> Hotdog
        self.next_hand_touch = 0.0
        self.events = []
        self.t0 = None
        self.frame_count = 0
        self.bin_zones = bin_zones or []
        self.det_counts = {c: 0 for c in CLASSES}          # every detection, incl. supply bins
        self.det_counts_analysis = {c: 0 for c in CLASSES}  # bin-ingredient detections excluded
        self.bin_excluded = 0
        # live dashboard publishing -- mirrors src/feed_analysis.py's snapshot()
        self.video_duration_s = video_duration_s
        self.live_path = live_path
        self.publish_every_s = publish_every_s
        self._last_publish_wall = 0.0
        self._wall_start = time.time()
        self._touching = False
        self.outgoing_total = 0
        self.sauce_counts = {}
        self.item_counts = {}
        self.frame_det_counts = {c: 0 for c in CLASSES}  # reset every frame -- "currently visible"

    def analysis_snapshot(self):
        states = {"on_counter": 0, "wrapping": 0, "wrapped": 0, "outgoing": 0}
        for hd in self.hotdogs.values():
            states[hd.state] += 1
        shown = sorted(self.hotdogs.values(), key=lambda h: h.last_seen, reverse=True)
        return {
            "video_time": _clock(self.now if self.t0 is not None else 0),
            "frame": self.frame_count,
            "states": states,
            "totals": {
                "hotdogs": len(self.hotdogs),
                "wrapping": states["wrapping"],
                "wrapped": states["wrapped"],
                "outgoing": self.outgoing_total,
                "sauces": sum(self.sauce_counts.values()),
                "items": sum(self.item_counts.values()),
                "line_touches": self.outgoing_total,
                "touches_without_order": 0,
                "bin_items_skipped": self.bin_excluded,
            },
            "sauces": dict(sorted(self.sauce_counts.items())),
            "items": dict(sorted(self.item_counts.items())),
            "detections_live": {k: v for k, v in self.frame_det_counts.items() if v},
            "detections_frames": {k: v for k, v in sorted(self.det_counts_analysis.items(), key=lambda kv: -kv[1]) if v},
            "detections_tracks": {"hot-dog": len(self.hotdogs)},
            "hotdogs": [hd.public() for hd in shown[:12]],
            "events": self.events[-50:],
            "exit_line": {"configured": True, "touching": self._touching},
            "rules": {
                "wrapping_cover_ratio": WRAPPING_COVER_RATIO, "wrapping_dwell_s": WRAPPING_DWELL_S,
                "wrapped_same_place_s": WRAPPED_SAME_PLACE_S, "wrapped_same_place_px": WRAPPED_SAME_PLACE_PX,
                "outgoing_cooldown_s": OUTGOING_COOLDOWN_S, "outgoing_window_s": OUTGOING_WINDOW_S,
                "lost_after_s": LOST_AFTER_S,
            },
        }

    def system_snapshot(self):
        elapsed = time.time() - self._wall_start
        video_s = self.now if self.t0 is not None else 0.0
        rate = video_s / elapsed if elapsed > 0 else 0.0
        eta = (self.video_duration_s - video_s) / rate if rate > 0 and self.video_duration_s else None
        return {
            "model": "YOLO11m-seg (DeepStream nvinfer, TensorRT FP16)",
            "decoder": "nvv4l2decoder (hardware)",
            "fps_avg": round(self.frame_count / elapsed, 2) if elapsed > 0 else None,
            "frames": self.frame_count,
            "video_s": round(video_s, 1),
            "duration_s": round(self.video_duration_s, 1) if self.video_duration_s else None,
            "eta_s": round(eta) if eta is not None else None,
            "recorder": "nvv4l2h264enc (hardware)",
            "pipeline": "DeepStream 9.1",
        }

    def publish_if_due(self):
        now = time.time()
        if now - self._last_publish_wall < self.publish_every_s:
            return
        self._last_publish_wall = now
        if not self.live_path:
            return
        payload = {"analysis": self.analysis_snapshot(), "system": self.system_snapshot()}
        tmp = self.live_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        import os
        os.replace(tmp, self.live_path)

    def log(self, kind, **kw):
        ev = {"t": round(self.now, 2), "event": kind, **kw}
        self.events.append(ev)
        print(f"[{ev['t']:6.2f}s] {kind}: {kw}")

    def handle_metadata(self, batch_meta):
        try:
            self._handle_metadata(batch_meta)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _handle_metadata(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            self.frame_count += 1
            pts_s = frame_meta.buffer_pts / 1e9
            if self.t0 is None:
                self.t0 = pts_s
            self.now = pts_s - self.t0

            # Single pass over the native iterator: pull every field we need
            # into plain Python values immediately.  pyservicemaker 0.0.1's
            # ObjectMetadata wrapper is only safe to read while the iterator
            # is advancing -- materializing it with list() (or holding a
            # reference across a second loop) segfaults the process, so
            # nothing below ever touches the wrapper object again.
            hotdog_objs, wrap_objs, hand_objs, sauce_objs, item_objs, clamshell_objs, wrapped_objs = \
                [], [], [], [], [], [], []
            self.frame_det_counts = {c: 0 for c in CLASSES}
            for o in frame_meta.object_items:
                cid = int(o.class_id)
                tid = int(o.object_id)
                r = o.rect_params
                box = (float(r.left), float(r.top), float(r.left + r.width), float(r.top + r.height))

                self.det_counts[CLASSES[cid]] += 1
                self.frame_det_counts[CLASSES[cid]] += 1

                # Bin-ingredient exclusion (config/zones.json zone_type == "bin"):
                # detections sitting inside a supply tray are stock, not
                # something added to a hotdog -- they're skipped from the
                # lifecycle/item analysis entirely but still drawn on video.
                if self.bin_zones and in_any_bin(box, self.bin_zones):
                    self.bin_excluded += 1
                    continue
                self.det_counts_analysis[CLASSES[cid]] += 1

                if cid == HOTDOG_CLASS:
                    hotdog_objs.append((tid, box))
                elif cid in WRAP_CLASSES:
                    wrap_objs.append((cid, box))
                    if cid in CLAMSHELL_CLASSES:
                        clamshell_objs.append((cid, box))
                elif cid == HAND_CLASS:
                    hand_objs.append((cid, box))
                elif cid in SAUCE_CLASSES:
                    sauce_objs.append((cid, box))
                elif cid in ITEM_CLASSES:
                    item_objs.append((cid, box))
                elif cid == WRAPPED_CLASS:
                    wrapped_objs.append((cid, box))
                elif cid == CLOSED_CLAMSHELL:
                    wrapped_objs.append((cid, box))  # closed clamshell counts as a wrapped signal too

            seen_tids = set()
            for tid, box in hotdog_objs:
                if tid == 0xFFFFFFFFFFFFFFFF:
                    continue  # untracked, skip -- identity matters for lifecycle
                seen_tids.add(tid)
                hd = self.hotdogs.get(tid)
                if hd is None:
                    hd = Hotdog(tid, box, self.now)
                    self.hotdogs[tid] = hd
                    self.log("hotdog_new", tid=tid)
                hd.box = box
                hd.last_seen = self.now
                hd.gone_since = None

                # sauces / items in the assembly area near this hotdog
                for scid, sbox in sauce_objs:
                    if iou_cover(sbox, box) > 0.05 or iou_cover(box, sbox) > 0.05:
                        name = SAUCE_CLASSES[scid]
                        if name not in hd.sauces:
                            hd.sauces.append(name)
                            self.sauce_counts[name] = self.sauce_counts.get(name, 0) + 1
                            self.log("sauce_added", tid=tid, sauce=name)
                for icid, ibox in item_objs:
                    if iou_cover(ibox, box) > 0.05 or iou_cover(box, ibox) > 0.05:
                        name = ITEM_CLASSES[icid]
                        if name not in hd.items:
                            hd.items.append(name)
                            self.item_counts[name] = self.item_counts.get(name, 0) + 1
                            self.log("item_added", tid=tid, item=name)

                # wrapping: a wrapper/clamshell covers >= WRAPPING_COVER_RATIO of the hotdog
                covered = 0.0
                covering_clamshell = None
                covering_cid = None
                for wcid, wbox in wrap_objs:
                    c = iou_cover(box, wbox)
                    if c > covered:
                        covered = c
                        covering_cid = wcid
                        if wcid in CLAMSHELL_CLASSES:
                            covering_clamshell = wbox
                if hd.state == "on_counter":
                    if covered >= WRAPPING_COVER_RATIO:
                        if hd.wrap_start is None:
                            hd.wrap_start = self.now
                        elif self.now - hd.wrap_start >= WRAPPING_DWELL_S:
                            hd.state = "wrapping"
                            hd.packaging = "clamshell" if covering_cid in CLAMSHELL_CLASSES else "wrapper"
                            self.log("wrapping_start", tid=tid, packaging=hd.packaging)
                    else:
                        hd.wrap_start = None
                if covering_clamshell is not None:
                    hd.last_clamshell_box = covering_clamshell

                # wrapped: a `wrapped`/`closed_reg_clamshell` detection over the hotdog's position
                if hd.state in ("on_counter", "wrapping"):
                    for _, wbox in wrapped_objs:
                        if iou_cover(box, wbox) > 0.2 or iou_cover(wbox, box) > 0.2:
                            hd.state = "wrapped"
                            hd.wrapped_at = self.now
                            self.log("wrapped", tid=tid, via="wrapped_class")
                            break

            # hotdogs not seen this frame: track "gone" for clamshell-close-in-place rule,
            # and drop very stale on_counter/wrapping ones
            for tid, hd in list(self.hotdogs.items()):
                if tid in seen_tids:
                    continue
                if hd.gone_since is None:
                    hd.gone_since = self.now
                gone_for = self.now - hd.gone_since

                if hd.state == "wrapping" and hd.last_clamshell_box is not None:
                    # is a clamshell still sitting in the same place?
                    still_clamshell = None
                    for o, cbox in clamshell_objs:
                        if math.hypot(*(a - b for a, b in zip(center(cbox), center(hd.last_clamshell_box)))) <= WRAPPED_SAME_PLACE_PX:
                            still_clamshell = cbox
                            break
                    if still_clamshell is not None:
                        if hd.clamshell_still_since is None:
                            hd.clamshell_still_since = self.now
                        elif self.now - hd.clamshell_still_since >= WRAPPED_SAME_PLACE_S:
                            hd.state = "wrapped"
                            hd.wrapped_at = self.now
                            self.log("wrapped", tid=tid, via="clamshell_closed_in_place")
                    else:
                        hd.clamshell_still_since = None

                if hd.state in ("on_counter", "wrapping") and gone_for > LOST_AFTER_S:
                    self.log("hotdog_lost", tid=tid)
                    del self.hotdogs[tid]

            # outgoing: a hand touching the exit line, rising edge, 2s cooldown,
            # marks the oldest wrapped-but-not-outgoing hotdog within the window
            hand_touch = any(
                point_seg_dist(center(hbox), P1, P2) < 35.0
                for _, hbox in hand_objs
            )
            self._touching = hand_touch
            if hand_touch and self.now >= self.next_hand_touch:
                candidates = [
                    hd for hd in self.hotdogs.values()
                    if hd.state == "wrapped" and hd.outgoing_at is None
                    and hd.wrapped_at is not None and self.now - hd.wrapped_at <= OUTGOING_WINDOW_S
                ]
                if candidates:
                    hd = min(candidates, key=lambda h: h.wrapped_at)
                    hd.state = "outgoing"
                    hd.outgoing_at = self.now
                    self.outgoing_total += 1
                    self.next_hand_touch = self.now + OUTGOING_COOLDOWN_S
                    self.log("outgoing", tid=hd.tid, sauces=hd.sauces, items=hd.items)

            self.publish_if_due()

    def finish(self, system_samples=None):
        counts = {"on_counter": 0, "wrapping": 0, "wrapped": 0, "outgoing": 0}
        for hd in self.hotdogs.values():
            counts[hd.state] += 1
        total_sauces = sum(len(hd.sauces) for hd in self.hotdogs.values())
        total_items = sum(len(hd.items) for hd in self.hotdogs.values())
        summary = {
            "frames": self.frame_count,
            "wall_duration_s": round(self.now, 2) if self.now else 0,
            "hotdog_state_counts": counts,
            "total_hotdog_tracks": len(self.hotdogs),
            "sauces_added": total_sauces,
            "items_added": total_items,
            "detection_counts_raw": self.det_counts,
            "detection_counts_analysis": self.det_counts_analysis,
            "bin_excluded_detections": self.bin_excluded,
            "bin_zones_loaded": len(self.bin_zones),
            "events": self.events,
        }
        if system_samples:
            vals = lambda k: [s[k] for s in system_samples if s.get(k) is not None]
            summary["system"] = {
                "samples": len(system_samples),
                "cpu_avg_pct_mean": round(sum(vals("cpu_avg_pct")) / len(vals("cpu_avg_pct")), 1) if vals("cpu_avg_pct") else None,
                "gpu_pct_mean": round(sum(vals("gpu_pct")) / len(vals("gpu_pct")), 1) if vals("gpu_pct") else None,
                "gpu_pct_max": max(vals("gpu_pct")) if vals("gpu_pct") else None,
                "ram_used_mb_mean": round(sum(vals("ram_used_mb")) / len(vals("ram_used_mb")), 0) if vals("ram_used_mb") else None,
                "tj_c_mean": round(sum(vals("tj_c")) / len(vals("tj_c")), 1) if vals("tj_c") else None,
                "tj_c_max": max(vals("tj_c")) if vals("tj_c") else None,
                "power_w_mean": round(sum(vals("power_w")) / len(vals("power_w")), 1) if vals("power_w") else None,
            }
        with open(EVENTS_OUT, "w") as f:
            for ev in self.events:
                f.write(json.dumps(ev) + "\n")
        with open(f"{WORK}/deepstream_test/summary_{RUN_TAG}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n=== SUMMARY ===")
        print(json.dumps(summary["hotdog_state_counts"], indent=2))
        print("detections (analysis, bin-excluded):", {k: v for k, v in self.det_counts_analysis.items() if v})
        print("bin-excluded detections:", self.bin_excluded)
        print(f"frames={self.frame_count} wall={summary['wall_duration_s']}s events={len(self.events)}")
        if system_samples:
            print("system:", json.dumps(summary["system"]))
        return summary


def main():
    bin_zones = load_bin_zones(ZONES_PATH)
    print(f"loaded {len(bin_zones)} bin zones from {ZONES_PATH}", flush=True)
    tracker_op = LifecycleTracker(bin_zones=bin_zones, video_duration_s=VIDEO_DURATION_S, live_path=LIVE_PATH)
    sampler = SystemSampler(SYSTEM_OUT).start()

    pipeline = (
        Pipeline("oa-deepstream-1min")
        .add("filesrc", "src", {"location": CLIP})
        .add("h264parse", "parser")
        .add("nvv4l2decoder", "decoder")
        .add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720,
                                      "batched-push-timeout": 40000})
        .add("nvinfer", "infer", {"config-file-path": CONFIG})
        .add("nvtracker", "tracker", {
            "tracker-width": 640, "tracker-height": 384,
            "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
            "ll-config-file": TRACKER_CONFIG,
        })
        .add("nvdsosd", "osd", {"display-mask": 1, "display-bbox": 1, "display-text": 1})
        .add("nvvideoconvert", "conv")
        .add("nvv4l2h264enc", "enc", {"bitrate": 4000000})
        .add("h264parse", "outparse")
        .add("matroskamux", "mux2")
        .add("filesink", "sink", {"location": OUT})
    )

    pipeline.link("src", "parser", "decoder")
    pipeline.link(("decoder", "mux"), ("", "sink_%u"))
    pipeline.link("mux", "infer", "tracker", "osd", "conv", "enc", "outparse", "mux2", "sink")
    # Attach after "tracker", not "infer" -- object_id (the tracking ID the
    # lifecycle state machine keys hotdogs by) is only assigned once nvtracker
    # has run; a probe on "infer" only ever sees the untracked sentinel.
    pipeline.attach("tracker", Probe("lifecycle", tracker_op))

    start = time.time()
    pipeline.start().wait()
    print(f"pipeline finished in {time.time() - start:.1f}s wall")
    samples = sampler.stop()
    tracker_op.finish(system_samples=samples)


if __name__ == "__main__":
    main()
