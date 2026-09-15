"""Run DeepStream over a video and write one JSON line of detections per frame.

This is the inference half of running the KDS pipeline on DeepStream.  It runs
INSIDE the DeepStream container (DeepStream is not installed natively on this
Jetson), and everything downstream -- the KDS screen reader, the order state
machines, the dashboard -- stays on the host, which is where its dependencies
live.  The two halves meet on this file: a JSONL stream keyed by media
timestamp, which the host replays in place of running YOLO itself.

Output, one line per frame:
    {"t": <pts seconds>, "f": <frame index>, "o": [[cls, conf, x1, y1, x2, y2, tid], ...]}

`t` is the media timestamp, NOT a frame index, so the host can line detections
up with its own decode of the same file without depending on either side
seeing exactly the same frames.

Two constraints are load-bearing and cost a debugging session each:
  * the probe attaches AFTER nvtracker -- object_id is only assigned there, and
    a probe on nvinfer sees only the untracked sentinel;
  * pyservicemaker's ObjectMetadata wrapper is valid only while its iterator is
    advancing, so every field is copied out in one pass and the wrapper is
    never re-read.  list()-ing the items segfaults the process.
"""
import json
import os
import sys
import time

from pyservicemaker import Pipeline, Probe, BatchMetadataOperator

WORK = os.environ.get("DS_WORK", "/work")
VIDEO = os.environ["DS_VIDEO"]
OUT = os.environ["DS_OUT"]
CONFIG = os.environ.get("DS_CONFIG", f"{WORK}/deepstream_test/config_infer.txt")
TRACKER_CFG = os.environ.get(
    "DS_TRACKER", "/opt/nvidia/deepstream/deepstream/samples/configs/deepstream-app/config_tracker_IOU.yml"
)
WIDTH = int(os.environ.get("DS_WIDTH", "1280"))
HEIGHT = int(os.environ.get("DS_HEIGHT", "720"))
# Only emit frames inside this media-time window; everything before it is
# decoded and inferred but thrown away.
T_FROM = float(os.environ.get("DS_FROM", "0"))
T_TO = float(os.environ.get("DS_TO", "0")) or float("inf")

UNTRACKED = 0xFFFFFFFFFFFFFFFF


class _Dumper(BatchMetadataOperator):
    def __init__(self, handle):
        super().__init__()
        self.h = handle
        self.frames = 0
        self.emitted = 0
        self.objects = 0
        self.t0 = time.time()
        self.last_pts = 0.0

    def handle_metadata(self, batch_meta):
        try:
            self._handle(batch_meta)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _handle(self, batch_meta):
        for frame_meta in batch_meta.frame_items:
            self.frames += 1
            pts = frame_meta.buffer_pts / 1e9
            self.last_pts = pts
            if pts < T_FROM or pts > T_TO:
                continue
            objs = []
            # ONE pass; every field copied to plain Python immediately.
            for o in frame_meta.object_items:
                r = o.rect_params
                tid = int(o.object_id)
                objs.append([
                    int(o.class_id),
                    round(float(o.confidence), 4),
                    int(r.left), int(r.top),
                    int(r.left + r.width), int(r.top + r.height),
                    -1 if tid == UNTRACKED else tid,
                ])
            self.objects += len(objs)
            self.emitted += 1
            self.h.write(json.dumps({"t": round(pts, 4), "f": self.frames, "o": objs}) + "\n")
            if self.emitted % 300 == 0:
                self.h.flush()
                el = time.time() - self.t0
                print("  %6d frames  %7.1fs media  %5.1f fps  %d objects"
                      % (self.frames, pts, self.frames / max(el, 1e-6), self.objects), flush=True)


def main():
    ext = os.path.splitext(VIDEO)[1].lower()
    demux = "matroskademux" if ext in (".mkv", ".webm") else "qtdemux"
    with open(OUT, "w") as handle:
        op = _Dumper(handle)
        p = (
            Pipeline("oa-ds-detect")
            .add("filesrc", "src", {"location": VIDEO})
            .add(demux, "demux")
            .add("h264parse", "parse")
            .add("nvv4l2decoder", "dec")
            .add("nvstreammux", "mux", {"batch-size": 1, "width": WIDTH, "height": HEIGHT,
                                        "batched-push-timeout": 40000, "live-source": 0})
            .add("nvinfer", "infer", {"config-file-path": CONFIG})
            .add("nvtracker", "tracker", {
                "ll-lib-file": "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so",
                "ll-config-file": TRACKER_CFG,
                "tracker-width": 640, "tracker-height": 384,
            })
            .add("fakesink", "sink", {"sync": 0})
        )
        p.link("src", "demux")
        p.link(("demux", "parse"), ("video_%u", ""))
        p.link("parse", "dec")
        p.link(("dec", "mux"), ("", "sink_%u"))
        p.link("mux", "infer", "tracker", "sink")
        p.attach("tracker", Probe("dump", op))
        print("running DeepStream over %s -> %s" % (VIDEO, OUT), flush=True)
        p.start().wait()
        handle.flush()
    el = time.time() - op.t0
    print("done: %d frames decoded, %d emitted, %d objects, %.0fs wall (%.1f fps)"
          % (op.frames, op.emitted, op.objects, el, op.frames / max(el, 1e-6)), flush=True)


if __name__ == "__main__":
    sys.exit(main())
