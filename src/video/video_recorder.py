"""Record the annotated detection feed to a video file.

Enabled with RECORD_VIDEO=<path>; off by default, so nothing changes unless it
is asked for.  Records exactly the frame the dashboard displays -- boxes,
masks, zones and trails -- and never alters what the pipeline detects.

Frames go to a background writer thread through a bounded queue, so encoding
never runs on the pinned main loop.  When the queue is full the producer blocks
rather than dropping, so the output contains every processed frame, in order.

Encoding uses the Jetson's hardware H.264 encoder (nvv4l2h264enc) when
GStreamer and the NVIDIA plugins are available, so the file is H.264 straight
away with no conversion afterwards.  Otherwise it falls back to OpenCV's mp4v.

    RECORD_VIDEO=1             auto-name in output/recordings/<source>_<time>.mp4
    RECORD_VIDEO=out/run.mp4   or an explicit path (fragmented MP4, so a
                               truncated file still plays to the last fragment)
    RECORD_FPS=30              playback rate (default: pipeline fps, else 30)
    RECORD_ENCODER=nvenc       nvenc (hardware H.264) or opencv
    RECORD_BITRATE=4000000     hardware encoder bitrate, bits per second
    RECORD_FOURCC=mp4v         codec tag for the OpenCV fallback
    RECORD_HUD=1               burn source-video timestamp + frame number
"""
import logging
import os
import queue
import threading
import time
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

_STOP = object()
_ACTIVE = None

# Same place the pipeline keeps its other artefacts (output/failures, ...).
RECORDINGS_DIR = Path("output/recordings")

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    _NVENC_OK = all(
        Gst.ElementFactory.find(e) for e in ("appsrc", "nvvidconv", "nvv4l2h264enc", "h264parse", "mp4mux")
    )
except Exception:  # gi not importable, or no NVIDIA GStreamer plugins
    Gst = None
    _NVENC_OK = False


def hardware_encoder_available() -> bool:
    return _NVENC_OK


def default_recording_path(source=None):
    """output/recordings/<source name>_<YYYYmmdd_HHMMSS>.mp4"""
    s = str(source) if source is not None else ""
    if s.isdigit():
        stem = f"camera{s}"
    elif "://" in s:
        stem = "stream"
    elif s:
        stem = Path(s).stem
    else:
        stem = "recording"
    return RECORDINGS_DIR / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"


class HardwareH264Writer:
    """BGR frames -> appsrc -> nvvidconv -> nvv4l2h264enc -> h264parse -> fragmented MP4."""

    def __init__(self, path, size, fps, bitrate=4_000_000):
        if not _NVENC_OK:
            raise RuntimeError("hardware H.264 encoder not available")
        w, h = size
        rate = max(1, int(round(fps)))
        self._duration = Gst.SECOND // rate
        self._frames = 0
        location = str(path).replace("\\", "\\\\").replace('"', '\\"')
        self._pipeline = Gst.parse_launch(
            "appsrc name=src is-live=false block=true format=time ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            f"nvv4l2h264enc bitrate={int(bitrate)} iframeinterval={rate * 2} ! h264parse ! "
            f'mp4mux fragment-duration=1000 ! filesink location="{location}"'
        )
        self._src = self._pipeline.get_by_name("src")
        self._src.set_property(
            "caps", Gst.Caps.from_string(f"video/x-raw,format=BGRx,width={w},height={h},framerate={rate}/1")
        )
        self._bus = self._pipeline.get_bus()
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("hardware H.264 encoder pipeline did not start")

    def _raise_on_error(self):
        msg = self._bus.pop_filtered(Gst.MessageType.ERROR)
        if msg is not None:
            err, debug = msg.parse_error()
            raise RuntimeError(f"{err.message} ({debug})")

    def write(self, frame_bgr):
        self._raise_on_error()
        bgrx = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
        buf = Gst.Buffer.new_wrapped(bgrx.tobytes())
        buf.pts = self._frames * self._duration
        buf.duration = self._duration
        self._frames += 1
        if self._src.emit("push-buffer", buf) != Gst.FlowReturn.OK:
            self._raise_on_error()
            raise RuntimeError("the hardware encoder stopped accepting frames")

    def release(self):
        self._src.emit("end-of-stream")
        # Wait for EOS so mp4mux writes its final fragment and the file is complete.
        self._bus.timed_pop_filtered(20 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)
        self._pipeline.set_state(Gst.State.NULL)


class VideoRecorder:
    def __init__(self, path, fps=30.0, fourcc="mp4v", hud=True, queue_size=64,
                 encoder="nvenc", bitrate=4_000_000):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps) if fps else 30.0
        self.fourcc = fourcc
        self.hud = hud
        self.encoder_wanted = str(encoder or "nvenc").lower()
        self.bitrate = int(bitrate)
        self.encoder = None
        self.frames_submitted = 0
        self.frames_written = 0
        self.error = None
        self._q = queue.Queue(maxsize=queue_size)
        self._writer = None
        self._size = None
        self._thread = threading.Thread(target=self._run, name="video-recorder", daemon=True)
        self._thread.start()

    def write(self, frame):
        if frame is None:
            return
        idx = self.frames_submitted
        self.frames_submitted += 1
        # Copy: the caller keeps drawing on / reusing its buffer after handoff.
        self._q.put((idx, frame.copy()))

    def close(self):
        self._q.put(_STOP)
        self._thread.join()
        return self.frames_written

    def _open(self, shape):
        h, w = shape[:2]
        self._size = (w, h)
        if self.encoder_wanted in ("nvenc", "hardware", "h264") and _NVENC_OK:
            try:
                self._writer = HardwareH264Writer(self.path, self._size, self.fps, self.bitrate)
                self.encoder = "nvv4l2h264enc"
            except RuntimeError as exc:
                logger.warning("Hardware encoder unavailable (%s); recording with OpenCV %s", exc, self.fourcc)
        if self._writer is None:
            self._writer = cv2.VideoWriter(
                str(self.path), cv2.VideoWriter_fourcc(*self.fourcc), self.fps, self._size
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"cv2.VideoWriter could not open {self.path} with fourcc={self.fourcc!r}"
                )
            self.encoder = f"opencv {self.fourcc}"
        logger.info("Recording %dx%d @ %.2f fps -> %s (%s)", w, h, self.fps, self.path, self.encoder)

    def _stamp(self, frame, idx):
        # Every file frame is processed, so idx / fps is the position in the
        # source video -- lets a reviewer seek the original to the same moment.
        secs = int(idx / self.fps)
        hh, rem = divmod(secs, 3600)
        mm, ss = divmod(rem, 60)
        text = f"{hh:02d}:{mm:02d}:{ss:02d}   frame {idx + 1:,}"
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, 0.55
        (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
        h = frame.shape[0]
        cv2.rectangle(frame, (8, h - th - 20), (8 + tw + 14, h - 8), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, h - 14), font, scale, (235, 235, 235), 1, cv2.LINE_AA)

    def _run(self):
        while True:
            item = self._q.get()
            if item is _STOP:
                break
            if self.error is not None:
                continue  # keep draining so the producer can never block forever
            idx, frame = item
            try:
                if self._writer is None:
                    self._open(frame.shape)
                if (frame.shape[1], frame.shape[0]) != self._size:
                    frame = cv2.resize(frame, self._size)
                if self.hud:
                    self._stamp(frame, idx)
                self._writer.write(frame)
                self.frames_written += 1
            except Exception as exc:  # recording must never take the pipeline down
                self.error = exc
                logger.error("Video recording stopped after %d frames: %s",
                             self.frames_written, exc)
        if self._writer is not None:
            self._writer.release()


def recorder_from_env(fps=None, source=None):
    """Return the process-wide recorder if RECORD_VIDEO is set, else None.

    Idempotent: a second call returns the same instance rather than opening a
    new writer over the same file.
    """
    global _ACTIVE
    path = os.environ.get("RECORD_VIDEO")
    if not path:
        return None
    if path.strip().lower() in ("1", "true", "yes", "on"):
        path = default_recording_path(source)
    if _ACTIVE is None:
        _ACTIVE = VideoRecorder(
            path,
            fps=float(os.environ.get("RECORD_FPS") or fps or 30),
            fourcc=os.environ.get("RECORD_FOURCC", "mp4v"),
            hud=os.environ.get("RECORD_HUD", "1").lower() not in ("0", "false", "no"),
            encoder=os.environ.get("RECORD_ENCODER", "nvenc"),
            bitrate=int(os.environ.get("RECORD_BITRATE") or 4_000_000),
        )
    return _ACTIVE
