"""Hardware-decoded video capture (NVIDIA nvv4l2decoder through GStreamer).

A drop-in for src.video.capture.VideoCaptureThread.  The main loop only uses
start(), get_frame(), consume_loop(), release(), _is_file_source and cap.get()
for the frame count, position and fps, and this class provides exactly those.

Decoding runs on the Jetson's hardware video decoder -- the element DeepStream
is built on -- and nvvidconv scales to the pipeline size, so the CPU only copies
the finished frame.  Measured on the camA recording: 379 fps decode + scale, but
handing a 1280x720 frame to Python costs ~7.6 ms, about the same as OpenCV's CPU
decode.  This frees a CPU core; it does not raise FPS.

Files go through a bounded blocking queue so every frame is processed, as with
the OpenCV capture; RTSP keeps only the newest frame.  `open_capture` falls back
to OpenCV for anything without a hardware chain (camera indexes, codecs other
than H.264/H.265, containers other than MKV/MP4/MOV).
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_LOOP_MARKER = object()
_DEMUXERS = {
    ".mkv": "matroskademux",
    ".webm": "matroskademux",
    ".mp4": "qtdemux",
    ".mov": "qtdemux",
    ".m4v": "qtdemux",
}
_H264_TAGS = {"h264", "avc1", "avc3", "x264"}
_H265_TAGS = {"hevc", "hvc1", "hev1", "h265", "x265"}

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
    _GST_OK = all(Gst.ElementFactory.find(e) for e in ("nvv4l2decoder", "nvvidconv", "appsink"))
except Exception:  # gi not importable, or no NVIDIA GStreamer plugins
    Gst = None
    _GST_OK = False


def gstreamer_available() -> bool:
    """True when the hardware decode chain can be built on this machine."""
    return _GST_OK


def _probe_file(path: str) -> Tuple[Optional[str], float, int]:
    """(codec family, fps, frame count) read through OpenCV's demuxer."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None, 0.0, 0
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
        tag = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip().lower()
        codec = "h264" if tag in _H264_TAGS else "h265" if tag in _H265_TAGS else None
        return codec, float(cap.get(cv2.CAP_PROP_FPS) or 0.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


class _CapShim:
    """The cv2.VideoCapture.get() properties the main loop reads.

    Position and count follow OpenCV's meaning for a file -- derived from the
    timestamp and the container duration -- so `count - position` still
    measures the time left in a variable-frame-rate recording.
    """

    def __init__(self, owner: "GstVideoCaptureThread"):
        self._owner = owner

    def get(self, prop) -> float:
        o = self._owner
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(o.frame_count)
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(round(o.last_pts_s * o.source_fps))
        if prop == cv2.CAP_PROP_FPS:
            return float(o.source_fps)
        if prop == cv2.CAP_PROP_POS_MSEC:
            return o.last_pts_s * 1000.0
        return 0.0

    def release(self) -> None:
        pass


class GstVideoCaptureThread:
    """Decode a file or RTSP stream on the hardware decoder into BGR frames."""

    backend = "nvv4l2decoder"

    def __init__(self, source, target_width=None, target_height=None, fps=None, realtime=False, start_at_s=0.0):
        if not _GST_OK:
            raise RuntimeError("GStreamer with the NVIDIA plugins is not importable")
        self.source = str(source)
        self._is_file_source = Path(self.source).is_file()
        self._is_rtsp = self.source.lower().startswith(("rtsp://", "rtsps://"))
        if not (self._is_file_source or self._is_rtsp):
            raise ValueError("no hardware decode chain for this source")

        self.realtime = realtime
        self._use_queue = self._is_file_source and not realtime
        self._frame_queue = queue.Queue(maxsize=30) if self._use_queue else None
        self._frame: Optional[Tuple[np.ndarray, float]] = None
        self._looped = False
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.error: Optional[str] = None
        self.frames_delivered = 0
        self.last_pts_s = 0.0
        # Seconds into the file to begin at.  Kept, not consumed: `_end_of_stream`
        # rewinds here rather than to 0, so a looping run never replays the part
        # of the recording the caller asked to skip.
        self.start_at_s = float(start_at_s or 0.0) if not self._is_rtsp else 0.0
        self._start_pts = int(self.start_at_s * Gst.SECOND) if self.start_at_s > 0 else 0
        self.last_pts_s = self.start_at_s

        if self._is_file_source:
            codec, probed_fps, count = _probe_file(self.source)
            demux = _DEMUXERS.get(Path(self.source).suffix.lower())
            if codec is None or demux is None:
                raise ValueError(
                    f"no hardware chain for a {Path(self.source).suffix or 'bare'} file with codec {codec}"
                )
        else:
            codec = os.environ.get("RTSP_CODEC", "h264").strip().lower()
            probed_fps, count, demux = 0.0, 0, None
            if codec not in ("h264", "h265"):
                raise ValueError(f"RTSP_CODEC must be h264 or h265, not {codec!r}")

        self.codec = codec
        self.source_fps = float(probed_fps or fps or 30.0)
        self.frame_count = count
        effective_fps = fps if fps is not None else probed_fps
        self._frame_interval = 1.0 / effective_fps if effective_fps and effective_fps > 0 else 0.0
        self.cap = _CapShim(self)

        parse = "h264parse" if codec == "h264" else "h265parse"
        size = f",width={int(target_width)},height={int(target_height)}" if target_width and target_height else ""
        if self._is_file_source:
            location = self.source.replace("\\", "\\\\").replace('"', '\\"')
            head = f'filesrc location="{location}" ! {demux} ! {parse}'
            sink_opts = "drop=false max-buffers=8"
        else:
            depay = "rtph264depay" if codec == "h264" else "rtph265depay"
            head = f'rtspsrc location="{self.source}" latency=200 ! {depay} ! {parse}'
            sink_opts = "drop=true max-buffers=1"
        self.pipeline_desc = (
            f"{head} ! nvv4l2decoder ! nvvidconv ! video/x-raw,format=BGRx{size} ! "
            f"appsink name=sink sync=false emit-signals=false {sink_opts}"
        )
        self._pipeline = Gst.parse_launch(self.pipeline_desc)
        self._sink = self._pipeline.get_by_name("sink")
        self._bus = self._pipeline.get_bus()

        # Preroll now, so a chain that cannot link or decode fails here, where
        # open_capture() can still fall back to OpenCV, not inside the main loop.
        self._pipeline.set_state(Gst.State.PAUSED)
        ret, _, _ = self._pipeline.get_state(15 * Gst.SECOND)
        if ret == Gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"hardware decode pipeline did not start: {self.pipeline_desc}")
        # NOT the place to seek: seek_simple() blocks until the new position is
        # prerolled, and a PAUSED pipeline with nobody pulling samples never
        # gets there.  The seek is the reader thread's first action instead.

    def _seek_to_start(self) -> bool:
        """Jump to ``start_at_s``.  A no-op at 0 or on RTSP.

        SNAP_BEFORE lands on the keyframe at or BEFORE the target, never after:
        landing after it would silently skip footage the caller asked for, and
        with keyframes tens of seconds apart in these recordings that is enough
        to put the two feeds on different minutes.  The frames between the
        keyframe and the target are then dropped in :meth:`_run`.

        ACCURATE would land exactly and is deliberately not used: on this
        decoder it never returns on an hour-long file.
        """
        if self.start_at_s <= 0:
            return False
        ok = self._pipeline.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT | Gst.SeekFlags.SNAP_BEFORE,
            int(self.start_at_s * Gst.SECOND),
        )
        if not ok:
            logger.warning(
                "Hardware decode could not seek %s to %.1fs; starting from 0",
                self.source,
                self.start_at_s,
            )
        return ok

    # ── Thread ───────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError(f"hardware decode pipeline did not play: {self.pipeline_desc}")
        self._thread = threading.Thread(target=self._run, name="gst-capture", daemon=True)
        self._thread.start()
        logger.info("Hardware video decode: %s", self.pipeline_desc)

    def _put_file_item(self, item) -> None:
        while self._running and self._frame_queue is not None:
            try:
                self._frame_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _to_frame(self, sample) -> Optional[np.ndarray]:
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            data = np.frombuffer(info.data, np.uint8)
            stride = data.size // h  # bytes per row, including any padding
            return data.reshape(h, stride)[:, : w * 4].reshape(h, w, 4)[:, :, :3].copy()
        finally:
            buf.unmap(info)

    def _end_of_stream(self) -> None:
        if self._is_rtsp:
            logger.warning("RTSP stream ended or failed (%s); reconnecting in 2 s", self.error)
            self._pipeline.set_state(Gst.State.NULL)
            time.sleep(2.0)
            self._pipeline.set_state(Gst.State.PLAYING)
            return
        if self._use_queue:
            self._put_file_item(_LOOP_MARKER)
        else:
            self._looped = True  # realtime file: signal the loop boundary
        # Rewind, as the OpenCV capture does, for a caller that keeps reading.
        self.last_pts_s = self.start_at_s
        if not self._pipeline.seek_simple(
            Gst.Format.TIME,
            Gst.SeekFlags.FLUSH | Gst.SeekFlags.KEY_UNIT | Gst.SeekFlags.SNAP_BEFORE,
            int(self.start_at_s * Gst.SECOND),
        ):
            time.sleep(0.1)

    def _run(self):
        timeout = Gst.SECOND // 5
        # The seek belongs here, as this thread's first action, and nowhere
        # else.  seek_simple() blocks until the new position is prerolled, so
        # whichever thread calls it must not be a thread the pipeline needs in
        # order to drain:
        #   * from __init__ (PAUSED, no reader yet) nothing pulls samples, so
        #     the preroll never completes;
        #   * from start() after PLAYING it deadlocks on a race -- this reader
        #     fills the 30-frame queue that the main loop is not consuming yet,
        #     stops pulling, and the seek waits on a pull that never comes.
        # Here nothing has been queued and nothing has been pulled, so neither
        # can happen.
        self._seek_to_start()
        while self._running:
            sample = self._sink.emit("try-pull-sample", timeout)
            if sample is None:
                msg = self._bus.pop_filtered(Gst.MessageType.ERROR)
                if msg is not None:
                    err, debug = msg.parse_error()
                    self.error = f"{err.message} ({debug})"
                    logger.error("Hardware video decode error: %s", self.error)
                    self._end_of_stream()
                elif self._sink.get_property("eos"):
                    self._end_of_stream()
                continue
            # Finish the seek: SNAP_BEFORE lands on the keyframe at or before
            # the requested offset, so discard what follows it until the media
            # clock reaches the offset itself.  Dropped before _to_frame(), so
            # nothing here pays for the BGR copy.
            if self._start_pts:
                buf_pts = sample.get_buffer().pts
                if buf_pts != Gst.CLOCK_TIME_NONE and buf_pts < self._start_pts:
                    continue
            started = time.time()
            frame = self._to_frame(sample)
            if frame is None:
                continue
            if self._is_file_source:
                pts = sample.get_buffer().pts
                current_time = pts / Gst.SECOND if pts != Gst.CLOCK_TIME_NONE else self.last_pts_s
                self.last_pts_s = current_time
            else:
                current_time = time.monotonic()
            self.frames_delivered += 1
            item = (frame, current_time)
            if self._use_queue:
                self._put_file_item(item)
            else:
                self._frame = item  # latest wins (live or realtime file)
            if self._frame_interval:
                sleep_time = self._frame_interval - (time.time() - started)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    # ── Same interface as VideoCaptureThread ─────────────────────────────────

    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        if self._frame_queue is not None:
            while self._running:
                try:
                    item = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    return None
                if item is _LOOP_MARKER:
                    self._looped = True
                    return None  # the main loop calls consume_loop() next
                return item
        return self._frame

    def consume_loop(self) -> bool:
        """Return True once when the video has finished and restarted."""
        if self._looped:
            self._looped = False
            return True
        return False

    def release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self._pipeline.set_state(Gst.State.NULL)


def open_capture(source, target_width=None, target_height=None, fps=None, realtime=False, ingest="gstreamer", start_at_s=0.0):
    """Hardware decode when possible, otherwise the OpenCV capture.

    The returned object has a `backend` attribute ("nvv4l2decoder" or
    "opencv") so the dashboard can show which one is running.
    """
    from src.video.capture import VideoCaptureThread

    if str(ingest or "").strip().lower() in ("gstreamer", "gst", "nvdec", "hardware", "deepstream", "auto"):
        if not _GST_OK:
            logger.warning("Hardware video decode unavailable (GStreamer / NVIDIA plugins not importable); using OpenCV")
        else:
            try:
                return GstVideoCaptureThread(
                    source, target_width, target_height, fps, realtime, start_at_s=start_at_s
                )
            except (ValueError, RuntimeError) as exc:
                logger.warning("Hardware video decode not used for %s: %s; using OpenCV", source, exc)
    capture = VideoCaptureThread(
        source,
        target_width=target_width,
        target_height=target_height,
        fps=fps,
        realtime=realtime,
        start_at_s=start_at_s,
    )
    capture.backend = "opencv"
    return capture
