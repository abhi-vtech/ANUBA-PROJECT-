"""A KDS screen image for the dashboard.

kds-ocr reads the KDS feed in its own process, so this process never sees
those pixels -- the dashboard's KDS panel would have no picture at all.  This
opens the same source a second time, cheaply, purely to show it.

It is a VIEW, nothing more: no frame here is parsed, and no decision depends
on it.  The old panel showed a frame annotated by `src/kds/overlay.py`; that
went with the in-process reader, so this is the plain screen, and the ticket
state drawn beside it comes from the reader's own emissions instead.

For a RECORDING the preview is kept level with the production video rather
than run at its own speed, or the panel would show a different minute from the
food on the main feed.  Frames are pulled forward sequentially to the target
time (never seeked per frame -- seeking an MKV repeatedly is far more
expensive than decoding through it).
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

#: Deliberately slow.  This is a reference picture beside the ticket list, and
#: every frame costs a decode on a box whose detector wants the CPU.
DEFAULT_FPS = 2.0

#: A recorded preview may run this far behind the production clock before it
#: gives up catching up frame-by-frame and seeks instead.  Tolerates the
#: normal case (we are a few frames behind) while still recovering if the
#: production loop jumps.
RESYNC_AHEAD_S = 5.0


class KdsPreview:
    """Decodes the KDS source at a low rate and hands frames to a callback."""

    def __init__(
        self,
        source: str,
        on_frame: Callable,
        fps: float = DEFAULT_FPS,
        live: bool = False,
        start_at_s: float = 0.0,
        offset_s: float = 0.0,
    ):
        self.source = source
        self.on_frame = on_frame
        self.interval = 1.0 / max(0.1, fps)
        self.live = live
        self.start_at_s = max(0.0, start_at_s)
        #: Added to the production media time to get the KDS media time.  The
        #: two recordings do not start at the same instant.
        self.offset_s = offset_s
        self._master_t: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.frames_shown = 0
        #: The most recent raw frame, for the composed recording. Kept as well
        #: as handed to the callback, because the dashboard JPEG-encodes its
        #: copy and the recorder needs the pixels.
        self.latest = None

    def set_master_time(self, t: float) -> None:
        self._master_t = float(t)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="kds-preview",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # -- internals -------------------------------------------------------

    def _run(self) -> None:
        try:
            import cv2
        except Exception:
            logger.warning("no cv2; KDS preview disabled")
            return
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.warning("KDS preview could not open its source; the panel "
                           "will show ticket state without a picture")
            return
        if not self.live and self.start_at_s:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start_at_s * 1000.0)
        logger.info("KDS preview running at %.1f fps", 1.0 / self.interval)
        try:
            while not self._stop.is_set():
                frame = self._next(cap, cv2)
                if frame is None:
                    break
                self.latest = frame
                try:
                    self.on_frame(frame)
                    self.frames_shown += 1
                except Exception:
                    logger.debug("KDS preview callback failed", exc_info=True)
                self._stop.wait(self.interval)
        finally:
            cap.release()
            logger.info("KDS preview stopped after %d frame(s)", self.frames_shown)

    def _next(self, cap, cv2):
        """The frame the production feed is currently level with."""
        if self.live or self._master_t is None:
            ok, frame = cap.read()
            return frame if ok else None

        target_ms = (self._master_t + self.offset_s) * 1000.0
        pos = cap.get(cv2.CAP_PROP_POS_MSEC)
        # Too far ahead of us to walk to -- jump.  Happens when the production
        # loop is seeked, or after the preview has been starved.
        if target_ms - pos > RESYNC_AHEAD_S * 1000.0:
            cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)
        last = None
        # Walk forward to the target. Bounded so a bad clock cannot spin here.
        for _ in range(600):
            if self._stop.is_set():
                return last
            ok, frame = cap.read()
            if not ok:
                return last
            last = frame
            if cap.get(cv2.CAP_PROP_POS_MSEC) >= target_ms:
                break
        return last
