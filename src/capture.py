import queue
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

_LOOP_MARKER = object()


class VideoCaptureThread:
    def __init__(
        self, source, target_width=None, target_height=None, fps=None, realtime=False
    ):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self._frame: Optional[Tuple[np.ndarray, float]] = None
        self._is_file_source = isinstance(source, str) and Path(source).is_file()
        self.realtime = realtime
        # Files normally use a blocking queue so every frame is processed. In
        # realtime mode they use latest-wins (drop frames) to keep real-time
        # wall-clock pace, the same way a live camera is handled.
        self._use_queue = self._is_file_source and not realtime
        self._frame_queue = queue.Queue(maxsize=30) if self._use_queue else None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.target_width = target_width
        self.target_height = target_height
        effective_fps = fps if fps is not None else self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_interval = (
            1.0 / effective_fps if effective_fps and effective_fps > 0 else 0.0
        )
        self._looped = False

    def _put_file_item(self, item) -> None:
        while self._running and self._frame_queue is not None:
            try:
                self._frame_queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            start = time.time()
            ret, frame = self.cap.read()
            if ret:
                current_time = (
                    self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if self._is_file_source
                    else time.monotonic()
                )
                if self.target_width and self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                item = (frame, current_time)
                if self._use_queue:
                    self._put_file_item(item)
                else:
                    self._frame = item  # latest-wins (live or realtime file)
                if self._frame_interval:
                    elapsed = time.time() - start
                    sleep_time = self._frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
            else:
                if self._is_file_source:
                    if self._use_queue:
                        self._put_file_item(_LOOP_MARKER)
                    else:
                        self._looped = True  # realtime file: signal loop boundary
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.01)

    def get_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        if self._frame_queue is not None:
            while self._running:
                try:
                    item = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    return None
                if item is _LOOP_MARKER:
                    self._looped = True
                    return None  # signal the main loop; it will call consume_loop() next
                return item
        return self._frame

    def consume_loop(self) -> bool:
        """Return True once when video has finished and restarted, then reset."""
        if self._looped:
            self._looped = False
            return True
        return False

    def change_source(self, new_source):
        self.cap.release()
        self.source = new_source
        self.cap = cv2.VideoCapture(new_source)
        self._is_file_source = isinstance(new_source, str) and Path(new_source).is_file()
        self._looped = False
        effective_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._frame_interval = (
            1.0 / effective_fps if effective_fps and effective_fps > 0 else 0.0
        )
        if self._frame_queue is not None:
            import queue as q_lib
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except q_lib.Empty:
                    break

    def release(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        self.cap.release()
