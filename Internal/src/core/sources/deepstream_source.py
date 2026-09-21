"""DeepStream source: a probe that pushes `Frame`s into a queue.

DeepStream inverts control -- GStreamer owns the loop and calls the probe -- so
this adapter buffers frames and `__iter__` drains them.  From the analysis
side the inversion is invisible: it still reads an iterator of `Frame`.

Two constraints from the current run_deepstream.py are preserved here, because
both are real and both cost a debugging session to find:

  * The probe must attach AFTER nvtracker.  `object_id` is only assigned once
    the tracker has run; a probe on nvinfer sees only the untracked sentinel.
  * pyservicemaker's ObjectMetadata wrapper is valid only while its iterator is
    advancing.  Calling `list()` on `frame_meta.object_items`, or holding a
    reference past the loop, segfaults the process.  So every field is copied
    into a plain `TrackedObject` in one pass and the wrapper is never touched
    again.
"""
from __future__ import annotations

import queue
from typing import Iterator, List, Optional, Sequence

from src.core.contract import UNTRACKED, Frame, TrackedObject

#: pyservicemaker reports "no tracking id" as uint64 max.
_UNTRACKED_SENTINEL = 0xFFFFFFFFFFFFFFFF


def make_probe_operator(labels: Sequence[str], sink: "queue.Queue"):
    """Build a BatchMetadataOperator that converts metadata into `Frame`s.

    Imported lazily inside the function so this module can be imported (and
    the class list validated) on a machine with no DeepStream install.
    """
    from pyservicemaker import BatchMetadataOperator

    class _FrameEmitter(BatchMetadataOperator):
        def __init__(self):
            super().__init__()
            self.t0: Optional[float] = None
            self.count = 0

        def handle_metadata(self, batch_meta):
            for frame_meta in batch_meta.frame_items:
                pts = frame_meta.buffer_pts / 1e9
                if self.t0 is None:
                    self.t0 = pts
                objects: List[TrackedObject] = []
                # Single pass; nothing below re-reads the wrapper.
                for obj in frame_meta.object_items:
                    cid = int(obj.class_id)
                    tid = int(obj.object_id)
                    rect = obj.rect_params
                    objects.append(TrackedObject(
                        track_id=UNTRACKED if tid == _UNTRACKED_SENTINEL else tid,
                        label=labels[cid] if 0 <= cid < len(labels) else str(cid),
                        bbox=(float(rect.left), float(rect.top),
                              float(rect.left + rect.width), float(rect.top + rect.height)),
                        confidence=float(getattr(obj, "confidence", 1.0) or 1.0),
                    ))
                sink.put(Frame(
                    index=self.count,
                    t=pts - self.t0,
                    width=int(getattr(frame_meta, "source_frame_width", 0) or 1280),
                    height=int(getattr(frame_meta, "source_frame_height", 0) or 720),
                    objects=objects,
                ))
                self.count += 1

    return _FrameEmitter()


class DeepStreamSource:
    """Owns the pipeline; yields frames the probe pushed.

    `build_pipeline` is a callable taking (config, probe_operator) and
    returning a started pyservicemaker Pipeline -- kept injectable so the
    element graph lives in the pipeline YAML/builder rather than in here.
    """

    _SENTINEL = object()

    def __init__(self, labels: Sequence[str], build_pipeline, config: dict,
                 max_queue: int = 256):
        self.labels = list(labels)
        self.build_pipeline = build_pipeline
        self.config = config
        self._queue: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._pipeline = None
        self._thread = None

    def __iter__(self) -> Iterator[Frame]:
        import threading

        operator = make_probe_operator(self.labels, self._queue)
        self._pipeline = self.build_pipeline(self.config, operator)

        def run():
            try:
                self._pipeline.start().wait()
            finally:
                self._queue.put(self._SENTINEL)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            yield item
        self.close()

    def close(self) -> None:
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
