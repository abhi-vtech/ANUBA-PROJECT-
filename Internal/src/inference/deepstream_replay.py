"""Serve detections produced by DeepStream in place of running YOLO on the host.

DeepStream is not installed natively on this Jetson -- it lives in a Docker
image -- and the container has none of what the rest of this pipeline needs
(FastAPI, the OCR engine, OpenCV bindings for the dashboard).  So inference and
analysis are split: `deepstream_test/ds_detect_dump.py` runs the model inside
the container and writes one JSON line of tracked detections per frame, and
this class replays that stream to `src/main.py` as if a local detector had just
produced it.  Nothing else in the pipeline knows the difference -- the KDS
reader, the state machines and the dashboard are untouched.

Lines are keyed by MEDIA TIMESTAMP, not frame index, so the host's own decode
of the same video lines up without both sides having to see identical frames.
The file is read forward-only, which is all the main loop ever needs and keeps
a 40-minute run to a few MB of memory.
"""
from __future__ import annotations

import bisect
import json
import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from src.domain.schemas import Detection

logger = logging.getLogger(__name__)


class DeepStreamReplayDetector:
    """Drop-in for `Detector.detect()` backed by a DeepStream JSONL dump."""

    backend = "deepstream"

    def __init__(
        self,
        jsonl_path: str,
        labels: Sequence[str],
        max_skew_s: float = 0.05,
        class_conf_overrides: Optional[dict] = None,
    ):
        self.path = Path(jsonl_path)
        self.labels = list(labels)
        # A detection is accepted for a frame when its timestamp is within this
        # of the frame's own.  One frame at 30 fps is 0.033s, so the default
        # tolerates a little under two frames of jitter and no more.
        self.max_skew_s = float(max_skew_s)
        self.class_conf_overrides = dict(class_conf_overrides or {})

        self._times: List[float] = []
        self._rows: List[list] = []
        self._load()
        self.misses = 0
        self.hits = 0

    def _load(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(
                "DeepStream detections not found: %s -- run "
                "scripts/run_deepstream_detect.sh first" % self.path
            )
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                self._times.append(float(d["t"]))
                self._rows.append(d.get("o") or [])
        if not self._times:
            raise ValueError("DeepStream detections file is empty: %s" % self.path)
        logger.info(
            "DeepStream detections: %d frames covering %.1fs -> %.1fs of media time (%s)",
            len(self._times), self._times[0], self._times[-1], self.path,
        )

    # ----------------------------------------------------------------- detect

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        """Detections for the frame whose media time is `self.media_time`.

        `frame` is accepted and ignored: the pixels were already looked at, in
        the container.  The caller sets `media_time` each frame before calling.
        """
        t = getattr(self, "media_time", None)
        if t is None:
            return []
        idx = bisect.bisect_left(self._times, t)
        best, best_gap = None, None
        for cand in (idx - 1, idx, idx + 1):
            if 0 <= cand < len(self._times):
                gap = abs(self._times[cand] - t)
                if best_gap is None or gap < best_gap:
                    best, best_gap = cand, gap
        if best is None or best_gap > self.max_skew_s:
            self.misses += 1
            return []
        self.hits += 1

        out: List[Detection] = []
        for cls, conf, x1, y1, x2, y2, tid in self._rows[best]:
            if not (0 <= cls < len(self.labels)):
                continue
            name = self.labels[cls]
            # The container config carries the same per-class gates, but apply
            # them again here so a host-side change takes effect without
            # re-running the model.
            threshold = self.class_conf_overrides.get(name, conf_threshold)
            if conf < threshold:
                continue
            out.append(
                Detection(
                    track_id=int(tid),
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    class_name=name,
                    confidence=float(conf),
                    polygon=None,   # boxes only; nothing downstream requires masks
                )
            )
        return out

    def coverage(self) -> str:
        total = self.hits + self.misses
        pct = 100.0 * self.hits / total if total else 0.0
        return "%d/%d frames matched (%.1f%%)" % (self.hits, total, pct)
