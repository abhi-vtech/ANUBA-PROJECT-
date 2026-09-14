"""Ultralytics source: .pt, .onnx or .engine, through the existing Detector.

Backend selection already works by filename in `src.inference.detector._backend_for`, so
this adapter takes a weights path and inherits that behaviour unchanged -- a
`.pt` run and a `.engine` run differ only in the string handed to it.

Everything that is genuinely runtime-specific -- opening the video, the
inference call, FP16, imgsz, the tracker config -- stays behind this class.
What comes out is `Frame`.
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

from src.core.contract import Frame, TrackedObject


class UltralyticsSource:
    def __init__(
        self,
        video: str,
        model_path: str,
        conf: float = 0.5,
        fps: Optional[float] = None,
        detector_kwargs: Optional[Dict[str, Any]] = None,
        max_frames: Optional[int] = None,
    ):
        self.video = video
        self.model_path = model_path
        self.conf = conf
        self.fps = fps
        self.detector_kwargs = detector_kwargs or {}
        self.max_frames = max_frames
        self._cap = None

    def __iter__(self) -> Iterator[Frame]:
        import cv2  # imported here so the core stays importable without cv2

        from src.inference.detector import Detector

        detector = Detector(self.model_path, **self.detector_kwargs)
        self._cap = cv2.VideoCapture(self.video)
        if not self._cap.isOpened():
            raise RuntimeError("could not open video: {0}".format(self.video))

        fps = self.fps or self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        index = 0
        try:
            while True:
                if self.max_frames is not None and index >= self.max_frames:
                    break
                ok, frame = self._cap.read()
                if not ok:
                    break
                detections = detector.detect(frame, conf_threshold=self.conf)
                yield Frame(
                    index=index,
                    t=index / fps,
                    width=width,
                    height=height,
                    objects=[
                        TrackedObject(
                            track_id=int(d.track_id),
                            label=d.class_name,
                            bbox=tuple(float(v) for v in d.bbox),
                            confidence=float(d.confidence),
                            polygon=d.polygon,
                        )
                        for d in detections
                    ],
                    extra={"bgr": frame},
                )
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
