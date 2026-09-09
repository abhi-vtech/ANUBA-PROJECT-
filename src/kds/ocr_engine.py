"""Pluggable OCR backend.

The rest of the KDS package only ever sees :class:`OcrBox` objects, so the
engine can be swapped in ``config/kds_visual.yaml`` (``ocr.engine``) without
touching the parser.  ``StubOcrEngine`` lets the whole pipeline -- parsing,
stability, FIFO, validation -- be unit tested with no model download and no
video file.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BBox = Tuple[int, int, int, int]


@dataclass
class OcrBox:
    """One recognised text box, in the coordinate space of the image passed in."""

    text: str
    bbox: BBox          # (x1, y1, x2, y2)
    confidence: float

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2.0

    @property
    def height(self) -> int:
        return max(1, self.bbox[3] - self.bbox[1])

    def scaled(self, fx: float, fy: float) -> "OcrBox":
        x1, y1, x2, y2 = self.bbox
        return OcrBox(
            self.text,
            (int(x1 / fx), int(y1 / fy), int(x2 / fx), int(y2 / fy)),
            self.confidence,
        )


class OcrEngine(ABC):
    """Recognise text in a BGR image."""

    @abstractmethod
    def read(self, image: np.ndarray) -> List[OcrBox]:
        raise NotImplementedError

    @property
    def name(self) -> str:
        return type(self).__name__


class RapidOcrEngine(OcrEngine):
    """RapidOCR (PP-OCR models on onnxruntime).

    Chosen because it is a pure pip install with no system binary, runs fast on
    CPU, and handles the high-contrast, low-resolution text of a photographed
    KDS monitor well.  Import is lazy so that unit tests and the production-only
    pipeline never pay for it.
    """

    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        self._engine = None
        self._unavailable_reason: Optional[str] = None

    def _ensure(self) -> bool:
        if self._engine is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        try:
            from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
        except ImportError as exc:
            self._unavailable_reason = str(exc)
            logger.error(
                "RapidOCR is not installed -- KDS text extraction is disabled. "
                "Install it with:  uv pip install rapidocr-onnxruntime onnxruntime"
            )
            return False
        self._engine = RapidOCR()
        logger.info("RapidOCR engine initialised")
        return True

    def read(self, image: np.ndarray) -> List[OcrBox]:
        if image is None or image.size == 0:
            return []
        if not self._ensure():
            return []
        try:
            result, _ = self._engine(image)
        except Exception:  # pragma: no cover - engine-internal failures
            logger.exception("RapidOCR failed on a KDS crop")
            return []
        if not result:
            return []

        boxes: List[OcrBox] = []
        for entry in result:
            # RapidOCR returns [polygon(4x2), text, score]
            polygon, text, score = entry[0], entry[1], float(entry[2])
            if score < self.min_confidence:
                continue
            text = (text or "").strip()
            if not text:
                continue
            xs = [float(p[0]) for p in polygon]
            ys = [float(p[1]) for p in polygon]
            boxes.append(
                OcrBox(
                    text=text,
                    bbox=(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
                    confidence=score,
                )
            )
        return boxes


class StubOcrEngine(OcrEngine):
    """Deterministic engine driven by a caller-supplied callback or script.

    Two modes:

    * ``responses`` -- a list of ``List[OcrBox]``, one per :meth:`read` call,
      with the last entry repeating once exhausted.
    * ``callback``  -- ``fn(image) -> List[OcrBox]``.

    Used by ``tests/test_kds_vision.py`` to drive the full pipeline frame by
    frame, including deliberate OCR dropouts.
    """

    def __init__(
        self,
        responses: Optional[Sequence[List[OcrBox]]] = None,
        callback: Optional[Callable[[np.ndarray], List[OcrBox]]] = None,
    ):
        self._responses = list(responses) if responses is not None else None
        self._callback = callback
        self.calls = 0

    def read(self, image: np.ndarray) -> List[OcrBox]:
        self.calls += 1
        if self._callback is not None:
            return self._callback(image)
        if not self._responses:
            return []
        if len(self._responses) == 1:
            return list(self._responses[0])
        return list(self._responses.pop(0))


def build_ocr_engine(config: Optional[Dict] = None) -> OcrEngine:
    """Construct the engine named by ``config['ocr']['engine']``."""
    config = config or {}
    ocr_cfg = config.get("ocr", {}) if isinstance(config, dict) else {}
    name = str(ocr_cfg.get("engine", "rapidocr")).lower()
    min_conf = float(ocr_cfg.get("min_confidence", 0.35))
    if name in ("stub", "none", "null"):
        return StubOcrEngine()
    if name != "rapidocr":
        logger.warning("Unknown OCR engine %r, falling back to rapidocr", name)
    return RapidOcrEngine(min_confidence=min_conf)
