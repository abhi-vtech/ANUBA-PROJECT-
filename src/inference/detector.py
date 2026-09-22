import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.domain.paths import resource
from src.domain.schemas import Detection

logger = logging.getLogger(__name__)

# The ONNX graph in rf_trained/ is exported with dynamic=False at this size.
_ONNX_EXPORT_IMGSZ = 640


def _bytetrack_config_path() -> str:
    """Path to config/tracker.yaml, after checking it actually selects ByteTrack.

    Ultralytics reads the tracking ALGORITHM from THIS file's own
    ``tracker_type`` field when ``model.track(tracker=...)`` is called with
    its path -- config/model.yaml's `tracker_type` (the one Detector takes as
    a constructor argument) never reaches Ultralytics at all, it only picks
    DeepSort vs the built-in tracker below.  The two have drifted apart more
    than once in this repo's history, and every time BoT-SORT won silently:
    its sparseOptFlow global motion compensation cost ~34 ms/frame on this
    footage for no measured accuracy gain.  Failing loudly here, once, at
    startup means that class of regression can't come back quietly again --
    it only supports ByteTrack, so anything else stops the run instead of
    costing a third of the frame budget unnoticed.
    """
    path = resource("config/tracker.yaml")
    declared = yaml.safe_load(Path(path).read_text()).get("tracker_type")
    if declared != "bytetrack":
        raise RuntimeError(
            f"config/tracker.yaml sets tracker_type: {declared!r}. This "
            "pipeline only supports ByteTrack -- set tracker_type: bytetrack."
        )
    return path


def _task_for(model_type) -> str:
    """Map the configured model_type to an Ultralytics task name.

    Only consulted for exported weights, which carry no task metadata of their
    own.  Defaults to "segment" because that is what this pipeline runs
    (config/model.yaml sets `model_type: yolo-seg`, and the downstream code
    reads masks); an explicit model_type always wins.
    """
    mt = str(model_type or "").lower()
    if "seg" in mt:
        return "segment"
    if "pose" in mt:
        return "pose"
    if "obb" in mt:
        return "obb"
    if "cls" in mt or "classify" in mt:
        return "classify"
    if mt:
        return "detect"
    return "segment"


def _backend_for(model_path) -> str:
    """Map a weights filename to the runtime that will execute it.

    Purely a dispatch helper: it selects device/precision defaults so the same
    Detector works with .pt, .onnx or .engine weights without any caller or
    downstream logic changing.
    """
    name = str(model_path or "").lower()
    if name.endswith(".onnx"):
        return "onnx"
    if name.endswith(".engine"):
        return "tensorrt"
    return "torch"


class Detector:
    def __init__(
        self,
        model_path: str,
        secondary_model_path: Optional[str] = None,
        target_classes=None,
        prompt_classes: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        tracker_type: str = "bytetrack",
        tracker_config: Optional[str] = None,
        # Per-class confidence overrides: {class_name: threshold}.
        # Classes listed here use their own threshold instead of the global
        # conf_threshold passed to detect().  All other classes are unchanged.
        class_conf_overrides: Optional[dict] = None,
        # FP16 inference.  None = auto: on whenever CUDA is present.
        half: Optional[bool] = None,
        # Inference resolution.  None = let Ultralytics choose (640).
        imgsz: Optional[int] = None,
    ):
        # Which backend will execute this file.  Only the export format differs;
        # detection, tracking and every downstream consumer are unchanged.
        self.backend = _backend_for(model_path)
        self.is_onnx = self.backend == "onnx"

        self.onnx_provider = None
        self.onnx_providers_in_use: List[str] = []
        if self.is_onnx:
            # ONNX graphs run in ONNX Runtime.  With the Jetson AI Lab build of
            # onnxruntime-gpu (see pyproject.toml) the CUDA and TensorRT
            # execution providers exist and the graph runs on the GPU; with the
            # CPU-only PyPI wheel it runs on the CPU at ~650 ms per frame.
            # Asking Ultralytics for CUDA when no CUDA provider is installed
            # makes it try to pip-install onnxruntime-gpu on every load and fall
            # back to CPU anyway, so the device follows what is installed.
            from src.inference.onnx_gpu import onnx_provider

            self.onnx_provider = onnx_provider()
            self.device = 0 if self.onnx_provider in ("tensorrt", "cuda") else "cpu"
        else:
            self.device = 0 if torch.cuda.is_available() else "cpu"

        # Jetson: FP16 halves memory bandwidth and engages Orin's Ampere tensor
        # cores — the single biggest inference win on this hardware.  Guarded to
        # CUDA because most fp16 CPU kernels are unimplemented in torch, so
        # forcing half on the CPU fallback raises at the first conv.  An ONNX
        # graph exported in fp32 and run on CPU cannot use it either.
        if self.is_onnx:
            self.half = False
        else:
            self.half = (self.device != "cpu") if half is None else (bool(half) and self.device != "cpu")

        self.imgsz = int(imgsz) if imgsz else None
        if self.is_onnx and self.imgsz not in (None, _ONNX_EXPORT_IMGSZ):
            # The graph was exported with dynamic=False, so its input is fixed
            # at export size.  A different imgsz would fail the shape check
            # inside ONNX Runtime rather than silently resize.
            logger.warning(
                "ONNX model is fixed at imgsz=%d; ignoring configured imgsz=%s. "
                "Re-export with that size, or with dynamic=True, to change it.",
                _ONNX_EXPORT_IMGSZ, self.imgsz,
            )
            self.imgsz = _ONNX_EXPORT_IMGSZ

        # Shared by every ultralytics inference call in this class.
        self._infer_kwargs: dict = {"half": self.half}
        if self.imgsz:
            self._infer_kwargs["imgsz"] = self.imgsz

        # Exported graphs (.onnx/.engine) carry no task metadata.  Without an
        # explicit task Ultralytics warns "Unable to automatically guess model
        # task" and assumes "detect", then decodes a segmentation head as if it
        # were detections -- which surfaces downstream as
        # `KeyError: <n>` from self.model.names, because the mis-parsed class
        # indices run past the real class count.  Derive the task instead.
        self.task = _task_for(model_type)
        self.model = (
            YOLO(model_path) if self.backend == "torch"
            else YOLO(model_path, task=self.task)
        )
        self.secondary_model = (
            (YOLO(secondary_model_path)
             if _backend_for(secondary_model_path) == "torch"
             else YOLO(secondary_model_path, task=self.task))
            if secondary_model_path else None
        )
        self.target_classes = set(target_classes) if target_classes else None
        self.prompt_classes = prompt_classes
        self.model_type = model_type
        self.tracker_type = tracker_type
        # Resolved once and reused by every model.track() call below, rather
        # than each one re-reading and re-validating config/tracker.yaml.
        self._tracker_config_path = _bytetrack_config_path()
        # Per-class confidence overrides (additive — does not affect any
        # class not explicitly listed here).
        self.class_conf_overrides: dict = class_conf_overrides or {}

        if self.prompt_classes:
            try:
                self.model.set_classes(self.prompt_classes)
            except AttributeError:
                pass

        self.deepsort = None
        if tracker_type == "deepsort":
            config = {}
            if tracker_config:
                data = yaml.safe_load(open(tracker_config))
                config = data.get("deepsort", {})

            embedder = config.get("embedder", "mobilenet")
            kwargs = {
                "max_age": config.get("max_age", 50),
                "n_init": config.get("n_init", 3),
                "max_cosine_distance": config.get("max_cosine_distance", 0.2),
                "nn_budget": config.get("nn_budget", 100),
                "embedder": embedder,
                "half": config.get("half", True),
                "bgr": config.get("bgr", True),
                "embedder_gpu": config.get("embedder_gpu", True),
            }

            from deep_sort_realtime.deepsort_tracker import DeepSort

            self.deepsort = DeepSort(**kwargs)

        self.system_roi = None
        import os
        import json
        roi_path = resource("config/system_roi.json")
        if os.path.exists(roi_path):
            try:
                with open(roi_path, 'r') as f:
                    self.system_roi = json.load(f)
            except Exception:
                pass

        # Open the ONNX session now, on the GPU provider chosen above, rather
        # than on the first real frame.  With TensorRT the first open builds the
        # FP16 engine (~8 minutes on this Jetson) or reads it back from cache.
        if self.is_onnx and self.device != "cpu":
            self._open_onnx_session(model_path)

    def _open_onnx_session(self, model_path: str) -> None:
        import contextlib
        import time
        from pathlib import Path

        from src.inference.onnx_gpu import session_providers, tensorrt_first

        cache = Path(model_path).resolve().parent / "ort_trt_cache"
        guard = tensorrt_first(cache) if self.onnx_provider == "tensorrt" else contextlib.nullcontext()
        started = time.perf_counter()
        with guard:
            self.model.track(
                np.zeros((720, 1280, 3), np.uint8),
                persist=True,
                verbose=False,
                conf=0.99,
                tracker=self._tracker_config_path,
                device=self.device,
                retina_masks=True,
                **self._infer_kwargs,
            )
        self.onnx_providers_in_use = session_providers(self.model)
        logger.info(
            "ONNX model ready on %s in %.1f s (execution providers: %s)",
            self.onnx_provider,
            time.perf_counter() - started,
            ", ".join(self.onnx_providers_in_use) or "unknown",
        )

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        if self.tracker_type == "deepsort":
            detections = self._detect_deepsort(frame, conf_threshold)
        else:
            detections = self._detect_builtin(frame, conf_threshold)
            
        if self.system_roi:
            filtered = []
            rx, ry, rw, rh = self.system_roi.get('x', 0), self.system_roi.get('y', 0), self.system_roi.get('w', 0), self.system_roi.get('h', 0)
            if rw > 0 and rh > 0:
                for det in detections:
                    if det.class_name in ("hot-dog", "burger_bun", "french_fries", "wrapping", "wrapped", "wrapper"):
                        cx = (det.bbox[0] + det.bbox[2]) / 2
                        cy = (det.bbox[1] + det.bbox[3]) / 2
                        if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh):
                            continue
                    filtered.append(det)
                return filtered
                
        return detections

    def _detect_builtin(
        self, frame: np.ndarray, conf_threshold: float
    ) -> List[Detection]:
        # Determine the lowest confidence we care about across all classes.
        # We must pass this to model.track() so YOLO's internal detection
        # stage does not silently drop classes with per-class overrides
        # (e.g. "wrapping" at 0.15) before they reach our Python filter.
        min_conf = min(
            [conf_threshold] + list(self.class_conf_overrides.values())
        )
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=min_conf,
            tracker=self._tracker_config_path,
            device=self.device,
            retina_masks=True,  # User requested tight masks, this prevents low-res mask bleed
            **self._infer_kwargs,
        )
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id
            # Extract masks if available
            masks = results[0].masks if hasattr(results[0], 'masks') else None
            
            for idx in range(len(boxes)):
                conf = boxes.conf[idx]
                cls = boxes.cls[idx]
                track_id = int(ids[idx]) if ids is not None else -1

                name = self.model.names[int(cls)]
                effective_threshold = self.class_conf_overrides.get(
                    name, conf_threshold
                )
                if float(conf) < effective_threshold:
                    continue
                if self.target_classes and name not in self.target_classes:
                    continue

                poly_pts = None
                bbox = tuple(map(int, boxes.xyxy[idx].tolist()))
                
                # Apply segmentation polygon if available
                if masks is not None and len(masks.xy) > idx:
                    pts = masks.xy[idx]
                    if pts is not None and len(pts) >= 3:
                        poly_pts = pts.tolist()
                        # Override YOLO's standard bbox with tight contour bounding rect
                        px, py, pw, ph = cv2.boundingRect(pts.astype(np.int32))
                        bbox = (px, py, px + pw, py + ph)

                det = Detection(
                    track_id=track_id,
                    bbox=bbox,
                    class_name=name,
                    confidence=float(conf),
                    polygon=poly_pts,
                )
                if poly_pts:
                    det.polygon = poly_pts
                    
                detections.append(det)

        if self.secondary_model:
            sec_results = self.secondary_model.track(
                frame,
                persist=True,
                verbose=False,
                tracker=self._tracker_config_path,
                device=self.device,
                **self._infer_kwargs,
            )
            if sec_results[0].boxes is not None:
                sec_boxes = sec_results[0].boxes
                sec_ids = sec_boxes.id
                for idx in range(len(sec_boxes)):
                    box = sec_boxes.xyxy[idx]
                    conf = sec_boxes.conf[idx]
                    cls = sec_boxes.cls[idx]

                    # Offset track ID to prevent conflicts, fallback if not tracked
                    track_id = int(sec_ids[idx]) if sec_ids is not None else -1
                    if track_id != -1:
                        track_id += 10000
                    else:
                        track_id = 10000 + idx

                    if float(conf) < conf_threshold:
                        continue
                    name = self.secondary_model.names[int(cls)]
                    if self.target_classes and name not in self.target_classes:
                        continue
                    detections.append(
                        Detection(
                            track_id=track_id,
                            bbox=tuple(map(int, box.tolist())),
                            class_name=name,
                            confidence=float(conf),
                        )
                    )

        return detections

    def _detect_deepsort(
        self, frame: np.ndarray, conf_threshold: float
    ) -> List[Detection]:
        results = self.model(frame, verbose=False, device=self.device, **self._infer_kwargs)[0]

        raw_detections = []
        for box in results.boxes:
            conf = float(box.conf)
            if conf < conf_threshold:
                continue
            cls = int(box.cls)
            name = self.model.names[cls]
            if self.target_classes and name not in self.target_classes:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            raw_detections.append(([x1, y1, w, h], conf, cls))

        if self.deepsort.embedder is not None:
            tracks = self.deepsort.update_tracks(raw_detections, frame=frame)
        else:
            unit = np.ones(128) / np.sqrt(128)
            embeds = [unit] * len(raw_detections)
            tracks = self.deepsort.update_tracks(raw_detections, embeds=embeds)

        detections = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            left, top, right, bottom = track.to_ltrb()
            cls_id = track.get_det_class()
            name = self.model.names.get(cls_id, str(cls_id))
            if self.target_classes and name not in self.target_classes:
                continue
            detections.append(
                Detection(
                    track_id=track.track_id,
                    bbox=(int(left), int(top), int(right), int(bottom)),
                    class_name=name,
                    confidence=track.get_det_conf() or 0.0,
                )
            )
        return detections
