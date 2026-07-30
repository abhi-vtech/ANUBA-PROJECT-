from typing import List, Optional

import numpy as np
import torch
import yaml
from ultralytics import YOLO

from src.paths import resource
from src.schemas import Detection


class Detector:
    def __init__(
        self,
        model_path: str,
        secondary_model_path: Optional[str] = None,
        target_classes=None,
        prompt_classes: Optional[List[str]] = None,
        model_type: Optional[str] = None,
        tracker_type: str = "botsort",
        tracker_config: Optional[str] = None,
    ):
        self.device = 0 if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        self.secondary_model = (
            YOLO(secondary_model_path) if secondary_model_path else None
        )
        self.target_classes = set(target_classes) if target_classes else None
        self.prompt_classes = prompt_classes
        self.model_type = model_type
        self.tracker_type = tracker_type

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

    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Detection]:
        if self.tracker_type == "deepsort":
            return self._detect_deepsort(frame, conf_threshold)
        return self._detect_builtin(frame, conf_threshold)

    def _detect_builtin(
        self, frame: np.ndarray, conf_threshold: float
    ) -> List[Detection]:
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            tracker=resource("config/tracker.yaml"),
            device=self.device,
        )
        detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            ids = boxes.id
            for idx in range(len(boxes)):
                box = boxes.xyxy[idx]
                conf = boxes.conf[idx]
                cls = boxes.cls[idx]
                track_id = int(ids[idx]) if ids is not None else -1

                if float(conf) < conf_threshold:
                    continue
                name = self.model.names[int(cls)]
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

        if self.secondary_model:
            sec_results = self.secondary_model.track(
                frame,
                persist=True,
                verbose=False,
                tracker=resource("config/tracker.yaml"),
                device=self.device,
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
        results = self.model(frame, verbose=False, device=self.device)[0]

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
