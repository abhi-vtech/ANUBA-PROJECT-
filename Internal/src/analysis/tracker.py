import numpy as np

from src.inference.detector import Detector


class Tracker:
    def __init__(self, model_path: str):
        self.detector = Detector(model_path)

    def update(self, frame: np.ndarray):
        return self.detector.detect(frame)
