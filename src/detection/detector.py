"""YOLOv8 vehicle detection wrapper."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    bbox: list[float]      # [x1, y1, x2, y2] absolute
    centroid: list[float]   # [cx, cy]


class VehicleDetector:
    """Wraps YOLOv8 for vehicle detection in aerial images."""

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 conf_threshold: float | None = None):
        self.model_path = model_path or config.YOLO_MODEL
        self.device = device or config.DEVICE
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.model = YOLO(self.model_path)
        self.vehicle_classes = config.VEHICLE_CLASSES
        logger.info("Loaded model %s on %s (conf=%.2f)",
                     self.model_path, self.device, self.conf_threshold)

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Run inference on a single image (BGR numpy array).

        Returns list of Detection objects for vehicle classes only.
        """
        results = self.model(image, device=self.device, conf=self.conf_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id not in self.vehicle_classes:
                    continue

                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append(Detection(
                    class_name=self.vehicle_classes[cls_id],
                    class_id=cls_id,
                    confidence=round(conf, 4),
                    bbox=[round(v, 2) for v in [x1, y1, x2, y2]],
                    centroid=[round(cx, 2), round(cy, 2)],
                ))

        logger.debug("Detected %d vehicles in image", len(detections))
        return detections

    def detect_file(self, image_path: str | Path) -> tuple[np.ndarray, list[Detection]]:
        """Load image from path and detect. Returns (image, detections)."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return img, self.detect(img)

    @property
    def model_version(self) -> str:
        return self.model_path
