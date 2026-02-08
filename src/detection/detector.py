"""Sistema de detección de vehículos con YOLOv8."""

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
    bbox: list[float]      # [x1, y1, x2, y2] absoluto
    centroid: list[float]   # [cx, cy]


class VehicleDetector:
    """Utiliza YOLOv8 para detección de vehículos en imágenes aéreas."""

    def __init__(self, model_path: str | None = None, device: str | None = None,
                 conf_threshold: float | None = None):
        self.model_path = model_path or config.YOLO_MODEL
        self.device = device or config.DEVICE
        self.conf_threshold = conf_threshold or config.CONFIDENCE_THRESHOLD
        self.model = YOLO(self.model_path)
        self.vehicle_classes = config.VEHICLE_CLASSES

        # Modelo COCO de respaldo para clases que el modelo afinado no detecta
        self.fallback_model = None
        self.fallback_classes = getattr(config, "FALLBACK_COCO_CLASSES", {})
        if self.fallback_classes:
            fallback_path = getattr(config, "YOLO_FALLBACK_MODEL", "yolov8m.pt")
            try:
                self.fallback_model = YOLO(fallback_path)
                logger.info("Loaded fallback COCO model %s for classes %s",
                            fallback_path, list(self.fallback_classes.values()))
            except Exception as e:
                logger.warning("Could not load fallback model: %s", e)

        logger.info("Loaded model %s on %s (conf=%.2f)",
                     self.model_path, self.device, self.conf_threshold)

    @staticmethod
    def _iou(box_a: list[float], box_b: list[float]) -> float:
        """Calcula IoU entre dos cajas [x1,y1,x2,y2]."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _extract_detections(self, results, class_map: dict[int, str]) -> list[Detection]:
        """Extrae objetos Detection de los resultados de YOLO usando el mapa de clases dado."""
        detections = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in class_map:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                detections.append(Detection(
                    class_name=class_map[cls_id],
                    class_id=cls_id,
                    confidence=round(conf, 4),
                    bbox=[round(v, 2) for v in [x1, y1, x2, y2]],
                    centroid=[round(cx, 2), round(cy, 2)],
                ))
        return detections

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Ejecuta inferencia en una sola imagen (array numpy BGR).

        Devuelve lista de objetos Detection solo para clases de vehículos.
        Usa modelo COCO de respaldo para clases faltantes (ej. bus).
        """
        # Modelo primario
        results = self.model(image, device=self.device, conf=self.conf_threshold, verbose=False)
        detections = self._extract_detections(results, self.vehicle_classes)

        # Modelo de respaldo para clases faltantes
        if self.fallback_model and self.fallback_classes:
            fb_results = self.fallback_model(
                image, device=self.device, conf=self.conf_threshold, verbose=False,
            )
            fb_dets = self._extract_detections(fb_results, self.fallback_classes)

            # Mezclar: solo agregar detecciones de respaldo que no se solapan con las existentes
            for fb in fb_dets:
                overlaps = any(self._iou(fb.bbox, d.bbox) > 0.3 for d in detections)
                if not overlaps:
                    detections.append(fb)

            if fb_dets:
                logger.debug("Fallback added %d bus detections",
                             sum(1 for d in fb_dets
                                 if not any(self._iou(d.bbox, ex.bbox) > 0.3
                                            for ex in detections if ex is not d)))

        logger.debug("Detected %d vehicles in image", len(detections))
        return detections

    def detect_file(self, image_path: str | Path) -> tuple[np.ndarray, list[Detection]]:
        """Carga imagen desde ruta y detecta. Devuelve (imagen, detecciones)."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
        return img, self.detect(img)

    @property
    def model_version(self) -> str:
        return self.model_path
