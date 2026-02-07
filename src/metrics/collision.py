"""Collision detection between vehicles based on bounding box analysis."""

from __future__ import annotations

import math
from typing import Any

from src.detection.detector import Detection
import config


def compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Compute Intersection over Union between two bboxes [x1,y1,x2,y2]."""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_centroid_distance(det1: Detection, det2: Detection) -> float:
    """Euclidean distance between centroids of two detections."""
    cx1, cy1 = det1.centroid
    cx2, cy2 = det2.centroid
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)


def _classify_severity(iou: float, distance: float) -> str:
    """Classify collision severity based on IoU and distance."""
    if iou > 0.15:
        return "HIGH"
    elif iou > config.COLLISION_IOU_THRESHOLD:
        return "MEDIUM"
    else:
        return "WARNING"


def detect_collisions(
    detections: list[Detection],
    iou_threshold: float | None = None,
    distance_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Detect potential collisions between vehicles.

    Compares all pairs of detections. A collision is flagged when:
    - IoU > iou_threshold (bounding boxes overlap), OR
    - Centroid distance < distance_threshold (very close proximity)

    Returns list of collision records with involved vehicles and severity.
    """
    iou_thresh = iou_threshold or config.COLLISION_IOU_THRESHOLD
    dist_thresh = distance_threshold or config.COLLISION_DISTANCE_THRESHOLD

    collisions: list[dict[str, Any]] = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det_a = detections[i]
            det_b = detections[j]

            iou = compute_iou(det_a.bbox, det_b.bbox)
            distance = compute_centroid_distance(det_a, det_b)

            if iou > iou_thresh or distance < dist_thresh:
                severity = _classify_severity(iou, distance)
                collisions.append({
                    "vehicle_a_idx": i,
                    "vehicle_b_idx": j,
                    "vehicle_a_class": det_a.class_name,
                    "vehicle_b_class": det_b.class_name,
                    "vehicle_a_centroid": det_a.centroid,
                    "vehicle_b_centroid": det_b.centroid,
                    "iou": round(iou, 4),
                    "distance": round(distance, 2),
                    "severity": severity,
                })

    # Sort by severity (HIGH first)
    severity_order = {"HIGH": 0, "MEDIUM": 1, "WARNING": 2}
    collisions.sort(key=lambda c: severity_order.get(c["severity"], 3))

    return collisions
