"""Visualization utilities: detection overlays and heatmaps."""

from __future__ import annotations

import cv2
import numpy as np
import scipy.ndimage

import config
from src.detection.detector import Detection

# Colors per class (BGR)
_COLORS = {
    "car": (0, 255, 0),
    "motorcycle": (255, 165, 0),
    "bicycle": (255, 255, 0),
    "bus": (0, 0, 255),
    "truck": (0, 128, 255),
    "cycle": (255, 255, 0),
}
_DEFAULT_COLOR = (200, 200, 200)


def draw_detections(image: np.ndarray, detections: list[Detection],
                    show_labels: bool = True) -> np.ndarray:
    """Draw bounding boxes and labels on image. Returns a copy."""
    img = image.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        color = _COLORS.get(d.class_name, _DEFAULT_COLOR)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        if show_labels:
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def generate_heatmap(image: np.ndarray, detections: list[Detection],
                     sigma: float = 15.0) -> np.ndarray:
    """Generate a weighted heatmap overlay.

    Returns BGR image with heatmap blended on top.
    """
    h, w = image.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        weight = config.CLASS_WEIGHTS.get(d.class_name, 1.0)
        heat[y1:y2, x1:x2] += weight

    if heat.max() > 0:
        heat = scipy.ndimage.gaussian_filter(heat, sigma=sigma)
        heat = heat / heat.max()

    # Convert to color heatmap
    heatmap_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Blend with original
    blended = cv2.addWeighted(image, 0.6, heatmap_color, 0.4, 0)
    return blended


def draw_collisions(image: np.ndarray, collisions: list[dict],
                     detections: list[Detection]) -> np.ndarray:
    """Draw collision indicators: red boxes around involved vehicles + connecting lines."""
    img = image.copy()

    _SEVERITY_COLORS = {
        "HIGH": (0, 0, 255),      # Red
        "MEDIUM": (0, 128, 255),   # Orange
        "WARNING": (0, 255, 255),  # Yellow
    }

    for col in collisions:
        idx_a = col["vehicle_a_idx"]
        idx_b = col["vehicle_b_idx"]
        severity = col["severity"]
        color = _SEVERITY_COLORS.get(severity, (0, 0, 255))

        det_a = detections[idx_a]
        det_b = detections[idx_b]

        # Draw thick boxes around involved vehicles
        for det in (det_a, det_b):
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        # Draw line between centroids
        ca = tuple(map(int, det_a.centroid))
        cb = tuple(map(int, det_b.centroid))
        cv2.line(img, ca, cb, color, 2, cv2.LINE_AA)

        # Label at midpoint
        mid_x = (ca[0] + cb[0]) // 2
        mid_y = (ca[1] + cb[1]) // 2
        label = f"{severity} IoU:{col['iou']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (mid_x - 2, mid_y - th - 6), (mid_x + tw + 2, mid_y), color, -1)
        cv2.putText(img, label, (mid_x, mid_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def draw_density_grid(image: np.ndarray, density_grid: list[list[int]]) -> np.ndarray:
    """Overlay density grid numbers on image."""
    img = image.copy()
    h, w = img.shape[:2]
    rows = len(density_grid)
    cols = len(density_grid[0]) if rows > 0 else 0

    cell_h = h // rows
    cell_w = w // cols

    for gy in range(rows):
        for gx in range(cols):
            count = density_grid[gy][gx]
            # Grid lines
            cv2.rectangle(img, (gx * cell_w, gy * cell_h),
                          ((gx + 1) * cell_w, (gy + 1) * cell_h),
                          (255, 255, 255), 1)
            # Count text
            if count > 0:
                cx = gx * cell_w + cell_w // 2 - 10
                cy = gy * cell_h + cell_h // 2 + 10
                cv2.putText(img, str(count), (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img
