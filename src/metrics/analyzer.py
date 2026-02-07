"""Traffic metrics computation: counts, density, occupancy, risk."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config
from src.detection.detector import Detection
from src.metrics.collision import detect_collisions


@dataclass
class TrafficMetrics:
    """All computed metrics for a single scene."""
    counts: dict[str, int]
    total_vehicles: int
    density_grid: list[list[int]]
    occupancy_pct: float
    zone_occupancy: dict[str, float]
    risk_level: str          # LOW / MEDIUM / HIGH / CRITICAL
    is_roundabout: bool
    roundabout_occupancy_pct: float | None
    collisions: list[dict] | None = None
    collision_count: int = 0


class TrafficAnalyzer:
    """Computes traffic metrics from detection results."""

    def __init__(self, grid_size: int | None = None):
        self.grid_size = grid_size or config.GRID_SIZE

    def analyze(self, detections: list[Detection], img_h: int, img_w: int,
                is_roundabout: bool = False) -> TrafficMetrics:
        """Full analysis pipeline."""
        counts = self._count_by_class(detections)
        total = sum(counts.values())
        density = self._density_grid(detections, img_h, img_w)
        occupancy = self._occupancy_pct(detections, img_h, img_w)
        zones = self._zone_occupancy(detections, img_h, img_w)
        roundabout_occ = self._roundabout_occupancy(detections, img_h, img_w) if is_roundabout else None
        collisions = detect_collisions(detections)
        risk = self._assess_risk(density, counts, collisions)

        return TrafficMetrics(
            counts=counts,
            total_vehicles=total,
            density_grid=density,
            occupancy_pct=round(occupancy, 2),
            zone_occupancy={k: round(v, 2) for k, v in zones.items()},
            risk_level=risk,
            is_roundabout=is_roundabout,
            roundabout_occupancy_pct=round(roundabout_occ, 2) if roundabout_occ is not None else None,
            collisions=collisions,
            collision_count=len(collisions),
        )

    # ── Individual metrics ─────────────────────────────────────────────

    @staticmethod
    def _count_by_class(detections: list[Detection]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return dict(sorted(counts.items()))

    def _density_grid(self, detections: list[Detection],
                      img_h: int, img_w: int) -> list[list[int]]:
        """NxN grid counting centroids per cell."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for d in detections:
            cx, cy = d.centroid
            gx = min(int(cx / img_w * self.grid_size), self.grid_size - 1)
            gy = min(int(cy / img_h * self.grid_size), self.grid_size - 1)
            grid[gy, gx] += 1
        return grid.tolist()

    @staticmethod
    def _occupancy_pct(detections: list[Detection],
                       img_h: int, img_w: int) -> float:
        """Percentage of image area covered by bounding boxes."""
        if not detections:
            return 0.0
        image_area = img_w * img_h
        total_bbox_area = 0.0
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            total_bbox_area += (x2 - x1) * (y2 - y1)
        return (total_bbox_area / image_area) * 100

    @staticmethod
    def _zone_occupancy(detections: list[Detection],
                        img_h: int, img_w: int) -> dict[str, float]:
        """Divide image into 3 horizontal zones and count proportion."""
        zones = {
            "upper": (0, 0, img_w, img_h // 3),
            "middle": (0, img_h // 3, img_w, 2 * img_h // 3),
            "lower": (0, 2 * img_h // 3, img_w, img_h),
        }
        zone_counts: dict[str, int] = {z: 0 for z in zones}
        for d in detections:
            cx, cy = d.centroid
            for zone_name, (x1, y1, x2, y2) in zones.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    zone_counts[zone_name] += 1
        total = max(len(detections), 1)
        return {z: (c / total) * 100 for z, c in zone_counts.items()}

    @staticmethod
    def _roundabout_occupancy(detections: list[Detection],
                              img_h: int, img_w: int) -> float:
        """Heuristic: % of detections within central circular region (r=40% of min dim)."""
        cx_img = img_w / 2
        cy_img = img_h / 2
        radius = 0.4 * min(img_w, img_h)
        inside = 0
        for d in detections:
            dx = d.centroid[0] - cx_img
            dy = d.centroid[1] - cy_img
            if math.sqrt(dx * dx + dy * dy) <= radius:
                inside += 1
        total = max(len(detections), 1)
        return (inside / total) * 100

    def _assess_risk(self, density_grid: list[list[int]],
                     counts: dict[str, int],
                     collisions: list[dict] | None = None) -> str:
        """Risk heuristic based on density, heavy vehicles, and collisions."""
        # Collisions automatically escalate to CRITICAL
        if collisions:
            high_collisions = any(c["severity"] == "HIGH" for c in collisions)
            if high_collisions:
                return "CRITICAL"

        max_density = max(max(row) for row in density_grid) if density_grid else 0
        has_heavy = any(counts.get(c, 0) > 0 for c in config.HEAVY_VEHICLE_CLASSES)
        has_collisions = bool(collisions)

        if has_collisions:
            return "HIGH"
        elif max_density >= config.RISK_DENSITY_THRESHOLD and has_heavy:
            return "HIGH"
        elif max_density >= config.RISK_DENSITY_THRESHOLD or has_heavy:
            return "MEDIUM"
        else:
            return "LOW"
