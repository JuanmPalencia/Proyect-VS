"""Traffic metrics computation: counts, density, occupancy, risk."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import config
from src.detection.detector import Detection
from src.metrics.collision import detect_collisions


# ──────────────────────────────────────────────────────────────────────
# Data container
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrafficMetrics:
    """All computed metrics for a single scene."""
    counts: dict[str, int]
    total_vehicles: int
    density_grid: list[list[int]]
    occupancy_pct: float
    traffic_density: str             # FLUIDO / MODERADO / DENSO / SATURADO
    zone_occupancy: dict[str, float]
    risk_level: str                  # LOW / MEDIUM / HIGH / CRITICAL
    is_roundabout: bool
    roundabout_occupancy_pct: float | None
    roundabout_level: str | None     # BAJA / MEDIA / ALTA / CRÍTICA
    collisions: list[dict] | None = None
    collision_count: int = 0


# ──────────────────────────────────────────────────────────────────────
# Analyzer
# ──────────────────────────────────────────────────────────────────────

class TrafficAnalyzer:
    """Computes traffic metrics from detection results."""

    def __init__(self, grid_size: int | None = None):
        self.grid_size = grid_size or config.GRID_SIZE

    def analyze(
        self,
        detections: list[Detection],
        img_h: int,
        img_w: int,
        is_roundabout: bool = False,
    ) -> TrafficMetrics:
        """Full analysis pipeline."""

        # ── Confidence filtering (important for incident quality) ──
        conf_thresh = config.CONFIDENCE_THRESHOLD
        detections_f = [d for d in detections if d.confidence >= conf_thresh]

        # ── Basic metrics ──
        counts = self._count_by_class(detections_f)
        total = sum(counts.values())
        density = self._density_grid(detections_f, img_h, img_w)
        occupancy = self._occupancy_pct(detections_f, img_h, img_w)
        zones = self._zone_occupancy(detections_f, img_h, img_w)

        roundabout_occ = (
            self._roundabout_occupancy(detections_f, img_h, img_w)
            if is_roundabout
            else None
        )

        # ── Parking detection ──
        parked_ratio = self._estimate_parked_ratio(detections_f)

        # ── Traffic density level (discounting parked vehicles) ──
        t_density = self._traffic_density_level(density, counts, parked_ratio)

        # ── Roundabout level ──
        roundabout_lvl = (
            self._roundabout_level(roundabout_occ) if roundabout_occ is not None else None
        )

        # ── Incident detection ──
        collisions = detect_collisions(detections_f)

        # ── Risk assessment ──
        risk = self._assess_risk(density, counts, collisions)

        return TrafficMetrics(
            counts=counts,
            total_vehicles=total,
            density_grid=density,
            occupancy_pct=round(occupancy, 2),
            traffic_density=t_density,
            zone_occupancy={k: round(v, 2) for k, v in zones.items()},
            risk_level=risk,
            is_roundabout=is_roundabout,
            roundabout_occupancy_pct=round(roundabout_occ, 2)
            if roundabout_occ is not None
            else None,
            roundabout_level=roundabout_lvl,
            collisions=collisions,
            collision_count=len(collisions),
        )

    # ──────────────────────────────────────────────────────────────────
    # Individual metrics
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _count_by_class(detections: list[Detection]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return dict(sorted(counts.items()))

    def _density_grid(
        self,
        detections: list[Detection],
        img_h: int,
        img_w: int,
    ) -> list[list[int]]:
        """NxN grid counting centroids per cell."""
        grid = np.zeros((self.grid_size, self.grid_size), dtype=int)
        for d in detections:
            cx, cy = d.centroid
            gx = min(int(cx / img_w * self.grid_size), self.grid_size - 1)
            gy = min(int(cy / img_h * self.grid_size), self.grid_size - 1)
            grid[gy, gx] += 1
        return grid.tolist()

    @staticmethod
    def _occupancy_pct(
        detections: list[Detection],
        img_h: int,
        img_w: int,
    ) -> float:
        """Percentage of image area covered by bounding boxes (sum heuristic)."""
        if not detections:
            return 0.0

        image_area = img_w * img_h
        total_bbox_area = 0.0

        for d in detections:
            x1, y1, x2, y2 = d.bbox
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            total_bbox_area += w * h

        return (total_bbox_area / image_area) * 100.0

    @staticmethod
    def _zone_occupancy(
        detections: list[Detection],
        img_h: int,
        img_w: int,
    ) -> dict[str, float]:
        """Divide image into 3 horizontal zones and compute % of vehicles."""
        zones = {
            "upper": (0, 0, img_w, img_h // 3),
            "middle": (0, img_h // 3, img_w, 2 * img_h // 3),
            "lower": (0, 2 * img_h // 3, img_w, img_h),
        }

        zone_counts = {z: 0 for z in zones}

        for d in detections:
            cx, cy = d.centroid
            for zone_name, (x1, y1, x2, y2) in zones.items():
                if x1 <= cx <= x2 and y1 <= cy <= y2:
                    zone_counts[zone_name] += 1

        total = max(len(detections), 1)
        return {z: (c / total) * 100.0 for z, c in zone_counts.items()}

    @staticmethod
    def _roundabout_occupancy(
        detections: list[Detection],
        img_h: int,
        img_w: int,
    ) -> float:
        """% of vehicles within central circular region (roundabout heuristic)."""
        cx_img = img_w / 2.0
        cy_img = img_h / 2.0
        radius = 0.4 * min(img_w, img_h)

        inside = 0
        for d in detections:
            dx = d.centroid[0] - cx_img
            dy = d.centroid[1] - cy_img
            if math.sqrt(dx * dx + dy * dy) <= radius:
                inside += 1

        total = max(len(detections), 1)
        return (inside / total) * 100.0

    @staticmethod
    def _estimate_parked_ratio(detections: list[Detection]) -> float:
        """Estimate the fraction of vehicles that appear parked.

        Heuristic: vehicles aligned in a row with regular spacing and
        similar bbox sizes are likely parked (aerial parking-lot pattern).
        A vehicle is "parked" if it has 2+ aligned, similarly-sized neighbors.
        """
        if len(detections) < 3:
            return 0.0

        parked = set()

        for i, d in enumerate(detections):
            w_i = d.bbox[2] - d.bbox[0]
            h_i = d.bbox[3] - d.bbox[1]
            size_i = max(w_i, h_i)
            if size_i == 0:
                continue

            aligned_count = 0
            for j, o in enumerate(detections):
                if i == j:
                    continue
                w_j = o.bbox[2] - o.bbox[0]
                h_j = o.bbox[3] - o.bbox[1]
                size_j = max(w_j, h_j)
                if size_j == 0:
                    continue

                # Similar size (within 40%)
                size_ratio = min(size_i, size_j) / max(size_i, size_j)
                if size_ratio < 0.6:
                    continue

                dx = abs(d.centroid[0] - o.centroid[0])
                dy = abs(d.centroid[1] - o.centroid[1])
                avg_size = (size_i + size_j) / 2

                # Close enough to be in the same row (within 3x bbox size)
                if dx > avg_size * 3 and dy > avg_size * 3:
                    continue

                # Aligned: one axis displacement is small relative to bbox
                aligned_x = dx < avg_size * 0.5  # same column
                aligned_y = dy < avg_size * 0.5  # same row

                if aligned_x or aligned_y:
                    aligned_count += 1

            # 2+ aligned neighbors → likely parked
            if aligned_count >= 2:
                parked.add(i)

        return len(parked) / len(detections) if detections else 0.0

    def _traffic_density_level(
        self,
        density_grid: list[list[int]],
        counts: dict[str, int],
        parked_ratio: float = 0.0,
    ) -> str:
        """Classify traffic density: FLUIDO / MODERADO / DENSO / SATURADO.

        Discounts parked vehicles — a parking lot full of cars is not
        the same as dense moving traffic.
        """
        flat = [v for row in density_grid for v in row]
        if not flat:
            return "FLUIDO"

        # Discount parked vehicles from effective density
        active_factor = max(1.0 - parked_ratio, 0.15)
        avg = (sum(flat) / len(flat)) * active_factor
        peak = max(flat) * active_factor
        has_heavy = any(counts.get(c, 0) > 0 for c in config.HEAVY_VEHICLE_CLASSES)

        if avg >= 3 or (peak >= 6 and has_heavy):
            return "SATURADO"
        if avg >= 2 or peak >= 4:
            return "DENSO"
        if avg >= 1 or peak >= 2:
            return "MODERADO"
        return "FLUIDO"

    @staticmethod
    def _roundabout_level(occupancy_pct: float) -> str:
        """Classify roundabout occupancy: BAJA / MEDIA / ALTA / CRÍTICA."""
        if occupancy_pct >= 75:
            return "CRÍTICA"
        if occupancy_pct >= 50:
            return "ALTA"
        if occupancy_pct >= 25:
            return "MEDIA"
        return "BAJA"

    def _assess_risk(
        self,
        density_grid: list[list[int]],
        counts: dict[str, int],
        collisions: list[dict] | None = None,
    ) -> str:
        """Risk heuristic based on density, vehicle mix, and critical events."""

        # ── Incident-based escalation ──
        if collisions:
            if any(c["severity"] == "HIGH" for c in collisions):
                return "CRITICAL"
            if any(c["severity"] == "MEDIUM" for c in collisions):
                return "HIGH"

        # ── Density-based risk ──
        max_density = max(max(row) for row in density_grid) if density_grid else 0
        has_heavy = any(
            counts.get(cls, 0) > 0 for cls in config.HEAVY_VEHICLE_CLASSES
        )

        if max_density >= config.RISK_DENSITY_THRESHOLD and has_heavy:
            return "HIGH"
        if max_density >= config.RISK_DENSITY_THRESHOLD:
            return "MEDIUM"
        if has_heavy:
            return "MEDIUM"

        return "LOW"
