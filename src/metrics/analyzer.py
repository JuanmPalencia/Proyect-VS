"""Cálculo de métricas de tráfico: conteos, densidad, ocupación, riesgo."""

from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np
import config
from src.detection.detector import Detection
from src.metrics.collision import detect_collisions


# ──────────────────────────────────────────────────────────────────────
# Contenedor de datos
# ──────────────────────────────────────────────────────────────────────

@dataclass
class TrafficMetrics:
    """Todas las métricas calculadas para una escena."""
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
# Analizador
# ──────────────────────────────────────────────────────────────────────

class TrafficAnalyzer:
    """Calcula métricas de tráfico a partir de resultados de detección."""

    def __init__(self, grid_size: int | None = None):
        self.grid_size = grid_size or config.GRID_SIZE

    def analyze(
        self,
        detections: list[Detection],
        img_h: int,
        img_w: int,
        is_roundabout: bool = False,
    ) -> TrafficMetrics:
        """Pipeline de análisis completo."""

        # Filtrado por confianza 
        conf_thresh = config.CONFIDENCE_THRESHOLD
        detections_f = [d for d in detections if d.confidence >= conf_thresh]

        # Métricas básicas
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

        # Detección de estacionados
        parked_ratio = self._estimate_parked_ratio(detections_f)

        # Nivel de densidad de tráfico (descontando vehículos estacionados)
        t_density = self._traffic_density_level(density, counts, parked_ratio)

        # Nivel de rotonda
        roundabout_lvl = (
            self._roundabout_level(roundabout_occ) if roundabout_occ is not None else None
        )

        # Detección de incidentes
        collisions = detect_collisions(detections_f)

        # Evaluación de riesgo
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

    # Métricas individuales

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
        """Cuadrícula NxN contando centroides por celda."""
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
        """Porcentaje del área de imagen cubierta por cajas delimitadoras (heurística de suma)."""
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
        """Divide la imagen en 3 zonas horizontales y calcula % de vehículos."""
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
        """% de vehículos dentro de región circular central (heurística de rotonda)."""
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
        """Estima la fracción de vehículos que aparentan estar estacionados.

        Heurística: vehículos alineados en fila con espaciado regular y
        tamaños de caja similares probablemente estén estacionados (patrón de estacionamiento aéreo).
        Un vehículo está "estacionado" si tiene 2+ vecinos alineados de tamaño similar.
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

                # Tamaño similar (dentro del 40%)
                size_ratio = min(size_i, size_j) / max(size_i, size_j)
                if size_ratio < 0.6:
                    continue

                dx = abs(d.centroid[0] - o.centroid[0])
                dy = abs(d.centroid[1] - o.centroid[1])
                avg_size = (size_i + size_j) / 2

                # Lo suficientemente cerca para estar en la misma fila (dentro de 3x tamaño de caja)
                if dx > avg_size * 3 and dy > avg_size * 3:
                    continue

                # Alineado: desplazamiento de un eje es pequeño relativo a la caja
                aligned_x = dx < avg_size * 0.5  # misma columna
                aligned_y = dy < avg_size * 0.5  # misma fila

                if aligned_x or aligned_y:
                    aligned_count += 1

            # 2+ vecinos alineados → probablemente estacionado
            if aligned_count >= 2:
                parked.add(i)

        return len(parked) / len(detections) if detections else 0.0

    def _traffic_density_level(
        self,
        density_grid: list[list[int]],
        counts: dict[str, int],
        parked_ratio: float = 0.0,
    ) -> str:
        """Clasifica densidad de tráfico: FLUIDO / MODERADO / DENSO / SATURADO.

        Descuenta vehículos estacionados — un estacionamiento lleno de autos no es
        lo mismo que tráfico denso en movimiento.
        """
        flat = [v for row in density_grid for v in row]
        if not flat:
            return "FLUIDO"

        # Descontar vehículos estacionados de la densidad efectiva
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
        """Clasifica ocupación de rotonda: BAJA / MEDIA / ALTA / CRÍTICA."""
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
        """Heurística de riesgo basada en densidad, mezcla de vehículos y eventos críticos."""

        # ── Escalamiento basado en incidentes ──
        if collisions:
            if any(c["severity"] == "HIGH" for c in collisions):
                return "CRITICAL"
            if any(c["severity"] == "MEDIUM" for c in collisions):
                return "HIGH"

        # ── Riesgo basado en densidad ──
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
