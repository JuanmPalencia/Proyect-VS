"""Detección de colisiones / eventos críticos entre vehículos basada en análisis de cajas delimitadoras."""

from __future__ import annotations

import math
from typing import Any

import config
from src.detection.detector import Detection


# ──────────────────────────────────────────────────────────────────────
# Funciones auxiliares
# ──────────────────────────────────────────────────────────────────────

def _sanitize_bbox(bbox: list[float]) -> list[float]:
    """Asegura que bbox sea válido [x1,y1,x2,y2] con x2>x1 y y2>y1."""
    x1, y1, x2, y2 = bbox
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [float(x1), float(y1), float(x2), float(y2)]


def compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
    """Calcula intersección sobre unión entre dos cajas [x1,y1,x2,y2]."""
    x1_1, y1_1, x2_1, y2_1 = _sanitize_bbox(bbox1)
    x1_2, y1_2, x2_2, y2_2 = _sanitize_bbox(bbox2)

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter_area = inter_w * inter_h

    bbox1_area = max(0.0, (x2_1 - x1_1)) * max(0.0, (y2_1 - y1_1))
    bbox2_area = max(0.0, (x2_2 - x1_2)) * max(0.0, (y2_2 - y1_2))
    union_area = bbox1_area + bbox2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def compute_centroid_distance(det1: Detection, det2: Detection) -> float:
    """Distancia euclidiana entre centroides de dos detecciones."""
    cx1, cy1 = det1.centroid
    cx2, cy2 = det2.centroid
    return math.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)


def _bbox_wh(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = _sanitize_bbox(bbox)
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _bbox_diagonal(bbox: list[float]) -> float:
    w, h = _bbox_wh(bbox)
    return math.sqrt(w * w + h * h)


def _classify_severity(iou: float, dist_norm: float | None) -> str:
    """Clasifica severidad basada en IoU y distancia normalizada."""
    if iou >= config.COLLISION_IOU_HIGH:
        return "HIGH"
    if iou >= config.COLLISION_IOU_THRESHOLD:
        return "MEDIUM"
    if dist_norm is not None and dist_norm <= config.COLLISION_DISTANCE_NORM_THRESHOLD:
        return "WARNING"
    return "WARNING"


def _event_type(iou: float, triggered_by_iou: bool) -> str:
    """Clasifica tipo de evento para mejor interpretabilidad en el reporte/demo."""
    if iou >= config.COLLISION_IOU_HIGH:
        return "POSSIBLE_COLLISION"
    if triggered_by_iou:
        return "OVERLAP_CONFLICT"
    return "NEAR_MISS"


def _looks_like_same_trajectory_pair(det_a: Detection, det_b: Detection) -> bool:
    """
    Heurística para suprimir falsos positivos de NEAR_MISS en imágenes únicas:
    - Vehículos alineados (uno detrás de otro) / estacionados en fila / cola detenida.
    Sin movimiento, aproximamos "misma trayectoria" por geometría:
      * desplazamiento lateral muy pequeño (dx pequeño) + gran separación longitudinal (dy grande)
      O
      * desplazamiento vertical muy pequeño (dy pequeño) + gran separación horizontal (dx grande)
    """
    ax, ay = det_a.centroid
    bx, by = det_b.centroid
    dx = abs(bx - ax)
    dy = abs(by - ay)

    wa, ha = _bbox_wh(det_a.bbox)
    wb, hb = _bbox_wh(det_b.bbox)
    mean_w = (wa + wb) / 2.0 if (wa + wb) > 0 else 1.0
    mean_h = (ha + hb) / 2.0 if (ha + hb) > 0 else 1.0

    # Factores ajustables (con valores seguros por defecto)
    # umbral de desplazamiento lateral "pequeño"
    same_lane_x = getattr(config, "NEARMISS_SAME_LANE_X_FACTOR", 0.55)   # * mean_w
    same_lane_y = getattr(config, "NEARMISS_SAME_LANE_Y_FACTOR", 0.55)   # * mean_h
    # umbral de separación longitudinal "grande"
    sep_y = getattr(config, "NEARMISS_SEPARATION_Y_FACTOR", 1.25)        # * mean_h
    sep_x = getattr(config, "NEARMISS_SEPARATION_X_FACTOR", 1.25)        # * mean_w

    # Caso 1: apilados verticalmente (mismo carril / cola)
    stacked = (dx <= same_lane_x * mean_w) and (dy >= sep_y * mean_h)
    # Caso 2: apilados horizontalmente (menos común, pero ayuda con algunas orientaciones de cámara)
    side_by_side_queue = (dy <= same_lane_y * mean_h) and (dx >= sep_x * mean_w)

    return stacked or side_by_side_queue


def _should_keep_near_miss(iou: float, dist_norm: float | None, det_a: Detection, det_b: Detection) -> bool:
    """
    Decide si un NEAR_MISS activado por distancia es significativo.
    Lo mantenemos si está muy cerca O si no parece misma-trayectoria/cola estacionada.

    Este es el filtro clave que reduce falsos positivos de autos estacionados.
    """
    if dist_norm is None:
        # si no tenemos distancia normalizada, mantener tal cual (comportamiento de respaldo)
        return True

    # Umbral estricto de near-miss (solo para situaciones de "misma trayectoria")
    strict = getattr(config, "NEARMISS_STRICT_DISTANCE_NORM", 0.65)

    if _looks_like_same_trajectory_pair(det_a, det_b):
        # Solo mantener si extremadamente cerca (de otro modo es probablemente estacionado/cola)
        return dist_norm <= strict

    return True


# ──────────────────────────────────────────────────────────────────────
# Principal
# ──────────────────────────────────────────────────────────────────────

def detect_collisions(
    detections: list[Detection],
    iou_threshold: float | None = None,
    distance_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """
    Detecta eventos críticos potenciales entre vehículos.

    Se activa cuando:
    - IoU > iou_threshold (solapamiento de cajas), O
    - Distancia de centroide normalizada < COLLISION_DISTANCE_NORM_THRESHOLD (si está habilitado), O
    - Distancia de centroide < distance_threshold (respaldo en píxeles)

    Devuelve lista de registros de eventos con vehículos involucrados y severidad.
    """
    iou_thresh = iou_threshold or config.COLLISION_IOU_THRESHOLD
    dist_thresh = distance_threshold or config.COLLISION_DISTANCE_THRESHOLD
    conf_thresh = getattr(config, "CONFIDENCE_THRESHOLD", 0.0)

    collisions: list[dict[str, Any]] = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            det_a = detections[i]
            det_b = detections[j]

            # Control de confianza (reduce falsos positivos)
            if min(det_a.confidence, det_b.confidence) < conf_thresh:
                continue

            iou = compute_iou(det_a.bbox, det_b.bbox)
            distance = compute_centroid_distance(det_a, det_b)

            # Distancia normalizada (independiente de escala)
            dist_norm = None
            if getattr(config, "COLLISION_USE_NORMALIZED_DISTANCE", False):
                diag_a = _bbox_diagonal(det_a.bbox)
                diag_b = _bbox_diagonal(det_b.bbox)
                mean_diag = (diag_a + diag_b) / 2.0
                if mean_diag > 0:
                    dist_norm = distance / mean_diag

            triggered_by_iou = False
            triggered_by_dist = False
            trigger = False

            # Condición de activación
            if iou > iou_thresh:
                trigger = True
                triggered_by_iou = True
            elif dist_norm is not None:
                trigger = dist_norm < config.COLLISION_DISTANCE_NORM_THRESHOLD
                triggered_by_dist = trigger
            else:
                trigger = distance < dist_thresh
                triggered_by_dist = trigger

            # Filtro extra: suprimir NEAR_MISS de bajo valor (estacionados/cola)
            if trigger and triggered_by_dist and (iou < iou_thresh):
                if not _should_keep_near_miss(iou, dist_norm, det_a, det_b):
                    continue

            if trigger:
                severity = _classify_severity(iou, dist_norm)
                event_type = _event_type(iou, triggered_by_iou)

                collisions.append({
                    "type": event_type,
                    "vehicle_a_idx": i,
                    "vehicle_b_idx": j,
                    "vehicle_a_class": det_a.class_name,
                    "vehicle_b_class": det_b.class_name,
                    "vehicle_a_centroid": det_a.centroid,
                    "vehicle_b_centroid": det_b.centroid,
                    "iou": round(iou, 4),
                    "distance": round(distance, 2),
                    "distance_norm": round(dist_norm, 3) if dist_norm is not None else None,
                    "severity": severity,
                })

    # Ordenar por severidad (HIGH primero)
    severity_order = {"HIGH": 0, "MEDIUM": 1, "WARNING": 2}
    collisions.sort(key=lambda c: severity_order.get(c["severity"], 3))

    return collisions
