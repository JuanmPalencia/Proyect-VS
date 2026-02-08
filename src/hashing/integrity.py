"""Construcción de hash determinístico y registro de evidencia."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _json_default(obj: Any) -> Any:
    """Maneja tipos numpy y otros objetos no nativos para serialización JSON."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def canonical_json(data: dict[str, Any]) -> str:
    """Serializa dict a una cadena JSON canónica determinística.

    - Claves ordenadas recursivamente
    - Sin espacios en blanco (separadores compactos)
    - Ensure_ascii para reproducibilidad multiplataforma
    - Maneja tipos numpy vía manejador default personalizado
    """
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, default=_json_default,
    )


def compute_hash(data: dict[str, Any]) -> str:
    """Hash SHA-256 de la representación JSON canónica.

    Normaliza los datos vía ida y vuelta JSON primero para asegurar que
    los tipos numpy, precisión de floats, etc. produzcan el mismo hash
    ya sea calculado desde el dict original o desde JSON re-parseado.
    """
    normalized = json.loads(canonical_json(data))
    canonical = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """Hash SHA-256 del contenido binario de un archivo."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def build_analysis_payload(
    scene_id: str,
    dataset_id: str,
    counts: dict[str, int],
    total_vehicles: int,
    density_grid: list[list[int]],
    occupancy_pct: float,
    zone_occupancy: dict[str, float],
    risk_level: str,
    model_version: str,
    is_roundabout: bool = False,
    roundabout_occupancy_pct: float | None = None,
    collision_count: int = 0,
    collisions: list[dict] | None = None,
    geo: dict | None = None,
) -> dict[str, Any]:
    """Construye el payload canónico de análisis (antes del hash)."""
    payload: dict[str, Any] = {
        "scene_id": scene_id,
        "dataset_id": dataset_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "counts": counts,
        "total_vehicles": total_vehicles,
        "density_grid": density_grid,
        "occupancy_pct": occupancy_pct,
        "zone_occupancy": zone_occupancy,
        "risk_level": risk_level,
        "is_roundabout": is_roundabout,
        "collision_count": collision_count,
    }
    if roundabout_occupancy_pct is not None:
        payload["roundabout_occupancy_pct"] = roundabout_occupancy_pct
    if collisions:
        payload["collisions"] = collisions
    if geo:
        payload["geo"] = geo
    return payload


def build_evidence_record(
    analysis_payload: dict[str, Any],
    image_hash: str | None = None,
) -> dict[str, Any]:
    """Construye el registro de evidencia para enviar a blockchain.

    Contiene el analysis_hash más metadatos para búsqueda.
    """
    analysis_hash = compute_hash(analysis_payload)
    record = {
        "analysis_hash": analysis_hash,
        "scene_id": analysis_payload["scene_id"],
        "dataset_id": analysis_payload["dataset_id"],
        "timestamp_utc": analysis_payload["timestamp_utc"],
        "model_version": analysis_payload["model_version"],
    }
    if image_hash:
        record["image_hash"] = image_hash
    return record


def verify_integrity(analysis_payload: dict[str, Any], expected_hash: str) -> bool:
    """Recalcula el hash y verifica que coincida con el almacenado."""
    return compute_hash(analysis_payload) == expected_hash
