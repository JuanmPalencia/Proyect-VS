"""Deterministic hashing and evidence record construction."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any


def canonical_json(data: dict[str, Any]) -> str:
    """Serialize dict to a deterministic canonical JSON string.

    - Keys sorted recursively
    - No whitespace (compact separators)
    - Ensure_ascii for cross-platform reproducibility
    """
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_hash(data: dict[str, Any]) -> str:
    """SHA-256 hash of the canonical JSON representation."""
    canonical = canonical_json(data)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(file_path: str) -> str:
    """SHA-256 hash of a file's binary content."""
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
    geo: dict | None = None,
) -> dict[str, Any]:
    """Build the canonical analysis payload (before hashing)."""
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
    }
    if roundabout_occupancy_pct is not None:
        payload["roundabout_occupancy_pct"] = roundabout_occupancy_pct
    if geo:
        payload["geo"] = geo
    return payload


def build_evidence_record(
    analysis_payload: dict[str, Any],
    image_hash: str | None = None,
) -> dict[str, Any]:
    """Build the evidence record to be sent to blockchain.

    Contains the analysis_hash plus metadata for lookup.
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
    """Recompute hash and verify it matches the stored one."""
    return compute_hash(analysis_payload) == expected_hash
