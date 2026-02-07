"""FastAPI backend: /analyze and /verify endpoints."""

from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import JSONResponse

from src.detection import VehicleDetector
from src.metrics import TrafficAnalyzer
from src.hashing import build_evidence_record, compute_hash, canonical_json
from src.hashing.integrity import build_analysis_payload, compute_file_hash
from src.blockchain import get_blockchain_adapter
from src.visualization import draw_detections
from src.simulator import TrafficSimulator

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Traffic Aerial Analysis API", version="1.0.0")

# Lazy singletons
_detector: VehicleDetector | None = None
_analyzer: TrafficAnalyzer | None = None
_chain = None
_simulator: TrafficSimulator | None = None


def _get_detector():
    global _detector
    if _detector is None:
        _detector = VehicleDetector()
    return _detector


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = TrafficAnalyzer()
    return _analyzer


def _get_chain():
    global _chain
    if _chain is None:
        _chain = get_blockchain_adapter()
    return _chain


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    dataset_id: str = Query("upload", description="Dataset identifier"),
    is_roundabout: bool = Query(False, description="Roundabout scene?"),
):
    """Analyze an aerial image: detect vehicles, compute metrics, register on blockchain."""
    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    h, w = img.shape[:2]
    scene_id = image.filename or "unknown"

    # Detect
    detector = _get_detector()
    detections = detector.detect(img)

    # Metrics
    analyzer = _get_analyzer()
    metrics = analyzer.analyze(detections, h, w, is_roundabout=is_roundabout)

    # Build canonical payload
    payload = build_analysis_payload(
        scene_id=scene_id,
        dataset_id=dataset_id,
        counts=metrics.counts,
        total_vehicles=metrics.total_vehicles,
        density_grid=metrics.density_grid,
        occupancy_pct=metrics.occupancy_pct,
        zone_occupancy=metrics.zone_occupancy,
        risk_level=metrics.risk_level,
        model_version=detector.model_version,
        is_roundabout=metrics.is_roundabout,
        roundabout_occupancy_pct=metrics.roundabout_occupancy_pct,
    )

    # Hash & evidence
    analysis_hash = compute_hash(payload)
    evidence = build_evidence_record(payload)

    # Register on blockchain
    chain = _get_chain()
    tx_result = chain.register(evidence)

    return {
        "analysis": payload,
        "analysis_hash": analysis_hash,
        "evidence": evidence,
        "blockchain": tx_result,
        "detections_count": len(detections),
    }


@app.get("/verify")
async def verify(analysis_hash: str = Query(..., description="SHA-256 hash to verify")):
    """Verify an analysis hash against the blockchain/ledger."""
    chain = _get_chain()
    record = chain.verify(analysis_hash)
    if record:
        return {
            "verified": True,
            "record": record,
        }
    return {
        "verified": False,
        "message": "No record found for this hash",
    }


@app.get("/records")
async def list_records(limit: int = Query(20, ge=1, le=100)):
    """List recent evidence records."""
    chain = _get_chain()
    return chain.list_records(limit=limit)


def _get_simulator():
    global _simulator
    if _simulator is None:
        _simulator = TrafficSimulator()
    return _simulator


@app.post("/simulate")
async def simulate(
    date: str = Query(..., description="Date YYYY-MM-DD"),
    time: str = Query(..., description="Time HH:MM"),
    scene_type: str = Query("urban_road", description="urban_road | roundabout | highway"),
    scene_id: str = Query("simulated_scene", description="Scene identifier"),
    total_vehicles: int = Query(0, description="Override total (0=auto)"),
    register: bool = Query(False, description="Register on blockchain?"),
):
    """What-if traffic simulation for a given date/time/scene."""
    from datetime import datetime as dt
    try:
        sim_dt = dt.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid date/time format"})

    sim = _get_simulator()
    result = sim.simulate(
        sim_datetime=sim_dt,
        scene_type=scene_type,
        scene_id=scene_id,
        override_total=total_vehicles if total_vehicles > 0 else None,
    )

    sim_hash = compute_hash(result)
    evidence = build_evidence_record(result)

    response = {
        "simulation": result,
        "simulation_hash": sim_hash,
        "evidence": evidence,
    }

    if register:
        chain = _get_chain()
        tx_result = chain.register(evidence)
        response["blockchain"] = tx_result

    return response


@app.get("/health")
async def health():
    return {"status": "ok", "model": _get_detector().model_version}
