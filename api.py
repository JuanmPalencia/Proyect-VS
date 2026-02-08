# FastAPI backend: endpoints /analyze, /verify, /records, /simulate y /health.
#
# Este módulo expone una API sencilla para:
# - Analizar imágenes aéreas (detección + métricas + evidencia verificable).
# - Verificar hashes en el ledger/blockchain.
# - Listar registros recientes.
# - Simular escenarios (what-if) sin necesidad de imagen.

from __future__ import annotations

import logging
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

# Configuración básica de logs
logging.basicConfig(level=logging.INFO)

# Instancia principal de FastAPI: define el título y versión visibles en /docs (Swagger)
app = FastAPI(title="Traffic Aerial Analysis API", version="1.0.0")

# Singletons "lazy" (se crean solo cuando se usan)
# Cargar modelos/recursos pesados al arrancar puede ser lento.
# Con lazy init, el primer request "paga" el coste, y luego se reutiliza.
_detector: VehicleDetector | None = None
_analyzer: TrafficAnalyzer | None = None
_chain = None
_simulator: TrafficSimulator | None = None

# Devuelve una única instancia de VehicleDetector. Se inicializa la primera vez que se llama
# para evitar cargar el modelo al arrancar.
def _get_detector():
    global _detector
    if _detector is None:
        _detector = VehicleDetector()
    return _detector

# Devuelve una única instancia de TrafficAnalyzer. Centraliza el cálculo de métricas a partir
# de las detecciones.
def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = TrafficAnalyzer()
    return _analyzer

# Devuelve el adaptador de blockchain/ledger. En función de la configuración, puede registrar
# en BSV o en un ledger local.
def _get_chain():
    global _chain
    if _chain is None:
        _chain = get_blockchain_adapter()
    return _chain

# Analiza una imagen aérea: detecta vehículos, calcula métricas y genera evidencia en un flujo.
@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    dataset_id: str = Query("upload", description="Dataset identifier"),
    is_roundabout: bool = Query(False, description="Roundabout scene?"),
):

    # 1) Lectura y decodificación de la imagen
    # UploadFile llega como bytes; lo convertimos a un array de numpy para OpenCV.
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Si la imagen no se puede decodificar, devolvemos 400 (petición inválida).
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image"})

    # Definimos dimensiones de la imagen (necesarias para cálculos de ocupación/densidad)
    h, w = img.shape[:2]

    # Identificador lógico de escena: por defecto usamos el nombre de archivo
    scene_id = image.filename or "unknown"

    # 2) Detección de vehículos
    # Usamos el detector singleton (carga el modelo una sola vez).
    detector = _get_detector()
    detections = detector.detect(img)

    # 3) Métricas de tráfico
    # El analyzer traduce detecciones
    analyzer = _get_analyzer()
    metrics = analyzer.analyze(detections, h, w, is_roundabout=is_roundabout)

    # 4) Payload canónico (estable)
    # Construimos un diccionario con el mínimo conjunto de campos relevantes.
    # Importante: el payload debe ser determinista para que el hash sea reproducible.
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

    # 5) Hash + evidencia
    # analysis_hash: huella digital SHA-256 del payload (integridad)
    # evidence: estructura completa que se registra (incluye hash, timestamp, etc.)
    analysis_hash = compute_hash(payload)
    evidence = build_evidence_record(payload)

    # 6) Registro en blockchain/ledger
    # El adaptador devuelve un resultado con info
    chain = _get_chain()
    tx_result = chain.register(evidence)

    return {
        "analysis": payload,
        "analysis_hash": analysis_hash,
        "evidence": evidence,
        "blockchain": tx_result,
        "detections_count": len(detections),
    }

# Verifica un hash contra el registro (blockchain/ledger). Si existe un registro para ese
# hash, devolvemos el record asociado.
@app.get("/verify")
async def verify(analysis_hash: str = Query(..., description="SHA-256 hash to verify")):
    chain = _get_chain()

    # Consultamos el ledger por hash; si está registrado devolvemos el detalle.
    record = chain.verify(analysis_hash)
    if record:
        return {
            "verified": True,
            "record": record,
        }

    # Si no aparece, se considera no verificado (o aún no registrado).
    return {
        "verified": False,
        "message": "No record found for this hash",
    }

# Lista registros recientes del ledger. limit controla cuántos elementos se devuelven (1..100).
@app.get("/records")
async def list_records(limit: int = Query(20, ge=1, le=100)):
    chain = _get_chain()
    return chain.list_records(limit=limit)

# Devuelve una única instancia del simulador (what-if). El simulador permite generar escenarios
# sin imagen de entrada.
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
    # Simulación what-if de tráfico para una fecha/hora y tipo de escena.
    # - Si total_vehicles > 0, fuerza el número de vehículos.
    # - Si register=True, registra la evidencia del escenario simulado.
    
    from datetime import datetime as dt

    # Parseamos fecha/hora; si el formato no cuadra, devolvemos 400.
    try:
        sim_dt = dt.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
    except ValueError:
        return JSONResponse(status_code=400, content={"error": "Invalid date/time format"})

    # Ejecutamos la simulación con el motor del simulador.
    sim = _get_simulator()
    result = sim.simulate(
        sim_datetime=sim_dt,
        scene_type=scene_type,
        scene_id=scene_id,
        override_total=total_vehicles if total_vehicles > 0 else None,
    )

    # Igual que en /analyze: hash para integridad + evidencia para registro.
    sim_hash = compute_hash(result)
    evidence = build_evidence_record(result)

    # Respuesta base: devolvemos simulación, hash y evidencia (aunque no se registre).
    response = {
        "simulation": result,
        "simulation_hash": sim_hash,
        "evidence": evidence,
    }

    # Registro opcional: útil si quieres trazabilidad también de escenarios what-if.
    if register:
        chain = _get_chain()
        tx_result = chain.register(evidence)
        response["blockchain"] = tx_result

    return response

# Endpoint simple de health-check (para Docker/K8s/monitorización). Incluye también la versión
# del modelo cargado para diagnóstico rápido.
@app.get("/health")
async def health():
    return {"status": "ok", "model": _get_detector().model_version}
