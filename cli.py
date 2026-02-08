# CLI entry point para análisis por lotes y verificación.
# Este script permite usar el sistema sin interfaz gráfica (Streamlit) ni API (FastAPI).

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Aseguramos que la raíz del proyecto está en el path de Python.
# Esto permite ejecutar: `python cli.py` sin problemas de imports relativos.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.detection import VehicleDetector
from src.metrics import TrafficAnalyzer
from src.hashing.integrity import (
    build_analysis_payload,
    build_evidence_record,
    canonical_json,
    compute_hash,
    compute_file_hash,
)
from src.blockchain import get_blockchain_adapter

# Logger con formato compacto para consola
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Analiza una imagen o un directorio completo de imágenes.
#   - Si `input` es un fichero analiza solo esa imagen.
#   - Si `input` es un directorio→ busca imágenes por extensión y procesa todas.
#   - Opcional: registra cada evidencia en blockchain/ledger con `--register`.
#   - Guarda un JSON final con todos los resultados en `--output`.
def cmd_analyze(args):

    import cv2  # import local para no cargar OpenCV si no se usa este comando

    # Inicializamos los componentes principales.
    # Se crean por comando (no singleton) porque el CLI suele ser ejecución puntual.
    detector = VehicleDetector()
    analyzer = TrafficAnalyzer()
    chain = get_blockchain_adapter()

    # Resolver rutas de entrada
    image_paths = []
    target = Path(args.input)

    # Caso 1: archivo
    if target.is_file():
        image_paths.append(target)

    # Caso 2: carpeta: buscamos extensiones habituales
    elif target.is_dir():
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            image_paths.extend(sorted(target.glob(ext)))

    # Caso 3: ruta inválida
    else:
        logger.error("Input not found: %s", args.input)
        return

    logger.info("Processing %d image(s)...", len(image_paths))
    results = []

    # Procesado por imagen
    for img_path in image_paths:
        # Leemos la imagen desde disco con OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        # Dimensiones necesarias para métricas
        h, w = img.shape[:2]

        # 1) Detección de vehículos (modelo)
        detections = detector.detect(img)

        # 2) Métricas de tráfico
        metrics = analyzer.analyze(detections, h, w, is_roundabout=args.roundabout)

        # 3) Payload canónico: estructura estable para hashing/verificación
        payload = build_analysis_payload(
            scene_id=img_path.name,
            dataset_id=args.dataset_id,
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

        # 4) Hash del análisis + evidencia
        # - analysis_hash: huella SHA-256 del payload
        # - evidence: registro que potencialmente se sube al ledger
        analysis_hash = compute_hash(payload)

        # En CLI añadimos también el hash del fichero de imagen para trazar el input original.
        evidence = build_evidence_record(payload, image_hash=compute_file_hash(str(img_path)))

        # 5) Registro opcional en blockchain/ledger
        if args.register:
            tx = chain.register(evidence)
            logger.info("Registered %s → %s", img_path.name, tx.get("tx_id", tx.get("evidence_id")))

        # Guardamos una entrada por imagen para exportar a JSON
        result_entry = {
            "analysis": payload,
            "analysis_hash": analysis_hash,
            "evidence": evidence,
        }
        results.append(result_entry)

        # Log “humano” para ver progreso rápido en consola
        logger.info(
            "  %s | %d vehicles | risk=%s | hash=%s...",
            img_path.name,
            metrics.total_vehicles,
            metrics.risk_level,
            analysis_hash[:16],
        )

    # Guardado a disco
    # Creamos carpetas si no existen
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Escribimos el JSON con indent para poder revisarlo fácilmente
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    logger.info("Results saved to %s", output_path)

# Verifica un hash contra el blockchain/ledger y devuelve VERIFIED + el record asociado si 
# existe y NOT FOUND + exit code 1 si no existe registro. El record se muestra en formato
# JSON para facilitar su uso en scripts o CI.
def cmd_verify(args):
    chain = get_blockchain_adapter()
    record = chain.verify(args.hash)

    if record:
        print("VERIFIED")
        print(json.dumps(record, indent=2))
    else:
        print("NOT FOUND")
        # Salida con código 1 para que scripts/CI puedan detectar fallo de verificación
        sys.exit(1)

# Procesa un dataset completo (modo batch)
def cmd_batch_dataset(args):
    from src.datasets import load_dataset  # import local para evitar dependencias si no se usa este comando
    import cv2

    detector = VehicleDetector()
    analyzer = TrafficAnalyzer()
    chain = get_blockchain_adapter()

    # Cargamos dataset
    ds = load_dataset(args.dataset_id, root=args.root)

    # Si el dataset es "roundabout", se analiza como rotonda
    is_roundabout = args.dataset_id == "roundabout"

    results = []

    # Recorremos las muestras del dataset
    for i, sample in enumerate(ds):
        if args.limit and i >= args.limit:
            break

        img = cv2.imread(str(sample.image_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        detections = detector.detect(img)
        metrics = analyzer.analyze(detections, h, w, is_roundabout=is_roundabout)

        payload = build_analysis_payload(
            scene_id=sample.image_path.name,
            dataset_id=sample.dataset_id,
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

        analysis_hash = compute_hash(payload)

        # En batch no se incluye file_hash
        evidence = build_evidence_record(payload)

        if args.register:
            chain.register(evidence)

        results.append({"analysis": payload, "analysis_hash": analysis_hash, "evidence": evidence})

        logger.info(
            "[%d] %s | %d vehicles | %s",
            i + 1,
            sample.image_path.name,
            metrics.total_vehicles,
            metrics.risk_level,
        )

    # Guardamos resultados finales
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)

    logger.info("Processed %d images → %s", len(results), output_path)

# Punto de entrada del CLI.
#   Define subcomandos y argumentos:
#   - analyze: analiza imágenes sueltas o carpetas
#   - verify: verifica un hash
#   - batch: procesa datasets completos
def main():
    parser = argparse.ArgumentParser(
        description="Traffic Aerial Analysis - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # Subcomando: analyze 
    p_analyze = sub.add_parser("analyze", help="Analyze image(s)")
    p_analyze.add_argument("input", help="Image file or directory")
    p_analyze.add_argument("-d", "--dataset-id", default="upload")
    p_analyze.add_argument("-o", "--output", default="data/results.json")
    p_analyze.add_argument("--roundabout", action="store_true")
    p_analyze.add_argument("--register", action="store_true", help="Register on blockchain")
    p_analyze.set_defaults(func=cmd_analyze)

    # Subcomando: verify 
    p_verify = sub.add_parser("verify", help="Verify a hash")
    p_verify.add_argument("hash", help="SHA-256 hash to verify")
    p_verify.set_defaults(func=cmd_verify)

    # Subcomando: batch 
    p_batch = sub.add_parser("batch", help="Process a Kaggle dataset")
    p_batch.add_argument("dataset_id", choices=["uav_traffic", "roundabout"])
    p_batch.add_argument("--root", default=None, help="Local dataset path (skip download)")
    p_batch.add_argument("-o", "--output", default="data/batch_results.json")
    p_batch.add_argument("-n", "--limit", type=int, default=None)
    p_batch.add_argument("--register", action="store_true")
    p_batch.set_defaults(func=cmd_batch_dataset)

    # Parseo de argumentos (si no hay subcomando, mostramos ayuda)
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    # Ejecutamos la función asociada al subcomando
    args.func(args)


if __name__ == "__main__":
    main()
