"""CLI entry point for batch analysis and verification."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root in path
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def cmd_analyze(args):
    """Analyze a single image or directory of images."""
    import cv2

    detector = VehicleDetector()
    analyzer = TrafficAnalyzer()
    chain = get_blockchain_adapter()

    image_paths = []
    target = Path(args.input)
    if target.is_file():
        image_paths.append(target)
    elif target.is_dir():
        for ext in ("*.jpg", "*.png", "*.jpeg"):
            image_paths.extend(sorted(target.glob(ext)))
    else:
        logger.error("Input not found: %s", args.input)
        return

    logger.info("Processing %d image(s)...", len(image_paths))
    results = []

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Cannot read %s, skipping", img_path)
            continue

        h, w = img.shape[:2]
        detections = detector.detect(img)
        metrics = analyzer.analyze(detections, h, w, is_roundabout=args.roundabout)

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

        analysis_hash = compute_hash(payload)
        evidence = build_evidence_record(payload, image_hash=compute_file_hash(str(img_path)))

        if args.register:
            tx = chain.register(evidence)
            logger.info("Registered %s → %s", img_path.name, tx.get("tx_id", tx.get("evidence_id")))

        result_entry = {
            "analysis": payload,
            "analysis_hash": analysis_hash,
            "evidence": evidence,
        }
        results.append(result_entry)
        logger.info("  %s | %d vehicles | risk=%s | hash=%s...",
                     img_path.name, metrics.total_vehicles, metrics.risk_level, analysis_hash[:16])

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    logger.info("Results saved to %s", output_path)


def cmd_verify(args):
    """Verify a hash against the blockchain/ledger."""
    chain = get_blockchain_adapter()
    record = chain.verify(args.hash)
    if record:
        print("VERIFIED")
        print(json.dumps(record, indent=2))
    else:
        print("NOT FOUND")
        sys.exit(1)


def cmd_batch_dataset(args):
    """Process a full Kaggle dataset (download + analyze)."""
    from src.datasets import load_dataset
    import cv2

    detector = VehicleDetector()
    analyzer = TrafficAnalyzer()
    chain = get_blockchain_adapter()

    ds = load_dataset(args.dataset_id, root=args.root)
    is_roundabout = args.dataset_id == "roundabout"
    results = []

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
        evidence = build_evidence_record(payload)

        if args.register:
            chain.register(evidence)

        results.append({"analysis": payload, "analysis_hash": analysis_hash, "evidence": evidence})
        logger.info("[%d] %s | %d vehicles | %s", i + 1, sample.image_path.name,
                     metrics.total_vehicles, metrics.risk_level)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=True)
    logger.info("Processed %d images → %s", len(results), output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Traffic Aerial Analysis - CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze image(s)")
    p_analyze.add_argument("input", help="Image file or directory")
    p_analyze.add_argument("-d", "--dataset-id", default="upload")
    p_analyze.add_argument("-o", "--output", default="data/results.json")
    p_analyze.add_argument("--roundabout", action="store_true")
    p_analyze.add_argument("--register", action="store_true", help="Register on blockchain")
    p_analyze.set_defaults(func=cmd_analyze)

    # verify
    p_verify = sub.add_parser("verify", help="Verify a hash")
    p_verify.add_argument("hash", help="SHA-256 hash to verify")
    p_verify.set_defaults(func=cmd_verify)

    # batch
    p_batch = sub.add_parser("batch", help="Process a Kaggle dataset")
    p_batch.add_argument("dataset_id", choices=["uav_traffic", "roundabout"])
    p_batch.add_argument("--root", default=None, help="Local dataset path (skip download)")
    p_batch.add_argument("-o", "--output", default="data/batch_results.json")
    p_batch.add_argument("-n", "--limit", type=int, default=None)
    p_batch.add_argument("--register", action="store_true")
    p_batch.set_defaults(func=cmd_batch_dataset)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
