"""Centralized configuration for the traffic analysis system."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LEDGER_PATH = DATA_DIR / "ledger.jsonl"
MODELS_DIR = ROOT_DIR / "models"

# ── Model ──────────────────────────────────────────────────────────────
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
DEVICE = os.getenv("DEVICE", "cpu")

# ── Detection classes (COCO subset relevant to traffic) ────────────────
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Metrics ────────────────────────────────────────────────────────────
GRID_SIZE = 5
RISK_DENSITY_THRESHOLD = 3  # detections per cell to flag HIGH
HEAVY_VEHICLE_CLASSES = {"bus", "truck"}

# ── Heatmap weights by class ──────────────────────────────────────────
CLASS_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.6,
    "bicycle": 0.3,
    "bus": 1.5,
    "truck": 1.5,
}

# ── BSV Blockchain ────────────────────────────────────────────────────
BSV_NETWORK = os.getenv("BSV_NETWORK", "main")
BSV_PRIVATE_KEY = os.getenv("BSV_PRIVATE_KEY", "")
ARC_URL = os.getenv("ARC_URL", "https://arc.gorillapool.io")
WOC_BASE = (
    "https://api.whatsonchain.com/v1/bsv/test"
    if BSV_NETWORK == "testnet"
    else "https://api.whatsonchain.com/v1/bsv/main"
)

# ── Dataset IDs (Kaggle) ─────────────────────────────────────────────
DATASETS = {
    "uav_traffic": {
        "kaggle_id": "sakshamjn/traffic-images-captured-from-uavs",
        "format": "yolo",
        "classes": {0: "car", 1: "motorcycle"},
        "local_root": "data/traffic_aerial_images_for_vehicle_detection/dataset",
    },
    "roundabout": {
        "kaggle_id": "javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection",
        "format": "voc",
        "classes": {0: "car", 1: "cycle", 2: "truck", 3: "bus"},
        "local_root": "data/roundabout_aerial_images_for_vehicle_detection",
    },
}

# ── Collision detection ───────────────────────────────────────────────
COLLISION_IOU_THRESHOLD = 0.05     # IoU above this = potential collision
COLLISION_DISTANCE_THRESHOLD = 40  # pixels, centroid distance below this = proximity warning
ENABLE_PLATE_OCR = True            # attempt license plate OCR on collision

# ── Heatmap weights extended ──────────────────────────────────────────
CLASS_WEIGHTS["cycle"] = 0.3  # alias used by roundabout dataset
