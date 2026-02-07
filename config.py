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
# YOLO (COCO) class IDs → class name
VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# ── Class aliases (cross-dataset consistency) ──────────────────────────
# Some datasets use different labels (e.g., "cycle" instead of "bicycle")
CLASS_ALIASES = {
    "cycle": "bicycle",
}

# ── Metrics ────────────────────────────────────────────────────────────
GRID_SIZE = 5
RISK_DENSITY_THRESHOLD = 3  # detections per cell to flag MEDIUM/HIGH
HEAVY_VEHICLE_CLASSES = {"bus", "truck"}

# ── Heatmap weights by class ──────────────────────────────────────────
CLASS_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.6,
    "bicycle": 0.3,
    "bus": 1.5,
    "truck": 1.5,
}

# Alias weight (dataset label)
CLASS_WEIGHTS["cycle"] = 0.3  # kept for backward compatibility

# ── BSV Blockchain ────────────────────────────────────────────────────
BSV_NETWORK = os.getenv("BSV_NETWORK", "main")
BSV_PRIVATE_KEY = os.getenv("BSV_PRIVATE_KEY", "")
ARC_URL = os.getenv("ARC_URL", "https://arc.gorillapool.io")
WOC_BASE = (
    "https://api.whatsonchain.com/v1/bsv/test"
    if BSV_NETWORK == "testnet"
    else "https://api.whatsonchain.com/v1/bsv/main"
)

# ── Dataset IDs (Kaggle) ───────────────────────────────────────────────
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

# ── Collision / Critical-event detection ───────────────────────────────
# IoU above this = overlap conflict
COLLISION_IOU_THRESHOLD = 0.05

# Strong overlap considered "possible collision"
COLLISION_IOU_HIGH = 0.15

# Pixel distance fallback (only used if normalized distance is off)
COLLISION_DISTANCE_THRESHOLD = 40

# Use scale-invariant normalized distance (recommended for UAV images)
COLLISION_USE_NORMALIZED_DISTANCE = True

# Near-miss threshold for normalized distance (distance / avg bbox diagonal)
# Lower = stricter (fewer near-misses). Recommended range: 0.8–1.3
COLLISION_DISTANCE_NORM_THRESHOLD = 1.2

# Enable license plate OCR only when incidents exist
ENABLE_PLATE_OCR = True

# ── Near-miss false-positive suppression (single-image heuristics) ─────
# These values reduce false positives due to parked vehicles / queues.
# They operate only for distance-triggered events (NEAR_MISS), not for IoU events.
#
# Interpretation:
# - "same lane" detection uses bbox size as scale:
#   dx small compared to bbox width and dy large compared to bbox height => likely queue/parked.
NEARMISS_SAME_LANE_X_FACTOR = 0.55     # dx <= factor * mean_bbox_width
NEARMISS_SAME_LANE_Y_FACTOR = 0.55     # dy <= factor * mean_bbox_height (alternate orientation)
NEARMISS_SEPARATION_Y_FACTOR = 1.25    # dy >= factor * mean_bbox_height
NEARMISS_SEPARATION_X_FACTOR = 1.25    # dx >= factor * mean_bbox_width

# If "queue-like", keep near-miss only if extremely close
NEARMISS_STRICT_DISTANCE_NORM = 0.65   # lower => fewer false near-misses in queues

# ── Optional: tuning notes (no code uses these) ────────────────────────
# If you see too many near-misses:
#   - decrease COLLISION_DISTANCE_NORM_THRESHOLD (e.g., 1.2 -> 1.0)
#   - decrease NEARMISS_STRICT_DISTANCE_NORM (e.g., 0.65 -> 0.55)
#
# If you see too few near-misses:
#   - increase COLLISION_DISTANCE_NORM_THRESHOLD (e.g., 1.2 -> 1.35)
#
# If queued/parked cars still trigger too much:
#   - increase NEARMISS_SEPARATION_Y_FACTOR (e.g., 1.25 -> 1.5)
#   - decrease NEARMISS_SAME_LANE_X_FACTOR (e.g., 0.55 -> 0.45)
