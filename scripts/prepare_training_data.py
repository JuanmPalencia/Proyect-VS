"""Convert roundabout CSV dataset to YOLO format and create train/val split.

Usage:
    python scripts/prepare_training_data.py

This script:
1. Reads the roundabout CSV (absolute bbox coords)
2. Converts to YOLO format (class_id cx cy w h normalized)
3. Creates train/val split (80/20)
4. Outputs to data/yolo_roundabout/{images,labels}/{train,val}/
"""

import shutil
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ROUNDABOUT_DIR = ROOT / "data" / "roundabout_aerial_images_for_vehicle_detection"
CSV_PATH = ROUNDABOUT_DIR / "data.csv"
IMAGES_DIR = ROUNDABOUT_DIR / "original" / "original" / "imgs"
OUTPUT_DIR = ROOT / "data" / "yolo_roundabout"

# Class mapping: name → YOLO class id
CLASS_MAP = {
    "car": 0,
    "cycle": 1,
    "truck": 2,
    "bus": 3,
}

TRAIN_RATIO = 0.8
IMG_W, IMG_H = 1920, 1080  # All roundabout images are this size
SEED = 42


def main():
    print(f"Reading CSV from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Total annotations: {len(df)}")
    print(f"  Classes: {df['class_name'].value_counts().to_dict()}")

    # Group annotations by image
    grouped = defaultdict(list)
    for _, row in df.iterrows():
        img_name = Path(str(row["image_name"])).name  # strip "original/imgs/" prefix
        cls_raw = row["class_name"]
        if pd.isna(cls_raw):
            continue
        cls = str(cls_raw).strip().lower()
        if cls not in CLASS_MAP:
            continue
        cls_id = CLASS_MAP[cls]
        x_min = float(row["x_min"])
        y_min = float(row["y_min"])
        x_max = float(row["x_max"])
        y_max = float(row["y_max"])

        # Convert absolute to YOLO normalized (cx, cy, w, h)
        cx = ((x_min + x_max) / 2) / IMG_W
        cy = ((y_min + y_max) / 2) / IMG_H
        w = (x_max - x_min) / IMG_W
        h = (y_max - y_min) / IMG_H

        # Clamp to [0, 1]
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        grouped[img_name].append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    print(f"  Unique images with annotations: {len(grouped)}")

    # Split
    all_images = sorted(grouped.keys())
    random.seed(SEED)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_imgs = set(all_images[:split_idx])
    val_imgs = set(all_images[split_idx:])
    print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    # Create output directories
    for split in ("train", "val"):
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Write files
    stats = {"train": 0, "val": 0, "skipped": 0}
    for img_name, labels in grouped.items():
        src_img = IMAGES_DIR / img_name
        if not src_img.exists():
            stats["skipped"] += 1
            continue

        split = "train" if img_name in train_imgs else "val"
        dst_img = OUTPUT_DIR / "images" / split / img_name
        dst_lbl = OUTPUT_DIR / "labels" / split / (Path(img_name).stem + ".txt")

        # Copy image (symlink would be faster but Windows compatibility)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Write YOLO label
        dst_lbl.write_text("\n".join(labels) + "\n")
        stats[split] += 1

    print(f"\nDone! Output: {OUTPUT_DIR}")
    print(f"  Train images: {stats['train']}")
    print(f"  Val images: {stats['val']}")
    print(f"  Skipped (image not found): {stats['skipped']}")

    # Write data.yaml for ultralytics
    yaml_path = OUTPUT_DIR / "data.yaml"
    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/val

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    yaml_path.write_text(yaml_content)
    print(f"  data.yaml: {yaml_path}")


if __name__ == "__main__":
    main()
