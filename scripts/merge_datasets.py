"""Merge UAV traffic + Roundabout datasets into a single YOLO dataset."""

import os
import random
import shutil
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "combined"

UAV_DIR = ROOT / "data" / "traffic_aerial_images_for_vehicle_detection" / "dataset"
ROUND_IMG_TRAIN = ROOT / "data" / "yolo_roundabout" / "images" / "train"
ROUND_IMG_VAL = ROOT / "data" / "yolo_roundabout" / "images" / "val"
ROUND_LBL_TRAIN = ROOT / "data" / "yolo_roundabout" / "labels" / "train"
ROUND_LBL_VAL = ROOT / "data" / "yolo_roundabout" / "labels" / "val"

# Unified classes: 0=car, 1=motorcycle, 2=truck, 3=bus
# UAV:        0=car, 1=motorcycle       → no remap needed
# Roundabout: 0=car, 1=cycle, 2=truck, 3=bus → remap 1(cycle)→1(motorcycle), rest same
# Classes are already compatible!

VAL_RATIO = 0.15  # for UAV split


def setup_dirs():
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    """Copy file, skip if already exists."""
    if not dst.exists():
        shutil.copy2(src, dst)


def add_uav_dataset():
    """Add UAV dataset with 85/15 train/val split."""
    print("Processing UAV dataset...")
    all_images = []

    for sec_dir in sorted(UAV_DIR.iterdir()):
        if not sec_dir.is_dir():
            continue
        images = sorted(sec_dir.glob("*.png"))
        for img in images:
            label = img.with_suffix(".txt")
            if label.exists():
                all_images.append((img, label, sec_dir.name))

    random.shuffle(all_images)
    split_idx = int(len(all_images) * (1 - VAL_RATIO))
    train_set = all_images[:split_idx]
    val_set = all_images[split_idx:]

    for subset, split_name in [(train_set, "train"), (val_set, "val")]:
        for img_path, lbl_path, sec_name in subset:
            # Prefix with section name to avoid filename collisions
            new_name = f"uav_{sec_name}_{img_path.name}"
            copy_file(img_path, OUT / "images" / split_name / new_name)
            copy_file(lbl_path, OUT / "labels" / split_name / new_name.replace(".png", ".txt"))

    print(f"  UAV: {len(train_set)} train, {len(val_set)} val")


def add_roundabout_dataset():
    """Add roundabout dataset (already split)."""
    print("Processing Roundabout dataset...")
    train_count = 0
    val_count = 0

    # Train
    for img in sorted(ROUND_IMG_TRAIN.glob("*.jpg")):
        lbl = ROUND_LBL_TRAIN / img.with_suffix(".txt").name
        new_name = f"round_{img.name}"
        copy_file(img, OUT / "images" / "train" / new_name)
        if lbl.exists():
            copy_file(lbl, OUT / "labels" / "train" / new_name.replace(".jpg", ".txt"))
        train_count += 1

    # Val
    for img in sorted(ROUND_IMG_VAL.glob("*.jpg")):
        lbl = ROUND_LBL_VAL / img.with_suffix(".txt").name
        new_name = f"round_{img.name}"
        copy_file(img, OUT / "images" / "val" / new_name)
        if lbl.exists():
            copy_file(lbl, OUT / "labels" / "val" / new_name.replace(".jpg", ".txt"))
        val_count += 1

    print(f"  Roundabout: {train_count} train, {val_count} val")


def write_data_yaml():
    """Write combined data.yaml."""
    yaml_content = f"""path: {OUT.as_posix()}
train: images/train
val: images/val

nc: 4
names: ['car', 'motorcycle', 'truck', 'bus']
"""
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\ndata.yaml written to: {yaml_path}")


def main():
    print("=" * 50)
    print("Merging UAV + Roundabout datasets")
    print("=" * 50)

    setup_dirs()
    add_uav_dataset()
    add_roundabout_dataset()
    write_data_yaml()

    # Summary
    train_imgs = len(list((OUT / "images" / "train").iterdir()))
    val_imgs = len(list((OUT / "images" / "val").iterdir()))
    print(f"\nTotal combined: {train_imgs} train, {val_imgs} val")
    print(f"Classes: car, motorcycle, truck, bus")
    print("Done!")


if __name__ == "__main__":
    main()
