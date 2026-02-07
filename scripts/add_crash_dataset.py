"""Add crash/accident dataset to the combined dataset, remapping class 0->4."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "data" / "combined"
CRASH = ROOT / "data" / "crash_dataset"

# Accident class in crash dataset is 0, remap to 4 in combined
REMAP = {0: 4}


def remap_label(src: Path, dst: Path):
    """Copy label file, remapping class IDs."""
    lines = src.read_text(encoding="utf-8").strip().splitlines()
    remapped = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            old_cls = int(parts[0])
            new_cls = REMAP.get(old_cls, old_cls)
            parts[0] = str(new_cls)
            remapped.append(" ".join(parts))
    dst.write_text("\n".join(remapped) + "\n", encoding="utf-8")


def add_split(split_name: str, img_dir: Path, lbl_dir: Path):
    """Add images and remapped labels from a split."""
    count = 0
    for img in sorted(img_dir.glob("*.*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / img.with_suffix(".txt").name
        new_img_name = f"crash_{img.name}"
        new_lbl_name = f"crash_{img.stem}.txt"

        dst_img = COMBINED / "images" / split_name / new_img_name
        dst_lbl = COMBINED / "labels" / split_name / new_lbl_name

        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        if lbl.exists() and not dst_lbl.exists():
            remap_label(lbl, dst_lbl)
        count += 1
    return count


def main():
    print("=" * 50)
    print("Adding crash dataset to combined")
    print("=" * 50)

    # Train
    train_count = add_split(
        "train",
        CRASH / "train" / "images",
        CRASH / "train" / "labels",
    )
    print(f"Train: {train_count} images added")

    # Val (Roboflow uses 'valid')
    val_count = add_split(
        "val",
        CRASH / "valid" / "images",
        CRASH / "valid" / "labels",
    )
    print(f"Val: {val_count} images added")

    # Update data.yaml
    yaml_content = f"""path: {COMBINED.as_posix()}
train: images/train
val: images/val

nc: 5
names: ['car', 'motorcycle', 'truck', 'bus', 'accident']
"""
    yaml_path = COMBINED / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")
    print(f"\nUpdated data.yaml: 5 classes (added 'accident')")

    # Summary
    train_total = len(list((COMBINED / "images" / "train").iterdir()))
    val_total = len(list((COMBINED / "images" / "val").iterdir()))
    print(f"Combined total: {train_total} train, {val_total} val")
    print("Done!")


if __name__ == "__main__":
    main()
