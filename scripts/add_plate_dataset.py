"""Add license plate dataset to the combined dataset, remapping class 0->5."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMBINED = ROOT / "data" / "combined"
PLATES = ROOT / "data" / "plate_dataset"

REMAP = {0: 5}


def remap_label(src: Path, dst: Path):
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
    count = 0
    for img in sorted(img_dir.glob("*.*")):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / img.with_suffix(".txt").name
        new_img_name = f"plate_{img.name}"
        new_lbl_name = f"plate_{img.stem}.txt"

        dst_img = COMBINED / "images" / split_name / new_img_name
        dst_lbl = COMBINED / "labels" / split_name / new_lbl_name

        if not dst_img.exists():
            shutil.copy2(img, dst_img)
        if lbl.exists() and not dst_lbl.exists():
            remap_label(lbl, dst_lbl)
        count += 1
    return count


def main():
    print("Adding license plate dataset to combined")

    train_count = add_split("train", PLATES / "train" / "images", PLATES / "train" / "labels")
    print(f"Train: {train_count} images added")

    val_count = add_split("val", PLATES / "valid" / "images", PLATES / "valid" / "labels")
    print(f"Val: {val_count} images added")

    yaml_content = f"""path: {COMBINED.as_posix()}
train: images/train
val: images/val

nc: 6
names: ['car', 'motorcycle', 'truck', 'bus', 'accident', 'license_plate']
"""
    (COMBINED / "data.yaml").write_text(yaml_content, encoding="utf-8")
    print(f"\nUpdated data.yaml: 6 classes")

    train_total = len(list((COMBINED / "images" / "train").iterdir()))
    val_total = len(list((COMBINED / "images" / "val").iterdir()))
    print(f"Combined total: {train_total} train, {val_total} val")
    print("Done!")


if __name__ == "__main__":
    main()
