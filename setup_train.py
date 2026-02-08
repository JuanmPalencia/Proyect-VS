# Script para preparar todo el dataset de entrenamiento (UAV + Roundabout) para YOLO.

import shutil
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd

# Semilla fija para que el split train/val sea reproducible entre ejecuciones
random.seed(42)

# RUTAS
# Carpeta del proyecto
ROOT = Path(__file__).resolve().parent

# Carpeta general de datos (datasets descargados + outputs generados)
DATA_DIR = ROOT / "data"

# Datasets fuente (ya extraídos previamente)
UAV_DIR = DATA_DIR / "traffic_aerial_images_for_vehicle_detection" / "dataset"
ROUNDABOUT_DIR = DATA_DIR / "roundabout_aerial_images_for_vehicle_detection"

# Salida intermedia: Roundabout convertido a YOLO
YOLO_ROUNDABOUT = DATA_DIR / "yolo_roundabout"

# Salida final: dataset combinado listo para entrenar
COMBINED_DIR = DATA_DIR / "combined"

# CONFIGURACIÓN
# Mapeo de clases del dataset de Roundabout
ROUNDABOUT_CLASSES = {
    "car": 0,
    "cycle": 1,       # Se mapea como "motorcycle" en el dataset unificado
    "truck": 2,
    "bus": 3,
}

# El dataset UAV ya viene con labels YOLO: 0=car, 1=motorcycle
# Clases unificadas para el dataset final combinado
UNIFIED_CLASSES = ["car", "motorcycle", "truck", "bus"]
UNIFIED_NC = len(UNIFIED_CLASSES)

# Porcentaje de entrenamiento
TRAIN_RATIO = 0.85

# Tamaño fijo esperado para imágenes de rotonda (se usa para normalizar bounding boxes)
ROUNDABOUT_IMG_SIZE = (1920, 1080)


# Comprobación de que los datasets existen y que su estructura mínima está correcta
def verify_datasets():
    print("\n" + "=" * 70)
    print("STEP 0: Verifying datasets")
    print("=" * 70)

    issues = []

    # Verificación del dataset UAV
    if not UAV_DIR.exists():
        issues.append(f"UAV dataset not found: {UAV_DIR}")
    else:
        # El dataset UAV se organiza en carpetas secXX
        sec_dirs = [d for d in UAV_DIR.iterdir() if d.is_dir() and d.name.startswith("sec")]
        print(f"UAV dataset found: {len(sec_dirs)} sections")

        # Contamos imágenes y labels para tener una idea rápida de consistencia
        total_images = 0
        total_labels = 0
        for sec in sec_dirs:
            images = list(sec.glob("*.png"))
            labels = list(sec.glob("*.txt"))
            total_images += len(images)
            total_labels += len(labels)
            print(f"   {sec.name}: {len(images)} images, {len(labels)} labels")

        print(f"TOTAL UAV: {total_images} images, {total_labels} labels")

        if total_images == 0:
            issues.append(f"UAV dataset has no images in sections")

    # Verificación del dataset Roundabout
    if not ROUNDABOUT_DIR.exists():
        issues.append(f"Roundabout dataset not found: {ROUNDABOUT_DIR}")
    else:
        # El dataset Roundabout usa un CSV de anotaciones
        csv_path = ROUNDABOUT_DIR / "data.csv"

        # Estructura esperada de las imágenes
        img_path = ROUNDABOUT_DIR / "original" / "original" / "imgs"

        if not csv_path.exists():
            issues.append(f"[ERROR] Roundabout CSV not found: {csv_path}")
        else:
            df = pd.read_csv(csv_path)
            print(f"Roundabout dataset found: {len(df)} annotations")
            print(f"   Classes: {df['class_name'].value_counts().to_dict()}")

        if not img_path.exists():
            issues.append(f"Roundabout images not found: {img_path}")
        else:
            images = list(img_path.glob("*.jpg"))
            print(f"Images: {len(images)}")

    # Si hay problemas, los listamos todos juntos para facilitar debugging
    if issues:
        print("\n" + "=" * 70)
        print("[WARNING]  ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        print("=" * 70)
        return False

    print("\n[OK] All datasets verified successfully!")
    return True


# Convierte el dataset Roundabout (CSV con coords tipo VOC) a formato YOLO.
def convert_roundabout_to_yolo():

    print("\n" + "=" * 70)
    print("STEP 1: Converting Roundabout dataset to YOLO format")
    print("=" * 70)

    csv_path = ROUNDABOUT_DIR / "data.csv"
    images_dir = ROUNDABOUT_DIR / "original" / "original" / "imgs"

    print(f" Reading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total annotations: {len(df)}")
    print(f"Classes: {df['class_name'].value_counts().to_dict()}")

    # Agrupamos anotaciones por imagen para generar un .txt por cada imagen
    grouped = defaultdict(list)

    # Tamaño fijo esperado en el dataset Roundabout (para normalización)
    img_w, img_h = ROUNDABOUT_IMG_SIZE

    for _, row in df.iterrows():
        # Aseguramos que el nombre de imagen está bien formateado
        img_name = Path(str(row["image_name"])).name
        cls_raw = row["class_name"]

        # Filtramos valores nulos o clases no contempladas
        if pd.isna(cls_raw):
            continue

        cls = str(cls_raw).strip().lower()
        if cls not in ROUNDABOUT_CLASSES:
            continue

        # Mapeo clase id (YOLO)
        cls_id = ROUNDABOUT_CLASSES[cls]

        # Coordenadas en formato VOC/absoluto (esquinas)
        x_min = float(row["x_min"])
        y_min = float(row["y_min"])
        x_max = float(row["x_max"])
        y_max = float(row["y_max"])

        # Convertimos a formato YOLO y normalizamos
        cx = ((x_min + x_max) / 2) / img_w
        cy = ((y_min + y_max) / 2) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        # Seguridad: limitamos a [0,1] para evitar anotaciones fuera de rango
        cx = max(0, min(1, cx))
        cy = max(0, min(1, cy))
        w = max(0, min(1, w))
        h = max(0, min(1, h))

        grouped[img_name].append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    print(f"Unique images with annotations: {len(grouped)}")

    # Split train/val
    all_images = sorted(grouped.keys())
    random.shuffle(all_images)
    split_idx = int(len(all_images) * TRAIN_RATIO)
    train_imgs = set(all_images[:split_idx])
    val_imgs = set(all_images[split_idx:])
    print(f"Train: {len(train_imgs)}, Val: {len(val_imgs)}")

    # Creamos estructura YOLO estándar
    for split in ("train", "val"):
        (YOLO_ROUNDABOUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_ROUNDABOUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Copiamos imágenes y escribimos labels
    stats = {"train": 0, "val": 0, "skipped": 0}

    for img_name, labels in grouped.items():
        src_img = images_dir / img_name

        # Si alguna imagen no existe en disco, la saltamos (dataset incompleto)
        if not src_img.exists():
            stats["skipped"] += 1
            continue

        split = "train" if img_name in train_imgs else "val"
        dst_img = YOLO_ROUNDABOUT / "images" / split / img_name
        dst_lbl = YOLO_ROUNDABOUT / "labels" / split / (Path(img_name).stem + ".txt")

        # Copiamos imagen solo si no existía ya (idempotencia: se puede re-ejecutar)
        if not dst_img.exists():
            shutil.copy2(src_img, dst_img)

        # Un fichero .txt por imagen, una línea por bounding box
        dst_lbl.write_text("\n".join(labels) + "\n")
        stats[split] += 1

    print(f"\nRoundabout YOLO conversion complete!")
    print(f"Train images: {stats['train']}")
    print(f"Val images: {stats['val']}")
    print(f"Skipped (not found): {stats['skipped']}")
    print(f"Output: {YOLO_ROUNDABOUT}")

    # Retornamos True si al menos se generó algo
    return stats["train"] > 0 or stats["val"] > 0


# Fusiona UAV y Roundabout en un dataset combinado único
def merge_datasets():
    print("\n" + "=" * 70)
    print("STEP 2: Merging UAV + Roundabout datasets")
    print("=" * 70)

    # Estructura final requerida por YOLO
    for split in ("train", "val"):
        (COMBINED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (COMBINED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Contadores para resumen final
    uav_train = 0
    uav_val = 0
    round_train = 0
    round_val = 0

    # Procesar dataset UAV
    print("\n Processing UAV traffic dataset...")

    uav_images = []

    if not UAV_DIR.exists():
        print("UAV directory not found!")
    else:
        # Cada subcarpeta secXX contiene imágenes + labels YOLO
        for sec_dir in sorted(UAV_DIR.iterdir()):
            if not sec_dir.is_dir() or not sec_dir.name.startswith("sec"):
                continue

            images = sorted(sec_dir.glob("*.png"))
            for img in images:
                label = img.with_suffix(".txt")
                if label.exists():
                    uav_images.append((img, label, sec_dir.name))

        print(f"Found {len(uav_images)} UAV images with labels")

        if uav_images:
            # Split train/val del UAV
            random.shuffle(uav_images)
            split_idx = int(len(uav_images) * TRAIN_RATIO)
            train_set = uav_images[:split_idx]
            val_set = uav_images[split_idx:]

            # Copia al dataset final, renombrando para no pisar ficheros
            for subset, split_name in [(train_set, "train"), (val_set, "val")]:
                for img_path, lbl_path, sec_name in subset:
                    new_name = f"uav_{sec_name}_{img_path.name}"
                    dst_img = COMBINED_DIR / "images" / split_name / new_name
                    dst_lbl = COMBINED_DIR / "labels" / split_name / new_name.replace(".png", ".txt")

                    if not dst_img.exists():
                        shutil.copy2(img_path, dst_img)
                    if not dst_lbl.exists():
                        shutil.copy2(lbl_path, dst_lbl)

            uav_train = len(train_set)
            uav_val = len(val_set)
            print(f"UAV: {uav_train} train, {uav_val} val")
        else:
            print("No UAV images found with labels")

    # Procesar dataset Roundabout ya convertido a YOLO
    print("\n Processing Roundabout dataset...")

    if not YOLO_ROUNDABOUT.exists():
        print("Roundabout YOLO directory not found. Run Step 1 first!")
    else:
        for split in ("train", "val"):
            img_dir = YOLO_ROUNDABOUT / "images" / split
            lbl_dir = YOLO_ROUNDABOUT / "labels" / split

            if not img_dir.exists():
                continue

            count = 0
            for img in sorted(img_dir.glob("*.jpg")):
                lbl = lbl_dir / img.with_suffix(".txt").name

                # Renombre para evitar conflictos con nombres del UAV
                new_name = f"round_{img.name}"
                dst_img = COMBINED_DIR / "images" / split / new_name
                dst_lbl = COMBINED_DIR / "labels" / split / new_name.replace(".jpg", ".txt")

                if not dst_img.exists():
                    shutil.copy2(img, dst_img)
                if lbl.exists() and not dst_lbl.exists():
                    shutil.copy2(lbl, dst_lbl)

                count += 1

            if split == "train":
                round_train = count
            else:
                round_val = count

        print(f"Roundabout: {round_train} train, {round_val} val")

    # Resumen final
    total_train = uav_train + round_train
    total_val = uav_val + round_val

    print(f"\n[OK] Merge complete!")
    print(f"   UAV:        {uav_train} train, {uav_val} val")
    print(f"   Roundabout: {round_train} train, {round_val} val")
    print(f"   ────────────────────────────────────")
    print(f"   TOTAL:      {total_train} train, {total_val} val")
    print(f"   Output: {COMBINED_DIR}")

    return total_train > 0 or total_val > 0


# PASO 3: CREAR DATA.YAML
def create_data_yaml():
    print("\n" + "=" * 70)
    print("STEP 3: Creating data.yaml")
    print("=" * 70)

    yaml_content = f"""# Urban VS - Traffic Aerial Vehicle Detection Dataset
# Combined: UAV Traffic + Roundabout Aerial Images

path: {COMBINED_DIR.as_posix()}
train: images/train
val: images/val

nc: {UNIFIED_NC}
names: {UNIFIED_CLASSES}

# Class mapping:
# 0: car          (from both datasets)
# 1: motorcycle   (UAV: motorcycle, Roundabout: cycle)
# 2: truck        (from roundabout)
# 3: bus          (from roundabout)
"""

    yaml_path = COMBINED_DIR / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    print(f"[OK] data.yaml created: {yaml_path}")
    print(f"\n Dataset configuration:")
    print(f"   Classes: {UNIFIED_CLASSES}")
    print(f"   NC: {UNIFIED_NC}")
    print(f"   Path: {COMBINED_DIR}")

    return yaml_path


# MAIN
def main():
    print("\n" + "=" * 70)
    print("Urban VS - Complete Training Data Setup")
    print("=" * 70)

    # Verificar estructura y disponibilidad de datasets
    if not verify_datasets():
        print("\n[ERROR] Dataset verification failed. Please check the paths.")
        print("\nExpected structure:")
        print(f"  {UAV_DIR}")
        print(f"  {ROUNDABOUT_DIR}")
        return

    # Convertir Roundabout a YOLO (si no se genera nada, paramos)
    if not convert_roundabout_to_yolo():
        print("\n[ERROR] Failed to convert Roundabout dataset")
        return

    # Fusionar datasets en uno único
    if not merge_datasets():
        print("\n[ERROR] Failed to merge datasets")
        return

    # Crear data.yaml final para entrenamiento
    yaml_path = create_data_yaml()

    # Resumen final
    print("\n" + "=" * 70)
    print("[OK] ALL DONE! Dataset is ready for training.")
    print("=" * 70)

    # Contamos archivos finales para estadísticas rápidas
    train_imgs = len(list((COMBINED_DIR / "images" / "train").glob("*")))
    val_imgs = len(list((COMBINED_DIR / "images" / "val").glob("*")))

    print(f"\n Final dataset statistics:")
    print(f"   Train images: {train_imgs}")
    print(f"   Val images:   {val_imgs}")
    print(f"   Total:        {train_imgs + val_imgs}")
    print(f"   Classes:      {UNIFIED_CLASSES}")

    # Mensajes finales orientados a “siguiente paso”
    print(f"\n Next steps:")
    print(f"   1. Verify data:")
    print(
        f"      python -c \"from pathlib import Path; "
        f"print('Train:', len(list(Path('data/combined/images/train').glob('*'))))\""
    )
    print(f"\n   2. Start training:")
    print(f"      python train.py --data {yaml_path} --epochs 50 --batch 8 --device cpu")
    print(f"\n   3. Or with GPU:")
    print(f"      python train.py --model yolo11m.pt --data {yaml_path} --epochs 100 --batch 16 --device 0")

    print(f"\n See README_TRAINING.md for detailed training guide")


if __name__ == "__main__":
    main()
