# Script de fine-tuning para YOLO sobre datasets de tráfico (UAV + Roundabout)

import argparse
import sys
from pathlib import Path

# Añadimos la raíz del proyecto al path para facilitar imports locales
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO


# Comprobamos si el dataset de entrenamiento está listo para usar

def check_data_prepared(data_path: Path) -> bool:
    # Si ni siquiera existe el YAML, no hay dataset preparado
    if not data_path.exists():
        return False

    # Validación mínima del contenido del YAML (no lo parseamos, solo comprobamos claves básicas)
    try:
        yaml_content = data_path.read_text()
        if "path:" not in yaml_content or "train:" not in yaml_content:
            return False
    except Exception:
        # Si falla la lectura, lo tratamos como no preparado
        return False

    # El dataset combinado se asume en el directorio padre del YAML
    combined_dir = data_path.parent

    # Estructura estándar de Ultralytics: images/train y images/val
    train_dir = combined_dir / "images" / "train"
    val_dir = combined_dir / "images" / "val"

    if not train_dir.exists() or not val_dir.exists():
        return False

    # Comprobación rápida de que hay imágenes realmente
    train_images = list(train_dir.glob("*"))
    val_images = list(val_dir.glob("*"))

    if len(train_images) == 0 or len(val_images) == 0:
        return False

    return True

# Prepara los datos de entrenamiento si no están listos.
def prepare_data_if_needed(data_path: Path, force_prepare: bool = False):
    # Imports locales: aquí no se usan directamente (los usa setup_train),
    # pero se dejan como “dependencias esperadas” si el script crece.
    from pathlib import Path
    import shutil
    import random
    from collections import defaultdict
    import pandas as pd

    print("\n" + "=" * 70)
    print(" CHECKING TRAINING DATA")
    print("=" * 70)

    # Caso normal: el dataset ya está y no queremos forzar regeneración
    if not force_prepare and check_data_prepared(data_path):
        print(f"[OK] Training data already prepared at: {data_path}")
        print(f"   Skipping data preparation step.")

        # Estadísticas rápidas para dejar constancia en consola
        combined_dir = data_path.parent
        train_count = len(list((combined_dir / "images" / "train").glob("*")))
        val_count = len(list((combined_dir / "images" / "val").glob("*")))
        print(f"\n Dataset stats:")
        print(f"   Train: {train_count} images")
        print(f"   Val:   {val_count} images")
        print(f"   Total: {train_count + val_count} images")
        return

    # Si llegamos aquí, falta dataset o está incompleto
    print("[WARNING]  Training data not found or incomplete")
    print(" Starting automatic data preparation...")
    print()

    # Ejecutamos setup_train.py como módulo para evitar pedirle al usuario un paso manual
    try:
        import importlib.util
        setup_script = Path(__file__).parent / "setup_train.py"

        if not setup_script.exists():
            print(f"[ERROR] Setup script not found: {setup_script}")
            print("   Please run: python setup_train.py")
            sys.exit(1)

        # Cargamos el archivo como un módulo dinámico (sin necesidad de instalarlo)
        spec = importlib.util.spec_from_file_location("setup_data", setup_script)
        setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup_module)

        # Lanzamos la preparación (esperamos que setup_train.py tenga main())
        print("Running data preparation...")
        setup_module.main()

        # Verificamos que realmente se generó el dataset esperado
        if not check_data_prepared(data_path):
            print("\n[ERROR] Data preparation failed!")
            print("   Please check the error messages above.")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("[OK] DATA PREPARATION COMPLETE")
        print("=" * 70)

    except Exception as e:
        # Si algo falla, dejamos instrucciones claras para el modo manual
        print(f"\n[ERROR] Error during data preparation: {e}")
        print("\nPlease run manually:")
        print("   python setup_train.py")
        sys.exit(1)


# ENTRENAMIENTO
def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for aerial vehicle detection")
    parser.add_argument("--model", default="yolo8m.pt", help="Base model (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--data", default="data/combined/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=960, help="Image size for training")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers")
    parser.add_argument("--device", default="cpu", help="Device (cpu, 0, 0,1, etc)")
    parser.add_argument("--project", default="runs/train", help="Project directory")
    parser.add_argument("--name", default="traffic_finetune", help="Run name")
    parser.add_argument("--force-prepare", action="store_true", help="Force data preparation even if already done")
    parser.add_argument("--skip-check", action="store_true", help="Skip data preparation check (not recommended)")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    # Convertimos el path del YAML a objeto Path
    data_path = Path(args.data)

    # Preparamos datos si hace falta
    if not args.skip_check:
        prepare_data_if_needed(data_path, force_prepare=args.force_prepare)

    # Validación final: si no existe el YAML, no se puede entrenar
    if not data_path.exists():
        print(f"\n[ERROR] Data file not found: {data_path}")
        print("   Please check the path or run: python setup_train.py")
        sys.exit(1)

    # Imprimimos configuración para trazabilidad del experimento
    print("\n" + "=" * 70)
    print("️  STARTING YOLO TRAINING")
    print("=" * 70)
    print(f"\n Training configuration:")
    print(f"   Model:      {args.model}")
    print(f"   Data:       {args.data}")
    print(f"   Epochs:     {args.epochs}")
    print(f"   Image size: {args.imgsz}")
    print(f"   Batch size: {args.batch}")
    print(f"   Device:     {args.device}")
    print(f"   Workers:    {args.workers}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Project:    {args.project}")
    print(f"   Name:       {args.name}")
    print()

    # Carga del modelo base
    # Ultralytics acepta rutas a pesos preentrenados (p.ej. yolo11m.pt) o runs previos.
    print(f"Loading model: {args.model}...")
    model = YOLO(args.model)

    # Entrenamiento
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        lr0=args.lr,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )

    # Fin del entrenamiento
    # Mostramos rutas importantes para que el usuario no tenga que buscarlas.
    print("\n" + "=" * 70)
    print("[OK] TRAINING COMPLETE")
    print("=" * 70)

    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_model_path = Path(args.project) / args.name / "weights" / "last.pt"

    print(f"\n Results:")
    print(f"   Best model:  {best_model_path}")
    print(f"   Last model:  {last_model_path}")
    print(f"   Results CSV: {Path(args.project) / args.name / 'results.csv'}")

    # Guía rápida para “enchufar” el modelo entrenado en el sistema (app + API)
    print(f"\n To use the trained model:")
    print(f"   1. Update .env file:")
    print(f"      YOLO_MODEL={best_model_path}")
    print(f"\n   2. Run the app:")
    print(f"      streamlit run app.py")
    print(f"\n   3. Or test with API:")
    print(f"      uvicorn api:app --reload")

    # Nota: evaluación rápida con el comando yolo val (Ultralytics CLI)
    print(f"\n To evaluate the model:")
    print(f"   yolo val model={best_model_path} data={args.data}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
