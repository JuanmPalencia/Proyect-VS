"""Fine-tuning script for YOLO on traffic datasets with automatic data preparation.

This script:
1. Checks if training data is prepared (data/combined/)
2. If not prepared, automatically runs data setup
3. Trains the YOLO model

Usage:
    python train.py --data data/combined/data.yaml --epochs 50
    python train.py --model yolo11m.pt --data data/combined/data.yaml --epochs 100 --imgsz 640 --batch 16 --device 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO


# ══════════════════════════════════════════════════════════════════════════
# DATA PREPARATION CHECK
# ══════════════════════════════════════════════════════════════════════════

def check_data_prepared(data_path: Path) -> bool:
    """Check if training data is already prepared.

    Args:
        data_path: Path to data.yaml

    Returns:
        True if data is ready, False otherwise
    """
    if not data_path.exists():
        return False

    # Check if data.yaml exists and has content
    try:
        yaml_content = data_path.read_text()
        if "path:" not in yaml_content or "train:" not in yaml_content:
            return False
    except Exception:
        return False

    # Get the dataset root from yaml
    combined_dir = data_path.parent

    # Check if train/val directories exist and have images
    train_dir = combined_dir / "images" / "train"
    val_dir = combined_dir / "images" / "val"

    if not train_dir.exists() or not val_dir.exists():
        return False

    # Check if there are actually images
    train_images = list(train_dir.glob("*"))
    val_images = list(val_dir.glob("*"))

    if len(train_images) == 0 or len(val_images) == 0:
        return False

    return True


def prepare_data_if_needed(data_path: Path, force_prepare: bool = False):
    """Prepare training data if not already done.

    Args:
        data_path: Path to data.yaml
        force_prepare: If True, force data preparation even if already done
    """
    from pathlib import Path
    import shutil
    import random
    from collections import defaultdict
    import pandas as pd

    print("\n" + "=" * 70)
    print(" CHECKING TRAINING DATA")
    print("=" * 70)

    if not force_prepare and check_data_prepared(data_path):
        print(f"[OK] Training data already prepared at: {data_path}")
        print(f"   Skipping data preparation step.")

        # Show quick stats
        combined_dir = data_path.parent
        train_count = len(list((combined_dir / "images" / "train").glob("*")))
        val_count = len(list((combined_dir / "images" / "val").glob("*")))
        print(f"\n Dataset stats:")
        print(f"   Train: {train_count} images")
        print(f"   Val:   {val_count} images")
        print(f"   Total: {train_count + val_count} images")
        return

    print("[WARNING]  Training data not found or incomplete")
    print(" Starting automatic data preparation...")
    print()

    # Import and run setup script
    try:
        # Try to import the setup script as a module
        import importlib.util
        setup_script = Path(__file__).parent / "setup_data_fixed.py"

        if not setup_script.exists():
            print(f"[ERROR] Setup script not found: {setup_script}")
            print("   Please run: python setup_data_fixed.py")
            sys.exit(1)

        # Load and execute setup script
        spec = importlib.util.spec_from_file_location("setup_data", setup_script)
        setup_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(setup_module)

        # Run the main function
        print("Running data preparation...")
        setup_module.main()

        # Verify it worked
        if not check_data_prepared(data_path):
            print("\n[ERROR] Data preparation failed!")
            print("   Please check the error messages above.")
            sys.exit(1)

        print("\n" + "=" * 70)
        print("[OK] DATA PREPARATION COMPLETE")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Error during data preparation: {e}")
        print("\nPlease run manually:")
        print("   python setup_data_fixed.py")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO for aerial vehicle detection")
    parser.add_argument("--model", default="yolo11s.pt", help="Base model (yolo11n/s/m/l/x.pt)")
    parser.add_argument("--data", default="data/combined/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
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

    # Convert data path to Path object
    data_path = Path(args.data)

    # Check and prepare data if needed (unless explicitly skipped)
    if not args.skip_check:
        prepare_data_if_needed(data_path, force_prepare=args.force_prepare)

    # Verify data path exists
    if not data_path.exists():
        print(f"\n[ERROR] Data file not found: {data_path}")
        print("   Please check the path or run: python setup_data_fixed.py")
        sys.exit(1)

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

    # Load model
    print(f"Loading model: {args.model}...")
    model = YOLO(args.model)

    # Train
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

    # Training complete
    print("\n" + "=" * 70)
    print("[OK] TRAINING COMPLETE")
    print("=" * 70)

    best_model_path = Path(args.project) / args.name / "weights" / "best.pt"
    last_model_path = Path(args.project) / args.name / "weights" / "last.pt"

    print(f"\n Results:")
    print(f"   Best model:  {best_model_path}")
    print(f"   Last model:  {last_model_path}")
    print(f"   Results CSV: {Path(args.project) / args.name / 'results.csv'}")

    print(f"\n To use the trained model:")
    print(f"   1. Update .env file:")
    print(f"      YOLO_MODEL={best_model_path}")
    print(f"\n   2. Run the app:")
    print(f"      streamlit run app.py")
    print(f"\n   3. Or test with API:")
    print(f"      uvicorn api:app --reload")

    print(f"\n To evaluate the model:")
    print(f"   yolo val model={best_model_path} data={args.data}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
