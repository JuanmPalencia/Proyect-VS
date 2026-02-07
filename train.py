"""Fine-tuning script for YOLO12 on traffic datasets.

Usage:
    python train.py --data data/combined/data.yaml --epochs 30
    python train.py --model yolo26m.pt --data data/combined/data.yaml --epochs 30 --imgsz 640 --batch 16 --device 0 --workers 8

The data YAML should follow ultralytics format:
    path: /path/to/dataset
    train: images/train
    val: images/val
    nc: 6
    names: ['car', 'motorcycle', 'truck', 'bus', 'accident', 'license_plate']
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(_file_).resolve().parent))

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO12 for aerial vehicle detection")
    parser.add_argument("--model", default="yolo26s.pt", help="Base model (YOLO26)")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")  
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--freeze", type=int, default=0, help="Freeze first N layers")
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="traffic_finetune")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
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
    )
    print(f"Training complete. Best model at: {args.project}/{args.name}/weights/best.pt")
    print("To use it: set YOLO_MODEL=runs/train/traffic_finetune/weights/best.pt in .env")


if _name_ == "_main_":
    main()