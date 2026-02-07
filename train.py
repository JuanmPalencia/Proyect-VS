"""Optional fine-tuning script for YOLOv8 on traffic datasets.

Usage:
    python train.py --data data/datasets.yaml --epochs 30

The data YAML should follow ultralytics format:
    path: /path/to/dataset
    train: images/train
    val: images/val
    nc: 4
    names: ['car', 'motorcycle', 'truck', 'bus']
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for aerial vehicle detection")
    parser.add_argument("--model", default="yolov8n.pt", help="Base model")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")  
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--freeze", type=int, default=10, help="Freeze first N layers")
    parser.add_argument("--device", default="cpu")
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


if __name__ == "__main__":
    main()
