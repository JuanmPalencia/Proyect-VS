import pandas as pd
import numpy as np
import kagglehub
from ultralytics import YOLO
import cv2
import json
import hashlib
from datetime import datetime, timezone

path = kagglehub.dataset_download("manasmittal2005/campus-images")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/labels_train_updated.csv")

model = YOLO("yolov8n.pt")

image = "C:\\Users\\ricky\\OneDrive\\Universidad\\3º Año\\0 - Club IA\\ProyectosIA\\calle.jpg"
img = cv2.imread(image)

results = model(img, device="cpu")

CLASSES = {
    0: "person",
    2: "car",
    3: "motorcycle"
}

detections = []

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in CLASSES:

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            detections.append({
                "class_id": cls_id,
                "class_name": CLASSES[cls_id],
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "centroid": [cx, cy]
            })

            label = f"{CLASSES[cls_id]} {conf:.2f}"

            x1, y1, x2, y2 = map(float, box.xyxy[0])
            x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])

            cv2.rectangle(img, (x1i, y1i), (x2i, y2i), (0,255,0), 2)
            cv2.putText(
                img,
                label,
                (x1i, y1i - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )

cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

counts = {}

for d in detections:
    cls = d["class_name"]
    counts[cls] = counts.get(cls, 0) + 1

print("Conteo por categoría:", counts)

H, W, _ = img.shape
image_area = W * H

total_bbox_area = 0.0

for d in detections:
    x1, y1, x2, y2 = d["bbox"]
    bbox_area = (x2 - x1) * (y2 - y1)
    total_bbox_area += bbox_area

occupancy_ratio = total_bbox_area / image_area
occupancy_percent = occupancy_ratio * 100

print(f"Ocupación total de la imagen: {occupancy_percent:.2f}%")
