import pandas as pd
import numpy as np
import kagglehub
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import scipy.ndimage

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
            cv2.putText(img, label, (x1i, y1i - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

counts = {}

for d in detections:
    cls = d["class_name"]
    counts[cls] = counts.get(cls, 0) + 1

print("Conteo por categoría:", counts)

H, W, _ = img.shape

CLASS_WEIGHT = {
    "car": 1.0,
    "person": 0.5,
    "motorcycle": 0.3
}

heatmap = np.zeros((H, W), dtype=float)

for d in detections:
    x1, y1, x2, y2 = map(int, d["bbox"])
    weight = CLASS_WEIGHT[d["class_name"]]
    
    heatmap[y1:y2, x1:x2] += weight

heatmap /= heatmap.max()

heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=15)

plt.imshow(heatmap, cmap='hot')
plt.colorbar()
plt.show()

heatmap_normalized = heatmap / heatmap.max()

total_occupancy = heatmap_normalized.sum()

num_pixels = heatmap_normalized.shape[0] * heatmap_normalized.shape[1]

occupancy_percent = (total_occupancy / num_pixels) * 100

print(f"Porcentaje de ocupación de la escena: {occupancy_percent:.2f}%")

H, W = heatmap_normalized.shape
ZONES = {
    "zona_superior": (0, 0, W, H//3),
    "zona_central": (0, H//3, W, 2*H//3),
    "zona_inferior": (0, 2*H//3, W, H)
}

zone_occupancy = {}
for zone, (x1, y1, x2, y2) in ZONES.items():
    zone_heatmap = heatmap_normalized[y1:y2, x1:x2]
    zone_occupancy[zone] = zone_heatmap.sum() / (zone_heatmap.size) * 100

print("Porcentaje de ocupación por zona:", zone_occupancy)