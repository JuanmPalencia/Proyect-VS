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

img_path = path + "/images_train/images_train"
image_name = "img_0047.jpg"

full_path = img_path + "/" + image_name

img = cv2.imread(full_path)

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
# Celdas de una imagen
GRID_SIZE = 4

density = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

for d in detections:
    cx, cy = d["centroid"]
    gx = int(cx / W * GRID_SIZE)
    gy = int(cy / H * GRID_SIZE)

    gx = min(gx, GRID_SIZE - 1)
    gy = min(gy, GRID_SIZE - 1)

    density[gy, gx] += 1

print("Mapa de densidad:\n", density)

ZONES = {
    "zona_superior": (0, 0, W, H//3),
    "zona_central": (0, H//3, W, 2*H//3),
    "zona_inferior": (0, 2*H//3, W, H)
}

zone_occupancy = {z: 0 for z in ZONES}

for d in detections:
    cx, cy = d["centroid"]
    for zone, (x1, y1, x2, y2) in ZONES.items():
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            zone_occupancy[zone] += 1

print("Ocupación por zonas:", zone_occupancy)

row = df[df["filename"] == image_name].iloc[0]
capture_timestamp = row["timestamp"]

analysis_result = {
    "scene_id": image_name,
    "capture_timestamp": capture_timestamp,
    "counts": counts,
    "density_grid": density.tolist(),
    "zone_occupancy": zone_occupancy
}

output_file = f"analysis_{image_name}.json"
with open(output_file, "w") as f:
    json.dump(analysis_result, f, indent=4)

print(f"Resultado guardado en: {output_file}")

results_list = []

for image_name in df["filename"].unique():
    results_list.append(analysis_result)

previous_hash = "0"*64 
for analysis_result in results_list:
    # json.dmps --> DE analysis_result = {"filename": "img1.jpg", "count": 3} A {"count": 3, "filename": "img1.jpg"}
    serialized = json.dumps(analysis_result, sort_keys=True).encode() # sort_keys = garantiza orden consistente
    combined = previous_hash.encode() + serialized
    block_hash = hashlib.sha256(combined).hexdigest()
    analysis_result["block_hash"] = block_hash
    previous_hash = block_hash

serialized_result = json.dumps(analysis_result, sort_keys=True).encode()

analysis_hash = hashlib.sha256(serialized_result).hexdigest()

analysis_result["hash"] = analysis_hash

print("Hash criptográfico del análisis:", analysis_hash)

output_file = f"analysis_{image_name}.json"
with open(output_file, "w") as f:
    json.dump(analysis_result, f, indent=4)

print(f"Resultado guardado en: {output_file}")