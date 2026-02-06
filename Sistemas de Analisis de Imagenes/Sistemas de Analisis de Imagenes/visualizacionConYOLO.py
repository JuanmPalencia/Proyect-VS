import pandas as pd
import numpy as np
import kagglehub
from ultralytics import YOLO
import cv2

# Entrenado a partir de COCO y sus clases
model = YOLO("yolov8n.pt")

image = "C:\\Users\\ricky\\OneDrive\\Universidad\\3º Año\\0 - Club IA\\ProyectosIA\\calle.jpg"

img = cv2.imread(image)

results = model(img, device="cpu")

CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    12: "stop sign",
    26: "umbrella"
}

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id in CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{CLASSES[cls_id]} {conf:.2f}"

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 ,0), 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
