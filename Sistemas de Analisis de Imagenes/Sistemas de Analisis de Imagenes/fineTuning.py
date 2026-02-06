from ultralytics import YOLO

model = YOLO("yolov8n.pt")

# Si tenemos menos imagenes, reducimos los epochs y el lr
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    lr0=0.001,
    batch=16,
    freeze=10
)
