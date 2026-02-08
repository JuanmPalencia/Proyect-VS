# Configuración centralizada del sistema de análisis de tráfico aéreo.
# - Rutas y ficheros
# - Configuración de modelos de detección
# - Umbrales de métricas y riesgo
# - Parámetros de detección de incidentes críticos
# - Configuración de blockchain (BSV)
# - Definición de datasets


import os
from pathlib import Path
from dotenv import load_dotenv

# Carga de variables de entorno desde un archivo .env (si existe)
load_dotenv()

# Rutas del proyecto
# Directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent

# Carpeta para datos generados
DATA_DIR = ROOT_DIR / "data"

# Fichero local que actúa como ledger cuando no hay blockchain real
LEDGER_PATH = DATA_DIR / "ledger.jsonl"

# Carpeta donde se almacenan los modelos de IA
# MODELS_DIR = ROOT_DIR / "models"

# Configuración del modelo de detección
# Ruta al modelo YOLO entrenado específicamente para tráfico aéreo
YOLO_MODEL = os.getenv(
    "YOLO_MODEL",
    "runs/detect/runs/train/traffic_finetune4/weights/best.pt"
)

# Modelo de respaldo (YOLO genérico) si el modelo fine-tuned no está disponible
YOLO_FALLBACK_MODEL = os.getenv("YOLO_FALLBACK_MODEL", "yolov8m.pt")

# Umbral mínimo de confianza para aceptar una detección
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.15"))

# Dispositivo de inferencia: "cpu" o "cuda"
DEVICE = os.getenv("DEVICE", "cpu")

# Clases COCO usadas por el modelo fallback
# Diccionario: id_clase -> nombre
FALLBACK_COCO_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Clases del modelo entrenado específicamente
# Mapeo de IDs del modelo fine-tuned a nombres de clases
VEHICLE_CLASSES = {
    0: "car",
    1: "motorcycle",
    2: "truck",
    3: "bus",
}

# Alias de clases (consistencia entre datasets)
# Algunos datasets usan nombres distintos para la misma clase
# (por ejemplo "cycle" en lugar de "bicycle")
CLASS_ALIASES = {
    "cycle": "bicycle",
}

# Parámetros de métricas de tráfico
# Tamaño de la rejilla usada para el mapa de densidad
GRID_SIZE = 5

# Número de detecciones por celda a partir del cual se eleva el nivel de riesgo
RISK_DENSITY_THRESHOLD = 3

# Clases consideradas como vehículos pesados
HEAVY_VEHICLE_CLASSES = {"bus", "truck"}

# Pesos para el mapa de calor
# Cada tipo de vehículo aporta un peso distinto a la densidad
CLASS_WEIGHTS = {
    "car": 1.0,
    "motorcycle": 0.6,
    "bicycle": 0.3,
    "bus": 1.5,
    "truck": 1.5,
}

# Peso para alias antiguos (compatibilidad con datasets antiguos)
CLASS_WEIGHTS["cycle"] = 0.3

# Configuración de Blockchain BSV
# Red BSV: "main" o "testnet"
BSV_NETWORK = os.getenv("BSV_NETWORK", "main")

# Clave privada usada para firmar transacciones
BSV_PRIVATE_KEY = os.getenv("BSV_PRIVATE_KEY", "")

# Endpoint del ARC (servicio de envío de transacciones)
ARC_URL = os.getenv("ARC_URL", "https://arc.gorillapool.io")

# API base de Whatsonchain según la red
WOC_BASE = (
    "https://api.whatsonchain.com/v1/bsv/test"
    if BSV_NETWORK == "testnet"
    else "https://api.whatsonchain.com/v1/bsv/main"
)

# Definición de datasets
# Metadatos necesarios para descargar y procesar datasets automáticamente
DATASETS = {
    "uav_traffic": {
        "kaggle_id": "sakshamjn/traffic-images-captured-from-uavs",
        "format": "yolo",
        "classes": {0: "car", 1: "motorcycle"},
        "local_root": "data/traffic_aerial_images_for_vehicle_detection/dataset",
    },
    "roundabout": {
        "kaggle_id": "javiersanchezsoriano/roundabout-aerial-images-for-vehicle-detection",
        "format": "voc",
        "classes": {0: "car", 1: "cycle", 2: "truck", 3: "bus"},
        "local_root": "data/roundabout_aerial_images_for_vehicle_detection",
    },
}

# Umbral de IoU a partir del cual se considera conflicto por solapamiento
COLLISION_IOU_THRESHOLD = 0.05

# IoU alto es posible colisión real
COLLISION_IOU_HIGH = 0.15

# Distancia en píxeles como respaldo si no se usa distancia normalizada
COLLISION_DISTANCE_THRESHOLD = 40

# Uso de distancia normalizada (invariante a escala) - recomendado
COLLISION_USE_NORMALIZED_DISTANCE = True

# Umbral de distancia normalizada para detectar near-miss
# Valores más bajos es más estricto
COLLISION_DISTANCE_NORM_THRESHOLD = 1.2

# Supresión de falsos positivos en near-miss
# Heurísticas pensadas para una sola imagen (sin tracking temporal)
# Ayudan a evitar falsos positivos en colas o vehículos aparcados.

# Factores para considerar vehículos en el mismo carril
NEARMISS_SAME_LANE_X_FACTOR = 0.55
NEARMISS_SAME_LANE_Y_FACTOR = 0.55

# Factores de separación mínima para aceptar un near-miss
NEARMISS_SEPARATION_Y_FACTOR = 1.25
NEARMISS_SEPARATION_X_FACTOR = 1.25

# Si los vehículos parecen estar en cola, solo se acepta el near-miss
# cuando la distancia es extremadamente pequeña
NEARMISS_STRICT_DISTANCE_NORM = 0.65
