"""Cargador para datasets formato YOLO (Dataset 1: Imágenes de Tráfico UAV)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

from .base import BaseDatasetLoader, ImageSample

logger = logging.getLogger(__name__)

# Mapeo de clases del Dataset 1 (desde los archivos de etiquetas YOLO)
_DEFAULT_CLASSES = {0: "car", 1: "motorcycle"}


class YOLODatasetLoader(BaseDatasetLoader):
    """Itera imágenes + etiquetas YOLO .txt lado a lado.

    Estructura esperada (flexible – auto-descubre):
        <root>/images/{split}/*.jpg
        <root>/labels/{split}/*.txt
    O plano:
        <root>/images/*.jpg
        <root>/labels/*.txt
    """

    def __init__(self, dataset_id: str, root=None, kaggle_slug=None,
                 class_map: dict[int, str] | None = None):
        super().__init__(dataset_id, root=root, kaggle_slug=kaggle_slug)
        self.class_map = class_map or _DEFAULT_CLASSES

    def _find_image_dirs(self) -> list[Path]:
        """Encuentra directorios que contienen imágenes."""
        candidates = []
        for p in self.root.rglob("*.jpg"):
            if p.parent not in candidates:
                candidates.append(p.parent)
        for p in self.root.rglob("*.png"):
            if p.parent not in candidates:
                candidates.append(p.parent)
        return candidates

    def _find_label_for_image(self, img_path: Path) -> Path | None:
        """Dada una imagen, encuentra su archivo de etiquetas YOLO .txt."""
        stem = img_path.stem
        # Intentar el directorio hermano labels/
        for labels_dir in [
            img_path.parent.parent / "labels" / img_path.parent.name,
            img_path.parent.parent / "labels",
            img_path.parent / "labels",
            img_path.parent,
        ]:
            candidate = labels_dir / f"{stem}.txt"
            if candidate.exists():
                return candidate
        return None

    def _parse_yolo_label(self, label_path: Path, img_w: int, img_h: int) -> list[dict]:
        """Parsea formato YOLO: class_id cx cy w h (normalizado)."""
        labels = []
        for line in label_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            # Convertir coordenadas del centro normalizadas a xyxy absolutas
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            class_name = self.class_map.get(cls_id, f"class_{cls_id}")
            labels.append({
                "class_name": class_name,
                "class_id": cls_id,
                "bbox": [x1, y1, x2, y2],
            })
        return labels

    def _iter_samples(self) -> Iterator[ImageSample]:
        import cv2

        img_dirs = self._find_image_dirs()
        if not img_dirs:
            logger.warning("No image directories found under %s", self.root)
            return

        for img_dir in img_dirs:
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                for img_path in sorted(img_dir.glob(ext)):
                    # Leer la imagen solo para dimensiones (para poder parsear etiquetas)
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                    h, w = img.shape[:2]

                    label_path = self._find_label_for_image(img_path)
                    labels = []
                    if label_path:
                        labels = self._parse_yolo_label(label_path, w, h)

                    yield ImageSample(
                        image_path=img_path,
                        dataset_id=self.dataset_id,
                        labels=labels,
                    )
