"""Cargador para datasets formato VOC/CSV (Dataset 2: Imágenes de Rotondas)."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import pandas as pd

from .base import BaseDatasetLoader, ImageSample

logger = logging.getLogger(__name__)

_DEFAULT_CLASSES = {0: "car", 1: "cycle", 2: "truck", 3: "bus"}
_CLASS_NAME_NORMALIZE = {
    "cars": "car", "car": "car",
    "cycles": "cycle", "cycle": "cycle", "bicycle": "cycle",
    "trucks": "truck", "truck": "truck",
    "buses": "bus", "bus": "bus",
    "vehicle": "car",
    "empty": "empty",
}


class VOCDatasetLoader(BaseDatasetLoader):
    """Carga el dataset de rotondas con anotaciones XML VOC y/o CSV.

    Auto-descubre:
      - Anotaciones XML en Annotations/ o annotations/
      - Archivo CSV con columnas tipo filename, class, xmin, ymin, xmax, ymax
      - Imágenes en images/ o JPEGImages/
    """

    def __init__(self, dataset_id: str, root=None, kaggle_slug=None,
                 class_map: dict[int, str] | None = None):
        super().__init__(dataset_id, root=root, kaggle_slug=kaggle_slug)
        self.class_map = class_map or _DEFAULT_CLASSES
        self._csv_data: pd.DataFrame | None = None

    def _find_csv(self) -> Path | None:
        for p in self.root.rglob("*.csv"):
            return p
        return None

    def _find_xml_dir(self) -> Path | None:
        for name in ("Annotations", "annotations", "xmls", "labels"):
            d = self.root / name
            if d.is_dir():
                return d
        # Check subdirectories
        for d in self.root.rglob("Annotations"):
            if d.is_dir():
                return d
        return None

    def _find_images_dir(self) -> list[Path]:
        candidates = []
        for name in ("images", "JPEGImages", "imgs"):
            for d in self.root.rglob(name):
                if d.is_dir():
                    candidates.append(d)
        if not candidates:
            # Fallback: encontrar directorios con imágenes
            for p in self.root.rglob("*.jpg"):
                if p.parent not in candidates:
                    candidates.append(p.parent)
            for p in self.root.rglob("*.png"):
                if p.parent not in candidates:
                    candidates.append(p.parent)
        return candidates

    def _parse_voc_xml(self, xml_path: Path) -> list[dict]:
        """Parsea anotaciones XML de formato Pascal VOC."""
        labels = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", "").lower().strip()
            class_name = _CLASS_NAME_NORMALIZE.get(name, name)
            if class_name == "empty":
                continue
            bndbox = obj.find("bndbox")
            if bndbox is None:
                continue
            x1 = float(bndbox.findtext("xmin", "0"))
            y1 = float(bndbox.findtext("ymin", "0"))
            x2 = float(bndbox.findtext("xmax", "0"))
            y2 = float(bndbox.findtext("ymax", "0"))
            labels.append({
                "class_name": class_name,
                "bbox": [x1, y1, x2, y2],
            })
        return labels

    def _load_csv_labels(self, image_name: str) -> list[dict]:
        """Carga etiquetas desde CSV para un nombre de imagen dado."""
        if self._csv_data is None:
            csv_path = self._find_csv()
            if csv_path is None:
                return []
            self._csv_data = pd.read_csv(csv_path)
            self._csv_data.columns = [c.strip().lower() for c in self._csv_data.columns]
            # Pre-extraer solo el nombre de archivo de cualquier prefijo de ruta (ej. "original/imgs/foo.jpg" → "foo.jpg")
            fname_col = self._detect_fname_col(self._csv_data)
            if fname_col:
                self._csv_data["_basename"] = self._csv_data[fname_col].astype(str).apply(
                    lambda x: Path(x.strip()).name
                )

        df = self._csv_data
        if "_basename" not in df.columns:
            return []

        rows = df[df["_basename"] == image_name]
        labels = []
        for _, row in rows.iterrows():
            # Intentar class_name, luego class, luego label
            cls_raw = row.get("class_name", row.get("class", row.get("label", None)))
            if pd.isna(cls_raw):
                continue
            class_name = _CLASS_NAME_NORMALIZE.get(str(cls_raw).strip().lower(), "unknown")
            if class_name in ("empty", "unknown"):
                continue
            x1 = float(row.get("x_min", row.get("xmin", row.get("x1", 0))))
            y1 = float(row.get("y_min", row.get("ymin", row.get("y1", 0))))
            x2 = float(row.get("x_max", row.get("xmax", row.get("x2", 0))))
            y2 = float(row.get("y_max", row.get("ymax", row.get("y2", 0))))
            if x2 > x1 and y2 > y1:
                labels.append({"class_name": class_name, "bbox": [x1, y1, x2, y2]})
        return labels

    @staticmethod
    def _detect_fname_col(df: pd.DataFrame) -> str | None:
        for col in ("image_name", "filename", "image", "file"):
            if col in df.columns:
                return col
        return None

    def _iter_samples(self) -> Iterator[ImageSample]:
        xml_dir = self._find_xml_dir()
        img_dirs = self._find_images_dir()

        if not img_dirs:
            logger.warning("No image directories found under %s", self.root)
            return

        for img_dir in img_dirs:
            for ext in ("*.jpg", "*.png", "*.jpeg"):
                for img_path in sorted(img_dir.glob(ext)):
                    labels = []

                    # Intentar primero anotación XML
                    if xml_dir:
                        xml_path = xml_dir / f"{img_path.stem}.xml"
                        if xml_path.exists():
                            labels = self._parse_voc_xml(xml_path)

                    # Recurrir a CSV si no se encontraron etiquetas XML
                    if not labels:
                        labels = self._load_csv_labels(img_path.name)

                    yield ImageSample(
                        image_path=img_path,
                        dataset_id=self.dataset_id,
                        labels=labels,
                    )
