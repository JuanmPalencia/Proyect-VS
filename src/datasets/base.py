"""Unified dataset loading interface."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import kagglehub

logger = logging.getLogger(__name__)


@dataclass
class ImageSample:
    """A single image with its ground-truth annotations (if any)."""
    image_path: Path
    dataset_id: str
    labels: list[dict] = field(default_factory=list)  # [{class_name, bbox:[x1,y1,x2,y2]}]


class BaseDatasetLoader:
    """Abstract loader – subclasses implement ``_iter_samples``."""

    def __init__(self, dataset_id: str, root: Path | str | None = None, kaggle_slug: str | None = None):
        self.dataset_id = dataset_id
        if root:
            self.root = Path(root)
        elif kaggle_slug:
            logger.info("Downloading dataset %s from Kaggle...", kaggle_slug)
            self.root = Path(kagglehub.dataset_download(kaggle_slug))
            logger.info("Dataset at %s", self.root)
        else:
            raise ValueError("Provide either root path or kaggle_slug")

    def __iter__(self) -> Iterator[ImageSample]:
        yield from self._iter_samples()

    def _iter_samples(self) -> Iterator[ImageSample]:
        raise NotImplementedError


def load_dataset(dataset_id: str, root: str | None = None) -> BaseDatasetLoader:
    """Factory: return the right loader based on dataset_id."""
    from .yolo_loader import YOLODatasetLoader
    from .voc_loader import VOCDatasetLoader
    from config import DATASETS

    cfg = DATASETS[dataset_id]
    fmt = cfg["format"]
    kaggle_slug = cfg["kaggle_id"]

    if fmt == "yolo":
        return YOLODatasetLoader(dataset_id=dataset_id, root=root, kaggle_slug=kaggle_slug if not root else None)
    elif fmt == "voc":
        return VOCDatasetLoader(dataset_id=dataset_id, root=root, kaggle_slug=kaggle_slug if not root else None,
                                class_map=cfg["classes"])
    else:
        raise ValueError(f"Unknown format: {fmt}")
