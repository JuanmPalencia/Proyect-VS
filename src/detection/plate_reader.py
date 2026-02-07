"""Vehicle identification: license plate OCR + vehicle description.

Privacy-first: only runs when a collision is detected.
Plates are stored as SHA-256 hashes, never in plain text.
"""

from __future__ import annotations

import hashlib
import logging
import re

import cv2
import numpy as np

from src.detection.detector import Detection

logger = logging.getLogger(__name__)

# Spanish plate pattern: 4 digits + 3 letters (e.g., 1234 ABC)
_PLATE_PATTERN = re.compile(r"\b\d{4}\s*[A-Z]{3}\b")
# Also match older format: letter(s) + 4 digits + letter(s)
_PLATE_PATTERN_OLD = re.compile(r"\b[A-Z]{1,2}\s*\d{4}\s*[A-Z]{1,3}\b")


class PlateReader:
    """Reads license plates and generates vehicle descriptions.

    Only activated when a collision is detected (privacy by design).
    """

    def __init__(self):
        self._tesseract_available = False
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._tesseract_available = True
            logger.info("[PlateReader] Tesseract OCR available")
        except Exception:
            logger.warning("[PlateReader] Tesseract not available - plate OCR disabled")

    def read_plate(self, image_crop: np.ndarray) -> str | None:
        """Attempt to read a license plate from a vehicle crop.

        Returns the plate text if found, None otherwise.
        """
        if not self._tesseract_available:
            return None

        import pytesseract

        # Preprocess for OCR
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Threshold
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Run OCR on multiple preprocessed versions
        for img in [enhanced, thresh, gray]:
            text = pytesseract.image_to_string(
                img, config="--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ).strip().upper()

            # Check for Spanish plate patterns
            match = _PLATE_PATTERN.search(text) or _PLATE_PATTERN_OLD.search(text)
            if match:
                plate = match.group(0).replace(" ", "")
                logger.info("[PlateReader] Plate detected: %s***", plate[:3])
                return plate

        return None

    @staticmethod
    def hash_plate(plate_text: str) -> str:
        """Hash a plate number with SHA-256. Never store plain text."""
        return hashlib.sha256(plate_text.encode("utf-8")).hexdigest()

    @staticmethod
    def describe_vehicle(
        image: np.ndarray,
        detection: Detection,
        img_h: int,
        img_w: int,
    ) -> dict:
        """Generate a description of the vehicle from its detection.

        Always available (no OCR needed).
        """
        x1, y1, x2, y2 = map(int, detection.bbox)

        # Ensure bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        bbox_w = x2 - x1
        bbox_h = y2 - y1
        bbox_area = bbox_w * bbox_h
        relative_size = bbox_area / (img_w * img_h) * 100

        # Position in image
        cx, cy = detection.centroid
        h_pos = "left" if cx < img_w / 3 else ("right" if cx > 2 * img_w / 3 else "center")
        v_pos = "upper" if cy < img_h / 3 else ("lower" if cy > 2 * img_h / 3 else "middle")

        # Dominant color from crop
        crop = image[y1:y2, x1:x2]
        dominant_color = _get_dominant_color(crop) if crop.size > 0 else "unknown"

        return {
            "class": detection.class_name,
            "confidence": detection.confidence,
            "position": f"{v_pos}-{h_pos}",
            "relative_size_pct": round(relative_size, 3),
            "dominant_color": dominant_color,
            "bbox_dimensions": [bbox_w, bbox_h],
        }

    def identify_vehicle(
        self,
        image: np.ndarray,
        detection: Detection,
        img_h: int,
        img_w: int,
    ) -> dict:
        """Full vehicle identification (only call when collision detected).

        Returns description + plate hash if plate is found.
        """
        description = self.describe_vehicle(image, detection, img_h, img_w)

        # Attempt plate OCR
        x1, y1, x2, y2 = map(int, detection.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        crop = image[y1:y2, x1:x2]

        plate_text = self.read_plate(crop) if crop.size > 0 else None

        result = {
            **description,
            "plate_detected": plate_text is not None,
        }

        if plate_text:
            result["plate_hash"] = self.hash_plate(plate_text)

        return result


def _get_dominant_color(crop: np.ndarray) -> str:
    """Estimate the dominant color name from a BGR image crop using k-means."""
    if crop.shape[0] < 2 or crop.shape[1] < 2:
        return "unknown"

    # Resize for speed
    small = cv2.resize(crop, (20, 20), interpolation=cv2.INTER_AREA)
    pixels = small.reshape(-1, 3).astype(np.float32)

    # K-means with k=3 to find dominant cluster
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS)

    # Find the largest cluster
    counts = np.bincount(labels.flatten())
    dominant_bgr = centers[counts.argmax()].astype(int)
    b, g, r = int(dominant_bgr[0]), int(dominant_bgr[1]), int(dominant_bgr[2])

    return _bgr_to_color_name(b, g, r)


def _bgr_to_color_name(b: int, g: int, r: int) -> str:
    """Map BGR values to a human-readable color name."""
    # Convert to HSV for better color classification
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Low saturation = grayscale
    if s < 30:
        if v < 60:
            return "black"
        elif v < 170:
            return "gray"
        else:
            return "white"

    # Classify by hue
    if h < 10 or h > 170:
        return "red"
    elif h < 25:
        return "orange"
    elif h < 35:
        return "yellow"
    elif h < 80:
        return "green"
    elif h < 130:
        return "blue"
    elif h < 170:
        return "purple"

    return "unknown"
