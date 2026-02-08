"""CLIP-based Vision-Language Model for traffic scene understanding.

This module uses OpenAI's CLIP to provide:
1. Scene classification (urban, highway, roundabout, etc.)
2. Context understanding (weather, time of day, traffic conditions)
3. Zero-shot vehicle type verification
4. Natural language scene descriptions
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPAnalyzer:
    """CLIP-based vision-language analyzer for traffic scenes."""

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        """Initialize CLIP model.

        Args:
            model_name: CLIP model variant (ViT-B/32, ViT-B/16, ViT-L/14)
            device: Device to run on (cpu, cuda)
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.preprocess = None

        try:
            import clip
            import torch

            self.model, self.preprocess = clip.load(model_name, device=device)
            self.model.eval()
            logger.info(f"Loaded CLIP model: {model_name} on {device}")
        except ImportError:
            logger.warning(
                "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")

    @property
    def is_available(self) -> bool:
        """Check if CLIP model is available."""
        return self.model is not None

    def classify_scene_type(self, image: np.ndarray) -> dict[str, float]:
        """Classify the type of traffic scene.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Dictionary of scene types with confidence scores
        """
        if not self.is_available:
            return {"unknown": 1.0}

        import clip
        import torch

        # Convert BGR to RGB PIL Image
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Scene type prompts
        scene_prompts = [
            "an aerial view of an urban road with traffic",
            "an aerial view of a highway with vehicles",
            "an aerial view of a roundabout with cars",
            "an aerial view of a parking lot",
            "an aerial view of an intersection with traffic lights",
            "an aerial view of a residential street",
        ]

        # Preprocess and encode
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(scene_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values = similarity[0].cpu().numpy()

        # Map to scene types
        scene_types = ["urban_road", "highway", "roundabout", "parking", "intersection", "residential"]
        results = {scene: float(score) for scene, score in zip(scene_types, values)}

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def detect_weather_conditions(self, image: np.ndarray) -> dict[str, float]:
        """Detect weather conditions in the scene.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Dictionary of weather conditions with confidence scores
        """
        if not self.is_available:
            return {"unknown": 1.0}

        import clip
        import torch

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        weather_prompts = [
            "a sunny day with clear skies",
            "a cloudy day with overcast skies",
            "a rainy day with wet roads",
            "a foggy day with low visibility",
            "a day with harsh shadows and bright sunlight",
        ]

        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(weather_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values = similarity[0].cpu().numpy()

        weather_types = ["sunny", "cloudy", "rainy", "foggy", "bright"]
        results = {weather: float(score) for weather, score in zip(weather_types, values)}

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def assess_traffic_density(self, image: np.ndarray) -> dict[str, float]:
        """Assess traffic density level using visual understanding.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Dictionary of density levels with confidence scores
        """
        if not self.is_available:
            return {"unknown": 1.0}

        import clip
        import torch

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        density_prompts = [
            "empty road with no vehicles",
            "light traffic with few vehicles",
            "moderate traffic with several vehicles",
            "heavy traffic with many vehicles",
            "congested traffic with vehicles bumper to bumper",
        ]

        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(density_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values = similarity[0].cpu().numpy()

        density_levels = ["empty", "light", "moderate", "heavy", "congested"]
        results = {level: float(score) for level, score in zip(density_levels, values)}

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    def analyze_scene(self, image: np.ndarray) -> dict[str, Any]:
        """Comprehensive scene analysis combining all CLIP capabilities.

        Args:
            image: BGR image (OpenCV format)

        Returns:
            Dictionary with scene_type, weather, density, and confidence scores
        """
        if not self.is_available:
            return {
                "available": False,
                "message": "CLIP model not loaded",
            }

        scene_types = self.classify_scene_type(image)
        weather = self.detect_weather_conditions(image)
        density = self.assess_traffic_density(image)

        # Get top predictions
        top_scene = max(scene_types.items(), key=lambda x: x[1])
        top_weather = max(weather.items(), key=lambda x: x[1])
        top_density = max(density.items(), key=lambda x: x[1])

        return {
            "available": True,
            "scene_type": {
                "prediction": top_scene[0],
                "confidence": top_scene[1],
                "all_scores": scene_types,
            },
            "weather": {
                "prediction": top_weather[0],
                "confidence": top_weather[1],
                "all_scores": weather,
            },
            "traffic_density": {
                "prediction": top_density[0],
                "confidence": top_density[1],
                "all_scores": density,
            },
        }

    def describe_scene(self, image: np.ndarray, custom_prompts: list[str] | None = None) -> dict[str, float]:
        """Generate scene description using custom text prompts.

        Args:
            image: BGR image (OpenCV format)
            custom_prompts: List of text descriptions to match against

        Returns:
            Dictionary mapping prompts to similarity scores
        """
        if not self.is_available:
            return {"error": 1.0}

        if custom_prompts is None:
            custom_prompts = [
                "safe traffic flow with good spacing",
                "dangerous traffic situation with potential collision",
                "organized traffic following lane rules",
                "chaotic traffic with poor organization",
            ]

        import clip
        import torch

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(custom_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values = similarity[0].cpu().numpy()

        results = {prompt: float(score) for prompt, score in zip(custom_prompts, values)}

        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
