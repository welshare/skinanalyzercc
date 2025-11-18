"""
Main skin analyzer service that orchestrates all analysis components.

This is the primary interface for analyzing facial skin attributes.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Union
import json
from datetime import datetime

from .detector import FaceDetector
from .parser import FaceParser
from .texture import TextureAnalyzer
from .pigmentation import PigmentationAnalyzer
from .tone import ToneAnalyzer


class SkinAnalyzer:
    """
    Complete skin analysis service.

    Combines face detection, skin segmentation, and multiple
    analysis modules to provide comprehensive skin attribute analysis.
    """

    def __init__(
        self,
        face_confidence: float = 0.7,
        use_deep_parsing: bool = False,
        parsing_model_path: Optional[str] = None
    ):
        """
        Initialize the skin analyzer.

        Args:
            face_confidence: Minimum confidence for face detection.
            use_deep_parsing: Use BiSeNet for skin segmentation.
            parsing_model_path: Path to BiSeNet model weights.
        """
        # Initialize all components
        self.face_detector = FaceDetector(confidence_threshold=face_confidence)
        self.face_parser = FaceParser(
            use_deep_model=use_deep_parsing,
            model_path=parsing_model_path
        )
        self.texture_analyzer = TextureAnalyzer()
        self.pigmentation_analyzer = PigmentationAnalyzer()
        self.tone_analyzer = ToneAnalyzer()

    def analyze(
        self,
        image: Union[str, Path, np.ndarray],
        return_visualizations: bool = False
    ) -> dict:
        """
        Perform complete skin analysis on an image.

        Args:
            image: Image path or numpy array (BGR format).
            return_visualizations: Include annotated images in result.

        Returns:
            Complete analysis results dictionary.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                return self._error_result(f"Could not load image: {image}")
        else:
            img = image.copy()

        # Step 1: Detect faces
        detection_result = self.face_detector.detect(img)

        if not detection_result["has_face"]:
            return {
                "success": False,
                "has_face": False,
                "error": "No face detected in image",
                "face_detection": detection_result
            }

        # Analyze the primary (largest) face
        primary_face = max(
            detection_result["faces"],
            key=lambda f: f["bbox"]["width"] * f["bbox"]["height"]
        )

        # Step 2: Extract face region
        face_image = self.face_detector.extract_face(img, primary_face, padding=0.3)

        # Step 3: Parse face and segment skin
        parsing_result = self.face_parser.parse(face_image)
        skin_mask = parsing_result["skin_mask"]
        skin_region = parsing_result["skin_region"]

        # Step 4: Analyze texture
        texture_result = self.texture_analyzer.analyze(skin_region, skin_mask)

        # Step 5: Analyze pigmentation
        pigmentation_result = self.pigmentation_analyzer.analyze(skin_region, skin_mask)

        # Step 6: Analyze tone
        tone_result = self.tone_analyzer.analyze(skin_region, skin_mask)

        # Compile results
        result = {
            "success": True,
            "has_face": True,
            "timestamp": datetime.utcnow().isoformat(),
            "image_size": {
                "width": img.shape[1],
                "height": img.shape[0]
            },
            "face_detection": {
                "face_count": detection_result["face_count"],
                "primary_face": primary_face,
                "confidence": primary_face["confidence"]
            },
            "skin_coverage": {
                "percentage": parsing_result["skin_percentage"]
            },
            "texture_analysis": texture_result,
            "pigmentation_analysis": pigmentation_result,
            "tone_analysis": tone_result,
            "summary": self._generate_summary(
                texture_result, pigmentation_result, tone_result
            )
        }

        # Add visualizations if requested
        if return_visualizations:
            result["visualizations"] = self._create_visualizations(
                face_image, skin_mask, pigmentation_result
            )

        return result

    def analyze_region(
        self,
        image: Union[str, Path, np.ndarray],
        region_name: str = "cheek"
    ) -> dict:
        """
        Analyze a specific facial region.

        Args:
            image: Image path or numpy array.
            region_name: Region to analyze (forehead, left_cheek, right_cheek, nose, chin).

        Returns:
            Analysis results for the specified region.
        """
        # Load and process image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()

        # Detect face
        detection_result = self.face_detector.detect(img)
        if not detection_result["has_face"]:
            return self._error_result("No face detected")

        primary_face = detection_result["faces"][0]
        face_image = self.face_detector.extract_face(img, primary_face)

        # Parse and get regions
        parsing_result = self.face_parser.parse(face_image)
        regions = self.face_parser.get_skin_regions(
            face_image, parsing_result["skin_mask"]
        )

        if region_name not in regions:
            return self._error_result(f"Unknown region: {region_name}")

        region_data = regions[region_name]
        region_image = region_data["image"]
        region_mask = region_data["mask"]

        # Analyze the specific region
        texture = self.texture_analyzer.analyze(region_image, region_mask)
        pigmentation = self.pigmentation_analyzer.analyze(region_image, region_mask)
        tone = self.tone_analyzer.analyze(region_image, region_mask)

        return {
            "success": True,
            "region": region_name,
            "texture": texture,
            "pigmentation": pigmentation,
            "tone": tone
        }

    def quick_check(self, image: Union[str, Path, np.ndarray]) -> dict:
        """
        Perform a quick face presence check.

        Args:
            image: Image path or numpy array.

        Returns:
            Basic check results (face detected, confidence, skin percentage).
        """
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        else:
            img = image.copy()

        detection = self.face_detector.detect(img)

        if not detection["has_face"]:
            return {
                "has_face": False,
                "face_count": 0,
                "is_suitable_for_analysis": False,
                "reason": "No face detected"
            }

        primary_face = max(
            detection["faces"],
            key=lambda f: f["bbox"]["width"] * f["bbox"]["height"]
        )

        # Quick skin check
        face_image = self.face_detector.extract_face(img, primary_face)
        parsing = self.face_parser.parse(face_image)

        # Determine if suitable for analysis
        is_suitable = (
            primary_face["confidence"] > 0.8 and
            parsing["skin_percentage"] > 20
        )

        return {
            "has_face": True,
            "face_count": detection["face_count"],
            "confidence": primary_face["confidence"],
            "skin_percentage": parsing["skin_percentage"],
            "is_suitable_for_analysis": is_suitable,
            "reason": None if is_suitable else "Low confidence or insufficient skin area"
        }

    def _generate_summary(
        self,
        texture: dict,
        pigmentation: dict,
        tone: dict
    ) -> dict:
        """Generate human-readable summary of analysis."""

        # Texture summary
        smoothness = texture.get("smoothness_score", 0)
        if smoothness > 0.7:
            texture_summary = "smooth"
        elif smoothness > 0.4:
            texture_summary = "moderate"
        else:
            texture_summary = "rough"

        # Pigmentation summary
        spot_count = pigmentation.get("total_spot_count", 0)
        evenness = pigmentation.get("evenness_score", 0)

        if spot_count < 5 and evenness > 0.8:
            pigmentation_summary = "even_tone"
        elif spot_count < 20:
            pigmentation_summary = "minor_irregularities"
        else:
            pigmentation_summary = "noticeable_spots"

        # Overall skin condition score (0-100)
        condition_score = (
            (smoothness * 30) +
            (evenness * 30) +
            ((1 - min(spot_count / 50, 1)) * 20) +
            (tone.get("color_uniformity", 0) * 20)
        )

        return {
            "texture": texture_summary,
            "pigmentation": pigmentation_summary,
            "skin_tone": tone.get("tone", "unknown"),
            "undertone": tone.get("undertone", "unknown"),
            "condition_score": round(condition_score, 1),
            "key_findings": self._get_key_findings(texture, pigmentation, tone)
        }

    def _get_key_findings(self, texture: dict, pigmentation: dict, tone: dict) -> list:
        """Extract key findings from analysis."""
        findings = []

        # Texture findings
        pore_density = texture.get("pore_analysis", {}).get("pore_density", 0)
        if pore_density > 3:
            findings.append("Visible pores detected")

        wrinkle_density = texture.get("wrinkle_analysis", {}).get("wrinkle_density", 0)
        if wrinkle_density > 5:
            findings.append("Fine lines detected")

        # Pigmentation findings
        dark_spots = pigmentation.get("dark_spot_count", 0)
        if dark_spots > 10:
            findings.append(f"{dark_spots} dark spots detected")

        redness = pigmentation.get("redness_areas", {}).get("redness_score", 0)
        if redness > 0.3:
            findings.append("Redness detected")

        melanin = pigmentation.get("melanin_distribution", {})
        if melanin.get("distribution") == "uneven":
            findings.append("Uneven melanin distribution")

        # Tone findings
        uniformity = tone.get("color_uniformity", 0)
        if uniformity < 0.5:
            findings.append("Uneven skin tone")

        return findings if findings else ["No significant concerns"]

    def _create_visualizations(
        self,
        face_image: np.ndarray,
        skin_mask: np.ndarray,
        pigmentation: dict
    ) -> dict:
        """Create visualization images for the analysis."""
        visualizations = {}

        # Skin mask overlay
        mask_viz = face_image.copy()
        mask_colored = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 0] = 0  # Remove blue
        mask_colored[:, :, 2] = 0  # Remove red (keep green)
        mask_viz = cv2.addWeighted(mask_viz, 0.7, mask_colored, 0.3, 0)
        visualizations["skin_mask_overlay"] = mask_viz

        # Spot detection overlay
        spots_viz = face_image.copy()
        for spot in pigmentation.get("dark_spots", [])[:50]:
            bbox = spot["bbox"]
            cv2.rectangle(
                spots_viz,
                (bbox["x"], bbox["y"]),
                (bbox["x"] + bbox["width"], bbox["y"] + bbox["height"]),
                (0, 0, 255),  # Red
                1
            )
        visualizations["spots_overlay"] = spots_viz

        return visualizations

    def _error_result(self, message: str) -> dict:
        """Create error result dictionary."""
        return {
            "success": False,
            "has_face": False,
            "error": message
        }

    def save_result(self, result: dict, output_path: Union[str, Path]) -> None:
        """
        Save analysis result to JSON file.

        Args:
            result: Analysis result dictionary.
            output_path: Output file path.
        """
        # Remove non-serializable items (numpy arrays)
        serializable = self._make_serializable(result)

        with open(output_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable types."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()
                    if not isinstance(v, np.ndarray)}
        elif isinstance(obj, list):
            return [self._make_serializable(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
