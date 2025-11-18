"""
Pigmentation analysis module for detecting spots, freckles, and uneven skin tone.

Uses LAB color space analysis and contour detection to identify
pigmentation irregularities.
"""

import cv2
import numpy as np
from typing import Optional, List
from enum import Enum


class SpotType(str, Enum):
    """Classification of pigmentation spots."""
    FRECKLE = "freckle"
    SMALL_SPOT = "small_spot"
    MEDIUM_SPOT = "medium_spot"
    LARGE_SPOT = "large_spot"  # Could be melasma, age spot
    DARK_CIRCLE = "dark_circle"
    REDNESS = "redness"


class PigmentationAnalyzer:
    """Analyzer for skin pigmentation irregularities."""

    def __init__(
        self,
        dark_threshold_sigma: float = 1.5,
        light_threshold_sigma: float = 1.5,
        min_spot_area: int = 5,
        max_spot_area: int = 1000
    ):
        """
        Initialize pigmentation analyzer.

        Args:
            dark_threshold_sigma: Std deviations below mean for dark spots.
            light_threshold_sigma: Std deviations above mean for light spots.
            min_spot_area: Minimum pixel area for spot detection.
            max_spot_area: Maximum pixel area for spot detection.
        """
        self.dark_threshold_sigma = dark_threshold_sigma
        self.light_threshold_sigma = light_threshold_sigma
        self.min_spot_area = min_spot_area
        self.max_spot_area = max_spot_area

    def analyze(
        self,
        skin_region: np.ndarray,
        skin_mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Analyze skin pigmentation.

        Args:
            skin_region: Skin region image (BGR).
            skin_mask: Optional binary mask for skin pixels.

        Returns:
            Dictionary with:
            - dark_spots: List of detected dark spots
            - light_spots: List of detected light spots
            - redness_areas: Areas with redness
            - evenness_score: Overall evenness (0-1)
            - melanin_distribution: Melanin distribution metrics
        """
        if skin_region is None or skin_region.size == 0:
            return self._empty_result()

        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(skin_region, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Detect dark spots (low L values)
        dark_spots = self._detect_dark_spots(l_channel, skin_mask)

        # Detect light spots (high L values)
        light_spots = self._detect_light_spots(l_channel, skin_mask)

        # Detect redness (high A values in LAB)
        redness = self._detect_redness(a_channel, skin_mask)

        # Calculate evenness score
        evenness = self._calculate_evenness(l_channel, skin_mask)

        # Analyze melanin distribution
        melanin = self._analyze_melanin(l_channel, skin_mask)

        # Calculate overall statistics
        total_spots = len(dark_spots) + len(light_spots)

        return {
            "dark_spots": dark_spots,
            "light_spots": light_spots,
            "redness_areas": redness,
            "total_spot_count": total_spots,
            "dark_spot_count": len(dark_spots),
            "light_spot_count": len(light_spots),
            "evenness_score": float(evenness),
            "melanin_distribution": melanin,
            "has_significant_spots": total_spots > 10
        }

    def _detect_dark_spots(
        self,
        l_channel: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[dict]:
        """
        Detect dark pigmentation spots.

        Args:
            l_channel: L channel from LAB color space.
            mask: Optional skin mask.

        Returns:
            List of spot dictionaries with location, size, and type.
        """
        # Apply mask to get statistics only from skin pixels
        if mask is not None:
            l_masked = l_channel[mask > 0]
        else:
            l_masked = l_channel.flatten()

        if len(l_masked) == 0:
            return []

        # Calculate threshold
        mean_l = np.mean(l_masked)
        std_l = np.std(l_masked)
        threshold = mean_l - (self.dark_threshold_sigma * std_l)

        # Create binary mask for dark spots
        dark_mask = (l_channel < threshold).astype(np.uint8) * 255

        # Apply skin mask if provided
        if mask is not None:
            dark_mask = cv2.bitwise_and(dark_mask, mask)

        # Find contours
        contours, _ = cv2.findContours(
            dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        spots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_spot_area < area < self.max_spot_area:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2

                # Get darkness level
                spot_region = l_channel[y:y+h, x:x+w]
                darkness = float(mean_l - np.mean(spot_region))

                spots.append({
                    "location": (int(center_x), int(center_y)),
                    "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "area": int(area),
                    "type": self._classify_spot(area, darkness),
                    "darkness_level": darkness
                })

        # Sort by area (largest first)
        spots.sort(key=lambda s: s["area"], reverse=True)

        return spots[:100]  # Limit to 100 spots

    def _detect_light_spots(
        self,
        l_channel: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[dict]:
        """
        Detect light/hypopigmented spots.

        Args:
            l_channel: L channel from LAB color space.
            mask: Optional skin mask.

        Returns:
            List of spot dictionaries.
        """
        if mask is not None:
            l_masked = l_channel[mask > 0]
        else:
            l_masked = l_channel.flatten()

        if len(l_masked) == 0:
            return []

        mean_l = np.mean(l_masked)
        std_l = np.std(l_masked)
        threshold = mean_l + (self.light_threshold_sigma * std_l)

        # Create binary mask for light spots
        light_mask = (l_channel > threshold).astype(np.uint8) * 255

        if mask is not None:
            light_mask = cv2.bitwise_and(light_mask, mask)

        contours, _ = cv2.findContours(
            light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        spots = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_spot_area < area < self.max_spot_area:
                x, y, w, h = cv2.boundingRect(cnt)
                center_x = x + w // 2
                center_y = y + h // 2

                spots.append({
                    "location": (int(center_x), int(center_y)),
                    "bbox": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)},
                    "area": int(area),
                    "type": "hypopigmentation"
                })

        spots.sort(key=lambda s: s["area"], reverse=True)
        return spots[:50]

    def _detect_redness(
        self,
        a_channel: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Detect areas of redness.

        The 'a' channel in LAB represents green-red, with higher values being more red.
        """
        if mask is not None:
            a_masked = a_channel[mask > 0]
        else:
            a_masked = a_channel.flatten()

        if len(a_masked) == 0:
            return {"redness_score": 0.0, "area_count": 0}

        mean_a = np.mean(a_masked)
        std_a = np.std(a_masked)

        # Higher 'a' values indicate redness
        threshold = mean_a + (1.5 * std_a)

        red_mask = (a_channel > threshold).astype(np.uint8) * 255

        if mask is not None:
            red_mask = cv2.bitwise_and(red_mask, mask)

        # Find redness areas
        contours, _ = cv2.findContours(
            red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Calculate overall redness
        red_pixels = np.sum(red_mask > 0)
        total_pixels = np.sum(mask > 0) if mask is not None else a_channel.size
        redness_ratio = red_pixels / total_pixels if total_pixels > 0 else 0

        # Normalize redness score
        redness_score = min(1.0, redness_ratio * 10)  # Scale up for sensitivity

        return {
            "redness_score": float(redness_score),
            "redness_percentage": float(redness_ratio * 100),
            "area_count": len(contours),
            "mean_a_value": float(mean_a)
        }

    def _calculate_evenness(
        self,
        l_channel: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate skin tone evenness score.

        Lower variance = more even skin tone = higher score.
        """
        if mask is not None:
            l_masked = l_channel[mask > 0]
        else:
            l_masked = l_channel.flatten()

        if len(l_masked) == 0:
            return 0.0

        mean_l = np.mean(l_masked)
        std_l = np.std(l_masked)

        # Coefficient of variation (CV) - lower is more even
        cv = std_l / mean_l if mean_l > 0 else 1

        # Convert to evenness score (0-1, higher is better)
        # Typical CV for skin is 0.05-0.15
        evenness = max(0, 1 - (cv * 5))

        return float(evenness)

    def _analyze_melanin(
        self,
        l_channel: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Analyze melanin distribution based on L channel.

        Lower L values generally correspond to higher melanin content.
        """
        if mask is not None:
            l_masked = l_channel[mask > 0]
        else:
            l_masked = l_channel.flatten()

        if len(l_masked) == 0:
            return {"melanin_index": 0.0, "distribution": "unknown"}

        mean_l = float(np.mean(l_masked))
        std_l = float(np.std(l_masked))
        min_l = float(np.min(l_masked))
        max_l = float(np.max(l_masked))

        # Melanin index (inverse of lightness, normalized 0-1)
        # L ranges from 0-255 in OpenCV
        melanin_index = (255 - mean_l) / 255

        # Classify distribution
        if std_l < 10:
            distribution = "very_even"
        elif std_l < 20:
            distribution = "even"
        elif std_l < 30:
            distribution = "moderate"
        else:
            distribution = "uneven"

        return {
            "melanin_index": float(melanin_index),
            "mean_lightness": mean_l,
            "lightness_std": std_l,
            "lightness_range": float(max_l - min_l),
            "distribution": distribution
        }

    def _classify_spot(self, area: int, darkness: float) -> str:
        """
        Classify a spot based on its area and darkness.

        Args:
            area: Spot area in pixels.
            darkness: How much darker than average (L units).

        Returns:
            Spot type classification.
        """
        if area < 30:
            return SpotType.FRECKLE
        elif area < 100:
            return SpotType.SMALL_SPOT
        elif area < 300:
            return SpotType.MEDIUM_SPOT
        else:
            return SpotType.LARGE_SPOT

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "dark_spots": [],
            "light_spots": [],
            "redness_areas": {"redness_score": 0.0, "area_count": 0},
            "total_spot_count": 0,
            "dark_spot_count": 0,
            "light_spot_count": 0,
            "evenness_score": 0.0,
            "melanin_distribution": {"melanin_index": 0.0, "distribution": "unknown"},
            "has_significant_spots": False
        }
