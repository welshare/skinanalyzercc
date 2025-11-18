"""
Skin tone analysis module.

Analyzes skin color properties including tone classification,
undertone detection, and color distribution.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from enum import Enum


class SkinTone(str, Enum):
    """Skin tone classification based on Fitzpatrick scale approximation."""
    VERY_LIGHT = "very_light"      # Type I
    LIGHT = "light"                 # Type II
    MEDIUM_LIGHT = "medium_light"   # Type III
    MEDIUM = "medium"               # Type IV
    MEDIUM_DARK = "medium_dark"     # Type V
    DARK = "dark"                   # Type VI


class Undertone(str, Enum):
    """Skin undertone classification."""
    COOL = "cool"      # Pink/red/blue undertones
    WARM = "warm"      # Yellow/golden/peachy undertones
    NEUTRAL = "neutral"  # Mix of both


class ToneAnalyzer:
    """Analyzer for skin tone and color properties."""

    def __init__(self):
        """Initialize tone analyzer."""
        pass

    def analyze(
        self,
        skin_region: np.ndarray,
        skin_mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Analyze skin tone and color properties.

        Args:
            skin_region: Skin region image (BGR).
            skin_mask: Optional binary mask for skin pixels.

        Returns:
            Dictionary with:
            - tone: Skin tone classification
            - undertone: Warm/cool/neutral
            - color_metrics: Color space metrics
            - dominant_color: Dominant skin color (BGR)
            - color_histogram: Simplified color distribution
        """
        if skin_region is None or skin_region.size == 0:
            return self._empty_result()

        # Extract skin pixels only
        if skin_mask is not None:
            # Get pixels where mask is non-zero
            skin_pixels = skin_region[skin_mask > 0]
        else:
            skin_pixels = skin_region.reshape(-1, 3)

        if len(skin_pixels) == 0:
            return self._empty_result()

        # Calculate color metrics in different color spaces
        hsv_metrics = self._analyze_hsv(skin_region, skin_mask)
        lab_metrics = self._analyze_lab(skin_region, skin_mask)

        # Classify skin tone
        tone = self._classify_tone(lab_metrics)

        # Detect undertone
        undertone = self._detect_undertone(lab_metrics, hsv_metrics)

        # Get dominant color
        dominant_color = self._get_dominant_color(skin_pixels)

        # Calculate color uniformity
        uniformity = self._calculate_uniformity(skin_pixels)

        return {
            "tone": tone,
            "undertone": undertone,
            "hsv_metrics": hsv_metrics,
            "lab_metrics": lab_metrics,
            "dominant_color": {
                "bgr": dominant_color.tolist(),
                "rgb": dominant_color[::-1].tolist(),
                "hex": self._bgr_to_hex(dominant_color)
            },
            "color_uniformity": float(uniformity),
            "brightness": float(lab_metrics["mean_l"])
        }

    def _analyze_hsv(
        self,
        skin_region: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> dict:
        """Analyze skin in HSV color space."""
        hsv = cv2.cvtColor(skin_region, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if mask is not None:
            h_vals = h[mask > 0]
            s_vals = s[mask > 0]
            v_vals = v[mask > 0]
        else:
            h_vals = h.flatten()
            s_vals = s.flatten()
            v_vals = v.flatten()

        if len(h_vals) == 0:
            return {"mean_h": 0, "mean_s": 0, "mean_v": 0}

        return {
            "mean_h": float(np.mean(h_vals)),
            "mean_s": float(np.mean(s_vals)),
            "mean_v": float(np.mean(v_vals)),
            "std_h": float(np.std(h_vals)),
            "std_s": float(np.std(s_vals)),
            "std_v": float(np.std(v_vals)),
            "median_h": float(np.median(h_vals)),
            "median_s": float(np.median(s_vals)),
            "median_v": float(np.median(v_vals))
        }

    def _analyze_lab(
        self,
        skin_region: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> dict:
        """Analyze skin in LAB color space."""
        lab = cv2.cvtColor(skin_region, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        if mask is not None:
            l_vals = l[mask > 0]
            a_vals = a[mask > 0]
            b_vals = b[mask > 0]
        else:
            l_vals = l.flatten()
            a_vals = a.flatten()
            b_vals = b.flatten()

        if len(l_vals) == 0:
            return {"mean_l": 0, "mean_a": 0, "mean_b": 0}

        return {
            "mean_l": float(np.mean(l_vals)),
            "mean_a": float(np.mean(a_vals)),
            "mean_b": float(np.mean(b_vals)),
            "std_l": float(np.std(l_vals)),
            "std_a": float(np.std(a_vals)),
            "std_b": float(np.std(b_vals)),
            "median_l": float(np.median(l_vals)),
            "median_a": float(np.median(a_vals)),
            "median_b": float(np.median(b_vals))
        }

    def _classify_tone(self, lab_metrics: dict) -> str:
        """
        Classify skin tone based on LAB L channel.

        Uses approximate thresholds based on Fitzpatrick scale mapping.
        """
        mean_l = lab_metrics.get("mean_l", 128)

        # L channel ranges from 0 (black) to 255 (white) in OpenCV
        if mean_l > 200:
            return SkinTone.VERY_LIGHT
        elif mean_l > 175:
            return SkinTone.LIGHT
        elif mean_l > 150:
            return SkinTone.MEDIUM_LIGHT
        elif mean_l > 125:
            return SkinTone.MEDIUM
        elif mean_l > 100:
            return SkinTone.MEDIUM_DARK
        else:
            return SkinTone.DARK

    def _detect_undertone(self, lab_metrics: dict, hsv_metrics: dict) -> str:
        """
        Detect skin undertone (warm/cool/neutral).

        Uses LAB a (green-red) and b (blue-yellow) channels.
        """
        mean_a = lab_metrics.get("mean_a", 128)
        mean_b = lab_metrics.get("mean_b", 128)
        mean_h = hsv_metrics.get("mean_h", 15)

        # In LAB:
        # Higher 'a' = more red (cool undertone)
        # Higher 'b' = more yellow (warm undertone)

        # Normalized around 128 (neutral point in OpenCV LAB)
        a_deviation = mean_a - 128
        b_deviation = mean_b - 128

        # Calculate warm/cool score
        # Positive = warm (yellow), Negative = cool (pink/red)
        undertone_score = b_deviation - (a_deviation * 0.5)

        # Also consider hue - lower hue values tend to be more red/pink
        if mean_h < 10:
            undertone_score -= 5  # Shift towards cool

        if undertone_score > 5:
            return Undertone.WARM
        elif undertone_score < -5:
            return Undertone.COOL
        else:
            return Undertone.NEUTRAL

    def _get_dominant_color(self, pixels: np.ndarray) -> np.ndarray:
        """
        Get the dominant skin color using k-means clustering.

        Args:
            pixels: Nx3 array of BGR pixel values.

        Returns:
            Dominant color as BGR array.
        """
        if len(pixels) == 0:
            return np.array([128, 128, 128], dtype=np.uint8)

        # Convert to float32 for k-means
        pixels_float = pixels.astype(np.float32)

        # Use k-means to find dominant color
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 3  # Find top 3 colors

        try:
            _, labels, centers = cv2.kmeans(
                pixels_float, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )

            # Get the most common cluster (dominant color)
            unique, counts = np.unique(labels, return_counts=True)
            dominant_idx = unique[np.argmax(counts)]
            dominant_color = centers[dominant_idx].astype(np.uint8)

            return dominant_color

        except cv2.error:
            # Fallback to mean color
            return np.mean(pixels, axis=0).astype(np.uint8)

    def _calculate_uniformity(self, pixels: np.ndarray) -> float:
        """
        Calculate color uniformity score.

        Higher score = more uniform skin color.
        """
        if len(pixels) == 0:
            return 0.0

        # Calculate standard deviation for each channel
        std_per_channel = np.std(pixels, axis=0)
        mean_std = np.mean(std_per_channel)

        # Convert to uniformity score (0-1)
        # Lower std = higher uniformity
        # Typical std for skin is 10-30
        uniformity = max(0, 1 - (mean_std / 50))

        return uniformity

    def _bgr_to_hex(self, bgr: np.ndarray) -> str:
        """Convert BGR color to hex string."""
        b, g, r = bgr
        return f"#{int(r):02x}{int(g):02x}{int(b):02x}"

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "tone": SkinTone.MEDIUM,
            "undertone": Undertone.NEUTRAL,
            "hsv_metrics": {"mean_h": 0, "mean_s": 0, "mean_v": 0},
            "lab_metrics": {"mean_l": 0, "mean_a": 0, "mean_b": 0},
            "dominant_color": {
                "bgr": [128, 128, 128],
                "rgb": [128, 128, 128],
                "hex": "#808080"
            },
            "color_uniformity": 0.0,
            "brightness": 0.0
        }

    def get_fitzpatrick_type(self, tone: str) -> int:
        """
        Map skin tone to Fitzpatrick skin type (I-VI).

        Note: This is an approximation based on lightness only.
        True Fitzpatrick typing requires UV response assessment.
        """
        mapping = {
            SkinTone.VERY_LIGHT: 1,
            SkinTone.LIGHT: 2,
            SkinTone.MEDIUM_LIGHT: 3,
            SkinTone.MEDIUM: 4,
            SkinTone.MEDIUM_DARK: 5,
            SkinTone.DARK: 6
        }
        return mapping.get(tone, 3)
