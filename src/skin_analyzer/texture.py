"""
Texture analysis module using GLCM and other classical CV techniques.

Analyzes skin texture properties like smoothness, roughness, pore visibility,
and wrinkle patterns.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from typing import Optional


class TextureAnalyzer:
    """Analyzer for skin texture using GLCM and morphological features."""

    def __init__(
        self,
        glcm_distances: list = [1, 3, 5],
        glcm_angles: list = [0, np.pi/4, np.pi/2, 3*np.pi/4],
        glcm_levels: int = 256
    ):
        """
        Initialize texture analyzer.

        Args:
            glcm_distances: Pixel distances for GLCM computation.
            glcm_angles: Angles for GLCM computation.
            glcm_levels: Number of gray levels for GLCM.
        """
        self.distances = glcm_distances
        self.angles = glcm_angles
        self.levels = glcm_levels

    def analyze(
        self,
        skin_region: np.ndarray,
        skin_mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Analyze skin texture.

        Args:
            skin_region: Skin region image (BGR).
            skin_mask: Optional binary mask for skin pixels.

        Returns:
            Dictionary with texture metrics:
            - glcm_features: GLCM-based texture features
            - roughness_score: Overall roughness (0-1)
            - smoothness_score: Overall smoothness (0-1)
            - pore_analysis: Pore detection results
            - wrinkle_analysis: Wrinkle detection results
        """
        if skin_region is None or skin_region.size == 0:
            return self._empty_result()

        # Convert to grayscale
        if len(skin_region.shape) == 3:
            gray = cv2.cvtColor(skin_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = skin_region.copy()

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # GLCM analysis
        glcm_features = self._compute_glcm_features(enhanced)

        # Pore analysis
        pore_analysis = self._analyze_pores(enhanced, skin_mask)

        # Wrinkle analysis
        wrinkle_analysis = self._analyze_wrinkles(enhanced, skin_mask)

        # Compute overall scores
        smoothness_score = self._compute_smoothness_score(glcm_features, pore_analysis)
        roughness_score = 1 - smoothness_score

        return {
            "glcm_features": glcm_features,
            "roughness_score": float(roughness_score),
            "smoothness_score": float(smoothness_score),
            "pore_analysis": pore_analysis,
            "wrinkle_analysis": wrinkle_analysis
        }

    def _compute_glcm_features(self, gray_image: np.ndarray) -> dict:
        """
        Compute GLCM texture features.

        Returns:
            Dictionary with contrast, dissimilarity, homogeneity,
            energy, correlation, and ASM.
        """
        # Ensure image is uint8 and properly scaled
        if gray_image.max() > 0:
            gray_scaled = ((gray_image / gray_image.max()) * (self.levels - 1)).astype(np.uint8)
        else:
            return self._empty_glcm_features()

        # Compute GLCM
        try:
            glcm = graycomatrix(
                gray_scaled,
                distances=self.distances,
                angles=self.angles,
                levels=self.levels,
                symmetric=True,
                normed=True
            )

            features = {
                "contrast": float(graycoprops(glcm, 'contrast').mean()),
                "dissimilarity": float(graycoprops(glcm, 'dissimilarity').mean()),
                "homogeneity": float(graycoprops(glcm, 'homogeneity').mean()),
                "energy": float(graycoprops(glcm, 'energy').mean()),
                "correlation": float(graycoprops(glcm, 'correlation').mean()),
                "ASM": float(graycoprops(glcm, 'ASM').mean())
            }

            return features

        except Exception as e:
            print(f"GLCM computation error: {e}")
            return self._empty_glcm_features()

    def _analyze_pores(
        self,
        gray_image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Detect and analyze skin pores.

        Uses blob detection to identify pore-like structures.
        """
        # Apply mask if provided
        if mask is not None:
            analysis_region = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            valid_pixels = np.sum(mask > 0)
        else:
            analysis_region = gray_image
            valid_pixels = gray_image.size

        if valid_pixels == 0:
            return {"pore_count": 0, "pore_density": 0.0, "avg_pore_size": 0.0}

        # Configure blob detector for pore detection
        params = cv2.SimpleBlobDetector_Params()

        # Filter by area (pores are small)
        params.filterByArea = True
        params.minArea = 3
        params.maxArea = 150

        # Filter by circularity (pores are roughly circular)
        params.filterByCircularity = True
        params.minCircularity = 0.3

        # Filter by convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5

        # Filter by inertia (roundness)
        params.filterByInertia = True
        params.minInertiaRatio = 0.3

        # Dark blobs (pores appear darker)
        params.filterByColor = True
        params.blobColor = 0

        detector = cv2.SimpleBlobDetector_create(params)

        # Invert image (pores are dark)
        inverted = cv2.bitwise_not(analysis_region)
        keypoints = detector.detect(inverted)

        # Calculate metrics
        pore_count = len(keypoints)
        pore_density = (pore_count / valid_pixels) * 10000  # per 10000 pixels
        avg_size = np.mean([kp.size for kp in keypoints]) if keypoints else 0

        return {
            "pore_count": int(pore_count),
            "pore_density": float(pore_density),
            "avg_pore_size": float(avg_size),
            "pore_locations": [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints[:50]]  # Limit to 50
        }

    def _analyze_wrinkles(
        self,
        gray_image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> dict:
        """
        Detect and analyze wrinkles using edge detection.

        Wrinkles appear as elongated edges/lines in the image.
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)

        # Detect edges using Canny
        edges = cv2.Canny(blurred, 30, 100)

        # Apply mask if provided
        if mask is not None:
            edges = cv2.bitwise_and(edges, edges, mask=mask)
            valid_pixels = np.sum(mask > 0)
        else:
            valid_pixels = gray_image.size

        if valid_pixels == 0:
            return {"wrinkle_density": 0.0, "wrinkle_intensity": 0.0}

        # Use morphological operations to connect nearby edges (wrinkle lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            connected,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=10,
            maxLineGap=5
        )

        line_count = len(lines) if lines is not None else 0

        # Calculate wrinkle density
        edge_pixels = np.sum(edges > 0)
        wrinkle_density = (edge_pixels / valid_pixels) * 100

        # Calculate wrinkle intensity based on edge strength
        if mask is not None:
            masked_edges = edges[mask > 0]
        else:
            masked_edges = edges.flatten()

        wrinkle_intensity = np.mean(masked_edges) / 255.0

        return {
            "wrinkle_density": float(wrinkle_density),
            "wrinkle_intensity": float(wrinkle_intensity),
            "line_count": int(line_count),
            "edge_pixel_ratio": float(edge_pixels / valid_pixels) if valid_pixels > 0 else 0.0
        }

    def _compute_smoothness_score(self, glcm_features: dict, pore_analysis: dict) -> float:
        """
        Compute overall smoothness score from texture features.

        Returns a value between 0 (rough) and 1 (smooth).
        """
        # Weights for different features
        # High homogeneity = smooth
        # High energy = uniform texture = smooth
        # Low contrast = smooth
        # Low pore density = smooth

        homogeneity = glcm_features.get("homogeneity", 0.5)
        energy = glcm_features.get("energy", 0.5)
        contrast = glcm_features.get("contrast", 100)
        pore_density = pore_analysis.get("pore_density", 0)

        # Normalize contrast (typically 0-1000)
        contrast_normalized = 1 - min(contrast / 500, 1)

        # Normalize pore density (typically 0-10)
        pore_normalized = 1 - min(pore_density / 5, 1)

        # Weighted average
        score = (
            0.3 * homogeneity +
            0.2 * energy +
            0.3 * contrast_normalized +
            0.2 * pore_normalized
        )

        return max(0, min(1, score))

    def _empty_result(self) -> dict:
        """Return empty result structure."""
        return {
            "glcm_features": self._empty_glcm_features(),
            "roughness_score": 0.0,
            "smoothness_score": 0.0,
            "pore_analysis": {"pore_count": 0, "pore_density": 0.0, "avg_pore_size": 0.0},
            "wrinkle_analysis": {"wrinkle_density": 0.0, "wrinkle_intensity": 0.0}
        }

    def _empty_glcm_features(self) -> dict:
        """Return empty GLCM features."""
        return {
            "contrast": 0.0,
            "dissimilarity": 0.0,
            "homogeneity": 0.0,
            "energy": 0.0,
            "correlation": 0.0,
            "ASM": 0.0
        }
