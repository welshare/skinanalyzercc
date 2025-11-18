"""
Face parsing and skin segmentation module.

Provides skin region extraction using color-space based segmentation
and optional deep learning models (BiSeNet).
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class FaceParser:
    """
    Face parser for skin region segmentation.

    Uses HSV/YCrCb color space for skin detection with optional
    deep learning model support.
    """

    # Skin color ranges in different color spaces
    # These are tuned for various skin tones
    HSV_LOWER = np.array([0, 20, 70], dtype=np.uint8)
    HSV_UPPER = np.array([20, 255, 255], dtype=np.uint8)

    YCRCB_LOWER = np.array([0, 135, 85], dtype=np.uint8)
    YCRCB_UPPER = np.array([255, 180, 135], dtype=np.uint8)

    def __init__(self, use_deep_model: bool = False, model_path: Optional[str] = None):
        """
        Initialize the face parser.

        Args:
            use_deep_model: Whether to use BiSeNet deep learning model.
            model_path: Path to BiSeNet model weights.
        """
        self.use_deep_model = use_deep_model
        self.model_path = model_path
        self._model = None

        if use_deep_model and model_path:
            self._load_deep_model()

    def _load_deep_model(self) -> None:
        """Load BiSeNet model for face parsing."""
        try:
            import torch
            import torchvision.transforms as transforms

            # Load model architecture and weights
            # This is a placeholder - actual implementation would load BiSeNet
            print(f"Loading BiSeNet model from {self.model_path}")
            # self._model = torch.load(self.model_path)
            # self._model.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        except ImportError:
            print("PyTorch not available, falling back to color-based segmentation")
            self.use_deep_model = False

    def parse(self, image: np.ndarray, face_bbox: Optional[dict] = None) -> dict:
        """
        Parse face and extract skin regions.

        Args:
            image: BGR image as numpy array.
            face_bbox: Optional face bounding box to focus on.

        Returns:
            Dictionary with:
            - skin_mask: Binary mask of skin regions
            - skin_region: Masked skin image
            - skin_percentage: Percentage of face that is skin
        """
        # Crop to face region if provided
        if face_bbox:
            x, y = face_bbox["x"], face_bbox["y"]
            w, h = face_bbox["width"], face_bbox["height"]
            # Add some padding
            pad = int(min(w, h) * 0.1)
            y1 = max(0, y - pad)
            y2 = min(image.shape[0], y + h + pad)
            x1 = max(0, x - pad)
            x2 = min(image.shape[1], x + w + pad)
            roi = image[y1:y2, x1:x2]
        else:
            roi = image

        if self.use_deep_model and self._model is not None:
            skin_mask = self._parse_with_model(roi)
        else:
            skin_mask = self._parse_with_color(roi)

        # Apply morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        # Extract skin region
        skin_region = cv2.bitwise_and(roi, roi, mask=skin_mask)

        # Calculate skin percentage
        total_pixels = roi.shape[0] * roi.shape[1]
        skin_pixels = np.sum(skin_mask > 0)
        skin_percentage = (skin_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        return {
            "skin_mask": skin_mask,
            "skin_region": skin_region,
            "skin_percentage": float(skin_percentage),
            "roi": roi
        }

    def _parse_with_color(self, image: np.ndarray) -> np.ndarray:
        """
        Segment skin using color-space based detection.

        Combines HSV and YCrCb color spaces for robust detection
        across different skin tones.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask_hsv = cv2.inRange(hsv, self.HSV_LOWER, self.HSV_UPPER)

        # Also check for reddish skin tones (higher hue values)
        hsv_lower2 = np.array([170, 20, 70], dtype=np.uint8)
        hsv_upper2 = np.array([180, 255, 255], dtype=np.uint8)
        mask_hsv2 = cv2.inRange(hsv, hsv_lower2, hsv_upper2)
        mask_hsv = cv2.bitwise_or(mask_hsv, mask_hsv2)

        # Convert to YCrCb
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        mask_ycrcb = cv2.inRange(ycrcb, self.YCRCB_LOWER, self.YCRCB_UPPER)

        # Combine masks (intersection for higher precision)
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        return skin_mask

    def _parse_with_model(self, image: np.ndarray) -> np.ndarray:
        """
        Segment skin using BiSeNet deep learning model.

        Returns a binary mask where skin pixels are 255.
        """
        import torch

        # Preprocess
        input_tensor = self._transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_batch = input_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            output = self._model(input_batch)
            if isinstance(output, tuple):
                output = output[0]

        # Get skin class (class 1 in CelebAMask-HQ)
        parsing = output.squeeze(0).argmax(0).cpu().numpy()
        skin_mask = (parsing == 1).astype(np.uint8) * 255

        # Resize back to original size
        skin_mask = cv2.resize(skin_mask, (image.shape[1], image.shape[0]))

        return skin_mask

    def get_skin_regions(
        self,
        image: np.ndarray,
        skin_mask: np.ndarray
    ) -> dict:
        """
        Extract specific facial skin regions for analysis.

        Args:
            image: Original face image.
            skin_mask: Binary skin mask.

        Returns:
            Dictionary with forehead, cheeks, nose, chin regions.
        """
        h, w = image.shape[:2]

        # Define approximate regions (assuming frontal face)
        regions = {
            "forehead": {
                "y1": 0, "y2": int(h * 0.3),
                "x1": int(w * 0.2), "x2": int(w * 0.8)
            },
            "left_cheek": {
                "y1": int(h * 0.3), "y2": int(h * 0.7),
                "x1": 0, "x2": int(w * 0.35)
            },
            "right_cheek": {
                "y1": int(h * 0.3), "y2": int(h * 0.7),
                "x1": int(w * 0.65), "x2": w
            },
            "nose": {
                "y1": int(h * 0.3), "y2": int(h * 0.65),
                "x1": int(w * 0.35), "x2": int(w * 0.65)
            },
            "chin": {
                "y1": int(h * 0.7), "y2": h,
                "x1": int(w * 0.25), "x2": int(w * 0.75)
            }
        }

        extracted = {}
        for name, coords in regions.items():
            region_mask = skin_mask[
                coords["y1"]:coords["y2"],
                coords["x1"]:coords["x2"]
            ]
            region_image = image[
                coords["y1"]:coords["y2"],
                coords["x1"]:coords["x2"]
            ]

            # Apply mask
            masked_region = cv2.bitwise_and(
                region_image, region_image,
                mask=region_mask
            )

            extracted[name] = {
                "image": masked_region,
                "mask": region_mask,
                "coords": coords
            }

        return extracted
