"""
Face detection module using OpenCV's YuNet detector.

YuNet is a lightweight face detector (~300KB) optimized for real-time performance.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional
import urllib.request


class FaceDetector:
    """Face detector using OpenCV's YuNet model."""

    MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    MODEL_NAME = "face_detection_yunet_2023mar.onnx"

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        top_k: int = 5000
    ):
        """
        Initialize the face detector.

        Args:
            model_path: Path to YuNet ONNX model. Downloads if not provided.
            confidence_threshold: Minimum confidence for face detection.
            nms_threshold: Non-maximum suppression threshold.
            top_k: Maximum number of faces to detect.
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k

        # Determine model path
        if model_path is None:
            model_dir = Path(__file__).parent.parent.parent / "models"
            model_dir.mkdir(exist_ok=True)
            model_path = model_dir / self.MODEL_NAME

            if not model_path.exists():
                self._download_model(model_path)

        self.model_path = str(model_path)
        self._detector = None
        self._input_size = (320, 320)

    def _download_model(self, model_path: Path) -> None:
        """Download the YuNet model."""
        print(f"Downloading YuNet model to {model_path}...")
        urllib.request.urlretrieve(self.MODEL_URL, model_path)
        print("Download complete.")

    def _get_detector(self, width: int, height: int) -> cv2.FaceDetectorYN:
        """Get or create detector with correct input size."""
        if self._detector is None or self._input_size != (width, height):
            self._input_size = (width, height)
            self._detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                self._input_size,
                self.confidence_threshold,
                self.nms_threshold,
                self.top_k
            )
        return self._detector

    def detect(self, image: np.ndarray) -> dict:
        """
        Detect faces in an image.

        Args:
            image: BGR image as numpy array.

        Returns:
            Dictionary with detection results:
            - has_face: bool
            - face_count: int
            - faces: list of face dictionaries with bbox, confidence, landmarks
        """
        if image is None or image.size == 0:
            return {"has_face": False, "face_count": 0, "faces": []}

        height, width = image.shape[:2]
        detector = self._get_detector(width, height)

        # Detect faces
        _, faces = detector.detect(image)

        if faces is None:
            return {"has_face": False, "face_count": 0, "faces": []}

        results = []
        for face in faces:
            # YuNet output format:
            # [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            bbox = {
                "x": int(face[0]),
                "y": int(face[1]),
                "width": int(face[2]),
                "height": int(face[3])
            }

            landmarks = {
                "right_eye": (float(face[4]), float(face[5])),
                "left_eye": (float(face[6]), float(face[7])),
                "nose_tip": (float(face[8]), float(face[9])),
                "right_mouth": (float(face[10]), float(face[11])),
                "left_mouth": (float(face[12]), float(face[13]))
            }

            confidence = float(face[14])

            results.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "confidence": confidence
            })

        return {
            "has_face": len(results) > 0,
            "face_count": len(results),
            "faces": results
        }

    def extract_face(
        self,
        image: np.ndarray,
        face: dict,
        padding: float = 0.2
    ) -> np.ndarray:
        """
        Extract face region from image with padding.

        Args:
            image: Original image.
            face: Face dictionary from detect().
            padding: Padding ratio around face bbox.

        Returns:
            Cropped face image.
        """
        bbox = face["bbox"]
        h, w = image.shape[:2]

        # Add padding
        pad_w = int(bbox["width"] * padding)
        pad_h = int(bbox["height"] * padding)

        x1 = max(0, bbox["x"] - pad_w)
        y1 = max(0, bbox["y"] - pad_h)
        x2 = min(w, bbox["x"] + bbox["width"] + pad_w)
        y2 = min(h, bbox["y"] + bbox["height"] + pad_h)

        return image[y1:y2, x1:x2].copy()
