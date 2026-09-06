"""
Robust Face Detection and Deep Neural Feature Embedding Engine.
Utilizes OpenCV DNN YuNet for high-accuracy face detection & 5-point landmark localization,
and OpenCV SFace (SphereFace Deep Neural Network) for 128-d deep facial recognition embeddings.
"""

import os
import urllib.request
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
import numpy as np
import cv2

from ..utils.crypto_utils import (
    compute_file_sha256,
    compute_bytes_sha256,
    compute_vector_sha256,
    to_bytes32_hex,
)
from ..config import BASE_DIR

# Model Download URLs from official OpenCV Zoo
YUNET_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


@dataclass
class DetectedFace:
    """Represents a genuine detected face with its neural embedding and cryptographic hashes."""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    landmarks: Dict[str, Tuple[int, int]]  # 5-point facial landmark coordinates
    embedding: List[float]  # 128-dimensional normalized neural embedding
    confidence: float
    face_crop_hash: str
    vector_hash: str
    face_crop_bytes: bytes = field(repr=False)
    embedding_model: str = "opencv_sface_128d"
    detector_model: str = "opencv_yunet_2023mar"


@dataclass
class ImageAnalysisResult:
    """Complete facial analysis result for an input image."""
    image_path: str
    image_sha256: str
    image_sha256_bytes32: str
    dimensions: Tuple[int, int]
    faces: List[DetectedFace]

    @property
    def primary_face(self) -> Optional[DetectedFace]:
        if not self.faces:
            return None
        # Return the largest face by area
        return max(self.faces, key=lambda f: f.bbox[2] * f.bbox[3])


class FaceEngine:
    """
    Production-grade Deep Neural Face Detection and Recognition Engine.
    Uses YuNet for face detection & landmark localization and SFace for 128-D neural embeddings.
    """

    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = models_dir or os.path.join(BASE_DIR, "models")
        os.makedirs(self.models_dir, exist_ok=True)

        self.yunet_path = os.path.join(self.models_dir, "face_detection_yunet_2023mar.onnx")
        self.sface_path = os.path.join(self.models_dir, "face_recognition_sface_2021dec.onnx")

        self._ensure_models_downloaded()

        # Initialize YuNet Face Detector
        self.detector = cv2.FaceDetectorYN.create(
            model=self.yunet_path,
            config="",
            input_size=(320, 320),
            score_threshold=0.6,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Initialize SFace Face Recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(
            model=self.sface_path,
            config="",
        )

    def _ensure_models_downloaded(self):
        """Ensures the required ONNX neural network weights are present locally."""
        if not os.path.exists(self.yunet_path):
            print(f"[FaceEngine] Downloading YuNet face detection model to {self.yunet_path}...")
            urllib.request.urlretrieve(YUNET_MODEL_URL, self.yunet_path)

        if not os.path.exists(self.sface_path):
            print(f"[FaceEngine] Downloading SFace neural recognition model to {self.sface_path}...")
            urllib.request.urlretrieve(SFACE_MODEL_URL, self.sface_path)

    def analyze_image(self, image_path: str) -> ImageAnalysisResult:
        """
        Loads an image, computes its SHA-256 digest, detects all real faces using YuNet,
        and computes deterministic 128-D deep neural embeddings using SFace.
        If no face is detected, returns an empty faces list (zero fake fallback).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")

        image_hash = compute_file_sha256(image_path)
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise ValueError(f"Failed to decode image from: {image_path}")

        height, width = img_bgr.shape[:2]

        # Configure dynamic input size for YuNet
        self.detector.setInputSize((width, height))
        _, detected_raw = self.detector.detect(img_bgr)

        faces: List[DetectedFace] = []

        # If no faces found, return empty faces list with clear diagnostics
        if detected_raw is None or len(detected_raw) == 0:
            return ImageAnalysisResult(
                image_path=image_path,
                image_sha256=image_hash,
                image_sha256_bytes32=to_bytes32_hex(image_hash),
                dimensions=(width, height),
                faces=[],
            )

        for raw_face in detected_raw:
            # Parse YuNet bounding box
            x, y, w, h = int(raw_face[0]), int(raw_face[1]), int(raw_face[2]), int(raw_face[3])
            confidence = float(raw_face[14])

            # Clamp coordinates to image boundaries
            x_clamped = max(0, min(x, width - 1))
            y_clamped = max(0, min(y, height - 1))
            w_clamped = max(1, min(w, width - x_clamped))
            h_clamped = max(1, min(h, height - y_clamped))

            # Crop face region
            face_roi = img_bgr[y_clamped : y_clamped + h_clamped, x_clamped : x_clamped + w_clamped]
            if face_roi.size == 0:
                continue

            # Compute crop hash
            _, crop_buf = cv2.imencode(".png", face_roi)
            crop_bytes = crop_buf.tobytes()
            crop_hash = compute_bytes_sha256(crop_bytes)

            # Extract 5-point facial landmarks from YuNet
            landmarks: Dict[str, Tuple[int, int]] = {
                "right_eye": (int(raw_face[4]), int(raw_face[5])),
                "left_eye": (int(raw_face[6]), int(raw_face[7])),
                "nose_tip": (int(raw_face[8]), int(raw_face[9])),
                "mouth_right": (int(raw_face[10]), int(raw_face[11])),
                "mouth_left": (int(raw_face[12]), int(raw_face[13])),
            }

            # Align face using 5-point landmarks & extract 128-D SFace deep embedding
            aligned_face = self.recognizer.alignCrop(img_bgr, raw_face)
            feature_vector = self.recognizer.feature(aligned_face)

            # Validate feature vector
            if feature_vector is None or feature_vector.shape != (1, 128):
                raise ValueError("Unexpected feature vector shape from SFace model")

            # Flatten and normalize
            embedding_arr = feature_vector.flatten().astype(np.float64)
            if not np.all(np.isfinite(embedding_arr)):
                raise ValueError("SFace embedding contains non-finite numerical values")

            norm = np.linalg.norm(embedding_arr)
            if norm > 1e-7:
                embedding_arr = embedding_arr / norm

            embedding = embedding_arr.tolist()
            vector_hash = compute_vector_sha256(embedding)

            faces.append(
                DetectedFace(
                    bbox=(x_clamped, y_clamped, w_clamped, h_clamped),
                    landmarks=landmarks,
                    embedding=embedding,
                    confidence=confidence,
                    face_crop_hash=crop_hash,
                    vector_hash=vector_hash,
                    face_crop_bytes=crop_bytes,
                    embedding_model="opencv_sface_128d",
                    detector_model="opencv_yunet_2023mar",
                )
            )

        return ImageAnalysisResult(
            image_path=image_path,
            image_sha256=image_hash,
            image_sha256_bytes32=to_bytes32_hex(image_hash),
            dimensions=(width, height),
            faces=faces,
        )
