"""
Unit tests for FaceEngine deep neural face detection (YuNet) and OpenCV SFace 128-D embeddings.
"""

import os
import cv2
import numpy as np
import pytest
from src.face_engine.detector import FaceEngine


def test_face_engine_analysis():
    engine = FaceEngine()
    test_img = "sample_images/who.jpg"
    if not os.path.exists(test_img):
        test_img = "sample_images/unknown.webp"
    assert os.path.exists(test_img)

    result = engine.analyze_image(test_img)
    assert result.image_sha256 is not None
    assert len(result.image_sha256) == 64
    assert result.image_sha256_bytes32.startswith("0x")
    assert len(result.faces) >= 1

    primary = result.primary_face
    assert primary is not None
    assert len(primary.embedding) == 128
    assert primary.confidence > 0.5
    assert len(primary.vector_hash) == 64
    assert primary.embedding_model == "opencv_sface_128d"
    assert primary.detector_model == "opencv_yunet_2023mar"
    assert "left_eye" in primary.landmarks
    assert "right_eye" in primary.landmarks
    assert "nose_tip" in primary.landmarks


def test_no_fake_face_fallback():
    """
    Verifies that an image without any human face (e.g. solid color or blank noise)
    returns 0 faces without manufacturing a central-ROI fake face.
    """
    engine = FaceEngine()
    # Create blank synthetic non-face image
    blank_img = np.zeros((300, 300, 3), dtype=np.uint8)
    blank_path = "sample_images/temp_blank_nonface.jpg"
    cv2.imwrite(blank_path, blank_img)

    try:
        result = engine.analyze_image(blank_path)
        assert len(result.faces) == 0
        assert result.primary_face is None
    finally:
        if os.path.exists(blank_path):
            os.remove(blank_path)


def test_face_engine_missing_file():
    engine = FaceEngine()
    with pytest.raises(FileNotFoundError):
        engine.analyze_image("sample_images/non_existent_file.jpg")
