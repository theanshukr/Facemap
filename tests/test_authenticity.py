"""
Authenticity and Tamper-Detection Verification Test.
Demonstrates:
1. Legitimate authentic image produces a 100% VALID verification outcome.
2. Tampered/altered image is instantly caught and REJECTED as a MISMATCH.
"""

import os
import sys
import json
import cv2
import numpy as np
from src.face_engine.detector import FaceEngine
from src.blockchain.verifier import ProofVerifier
from src.utils.crypto_utils import compute_file_sha256, to_bytes32_hex


def test_authenticity_and_tamper_detection():
    # 1. Setup original portrait
    original_img_path = "sample_images/who.jpg"
    if not os.path.exists(original_img_path):
        original_img_path = "sample_images/unknown.webp"
    assert os.path.exists(original_img_path)

    # 2. Extract features & create verified proof record
    engine = FaceEngine()
    analysis = engine.analyze_image(original_img_path)
    primary_face = analysis.primary_face

    receipt_file = "test_auth_receipt.json"
    with open(receipt_file, "w") as f:
        json.dump({
            "image_sha256": analysis.image_sha256,
            "face_vector_hash": primary_face.vector_hash,
            "matched_social_url": "https://www.facebook.com/TheMarkJustice",
            "similarity_score": 0.7290,
            "blockchain": {
                "network": "Polygon Amoy Testnet",
                "contract_address": "0x739c9f7F98e210174092bF5cf91C5C533A2a7c4f",
                "tx_hash": "0xaf7861abc4c0bf0115fa80e06c998dd4419c1f887895d0e0f3bdcb075124a198",
                "block_number": 46808500,
            }
        }, f, indent=2)

    verifier = ProofVerifier(network_key="polygon_amoy")

    try:
        # TEST A: Authentic Image -> MUST PASS
        outcome_authentic = verifier.verify_by_receipt_file(receipt_file, original_img_path)
        assert outcome_authentic.is_valid is True
        assert outcome_authentic.image_hash_matched is True

        # TEST B: Tampered / Modified Image -> MUST FAIL
        tampered_img_path = "sample_images/temp_tampered_portrait.jpg"
        img_data = cv2.imread(original_img_path)
        # Tamper with 1 pixel
        img_data[10, 10] = [0, 0, 0]
        cv2.imwrite(tampered_img_path, img_data)

        try:
            outcome_tampered = verifier.verify_by_receipt_file(receipt_file, tampered_img_path)
            assert outcome_tampered.is_valid is False
            assert outcome_tampered.image_hash_matched is False
        finally:
            if os.path.exists(tampered_img_path):
                os.remove(tampered_img_path)

    finally:
        if os.path.exists(receipt_file):
            os.remove(receipt_file)
