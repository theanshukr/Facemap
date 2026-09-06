"""
End-to-End Integration Test for Face ID + Blockchain Verification Pipeline.
Tests the full lifecycle:
Input Face Photo -> Detection + SFace Embedding -> Candidate Matching ->
Canonical Match Record -> Deterministic Hashing -> Simulated Notarization ->
Receipt Generation -> Independent Re-Verification -> PASS.
"""

import os
import json
import pytest
from pathlib import Path

from src.face_engine.detector import FaceEngine
from src.matcher.match_record import build_match_record, CanonicalMatchRecord
from src.blockchain.verifier import ProofVerifier, MatchVerificationOutcome
from src.config import BASE_DIR


def test_end_to_end_verification_lifecycle(tmp_path):
    """
    Validates end-to-end dataflow from face image to canonical match record,
    simulated on-chain anchoring, receipt generation, and independent re-verification.
    """
    image_path = os.path.join(BASE_DIR, "sample_images", "mark.jpg")
    if not os.path.exists(image_path):
        pytest.skip("Sample image mark.jpg not present")

    # 1. Detect & Encode Face
    engine = FaceEngine()
    analysis = engine.analyze_image(image_path)
    assert len(analysis.faces) >= 1
    primary_face = analysis.primary_face
    assert primary_face is not None
    assert len(primary_face.embedding) == 128
    assert len(analysis.image_sha256) == 64

    # 2. Construct Canonical Match Record
    discovery_ts = 1725585000
    candidate_content_hash = "3a7b9c1d5e2f4a6b8c0d2e4f6a8b0c2d4e6f8a0b2c4d6e8f0a2b4c6d8e0f2a4b"
    post_url = "https://www.instagram.com/p/CzIy6Htgn6g?utm_source=ig_web_copy_link"
    similarity = 0.9454

    match_record = build_match_record(
        platform="Instagram",
        content_type="Post",
        post_url=post_url,
        candidate_content_hash=candidate_content_hash,
        face_similarity=similarity,
        face_verification="VERIFIED",
        search_engine="Google Lens (SerpApi)",
        discovery_timestamp=discovery_ts,
    )

    match_record_hash = match_record.compute_match_record_hash()
    assert len(match_record_hash) == 64

    # 3. Simulate Blockchain Receipt
    contract_addr = "0x1aD68F403Da3B0C800CC7A8666bD107efdd0B331"
    receipt_data = {
        "input_provenance": {
            "input_image_path": image_path,
            "input_image_hash": analysis.image_sha256,
            "face_vector_hash": primary_face.vector_hash,
        },
        "discovered_match": match_record.to_dict(),
        "blockchain_proof": {
            "match_record_hash": match_record_hash,
            "network": "Polygon Amoy Testnet",
            "chain_id": 80002,
            "contract_address": contract_addr,
            "transaction_hash": "0x0724d4ab5210720eb7029c983a6249583d23c0121e718f6a1debf36275f76db5",
            "block_number": 46814965,
            "explorer_url": f"https://amoy.polygonscan.com/tx/0x0724d4ab5210720eb7029c983a6249583d23c0121e718f6a1debf36275f76db5",
        },
    }

    receipt_file = tmp_path / "test_live_receipt.json"
    with open(receipt_file, "w", encoding="utf-8") as f:
        json.dump(receipt_data, f, indent=2)

    # 4. Independent Verification
    with open(receipt_file, "r", encoding="utf-8") as f:
        loaded_receipt = json.load(f)

    disc_match = loaded_receipt["discovered_match"]
    reconstructed_rec = CanonicalMatchRecord(**disc_match)
    recalculated_hash = reconstructed_rec.compute_match_record_hash()

    assert recalculated_hash == match_record_hash
    assert recalculated_hash == loaded_receipt["blockchain_proof"]["match_record_hash"]

    # 5. Tamper Verification Failure on Altered Field
    disc_match_tampered = dict(disc_match)
    disc_match_tampered["face_similarity"] = 0.9999
    tampered_rec = CanonicalMatchRecord(**disc_match_tampered)
    tampered_hash = tampered_rec.compute_match_record_hash()

    assert tampered_hash != match_record_hash
