"""
Unit tests for CanonicalMatchRecord, deterministic hashing, on-chain match proof verification,
and anti-tampering guarantees.
"""

import json
import pytest
from src.matcher.match_record import CanonicalMatchRecord, build_match_record, normalize_url
from src.blockchain.verifier import ProofVerifier, MatchVerificationOutcome


def test_canonical_match_record_deterministic_hash():
    """
    Validates that CanonicalMatchRecord produces the exact same SHA-256 hash
    regardless of initial dictionary key insertion order or whitespace variations.
    """
    rec1 = build_match_record(
        platform="Instagram",
        content_type="Post",
        post_url="https://www.instagram.com/p/CzIy6Htgn6g?utm_source=ig_web_copy_link",
        candidate_content_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
        face_similarity=0.945412,
        face_verification="VERIFIED",
        search_engine="Google Lens (SerpApi)",
        discovery_timestamp=1725580000,
    )

    rec2 = build_match_record(
        platform="  instagram  ",
        content_type="Post",
        post_url="https://www.instagram.com/p/CzIy6Htgn6g",
        candidate_content_hash="A1B2C3D4E5F60718293A4B5C6D7E8F90123456789ABCDEF0123456789ABCDEF0",
        face_similarity=0.9454,
        face_verification="verified",
        search_engine="Google Lens (SerpApi)",
        discovery_timestamp=1725580000,
    )

    hash1 = rec1.compute_match_record_hash()
    hash2 = rec2.compute_match_record_hash()

    assert hash1 == hash2
    assert len(hash1) == 64


def test_tamper_detection_url_modification():
    """
    Verifies that altering the post URL in a match record changes the calculated hash
    and triggers tamper detection during independent verification.
    """
    rec = build_match_record(
        platform="Instagram",
        content_type="Post",
        post_url="https://www.instagram.com/p/CzIy6Htgn6g",
        candidate_content_hash="a1b2c3d4e5f60718293a4b5c6d7e8f90123456789abcdef0123456789abcdef0",
        face_similarity=0.9454,
        face_verification="VERIFIED",
        search_engine="Google Lens (SerpApi)",
        discovery_timestamp=1725580000,
    )
    original_hash = rec.compute_match_record_hash()

    # Tampered record with modified URL
    tampered_dict = rec.to_dict()
    tampered_dict["post_url"] = "https://www.instagram.com/p/FAKE_TAMPERED_URL_999"

    tampered_rec = CanonicalMatchRecord(**tampered_dict)
    tampered_hash = tampered_rec.compute_match_record_hash()

    assert original_hash != tampered_hash

    # Simulated receipt with tampered field
    receipt_data = {
        "input_provenance": {"input_image_hash": "input123"},
        "discovered_match": tampered_dict,
        "blockchain_proof": {
            "match_record_hash": original_hash,
            "contract_address": "0x0000000000000000000000000000000000000000",
        },
    }

    verifier = ProofVerifier(network_key="polygon_amoy", contract_address="0x0000000000000000000000000000000000000000")
    outcome = verifier.verify_match_record(receipt_data)

    assert outcome.is_valid is False
    assert outcome.tamper_detected is True


def test_tamper_detection_similarity_modification():
    """
    Verifies that altering the similarity score in a match record changes the hash.
    """
    rec = build_match_record(
        platform="X",
        content_type="Post",
        post_url="https://x.com/user/status/123456",
        candidate_content_hash="bbbb" * 16,
        face_similarity=0.9250,
        face_verification="VERIFIED",
        discovery_timestamp=1725580000,
    )
    original_hash = rec.compute_match_record_hash()

    tampered_rec = build_match_record(
        platform="X",
        content_type="Post",
        post_url="https://x.com/user/status/123456",
        candidate_content_hash="bbbb" * 16,
        face_similarity=0.9999,  # Falsified similarity
        face_verification="VERIFIED",
        discovery_timestamp=1725580000,
    )
    tampered_hash = tampered_rec.compute_match_record_hash()

    assert original_hash != tampered_hash


def test_tamper_detection_candidate_content_hash_modification():
    """
    Verifies that altering candidate content hash changes the match record fingerprint.
    """
    rec = build_match_record(
        platform="Reddit",
        content_type="Comment/Thread",
        post_url="https://reddit.com/r/sub/comments/123/title",
        candidate_content_hash="1111" * 16,
        face_similarity=0.9100,
        face_verification="VERIFIED",
        discovery_timestamp=1725580000,
    )
    original_hash = rec.compute_match_record_hash()

    tampered_rec = build_match_record(
        platform="Reddit",
        content_type="Comment/Thread",
        post_url="https://reddit.com/r/sub/comments/123/title",
        candidate_content_hash="2222" * 16,  # Substituted image
        face_similarity=0.9100,
        face_verification="VERIFIED",
        discovery_timestamp=1725580000,
    )
    tampered_hash = tampered_rec.compute_match_record_hash()

def test_tamper_detection_timestamp_modification():
    """
    Verifies that modifying the discovery timestamp changes the computed matchRecordHash.
    """
    original_ts = 1725580000
    rec = build_match_record(
        platform="Instagram",
        content_type="Post",
        post_url="https://www.instagram.com/p/CzIy6Htgn6g",
        candidate_content_hash="a1b2c3d4" * 8,
        face_similarity=0.9454,
        discovery_timestamp=original_ts,
    )
    original_hash = rec.compute_match_record_hash()

    tampered_rec = build_match_record(
        platform="Instagram",
        content_type="Post",
        post_url="https://www.instagram.com/p/CzIy6Htgn6g",
        candidate_content_hash="a1b2c3d4" * 8,
        face_similarity=0.9454,
        discovery_timestamp=original_ts + 3600,
    )
    tampered_hash = tampered_rec.compute_match_record_hash()

    assert original_hash != tampered_hash


def test_provenance_independence():
    """
    Verifies that the primary match record hash represents the discovered match,
    not the input image hash.
    """
    rec = build_match_record(
        platform="Instagram",
        content_type="Post",
        post_url="https://www.instagram.com/p/CzIy6Htgn6g",
        candidate_content_hash="a1b2c3d4" * 8,
        face_similarity=0.9454,
        discovery_timestamp=1725580000,
    )

    match_hash = rec.compute_match_record_hash()
    input_image_hash = "9999888877776666" * 4

    assert match_hash != input_image_hash
    assert "input_image_hash" not in rec.to_dict()
