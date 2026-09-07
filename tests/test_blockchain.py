"""
Unit tests for Blockchain Notary, Verifier, Smart Contract ABI, and Strict Live Mode Guarantees.
"""

import pytest
from src.blockchain.notary import BlockchainNotary
from src.blockchain.verifier import ProofVerifier
from src.blockchain.contract_abi import FACE_PROOF_REGISTRY_ABI


def test_contract_abi_loaded():
    assert len(FACE_PROOF_REGISTRY_ABI) > 0
    function_names = [item.get("name") for item in FACE_PROOF_REGISTRY_ABI if item.get("type") == "function"]
    assert "registerProof" in function_names
    assert "getProof" in function_names
    assert "getTotalProofs" in function_names


def test_live_demo_strictly_rejects_missing_contract():
    """
    Verifies that when require_live_contract=True is set, notary raises RuntimeError
    if the contract is missing, preventing silent fallback to simulation.
    """
    notary = BlockchainNotary(
        network_key="polygon_amoy",
        contract_address="0x0000000000000000000000000000000000000000",
        private_key="0x" + "1" * 64,
    )
    
    with pytest.raises(RuntimeError) as exc_info:
        notary.anchor_proof(
            image_sha256="a" * 64,
            face_vector_hash="b" * 64,
            matched_image_hash="c" * 64,
            social_post_url="https://x.com/example/status/123",
            similarity_score_scaled=9500,
            metadata={"test": True},
            require_live_contract=True,
        )
    assert "contract" in str(exc_info.value).lower()


def test_live_demo_strictly_rejects_missing_private_key():
    """
    Verifies that require_live_contract=True raises RuntimeError if PRIVATE_KEY is absent.
    """
    notary = BlockchainNotary(
        network_key="polygon_amoy",
        contract_address="0x0000000000000000000000000000000000000000",
        private_key="",
    )
    with pytest.raises(RuntimeError) as exc_info:
        notary.anchor_proof(
            image_sha256="a" * 64,
            face_vector_hash="b" * 64,
            matched_image_hash="c" * 64,
            social_post_url="https://x.com/example/status/123",
            similarity_score_scaled=9500,
            metadata={"test": True},
            require_live_contract=True,
        )
    assert "private_key" in str(exc_info.value).lower()


def test_contract_verifier_invalid_address():
    verifier = ProofVerifier(network_key="polygon_amoy", contract_address="0xInvalid")
    test_img = "sample_images/who.jpg" if os.path.exists("sample_images/who.jpg") else "sample_images/unknown.webp"
    outcome = verifier.verify_image_by_contract(test_img, "0xInvalid")
    assert outcome.is_valid is False
    assert "Invalid contract address" in outcome.status
