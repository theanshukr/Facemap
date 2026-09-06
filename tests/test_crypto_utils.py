"""
Unit tests for cryptographic utilities.
"""

import pytest
import numpy as np
from src.utils.crypto_utils import (
    compute_bytes_sha256,
    compute_vector_sha256,
    to_bytes32_hex,
    compute_merkle_root,
)


def test_compute_bytes_sha256():
    data = b"Hello Blockchain"
    h = compute_bytes_sha256(data)
    assert isinstance(h, str)
    assert len(h) == 64


def test_compute_vector_sha256():
    v1 = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    v2 = [0.1, 0.2, 0.3]
    h1 = compute_vector_sha256(v1)
    h2 = compute_vector_sha256(v2)
    assert h1 == h2
    assert len(h1) == 64


def test_to_bytes32_hex():
    h = "abcd"
    b32 = to_bytes32_hex(h)
    assert b32.startswith("0x")
    assert len(b32) == 66  # "0x" + 64 hex characters


def test_compute_merkle_root():
    hashes = [
        "a" * 64,
        "b" * 64,
        "c" * 64,
    ]
    root = compute_merkle_root(hashes)
    assert isinstance(root, str)
    assert len(root) == 64
