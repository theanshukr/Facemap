"""
Cryptographic utility functions for hashing and proof generation.
"""

import hashlib
import json
from typing import Union, List
import numpy as np


def compute_file_sha256(file_path: str) -> str:
    """Compute SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_bytes_sha256(data: bytes) -> str:
    """Compute SHA-256 hash of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_vector_sha256(vector: Union[np.ndarray, List[float]]) -> str:
    """Compute deterministic SHA-256 hash of a face embedding vector."""
    if isinstance(vector, np.ndarray):
        arr = np.round(vector.astype(np.float64), 6).tolist()
    else:
        arr = [round(float(x), 6) for x in vector]
    json_bytes = json.dumps(arr, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


def to_bytes32_hex(hex_str: str) -> str:
    """Format hex string as 0x-prefixed 32-byte (64 hex characters) string."""
    clean = hex_str.lower().replace("0x", "")
    if len(clean) > 64:
        clean = clean[:64]
    else:
        clean = clean.zfill(64)
    return "0x" + clean


def compute_merkle_root(leaf_hashes: List[str]) -> str:
    """Compute SHA-256 Merkle root of a list of hex hash strings."""
    if not leaf_hashes:
        return hashlib.sha256(b"").hexdigest()
    
    current_level = [bytes.fromhex(h.replace("0x", "")) for h in leaf_hashes]
    
    while len(current_level) > 1:
        next_level = []
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            if i + 1 < len(current_level):
                right = current_level[i + 1]
            else:
                right = left  # Duplicate last element if odd number
            
            combined = hashlib.sha256(left + right).digest()
            next_level.append(combined)
        current_level = next_level
        
    return current_level[0].hex()
