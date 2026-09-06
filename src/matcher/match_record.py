"""
Canonical Match Record and Deterministic Hashing Module.
Builds immutable, deterministic fingerprints for verified social-media matches.
Distinguishes between input image provenance, candidate content hash, and match record hash.
"""

import json
import hashlib
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urlunparse


def normalize_url(url: str) -> str:
    """Normalizes URL for deterministic hashing (strips tracking query params, trailing slashes, normalizes scheme/host)."""
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            pass  # preserve canonical domain
        path = parsed.path.rstrip("/")
        # Filter common tracking query parameters
        clean_query = ""
        if parsed.query:
            query_pairs = [p for p in parsed.query.split("&") if p]
            filtered = [
                p for p in query_pairs
                if not any(p.lower().startswith(prefix) for prefix in ["utm_", "igshid", "fbclid", "s=", "t="])
            ]
            clean_query = "&".join(sorted(filtered))
        return urlunparse((scheme, netloc, path, "", clean_query, ""))
    except Exception:
        return url.strip()


@dataclass
class CanonicalMatchRecord:
    """
    Deterministic representation of a verified social-media match discovery.
    Strictly excludes non-deterministic or transient data (e.g. raw embeddings).
    """
    platform: str
    content_type: str
    post_url: str
    candidate_content_hash: str
    face_similarity: float
    face_verification: str = "VERIFIED"
    search_engine: str = "Reverse Image Search"
    discovery_timestamp: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Returns sorted, normalized dictionary representation."""
        return {
            "candidate_content_hash": str(self.candidate_content_hash).lower().strip(),
            "content_type": str(self.content_type).strip(),
            "discovery_timestamp": int(self.discovery_timestamp),
            "face_similarity": float(round(self.face_similarity, 4)),
            "face_verification": str(self.face_verification).strip().upper(),
            "platform": str(self.platform).strip().upper(),
            "post_url": normalize_url(self.post_url),
            "search_engine": str(self.search_engine).strip(),
        }

    def to_canonical_json(self) -> str:
        """Generates deterministic, canonical JSON string with fixed key ordering and compact separators."""
        data = self.to_dict()
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def compute_match_record_hash(self) -> str:
        """Computes SHA-256 digest of the canonical JSON string."""
        canonical_str = self.to_canonical_json()
        return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


def build_match_record(
    platform: str,
    content_type: str,
    post_url: str,
    candidate_content_hash: str,
    face_similarity: float,
    face_verification: str = "VERIFIED",
    search_engine: str = "Reverse Image Search",
    discovery_timestamp: Optional[int] = None,
) -> CanonicalMatchRecord:
    """Helper function to build a CanonicalMatchRecord instance."""
    import time
    ts = int(discovery_timestamp) if discovery_timestamp is not None else int(time.time())
    return CanonicalMatchRecord(
        platform=platform,
        content_type=content_type,
        post_url=post_url,
        candidate_content_hash=candidate_content_hash,
        face_similarity=face_similarity,
        face_verification=face_verification,
        search_engine=search_engine,
        discovery_timestamp=ts,
    )


@dataclass
class RelatedVerifiedMatch:
    """
    Representation of a secondary verified social media post/reel belonging to the same person.
    Independently verified with YuNet + SFace (>= 70%).
    """
    platform: str
    content_type: str
    post_url: str
    face_similarity: float
    face_verification: str = "VERIFIED"
    candidate_content_hash: str = ""
    search_engine: str = "Profile Discovery"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": str(self.platform).strip().upper(),
            "content_type": str(self.content_type).strip(),
            "post_url": normalize_url(self.post_url),
            "face_similarity": float(round(self.face_similarity, 4)),
            "face_verification": str(self.face_verification).strip().upper(),
            "candidate_content_hash": str(self.candidate_content_hash).lower().strip(),
            "search_engine": str(self.search_engine).strip(),
        }

