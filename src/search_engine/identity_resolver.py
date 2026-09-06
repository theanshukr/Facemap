"""
Identity Resolver and Multi-Source Name Convergence Engine.
Derives high-confidence person identity candidates from multiple independent web discoveries
using domain voting, linguistic validation, and noise rejection.
Never hardcodes names; strictly enforces multi-source cross-domain corroboration.
"""

import re
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urlparse

from .name_extractor import DynamicNameExtractor


GENERIC_TITLE_BLACKLIST: Set[str] = {
    "home", "homepage", "profile", "user", "channel", "account",
    "video", "videos", "photo", "photos", "image", "images", "picture", "pictures",
    "watch", "online", "official", "media", "social", "news", "breaking",
    "celebrity", "actor", "actress", "model", "influencer", "director", "producer",
    "movies", "movie", "films", "film", "cast", "crew", "database", "wiki", "wikipedia",
    "latest", "trending", "exclusive", "interview", "press", "conference", "review",
    "episode", "season", "trailer", "teaser", "highlights", "clip", "clips",
    "topic", "community", "discussion", "thread", "comment", "post", "posts",
    "reddit", "twitter", "instagram", "facebook", "youtube", "tiktok", "linkedin",
    "fandom", "letterboxd", "imdb", "plex", "cinema", "gallery", "wallpapers",
    "about", "biography", "contact", "privacy", "terms", "search", "results",
    "who is", "what is", "unknown", "anonymous", "person", "man", "woman", "guy",
}


@dataclass
class IdentityCandidate:
    """Represents a resolved person identity candidate derived from multi-source convergence."""
    name: str
    confidence: float
    source_count: int
    supporting_domains: List[str] = field(default_factory=list)
    supporting_titles: List[str] = field(default_factory=list)
    is_confident: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "confidence": self.confidence,
            "source_count": self.source_count,
            "supporting_domains": self.supporting_domains,
            "supporting_titles": self.supporting_titles,
            "is_confident": self.is_confident,
            "details": self.details,
        }


class IdentityResolver:
    """
    Analyzes multiple independent search discoveries to identify person identity candidates.
    Enforces multi-source convergence: a candidate name is only accepted if supported
    by at least 2 distinct independent domains and passes linguistic & entity validation.
    """

    @classmethod
    def extract_domain(cls, url: str) -> str:
        """Extracts clean primary domain from URL."""
        if not url:
            return ""
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc.lower().split(":")[0]
            if netloc.startswith("www."):
                netloc = netloc[4:]
            return netloc
        except Exception:
            return ""

    @classmethod
    def is_valid_person_name(cls, name_candidate: Optional[str]) -> bool:
        """
        Validates that an extracted string conforms to a genuine human person name structure.
        Rejects generic phrases, numbers, lowercase words, and blacklisted entities.
        """
        if not name_candidate or not isinstance(name_candidate, str):
            return False

        name = name_candidate.strip()
        words = [w for w in re.split(r"\s+", name) if w]

        # Must be 2 to 4 words (e.g. First Last, First Middle Last)
        if not (2 <= len(words) <= 4):
            return False

        for word in words:
            # Each word must be capitalized alphabetic, length >= 2
            if not (word.isalpha() and word[0].isupper() and len(word) >= 2):
                return False
            # Check individual word against blacklist
            if word.lower() in GENERIC_TITLE_BLACKLIST:
                return False

        # Check full combined phrase against blacklist
        if name.lower() in GENERIC_TITLE_BLACKLIST:
            return False

        # Reject common conjunctions or prepositions
        conjunctions = {"and", "or", "the", "in", "on", "at", "for", "with", "by", "of", "from", "to", "is", "a", "an"}
        if any(w.lower() in conjunctions for w in words):
            return False

        return True

    @classmethod
    def resolve_identity(
        cls,
        candidates: List[Any],
        min_sources: int = 2,
        min_confidence: float = 0.60,
    ) -> Optional[IdentityCandidate]:
        """
        Aggregates name signals across candidate discoveries and determines if multi-source
        convergence establishes a confident identity candidate.
        
        Rules:
        - Must have at least `min_sources` (default 2) independent domain sources.
        - Names are normalized and validated.
        - Biometric corroboration (similarity >= 0.70 on candidate image) boosts confidence.
        - Returns IdentityCandidate if consensus is reached, or None.
        """
        if not candidates:
            return None

        # Domain votes: name -> set of independent domains
        name_domains: Dict[str, Set[str]] = {}
        # Titles supporting each name
        name_titles: Dict[str, List[str]] = {}
        # Biometric match flags supporting each name
        name_biometric_hits: Dict[str, int] = {}

        for cand in candidates:
            post_url = getattr(cand, "post_url", None)
            if not post_url and hasattr(cand, "candidate"):
                post_url = getattr(cand.candidate, "post_url", "")
            
            domain = cls.extract_domain(post_url or "")
            if not domain:
                continue

            title = getattr(cand, "title", None)
            if title is None and hasattr(cand, "candidate"):
                title = getattr(cand.candidate, "title", "")

            author = getattr(cand, "author", None)
            if author is None and hasattr(cand, "candidate"):
                author = getattr(cand.candidate, "author", "")

            snippet = getattr(cand, "snippet", None)
            if snippet is None and hasattr(cand, "candidate"):
                snippet = getattr(cand.candidate, "snippet", "")

            # Check if this candidate has a verified biometric score
            sim_score = getattr(cand, "similarity_score", 0.0)
            is_bio_match = bool(sim_score and sim_score >= 0.70)

            # Extract name candidates using multiple signals
            extracted_names: Set[str] = set()

            # Signal 1: Title
            if title:
                n = DynamicNameExtractor.extract_from_title(title)
                if n and cls.is_valid_person_name(n):
                    extracted_names.add(n)

            # Signal 2: Author / Handle
            if author:
                n = DynamicNameExtractor.clean_handle_or_slug(author)
                if n and cls.is_valid_person_name(n):
                    extracted_names.add(n)

            # Signal 3: URL Slug
            if post_url:
                n = DynamicNameExtractor.extract_from_url(post_url)
                if n and cls.is_valid_person_name(n):
                    extracted_names.add(n)

            # Signal 4: Snippet
            if snippet:
                n = DynamicNameExtractor.extract_from_title(snippet)
                if n and cls.is_valid_person_name(n):
                    extracted_names.add(n)

            for raw_name in extracted_names:
                # Normalize name casing & whitespace
                norm_name = " ".join(w.capitalize() for w in raw_name.split())
                if not cls.is_valid_person_name(norm_name):
                    continue

                if norm_name not in name_domains:
                    name_domains[norm_name] = set()
                    name_titles[norm_name] = []
                    name_biometric_hits[norm_name] = 0

                name_domains[norm_name].add(domain)
                if title and title not in name_titles[norm_name]:
                    name_titles[norm_name].append(title)
                if is_bio_match:
                    name_biometric_hits[norm_name] += 1

        if not name_domains:
            return None

        # Sort candidate names by number of independent domain sources descending
        sorted_names = sorted(
            name_domains.keys(),
            key=lambda k: (len(name_domains[k]), name_biometric_hits.get(k, 0)),
            reverse=True,
        )

        top_name = sorted_names[0]
        top_domains = name_domains[top_name]
        top_source_count = len(top_domains)

        if top_source_count < min_sources:
            return None

        # Check for ambiguity if second place exists
        if len(sorted_names) > 1:
            second_name = sorted_names[1]
            second_count = len(name_domains[second_name])
            # If two completely different names tie at the top with same sources, flag ambiguity
            if top_source_count == second_count and top_name.lower() != second_name.lower():
                # Prefer the one with higher biometric hits
                if name_biometric_hits.get(top_name, 0) == name_biometric_hits.get(second_name, 0):
                    return None

        # Compute confidence: base 0.50 + 0.12 per additional domain + 0.15 for biometric validation
        base_conf = 0.50 + (0.12 * top_source_count)
        if name_biometric_hits.get(top_name, 0) > 0:
            base_conf += 0.15
        confidence = min(0.99, base_conf)

        is_confident = bool(top_source_count >= min_sources and confidence >= min_confidence)

        return IdentityCandidate(
            name=top_name,
            confidence=round(confidence, 2),
            source_count=top_source_count,
            supporting_domains=sorted(list(top_domains)),
            supporting_titles=name_titles.get(top_name, [])[:5],
            is_confident=is_confident,
            details={
                "biometric_corroborations": name_biometric_hits.get(top_name, 0),
                "total_candidate_names_evaluated": len(sorted_names),
            },
        )
