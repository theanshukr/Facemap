"""
Facial Cross-Verification, Metric Evaluation, and Multi-Candidate Ranking Agent.
Calculates authentic Cosine Similarity and L2 Euclidean distance between 128-D neural face embeddings.
Ranks all discovered reverse-image candidates and ensures only genuine facial identity matches qualify.
Prioritizes genuine social media platform matches (Facebook, Instagram, X, LinkedIn, Reddit, etc.) over generic web aggregators.
"""

import os
from typing import Optional, List, Tuple, Callable, Any, Dict
from dataclasses import dataclass
import numpy as np

from ..face_engine.detector import FaceEngine, DetectedFace, ImageAnalysisResult
from ..config import SIMILARITY_THRESHOLD
from ..utils.crypto_utils import compute_file_sha256, to_bytes32_hex
from ..search_engine.social_validator import SocialUrlValidator, UrlStatus, ContentType, UrlValidationResult
from ..search_engine.social_extractor import SocialMediaExtractor


@dataclass
class MatchEvaluationResult:
    """Detailed evaluation metrics between original face and candidate face."""
    is_match: bool
    similarity_score: float  # True Cosine similarity [-1.0, 1.0]
    similarity_int_scaled: int  # Scaled for smart contract (e.g. 7289 = 72.89%)
    euclidean_distance: float
    original_vector_hash: str
    matched_vector_hash: str
    matched_image_hash: str
    matched_image_hash_bytes32: str
    original_face: DetectedFace
    candidate_face: Optional[DetectedFace]
    detected_faces_count: int
    selected_face_index: int
    details: str


@dataclass
class RankedCandidateMatch:
    """Represents a candidate ranked by facial similarity against the input photo."""
    rank: int
    candidate: Any  # SocialMatchCandidate
    candidate_image_path: Optional[str]
    evaluation: MatchEvaluationResult
    is_match: bool
    similarity_score: float
    is_social_media: bool = False
    url_validation: Optional[UrlValidationResult] = None
    content_type: str = ""


@dataclass
class CandidateRankingReport:
    """Comprehensive report containing evaluated biometric candidates and skipped discoveries."""
    ranked_candidates: List[RankedCandidateMatch]
    metadata_only_count: int
    skipped_log: List[Dict[str, str]]

    def get_verified_social_matches(self, threshold: Optional[float] = None) -> List[RankedCandidateMatch]:
        """Returns all genuine social media candidates meeting biometric threshold (>= 70%) and accessibility."""
        th = threshold if threshold is not None else SIMILARITY_THRESHOLD
        res = [
            c for c in self.ranked_candidates
            if c.is_social_media and c.is_match and c.similarity_score >= th
            and c.candidate_image_path and c.evaluation and c.evaluation.detected_faces_count >= 1
            and (c.url_validation is None or (c.url_validation.is_valid_content and c.url_validation.status == UrlStatus.ACCESSIBLE))
        ]
        res.sort(key=lambda c: c.similarity_score, reverse=True)
        return res

    def get_verified_web_discoveries(self, threshold: Optional[float] = None) -> List[RankedCandidateMatch]:
        """Returns all non-social web discoveries meeting biometric threshold (>= 70%)."""
        th = threshold if threshold is not None else SIMILARITY_THRESHOLD
        res = [
            c for c in self.ranked_candidates
            if (not c.is_social_media) and c.is_match and c.similarity_score >= th
            and c.candidate_image_path and c.evaluation and c.evaluation.detected_faces_count >= 1
            and (c.url_validation is None or c.url_validation.is_valid_content)
        ]
        res.sort(key=lambda c: c.similarity_score, reverse=True)
        return res

    def select_final_social_match(self, threshold: Optional[float] = None) -> Tuple[Optional[RankedCandidateMatch], str]:
        """
        Implements deterministic Selection Strategy:
        1. Eligibility filter: genuine social media, >= threshold, accessible, downloaded, 1+ face.
        2. Sort strictly by face similarity descending (No platform priority).
        3. Highest similarity = Primary Winner.
        4. If none -> return (None, reason).
        """
        th = threshold if threshold is not None else SIMILARITY_THRESHOLD
        eligible_cands = self.get_verified_social_matches(threshold=th)
        if not eligible_cands:
            return None, f"No eligible social-media candidates found meeting accessibility and biometric criteria (similarity ≥ {th * 100:.0f}%)."

        winner = eligible_cands[0]
        platform_name = (winner.candidate.platform or "Social").upper()
        reason = f"Selected as highest-similarity verified genuine social-media match on {platform_name} with {winner.similarity_score * 100:.2f}% face similarity."
        return winner, reason

    def select_top_non_social_match(self, threshold: Optional[float] = None) -> Tuple[Optional[RankedCandidateMatch], str]:
        """
        Selects the highest similarity non-social / web match (e.g. Letterboxd, Cinema.be, Plex, TV-Media, etc.)
        when no genuine social media post is found.
        """
        th = threshold if threshold is not None else SIMILARITY_THRESHOLD
        eligible_non_social = self.get_verified_web_discoveries(threshold=th)
        if not eligible_non_social:
            return None, f"No non-social web candidates found meeting biometric threshold (similarity ≥ {th * 100:.0f}%)."

        top = eligible_non_social[0]
        plat_name = (top.candidate.platform or "Web").upper()
        reason = f"Selected as top matched non-social web discovery on {plat_name} with {top.similarity_score * 100:.2f}% face similarity (Social match not found)."
        return top, reason


    def get_classified_results(
        self,
        primary_winner: Optional[RankedCandidateMatch] = None,
        additional_ranked: Optional[List[RankedCandidateMatch]] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Categorizes all evaluated candidates into:
        - primary_winner: complete match information for the top social winner
        - related_matches: additional verified social matches (similarity >= threshold) distinct from winner
        - other_candidates: non-social or rejected candidates
        """
        th = threshold if threshold is not None else SIMILARITY_THRESHOLD
        winner = primary_winner or self.select_final_social_match(threshold=th)[0]

        all_ranked = list(self.ranked_candidates)
        if additional_ranked:
            all_ranked.extend(additional_ranked)

        winner_url = SocialMediaExtractor.canonicalize_url(winner.candidate.post_url) if winner else ""
        
        seen_urls = {winner_url} if winner_url else set()
        related: List[RankedCandidateMatch] = []
        others: List[RankedCandidateMatch] = []

        for r in all_ranked:
            c_url = SocialMediaExtractor.canonicalize_url(r.candidate.post_url)
            if not c_url or c_url in seen_urls:
                continue
            seen_urls.add(c_url)

            # Check if it qualifies as related verified content (>= threshold, valid social, 1+ face)
            is_valid_url = r.url_validation.is_valid_content if r.url_validation else True
            is_accessible = (r.url_validation.status == UrlStatus.ACCESSIBLE) if r.url_validation else True
            has_face = r.evaluation.detected_faces_count >= 1 if r.evaluation else False
            
            if r.is_social_media and r.is_match and r.similarity_score >= th and is_valid_url and is_accessible and has_face:
                related.append(r)
            else:
                others.append(r)

        # Sort related by similarity descending
        related.sort(key=lambda x: x.similarity_score, reverse=True)

        return {
            "primary_winner": winner,
            "related_matches": related,
            "other_candidates": others,
        }

    @property
    def top_accessible_social_match(self) -> Optional[RankedCandidateMatch]:
        """Returns the highest-similarity verified candidate on genuine social media that is ACCESSIBLE."""
        winner, _ = self.select_final_social_match()
        return winner

    @property
    def top_social_match(self) -> Optional[RankedCandidateMatch]:
        """
        Returns the top verified social match based on Highest Similarity Genuine Social Match (>= 70%).
        """
        winner, _ = self.select_final_social_match()
        return winner

    @property
    def top_candidate(self) -> Optional[RankedCandidateMatch]:
        """Returns the highest-scoring candidate overall."""
        return self.ranked_candidates[0] if self.ranked_candidates else None



class FaceMatcher:
    """
    Compares deep neural face embeddings across images to verify genuine visual identity match.
    Supports multi-candidate ranking and multi-face candidate image evaluation.
    Enforces URL accessibility and content validation before allowing candidate eligibility.
    """

    def __init__(self, threshold: float = SIMILARITY_THRESHOLD):
        self.threshold = threshold
        self.face_engine = FaceEngine()

    def compare_embeddings(
        self, emb1: List[float], emb2: List[float]
    ) -> Tuple[float, float]:
        """
        Computes standard cosine similarity and Euclidean distance between two embedding vectors.
        """
        v1 = np.array(emb1, dtype=np.float64)
        v2 = np.array(emb2, dtype=np.float64)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0, float("inf")

        u1 = v1 / norm1
        u2 = v2 / norm2

        cos_sim = float(np.dot(u1, u2))
        cos_sim = max(-1.0, min(1.0, cos_sim))

        euc_dist = float(np.linalg.norm(u1 - u2))

        return cos_sim, euc_dist

    def evaluate_match(
        self,
        original_face: DetectedFace,
        candidate_image_path: str,
    ) -> MatchEvaluationResult:
        """
        Runs facial analysis on the candidate image and compares input face against
        ALL detected faces in that image, selecting the highest-scoring face.
        """
        candidate_image_hash = compute_file_sha256(candidate_image_path)
        cand_analysis = self.face_engine.analyze_image(candidate_image_path)

        if not cand_analysis.faces:
            return MatchEvaluationResult(
                is_match=False,
                similarity_score=0.0,
                similarity_int_scaled=0,
                euclidean_distance=float("inf"),
                original_vector_hash=original_face.vector_hash,
                matched_vector_hash="",
                matched_image_hash=candidate_image_hash,
                matched_image_hash_bytes32=to_bytes32_hex(candidate_image_hash),
                original_face=original_face,
                candidate_face=None,
                detected_faces_count=0,
                selected_face_index=-1,
                details="No faces detected in candidate image.",
            )

        best_cos_sim = -1.0
        best_euc_dist = float("inf")
        best_cand_face: Optional[DetectedFace] = None
        best_face_idx = 0

        for idx, cand_face in enumerate(cand_analysis.faces):
            cos_sim, euc_dist = self.compare_embeddings(
                original_face.embedding, cand_face.embedding
            )
            if cos_sim > best_cos_sim:
                best_cos_sim = cos_sim
                best_euc_dist = euc_dist
                best_cand_face = cand_face
                best_face_idx = idx

        is_match = best_cos_sim >= self.threshold
        scaled_int = int(round(max(0.0, best_cos_sim) * 10000))

        details = (
            f"Cosine similarity: {best_cos_sim:.4f} (Threshold: {self.threshold:.2f}), "
            f"Euclidean distance: {best_euc_dist:.4f}, Evaluated {len(cand_analysis.faces)} face(s). "
            f"Verdict: {'MATCH' if is_match else 'NO MATCH'}"
        )

        return MatchEvaluationResult(
            is_match=is_match,
            similarity_score=best_cos_sim,
            similarity_int_scaled=scaled_int,
            euclidean_distance=best_euc_dist,
            original_vector_hash=original_face.vector_hash,
            matched_vector_hash=best_cand_face.vector_hash if best_cand_face else "",
            matched_image_hash=candidate_image_hash,
            matched_image_hash_bytes32=to_bytes32_hex(candidate_image_hash),
            original_face=original_face,
            candidate_face=best_cand_face,
            detected_faces_count=len(cand_analysis.faces),
            selected_face_index=best_face_idx,
            details=details,
        )

    def rank_candidates(
        self,
        original_face: DetectedFace,
        candidates: List[Any],
        download_fn: Callable[[Any], Tuple[Optional[str], Optional[str]]],
        log_callback: Optional[Callable[[str], None]] = None,
        validate_urls: bool = True,
    ) -> CandidateRankingReport:
        """
        Validates URLs, downloads candidate images, and evaluates facial similarity against original_face.
        Ensures broken/invalid URLs are skipped and automatically falls back to subsequent candidates.
        Categorizes candidates into eligible ranked matches and skipped discoveries.
        CRITICAL: Never uses input image as fallback.
        """
        evaluated_results: List[RankedCandidateMatch] = []
        skipped_log: List[Dict[str, str]] = []
        metadata_only_count = 0

        for idx, candidate in enumerate(candidates, 1):
            plat_name = str(getattr(candidate, "platform", "web")).upper()
            post_url = str(getattr(candidate, "post_url", ""))

            # 1. URL & Accessibility Validation
            url_val = SocialUrlValidator.validate_social_url(
                post_url,
                check_live=validate_urls,
                allow_web_fallback=not bool(getattr(candidate, "is_social_media", False)),
            )
            candidate.url_validation = url_val
            candidate.content_type = url_val.content_type.value

            if not url_val.is_valid_content:
                reason = f"SKIPPED - {url_val.status.value}: {url_val.reason}"
                skipped_log.append({
                    "candidate_num": str(idx),
                    "url": post_url,
                    "platform": plat_name,
                    "reason": reason,
                })
                if log_callback:
                    log_callback(
                        f"[Candidate #{idx}] Platform: {plat_name} | "
                        f"URL Validation: {url_val.status.value} | "
                        f"Content Type: {url_val.content_type.value} | "
                        f"Decision: SKIPPED ({url_val.reason})"
                    )
                continue

            # 2. Check Candidate Image Availability
            if not candidate.has_image:
                metadata_only_count += 1
                reason = "SKIPPED: no image_url provided by search provider (metadata-only discovery)"
                skipped_log.append({
                    "candidate_num": str(idx),
                    "url": post_url,
                    "platform": plat_name,
                    "reason": reason,
                })
                if log_callback:
                    log_callback(
                        f"[Candidate #{idx}] Platform: {plat_name} | "
                        f"URL Validation: {url_val.status.value} | "
                        f"Content Type: {url_val.content_type.value} | "
                        f"Decision: SKIPPED (No candidate image URL)"
                    )
                continue

            # 3. Download Candidate Image
            cand_img_path, download_err = download_fn(candidate)

            if download_err or not cand_img_path or not os.path.exists(cand_img_path):
                reason = download_err or "SKIPPED: image download failed"
                skipped_log.append({
                    "candidate_num": str(idx),
                    "url": post_url,
                    "platform": plat_name,
                    "reason": reason,
                })
                if log_callback:
                    log_callback(
                        f"[Candidate #{idx}] Platform: {plat_name} | "
                        f"URL Validation: {url_val.status.value} | "
                        f"Content Type: {url_val.content_type.value} | "
                        f"Image Download: FAILED | "
                        f"Decision: SKIPPED ({download_err})"
                    )
                continue

            # 4. Deep Neural Face Detection & Biometric Comparison
            eval_res = self.evaluate_match(original_face, cand_img_path)

            if eval_res.detected_faces_count == 0:
                reason = "SKIPPED: no face detected in candidate image"
                skipped_log.append({
                    "candidate_num": str(idx),
                    "url": post_url,
                    "platform": plat_name,
                    "reason": reason,
                })
                if log_callback:
                    log_callback(
                        f"[Candidate #{idx}] Platform: {plat_name} | "
                        f"URL Validation: {url_val.status.value} | "
                        f"Content Type: {url_val.content_type.value} | "
                        f"Image Download: SUCCESS | "
                        f"Faces: 0 | "
                        f"Decision: SKIPPED (No detectable face)"
                    )
                continue

            if not eval_res.is_match:
                reason = f"SKIPPED: similarity {eval_res.similarity_score * 100:.2f}% below threshold {self.threshold * 100:.0f}%"
                skipped_log.append({
                    "candidate_num": str(idx),
                    "url": post_url,
                    "platform": plat_name,
                    "reason": reason,
                })
                if log_callback:
                    log_callback(
                        f"[Candidate #{idx}] Platform: {plat_name} | "
                        f"URL Validation: {url_val.status.value} | "
                        f"Content Type: {url_val.content_type.value} | "
                        f"Image Download: SUCCESS | "
                        f"Faces: {eval_res.detected_faces_count} | "
                        f"SFace Similarity: {eval_res.similarity_score * 100:.2f}% | "
                        f"Decision: SKIPPED (Below threshold {self.threshold * 100:.0f}%)"
                    )
                continue

            # 5. Candidate is Fully Valid and Eligible
            if log_callback:
                log_callback(
                    f"[Candidate #{idx}] Platform: {plat_name} | "
                    f"URL Validation: {url_val.status.value} | "
                    f"Content Type: {url_val.content_type.value} | "
                    f"Image Download: SUCCESS | "
                    f"Faces: {eval_res.detected_faces_count} | "
                    f"SFace Similarity: {eval_res.similarity_score * 100:.2f}% | "
                    f"Decision: ELIGIBLE MATCH"
                )

            evaluated_results.append(
                RankedCandidateMatch(
                    rank=0,
                    candidate=candidate,
                    candidate_image_path=cand_img_path,
                    evaluation=eval_res,
                    is_match=True,
                    similarity_score=eval_res.similarity_score,
                    is_social_media=bool(getattr(candidate, "is_social_media", False)),
                    url_validation=url_val,
                    content_type=url_val.content_type.value,
                )
            )

        # Sort based on Selection Strategy:
        # 1. Genuine social media candidates with ACCESSIBLE status come first
        # 2. Highest Biometric Similarity Score descending
        def _rank_priority(r: RankedCandidateMatch) -> Tuple[int, int, float]:
            is_soc = 1 if r.is_social_media else 0
            url_st = r.url_validation.status if r.url_validation else UrlStatus.ACCESSIBLE
            access_tier = 2 if url_st == UrlStatus.ACCESSIBLE else (1 if url_st == UrlStatus.BLOCKED_BY_PLATFORM else 0)
            return (is_soc, access_tier, r.similarity_score)

        evaluated_results.sort(key=_rank_priority, reverse=True)

        # Assign ranks
        for rank_idx, item in enumerate(evaluated_results, 1):
            item.rank = rank_idx

        return CandidateRankingReport(
            ranked_candidates=evaluated_results,
            metadata_only_count=metadata_only_count,
            skipped_log=skipped_log,
        )
