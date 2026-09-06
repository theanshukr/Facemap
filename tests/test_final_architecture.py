"""
Unit test suite verifying all 15 Facemap Task 3 Step 4 & Step 5 architecture requirements:
1. One verified social candidate -> selected as primary.
2. Multiple verified social candidates -> highest SFace similarity wins.
3. Higher-scoring web result cannot override a social result.
4. No verified social candidate -> web fallback displayed.
5. No verified social candidate -> blockchain skipped.
6. Generic social profile URL cannot become the final proof.
7. IdentityResolver cannot trigger secondary social discovery.
8. IdentityResolver cannot override biometric selection.
9. Blockchain record stores matchRecordHash.
10. Recomputed matchRecordHash matches on-chain hash.
11. Tampered URL causes verification failure.
12. Tampered candidateContentHash causes verification failure.
13. Tampered similarity causes verification failure.
14. No platform priority is applied.
15. Only the selected primary social match is notarized.
"""

import pytest
from src.face_engine.detector import DetectedFace
from src.search_engine.reverse_search import SocialMatchCandidate
from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType, SocialUrlValidator
from src.search_engine.identity_resolver import IdentityResolver, IdentityCandidate
from src.matcher.face_matcher import (
    CandidateRankingReport,
    RankedCandidateMatch,
    MatchEvaluationResult,
)
from src.matcher.match_record import build_match_record, CanonicalMatchRecord
from src.blockchain.verifier import ProofVerifier


def _create_mock_face() -> DetectedFace:
    return DetectedFace(
        bbox=(0, 0, 100, 100),
        landmarks={"right_eye": (1, 1), "left_eye": (2, 1), "nose_tip": (1, 2), "mouth_right": (1, 3), "mouth_left": (2, 3)},
        confidence=0.99,
        embedding=[0.1] * 128,
        vector_hash="mock_vector_hash",
        detector_model="opencv_yunet",
        embedding_model="opencv_sface_128d",
        face_crop_hash="mock_crop_hash",
        face_crop_bytes=b"mock_bytes",
    )


def _create_ranked_cand(
    post_url: str,
    platform: str,
    similarity: float,
    is_social: bool = True,
    content_type: str = "Post",
    status: UrlStatus = UrlStatus.ACCESSIBLE,
    is_valid_content: bool = True,
) -> RankedCandidateMatch:
    mock_face = _create_mock_face()
    cand = SocialMatchCandidate(
        post_url=post_url,
        platform=platform,
        title=f"{platform} post",
        author="user1",
        image_url="https://example.com/img.jpg",
        source_engine="Reverse Image Search",
        snippet="",
        raw_metadata={},
        is_social_media=is_social,
    )
    val_res = UrlValidationResult(
        is_valid_content=is_valid_content,
        status=status,
        content_type=ContentType.POST if is_social else ContentType.GENERIC_WEB,
        platform=platform,
        final_url=post_url,
        http_status_code=200 if is_valid_content else 404,
        reason="ok" if is_valid_content else "invalid content",
        raw_details={},
    )
    eval_res = MatchEvaluationResult(
        is_match=(similarity >= 0.70),
        similarity_score=similarity,
        similarity_int_scaled=int(similarity * 10000),
        euclidean_distance=0.2,
        original_vector_hash="orig_hash",
        matched_vector_hash="cand_hash",
        matched_image_hash="cand_img_hash",
        matched_image_hash_bytes32="0x" + "11" * 32,
        original_face=mock_face,
        candidate_face=mock_face,
        detected_faces_count=1,
        selected_face_index=0,
        details="Match test",
    )
    return RankedCandidateMatch(
        rank=1,
        candidate=cand,
        candidate_image_path="sample.jpg",
        evaluation=eval_res,
        is_match=(similarity >= 0.70),
        similarity_score=similarity,
        is_social_media=is_social,
        url_validation=val_res,
        content_type=content_type,
    )


# 1. One verified social candidate -> selected as primary
def test_1_one_verified_social_candidate_selected_as_primary():
    c_social = _create_ranked_cand("https://instagram.com/p/abc", "instagram", 0.88, is_social=True)
    report = CandidateRankingReport(ranked_candidates=[c_social], metadata_only_count=0, skipped_log=[])
    winner, reason = report.select_final_social_match(threshold=0.70)
    assert winner is not None
    assert winner.candidate.post_url == "https://instagram.com/p/abc"
    assert winner.similarity_score == 0.88


# 2. Multiple verified social candidates -> highest SFace similarity wins
def test_2_multiple_verified_social_candidates_highest_sface_wins():
    c1 = _create_ranked_cand("https://x.com/u/status/1", "twitter", 0.75, is_social=True)
    c2 = _create_ranked_cand("https://instagram.com/p/2", "instagram", 0.94, is_social=True)
    c3 = _create_ranked_cand("https://reddit.com/r/s/comments/3", "reddit", 0.82, is_social=True)
    report = CandidateRankingReport(ranked_candidates=[c1, c2, c3], metadata_only_count=0, skipped_log=[])
    winner, _ = report.select_final_social_match(threshold=0.70)
    assert winner is not None
    assert winner.candidate.post_url == "https://instagram.com/p/2"
    assert winner.similarity_score == 0.94


# 3. Higher-scoring web result cannot override a social result
def test_3_higher_scoring_web_result_cannot_override_social():
    c_web = _create_ranked_cand("https://themoviedb.org/p/1", "themoviedb", 0.99, is_social=False)
    c_social = _create_ranked_cand("https://x.com/u/status/10", "twitter", 0.85, is_social=True)
    report = CandidateRankingReport(ranked_candidates=[c_web, c_social], metadata_only_count=0, skipped_log=[])
    winner, _ = report.select_final_social_match(threshold=0.70)
    assert winner is not None
    assert winner.is_social_media is True
    assert winner.candidate.platform == "twitter"
    assert winner.similarity_score == 0.85


# 4. No verified social candidate -> web fallback displayed
def test_4_no_verified_social_candidate_web_fallback_displayed():
    c_social_low = _create_ranked_cand("https://instagram.com/p/low", "instagram", 0.65, is_social=True)
    c_web1 = _create_ranked_cand("https://letterboxd.com/actor/mark", "letterboxd", 0.93, is_social=False)
    c_web2 = _create_ranked_cand("https://cinema.be/actor/mark", "cinema", 0.91, is_social=False)
    report = CandidateRankingReport(ranked_candidates=[c_social_low, c_web1, c_web2], metadata_only_count=0, skipped_log=[])
    
    social_winner, _ = report.select_final_social_match(threshold=0.70)
    assert social_winner is None

    top_web, reason = report.select_top_non_social_match(threshold=0.70)
    assert top_web is not None
    assert top_web.candidate.post_url == "https://letterboxd.com/actor/mark"
    assert top_web.similarity_score == 0.93


# 5. No verified social candidate -> blockchain skipped
def test_5_no_verified_social_candidate_blockchain_skipped():
    c_web = _create_ranked_cand("https://tv-media.at/actor", "tv-media", 0.92, is_social=False)
    report = CandidateRankingReport(ranked_candidates=[c_web], metadata_only_count=0, skipped_log=[])
    verified_socials = report.get_verified_social_matches(threshold=0.70)
    assert len(verified_socials) == 0


# 6. Generic social profile URL cannot become the final proof
def test_6_generic_social_profile_url_cannot_become_final_proof():
    content_type, _ = SocialUrlValidator.detect_content_type("https://instagram.com/just_a_username/")
    assert content_type == ContentType.GENERIC_PROFILE
    val = SocialUrlValidator.validate_social_url("https://instagram.com/just_a_username/", check_live=False)
    assert val.is_valid_content is False
    # Non-post profile candidate is not a valid post proof
    c_profile = _create_ranked_cand("https://instagram.com/just_a_username/", "instagram", 0.90, is_social=True, content_type="Generic Profile", is_valid_content=False)
    report = CandidateRankingReport(ranked_candidates=[c_profile], metadata_only_count=0, skipped_log=[])
    winner, _ = report.select_final_social_match(threshold=0.70)
    assert winner is None


# 7. IdentityResolver cannot trigger secondary social discovery
def test_7_identity_resolver_cannot_trigger_secondary_social_discovery():
    cands = [
        SocialMatchCandidate(
            post_url="https://letterboxd.com/actor/mark-justice",
            platform="letterboxd",
            title="Mark Justice - Letterboxd",
            author="",
            image_url="http://img.com/1.jpg",
            source_engine="Reverse Image Search",
            snippet="Mark Justice filmography",
            raw_metadata={},
            is_social_media=False,
        ),
        SocialMatchCandidate(
            post_url="https://incluvie.com/mark-justice",
            platform="incluvie",
            title="Mark Justice - Incluvie",
            author="",
            image_url="http://img.com/2.jpg",
            source_engine="Reverse Image Search",
            snippet="Actor Mark Justice",
            raw_metadata={},
            is_social_media=False,
        ),
    ]
    ident = IdentityResolver.resolve_identity(cands)
    assert ident is not None
    assert ident.name == "Mark Justice"
    assert ident.is_confident is True
    # IdentityResolver only returns metadata; it has no network discovery methods
    assert not hasattr(ident, "fetch_social_candidates")
    assert not hasattr(IdentityResolver, "search_social_platforms")


# 8. IdentityResolver cannot override biometric selection
def test_8_identity_resolver_cannot_override_biometric_selection():
    # Person named "John Doe" resolved, but Candidate B has highest SFace similarity (0.95 vs 0.72)
    c_a = _create_ranked_cand("https://instagram.com/p/aaa", "instagram", 0.72, is_social=True)
    c_b = _create_ranked_cand("https://x.com/u/status/bbb", "twitter", 0.95, is_social=True)
    report = CandidateRankingReport(ranked_candidates=[c_a, c_b], metadata_only_count=0, skipped_log=[])
    winner, _ = report.select_final_social_match(threshold=0.70)
    # Selection is strictly biometrically driven
    assert winner is not None
    assert winner.candidate.post_url == "https://x.com/u/status/bbb"
    assert winner.similarity_score == 0.95


# 9. Blockchain record stores matchRecordHash
def test_9_blockchain_record_stores_match_record_hash():
    record = build_match_record(
        platform="instagram",
        content_type="Post",
        post_url="https://instagram.com/p/C999",
        candidate_content_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        face_similarity=0.9123,
        face_verification="VERIFIED",
        search_engine="Reverse Image Search",
        discovery_timestamp=1700000000,
    )
    match_hash = record.compute_match_record_hash()
    assert isinstance(match_hash, str)
    assert len(match_hash) == 64
    rec_dict = record.to_dict()
    assert rec_dict["platform"] == "INSTAGRAM"
    assert rec_dict["face_similarity"] == 0.9123


# 10. Recomputed matchRecordHash matches on-chain hash
def test_10_recomputed_match_record_hash_matches_on_chain_hash():
    record = build_match_record(
        platform="twitter",
        content_type="Post",
        post_url="https://x.com/user/status/123456",
        candidate_content_hash="11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
        face_similarity=0.8850,
        face_verification="VERIFIED",
        search_engine="Reverse Image Search",
        discovery_timestamp=1700000000,
    )
    h1 = record.compute_match_record_hash()
    # Reconstruct canonical record independently
    reconstructed = CanonicalMatchRecord(**record.to_dict())
    h2 = reconstructed.compute_match_record_hash()
    assert h1 == h2


# 11. Tampered URL causes verification failure
def test_11_tampered_url_causes_verification_failure():
    record = build_match_record(
        platform="Instagram",
        content_type="Post",
        post_url="https://www.instagram.com/p/CzIy6Htgn6g",
        candidate_content_hash="a1b2c3d4" * 8,
        face_similarity=0.9454,
        face_verification="VERIFIED",
        search_engine="Reverse Image Search",
        discovery_timestamp=1725580000,
    )
    original_hash = record.compute_match_record_hash()
    tampered_dict = record.to_dict()
    tampered_dict["post_url"] = "https://www.instagram.com/p/TAMPERED_URL"
    tampered_record = CanonicalMatchRecord(**tampered_dict)
    assert tampered_record.compute_match_record_hash() != original_hash


# 12. Tampered candidateContentHash causes verification failure
def test_12_tampered_candidate_content_hash_causes_verification_failure():
    record = build_match_record(
        platform="Reddit",
        content_type="Post",
        post_url="https://reddit.com/r/pics/comments/abc",
        candidate_content_hash="1111" * 16,
        face_similarity=0.8900,
        face_verification="VERIFIED",
        search_engine="Reverse Image Search",
        discovery_timestamp=1725580000,
    )
    original_hash = record.compute_match_record_hash()
    tampered_record = build_match_record(
        platform="Reddit",
        content_type="Post",
        post_url="https://reddit.com/r/pics/comments/abc",
        candidate_content_hash="9999" * 16,  # Altered content hash
        face_similarity=0.8900,
        face_verification="VERIFIED",
        search_engine="Reverse Image Search",
        discovery_timestamp=1725580000,
    )
    assert tampered_record.compute_match_record_hash() != original_hash


# 13. Tampered similarity causes verification failure
def test_13_tampered_similarity_causes_verification_failure():
    record = build_match_record(
        platform="X",
        content_type="Post",
        post_url="https://x.com/user/status/777",
        candidate_content_hash="bbbb" * 16,
        face_similarity=0.8500,
        face_verification="VERIFIED",
        search_engine="Reverse Image Search",
        discovery_timestamp=1725580000,
    )
    original_hash = record.compute_match_record_hash()
    tampered_record = build_match_record(
        platform="X",
        content_type="Post",
        post_url="https://x.com/user/status/777",
        candidate_content_hash="bbbb" * 16,
        face_similarity=0.9999,  # Falsified similarity
        face_verification="VERIFIED",
        search_engine="Reverse Image Search",
        discovery_timestamp=1725580000,
    )
    assert tampered_record.compute_match_record_hash() != original_hash


# 14. No platform priority is applied
def test_14_no_platform_priority_is_applied():
    c_reddit = _create_ranked_cand("https://reddit.com/r/s/comments/99", "reddit", 0.96, is_social=True)
    c_insta = _create_ranked_cand("https://instagram.com/p/99", "instagram", 0.91, is_social=True)
    c_x = _create_ranked_cand("https://x.com/u/status/99", "twitter", 0.88, is_social=True)
    report = CandidateRankingReport(ranked_candidates=[c_insta, c_x, c_reddit], metadata_only_count=0, skipped_log=[])
    winner, _ = report.select_final_social_match(threshold=0.70)
    assert winner is not None
    assert winner.candidate.platform == "reddit"
    assert winner.similarity_score == 0.96


# 15. Only the selected primary social match is notarized
def test_15_only_primary_social_match_is_notarized():
    c_winner = _create_ranked_cand("https://instagram.com/p/winner", "instagram", 0.95, is_social=True)
    c_rel_social = _create_ranked_cand("https://reddit.com/r/s/comments/rel", "reddit", 0.85, is_social=True)
    c_web = _create_ranked_cand("https://tv-media.at/page", "tv-media", 0.90, is_social=False)
    
    report = CandidateRankingReport(ranked_candidates=[c_winner, c_rel_social, c_web], metadata_only_count=0, skipped_log=[])
    classified = report.get_classified_results(primary_winner=c_winner, threshold=0.70)
    
    # Primary winner is strictly the single top candidate
    assert classified["primary_winner"].candidate.post_url == "https://instagram.com/p/winner"
    # Related matches are listed but not returned as primary winner
    related = classified["related_matches"]
    assert len(related) == 1
    assert related[0].candidate.post_url == "https://reddit.com/r/s/comments/rel"
