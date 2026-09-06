"""
Unit tests for Profile Crawler and Multi-Post Face Verification (Option 2).
Covers:
- profile URL extraction
- direct post URL extraction
- duplicate winner removal
- related candidate >=70% acceptance
- related candidate <70% rejection
- profile crawling failure does not break primary pipeline
- CDN/image URL is not displayed as post_url
"""

import pytest
from unittest.mock import MagicMock, patch

from src.search_engine.profile_crawler import ProfileCrawler
from src.search_engine.reverse_search import SocialMatchCandidate
from src.matcher.face_matcher import CandidateRankingReport, RankedCandidateMatch, MatchEvaluationResult
from src.face_engine.detector import DetectedFace
from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType
from src.matcher.match_record import RelatedVerifiedMatch


def test_extract_profile_handle():
    """Test extracting platform, handle, and profile URL from post URLs."""
    crawler = ProfileCrawler()

    # Instagram Reel
    res_insta = crawler.extract_profile_handle("https://www.instagram.com/reel/C3b4c5d6e7/", author="alex_vlogs")
    assert res_insta is not None
    assert res_insta["platform"] == "instagram"
    assert res_insta["handle"] == "alex_vlogs"
    assert "instagram.com/alex_vlogs" in res_insta["profile_url"]

    # X / Twitter Status
    res_x = crawler.extract_profile_handle("https://x.com/johndoe/status/1234567890")
    assert res_x is not None
    assert res_x["platform"] == "twitter"
    assert res_x["handle"] == "johndoe"
    assert "x.com/johndoe" in res_x["profile_url"]

    # Reddit User
    res_reddit = crawler.extract_profile_handle("https://www.reddit.com/user/reddituser123")
    assert res_reddit is not None
    assert res_reddit["platform"] == "reddit"
    assert res_reddit["handle"] == "reddituser123"


def test_direct_post_url_extraction_not_cdn():
    """Test that discovered candidates use genuine direct post URLs and not CDN thumbnail URLs."""
    primary = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/C3b4c5d6e7",
        platform="instagram",
        title="Alex Reel 1",
        author="alex_vlogs",
        image_url="https://scontent.cdninstagram.com/v/t51.2885-15/thumbnail1.jpg",
        source_engine="Google Lens",
        snippet="Alex's First Reel",
        raw_metadata={},
        is_social_media=True,
    )

    other = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/D4e5f6g7h8",
        platform="instagram",
        title="Alex Reel 2",
        author="alex_vlogs",
        image_url="https://scontent.cdninstagram.com/v/t51.2885-15/thumbnail2.jpg",
        source_engine="Google Lens",
        snippet="Alex's Second Reel",
        raw_metadata={},
        is_social_media=True,
    )

    crawler = ProfileCrawler()
    related = crawler.discover_related_posts(
        primary_candidate=primary,
        existing_discoveries=[primary, other],
    )

    assert len(related) == 1
    # Ensure post_url is the direct Instagram post URL, not the CDN thumbnail link
    assert related[0].post_url == "https://www.instagram.com/reel/D4e5f6g7h8"
    assert "cdninstagram.com" not in related[0].post_url
    assert related[0].image_url == "https://scontent.cdninstagram.com/v/t51.2885-15/thumbnail2.jpg"


def test_duplicate_winner_removal():
    """Test that the primary winner is strictly excluded from related_matches."""
    primary = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/C3b4c5d6e7",
        platform="instagram",
        title="Alex Reel 1",
        author="alex_vlogs",
        image_url="https://cdn.example.com/img1.jpg",
        source_engine="Google Lens",
        snippet="Reel 1",
        raw_metadata={},
        is_social_media=True,
    )

    crawler = ProfileCrawler()
    # If the candidate list only contains the primary winner itself
    related = crawler.discover_related_posts(
        primary_candidate=primary,
        existing_discoveries=[primary],
    )
    assert len(related) == 0


def test_related_candidate_similarity_acceptance_and_rejection():
    """Test that related candidate >=70% is accepted as related_match, and <70% is rejected to other_candidates."""
    face_orig = DetectedFace(
        bbox=(0, 0, 100, 100),
        confidence=0.99,
        landmarks={},
        embedding=[0.1] * 128,
        face_crop_hash="hash_crop",
        vector_hash="hash_orig",
        face_crop_bytes=b"dummy_crop_bytes",
    )


    cand_winner = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/WINNER1",
        platform="instagram",
        title="Winner",
        author="alex",
        image_url="https://cdn.example.com/w.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )
    eval_winner = MatchEvaluationResult(
        is_match=True,
        similarity_score=0.92,
        similarity_int_scaled=9200,
        euclidean_distance=0.4,
        original_vector_hash="hash_orig",
        matched_vector_hash="hash_w",
        matched_image_hash="hash_img_w",
        matched_image_hash_bytes32="0x" + "1" * 64,
        original_face=face_orig,
        candidate_face=face_orig,
        detected_faces_count=1,
        selected_face_index=0,
        details="Match",
    )
    val_winner = UrlValidationResult(
        is_valid_content=True,
        status=UrlStatus.ACCESSIBLE,
        content_type=ContentType.REEL,
        platform="instagram",
        final_url="https://www.instagram.com/reel/WINNER1",
        http_status_code=200,
        reason="OK",
        raw_details={},
    )
    match_winner = RankedCandidateMatch(
        rank=1,
        candidate=cand_winner,
        candidate_image_path="w.jpg",
        evaluation=eval_winner,
        is_match=True,
        similarity_score=0.92,
        is_social_media=True,
        url_validation=val_winner,
        content_type="Reel",
    )

    # Related 1: 85% similarity (Passes >= 70%)
    cand_rel1 = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/REL1_HIGH",
        platform="instagram",
        title="Rel1",
        author="alex",
        image_url="https://cdn.example.com/r1.jpg",
        source_engine="Profile Discovery",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )
    eval_rel1 = MatchEvaluationResult(
        is_match=True,
        similarity_score=0.85,
        similarity_int_scaled=8500,
        euclidean_distance=0.55,
        original_vector_hash="hash_orig",
        matched_vector_hash="hash_r1",
        matched_image_hash="hash_img_r1",
        matched_image_hash_bytes32="0x" + "2" * 64,
        original_face=face_orig,
        candidate_face=face_orig,
        detected_faces_count=1,
        selected_face_index=0,
        details="Match",
    )
    match_rel1 = RankedCandidateMatch(
        rank=2,
        candidate=cand_rel1,
        candidate_image_path="r1.jpg",
        evaluation=eval_rel1,
        is_match=True,
        similarity_score=0.85,
        is_social_media=True,
        url_validation=val_winner,
        content_type="Reel",
    )

    # Related 2: 55% similarity (Fails < 70%)
    cand_rel2 = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/REL2_LOW",
        platform="instagram",
        title="Rel2",
        author="alex",
        image_url="https://cdn.example.com/r2.jpg",
        source_engine="Profile Discovery",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )
    eval_rel2 = MatchEvaluationResult(
        is_match=False,
        similarity_score=0.55,
        similarity_int_scaled=5500,
        euclidean_distance=0.95,
        original_vector_hash="hash_orig",
        matched_vector_hash="hash_r2",
        matched_image_hash="hash_img_r2",
        matched_image_hash_bytes32="0x" + "3" * 64,
        original_face=face_orig,
        candidate_face=face_orig,
        detected_faces_count=1,
        selected_face_index=0,
        details="Below threshold",
    )
    match_rel2 = RankedCandidateMatch(
        rank=3,
        candidate=cand_rel2,
        candidate_image_path="r2.jpg",
        evaluation=eval_rel2,
        is_match=False,
        similarity_score=0.55,
        is_social_media=True,
        url_validation=val_winner,
        content_type="Reel",
    )

    report = CandidateRankingReport(
        ranked_candidates=[match_winner, match_rel1, match_rel2],
        metadata_only_count=0,
        skipped_log=[],
    )

    classified = report.get_classified_results(
        primary_winner=match_winner,
        threshold=0.70,
    )

    # Winner should be match_winner
    assert classified["primary_winner"] == match_winner

    # match_rel1 (85%) should be in related_matches
    assert len(classified["related_matches"]) == 1
    assert classified["related_matches"][0].candidate.post_url == "https://www.instagram.com/reel/REL1_HIGH"

    # match_rel2 (55%) should be in other_candidates
    assert len(classified["other_candidates"]) == 1
    assert classified["other_candidates"][0].candidate.post_url == "https://www.instagram.com/reel/REL2_LOW"


def test_failure_isolation_profile_crawler_does_not_break_pipeline():
    """Test that if profile crawling network fails/raises exception, it returns empty list gracefully."""
    crawler = ProfileCrawler()
    primary = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/C3b4c5d6e7",
        platform="instagram",
        title="Alex Reel",
        author="alex_vlogs",
        image_url="https://cdn.example.com/img.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    with patch.object(crawler.session, "get", side_effect=Exception("Network error")):
        # Should not raise exception
        result = crawler.discover_related_posts(primary_candidate=primary)
        assert isinstance(result, list)
        assert result == []
