"""
Unit tests for FaceMatcher, multi-candidate ranking, multi-face verification,
and anti-self-match guarantees.
"""

import os
import pytest
from src.matcher.face_matcher import FaceMatcher, RankedCandidateMatch
from src.face_engine.detector import FaceEngine, DetectedFace
from src.search_engine.reverse_search import SocialMatchCandidate


def test_face_matcher_self_comparison():
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)
    img_path = "sample_images/who.jpg"
    if not os.path.exists(img_path):
        img_path = "sample_images/mark_justice.jpg"

    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    res = matcher.evaluate_match(primary_face, img_path)
    assert res.is_match is True
    assert res.similarity_score >= 0.95
    assert res.similarity_int_scaled >= 9500
    assert res.euclidean_distance < 0.3


def test_positive_vs_negative_face_pair_ranking():
    """
    Validates that same-person image pairs score above the empirical threshold (0.40)
    and rank significantly higher than non-matching faces.
    """
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)

    img1 = "sample_images/mark.jpg"
    img2 = "sample_images/mark_justice.jpg"

    if os.path.exists(img1) and os.path.exists(img2):
        analysis1 = engine.analyze_image(img1)
        res_pos = matcher.evaluate_match(analysis1.primary_face, img2)
        assert res_pos.is_match is True
        assert res_pos.similarity_score >= 0.40

    # Test comparison against orthogonal/random embedding
    emb_target = [1.0] + [0.0] * 127
    emb_diff = [0.0, 1.0] + [0.0] * 126
    cos_sim, dist = matcher.compare_embeddings(emb_target, emb_diff)
    assert cos_sim == 0.0
    assert dist > 1.0


def test_rank_candidates_sorting_and_skipping():
    """
    Validates that rank_candidates:
    1. Evaluates and sorts biometric candidates in descending order.
    2. Correctly categorizes and skips candidates without usable images.
    3. Handles failed downloads gracefully.
    """
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)
    img_path = "sample_images/who.jpg"

    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    candidates = [
        # Candidate 1: Valid image URL that matches
        SocialMatchCandidate(
            post_url="https://x.com/user1/status/111",
            platform="twitter",
            title="Candidate 1",
            author="user1",
            image_url="http://example.com/match.jpg",
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        ),
        # Candidate 2: Valid image URL but download fails
        SocialMatchCandidate(
            post_url="https://x.com/user2/status/222",
            platform="twitter",
            title="Candidate 2",
            author="user2",
            image_url="http://example.com/fail_download.jpg",
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        ),
        # Candidate 3: Metadata only (no image URL)
        SocialMatchCandidate(
            post_url="https://x.com/user3/status/333",
            platform="twitter",
            title="Candidate 3",
            author="user3",
            image_url=None,
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        ),
    ]

    def mock_download(cand):
        if cand.image_url == "http://example.com/match.jpg":
            return img_path, None
        elif cand.image_url == "http://example.com/fail_download.jpg":
            return None, "SKIPPED: image download returned HTTP 404"
        return None, "SKIPPED: no image_url provided"

    report = matcher.rank_candidates(primary_face, candidates, mock_download, validate_urls=False)
    
    assert report.metadata_only_count == 1
    assert len(report.skipped_log) == 2  # Cand 2 and Cand 3 skipped
    assert len(report.ranked_candidates) == 1
    
    top_match = report.ranked_candidates[0]
    assert top_match.rank == 1
    assert top_match.is_match is True
    assert top_match.candidate.post_url == "https://x.com/user1/status/111"


def test_anti_self_match_prevention():
    """
    Verifies that a candidate with missing image or failed download is NEVER
    substituted with the input image.
    """
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)
    img_path = "sample_images/who.jpg"
    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    cand_no_img = SocialMatchCandidate(
        post_url="https://x.com/someone/status/999",
        platform="twitter",
        title="No Image Post",
        author="someone",
        image_url=None,
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
    )

    def failing_download(cand):
        return None, "SKIPPED: no image_url provided"

    report = matcher.rank_candidates(primary_face, [cand_no_img], failing_download, validate_urls=False)
    assert len(report.ranked_candidates) == 0
    assert report.metadata_only_count == 1


def test_selection_strategy_priority_ordering():
    """
    Verifies that candidate ranking enforces genuine social candidate prioritization
    sorted strictly by similarity score descending without platform bias.
    """
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)
    img_path = "sample_images/who.jpg"
    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    cand_reddit = SocialMatchCandidate(
        post_url="https://www.reddit.com/r/subreddit/comments/123/title",
        platform="reddit",
        title="Reddit Post",
        author="user_reddit",
        image_url="http://example.com/reddit.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )
    cand_x = SocialMatchCandidate(
        post_url="https://x.com/user_x/status/456",
        platform="twitter",
        title="X Post",
        author="user_x",
        image_url="http://example.com/x.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )
    cand_insta_low = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/789",
        platform="instagram",
        title="Instagram Reel 1",
        author="user_insta",
        image_url="http://example.com/insta1.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )
    cand_web = SocialMatchCandidate(
        post_url="https://example.com/article",
        platform="web",
        title="Generic Web Article",
        author="web_user",
        image_url="http://example.com/web.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=False,
    )

    def mock_download(cand):
        return img_path, None

    report = matcher.rank_candidates(
        primary_face,
        [cand_reddit, cand_x, cand_insta_low, cand_web],
        mock_download,
        validate_urls=False,
    )

    assert len(report.ranked_candidates) == 4
    # All genuine social candidates rank before non-social web candidate
    assert report.ranked_candidates[0].is_social_media is True
    assert report.ranked_candidates[1].is_social_media is True
    assert report.ranked_candidates[2].is_social_media is True
    assert report.ranked_candidates[3].is_social_media is False
    assert report.ranked_candidates[3].candidate.platform == "web"


def _make_mock_face():
    return DetectedFace(
        bbox=(0, 0, 10, 10),
        landmarks={"right_eye": (1, 1), "left_eye": (2, 1), "nose_tip": (1, 2), "mouth_right": (1, 3), "mouth_left": (2, 3)},
        confidence=0.99,
        embedding=[1.0] * 128,
        embedding_model="opencv_sface_128d",
        vector_hash="mock_vector_hash",
        face_crop_hash="mock_crop_hash",
        face_crop_bytes=b"mock_bytes",
    )


def test_reddit_97_vs_x_96_vs_instagram_93_selects_reddit_highest_similarity():
    """Reddit 97% vs X 96% vs Instagram 93% -> Reddit (97%) must be selected based on highest similarity."""
    from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType
    from src.matcher.face_matcher import CandidateRankingReport, RankedCandidateMatch, MatchEvaluationResult

    mock_face = _make_mock_face()

    cand_insta = SocialMatchCandidate(post_url="https://instagram.com/p/123", platform="instagram", title="Insta", author="u1", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    cand_x = SocialMatchCandidate(post_url="https://x.com/u2/status/456", platform="twitter", title="X", author="u2", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    cand_reddit = SocialMatchCandidate(post_url="https://reddit.com/r/sub/comments/789", platform="reddit", title="Reddit", author="u3", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)

    r_insta = RankedCandidateMatch(
        rank=3, candidate=cand_insta, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.93, 9300, 0.3, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.93, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "instagram", cand_insta.post_url, 200, "ok", {}),
        content_type="Post",
    )
    r_x = RankedCandidateMatch(
        rank=2, candidate=cand_x, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.96, 9600, 0.2, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.96, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "twitter", cand_x.post_url, 200, "ok", {}),
        content_type="Post",
    )
    r_reddit = RankedCandidateMatch(
        rank=1, candidate=cand_reddit, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.97, 9700, 0.1, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.97, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.COMMENT, "reddit", cand_reddit.post_url, 200, "ok", {}),
        content_type="Comment/Thread",
    )

    report = CandidateRankingReport(ranked_candidates=[r_reddit, r_x, r_insta], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is not None
    assert "reddit" in selected.candidate.platform.lower()
    assert selected.similarity_score == 0.97
    assert "highest-similarity" in reason


def test_x_96_vs_reddit_92_selects_x_highest_similarity():
    """X 96% vs Reddit 92% -> X (96%) must be selected based on highest similarity."""
    from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType
    from src.matcher.face_matcher import CandidateRankingReport, RankedCandidateMatch, MatchEvaluationResult

    mock_face = _make_mock_face()

    cand_x = SocialMatchCandidate(post_url="https://x.com/u2/status/456", platform="twitter", title="X", author="u2", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    cand_reddit = SocialMatchCandidate(post_url="https://reddit.com/r/sub/comments/789", platform="reddit", title="Reddit", author="u3", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)

    r_x = RankedCandidateMatch(
        rank=1, candidate=cand_x, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.96, 9600, 0.2, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.96, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "twitter", cand_x.post_url, 200, "ok", {}),
        content_type="Post",
    )
    r_reddit = RankedCandidateMatch(
        rank=2, candidate=cand_reddit, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.92, 9200, 0.3, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.92, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.COMMENT, "reddit", cand_reddit.post_url, 200, "ok", {}),
        content_type="Comment/Thread",
    )

    report = CandidateRankingReport(ranked_candidates=[r_x, r_reddit], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is not None
    assert "twitter" in selected.candidate.platform.lower()
    assert selected.similarity_score == 0.96
    assert "highest-similarity" in reason


def test_reddit_91_vs_instagram_75_selects_reddit():
    """Reddit 91% vs Instagram 75% -> Reddit (91%) must be selected."""
    from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType
    from src.matcher.face_matcher import CandidateRankingReport, RankedCandidateMatch, MatchEvaluationResult

    mock_face = _make_mock_face()

    cand_insta = SocialMatchCandidate(post_url="https://instagram.com/p/123", platform="instagram", title="Insta", author="u1", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    cand_reddit = SocialMatchCandidate(post_url="https://reddit.com/r/sub/comments/789", platform="reddit", title="Reddit", author="u3", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)

    r_insta = RankedCandidateMatch(
        rank=2, candidate=cand_insta, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.75, 7500, 0.5, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.75, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "instagram", cand_insta.post_url, 200, "ok", {}),
        content_type="Post",
    )
    r_reddit = RankedCandidateMatch(
        rank=1, candidate=cand_reddit, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.91, 9100, 0.3, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.91, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.COMMENT, "reddit", cand_reddit.post_url, 200, "ok", {}),
        content_type="Comment/Thread",
    )

    report = CandidateRankingReport(ranked_candidates=[r_reddit, r_insta], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is not None
    assert "reddit" in selected.candidate.platform.lower()
    assert selected.similarity_score == 0.91
    assert "highest-similarity" in reason


def test_only_web_candidate_at_99_results_in_no_match():
    """Only WEB candidate at 99% (e.g. TMDB/TV-Media) -> NO MATCH."""
    from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType
    from src.matcher.face_matcher import CandidateRankingReport, RankedCandidateMatch, MatchEvaluationResult

    mock_face = _make_mock_face()

    cand_web = SocialMatchCandidate(post_url="https://themoviedb.org/person/123", platform="web", title="TMDB Profile", author="", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=False)
    r_web = RankedCandidateMatch(
        rank=1, candidate=cand_web, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.99, 9900, 0.05, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.99, is_social_media=False,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.GENERIC_WEB, "web", cand_web.post_url, 200, "ok", {}),
        content_type="Web Page",
    )

    report = CandidateRankingReport(ranked_candidates=[r_web], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is None
    assert "No eligible social-media candidates found" in reason


def test_candidate_below_70_excluded():
    """Candidate with 68% similarity (< 70%) must be excluded from final selection."""
    from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType
    from src.matcher.face_matcher import CandidateRankingReport, RankedCandidateMatch, MatchEvaluationResult

    mock_face = _make_mock_face()

    cand_insta = SocialMatchCandidate(post_url="https://instagram.com/p/123", platform="instagram", title="Insta", author="u1", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    r_insta = RankedCandidateMatch(
        rank=1, candidate=cand_insta, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(False, 0.68, 6800, 0.7, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=False, similarity_score=0.68, is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "instagram", cand_insta.post_url, 200, "ok", {}),
        content_type="Post",
    )

    report = CandidateRankingReport(ranked_candidates=[r_insta], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is None


def test_generic_profile_urls_excluded():
    """Generic Instagram profile URL (e.g. /username) must be excluded."""
    from src.search_engine.social_validator import SocialUrlValidator, UrlStatus, ContentType
    validator = SocialUrlValidator()
    res = validator.validate_social_url("https://www.instagram.com/rickmspelman", check_live=False)
    assert res.is_valid_content is False
    assert res.content_type == ContentType.GENERIC_PROFILE


def test_inaccessible_urls_excluded():
    """Inaccessible candidate (e.g. 404/INACCESSIBLE status) must be excluded from final selection."""
    from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType
    from src.matcher.face_matcher import CandidateRankingReport, RankedCandidateMatch, MatchEvaluationResult

    mock_face = _make_mock_face()

    cand_insta = SocialMatchCandidate(post_url="https://instagram.com/p/123", platform="instagram", title="Insta", author="u1", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    r_insta = RankedCandidateMatch(
        rank=1, candidate=cand_insta, candidate_image_path="sample.jpg",
        evaluation=MatchEvaluationResult(True, 0.95, 9500, 0.2, "h1", "h2", "h3", "h4", mock_face, mock_face, 1, 0, ""),
        is_match=True, similarity_score=0.95, is_social_media=True,
        url_validation=UrlValidationResult(False, UrlStatus.INACCESSIBLE, ContentType.POST, "instagram", cand_insta.post_url, 404, "404 Not Found", {}),
        content_type="Post",
    )

    report = CandidateRankingReport(ranked_candidates=[r_insta], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is None


def test_image_download_failure_excluded():
    """Candidates whose image download fails must be excluded from selection."""
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.70)
    img_path = "sample_images/who.jpg"
    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    cand = SocialMatchCandidate(
        post_url="https://instagram.com/p/broken_image",
        platform="instagram",
        title="Broken Post",
        author="user",
        image_url="http://example.com/broken.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
    )

    def failing_download(c):
        return None, "HTTP 404 Not Found"

    report = matcher.rank_candidates(primary_face, [cand], failing_download, validate_urls=False)
    selected, _ = report.select_final_social_match(threshold=0.70)
    assert selected is None
    assert len(report.ranked_candidates) == 0



