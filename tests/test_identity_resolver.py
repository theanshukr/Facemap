"""
Comprehensive unit tests for Identity-Assisted Social Discovery,
including multi-source name convergence, noise rejection, query generation,
and failure isolation.
"""

import pytest
from src.search_engine.identity_resolver import IdentityResolver, IdentityCandidate
from src.search_engine.identity_social_discoverer import IdentitySocialDiscoverer
from src.search_engine.reverse_search import SocialMatchCandidate
from src.matcher.face_matcher import FaceMatcher, RankedCandidateMatch, MatchEvaluationResult, CandidateRankingReport
from src.face_engine.detector import DetectedFace
from src.search_engine.social_validator import UrlValidationResult, UrlStatus, ContentType


def _make_mock_face(vector_val: float = 1.0) -> DetectedFace:
    return DetectedFace(
        bbox=(0, 0, 10, 10),
        landmarks={"right_eye": (1, 1), "left_eye": (2, 1), "nose_tip": (1, 2), "mouth_right": (1, 3), "mouth_left": (2, 3)},
        confidence=0.99,
        embedding=[vector_val] * 128,
        embedding_model="opencv_sface_128d",
        vector_hash="mock_vector_hash",
        face_crop_hash="mock_crop_hash",
        face_crop_bytes=b"mock_bytes",
    )


def test_multi_source_name_convergence():
    """
    Validates that when multiple independent domain sources identify the same person name,
    IdentityResolver converges on that candidate with high confidence.
    """
    candidates = [
        SocialMatchCandidate(
            post_url="https://letterboxd.com/actor/john-doe",
            platform="web",
            title="John Doe - Letterboxd",
            author="Letterboxd",
            image_url="http://example.com/1.jpg",
            source_engine="Google Lens",
            snippet="Films starring John Doe",
            raw_metadata={},
        ),
        SocialMatchCandidate(
            post_url="https://incluvie.com/actors/john-doe",
            platform="web",
            title="John Doe | Incluvie",
            author="Incluvie",
            image_url="http://example.com/2.jpg",
            source_engine="Google Lens",
            snippet="Inclusive ratings for John Doe",
            raw_metadata={},
        ),
        SocialMatchCandidate(
            post_url="https://plex.tv/people/john-doe",
            platform="web",
            title="John Doe - Plex",
            author="Plex",
            image_url="http://example.com/3.jpg",
            source_engine="Google Lens",
            snippet="Watch movies with John Doe",
            raw_metadata={},
        ),
        SocialMatchCandidate(
            post_url="https://cinema.be/actor/john-doe",
            platform="web",
            title="John Doe - Cinema.be",
            author="Cinema",
            image_url="http://example.com/4.jpg",
            source_engine="Google Lens",
            snippet="Belgian cinema guide for John Doe",
            raw_metadata={},
        ),
    ]

    identity = IdentityResolver.resolve_identity(candidates, min_sources=2)
    assert identity is not None
    assert identity.name == "John Doe"
    assert identity.source_count == 4
    assert identity.is_confident is True
    assert identity.confidence >= 0.85
    assert "letterboxd.com" in identity.supporting_domains
    assert "incluvie.com" in identity.supporting_domains
    assert "plex.tv" in identity.supporting_domains
    assert "cinema.be" in identity.supporting_domains


def test_conflicting_names_resolution():
    """
    Validates that when there are conflicting names across results,
    the name with dominant multi-domain consensus is selected,
    or ambiguity prevents a false resolution.
    """
    candidates = [
        SocialMatchCandidate(
            post_url="https://site1.com/person/alice-smith",
            platform="web",
            title="Alice Smith - Bio",
            author="site1",
            image_url="http://example.com/1.jpg",
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        ),
        SocialMatchCandidate(
            post_url="https://site2.org/people/alice-smith",
            platform="web",
            title="Alice Smith - Database",
            author="site2",
            image_url="http://example.com/2.jpg",
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        ),
        SocialMatchCandidate(
            post_url="https://site3.net/people/alice-smith",
            platform="web",
            title="Alice Smith | Directory",
            author="site3",
            image_url="http://example.com/3.jpg",
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        ),
        SocialMatchCandidate(
            post_url="https://site4.com/actor/bob-jones",
            platform="web",
            title="Bob Jones - Actor",
            author="site4",
            image_url="http://example.com/4.jpg",
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        ),
    ]

    identity = IdentityResolver.resolve_identity(candidates, min_sources=2)
    assert identity is not None
    assert identity.name == "Alice Smith"
    assert identity.source_count == 3


def test_generic_title_rejection():
    """
    Validates that generic page titles, article phrases, and common non-person words
    are rejected and never parsed as a person identity.
    """
    candidates = [
        SocialMatchCandidate(
            post_url="https://site1.com/trending-news",
            platform="web",
            title="Breaking News Latest Updates",
            author="site1",
            image_url="http://example.com/1.jpg",
            source_engine="Google Lens",
            snippet="Check out the latest trending news",
            raw_metadata={},
        ),
        SocialMatchCandidate(
            post_url="https://site2.com/movies/top-action",
            platform="web",
            title="Top Action Movies Database",
            author="site2",
            image_url="http://example.com/2.jpg",
            source_engine="Google Lens",
            snippet="Online movie database",
            raw_metadata={},
        ),
    ]

    identity = IdentityResolver.resolve_identity(candidates, min_sources=2)
    assert identity is None


def test_url_slug_name_extraction():
    """
    Validates extracting clean human names from URL slug segments.
    """
    assert IdentityResolver.is_valid_person_name("Mark Justice") is True
    assert IdentityResolver.is_valid_person_name("Ashish Chanchlani") is True
    assert IdentityResolver.is_valid_person_name("Breaking News") is False
    assert IdentityResolver.is_valid_person_name("Social Media") is False
    assert IdentityResolver.is_valid_person_name("Top Movies") is False
    assert IdentityResolver.is_valid_person_name("Singleword") is False
    assert IdentityResolver.is_valid_person_name("a lowercase name") is False


def test_social_search_query_generation():
    """
    Validates that IdentitySocialDiscoverer generates diverse search queries targeting
    direct social media content paths for the resolved identity without hardcoding.
    """
    queries = IdentitySocialDiscoverer.generate_search_queries("Jane Doe")
    assert len(queries) >= 4
    joined = " ".join(queries).lower()
    assert '"jane doe"' in joined
    assert "instagram.com" in joined
    assert "x.com" in joined or "twitter.com" in joined
    assert "reddit.com" in joined


def test_name_match_but_face_mismatch_rejection():
    """
    CRITICAL: Validates that a social media post found by the person's name
    is strictly REJECTED if SFace facial comparison does NOT meet the 70% threshold.
    Name match != Biometric verification.
    """
    mock_target_face = _make_mock_face(1.0)
    mock_diff_face = _make_mock_face(0.0)  # Different face

    cand_social = SocialMatchCandidate(
        post_url="https://x.com/janedoe/status/999888",
        platform="twitter",
        title="Jane Doe Tweet",
        author="janedoe",
        image_url="http://example.com/diff_person.jpg",
        source_engine="Identity Discovery",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    r_social = RankedCandidateMatch(
        rank=1,
        candidate=cand_social,
        candidate_image_path="diff.jpg",
        evaluation=MatchEvaluationResult(
            is_match=False,
            similarity_score=0.35,  # 35% < 70%
            similarity_int_scaled=3500,
            euclidean_distance=1.1,
            original_vector_hash="h1",
            matched_vector_hash="h2",
            matched_image_hash="h3",
            matched_image_hash_bytes32="0x" + "00" * 32,
            original_face=mock_target_face,
            candidate_face=mock_diff_face,
            detected_faces_count=1,
            selected_face_index=0,
            details="",
        ),
        is_match=False,
        similarity_score=0.35,
        is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "twitter", cand_social.post_url, 200, "ok", {}),
        content_type="Post",
    )

    report = CandidateRankingReport(ranked_candidates=[r_social], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is None
    assert "No eligible social-media candidates found" in reason


def test_verified_social_post_acceptance_above_70():
    """
    Validates that when a discovered social post passes YuNet + SFace >= 70%
    against the original face, it is accepted and verified.
    """
    mock_face = _make_mock_face(1.0)

    cand_social = SocialMatchCandidate(
        post_url="https://instagram.com/reel/112233",
        platform="instagram",
        title="Mark Reel",
        author="mark",
        image_url="http://example.com/mark_reel.jpg",
        source_engine="Identity Discovery",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    r_social = RankedCandidateMatch(
        rank=1,
        candidate=cand_social,
        candidate_image_path="mark.jpg",
        evaluation=MatchEvaluationResult(
            is_match=True,
            similarity_score=0.94,
            similarity_int_scaled=9400,
            euclidean_distance=0.3,
            original_vector_hash="h1",
            matched_vector_hash="h2",
            matched_image_hash="h3",
            matched_image_hash_bytes32="0x" + "00" * 32,
            original_face=mock_face,
            candidate_face=mock_face,
            detected_faces_count=1,
            selected_face_index=0,
            details="",
        ),
        is_match=True,
        similarity_score=0.94,
        is_social_media=True,
        url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.REEL, "instagram", cand_social.post_url, 200, "ok", {}),
        content_type="Reel",
    )

    report = CandidateRankingReport(ranked_candidates=[r_social], metadata_only_count=0, skipped_log=[])
    selected, reason = report.select_final_social_match(threshold=0.70)
    assert selected is not None
    assert selected.similarity_score == 0.94
    assert selected.candidate.post_url == "https://instagram.com/reel/112233"
    assert "highest-similarity verified genuine social-media match" in reason


def test_duplicate_post_removal():
    """
    Validates that duplicate post URLs from different stages are deduplicated and canonicalized.
    """
    from unittest.mock import MagicMock
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "organic_results": [
            {"link": "https://instagram.com/p/abc?utm_source=different", "title": "Duplicate IG", "thumbnail": "http://img.com/1.jpg"},
            {"link": "https://instagram.com/p/new_post", "title": "New IG Post", "thumbnail": "http://img.com/2.jpg"},
        ]
    }
    mock_session.get.return_value = mock_resp

    discoverer = IdentitySocialDiscoverer(session=mock_session, serpapi_key="mock_key")
    identity = IdentityCandidate(
        name="Test Person",
        confidence=0.9,
        source_count=3,
        supporting_domains=["d1.com", "d2.com", "d3.com"],
        is_confident=True,
    )

    existing = [
        SocialMatchCandidate(
            post_url="https://instagram.com/p/abc?utm_source=ig",
            platform="instagram",
            title="Post",
            author="user",
            image_url=None,
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
            is_social_media=True,
        )
    ]

    # Existing clean url is https://instagram.com/p/abc
    # Test discoverer honors existing URLs and never duplicates
    res = discoverer.discover_social_posts(identity=identity, existing_discoveries=existing)
    for c in res:
        assert c.post_url != "https://instagram.com/p/abc"
    assert any(c.post_url == "https://instagram.com/p/new_post" for c in res)



def test_failure_isolation_empty_identity():
    """
    Validates that when identity resolution fails or has < 2 sources,
    it returns None cleanly and does not raise exceptions.
    """
    single_cand = [
        SocialMatchCandidate(
            post_url="https://randomblog.com/post/1",
            platform="web",
            title="Single Post - Random",
            author="random",
            image_url=None,
            source_engine="Google Lens",
            snippet="",
            raw_metadata={},
        )
    ]
    identity = IdentityResolver.resolve_identity(single_cand, min_sources=2)
    assert identity is None


def test_direct_content_url_extractions():
    """
    Validates that direct content URLs on Instagram, X, Reddit, YouTube, TikTok
    are recognized as direct content, while generic profiles are rejected.
    """
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.instagram.com/p/C123456789/") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.instagram.com/reel/C987654321/") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://x.com/someuser/status/1817467102854615084") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://twitter.com/someuser/status/1017443137482919937") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.reddit.com/r/movies/comments/1abc23/discussion_thread/") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.youtube.com/shorts/abcdef12345") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.tiktok.com/@actor/video/7123456789012345678") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.facebook.com/watch/?v=123456789") is True
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.facebook.com/reel/123456789") is True

    # Rejection of generic profiles and search pages
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.instagram.com/themarkjustice") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://x.com/TheMarkJustice") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://twitter.com/TheMarkJustice") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.reddit.com/user/someuser") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.youtube.com/@TheMarkJustice") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.tiktok.com/@TheMarkJustice") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.facebook.com/theofficialmarkjustice") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.instagram.com/explore/tags/actor") is False
    assert IdentitySocialDiscoverer.is_direct_content_url("https://www.reddit.com/r/movies") is False


def test_dynamic_identity_name_search_queries():
    """
    Validates dynamic query generation from IdentityResolver output without hardcoded names.
    """
    name = "Robert Downey"
    queries = IdentitySocialDiscoverer.generate_search_queries(name)
    assert any(f'"{name}" site:instagram.com' == q for q in queries)
    assert any(f'"{name}" site:x.com' == q for q in queries)
    assert any(f'"{name}" site:reddit.com' == q for q in queries)
    assert any(f'"{name}" site:youtube.com' == q for q in queries)
    assert any(f'"{name}" Instagram post' == q for q in queries)
    assert any(f'"{name}" Reddit photo' == q for q in queries)


def test_multiple_faces_in_candidate_image_selects_best_face():
    """
    Validates that when a candidate image contains multiple faces,
    FaceMatcher compares against all detected faces and uses the highest valid similarity.
    """
    matcher = FaceMatcher(threshold=0.70)
    img_path = "sample_images/mark.jpg"

    # Directional embeddings: target is [1, 0, 0...], face0 is [0, 1, 0...], face1 is [1, 0, 0...], face2 is [0, 0, 1...]
    target_emb = [1.0] + [0.0] * 127
    face0_emb = [0.0, 1.0] + [0.0] * 126
    face1_emb = [1.0] + [0.0] * 127
    face2_emb = [0.0, 0.0, 1.0] + [0.0] * 125

    target_face = _make_mock_face(1.0)
    target_face.embedding = target_emb

    face0 = _make_mock_face(1.0)
    face0.embedding = face0_emb
    face1 = _make_mock_face(1.0)
    face1.embedding = face1_emb
    face2 = _make_mock_face(1.0)
    face2.embedding = face2_emb

    from unittest.mock import patch
    from src.face_engine.detector import ImageAnalysisResult

    mock_analysis = ImageAnalysisResult(
        image_path=img_path,
        image_sha256="hash123",
        image_sha256_bytes32="0x" + "00" * 32,
        dimensions=(100, 100),
        faces=[face0, face1, face2],
    )

    with patch.object(matcher.face_engine, "analyze_image", return_value=mock_analysis):
        res = matcher.evaluate_match(target_face, img_path)

    assert res.detected_faces_count == 3
    assert res.selected_face_index == 1
    assert res.is_match is True
    assert res.similarity_score >= 0.95





def test_merged_candidate_pool_highest_biometric_similarity_wins():
    """
    Validates merging reverse search candidates with identity discovery candidates:
    Initial Reverse Search: Instagram 72.5%, Reddit 83.1%
    Identity Discovery: X 91.7%, Instagram 88.4%, Reddit 79.2%
    Combined pool winner: X 91.7%
    """
    mock_face = _make_mock_face(1.0)

    cand_ig_init = SocialMatchCandidate(post_url="https://instagram.com/p/init1", platform="instagram", title="IG 1", author="u", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    cand_rd_init = SocialMatchCandidate(post_url="https://reddit.com/r/sub/comments/init2", platform="reddit", title="RD 1", author="u", image_url="img", source_engine="Google Lens", snippet="", raw_metadata={}, is_social_media=True)
    cand_x_id = SocialMatchCandidate(post_url="https://x.com/u/status/id1", platform="twitter", title="X 1", author="u", image_url="img", source_engine="Identity Discovery", snippet="", raw_metadata={}, is_social_media=True)
    cand_ig_id = SocialMatchCandidate(post_url="https://instagram.com/reel/id2", platform="instagram", title="IG 2", author="u", image_url="img", source_engine="Identity Discovery", snippet="", raw_metadata={}, is_social_media=True)
    cand_rd_id = SocialMatchCandidate(post_url="https://reddit.com/r/sub/comments/id3", platform="reddit", title="RD 2", author="u", image_url="img", source_engine="Identity Discovery", snippet="", raw_metadata={}, is_social_media=True)

    r_ig_init = RankedCandidateMatch(rank=1, candidate=cand_ig_init, candidate_image_path="s.jpg", evaluation=MatchEvaluationResult(True, 0.725, 7250, 0.5, "h", "h", "h", "h", mock_face, mock_face, 1, 0, ""), is_match=True, similarity_score=0.725, is_social_media=True, url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "instagram", cand_ig_init.post_url, 200, "ok", {}), content_type="Post")
    r_rd_init = RankedCandidateMatch(rank=2, candidate=cand_rd_init, candidate_image_path="s.jpg", evaluation=MatchEvaluationResult(True, 0.831, 8310, 0.4, "h", "h", "h", "h", mock_face, mock_face, 1, 0, ""), is_match=True, similarity_score=0.831, is_social_media=True, url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.COMMENT, "reddit", cand_rd_init.post_url, 200, "ok", {}), content_type="Comment/Thread")
    
    r_x_id = RankedCandidateMatch(rank=3, candidate=cand_x_id, candidate_image_path="s.jpg", evaluation=MatchEvaluationResult(True, 0.917, 9170, 0.2, "h", "h", "h", "h", mock_face, mock_face, 1, 0, ""), is_match=True, similarity_score=0.917, is_social_media=True, url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.POST, "twitter", cand_x_id.post_url, 200, "ok", {}), content_type="Post")
    r_ig_id = RankedCandidateMatch(rank=4, candidate=cand_ig_id, candidate_image_path="s.jpg", evaluation=MatchEvaluationResult(True, 0.884, 8840, 0.3, "h", "h", "h", "h", mock_face, mock_face, 1, 0, ""), is_match=True, similarity_score=0.884, is_social_media=True, url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.REEL, "instagram", cand_ig_id.post_url, 200, "ok", {}), content_type="Reel")
    r_rd_id = RankedCandidateMatch(rank=5, candidate=cand_rd_id, candidate_image_path="s.jpg", evaluation=MatchEvaluationResult(True, 0.792, 7920, 0.45, "h", "h", "h", "h", mock_face, mock_face, 1, 0, ""), is_match=True, similarity_score=0.792, is_social_media=True, url_validation=UrlValidationResult(True, UrlStatus.ACCESSIBLE, ContentType.COMMENT, "reddit", cand_rd_id.post_url, 200, "ok", {}), content_type="Comment/Thread")

    # Combine into common pool
    all_ranked = [r_ig_init, r_rd_init, r_x_id, r_ig_id, r_rd_id]
    report = CandidateRankingReport(ranked_candidates=all_ranked, metadata_only_count=0, skipped_log=[])
    
    winner, reason = report.select_final_social_match(threshold=0.70)
    assert winner is not None
    assert winner.candidate.post_url == cand_x_id.post_url
    assert winner.similarity_score == 0.917

    classified = report.get_classified_results(primary_winner=winner, threshold=0.70)
    assert classified["primary_winner"].candidate.post_url == cand_x_id.post_url
    assert len(classified["related_matches"]) == 4
    # Highest similarity in related matches should be IG 88.4%
    assert classified["related_matches"][0].similarity_score == 0.884
    assert classified["related_matches"][1].similarity_score == 0.831
    assert classified["related_matches"][2].similarity_score == 0.792
    assert classified["related_matches"][3].similarity_score == 0.725

