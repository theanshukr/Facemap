"""
Unit tests for Social Media URL Validator, Accessibility Probing,
Content Type Classification, and Automatic Candidate Fallback.
"""

import pytest
from unittest.mock import MagicMock, patch
import requests

from src.search_engine.social_validator import (
    SocialUrlValidator,
    UrlStatus,
    ContentType,
    UrlValidationResult,
)
from src.search_engine.reverse_search import SocialMatchCandidate
from src.matcher.face_matcher import FaceMatcher
from src.face_engine.detector import FaceEngine


def test_valid_facebook_video_url_accepted():
    url = "https://www.facebook.com/zoomtv/videos/youtuber-ashishchanchlani-announced-on-social-media-that-he-is-unwell-and-has-ca/637623788643670"
    res = SocialUrlValidator.validate_social_url(url, check_live=False)
    assert res.content_type == ContentType.VIDEO
    assert res.is_valid_content is True
    assert res.status == UrlStatus.ACCESSIBLE


def test_valid_instagram_reel_url_accepted():
    url = "https://www.instagram.com/reel/Cu-Ch94gYzk/"
    res = SocialUrlValidator.validate_social_url(url, check_live=False)
    assert res.content_type == ContentType.REEL
    assert res.is_valid_content is True
    assert res.status == UrlStatus.ACCESSIBLE


def test_generic_facebook_profile_rejected_as_final_post():
    url = "https://www.facebook.com/SomePerson"
    res = SocialUrlValidator.validate_social_url(url, check_live=False)
    assert res.content_type == ContentType.GENERIC_PROFILE
    assert res.is_valid_content is False
    assert res.status == UrlStatus.INVALID_STRUCTURE
    assert "generic profile" in res.reason.lower()


def test_malformed_url_rejected():
    invalid_urls = [
        "not-a-valid-url",
        "ftp://example.com/video.mp4",
        "https://",
        "",
    ]
    for u in invalid_urls:
        res = SocialUrlValidator.validate_social_url(u, check_live=False)
        assert res.is_valid_content is False
        assert res.status == UrlStatus.INVALID_STRUCTURE


def test_404_url_rejected_as_inaccessible():
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_session.get.return_value = mock_resp

    url = "https://www.facebook.com/watch/?v=999999999999"
    res = SocialUrlValidator.validate_social_url(url, session=mock_session, check_live=True)
    assert res.status == UrlStatus.INACCESSIBLE
    assert res.is_valid_content is False
    assert "404" in res.reason


def test_platform_403_classified_as_blocked_by_platform():
    """
    403/429 from social platform anti-bot walls must NOT be marked dead/broken.
    It should be classified as BLOCKED_BY_PLATFORM and remain eligible if content structure is valid.
    """
    mock_session = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_session.get.return_value = mock_resp

    url = "https://www.instagram.com/reel/Cu-Ch94gYzk/"
    res = SocialUrlValidator.validate_social_url(url, session=mock_session, check_live=True)
    assert res.status == UrlStatus.BLOCKED_BY_PLATFORM
    assert res.is_valid_content is True
    assert res.content_type == ContentType.REEL


def test_broken_candidate_1_falls_back_to_valid_candidate_2():
    """
    Verifies that if candidate #1 has higher similarity (e.g. 96.13%) but a broken/404 URL,
    FaceMatcher skips #1 and successfully selects candidate #2 (valid Instagram Reel 94.82%).
    """
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)
    img_path = "sample_images/who.jpg"

    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    cand1_broken_fb = SocialMatchCandidate(
        post_url="https://www.facebook.com/watch/?v=broken123",
        platform="facebook",
        title="Broken Facebook Video",
        author="zoomtv",
        image_url="http://example.com/fb_img.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    cand2_valid_ig = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/Cu-Ch94gYzk/",
        platform="instagram",
        title="Valid Instagram Reel",
        author="ashish",
        image_url="http://example.com/ig_img.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    def mock_download(cand):
        # Both images download successfully
        return img_path, None

    # Patch validate_social_url so cand1 returns 404 INACCESSIBLE, cand2 returns 200 ACCESSIBLE
    def mock_validate(url, **kwargs):
        if "broken123" in url:
            return UrlValidationResult(
                is_valid_content=False,
                status=UrlStatus.INACCESSIBLE,
                content_type=ContentType.VIDEO,
                platform="facebook",
                final_url=url,
                http_status_code=404,
                reason="Dead resource (HTTP 404 Not Found)",
                raw_details={},
            )
        else:
            return UrlValidationResult(
                is_valid_content=True,
                status=UrlStatus.ACCESSIBLE,
                content_type=ContentType.REEL,
                platform="instagram",
                final_url=url,
                http_status_code=200,
                reason="URL reachable",
                raw_details={},
            )

    with patch.object(SocialUrlValidator, "validate_social_url", side_effect=mock_validate):
        report = matcher.rank_candidates(
            original_face=primary_face,
            candidates=[cand1_broken_fb, cand2_valid_ig],
            download_fn=mock_download,
            validate_urls=True,
        )

    # Cand 1 was skipped because of broken URL
    assert len(report.skipped_log) == 1
    assert "INACCESSIBLE" in report.skipped_log[0]["reason"]
    assert report.skipped_log[0]["url"] == cand1_broken_fb.post_url

    # Cand 2 was selected as top match
    assert len(report.ranked_candidates) == 1
    top_match = report.top_social_match
    assert top_match is not None
    assert top_match.candidate.post_url == cand2_valid_ig.post_url
    assert top_match.content_type == "Reel"
    assert top_match.url_validation.status == UrlStatus.ACCESSIBLE


def test_no_valid_social_candidate_returns_empty_ranked_pool():
    """
    If all discovered candidates have broken URLs or invalid structure,
    no candidate is eligible and report.ranked_candidates is empty.
    """
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)
    img_path = "sample_images/who.jpg"
    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    cand_generic = SocialMatchCandidate(
        post_url="https://www.facebook.com/SomePerson",
        platform="facebook",
        title="Generic Profile",
        author="SomePerson",
        image_url="http://example.com/img.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    def mock_download(cand):
        return img_path, None

    report = matcher.rank_candidates(
        original_face=primary_face,
        candidates=[cand_generic],
        download_fn=mock_download,
        validate_urls=True,
    )

    assert len(report.ranked_candidates) == 0
    assert len(report.skipped_log) == 1
    assert report.top_social_match is None


def test_blocked_by_platform_not_selected_over_accessible_candidate():
    """
    Validates user priority requirement:
    #1 Facebook 96.13% BLOCKED_BY_PLATFORM
    #2 Instagram 94.82% ACCESSIBLE
    -> Instagram #2 (ACCESSIBLE) is selected as top verified match over Facebook #1!
    """
    engine = FaceEngine()
    matcher = FaceMatcher(threshold=0.40)
    img_path = "sample_images/who.jpg"
    analysis = engine.analyze_image(img_path)
    primary_face = analysis.primary_face

    cand_fb_blocked = SocialMatchCandidate(
        post_url="https://www.facebook.com/zoomtv/videos/123456",
        platform="facebook",
        title="FB Video 96%",
        author="zoomtv",
        image_url="http://example.com/fb.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    cand_ig_accessible = SocialMatchCandidate(
        post_url="https://www.instagram.com/reel/ABC123xyz",
        platform="instagram",
        title="IG Reel 94%",
        author="ashish",
        image_url="http://example.com/ig.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )

    def mock_download(cand):
        return img_path, None

    def mock_validate(url, **kwargs):
        if "facebook.com" in url:
            return UrlValidationResult(
                is_valid_content=True,
                status=UrlStatus.BLOCKED_BY_PLATFORM,
                content_type=ContentType.VIDEO,
                platform="facebook",
                final_url=url,
                http_status_code=403,
                reason="Platform bot protection 403",
                raw_details={},
            )
        else:
            return UrlValidationResult(
                is_valid_content=True,
                status=UrlStatus.ACCESSIBLE,
                content_type=ContentType.REEL,
                platform="instagram",
                final_url=url,
                http_status_code=200,
                reason="HTTP 200 OK",
                raw_details={},
            )

    with patch.object(SocialUrlValidator, "validate_social_url", side_effect=mock_validate):
        report = matcher.rank_candidates(
            original_face=primary_face,
            candidates=[cand_fb_blocked, cand_ig_accessible],
            download_fn=mock_download,
            validate_urls=True,
        )

    assert len(report.ranked_candidates) == 2
    # Candidate #1 in ranked list must be Instagram because it is ACCESSIBLE (tier 4 > tier 3)
    assert report.top_social_match.candidate.post_url == cand_ig_accessible.post_url
    assert report.top_social_match.url_validation.status == UrlStatus.ACCESSIBLE
    assert report.ranked_candidates[0].candidate.post_url == cand_ig_accessible.post_url
    assert report.ranked_candidates[1].candidate.post_url == cand_fb_blocked.post_url
