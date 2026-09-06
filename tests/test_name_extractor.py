"""
Unit tests for Dynamic Name and Attribution Extractor.
Verifies dynamic extraction of person names from titles, snippets, author handles, and URL slugs
without hardcoding or predefined lookup tables.
"""

from src.search_engine.name_extractor import DynamicNameExtractor
from src.search_engine.reverse_search import SocialMatchCandidate


def test_extract_name_from_news_title():
    title = "YouTuber Ashish Chanchlani announced on social media that he is unwell and has cancelled"
    name = DynamicNameExtractor.extract_from_title(title)
    assert name == "Ashish Chanchlani"


def test_extract_name_with_prefix_and_suffix():
    title = "Actor Mark Justice - Exclusive Interview | ZoomTV Official"
    name = DynamicNameExtractor.extract_from_title(title)
    assert name == "Mark Justice"


def test_extract_name_from_pascal_handle():
    handle = "TheMarkJustice"
    name = DynamicNameExtractor.clean_handle_or_slug(handle)
    assert name == "Mark Justice"


def test_extract_name_from_snake_handle():
    handle = "ashish_chanchlani"
    name = DynamicNameExtractor.clean_handle_or_slug(handle)
    assert name == "Ashish Chanchlani"


def test_extract_name_from_url_slug():
    url = "https://www.facebook.com/zoomtv/videos/mark-justice-interview/637623788643670"
    name = DynamicNameExtractor.extract_from_url(url)
    assert name == "Mark Justice"


def test_extract_name_fallback_on_generic_phrase():
    title = "High School Basketball Highlights 2024"
    name = DynamicNameExtractor.extract_name(title=title)
    # Generic phrase should not falsely extract a person name
    assert name is None


def test_candidate_auto_extracts_name_attribution():
    cand = SocialMatchCandidate(
        post_url="https://www.facebook.com/zoomtv/videos/youtuber-ashishchanchlani-announced-on-social-media-that-he-is-unwell/123",
        platform="facebook",
        title="YouTuber Ashish Chanchlani announced on social media that he is unwell",
        author="zoomtv",
        image_url="http://example.com/img.jpg",
        source_engine="Google Lens",
        snippet="",
        raw_metadata={},
        is_social_media=True,
    )
    assert cand.name_attribution == "Ashish Chanchlani"
