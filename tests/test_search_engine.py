"""
Unit tests for Reverse Image Search, URL Canonicalization, and Candidate Image Handling.
"""

from src.search_engine.social_extractor import SocialMediaExtractor
from src.search_engine.reverse_search import ReverseImageSearchEngine, SocialMatchCandidate


def test_url_canonicalization():
    raw_url = "https://x.com/elonmusk/status/1789201408849?utm_source=twitter&utm_medium=social&ref_src=twsrc%5Etfw#some-hash"
    clean_url = SocialMediaExtractor.canonicalize_url(raw_url)
    assert "utm_source" not in clean_url
    assert "utm_medium" not in clean_url
    assert "ref_src" not in clean_url
    assert clean_url == "https://x.com/elonmusk/status/1789201408849"


def test_is_post_url():
    post_url = "https://x.com/elonmusk/status/1789201408849"
    profile_url = "https://x.com/elonmusk"
    assert SocialMediaExtractor.is_post_url(post_url) is True
    assert SocialMediaExtractor.is_post_url(profile_url) is False


def test_social_media_extractor_twitter():
    url = "https://x.com/elonmusk/status/1789201408849"
    info = SocialMediaExtractor.identify_platform(url)
    assert info is not None
    assert info["platform"] == "twitter"
    assert info["handle"] == "elonmusk"
    assert info["is_post"] is True


def test_social_media_extractor_linkedin():
    url = "https://www.linkedin.com/posts/satyanadella-leadership-update"
    info = SocialMediaExtractor.identify_platform(url)
    assert info is not None
    assert info["platform"] == "linkedin"
    assert info["is_post"] is True


def test_social_domain_filter():
    engine = ReverseImageSearchEngine()
    assert engine._is_social_media_domain("https://twitter.com/elonmusk") is True
    assert engine._is_social_media_domain("https://github.com/torvalds") is True
    assert engine._is_social_media_domain("https://generic-news-blog.org/article") is False


def test_candidate_image_url_separation():
    """
    Verifies that post_url and image_url remain strictly distinct fields on candidate objects.
    """
    cand = SocialMatchCandidate(
        post_url="https://www.facebook.com/TheMarkJustice/",
        platform="facebook",
        title="Mark Justice",
        author="TheMarkJustice",
        image_url="https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcTjmOlyNNvAt6-90nUeuv5FsubH5NP5FmtslW4rc0A_Dbd8NzDq",
        source_engine="Google Lens",
        snippet="Mark Justice Facebook",
        raw_metadata={},
    )
    assert cand.has_image is True
    assert cand.post_url != cand.image_url
    assert "facebook.com" in cand.post_url
    assert "gstatic.com" in cand.image_url
