"""
Social Media Post Extractor, URL Canonicalizer, and Domain Parser.
Analyzes URLs, strips tracking parameters, and extracts platform-specific post metadata.
"""

import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

# Common query tracking parameters to strip
TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "ref",
    "ref_src",
    "fbclid",
    "gclid",
    "s",
    "t",
    "context",
    "feature",
    "si",
}


class SocialMediaExtractor:
    """Extracts platform, username, and post identification from URLs."""

    PLATFORM_PATTERNS = {
        "twitter": [
            r"https?://(?:www\.)?(?:twitter\.com|x\.com)/([A-Za-z0-9_]+)/status/(\d+)",
            r"https?://(?:www\.)?(?:twitter\.com|x\.com)/([A-Za-z0-9_]+)",
        ],
        "linkedin": [
            r"https?://(?:www\.)?linkedin\.com/posts/([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?linkedin\.com/feed/update/urn:li:activity:(\d+)",
            r"https?://(?:www\.)?linkedin\.com/in/([A-Za-z0-9_-]+)",
        ],
        "instagram": [
            r"https?://(?:www\.)?instagram\.com/p/([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?instagram\.com/reel/([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?instagram\.com/([A-Za-z0-9_.]+)",
        ],
        "reddit": [
            r"https?://(?:www\.)?reddit\.com/r/([^/]+)/comments/([a-z0-9]+)",
            r"https?://(?:www\.)?reddit\.com/user/([A-Za-z0-9_-]+)",
        ],
        "github": [
            r"https?://(?:www\.)?github\.com/([A-Za-z0-9_-]+)/([A-Za-z0-9_.-]+)",
            r"https?://(?:www\.)?github\.com/([A-Za-z0-9_-]+)",
        ],
        "facebook": [
            r"https?://(?:www\.)?facebook\.com/(?:watch/?\?v=\d+|watch/?)",
            r"https?://(?:www\.)?facebook\.com/(?:[A-Za-z0-9_.]+/)?videos/(\d+)",
            r"https?://(?:www\.)?facebook\.com/reel/(\d+)",
            r"https?://(?:www\.)?facebook\.com/photo\.php",
            r"https?://(?:www\.)?facebook\.com/([A-Za-z0-9_.]+)/posts/(\d+)",
            r"https?://(?:www\.)?facebook\.com/([A-Za-z0-9_.]+)",
        ],
        "youtube": [
            r"https?://(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?youtube\.com/shorts/([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?youtube\.com/@([A-Za-z0-9_.-]+)",
            r"https?://(?:www\.)?youtube\.com/channel/([A-Za-z0-9_-]+)",
        ],
        "threads": [
            r"https?://(?:www\.)?threads\.net/@([A-Za-z0-9_.]+)/post/([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?threads\.net/@([A-Za-z0-9_.]+)",
        ],
        "bluesky": [
            r"https?://(?:www\.)?bsky\.app/profile/([A-Za-z0-9_.-]+)/post/([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?bsky\.app/profile/([A-Za-z0-9_.-]+)",
        ],
        "medium": [
            r"https?://(?:www\.)?medium\.com/@([A-Za-z0-9_.-]+)/([A-Za-z0-9_-]+)",
            r"https?://(?:www\.)?medium\.com/@([A-Za-z0-9_.-]+)",
        ]
    }

    POST_URL_INDICATORS = [
        "/status/",
        "/statuses/",
        "/posts/",
        "/videos/",
        "/video/",
        "/p/",
        "/reel/",
        "/reels/",
        "/comments/",
        "/watch",
        "/shorts/",
        "/post/",
        "/photo.php",
        "/photo/",
        "/story.php",
    ]

    @classmethod
    def canonicalize_url(cls, url: str) -> str:
        """
        Removes tracking parameters (utm_*, ref, etc.) and normalizes URLs.
        """
        if not url:
            return ""
        parsed = urlparse(url)
        # Filter query params
        query_items = parse_qsl(parsed.query, keep_blank_values=False)
        clean_query_items = [
            (k, v) for k, v in query_items if k.lower() not in TRACKING_PARAMS
        ]
        clean_query = urlencode(clean_query_items)

        # Normalize path: remove multiple slashes, strip trailing slash unless root
        clean_path = re.sub(r"/+", "/", parsed.path)
        if len(clean_path) > 1 and clean_path.endswith("/"):
            clean_path = clean_path.rstrip("/")

        clean_url = urlunparse((
            parsed.scheme.lower() or "https",
            parsed.netloc.lower(),
            clean_path,
            parsed.params,
            clean_query,
            "",  # strip fragment
        ))
        return clean_url

    @classmethod
    def is_post_url(cls, url: str) -> bool:
        """
        Checks if the URL points to a specific post/content rather than a root domain.
        """
        lower_url = url.lower()
        return any(ind in lower_url for ind in cls.POST_URL_INDICATORS)

    @classmethod
    def identify_platform(cls, url: str) -> Optional[Dict[str, str]]:
        """Identifies if a URL belongs to a known social media platform."""
        clean_url = cls.canonicalize_url(url)
        for platform, patterns in cls.PLATFORM_PATTERNS.items():
            for pattern in patterns:
                match = re.match(pattern, clean_url, re.IGNORECASE)
                if match:
                    handle = match.group(1) if match.groups() else ""
                    return {
                        "platform": platform,
                        "handle": handle,
                        "url": clean_url,
                        "is_post": cls.is_post_url(clean_url),
                    }
        
        parsed = urlparse(clean_url)
        domain = parsed.netloc.lower()
        if any(d in domain for d in ["twitter.com", "x.com", "linkedin.com", "instagram.com", "github.com", "facebook.com", "reddit.com", "youtube.com"]):
            parts = [p for p in parsed.path.split("/") if p]
            return {
                "platform": domain.split(".")[-2] if "." in domain else domain,
                "handle": parts[0] if parts else "unknown",
                "url": clean_url,
                "is_post": cls.is_post_url(clean_url),
            }

        return None
