"""
Social Media URL Validator and Accessibility Classifier.
Validates URL syntax, detects specific content types (Video, Reel, Post, Shorts, Photo, etc.),
rejects generic homepages/profiles, and performs lightweight accessibility probes.
Classifies HTTP responses as ACCESSIBLE, BLOCKED_BY_PLATFORM (403/429/login walls),
INACCESSIBLE (404/410/DNS/timeout), or INVALID_STRUCTURE.
"""

import re
import requests
from enum import Enum
from typing import Optional, Dict, Any, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass

from ..config import SUPPORTED_SOCIAL_DOMAINS
from .social_extractor import SocialMediaExtractor


class UrlStatus(str, Enum):
    ACCESSIBLE = "ACCESSIBLE"
    BLOCKED_BY_PLATFORM = "BLOCKED_BY_PLATFORM"
    INACCESSIBLE = "INACCESSIBLE"
    INVALID_STRUCTURE = "INVALID_STRUCTURE"


class ContentType(str, Enum):
    VIDEO = "Video"
    REEL = "Reel"
    POST = "Post"
    SHORTS = "Shorts"
    PHOTO = "Photo"
    COMMENT = "Comment/Thread"
    GENERIC_PROFILE = "Generic Profile"
    HOMEPAGE = "Homepage"
    GENERIC_WEB = "Web Page"
    UNKNOWN = "Unknown"


@dataclass
class UrlValidationResult:
    """Detailed result of social media URL validation and accessibility probe."""
    is_valid_content: bool
    status: UrlStatus
    content_type: ContentType
    platform: str
    final_url: str
    http_status_code: Optional[int]
    reason: str
    raw_details: Dict[str, Any]


class SocialUrlValidator:
    """
    Validates and classifies URLs to ensure only genuine, accessible social-media content
    qualifies as the selected verified match.
    """

    # Recognized content path patterns for supported platforms
    CONTENT_PATTERNS = {
        "facebook": [
            (re.compile(r"/videos?/(?:[^/]+/)?\d+", re.I), ContentType.VIDEO),
            (re.compile(r"/videos?/", re.I), ContentType.VIDEO),
            (re.compile(r"/watch/?", re.I), ContentType.VIDEO),
            (re.compile(r"/reel/[A-Za-z0-9_-]+", re.I), ContentType.REEL),
            (re.compile(r"/reels/[A-Za-z0-9_-]+", re.I), ContentType.REEL),
            (re.compile(r"/posts?/\d+", re.I), ContentType.POST),
            (re.compile(r"/photos?/", re.I), ContentType.PHOTO),
            (re.compile(r"photo\.php", re.I), ContentType.PHOTO),
            (re.compile(r"story\.php", re.I), ContentType.POST),
            (re.compile(r"/permalink\.php", re.I), ContentType.POST),
            (re.compile(r"/groups/[^/]+/posts/\d+", re.I), ContentType.POST),
            (re.compile(r"/groups/[^/]+/permalink/\d+", re.I), ContentType.POST),
        ],
        "instagram": [
            (re.compile(r"/reel/[A-Za-z0-9_-]+", re.I), ContentType.REEL),
            (re.compile(r"/reels/[A-Za-z0-9_-]+", re.I), ContentType.REEL),
            (re.compile(r"/p/[A-Za-z0-9_-]+", re.I), ContentType.POST),
            (re.compile(r"/tv/[A-Za-z0-9_-]+", re.I), ContentType.VIDEO),
        ],
        "youtube": [
            (re.compile(r"/watch\?v=[A-Za-z0-9_-]+", re.I), ContentType.VIDEO),
            (re.compile(r"/shorts/[A-Za-z0-9_-]+", re.I), ContentType.SHORTS),
            (re.compile(r"/live/[A-Za-z0-9_-]+", re.I), ContentType.VIDEO),
            (re.compile(r"youtu\.be/[A-Za-z0-9_-]+", re.I), ContentType.VIDEO),
        ],
        "twitter": [
            (re.compile(r"/status(?:es)?/\d+", re.I), ContentType.POST),
        ],
        "x": [
            (re.compile(r"/status(?:es)?/\d+", re.I), ContentType.POST),
        ],
        "linkedin": [
            (re.compile(r"/posts/[A-Za-z0-9_-]+", re.I), ContentType.POST),
            (re.compile(r"/feed/update/urn:li:activity:\d+", re.I), ContentType.POST),
            (re.compile(r"/pulse/[A-Za-z0-9_-]+", re.I), ContentType.POST),
        ],
        "reddit": [
            (re.compile(r"/r/[^/]+/comments/[a-z0-9]+", re.I), ContentType.COMMENT),
        ],
        "tiktok": [
            (re.compile(r"/video/\d+", re.I), ContentType.VIDEO),
            (re.compile(r"/v/\d+", re.I), ContentType.VIDEO),
        ],
        "threads": [
            (re.compile(r"/post/[A-Za-z0-9_-]+", re.I), ContentType.POST),
        ],
        "bluesky": [
            (re.compile(r"/profile/[^/]+/post/[A-Za-z0-9_-]+", re.I), ContentType.POST),
        ],
    }

    @classmethod
    def detect_content_type(cls, url: str) -> Tuple[ContentType, str]:
        """
        Determines the platform and specific content type from URL structure.
        Distinguishes content from generic profiles or homepages.
        """
        if not url:
            return ContentType.UNKNOWN, "unknown"

        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.rstrip("/")

        # Identify platform
        plat_info = SocialMediaExtractor.identify_platform(url)
        platform_key = plat_info["platform"].lower() if plat_info else ""

        # Homepage check
        if not path or path in ["", "/"]:
            return ContentType.HOMEPAGE, platform_key or "web"

        # Check for specific content patterns
        for plat, patterns in cls.CONTENT_PATTERNS.items():
            if plat in domain or plat == platform_key:
                for pattern, c_type in patterns:
                    if pattern.search(url):
                        return c_type, plat

        # If it belongs to a social platform but didn't match content pattern, it is a generic profile/page
        if any(soc in domain for soc in SUPPORTED_SOCIAL_DOMAINS) or (plat_info and plat_info["platform"] in cls.CONTENT_PATTERNS):
            return ContentType.GENERIC_PROFILE, platform_key or domain

        # General web page
        return ContentType.GENERIC_WEB, "web"

    @classmethod
    def probe_accessibility(
        cls, url: str, session: Optional[requests.Session] = None, timeout: int = 8
    ) -> Tuple[UrlStatus, Optional[int], str]:
        """
        Performs a lightweight HTTP probe to verify URL accessibility.
        Follows normal redirects.
        Classifies responses into ACCESSIBLE, BLOCKED_BY_PLATFORM, INACCESSIBLE.
        """
        sess = session or requests.Session()
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            # Use GET with stream=True so we don't download large bodies (videos, etc.)
            resp = sess.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
            status_code = resp.status_code

            # 2xx Success
            if 200 <= status_code < 300:
                # Check for soft-404 / error body headers if available
                return UrlStatus.ACCESSIBLE, status_code, "URL successfully reached with HTTP 200 OK"

            # 403 Forbidden / 429 Too Many Requests / 401 Unauthorized
            # Social platforms frequently block automated bot scrapers
            if status_code in (401, 403, 429):
                return UrlStatus.BLOCKED_BY_PLATFORM, status_code, f"Platform blocked automated request with HTTP {status_code} (anti-bot protection)"

            # 404 Not Found / 410 Gone
            if status_code in (404, 410):
                return UrlStatus.INACCESSIBLE, status_code, f"Dead / missing resource (HTTP {status_code} Not Found)"

            # 5xx Server Error
            if status_code >= 500:
                return UrlStatus.INACCESSIBLE, status_code, f"Server error (HTTP {status_code})"

            # Other unexpected codes
            return UrlStatus.INACCESSIBLE, status_code, f"Unexpected response status HTTP {status_code}"

        except requests.exceptions.Timeout:
            return UrlStatus.INACCESSIBLE, None, "Connection timed out during accessibility probe"
        except requests.exceptions.SSLError:
            return UrlStatus.INACCESSIBLE, None, "SSL certificate validation error"
        except requests.exceptions.ConnectionError:
            return UrlStatus.INACCESSIBLE, None, "Failed to resolve host or establish connection"
        except Exception as e:
            return UrlStatus.INACCESSIBLE, None, f"Accessibility probe error: {str(e)}"

    @classmethod
    def validate_social_url(
        cls,
        url: str,
        session: Optional[requests.Session] = None,
        check_live: bool = True,
        allow_web_fallback: bool = False,
    ) -> UrlValidationResult:
        """
        Complete validation pipeline for a candidate URL:
        1. Syntax and scheme verification (HTTPS/HTTP).
        2. Canonicalization.
        3. Content type detection (Video, Reel, Post, Shorts, Photo vs Generic Profile).
        4. HTTP accessibility probe.
        5. Eligibility verdict.
        """
        if not url or not isinstance(url, str):
            return UrlValidationResult(
                is_valid_content=False,
                status=UrlStatus.INVALID_STRUCTURE,
                content_type=ContentType.UNKNOWN,
                platform="unknown",
                final_url="",
                http_status_code=None,
                reason="URL is empty or invalid data type",
                raw_details={},
            )

        clean_url = SocialMediaExtractor.canonicalize_url(url.strip())
        parsed = urlparse(clean_url)

        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return UrlValidationResult(
                is_valid_content=False,
                status=UrlStatus.INVALID_STRUCTURE,
                content_type=ContentType.UNKNOWN,
                platform="unknown",
                final_url=clean_url,
                http_status_code=None,
                reason="Invalid URL scheme or missing hostname",
                raw_details={},
            )

        content_type, platform = cls.detect_content_type(clean_url)

        # Reject generic homepages and generic profiles as final social-media content
        if content_type == ContentType.HOMEPAGE:
            return UrlValidationResult(
                is_valid_content=False,
                status=UrlStatus.INVALID_STRUCTURE,
                content_type=content_type,
                platform=platform,
                final_url=clean_url,
                http_status_code=None,
                reason="Generic homepage URL does not represent a specific social media post",
                raw_details={},
            )

        if content_type == ContentType.GENERIC_PROFILE and not allow_web_fallback:
            return UrlValidationResult(
                is_valid_content=False,
                status=UrlStatus.INVALID_STRUCTURE,
                content_type=content_type,
                platform=platform,
                final_url=clean_url,
                http_status_code=None,
                reason="Generic profile/page URL does not represent a specific social media post/content",
                raw_details={},
            )

        # Perform accessibility probe if requested
        if check_live:
            status, code, reason_msg = cls.probe_accessibility(clean_url, session=session)
        else:
            status = UrlStatus.ACCESSIBLE
            code = 200
            reason_msg = "Static validation passed (live probe disabled)"

        # Determine validity:
        # ACCESSIBLE or BLOCKED_BY_PLATFORM are eligible if content_type is genuine content
        is_genuine_content = content_type in (
            ContentType.VIDEO,
            ContentType.REEL,
            ContentType.POST,
            ContentType.SHORTS,
            ContentType.PHOTO,
            ContentType.COMMENT,
            ContentType.GENERIC_WEB,
        )

        is_valid = (
            (status in (UrlStatus.ACCESSIBLE, UrlStatus.BLOCKED_BY_PLATFORM))
            and is_genuine_content
        )

        return UrlValidationResult(
            is_valid_content=is_valid,
            status=status,
            content_type=content_type,
            platform=platform,
            final_url=clean_url,
            http_status_code=code,
            reason=reason_msg,
            raw_details={"domain": parsed.netloc, "path": parsed.path},
        )
