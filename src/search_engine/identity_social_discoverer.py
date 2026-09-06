"""
Identity-Assisted Social Media Discovery Engine.
Executes targeted social media discovery using resolved person identity candidates.
Discovers direct posts, reels, threads, and media binaries across supported social platforms
for subsequent independent biometric facial verification.
Never hardcodes identities or URLs; isolates failures completely.
"""

import re
import requests
from typing import List, Optional, Dict, Any, Set, Callable
from urllib.parse import urlparse

from ..config import (
    SERPAPI_API_KEY,
    BING_SEARCH_API_KEY,
    SCRAPE_DO_TOKEN,
    SUPPORTED_SOCIAL_DOMAINS,
)
from .social_extractor import SocialMediaExtractor
from .reverse_search import SocialMatchCandidate
from .identity_resolver import IdentityCandidate
from .social_validator import SocialUrlValidator, ContentType, UrlStatus


class IdentitySocialDiscoverer:
    """
    Performs secondary social-media discovery using a verified multi-source identity candidate.
    Searches for direct social posts and reels with media thumbnails for biometric evaluation.
    """

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        bing_key: Optional[str] = None,
        scrape_do_token: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.serpapi_key = serpapi_key or SERPAPI_API_KEY
        self.bing_key = bing_key or BING_SEARCH_API_KEY
        self.scrape_do_token = scrape_do_token or SCRAPE_DO_TOKEN
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        })

    @classmethod
    def generate_search_queries(cls, identity_name: str) -> List[str]:
        """
        Generates targeted social search query strings for the identity candidate.
        Includes platform-specific site queries and query variations.
        """
        if not identity_name:
            return []

        clean_name = identity_name.strip()
        queries = [
            f'"{clean_name}" site:instagram.com',
            f'"{clean_name}" site:x.com',
            f'"{clean_name}" site:twitter.com',
            f'"{clean_name}" site:reddit.com',
            f'"{clean_name}" site:youtube.com',
            f'"{clean_name}" site:tiktok.com',
            f'"{clean_name}" site:facebook.com',
            f'"{clean_name}" Instagram post',
            f'"{clean_name}" Instagram reel',
            f'"{clean_name}" X status',
            f'"{clean_name}" Twitter post',
            f'"{clean_name}" Reddit photo',
            f'"{clean_name}" YouTube video',
            f'"{clean_name}" TikTok video',
            f'"{clean_name}" Facebook post',
        ]
        return queries

    def is_genuine_social_url(self, url: str) -> bool:
        """Checks if a URL belongs to a supported genuine social media platform."""
        if not url:
            return False
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower().split(":")[0]
            if domain.startswith("www."):
                domain = domain[4:]
            for soc in SUPPORTED_SOCIAL_DOMAINS:
                if domain == soc or domain.endswith("." + soc):
                    return True
            return False
        except Exception:
            return False

    @classmethod
    def is_direct_content_url(cls, url: str) -> bool:
        """
        Validates if a URL points to direct post/reel/video/thread content.
        Rejects generic profiles, homepages, search pages, and category pages.
        """
        if not url:
            return False
        c_type, _ = SocialUrlValidator.detect_content_type(url)
        return c_type in (
            ContentType.POST,
            ContentType.REEL,
            ContentType.VIDEO,
            ContentType.SHORTS,
            ContentType.PHOTO,
            ContentType.COMMENT,
        )

    def _extract_og_image(self, url: str) -> Optional[str]:
        """Attempts to fetch OpenGraph or Twitter card image for direct post URLs lacking thumbnails."""
        try:
            resp = self.session.get(url, timeout=5, allow_redirects=True)
            if resp.status_code == 200:
                html = resp.text
                og_match = re.search(r'<meta\s+property=["\']og:image["\']\s+content=["\']([^"\']+)["\']', html, re.I)
                if not og_match:
                    og_match = re.search(r'<meta\s+content=["\']([^"\']+)["\']\s+property=["\']og:image["\']', html, re.I)
                if og_match:
                    return og_match.group(1).replace("&amp;", "&")
                tw_match = re.search(r'<meta\s+name=["\']twitter:image["\']\s+content=["\']([^"\']+)["\']', html, re.I)
                if tw_match:
                    return tw_match.group(1).replace("&amp;", "&")
        except Exception:
            pass
        return None

    def discover_social_posts(
        self,
        identity: IdentityCandidate,
        existing_discoveries: Optional[List[Any]] = None,
        max_results: int = 30,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> List[SocialMatchCandidate]:
        """
        Searches supported social platforms for direct content from the resolved identity.
        
        Failure isolation: Never raises uncaught exceptions. If search fails or returns nothing,
        an empty list is returned without breaking the primary pipeline.
        
        Ensures:
        - Specific user-facing social content URLs are identified.
        - Usable candidate image/thumbnail URLs are extracted for biometric cross-verification.
        - Existing discoveries are deduplicated out.
        """
        if not identity or not identity.name or not identity.is_confident:
            return []

        discovered: List[SocialMatchCandidate] = []
        seen_urls: Set[str] = set()

        if existing_discoveries:
            for d in existing_discoveries:
                u = getattr(d, "post_url", None)
                if not u and hasattr(d, "candidate"):
                    u = getattr(d.candidate, "post_url", "")
                if u:
                    seen_urls.add(SocialMediaExtractor.canonicalize_url(u))

        clean_name = identity.name.strip()

        # Query batching: Group targeted direct searches
        targeted_search_groups = [
            (
                f'"{clean_name}" (site:instagram.com/p/ OR site:instagram.com/reel/ OR site:instagram.com/reels/)',
                f'"{clean_name}" site:instagram.com',
            ),
            (
                f'"{clean_name}" (site:x.com/*/status/ OR site:twitter.com/*/status/)',
                f'"{clean_name}" site:x.com',
            ),
            (
                f'"{clean_name}" site:reddit.com/r/*/comments/',
                f'"{clean_name}" site:reddit.com',
            ),
            (
                f'"{clean_name}" (site:youtube.com/watch OR site:youtube.com/shorts/)',
                f'"{clean_name}" site:youtube.com',
            ),
            (
                f'"{clean_name}" (site:tiktok.com/@*/video/ OR site:tiktok.com/v/)',
                f'"{clean_name}" site:tiktok.com',
            ),
            (
                f'"{clean_name}" (site:facebook.com/*/posts/ OR site:facebook.com/*/videos/ OR site:facebook.com/reel/ OR site:facebook.com/watch)',
                f'"{clean_name}" site:facebook.com',
            ),
        ]

        # 1. Query SerpApi Google Organic Search for direct social posts
        if self.serpapi_key:
            for query_string, display_query in targeted_search_groups:
                if len(discovered) >= max_results:
                    break
                if log_callback:
                    log_callback(f"Query:\n  {display_query}")
                try:
                    serp_params = {
                        "engine": "google",
                        "q": query_string,
                        "api_key": self.serpapi_key,
                        "num": 10,
                    }
                    resp = self.session.get("https://serpapi.com/search.json", params=serp_params, timeout=12)
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("organic_results", []):
                            link = item.get("link", "")
                            clean_link = SocialMediaExtractor.canonicalize_url(link)
                            if not clean_link or clean_link in seen_urls:
                                continue

                            if not self.is_genuine_social_url(clean_link):
                                continue

                            # Enforce direct content preference (reject generic profiles)
                            if not self.is_direct_content_url(clean_link):
                                # Check if sitelinks contain direct content
                                sitelinks = item.get("sitelinks", {}).get("inline", []) + item.get("sitelinks", {}).get("expanded", [])
                                for sl in sitelinks:
                                    sl_link = SocialMediaExtractor.canonicalize_url(sl.get("link", ""))
                                    if sl_link and sl_link not in seen_urls and self.is_direct_content_url(sl_link):
                                        seen_urls.add(sl_link)
                                        plat_info = SocialMediaExtractor.identify_platform(sl_link)
                                        plat_label = plat_info["platform"] if plat_info else "social"
                                        thumb = sl.get("thumbnail") or item.get("thumbnail")
                                        cand = SocialMatchCandidate(
                                            post_url=sl_link,
                                            platform=plat_label,
                                            title=sl.get("title", f"{clean_name} on {plat_label}"),
                                            author=plat_info.get("handle", "") if plat_info else clean_name,
                                            image_url=thumb if (thumb and str(thumb).startswith("http")) else None,
                                            source_engine=f"Identity Discovery ({clean_name})",
                                            snippet=item.get("snippet", ""),
                                            raw_metadata=sl,
                                            is_social_media=True,
                                            is_post=True,
                                            name_attribution=clean_name,
                                        )
                                        discovered.append(cand)
                                        if log_callback:
                                            log_callback(f"  ✓ Direct content found: {sl_link}")
                                continue

                            seen_urls.add(clean_link)

                            thumbnail = (
                                item.get("thumbnail")
                                or item.get("rich_snippet", {}).get("top", {}).get("extensions", [None])[0]
                            )
                            cand_img = thumbnail if isinstance(thumbnail, str) and thumbnail.startswith("http") else None

                            plat_info = SocialMediaExtractor.identify_platform(clean_link)
                            plat_label = plat_info["platform"] if plat_info else "social"

                            discovered.append(
                                SocialMatchCandidate(
                                    post_url=clean_link,
                                    platform=plat_label,
                                    title=item.get("title", f"{clean_name} on {plat_label}"),
                                    author=plat_info.get("handle", "") if plat_info else clean_name,
                                    image_url=cand_img,
                                    source_engine=f"Identity Discovery ({clean_name})",
                                    snippet=item.get("snippet", ""),
                                    raw_metadata=item,
                                    is_social_media=True,
                                    is_post=True,
                                    name_attribution=clean_name,
                                )
                            )
                            if log_callback:
                                log_callback(f"  ✓ Direct content found: {clean_link}")
                except Exception:
                    pass

        # 2. Query SerpApi Google Images for social-hosted images of that person
        if self.serpapi_key and len(discovered) < max_results:
            image_search_queries = [
                (f'"{clean_name}" site:instagram.com', "Instagram Visuals"),
                (f'"{clean_name}" site:x.com OR site:twitter.com', "X / Twitter Visuals"),
                (f'"{clean_name}" site:reddit.com', "Reddit Visuals"),
                (f'"{clean_name}" site:youtube.com OR site:facebook.com', "YouTube & Facebook Visuals"),
            ]
            for img_query, label in image_search_queries:
                if len(discovered) >= max_results:
                    break
                try:
                    img_params = {
                        "engine": "google_images",
                        "q": img_query,
                        "api_key": self.serpapi_key,
                        "num": 10,
                    }
                    resp = self.session.get("https://serpapi.com/search.json", params=img_params, timeout=12)
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("images_results", []):
                            link = item.get("link") or item.get("context_aspect_ratio_source_url") or item.get("original")
                            clean_link = SocialMediaExtractor.canonicalize_url(link or "")
                            if not clean_link or clean_link in seen_urls:
                                continue

                            if not self.is_genuine_social_url(clean_link):
                                continue

                            # Must be direct content URL
                            if not self.is_direct_content_url(clean_link):
                                continue

                            cand_img = item.get("thumbnail") or item.get("original")
                            if not cand_img or not str(cand_img).startswith("http"):
                                continue

                            seen_urls.add(clean_link)
                            plat_info = SocialMediaExtractor.identify_platform(clean_link)
                            plat_label = plat_info["platform"] if plat_info else "social"

                            discovered.append(
                                SocialMatchCandidate(
                                    post_url=clean_link,
                                    platform=plat_label,
                                    title=item.get("title", f"{clean_name} image"),
                                    author=plat_info.get("handle", "") if plat_info else clean_name,
                                    image_url=cand_img,
                                    source_engine=f"Identity Visual Discovery ({clean_name})",
                                    snippet=item.get("snippet", ""),
                                    raw_metadata=item,
                                    is_social_media=True,
                                    is_post=True,
                                    name_attribution=clean_name,
                                )
                            )
                            if log_callback:
                                log_callback(f"  ✓ Direct content found (visual): {clean_link}")
                except Exception:
                    pass

        # 3. For candidates missing candidate image URL, attempt fast og:image extraction
        for cand in discovered:
            if not cand.has_image:
                og_img = self._extract_og_image(cand.post_url)
                if og_img:
                    cand.image_url = og_img

        return discovered[:max_results]

