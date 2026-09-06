"""
Profile & Multi-Post Discovery Crawler.
Discovers additional posts and reels associated with a creator's account/profile
after the primary winner has been selected.
Ensures failure isolation, canonical URL extraction, and deduplication against the winner.
"""

import re
import requests
from typing import List, Optional, Dict, Any, Set
from urllib.parse import urlparse

from ..config import (
    SERPAPI_API_KEY,
    SCRAPE_DO_TOKEN,
    SUPPORTED_SOCIAL_DOMAINS,
)
from .social_extractor import SocialMediaExtractor
from .reverse_search import SocialMatchCandidate
from .social_validator import SocialUrlValidator, ContentType


class ProfileCrawler:
    """
    Discovers additional social media posts and reels from an identified creator/account.
    Works as an isolated second-stage enhancement after primary winner selection.
    """

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        scrape_do_token: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.serpapi_key = serpapi_key or SERPAPI_API_KEY
        self.scrape_do_token = scrape_do_token or SCRAPE_DO_TOKEN
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        })

    @classmethod
    def extract_profile_handle(cls, post_url: str, author: str = "") -> Optional[Dict[str, str]]:
        """
        Extracts platform and creator handle from a post URL or author string.
        Returns dict with 'platform', 'handle', 'profile_url' or None.
        """
        if not post_url:
            return None

        clean_url = SocialMediaExtractor.canonicalize_url(post_url)
        plat_info = SocialMediaExtractor.identify_platform(clean_url)
        if not plat_info:
            return None

        platform = plat_info.get("platform", "").lower()
        handle = plat_info.get("handle", "").strip()

        # In Instagram reel or post URLs (/reel/XYZ or /p/XYZ), the extracted handle is the shortcode/post ID.
        # If author is provided, prefer the author's username/handle.
        is_content_path = (
            "/reel/" in clean_url or "/p/" in clean_url or "/status/" in clean_url or "/comments/" in clean_url or "/watch" in clean_url
        )
        if author:
            clean_author = re.sub(r"[^A-Za-z0-9_.-]", "", author)
            if clean_author and (is_content_path or not handle or handle.lower() in ["p", "reel", "reels", "posts", "status", "unknown"]):
                handle = clean_author
        elif is_content_path:
            # When it is a pure post/reel URL and no author handle is provided
            pass

        if not handle:
            return None


        # Build clean profile URL
        profile_url = ""
        if platform == "instagram":
            profile_url = f"https://www.instagram.com/{handle}"
        elif platform in ["twitter", "x"]:
            profile_url = f"https://x.com/{handle}"
        elif platform == "reddit":
            profile_url = f"https://www.reddit.com/user/{handle}"
        elif platform == "youtube":
            profile_url = f"https://www.youtube.com/@{handle}"
        elif platform == "facebook":
            profile_url = f"https://www.facebook.com/{handle}"
        elif platform == "threads":
            profile_url = f"https://www.threads.net/@{handle}"
        elif platform == "linkedin":
            profile_url = f"https://www.linkedin.com/in/{handle}"

        return {
            "platform": platform,
            "handle": handle,
            "profile_url": profile_url,
        }

    def discover_related_posts(
        self,
        primary_candidate: SocialMatchCandidate,
        existing_discoveries: Optional[List[SocialMatchCandidate]] = None,
        max_results: int = 10,
    ) -> List[SocialMatchCandidate]:
        """
        Discovers additional posts and reels associated with the primary winner's profile/account.
        
        Failure isolation: Never throws uncaught exceptions. If crawling fails or returns nothing,
        an empty list is returned without breaking the primary pipeline.
        
        Ensures:
        - Primary winner URL is deduplicated out.
        - Only genuine social media post/reel URLs are returned.
        - Media image/thumbnail URLs are extracted for independent biometric verification.
        - Post URLs are never CDN / cutout image links.
        """
        related_candidates: List[SocialMatchCandidate] = []
        seen_post_urls: Set[str] = set()

        primary_clean_url = SocialMediaExtractor.canonicalize_url(primary_candidate.post_url)
        seen_post_urls.add(primary_clean_url)

        # 1. Harvest related items from existing reverse-search discoveries from the same author/account or visual cluster
        if existing_discoveries:
            prof_info = self.extract_profile_handle(primary_candidate.post_url, primary_candidate.author)
            primary_handle = (prof_info.get("handle") if prof_info else "").lower()
            primary_plat = (prof_info.get("platform") if prof_info else "").lower()

            for cand in existing_discoveries:
                cand_clean_url = SocialMediaExtractor.canonicalize_url(cand.post_url)
                if not cand_clean_url or cand_clean_url in seen_post_urls:
                    continue

                # Ensure it is a genuine post URL and not an image CDN link
                if not cand.is_social_media and not SocialMediaExtractor.is_post_url(cand_clean_url):
                    continue

                cand_prof = self.extract_profile_handle(cand_clean_url, cand.author)
                cand_handle = (cand_prof.get("handle") if cand_prof else "").lower()
                cand_plat = (cand_prof.get("platform") if cand_prof else "").lower()

                # Check if it matches the same author or same platform handle or related cluster
                is_author_match = primary_handle and (cand_handle == primary_handle or primary_handle in cand.title.lower() or primary_handle in cand.snippet.lower())
                is_same_plat = primary_plat and (cand_plat == primary_plat)

                if is_author_match or (is_same_plat and cand.has_image and cand.is_social_media):
                    seen_post_urls.add(cand_clean_url)
                    cand_copy = SocialMatchCandidate(
                        post_url=cand_clean_url,
                        platform=cand.platform,
                        title=cand.title,
                        author=cand.author,
                        image_url=cand.image_url,
                        source_engine=f"Related Discovery ({cand.source_engine})",
                        snippet=cand.snippet,
                        raw_metadata=cand.raw_metadata,
                        is_social_media=True,
                        is_post=True,
                        content_type=cand.content_type,
                    )
                    related_candidates.append(cand_copy)

        # 2. If SerpApi is available and handle is identified, query for recent posts/reels of that profile
        try:
            prof_info = self.extract_profile_handle(primary_candidate.post_url, primary_candidate.author)
            if prof_info and prof_info.get("handle") and self.serpapi_key and len(related_candidates) < max_results:
                handle = prof_info["handle"]
                platform = prof_info["platform"]
                
                query_str = ""
                if platform == "instagram":
                    query_str = f"site:instagram.com/{handle}/reel OR site:instagram.com/{handle}/p"
                elif platform in ["twitter", "x"]:
                    query_str = f"site:x.com/{handle}/status OR site:twitter.com/{handle}/status"
                elif platform == "reddit":
                    query_str = f"site:reddit.com/user/{handle}"
                elif platform == "youtube":
                    query_str = f"site:youtube.com/@{handle} shorts OR watch"

                if query_str:
                    serp_params = {
                        "engine": "google",
                        "q": query_str,
                        "api_key": self.serpapi_key,
                        "num": 10,
                    }
                    resp = self.session.get("https://serpapi.com/search.json", params=serp_params, timeout=15)
                    if resp.status_code == 200:
                        data = resp.json()
                        for item in data.get("organic_results", []):
                            link = item.get("link", "")
                            clean_link = SocialMediaExtractor.canonicalize_url(link)
                            if not clean_link or clean_link in seen_post_urls:
                                continue
                            if not SocialMediaExtractor.is_post_url(clean_link):
                                continue

                            seen_post_urls.add(clean_link)
                            thumbnail = (
                                item.get("thumbnail")
                                or item.get("rich_snippet", {}).get("top", {}).get("extensions", [None])[0]
                            )
                            cand_img = thumbnail if isinstance(thumbnail, str) and thumbnail.startswith("http") else None

                            plat_info_cand = SocialMediaExtractor.identify_platform(clean_link)
                            plat_label = plat_info_cand["platform"] if plat_info_cand else platform

                            related_candidates.append(
                                SocialMatchCandidate(
                                    post_url=clean_link,
                                    platform=plat_label,
                                    title=item.get("title", f"{handle}'s post"),
                                    author=handle,
                                    image_url=cand_img,
                                    source_engine="Profile Discovery (Google Index)",
                                    snippet=item.get("snippet", ""),
                                    raw_metadata=item,
                                    is_social_media=True,
                                    is_post=True,
                                )
                            )
        except Exception as e:
            # Failure isolation: silently catch search API errors during profile crawl
            pass

        return related_candidates[:max_results]
