"""
Reverse Image Search Engine.
Discovers real matching social media posts using Google Lens (SerpApi), Bing Visual Search,
and verified visual web search APIs.
Extracts direct candidate image URLs and downloads verified image binaries for biometric cross-matching.
Distinguishes genuine social media platforms from generic web directory aggregators.
"""

import os
import shutil
import tempfile
import cv2
import numpy as np
import requests
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse

from ..config import (
    BASE_DIR,
    SERPAPI_API_KEY,
    BING_SEARCH_API_KEY,
    SCRAPE_DO_TOKEN,
    SUPPORTED_SOCIAL_DOMAINS,
)
from .social_extractor import SocialMediaExtractor
from .name_extractor import DynamicNameExtractor


@dataclass
class SocialMatchCandidate:
    """Represents a candidate discovery found via reverse search."""
    post_url: str
    platform: str
    title: str
    author: str
    image_url: Optional[str]
    source_engine: str
    snippet: str
    raw_metadata: Dict[str, Any]
    is_social_media: bool = False
    is_post: bool = True
    content_type: str = ""
    url_validation: Any = None
    name_attribution: Optional[str] = None

    def __post_init__(self):
        if self.name_attribution is None:
            self.name_attribution = DynamicNameExtractor.extract_name(
                title=self.title,
                author=self.author,
                snippet=self.snippet,
                url=self.post_url,
                raw_metadata=self.raw_metadata,
            )

    @property
    def has_image(self) -> bool:
        """Returns True if the search result exposed a usable candidate image URL."""
        return bool(self.image_url and self.image_url.strip().startswith("http"))


class ReverseImageSearchEngine:
    """
    Executes genuine reverse image search across multiple search providers.
    Filters for live social media profiles/posts and extracts candidate media.
    Supports SerpApi Google Lens, Bing Visual Search, and Scrape.do API.
    """

    def __init__(
        self,
        serpapi_key: Optional[str] = None,
        bing_key: Optional[str] = None,
        scrape_do_token: Optional[str] = None,
    ):
        self.serpapi_key = serpapi_key or SERPAPI_API_KEY
        self.bing_key = bing_key or BING_SEARCH_API_KEY
        self.scrape_do_token = scrape_do_token or SCRAPE_DO_TOKEN
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        })

    def has_live_provider(self) -> bool:
        """Returns True if at least one live reverse-search provider key is configured."""
        return bool(self.serpapi_key or self.bing_key or self.scrape_do_token)

    def is_genuine_social_media(self, url: str) -> bool:
        """Checks if a URL belongs to a genuine primary social media platform."""
        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(":")[0]
        for soc in SUPPORTED_SOCIAL_DOMAINS:
            if domain == soc or domain.endswith("." + soc):
                return True
        return False

    def _is_social_media_domain(self, url: str) -> bool:
        """Alias for is_genuine_social_media."""
        return self.is_genuine_social_media(url)

    def search_by_image(
        self,
        image_path: Optional[str] = None,
        image_url: Optional[str] = None,
        max_results: int = 30,
        require_live: bool = False,
    ) -> List[SocialMatchCandidate]:
        """
        Performs reverse image search using available live APIs.
        Combines & ranks results across engines with deduplication and URL canonicalization.
        """
        if require_live and not self.has_live_provider():
            raise ValueError(
                "No live search provider configured in .env. Please provide SERPAPI_API_KEY, "
                "BING_SEARCH_API_KEY, or SCRAPE_DO_TOKEN for live discovery."
            )

        all_candidates: List[SocialMatchCandidate] = []

        # 1. Query SerpApi Google Lens if key is available (Prioritized)
        if self.serpapi_key:
            serp_results = self._search_serpapi_lens(image_path=image_path, image_url=image_url)
            all_candidates.extend(serp_results)

        # 2. Query Bing Visual Search if key is available
        if self.bing_key:
            bing_results = self._search_bing_visual(image_path=image_path, image_url=image_url)
            all_candidates.extend(bing_results)

        # 3. Query Scrape.do Web Engine if token is available
        if self.scrape_do_token and len(all_candidates) < 3:
            scrape_results = self._search_scrape_do(image_path=image_path, image_url=image_url)
            all_candidates.extend(scrape_results)

        # Canonicalize, label social vs web, and deduplicate
        social_candidates: List[SocialMatchCandidate] = []
        web_candidates: List[SocialMatchCandidate] = []
        seen_urls = set()

        for cand in all_candidates:
            clean_url = SocialMediaExtractor.canonicalize_url(cand.post_url)
            cand.post_url = clean_url
            if clean_url not in seen_urls:
                seen_urls.add(clean_url)
                cand.is_social_media = self.is_genuine_social_media(clean_url)
                if cand.is_social_media:
                    social_candidates.append(cand)
                else:
                    web_candidates.append(cand)

        # Prioritize genuine social media candidates with images, then web candidates with images
        prioritized = [c for c in social_candidates if c.has_image]
        prioritized += [c for c in web_candidates if c.has_image]
        prioritized += [c for c in social_candidates if not c.has_image]
        prioritized += [c for c in web_candidates if not c.has_image]

        return prioritized[:max_results]

    def _upload_temp_image(self, image_path: str) -> Optional[str]:
        """Uploads a local image to a hosting provider to obtain a public visual URL."""
        if not hasattr(self, "_upload_cache"):
            self._upload_cache = {}
        if image_path in self._upload_cache:
            return self._upload_cache[image_path]

        try:
            with open(image_path, "rb") as f:
                resp = self.session.post(
                    "https://freeimage.host/api/1/upload",
                    data={"key": "6d207e02198a847aa98d0a2a901485a5", "action": "upload", "format": "json"},
                    files={"source": f},
                    timeout=15,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "image" in data and "url" in data["image"]:
                        url = data["image"]["url"]
                        self._upload_cache[image_path] = url
                        return url
        except Exception:
            pass

        try:
            with open(image_path, "rb") as f:
                resp = self.session.post("https://uguu.se/upload", files={"files[]": f}, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("success") and data.get("files"):
                        url = data["files"][0]["url"]
                        self._upload_cache[image_path] = url
                        return url
        except Exception:
            pass

        return None

    def _search_serpapi_lens(
        self, image_path: Optional[str] = None, image_url: Optional[str] = None
    ) -> List[SocialMatchCandidate]:
        """
        Pure Visual Reverse-Image Search via Google Lens (SerpApi).
        Extracts both post/page URL and high-quality candidate image thumbnail/original URLs.
        """
        results: List[SocialMatchCandidate] = []
        target_img_url = image_url

        if not target_img_url and image_path:
            target_img_url = self._upload_temp_image(image_path)

        if target_img_url:
            lens_params = {
                "engine": "google_lens",
                "url": target_img_url,
                "api_key": self.serpapi_key,
            }
            try:
                resp = self.session.get("https://serpapi.com/search.json", params=lens_params, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    
                    for item in data.get("visual_matches", []):
                        link = item.get("link", "")
                        if link:
                            cand_img = item.get("thumbnail") or item.get("original") or item.get("image") or item.get("thumbnail_large")
                            plat_info = SocialMediaExtractor.identify_platform(link)
                            is_soc = self.is_genuine_social_media(link)
                            
                            platform_label = plat_info["platform"] if plat_info else f"web ({item.get('source', 'site')})"
                            
                            results.append(
                                SocialMatchCandidate(
                                    post_url=link,
                                    platform=platform_label,
                                    title=item.get("title", ""),
                                    author=plat_info["handle"] if plat_info else item.get("source", ""),
                                    image_url=cand_img if cand_img else None,
                                    source_engine="Reverse Image Search",
                                    snippet=item.get("title", ""),
                                    raw_metadata=item,
                                    is_social_media=is_soc,
                                    is_post=plat_info.get("is_post", True) if plat_info else True,
                                )
                            )

                    for item in data.get("exact_matches", []):
                        link = item.get("link", "")
                        if link:
                            cand_img = item.get("thumbnail") or item.get("original") or item.get("image")
                            plat_info = SocialMediaExtractor.identify_platform(link)
                            is_soc = self.is_genuine_social_media(link)
                            platform_label = plat_info["platform"] if plat_info else f"web ({item.get('source', 'site')})"
                            
                            results.append(
                                SocialMatchCandidate(
                                    post_url=link,
                                    platform=platform_label,
                                    title=item.get("title", ""),
                                    author=plat_info["handle"] if plat_info else item.get("source", ""),
                                    image_url=cand_img if cand_img else None,
                                    source_engine="Reverse Image Search",
                                    snippet=item.get("title", ""),
                                    raw_metadata=item,
                                    is_social_media=is_soc,
                                    is_post=plat_info.get("is_post", True) if plat_info else True,
                                )
                            )
            except Exception as e:
                print(f"[ReverseSearch] SerpApi warning: {e}")

        return results

    def _search_bing_visual(
        self, image_path: Optional[str] = None, image_url: Optional[str] = None
    ) -> List[SocialMatchCandidate]:
        """Queries Bing Visual Search API and extracts content and thumbnail URLs."""
        endpoint = "https://api.bing.microsoft.com/v7.0/images/visualsearch"
        headers = {"Ocp-Apim-Subscription-Key": self.bing_key}
        
        results: List[SocialMatchCandidate] = []
        try:
            if image_path:
                with open(image_path, "rb") as f:
                    files = {"image": f}
                    resp = self.session.post(endpoint, headers=headers, files=files, timeout=30)
            elif image_url:
                body = {"imageInfo": {"url": image_url}}
                resp = self.session.post(endpoint, headers=headers, json=body, timeout=30)
            else:
                return []

            if resp.status_code == 200:
                data = resp.json()
                for tag in data.get("tags", []):
                    for action in tag.get("actions", []):
                        if action.get("actionType") in ["VisualSearch", "PagesIncluding"]:
                            for item in action.get("data", {}).get("value", []):
                                host_page = item.get("hostPageUrl", "")
                                if host_page:
                                    cand_img = item.get("contentUrl") or item.get("thumbnailUrl")
                                    plat_info = SocialMediaExtractor.identify_platform(host_page)
                                    is_soc = self.is_genuine_social_media(host_page)
                                    platform_label = plat_info["platform"] if plat_info else f"web ({item.get('hostPageDisplayUrl', 'site')})"
                                    
                                    results.append(
                                        SocialMatchCandidate(
                                            post_url=host_page,
                                            platform=platform_label,
                                            title=item.get("name", ""),
                                            author=plat_info["handle"] if plat_info else "",
                                            image_url=cand_img if cand_img else None,
                                            source_engine="Bing Visual Search",
                                            snippet=item.get("name", ""),
                                            raw_metadata=item,
                                            is_social_media=is_soc,
                                            is_post=plat_info.get("is_post", True) if plat_info else True,
                                        )
                                    )
        except Exception as e:
            print(f"[ReverseSearch] Bing Visual warning: {e}")

        return results

    def _search_scrape_do(
        self, image_path: Optional[str] = None, image_url: Optional[str] = None
    ) -> List[SocialMatchCandidate]:
        """Queries Bing Visual Search via Scrape.do proxy and parses visual thumbnail links."""
        results: List[SocialMatchCandidate] = []
        target_img_url = image_url
        if not target_img_url and image_path:
            target_img_url = self._upload_temp_image(image_path)

        if not target_img_url:
            return []

        try:
            target_url = f"https://www.bing.com/images/search?view=detailv2&iss=sbi&q=imgurl:{target_img_url}"
            params = {
                "token": self.scrape_do_token,
                "url": target_url,
            }
            resp = self.session.get("https://api.scrape.do", params=params, timeout=25)
            if resp.status_code == 200:
                from bs4 import BeautifulSoup
                import json
                soup = BeautifulSoup(resp.text, "html.parser")
                
                for a in soup.find_all("a", href=True):
                    href = a["href"].strip()
                    if not (href.startswith("http://") or href.startswith("https://")):
                        continue
                    if "bing.com/images" in href or "bing.com/search" in href or "javascript:" in href:
                        continue

                    cand_img = None
                    img_tag = a.find("img")
                    if img_tag and img_tag.get("src") and (img_tag["src"].startswith("http://") or img_tag["src"].startswith("https://")):
                        cand_img = img_tag["src"]
                    elif a.get("m"):
                        try:
                            m_data = json.loads(a["m"])
                            m_url = m_data.get("murl") or m_data.get("turl")
                            if m_url and (m_url.startswith("http://") or m_url.startswith("https://")):
                                cand_img = m_url
                        except Exception:
                            pass

                    plat_info = SocialMediaExtractor.identify_platform(href)
                    is_soc = self.is_genuine_social_media(href)
                    platform_label = plat_info["platform"] if plat_info else "web"

                    results.append(
                        SocialMatchCandidate(
                            post_url=href,
                            platform=platform_label,
                            title=a.get_text().strip() or (plat_info["handle"] if plat_info else "Web Result"),
                            author=plat_info["handle"] if plat_info else "",
                            image_url=cand_img,
                            source_engine="Scrape.do Visual Proxy",
                            snippet=a.get_text().strip(),
                            raw_metadata={},
                            is_social_media=is_soc,
                            is_post=plat_info.get("is_post", True) if plat_info else True,
                        )
                    )
        except Exception as e:
            print(f"[ReverseSearch] Scrape.do warning: {e}")

        return results

    def download_candidate_image(
        self, candidate: SocialMatchCandidate, temp_dir: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Downloads a candidate image to a temporary file for biometric facial cross-matching.
        Validates HTTP response, Content-Type, and decodes the image.
        Returns (file_path, None) on success, or (None, failure_reason) on failure.
        CRITICAL: Never returns the input image as fallback.
        """
        if not candidate.image_url:
            return None, "SKIPPED: no image_url provided by search provider"

        url = candidate.image_url.strip()
        if not (url.startswith("http://") or url.startswith("https://")):
            return None, "SKIPPED: invalid image_url schema"

        target_dir = temp_dir or os.path.join(BASE_DIR, "temp", "candidate_images")
        os.makedirs(target_dir, exist_ok=True)

        try:
            resp = self.session.get(url, timeout=15)
            if resp.status_code != 200:
                return None, f"SKIPPED: image download returned HTTP {resp.status_code}"

            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                return None, "SKIPPED: response Content-Type is text/html (not an image binary)"

            if len(resp.content) < 300:
                return None, "SKIPPED: downloaded content too small (<300 bytes)"

            # Verify image can be decoded
            img_arr = np.frombuffer(resp.content, dtype=np.uint8)
            decoded = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                return None, "SKIPPED: downloaded image could not be decoded"

            suffix = ".jpg"
            if "png" in content_type:
                suffix = ".png"
            elif "webp" in content_type:
                suffix = ".webp"

            temp_file = tempfile.NamedTemporaryFile(dir=target_dir, delete=False, suffix=suffix)
            temp_file.write(resp.content)
            temp_file.close()

            return temp_file.name, None

        except requests.exceptions.Timeout:
            return None, "SKIPPED: image download timed out"
        except Exception as e:
            return None, f"SKIPPED: download error ({str(e)})"
