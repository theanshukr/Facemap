"""
Dynamic Name and Attribution Extractor from Discovered Search Metadata.
Extracts person names from titles, snippets, author handles, and URL slugs
using dynamic linguistic patterns, capitalization analysis, and noise stripping.
No hardcoded person dictionaries or pre-selected identities.
"""

import re
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse


# Common descriptive prefixes in social/news titles to strip
TITLE_PREFIX_NOISE = [
    r"^(?:youtuber|actor|actress|influencer|singer|politician|ceo|founder|director|dr\.|mr\.|mrs\.|ms\.|prof\.)\s+",
    r"^(?:watch|video|photo|exclusive|news|breaking|update|official|profile|biography|about|who is)\s*[:\-–—]\s*",
    r"^(?:photos?\s+of|videos?\s+of|posts?\s+by|tweets?\s+by|reel\s+by)\s+",
]

# Common suffixes in social titles to strip
TITLE_SUFFIX_NOISE = [
    r"\s*[\|\-–—]\s*(?:facebook|instagram|youtube|twitter|x|linkedin|reddit|threads|bluesky|tiktok|imdb|wiki|wikipedia|news|tv|zoomtv|ndtv|bbc).*$",
    r"\s+(?:announced|said|spotted|seen|celebrates|attends|speaks|shares|reveals|reacts|posts|unveils|arrives|dies|wins|married).*$",
    r"\s*\(.*?\)\s*$",
    r"\s*\[.*?\]\s*$",
    r"\s*#\w+.*$",
]


class DynamicNameExtractor:
    """
    Extracts person names dynamically from discovered metadata, page titles, and URL slugs.
    """

    @classmethod
    def clean_handle_or_slug(cls, text: str) -> Optional[str]:
        """
        Converts camelCase, PascalCase, or snake_case handles/slugs into space-separated capitalized words.
        E.g., 'TheMarkJustice' -> 'Mark Justice', 'ashish_chanchlani' -> 'Ashish Chanchlani'
        """
        if not text:
            return None

        # Remove prefix words like 'the', 'official', 'real', 'iam'
        clean = re.sub(r"^(?:the|official|real|iam|its|mr|dr)_?", "", text, flags=re.IGNORECASE)
        if not clean or len(clean) < 3:
            clean = text

        # Split snake_case or kebab-case or dots
        if "_" in clean or "-" in clean or "." in clean:
            words = [w for w in re.split(r"[-_.]+", clean) if w and not w.isdigit()]
            # Filter out common media/action junk words
            junk_words = {
                "video", "videos", "post", "posts", "watch", "photo", "photos",
                "reel", "reels", "status", "interview", "exclusive", "news",
                "official", "live", "story", "stories", "vlog", "channel"
            }
            words = [w for w in words if w.lower() not in junk_words]
            if 1 <= len(words) <= 4:
                # Check if words look like a name (only alpha chars, length >= 2)
                if all(w.isalpha() and len(w) >= 2 for w in words):
                    return " ".join(w.capitalize() for w in words)

        # Split PascalCase / camelCase (e.g. MarkJustice, AshishChanchlani)
        split_pascal = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", clean)
        if 2 <= len(split_pascal) <= 4:
            if all(w.isalpha() and len(w) >= 2 for w in split_pascal):
                return " ".join(w.capitalize() for w in split_pascal)

        return None

    @classmethod
    def extract_from_title(cls, title: str) -> Optional[str]:
        """
        Extracts candidate person name from search result title string.
        """
        if not title or not isinstance(title, str):
            return None

        text = title.strip()

        # 1. Strip suffix noise (site names, actions, tags)
        for s_pat in TITLE_SUFFIX_NOISE:
            text = re.sub(s_pat, "", text, flags=re.IGNORECASE).strip()

        # 2. Strip prefix noise (YouTuber, Actor, Watch:, etc.)
        for p_pat in TITLE_PREFIX_NOISE:
            text = re.sub(p_pat, "", text, flags=re.IGNORECASE).strip()

        if not text:
            return None

        # 3. Look for Capitalized 2-3 Word Pattern at the beginning or after a colon/dash
        # E.g., 'Ashish Chanchlani' or 'Mark Justice'
        name_match = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b", text)
        if name_match:
            candidate_name = name_match.group(1).strip()
            words = candidate_name.split()
            junk_words = {
                "social", "media", "breaking", "news", "video", "videos", "post", "posts", "official",
                "high", "school", "basketball", "football", "soccer", "cricket", "sports", "highlights",
                "season", "tournament", "championship", "series", "episode", "trailer", "game",
                "music", "concert", "awards", "festival", "conference", "interview", "press", "live",
                "online", "channel", "group", "community", "world", "daily", "weekly"
            }
            if not any(w.lower() in junk_words for w in words) and all(len(w) >= 2 for w in words):
                return candidate_name

        # 4. Handle concatenated slug-like titles, e.g. "youtuber-ashishchanchlani-announced..."
        if "-" in text or "_" in text:
            extracted = cls.clean_handle_or_slug(text)
            if extracted:
                return extracted

        return None

    @classmethod
    def extract_from_url(cls, url: str) -> Optional[str]:
        """
        Extracts person name from URL path or slug.
        E.g., https://www.facebook.com/zoomtv/videos/mark-justice-exclusive/637623788643670 -> Mark Justice
        E.g., https://www.facebook.com/TheMarkJustice -> Mark Justice
        """
        if not url:
            return None

        parsed = urlparse(url)
        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]

        # Scan parts in reverse (slugs first, then channel names)
        for part in reversed(parts):
            # Direct username or slug segment
            if part.lower() not in {"videos", "video", "posts", "post", "reel", "reels", "watch", "photos", "p", "status", "groups", "channel"}:
                cleaned = cls.clean_handle_or_slug(part)
                if cleaned:
                    return cleaned

        return None

    @classmethod
    def extract_name(
        cls,
        title: Optional[str] = None,
        author: Optional[str] = None,
        snippet: Optional[str] = None,
        url: Optional[str] = None,
        raw_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Main entrypoint: dynamically extracts person name from discovered search metadata.
        Returns cleaned person name or None if no reliable name could be determined.
        """
        # 1. Try Title first (often contains the full formatted name)
        if title:
            name = cls.extract_from_title(title)
            if name:
                return name

        # 2. Try Raw Metadata (e.g. source, title, headline)
        if raw_metadata:
            meta_title = raw_metadata.get("title") or raw_metadata.get("name") or raw_metadata.get("snippet")
            if meta_title and meta_title != title:
                name = cls.extract_from_title(str(meta_title))
                if name:
                    return name

        # 3. Try Snippet
        if snippet and snippet != title:
            name = cls.extract_from_title(snippet)
            if name:
                return name

        # 4. Try URL slug / path
        if url:
            name = cls.extract_from_url(url)
            if name:
                return name

        # 5. Try Author Handle
        if author:
            name = cls.clean_handle_or_slug(author)
            if name:
                return name

        return None
