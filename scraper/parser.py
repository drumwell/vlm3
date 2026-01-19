"""
vBulletin HTML Parser

Parses vBulletin 6.x forum pages to extract structured data:
- Forum listings (subforums)
- Thread listings
- Post content (text, author, timestamp, images)
"""

import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urljoin, urlparse, parse_qs

from bs4 import BeautifulSoup, Tag


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Forum:
    """Represents a forum or subforum."""

    forum_id: str
    title: str
    url: str
    description: str = ""
    parent_id: Optional[str] = None
    thread_count: int = 0
    post_count: int = 0
    subforums: List["Forum"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["subforums"] = [sf.to_dict() for sf in self.subforums]
        return d


@dataclass
class Thread:
    """Represents a forum thread."""

    thread_id: str
    title: str
    url: str
    forum_id: str
    author: str = ""
    author_id: str = ""
    created_at: str = ""
    reply_count: int = 0
    view_count: int = 0
    last_post_at: str = ""
    last_poster: str = ""
    is_sticky: bool = False
    is_locked: bool = False
    page_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Post:
    """Represents a single post in a thread."""

    post_id: str
    thread_id: str
    author: str
    author_id: str = ""
    content_html: str = ""
    content_text: str = ""
    created_at: str = ""
    post_number: int = 0
    is_first_post: bool = False
    images: List[str] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Pagination:
    """Pagination information."""

    current_page: int = 1
    total_pages: int = 1
    has_next: bool = False
    has_prev: bool = False
    next_url: Optional[str] = None
    prev_url: Optional[str] = None


# ============================================================================
# vBulletin 6.x Parser
# ============================================================================


class VBulletinParser:
    """
    Parser for vBulletin 6.x HTML pages.

    vBulletin 6 uses a modern responsive design with specific CSS classes
    and data attributes that we can leverage for parsing.
    """

    def __init__(self, base_url: str):
        """
        Initialize parser.

        Args:
            base_url: Base URL of the forum for resolving relative links
        """
        self.base_url = base_url.rstrip("/")

    def _resolve_url(self, url: str) -> str:
        """Resolve relative URL to absolute."""
        if not url:
            return ""
        if url.startswith("//"):
            return "https:" + url
        if not url.startswith("http"):
            return urljoin(self.base_url + "/", url)
        return url

    def _extract_id(self, url: str, prefix: str = "", allow_slug: bool = True) -> str:
        """
        Extract ID from URL path.

        Handles both numeric IDs (/forum/123-name) and slug-based URLs (/forum/s14/faqs/no-start).

        Args:
            url: URL to extract ID from
            prefix: Optional prefix to filter by
            allow_slug: If True, return last path segment as ID when no numeric ID found
        """
        if not url:
            return ""
        # Handle URLs like /threads/12345-thread-title or /forum/67-name
        path = urlparse(url).path
        parts = [p for p in path.strip("/").split("/") if p]

        # First, try to find a numeric ID
        for part in parts:
            if "-" in part:
                potential_id = part.split("-")[0]
                if potential_id.isdigit():
                    return potential_id
            elif part.isdigit():
                return part

        # If no numeric ID and slugs are allowed, use the last path segment
        # (excluding common prefixes like 'forum', 'forums', 'threads')
        if allow_slug and parts:
            # Filter out common path prefixes
            slug_parts = [p for p in parts if p not in ('forum', 'forums', 'threads', 'thread')]
            if slug_parts:
                return slug_parts[-1]  # Return the last meaningful segment

        return ""

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        if not text:
            return ""
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _parse_count(self, text: str) -> int:
        """Parse count from text like '1,234' or '1.2K'."""
        if not text:
            return 0
        text = text.strip().upper()
        # Handle K/M suffixes
        multiplier = 1
        if text.endswith("K"):
            multiplier = 1000
            text = text[:-1]
        elif text.endswith("M"):
            multiplier = 1000000
            text = text[:-1]
        # Remove commas (but keep decimal points for K/M values)
        text = text.replace(",", "")
        try:
            return int(float(text) * multiplier)
        except ValueError:
            return 0

    def _parse_timestamp(self, text: str) -> str:
        """Parse timestamp text to ISO format."""
        if not text:
            return ""
        text = self._clean_text(text)
        # Try common formats
        formats = [
            "%b %d, %Y",        # Dec 15, 2023
            "%B %d, %Y",       # December 15, 2023
            "%m-%d-%Y",        # 12-15-2023
            "%Y-%m-%d",        # 2023-12-15
            "%d %b %Y",        # 15 Dec 2023
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(text, fmt)
                return dt.isoformat()
            except ValueError:
                continue
        # Return original if parsing fails
        return text

    # ========================================================================
    # Forum/Subforum Parsing
    # ========================================================================

    def parse_forum_index(self, html: str) -> List[Forum]:
        """
        Parse the main forum index page to extract all forums and subforums.

        Args:
            html: Raw HTML of forum index page

        Returns:
            List of Forum objects representing top-level forums
        """
        soup = BeautifulSoup(html, "html.parser")
        forums = []

        # vBulletin 6 uses various structures - try multiple selectors
        # Look for forum category containers
        forum_containers = soup.select(
            ".b-channel, .forumbit, .forum-item, "
            "[class*='forum'], [data-node-id]"
        )

        # Also look for links with forum patterns
        forum_links = soup.select(
            "a[href*='/forum/'], a[href*='/forums/'], "
            ".b-channel__title a, .forum-title a"
        )

        seen_ids = set()

        for link in forum_links:
            href = link.get("href", "")
            if not href:
                continue

            url = self._resolve_url(href)

            # Check if this is a thread URL (numeric ID in the last path segment)
            # Thread URLs look like: /forum/s14/general/1330282-thread-title
            # Forum URLs look like: /forum/5-general or /forum/s14/general
            path = urlparse(url).path
            parts = [p for p in path.strip("/").split("/") if p]
            if len(parts) >= 2:
                last_segment = parts[-1]
                # If last segment starts with digits followed by dash, it's likely a thread
                if "-" in last_segment:
                    first_part = last_segment.split("-")[0]
                    # Thread IDs are typically large numbers, forum IDs are small
                    # Also check path depth - threads in nested forums have 4+ parts
                    if first_part.isdigit() and (int(first_part) > 1000 or len(parts) > 3):
                        continue  # This is a thread, not a forum

            # Get the forum ID (numeric or slug)
            forum_id = self._extract_id(url, allow_slug=True)

            if not forum_id or forum_id in seen_ids:
                continue
            seen_ids.add(forum_id)

            title = self._clean_text(link.get_text())
            if not title:
                continue

            # Try to find description
            description = ""
            parent = link.find_parent(["div", "li", "tr"])
            if parent:
                desc_elem = parent.select_one(
                    ".forum-desc, .b-channel__description, "
                    "[class*='description']"
                )
                if desc_elem:
                    description = self._clean_text(desc_elem.get_text())

            forums.append(
                Forum(
                    forum_id=forum_id,
                    title=title,
                    url=url,
                    description=description,
                )
            )

        return forums

    def parse_forum_page(self, html: str, forum_id: str) -> tuple[List[Forum], List[Thread], Pagination]:
        """
        Parse a forum page to extract subforums, threads, and pagination.

        Args:
            html: Raw HTML of forum page
            forum_id: ID of current forum

        Returns:
            Tuple of (subforums, threads, pagination)
        """
        soup = BeautifulSoup(html, "html.parser")
        subforums = []
        threads = []

        # Parse subforums
        subforum_links = soup.select(
            ".b-channel__title a, .subforum-item a, "
            "[class*='subforum'] a, .child-forum a"
        )
        seen_forum_ids = set()

        for link in subforum_links:
            href = link.get("href", "")
            if "/forum/" not in href and "/forums/" not in href:
                continue
            url = self._resolve_url(href)
            sub_id = self._extract_id(url)
            if sub_id and sub_id not in seen_forum_ids and sub_id != forum_id:
                seen_forum_ids.add(sub_id)
                subforums.append(
                    Forum(
                        forum_id=sub_id,
                        title=self._clean_text(link.get_text()),
                        url=url,
                        parent_id=forum_id,
                    )
                )

        # Parse threads
        thread_containers = soup.select(
            ".b-post, .thread-item, [class*='threadbit'], "
            "tr[id*='thread'], li[class*='thread'], .topic-item"
        )

        for container in thread_containers:
            thread = self._parse_thread_from_listing(container, forum_id)
            if thread:
                threads.append(thread)

        # If no containers found, try direct thread links
        if not threads:
            thread_links = soup.select("a[href*='/threads/']")
            seen_thread_ids = set()
            for link in thread_links:
                href = link.get("href", "")
                url = self._resolve_url(href)
                thread_id = self._extract_id(url)
                if thread_id and thread_id not in seen_thread_ids:
                    seen_thread_ids.add(thread_id)
                    title = self._clean_text(link.get_text())
                    if title and len(title) > 3:  # Filter out navigation links
                        threads.append(
                            Thread(
                                thread_id=thread_id,
                                title=title,
                                url=url,
                                forum_id=forum_id,
                            )
                        )

        # Parse pagination
        pagination = self._parse_pagination(soup)

        return subforums, threads, pagination

    def _parse_thread_from_listing(self, container: Tag, forum_id: str) -> Optional[Thread]:
        """Parse thread information from a listing container."""
        # Find thread link
        # vBulletin 6 uses <a class="topic-title"> directly, not nested in a container
        link = container.select_one(
            "a[href*='/threads/'], .thread-title a, "
            ".b-post__title a, [class*='title'] a, "
            "a.topic-title, a.js-topic-title"
        )
        if not link:
            return None

        href = link.get("href", "")
        url = self._resolve_url(href)
        thread_id = self._extract_id(url)
        if not thread_id:
            return None

        title = self._clean_text(link.get_text())
        if not title:
            return None

        thread = Thread(
            thread_id=thread_id,
            title=title,
            url=url,
            forum_id=forum_id,
        )

        # Extract author
        author_elem = container.select_one(
            ".b-post__author a, .author a, [class*='starter'] a, "
            ".username, [class*='author']"
        )
        if author_elem:
            thread.author = self._clean_text(author_elem.get_text())
            author_href = author_elem.get("href", "")
            thread.author_id = self._extract_id(author_href)

        # Extract reply/view counts
        stats = container.select("[class*='count'], [class*='stats'], .b-post__stats span")
        for stat in stats:
            text = self._clean_text(stat.get_text()).lower()
            if "repl" in text or "post" in text:
                thread.reply_count = self._parse_count(text)
            elif "view" in text:
                thread.view_count = self._parse_count(text)

        # Check sticky/locked status
        classes = " ".join(container.get("class", []))
        thread.is_sticky = "sticky" in classes or "pinned" in classes
        thread.is_locked = "locked" in classes or "closed" in classes

        return thread

    # ========================================================================
    # Thread/Post Parsing
    # ========================================================================

    def parse_thread_page(self, html: str, thread_id: str) -> tuple[List[Post], Pagination]:
        """
        Parse a thread page to extract posts and pagination.

        Args:
            html: Raw HTML of thread page
            thread_id: ID of current thread

        Returns:
            Tuple of (posts, pagination)
        """
        soup = BeautifulSoup(html, "html.parser")
        posts = []

        # Find post containers
        post_containers = soup.select(
            ".b-post, [class*='postbit'], [id*='post_'], "
            ".message-container, [data-post-id], article[class*='post']"
        )

        for i, container in enumerate(post_containers):
            post = self._parse_post(container, thread_id, i)
            if post:
                posts.append(post)

        pagination = self._parse_pagination(soup)

        return posts, pagination

    def _parse_post(self, container: Tag, thread_id: str, index: int) -> Optional[Post]:
        """Parse a single post from its container."""
        # Try to find post ID
        post_id = container.get("data-post-id", "")
        if not post_id:
            post_id = container.get("id", "").replace("post_", "").replace("post", "")
        if not post_id:
            # Try to find from anchor
            anchor = container.select_one("[id*='post']")
            if anchor:
                post_id = anchor.get("id", "").replace("post_", "").replace("post", "")
        if not post_id:
            post_id = f"{thread_id}_{index}"

        # Find author
        author = ""
        author_id = ""
        author_elem = container.select_one(
            ".b-post__author a, .username, [class*='author'] a, "
            ".postcontent .username, [itemprop='author']"
        )
        if author_elem:
            author = self._clean_text(author_elem.get_text())
            author_href = author_elem.get("href", "")
            author_id = self._extract_id(author_href)

        # Find content
        content_elem = container.select_one(
            ".b-post__content, .postcontent, [class*='message-body'], "
            ".post-content, [itemprop='text'], .js-post__content-text"
        )

        content_html = ""
        content_text = ""
        images = []
        quotes = []

        if content_elem:
            content_html = str(content_elem)
            content_text = self._clean_text(content_elem.get_text())

            # Extract images
            for img in content_elem.select("img"):
                src = img.get("src", "") or img.get("data-src", "")
                if src and not self._is_emoji_or_icon(src):
                    images.append(self._resolve_url(src))

            # Extract quotes
            for quote in content_elem.select("blockquote, .bbcode_quote, [class*='quote']"):
                quote_text = self._clean_text(quote.get_text())
                if quote_text:
                    quotes.append(quote_text[:500])  # Truncate long quotes

        # Find timestamp
        created_at = ""
        time_elem = container.select_one(
            "time, [class*='date'], .b-post__timestamp, "
            "[datetime], .postcontent .date"
        )
        if time_elem:
            created_at = time_elem.get("datetime", "") or time_elem.get("title", "")
            if not created_at:
                created_at = self._parse_timestamp(time_elem.get_text())

        return Post(
            post_id=post_id,
            thread_id=thread_id,
            author=author,
            author_id=author_id,
            content_html=content_html,
            content_text=content_text,
            created_at=created_at,
            post_number=index + 1,
            is_first_post=(index == 0),
            images=images,
            quotes=quotes,
        )

    def _is_emoji_or_icon(self, src: str) -> bool:
        """Check if image URL is likely an emoji or icon."""
        src_lower = src.lower()
        skip_patterns = [
            "emoji", "smil", "icon", "avatar",
            "/misc/", "/images/buttons/",
            ".gif",  # Most animated emojis
        ]
        return any(p in src_lower for p in skip_patterns)

    # ========================================================================
    # Pagination
    # ========================================================================

    def _parse_pagination(self, soup: BeautifulSoup) -> Pagination:
        """Parse pagination information from page."""
        pagination = Pagination()

        # Look for pagination container
        pager = soup.select_one(
            ".pagination, .b-pagination, [class*='pagenav'], "
            ".pageNav, [class*='pager']"
        )

        if not pager:
            return pagination

        # Find current page
        current = pager.select_one(
            ".current, [class*='active'], [aria-current='page'], "
            "strong, .b-pagination__current"
        )
        if current:
            try:
                pagination.current_page = int(self._clean_text(current.get_text()))
            except ValueError:
                pass

        # Find total pages (often in "Page X of Y" or last page link)
        page_links = pager.select("a[href]")
        max_page = pagination.current_page

        for link in page_links:
            text = self._clean_text(link.get_text())
            try:
                page_num = int(text)
                max_page = max(max_page, page_num)
            except ValueError:
                pass

        # Check for "last" link
        last_link = pager.select_one("[class*='last'] a, a[title*='Last']")
        if last_link:
            href = last_link.get("href", "")
            # Try to extract page number from URL
            if "page=" in href:
                try:
                    page_num = int(href.split("page=")[-1].split("&")[0])
                    max_page = max(max_page, page_num)
                except ValueError:
                    pass

        pagination.total_pages = max_page
        pagination.has_next = pagination.current_page < pagination.total_pages
        pagination.has_prev = pagination.current_page > 1

        # Find next/prev URLs
        next_link = pager.select_one(
            "a[class*='next'], [class*='next'] a, a[rel='next'], [aria-label*='Next'] a"
        )
        if next_link:
            pagination.next_url = self._resolve_url(next_link.get("href", ""))

        prev_link = pager.select_one(
            "a[class*='prev'], [class*='prev'] a, a[rel='prev'], [aria-label*='Previous'] a"
        )
        if prev_link:
            pagination.prev_url = self._resolve_url(prev_link.get("href", ""))

        return pagination
