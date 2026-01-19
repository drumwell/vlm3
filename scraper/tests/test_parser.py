"""
Unit tests for scraper/parser.py

Tests the vBulletin HTML parser: forum discovery, thread listing,
post extraction, and pagination.
"""

from pathlib import Path

import pytest

from scraper.parser import (
    VBulletinParser,
    Forum,
    Thread,
    Post,
    Pagination,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parser():
    """Create a parser instance for testing."""
    return VBulletinParser("https://example.com")


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def forum_index_html(fixtures_dir):
    """Load forum index fixture."""
    return (fixtures_dir / "forum_index.html").read_text()


@pytest.fixture
def forum_page_html(fixtures_dir):
    """Load forum page fixture."""
    return (fixtures_dir / "forum_page.html").read_text()


@pytest.fixture
def thread_page_html(fixtures_dir):
    """Load thread page fixture."""
    return (fixtures_dir / "thread_page.html").read_text()


@pytest.fixture
def empty_forum_html(fixtures_dir):
    """Load empty forum fixture."""
    return (fixtures_dir / "empty_forum.html").read_text()


# =============================================================================
# Forum Index Parsing Tests
# =============================================================================


class TestForumIndexParsing:
    """Tests for parsing the main forum index page."""

    def test_extracts_forums(self, parser, forum_index_html):
        """Should extract all top-level forums from index."""
        forums = parser.parse_forum_index(forum_index_html)

        assert len(forums) == 3

    def test_forum_has_id(self, parser, forum_index_html):
        """Each forum should have an ID extracted from URL."""
        forums = parser.parse_forum_index(forum_index_html)

        forum_ids = [f.forum_id for f in forums]
        assert "5" in forum_ids
        assert "10" in forum_ids
        assert "20" in forum_ids

    def test_forum_has_title(self, parser, forum_index_html):
        """Each forum should have a title."""
        forums = parser.parse_forum_index(forum_index_html)

        titles = [f.title for f in forums]
        assert "General E30 M3 Discussion" in titles
        assert "FAQs" in titles
        assert "Classifieds" in titles

    def test_forum_has_url(self, parser, forum_index_html):
        """Each forum should have an absolute URL."""
        forums = parser.parse_forum_index(forum_index_html)

        for forum in forums:
            assert forum.url.startswith("https://")
            assert "/forum/" in forum.url

    def test_forum_has_description(self, parser, forum_index_html):
        """Forums should have descriptions when available."""
        forums = parser.parse_forum_index(forum_index_html)

        faqs_forum = next(f for f in forums if f.forum_id == "10")
        assert "Frequently asked questions" in faqs_forum.description

    def test_forum_to_dict(self, parser, forum_index_html):
        """Forum.to_dict should serialize correctly."""
        forums = parser.parse_forum_index(forum_index_html)
        forum_dict = forums[0].to_dict()

        assert "forum_id" in forum_dict
        assert "title" in forum_dict
        assert "url" in forum_dict
        assert "subforums" in forum_dict


# =============================================================================
# Forum Page Parsing Tests
# =============================================================================


class TestForumPageParsing:
    """Tests for parsing a forum page (subforums + thread listings)."""

    def test_extracts_subforums(self, parser, forum_page_html):
        """Should extract subforums from forum page."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        assert len(subforums) >= 3
        subforum_ids = [sf.forum_id for sf in subforums]
        assert "42" in subforum_ids  # No-Start
        assert "43" in subforum_ids  # Water Leaks
        assert "44" in subforum_ids  # ECU Chips

    def test_subforum_has_parent_id(self, parser, forum_page_html):
        """Subforums should have parent_id set."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        for subforum in subforums:
            assert subforum.parent_id == "10"

    def test_extracts_threads(self, parser, forum_page_html):
        """Should extract thread listings from forum page."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        assert len(threads) >= 3

    def test_thread_has_id(self, parser, forum_page_html):
        """Each thread should have an ID."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        thread_ids = [t.thread_id for t in threads]
        assert "1001" in thread_ids
        assert "1002" in thread_ids

    def test_thread_has_title(self, parser, forum_page_html):
        """Each thread should have a title."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        titles = [t.title for t in threads]
        assert any("E30 won't start" in t for t in titles)
        assert any("READ FIRST" in t for t in titles)

    def test_thread_has_author(self, parser, forum_page_html):
        """Threads should have author information."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        thread_1001 = next(t for t in threads if t.thread_id == "1001")
        assert thread_1001.author == "johnsmith"

    def test_thread_has_forum_id(self, parser, forum_page_html):
        """Threads should have their forum_id set."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        for thread in threads:
            assert thread.forum_id == "10"

    def test_thread_sticky_detected(self, parser, forum_page_html):
        """Sticky threads should be marked as sticky."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        thread_1002 = next(t for t in threads if t.thread_id == "1002")
        assert thread_1002.is_sticky is True

    def test_thread_locked_detected(self, parser, forum_page_html):
        """Locked threads should be marked as locked."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        thread_1003 = next(t for t in threads if t.thread_id == "1003")
        assert thread_1003.is_locked is True

    def test_thread_to_dict(self, parser, forum_page_html):
        """Thread.to_dict should serialize correctly."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")
        thread_dict = threads[0].to_dict()

        assert "thread_id" in thread_dict
        assert "title" in thread_dict
        assert "author" in thread_dict
        assert "is_sticky" in thread_dict

    def test_empty_forum_returns_empty_lists(self, parser, empty_forum_html):
        """Empty forum should return empty thread list."""
        subforums, threads, pagination = parser.parse_forum_page(empty_forum_html, "99")

        assert threads == []


# =============================================================================
# Pagination Tests
# =============================================================================


class TestPaginationParsing:
    """Tests for pagination extraction."""

    def test_pagination_current_page(self, parser, forum_page_html):
        """Should identify current page number."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        assert pagination.current_page == 1

    def test_pagination_total_pages(self, parser, forum_page_html):
        """Should identify total page count."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        assert pagination.total_pages == 5  # From "Last" link

    def test_pagination_has_next(self, parser, forum_page_html):
        """First page should have next page."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        assert pagination.has_next is True
        assert pagination.has_prev is False

    def test_pagination_next_url(self, parser, forum_page_html):
        """Should extract next page URL."""
        subforums, threads, pagination = parser.parse_forum_page(forum_page_html, "10")

        assert pagination.next_url is not None
        assert "page=2" in pagination.next_url

    def test_single_page_pagination(self, parser, empty_forum_html):
        """Single page forum should have no next/prev."""
        subforums, threads, pagination = parser.parse_forum_page(empty_forum_html, "99")

        assert pagination.total_pages == 1
        assert pagination.has_next is False
        assert pagination.has_prev is False


# =============================================================================
# Thread/Post Parsing Tests
# =============================================================================


class TestThreadPageParsing:
    """Tests for parsing thread pages (post extraction)."""

    def test_extracts_posts(self, parser, thread_page_html):
        """Should extract all posts from thread page."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        assert len(posts) == 3

    def test_post_has_id(self, parser, thread_page_html):
        """Each post should have an ID."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_ids = [p.post_id for p in posts]
        assert "5001" in post_ids
        assert "5002" in post_ids
        assert "5003" in post_ids

    def test_post_has_thread_id(self, parser, thread_page_html):
        """Posts should have their thread_id set."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        for post in posts:
            assert post.thread_id == "1001"

    def test_post_has_author(self, parser, thread_page_html):
        """Posts should have author information."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5001 = next(p for p in posts if p.post_id == "5001")
        assert post_5001.author == "johnsmith"

        post_5002 = next(p for p in posts if p.post_id == "5002")
        assert post_5002.author == "m3expert"

    def test_post_has_content_text(self, parser, thread_page_html):
        """Posts should have text content extracted."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5001 = next(p for p in posts if p.post_id == "5001")
        assert "1988 E30 M3 won't start" in post_5001.content_text
        assert "fuel pump relay" in post_5001.content_text

    def test_post_has_content_html(self, parser, thread_page_html):
        """Posts should preserve HTML content."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5001 = next(p for p in posts if p.post_id == "5001")
        assert "<p>" in post_5001.content_html

    def test_post_has_timestamp(self, parser, thread_page_html):
        """Posts should have timestamp."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5001 = next(p for p in posts if p.post_id == "5001")
        assert post_5001.created_at != ""
        assert "2023-12-01" in post_5001.created_at

    def test_first_post_flagged(self, parser, thread_page_html):
        """First post should be flagged as is_first_post."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        assert posts[0].is_first_post is True
        assert posts[1].is_first_post is False

    def test_post_number_sequential(self, parser, thread_page_html):
        """Post numbers should be sequential."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        assert posts[0].post_number == 1
        assert posts[1].post_number == 2
        assert posts[2].post_number == 3

    def test_post_to_dict(self, parser, thread_page_html):
        """Post.to_dict should serialize correctly."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")
        post_dict = posts[0].to_dict()

        assert "post_id" in post_dict
        assert "thread_id" in post_dict
        assert "author" in post_dict
        assert "content_text" in post_dict
        assert "images" in post_dict


# =============================================================================
# Image Extraction Tests
# =============================================================================


class TestImageExtraction:
    """Tests for extracting images from posts."""

    def test_extracts_images(self, parser, thread_page_html):
        """Should extract image URLs from posts."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5001 = next(p for p in posts if p.post_id == "5001")
        assert len(post_5001.images) == 1
        assert "engine_bay.jpg" in post_5001.images[0]

    def test_extracts_multiple_images(self, parser, thread_page_html):
        """Should extract multiple images from a post."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5003 = next(p for p in posts if p.post_id == "5003")
        # Should have 2 real images, not the smiley
        assert len(post_5003.images) == 2
        assert any("spark_plug_check.jpg" in img for img in post_5003.images)
        assert any("fuel_rail.jpg" in img for img in post_5003.images)

    def test_filters_emoji_images(self, parser, thread_page_html):
        """Should filter out emoji/smiley images."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5003 = next(p for p in posts if p.post_id == "5003")
        # Should not include the smiley
        assert not any("smilies" in img for img in post_5003.images)
        assert not any(".gif" in img for img in post_5003.images)

    def test_resolves_relative_image_urls(self, parser, thread_page_html):
        """Relative image URLs should be resolved to absolute."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        for post in posts:
            for img_url in post.images:
                assert img_url.startswith("http")


# =============================================================================
# Quote Extraction Tests
# =============================================================================


class TestQuoteExtraction:
    """Tests for extracting quoted content from posts."""

    def test_extracts_quotes(self, parser, thread_page_html):
        """Should extract quoted text from posts."""
        posts, pagination = parser.parse_thread_page(thread_page_html, "1001")

        post_5002 = next(p for p in posts if p.post_id == "5002")
        assert len(post_5002.quotes) == 1
        assert "fuel pump relay" in post_5002.quotes[0]


# =============================================================================
# URL Utilities Tests
# =============================================================================


class TestParserUrlUtilities:
    """Tests for parser URL utility methods."""

    def test_resolve_url_absolute(self, parser):
        """Absolute URLs should pass through."""
        result = parser._resolve_url("https://other.com/image.jpg")
        assert result == "https://other.com/image.jpg"

    def test_resolve_url_relative(self, parser):
        """Relative URLs should be resolved."""
        result = parser._resolve_url("/forum/42")
        assert result == "https://example.com/forum/42"

    def test_resolve_url_protocol_relative(self, parser):
        """Protocol-relative URLs should get https."""
        result = parser._resolve_url("//cdn.example.com/img.jpg")
        assert result == "https://cdn.example.com/img.jpg"

    def test_extract_id_from_url(self, parser):
        """Should extract ID from URL path."""
        result = parser._extract_id("/threads/12345-thread-title")
        assert result == "12345"

    def test_extract_id_handles_no_slug(self, parser):
        """Should handle URLs without slug."""
        result = parser._extract_id("/forum/42")
        assert result == "42"

    def test_clean_text_normalizes_whitespace(self, parser):
        """Should normalize whitespace in text."""
        result = parser._clean_text("  Hello   World  \n\t  ")
        assert result == "Hello World"

    def test_parse_count_with_comma(self, parser):
        """Should parse counts with comma separators."""
        assert parser._parse_count("1,234") == 1234

    def test_parse_count_with_k_suffix(self, parser):
        """Should parse counts with K suffix."""
        assert parser._parse_count("10K") == 10000
        assert parser._parse_count("1.5K") == 1500

    def test_parse_count_with_m_suffix(self, parser):
        """Should parse counts with M suffix."""
        assert parser._parse_count("2M") == 2000000


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_malformed_html(self, parser):
        """Should handle malformed HTML gracefully."""
        bad_html = "<html><body><div class='unclosed"

        # Should not raise
        forums = parser.parse_forum_index(bad_html)
        assert forums == []

    def test_empty_html(self, parser):
        """Should handle empty HTML."""
        forums = parser.parse_forum_index("")
        assert forums == []

    def test_no_forums_found(self, parser):
        """Should return empty list if no forums found."""
        html = "<html><body><p>No forums here</p></body></html>"
        forums = parser.parse_forum_index(html)
        assert forums == []

    def test_missing_optional_fields(self, parser):
        """Should handle missing optional fields gracefully."""
        html = """
        <div class="b-channel">
            <a href="/forum/1-test">Test Forum</a>
        </div>
        """
        forums = parser.parse_forum_index(html)

        assert len(forums) == 1
        assert forums[0].description == ""  # Optional field has default
