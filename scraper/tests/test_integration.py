#!/usr/bin/env python3
"""
Integration tests for the full scraper pipeline.

Tests the complete flow:
1. Forum discovery → Thread listing → Post scraping → Image download
2. Data validation between stages
3. Checkpoint/resume functionality
"""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scraper.core import (
    ForumConfig,
    Selectors,
    AuthConfig,
    ScraperSession,
    Checkpoint,
    save_json,
    load_json,
    append_jsonl,
    load_jsonl,
    validate_record,
    load_jsonl_validated,
    THREAD_SCHEMA,
    POST_SCHEMA,
    ValidationError,
)
from scraper.parser import VBulletinParser


class TestFullPipelineIntegration:
    """Integration tests for the complete scraper pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    @pytest.fixture
    def fixtures_dir(self):
        """Path to test fixtures."""
        return Path(__file__).parent / "fixtures"

    @pytest.fixture
    def mock_config(self):
        """Create a mock forum configuration."""
        return ForumConfig(
            name="Test Forum",
            forum_id="test",
            base_url="https://test.forum.com",
            platform="vbulletin6",
            selectors=Selectors(
                forum_link=".b-channel__title a, .forum-title a",
                thread_container=".b-post, .threadbit",
                thread_link=".b-post__title a",
                thread_author=".b-post__author a",
                thread_stats=".b-post__stats",
                post_container=".b-post",
                post_author=".b-post__author a, .username",
                post_content=".b-post__content",
                post_timestamp="time.b-post__timestamp",
                pagination_next=".pagination .next, a[rel='next'], a.next",
                pagination_last=".pagination .last, a.last",
                subforum_link=".subforum-item a",
            ),
            auth=AuthConfig(enabled=False),
            min_delay=0.0,
            max_delay=0.0,
            retry_delay=0.1,
            max_retries=1,
            backoff_multiplier=1.0,
            timeout=30,
            user_agent="Mozilla/5.0",
            headers={},
            storage_base=Path("/tmp/test"),
            checkpoint_interval=10,
        )

    @pytest.fixture
    def mock_session(self, mock_config, fixtures_dir):
        """Create a mock scraper session that returns fixture data."""
        logger = Mock()
        session = ScraperSession(mock_config, logger)

        # Load fixture files
        fixtures = {
            "index": (fixtures_dir / "forum_index.html").read_text(),
            "forum": (fixtures_dir / "forum_page.html").read_text(),
            "thread": (fixtures_dir / "thread_page.html").read_text(),
            "empty": (fixtures_dir / "empty_forum.html").read_text(),
        }

        def mock_get_html(url):
            if "forum/10" in url or "faqs" in url.lower():
                return fixtures["forum"]
            elif "threads/" in url:
                return fixtures["thread"]
            elif url.endswith("/") or "index" in url or url == mock_config.base_url:
                return fixtures["index"]
            else:
                return fixtures["empty"]

        session.get_html = mock_get_html
        return session

    def test_forum_discovery_to_thread_listing(self, mock_session, mock_config, temp_dir):
        """Test forum discovery followed by thread listing."""
        parser = VBulletinParser(mock_config.base_url)

        # Stage 1: Discover forums from index
        index_html = mock_session.get_html(mock_config.base_url)
        forums = parser.parse_forum_index(index_html)

        assert len(forums) == 3
        assert forums[0].forum_id == "5"
        assert forums[1].forum_id == "10"
        assert forums[2].forum_id == "20"

        # Save forum data (mimics stage 1 output)
        forums_output = temp_dir / "forums.json"
        save_json(
            {
                "base_url": mock_config.base_url,
                "forums": [f.to_dict() for f in forums],
                "total_forums": len(forums),
            },
            forums_output,
        )

        # Verify saved data
        loaded = load_json(forums_output)
        assert loaded["total_forums"] == 3
        assert loaded["forums"][0]["forum_id"] == "5"

        # Stage 2: Get threads from a forum
        forum_html = mock_session.get_html(f"{mock_config.base_url}/forum/10-faqs")
        subforums, threads, pagination = parser.parse_forum_page(forum_html, "10")

        assert len(subforums) == 3  # 3 subforums in fixture
        assert len(threads) == 3  # 3 threads in fixture
        assert threads[0].thread_id == "1001"
        assert threads[0].title == "Help! My E30 won't start"

        # Save threads as JSONL (mimics stage 2 output)
        threads_output = temp_dir / "threads_10.jsonl"
        for thread in threads:
            append_jsonl(thread.to_dict(), threads_output)

        # Verify JSONL output
        loaded_threads = list(load_jsonl(threads_output))
        assert len(loaded_threads) == 3

    def test_thread_to_posts_pipeline(self, mock_session, mock_config, temp_dir):
        """Test thread scraping to post extraction."""
        parser = VBulletinParser(mock_config.base_url)

        # Get thread page
        thread_html = mock_session.get_html(
            f"{mock_config.base_url}/threads/1001-help-my-e30-wont-start"
        )

        # Parse posts
        posts, pagination = parser.parse_thread_page(thread_html, "1001")

        assert len(posts) == 3
        assert posts[0].post_id == "5001"
        assert posts[0].author == "johnsmith"
        assert "E30 M3 won't start" in posts[0].content_text
        assert len(posts[0].images) == 1

        assert posts[1].post_id == "5002"
        assert posts[1].author == "m3expert"

        assert posts[2].post_id == "5003"
        assert len(posts[2].images) == 2  # Excludes smilies

        # Verify pagination
        assert pagination.has_next
        assert pagination.next_url is not None

        # Save posts as JSONL
        posts_output = temp_dir / "posts_1001.jsonl"
        for post in posts:
            append_jsonl(post.to_dict(), posts_output)

        # Validate saved posts
        loaded_posts = list(load_jsonl(posts_output))
        assert len(loaded_posts) == 3
        assert all("post_id" in p for p in loaded_posts)
        assert all("author" in p for p in loaded_posts)
        assert all("content_text" in p for p in loaded_posts)

    def test_checkpoint_resume_across_stages(self, mock_config, temp_dir):
        """Test checkpoint save/load functionality for resume capability."""
        checkpoint_path = temp_dir / "checkpoint.json"

        # Stage 2 simulation: Start scraping threads
        checkpoint = Checkpoint(stage="threads_42")
        checkpoint.mark_completed("thread_1001")
        checkpoint.mark_completed("thread_1002")
        checkpoint.save(checkpoint_path)

        # Simulate restart - load checkpoint
        restored = Checkpoint.load_or_create(checkpoint_path, "threads_42")

        assert restored.is_completed("thread_1001")
        assert restored.is_completed("thread_1002")
        assert not restored.is_completed("thread_1003")

        # Continue with remaining work
        restored.mark_completed("thread_1003")
        restored.save(checkpoint_path)

        # Final verification
        final = Checkpoint.load_or_create(checkpoint_path, "threads_42")
        assert len(final.completed_ids) == 3

    def test_data_validation_between_stages(self, temp_dir):
        """Test schema validation for data passed between stages."""
        # Valid thread record
        valid_thread = {
            "thread_id": "1001",
            "title": "Test Thread",
            "url": "https://test.com/threads/1001",
            "forum_id": "42",
            "author": "testuser",
            "replies": 10,
            "views": 100,
        }

        # Valid post record
        valid_post = {
            "post_id": "5001",
            "thread_id": "1001",
            "author": "testuser",
            "content": "This is the post content",
            "timestamp": "2023-12-01T10:30:00",
            "images": ["https://example.com/image.jpg"],
        }

        # Test thread validation (returns warnings list, raises on error)
        warnings = validate_record(valid_thread, THREAD_SCHEMA, "thread")
        assert len(warnings) == 0, f"Thread validation failed: {warnings}"

        # Test post validation
        warnings = validate_record(valid_post, POST_SCHEMA, "post")
        assert len(warnings) == 0, f"Post validation failed: {warnings}"

        # Test invalid records - should raise ValidationError
        invalid_thread = {"thread_id": "1001"}  # Missing required fields
        with pytest.raises(ValidationError):
            validate_record(invalid_thread, THREAD_SCHEMA, "thread")

        # Write valid data and load with validation
        valid_file = temp_dir / "valid_threads.jsonl"
        append_jsonl(valid_thread, valid_file)

        records = load_jsonl_validated(valid_file, THREAD_SCHEMA, "thread")
        assert len(records) == 1

    def test_full_pipeline_with_empty_forum(self, mock_session, mock_config):
        """Test pipeline handles empty forums gracefully."""
        parser = VBulletinParser(mock_config.base_url)

        # Request empty forum
        empty_html = mock_session.get_html(f"{mock_config.base_url}/forum/99-empty")

        subforums, threads, pagination = parser.parse_forum_page(empty_html, "99")

        assert len(subforums) == 0
        assert len(threads) == 0
        assert not pagination.has_next

    def test_image_url_extraction(self, mock_session, mock_config):
        """Test image URL extraction from posts."""
        parser = VBulletinParser(mock_config.base_url)

        thread_html = mock_session.get_html(
            f"{mock_config.base_url}/threads/1001-test"
        )
        posts, _ = parser.parse_thread_page(thread_html, "1001")

        # Collect all images
        all_images = []
        for post in posts:
            all_images.extend(post.images)

        # Should have 3 real images (excluding smilies)
        assert len(all_images) == 3
        assert "engine_bay.jpg" in all_images[0]
        assert "spark_plug_check.jpg" in all_images[1]
        assert "fuel_rail.jpg" in all_images[2]

        # Verify no smilies included
        for img in all_images:
            assert "smilies" not in img
            assert ".gif" not in img.lower() or "animated" not in img.lower()

    def test_pagination_chain(self, mock_config, fixtures_dir):
        """Test pagination across multiple pages."""
        parser = VBulletinParser(mock_config.base_url)

        # Parse forum page with pagination
        forum_html = (fixtures_dir / "forum_page.html").read_text()
        _, _, pagination = parser.parse_forum_page(forum_html, "10")

        assert pagination.current_page == 1
        assert pagination.total_pages == 5
        assert pagination.has_next
        assert "/forum/10-faqs?page=2" in pagination.next_url

        # Parse thread page with pagination
        thread_html = (fixtures_dir / "thread_page.html").read_text()
        _, pagination = parser.parse_thread_page(thread_html, "1001")

        assert pagination.has_next
        assert "page=2" in pagination.next_url


class TestDataIntegrity:
    """Tests for data integrity across pipeline stages."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path, ignore_errors=True)

    def test_jsonl_append_and_read(self, temp_dir):
        """Test JSONL file operations maintain data integrity."""
        output_file = temp_dir / "data.jsonl"

        # Write records
        records = [
            {"id": "1", "data": "first"},
            {"id": "2", "data": "second"},
            {"id": "3", "data": "third"},
        ]

        for record in records:
            append_jsonl(record, output_file)

        # Read back
        loaded = list(load_jsonl(output_file))

        assert len(loaded) == 3
        for i, record in enumerate(loaded):
            assert record["id"] == records[i]["id"]
            assert record["data"] == records[i]["data"]

    def test_json_unicode_handling(self, temp_dir):
        """Test JSON handling of unicode characters (common in forums)."""
        output_file = temp_dir / "unicode.jsonl"

        records = [
            {"content": "BMW E30 M3 — 燃料噴射系統"},
            {"content": "Öldruckschalter prüfen"},
            {"content": "🔧 Fixing the engine"},
        ]

        for record in records:
            append_jsonl(record, output_file)

        loaded = list(load_jsonl(output_file))
        assert len(loaded) == 3
        assert "燃料噴射系統" in loaded[0]["content"]
        assert "Öldruckschalter" in loaded[1]["content"]

    def test_checkpoint_thread_safety(self, temp_dir):
        """Test checkpoint operations are safe for concurrent access."""
        checkpoint_path = temp_dir / "checkpoint.json"

        checkpoint = Checkpoint(stage="test")

        # Simulate rapid updates
        for i in range(100):
            checkpoint.mark_completed(f"item_{i}")

        checkpoint.save(checkpoint_path)

        # Verify all items saved
        loaded = Checkpoint.load_or_create(checkpoint_path, "test")
        assert len(loaded.completed_ids) == 100


class TestPipelineResilience:
    """Tests for pipeline error handling and recovery."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock forum configuration."""
        return ForumConfig(
            name="Test Forum",
            forum_id="test",
            base_url="https://test.forum.com",
            platform="vbulletin6",
            selectors=Selectors(),
            auth=AuthConfig(enabled=False),
            min_delay=0.0,
            max_delay=0.0,
            retry_delay=0.1,
            max_retries=1,
            backoff_multiplier=1.0,
            timeout=30,
            user_agent="Mozilla/5.0",
            headers={},
            storage_base=Path("/tmp/test"),
            checkpoint_interval=10,
        )

    def test_handles_malformed_html(self, mock_config):
        """Test parser handles malformed HTML gracefully."""
        parser = VBulletinParser(mock_config.base_url)

        malformed_html = "<html><div class='unclosed'><p>Content"

        # Should not raise exception
        forums = parser.parse_forum_index(malformed_html)
        assert isinstance(forums, list)

    def test_handles_missing_elements(self, mock_config):
        """Test parser handles missing expected elements."""
        parser = VBulletinParser(mock_config.base_url)

        minimal_html = "<html><body><p>No forum structure here</p></body></html>"

        forums = parser.parse_forum_index(minimal_html)
        assert len(forums) == 0

        subforums, threads, pagination = parser.parse_forum_page(minimal_html, "42")
        assert len(subforums) == 0
        assert len(threads) == 0

    def test_session_handles_errors_gracefully(self, mock_config):
        """Test session handles errors gracefully and logs them."""
        logger = Mock()
        session = ScraperSession(mock_config, logger)

        # Test that request exceptions are handled (returns None)
        def mock_get_error(*args, **kwargs):
            raise requests.exceptions.ConnectionError("Connection failed")

        with patch.object(session.session, "get", mock_get_error):
            result = session.get_html("https://test.forum.com/page")

        # Should return None on error
        assert result is None
        assert session.error_count == 1

    def test_session_successful_request(self, mock_config):
        """Test session handles successful requests."""
        logger = Mock()
        session = ScraperSession(mock_config, logger)

        def mock_get_success(*args, **kwargs):
            response = Mock()
            response.status_code = 200
            response.text = "<html><body>Success</body></html>"
            response.content = b"<html><body>Success</body></html>"
            response.raise_for_status = Mock()
            return response

        with patch.object(session.session, "get", mock_get_success):
            result = session.get_html("https://test.forum.com/page")

        assert result == "<html><body>Success</body></html>"
        assert session.request_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
