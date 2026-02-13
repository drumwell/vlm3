"""
Tests for 01_reconstruct.py - Thread Reconstruction

Tests the grouping of raw posts into complete threads.
"""

import json
import pytest
from pathlib import Path

from pipeline import reconstruct


# ============================================================================
# Test: filter_images
# ============================================================================

class TestFilterImages:
    """Tests for image URL filtering."""

    def test_filter_images_strips_sponsor_urls(self):
        """Removes member_sponsors/ URLs."""
        images = [
            "https://example.com/photo.jpg",
            "https://example.com/member_sponsors/logo.jpg",
        ]
        result = reconstruct.filter_images(images)
        assert len(result) == 1
        assert "photo.jpg" in result[0]

    def test_filter_images_strips_avatars(self):
        """Removes avatar URLs."""
        images = [
            "https://example.com/photo.jpg",
            "https://example.com/avatar/user123.jpg",
        ]
        result = reconstruct.filter_images(images)
        assert len(result) == 1

    def test_filter_images_strips_smilies(self):
        """Removes smiley URLs."""
        images = [
            "https://example.com/photo.jpg",
            "https://example.com/smilies/smile.gif",
        ]
        result = reconstruct.filter_images(images)
        assert len(result) == 1

    def test_filter_images_empty_list(self):
        """Empty input returns empty output."""
        assert reconstruct.filter_images([]) == []

    def test_filter_images_none_input(self):
        """None input returns empty list."""
        assert reconstruct.filter_images(None) == []

    def test_filter_images_keeps_content_images(self):
        """Preserves legitimate content images."""
        images = [
            "https://s6.postimg.org/xrdaz63pd/IMG_1988.jpg",
            "https://imgur.com/abc123.jpg",
        ]
        result = reconstruct.filter_images(images)
        assert len(result) == 2


# ============================================================================
# Test: load_posts
# ============================================================================

class TestLoadPosts:
    """Tests for loading posts from JSONL files."""

    def test_load_posts_groups_by_thread(self, raw_dir_with_posts):
        """Posts are grouped by thread_id."""
        posts = reconstruct.load_posts(raw_dir_with_posts, ["d-i-y"])
        assert "1258841" in posts
        assert len(posts["1258841"]) == 3

    def test_load_posts_adds_forum_field(self, raw_dir_with_posts):
        """Each post gets a forum field."""
        posts = reconstruct.load_posts(raw_dir_with_posts, ["d-i-y"])
        for post in posts["1258841"]:
            assert post["forum"] == "d-i-y"

    def test_load_posts_missing_forum_skipped(self, raw_dir_with_posts):
        """Missing forum files produce a warning, not an error."""
        posts = reconstruct.load_posts(raw_dir_with_posts, ["d-i-y", "nonexistent"])
        assert "1258841" in posts

    def test_load_posts_empty_targets(self, raw_dir_with_posts):
        """Empty target list returns empty dict."""
        posts = reconstruct.load_posts(raw_dir_with_posts, [])
        assert len(posts) == 0

    def test_load_posts_filters_sponsor_images(self, raw_dir_with_posts):
        """Sponsor images are stripped from posts."""
        posts = reconstruct.load_posts(raw_dir_with_posts, ["d-i-y"])
        first_post = [p for p in posts["1258841"] if p["post_number"] == 1][0]
        for url in first_post["images"]:
            assert "member_sponsors" not in url


# ============================================================================
# Test: load_thread_metadata
# ============================================================================

class TestLoadThreadMetadata:
    """Tests for loading thread metadata."""

    def test_load_metadata_returns_dict(self, raw_dir_with_posts):
        """Returns dict keyed by thread_id."""
        metadata = reconstruct.load_thread_metadata(raw_dir_with_posts, ["d-i-y"])
        assert "1258841" in metadata
        assert metadata["1258841"]["title"] == "Valve Adjustment"

    def test_load_metadata_includes_sticky_locked(self, raw_dir_with_posts):
        """Metadata includes is_sticky and is_locked fields."""
        metadata = reconstruct.load_thread_metadata(raw_dir_with_posts, ["d-i-y"])
        assert "is_sticky" in metadata["1258841"]
        assert "is_locked" in metadata["1258841"]


# ============================================================================
# Test: reconstruct_thread
# ============================================================================

class TestReconstructThread:
    """Tests for thread reconstruction."""

    def test_reconstruct_thread_sorts_by_post_number(self, sample_post, sample_reply, sample_noise_reply, sample_thread_metadata):
        """Posts sorted chronologically by post_number."""
        posts = [sample_noise_reply, sample_post, sample_reply]  # Out of order
        for p in posts:
            p["forum"] = "d-i-y"
            p["images"] = reconstruct.filter_images(p.get("images", []))

        thread = reconstruct.reconstruct_thread(
            "1258841", posts, {"1258841": sample_thread_metadata}
        )
        numbers = [p["post_number"] for p in thread["posts"]]
        assert numbers == sorted(numbers)

    def test_reconstruct_thread_uses_metadata_title(self, sample_post, sample_thread_metadata):
        """Thread title comes from metadata, not post."""
        sample_post["forum"] = "d-i-y"
        sample_post["images"] = []
        sample_thread_metadata["title"] = "Custom Title"

        thread = reconstruct.reconstruct_thread(
            "1258841", [sample_post], {"1258841": sample_thread_metadata}
        )
        assert thread["thread_title"] == "Custom Title"

    def test_reconstruct_thread_output_schema(self, sample_post, sample_thread_metadata):
        """Output has all required fields."""
        sample_post["forum"] = "d-i-y"
        sample_post["images"] = []

        thread = reconstruct.reconstruct_thread(
            "1258841", [sample_post], {"1258841": sample_thread_metadata}
        )

        required_keys = [
            "thread_id", "thread_title", "forum", "author",
            "post_count", "first_post_date", "last_post_date",
            "is_sticky", "is_locked", "posts",
        ]
        for key in required_keys:
            assert key in thread, f"Missing key: {key}"

    def test_reconstruct_thread_post_schema(self, sample_post, sample_thread_metadata):
        """Each post has required fields."""
        sample_post["forum"] = "d-i-y"
        sample_post["images"] = []

        thread = reconstruct.reconstruct_thread(
            "1258841", [sample_post], {"1258841": sample_thread_metadata}
        )

        post = thread["posts"][0]
        required_keys = [
            "post_id", "author", "post_number", "is_first_post",
            "content_text", "content_images", "quotes",
        ]
        for key in required_keys:
            assert key in post, f"Missing post key: {key}"

    def test_reconstruct_thread_missing_metadata(self, sample_post):
        """Works without metadata (falls back to post data)."""
        sample_post["forum"] = "d-i-y"
        sample_post["images"] = []

        thread = reconstruct.reconstruct_thread("1258841", [sample_post], {})
        assert thread["thread_title"] == sample_post["thread_title"]
        assert thread["is_sticky"] is False

    def test_reconstruct_thread_empty_posts(self):
        """Returns None for empty posts list instead of crashing."""
        result = reconstruct.reconstruct_thread("12345", [], {})
        assert result is None


# ============================================================================
# Test: end-to-end with files
# ============================================================================

class TestReconstructEndToEnd:
    """Integration tests using file I/O."""

    def test_full_pipeline(self, raw_dir_with_posts, tmp_path, config_file):
        """Reconstruct from raw files, verify output."""
        output_dir = tmp_path / "threads"
        output_dir.mkdir()

        config = reconstruct.load_config(config_file)
        target_forums = config["forums"]["targets"]
        # Only d-i-y has data in our fixture
        posts_by_thread = reconstruct.load_posts(raw_dir_with_posts, ["d-i-y"])
        metadata = reconstruct.load_thread_metadata(raw_dir_with_posts, ["d-i-y"])

        for tid, posts in posts_by_thread.items():
            thread = reconstruct.reconstruct_thread(tid, posts, metadata)
            with open(output_dir / f"{tid}.json", "w") as f:
                json.dump(thread, f, indent=2)

        output_files = list(output_dir.glob("*.json"))
        assert len(output_files) == 1

        with open(output_files[0]) as f:
            thread = json.load(f)
        assert thread["thread_id"] == "1258841"
        assert thread["post_count"] == 3
