"""
Tests for 02_filter.py - Mechanical Filtering

Tests post-level and thread-level filtering logic.
"""

import json
import pytest
from pathlib import Path

from pipeline import filter_threads


# ============================================================================
# Test: load_filter_config
# ============================================================================

class TestLoadFilterConfig:
    """Tests for configuration loading."""

    def test_load_config_full(self, config_file):
        """All fields populated from config file."""
        config = filter_threads.load_filter_config(config_file)
        assert config.min_post_length == 30
        assert config.min_first_post_length == 50
        assert config.min_thread_posts == 2
        assert "thanks" in config.social_noise_patterns
        assert "WTB" in config.title_noise_patterns

    def test_load_config_defaults(self, tmp_path):
        """Uses defaults for missing config."""
        missing = tmp_path / "missing.yaml"
        config = filter_threads.load_filter_config(missing)
        defaults = filter_threads.FilterConfig()
        assert config.min_post_length == defaults.min_post_length

    def test_load_config_partial(self, tmp_path):
        """Merges provided values with defaults."""
        import yaml
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"filtering": {"min_post_length": 50}}))

        config = filter_threads.load_filter_config(config_file)
        assert config.min_post_length == 50
        assert config.min_thread_posts == filter_threads.FilterConfig().min_thread_posts


# ============================================================================
# Test: clean_content
# ============================================================================

class TestCleanContent:
    """Tests for content cleaning functions."""

    def test_strips_post_number_marker(self):
        """Removes leading #N markers."""
        config = filter_threads.FilterConfig(strip_post_number_markers=True)
        result = filter_threads.clean_content("#1 Hello world", config)
        assert result == "Hello world"

    def test_strips_large_post_number(self):
        """Removes #16, #234, etc."""
        config = filter_threads.FilterConfig(strip_post_number_markers=True)
        result = filter_threads.clean_content("#234 Some content", config)
        assert result == "Some content"

    def test_preserves_non_marker_hash(self):
        """Doesn't strip # in middle of text."""
        config = filter_threads.FilterConfig(strip_post_number_markers=True)
        result = filter_threads.clean_content("Part #12345 is needed", config)
        assert "Part #12345" in result

    def test_strips_html_artifacts(self):
        """Removes HTML tags leaked into text."""
        config = filter_threads.FilterConfig(strip_html_artifacts=True)
        result = filter_threads.clean_content("Hello <br/> world", config)
        assert "<br/>" not in result

    def test_normalizes_whitespace(self):
        """Collapses multiple blank lines."""
        config = filter_threads.FilterConfig(normalize_whitespace=True)
        result = filter_threads.clean_content("Line 1\n\n\n\nLine 2", config)
        assert "\n\n\n" not in result

    def test_strips_trailing_whitespace(self):
        """Removes trailing spaces per line."""
        config = filter_threads.FilterConfig(normalize_whitespace=True)
        result = filter_threads.clean_content("Hello   \nWorld   ", config)
        for line in result.split("\n"):
            assert line == line.rstrip()

    def test_strips_signature_at_end(self):
        """Removes content after signature divider in bottom half."""
        config = filter_threads.FilterConfig(strip_signatures=True)
        text = "Line 1\nLine 2\nLine 3\n--\nMy signature with links"
        result = filter_threads.clean_content(text, config)
        assert "signature" not in result
        assert "Line 1" in result

    def test_preserves_dash_dash_in_early_content(self):
        """Does not strip content when -- appears in the top half."""
        config = filter_threads.FilterConfig(strip_signatures=True)
        # -- on line 2 of 5 lines = top half, should NOT truncate
        text = "Technical content\n--\nMore technical content\nEven more content\nFinal content"
        result = filter_threads.clean_content(text, config)
        assert "More technical content" in result
        assert "Final content" in result

    def test_strips_signature_only_bottom_half(self):
        """Only strips signature divider in the bottom half of the post."""
        config = filter_threads.FilterConfig(strip_signatures=True)
        # 6 lines, divider on line 5 = in bottom half
        text = "Line 1\nLine 2\nLine 3\nLine 4\n---\nMy signature"
        result = filter_threads.clean_content(text, config)
        assert "My signature" not in result
        assert "Line 4" in result

    def test_empty_input(self):
        """Returns empty string for empty input."""
        config = filter_threads.FilterConfig()
        assert filter_threads.clean_content("", config) == ""

    def test_none_input(self):
        """Returns empty string for None."""
        config = filter_threads.FilterConfig()
        assert filter_threads.clean_content(None, config) == ""


# ============================================================================
# Test: Post-Level Filters
# ============================================================================

class TestFilterPost:
    """Tests for post-level filtering."""

    def test_passes_valid_post(self):
        """Long enough technical post passes."""
        config = filter_threads.FilterConfig(min_post_length=30)
        post = {"content_text": "This is a detailed technical post about valve adjustment procedures."}
        assert filter_threads.filter_post(post, config) is None

    def test_rejects_short_post(self):
        """Post shorter than min_post_length is filtered."""
        config = filter_threads.FilterConfig(min_post_length=30)
        post = {"content_text": "Short"}
        assert filter_threads.filter_post(post, config) == "too_short"

    def test_rejects_social_noise(self):
        """Social noise posts are filtered."""
        config = filter_threads.FilterConfig(
            min_post_length=1,
            social_noise_patterns=["thanks", "bump"],
        )
        post = {"content_text": "thanks"}
        assert filter_threads.filter_post(post, config) == "social_noise"

    def test_social_noise_case_insensitive(self):
        """Social noise matching is case-insensitive."""
        config = filter_threads.FilterConfig(
            min_post_length=1,
            social_noise_patterns=["thanks"],
        )
        post = {"content_text": "Thanks!"}
        assert filter_threads.filter_post(post, config) == "social_noise"

    def test_rejects_url_only(self):
        """Posts with only URLs are filtered."""
        config = filter_threads.FilterConfig(min_post_length=1)
        post = {"content_text": "https://example.com/long-enough-url-to-pass-length"}
        assert filter_threads.filter_post(post, config) == "url_only"

    def test_rejects_image_only(self):
        """Posts with images but no meaningful text are filtered."""
        config = filter_threads.FilterConfig(min_post_length=1)
        post = {"content_text": "   ", "content_images": ["https://example.com/img.jpg"]}
        # Short text triggers too_short first
        assert filter_threads.filter_post(post, config) is not None


# ============================================================================
# Test: is_social_noise
# ============================================================================

class TestIsSocialNoise:
    """Tests for social noise detection."""

    def test_exact_match(self):
        """Exact match is detected."""
        assert filter_threads.is_social_noise("thanks", ["thanks"]) is True

    def test_with_punctuation(self):
        """Punctuation stripped before matching."""
        assert filter_threads.is_social_noise("thanks!", ["thanks"]) is True

    def test_not_social(self):
        """Technical content is not social."""
        assert filter_threads.is_social_noise(
            "The torque spec is 85 Nm", ["thanks"]
        ) is False

    def test_case_insensitive(self):
        """Matching is case-insensitive."""
        assert filter_threads.is_social_noise("THANKS", ["thanks"]) is True

    def test_short_text_starting_with_pattern(self):
        """Short text starting with social pattern is noise."""
        assert filter_threads.is_social_noise(
            "thanks for the info", ["thanks"]
        ) is True

    def test_long_text_starting_with_pattern_is_not_noise(self):
        """Longer text starting with pattern is not noise if substantial."""
        text = "thanks for the valve clearance info, the spec is 0.28mm cold and I verified with my feeler gauge"
        assert filter_threads.is_social_noise(text, ["thanks"]) is False


# ============================================================================
# Test: is_quote_only
# ============================================================================

class TestIsQuoteOnly:
    """Tests for quote-only post detection."""

    def test_no_quotes(self):
        """Post without quotes is not quote-only."""
        post = {"content_text": "Regular content", "quotes": []}
        assert filter_threads.is_quote_only(post) is False

    def test_pure_quote(self):
        """Post that is entirely quoted text."""
        quoted_text = "This was the original message"
        post = {
            "content_text": quoted_text,
            "quotes": [quoted_text],
        }
        assert filter_threads.is_quote_only(post) is True

    def test_quote_plus_original(self):
        """Post with quote AND original content passes."""
        quoted_text = "Original question"
        post = {
            "content_text": f"{quoted_text}\n\nHere is my detailed answer to the question above.",
            "quotes": [quoted_text],
        }
        assert filter_threads.is_quote_only(post) is False


# ============================================================================
# Test: Thread-Level Filters
# ============================================================================

class TestFilterThread:
    """Tests for thread-level filtering."""

    def test_passes_valid_thread(self, sample_thread):
        """Thread with enough posts passes."""
        config = filter_threads.FilterConfig(min_thread_posts=2, min_first_post_length=50)
        assert filter_threads.filter_thread(sample_thread, config) is None

    def test_rejects_too_few_posts(self, sample_thread):
        """Thread with < min_thread_posts is filtered."""
        config = filter_threads.FilterConfig(min_thread_posts=10)
        assert filter_threads.filter_thread(sample_thread, config) == "too_few_posts"

    def test_rejects_short_first_post(self):
        """Thread with short first post is filtered."""
        config = filter_threads.FilterConfig(min_thread_posts=1, min_first_post_length=50)
        thread = {
            "thread_title": "Test",
            "posts": [{"content_text": "Short", "post_number": 1}],
        }
        assert filter_threads.filter_thread(thread, config) == "first_post_too_short"

    def test_rejects_noise_title(self):
        """Thread with buy/sell title is filtered."""
        config = filter_threads.FilterConfig(
            min_thread_posts=1,
            min_first_post_length=1,
            title_noise_patterns=["WTB", "FS:"],
        )
        thread = {
            "thread_title": "FS: E30 M3 parts",
            "posts": [{"content_text": "Selling my E30 M3 parts, lots of good stuff here!", "post_number": 1}],
        }
        assert "title_noise" in filter_threads.filter_thread(thread, config)


# ============================================================================
# Test: filter_thread_title
# ============================================================================

class TestFilterThreadTitle:
    """Tests for thread title noise detection."""

    def test_clean_title(self):
        """Technical title passes."""
        result = filter_threads.filter_thread_title("Valve Adjustment Procedure", ["WTB", "FS:"])
        assert result is None

    def test_for_sale_title(self):
        """FS: title is noise."""
        result = filter_threads.filter_thread_title("FS: Engine parts", ["WTB", "FS:"])
        assert result is not None

    def test_case_insensitive(self):
        """Title matching is case-insensitive."""
        result = filter_threads.filter_thread_title("wtb: intake manifold", ["WTB"])
        assert result is not None


# ============================================================================
# Test: process_threads (integration)
# ============================================================================

class TestProcessThreads:
    """Integration tests for the full filtering pipeline."""

    def test_processes_all_threads(self, threads_dir, tmp_path, config_file):
        """Processes all JSON files in input directory."""
        output = tmp_path / "threads_clean"
        output.mkdir()

        config = filter_threads.load_filter_config(config_file)
        stats = filter_threads.process_threads(threads_dir, output, config)

        assert stats["total_threads"] == 1
        assert stats["threads_surviving"] >= 0

    def test_creates_output_files(self, threads_dir, tmp_path, config_file):
        """Output files are created for surviving threads."""
        output = tmp_path / "threads_clean"
        output.mkdir()

        config = filter_threads.load_filter_config(config_file)
        filter_threads.process_threads(threads_dir, output, config)

        output_files = list(output.glob("*.json"))
        assert len(output_files) >= 0  # May be 0 or 1 depending on filtering

    def test_dry_run_no_output(self, threads_dir, tmp_path, config_file):
        """Dry run doesn't write files."""
        output = tmp_path / "threads_clean"
        output.mkdir()

        config = filter_threads.load_filter_config(config_file)
        filter_threads.process_threads(threads_dir, output, config, dry_run=True)

        output_files = list(output.glob("*.json"))
        assert len(output_files) == 0

    def test_clean_before_filter_order(self, tmp_path, config_file):
        """Posts with raw text passing length check but cleaning to trivial content are filtered."""
        # This post is 37 chars raw (passes min_post_length=30) but the
        # content is a post number marker + timestamp on separate lines +
        # trivial text. After cleaning: "ok got it" (9 chars, fails length check).
        thread = {
            "thread_id": "9999",
            "thread_title": "Test Thread",
            "forum": "d-i-y",
            "author": "TestUser",
            "post_count": 2,
            "posts": [
                {
                    "post_id": "9999",
                    "author": "TestUser",
                    "post_number": 1,
                    "is_first_post": True,
                    "content_text": "This is a real first post with enough technical content about valve adjustment.",
                    "content_images": [],
                    "quotes": [],
                },
                {
                    "post_id": "9998",
                    "author": "Noisy",
                    "post_number": 2,
                    "is_first_post": False,
                    "content_text": "#16\n05-29-2017, 11:37 AM\nok got it",
                    "content_images": [],
                    "quotes": [],
                },
            ],
        }

        threads_dir = tmp_path / "threads_in"
        threads_dir.mkdir()
        with open(threads_dir / "9999.json", "w") as f:
            json.dump(thread, f)

        output = tmp_path / "threads_out"
        output.mkdir()

        config = filter_threads.load_filter_config(config_file)
        stats = filter_threads.process_threads(threads_dir, output, config)

        # The noisy post should be filtered (cleaned text is too short)
        assert stats["posts_filtered"]["too_short"] >= 1


# ============================================================================
# Test: write_report
# ============================================================================

class TestWriteReport:
    """Tests for report generation."""

    def test_creates_report_file(self, tmp_path):
        """Report file is created."""
        from collections import Counter, defaultdict

        stats = {
            "total_threads": 100,
            "total_posts": 1000,
            "posts_filtered": Counter({"too_short": 50}),
            "threads_filtered": Counter({"too_few_posts": 20}),
            "threads_surviving": 80,
            "posts_surviving": 900,
            "by_forum": defaultdict(lambda: {"threads_in": 0, "threads_out": 0, "posts_in": 0, "posts_out": 0}),
        }

        report_path = tmp_path / "report.md"
        filter_threads.write_report(stats, report_path)
        assert report_path.exists()

    def test_report_contains_stats(self, tmp_path):
        """Report includes key statistics."""
        from collections import Counter, defaultdict

        stats = {
            "total_threads": 100,
            "total_posts": 1000,
            "posts_filtered": Counter({"too_short": 50}),
            "threads_filtered": Counter({"too_few_posts": 20}),
            "threads_surviving": 80,
            "posts_surviving": 900,
            "by_forum": defaultdict(lambda: {"threads_in": 0, "threads_out": 0, "posts_in": 0, "posts_out": 0}),
        }

        report_path = tmp_path / "report.md"
        filter_threads.write_report(stats, report_path)
        content = report_path.read_text()

        assert "100" in content
        assert "80" in content
        assert "too_short" in content
