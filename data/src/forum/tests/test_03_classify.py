"""
Tests for 03_classify.py - Thread Classification

Tests classification logic and gate checks (no real API calls).
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pipeline import classify


# ============================================================================
# Test: passes_gate
# ============================================================================

class TestPassesGate:
    """Tests for the classification quality gate."""

    def test_high_depth_s14_passes(self, forum_config):
        """high depth + s14_specific passes."""
        classification = {
            "thread_type": "how_to",
            "technical_depth": "high",
            "s14_specific": True,
            "has_resolution": True,
            "confidence": 0.9,
        }
        assert classify.passes_gate(classification, forum_config) is True

    def test_medium_depth_s14_passes(self, forum_config):
        """medium depth + s14_specific passes."""
        classification = {
            "technical_depth": "medium",
            "s14_specific": True,
        }
        assert classify.passes_gate(classification, forum_config) is True

    def test_low_depth_fails(self, forum_config):
        """low depth fails regardless of s14_specific."""
        classification = {
            "technical_depth": "low",
            "s14_specific": True,
        }
        assert classify.passes_gate(classification, forum_config) is False

    def test_not_s14_fails(self, forum_config):
        """Not s14_specific fails regardless of depth."""
        classification = {
            "technical_depth": "high",
            "s14_specific": False,
        }
        assert classify.passes_gate(classification, forum_config) is False

    def test_low_and_not_s14_fails(self, forum_config):
        """Both low and not s14 fails."""
        classification = {
            "technical_depth": "low",
            "s14_specific": False,
        }
        assert classify.passes_gate(classification, forum_config) is False

    def test_missing_fields_fail(self, forum_config):
        """Missing fields treated as failing defaults."""
        assert classify.passes_gate({}, forum_config) is False


# ============================================================================
# Test: build_thread_preview
# ============================================================================

class TestBuildThreadPreview:
    """Tests for thread preview construction."""

    def test_includes_title(self, sample_thread):
        """Preview includes thread title."""
        preview = classify.build_thread_preview(sample_thread, 2000, 3)
        assert "Valve Adjustment" in preview

    def test_includes_first_post(self, sample_thread):
        """Preview includes first post content."""
        preview = classify.build_thread_preview(sample_thread, 2000, 3)
        assert "valve shim" in preview.lower()

    def test_truncates_to_max_chars(self, sample_thread):
        """Preview is truncated at max_chars."""
        preview = classify.build_thread_preview(sample_thread, 100, 3)
        assert len(preview) <= 120  # Allow for truncation marker

    def test_limits_replies(self, sample_thread):
        """Only includes max_replies replies."""
        preview = classify.build_thread_preview(sample_thread, 5000, 1)
        # Should have title + first post + 1 reply (not all)
        assert "Split_S" in preview  # First post author

    def test_empty_thread(self):
        """Handles thread with no posts."""
        thread = {"thread_title": "Empty", "posts": []}
        preview = classify.build_thread_preview(thread, 2000, 3)
        assert "Empty" in preview


# ============================================================================
# Test: classify_thread (mocked API)
# ============================================================================

class TestClassifyThread:
    """Tests for thread classification with mocked API."""

    def test_successful_classification(self, sample_thread, forum_config):
        """Returns classification dict on success."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "thread_type": "how_to",
            "technical_depth": "high",
            "s14_specific": True,
            "has_resolution": True,
            "confidence": 0.92,
        }))]
        mock_client.messages.create.return_value = mock_response

        result = classify.classify_thread(sample_thread, forum_config, mock_client)
        assert result is not None
        assert result["thread_type"] == "how_to"
        assert result["technical_depth"] == "high"

    def test_handles_markdown_wrapped_json(self, sample_thread, forum_config):
        """Handles JSON wrapped in markdown code blocks."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"thread_type": "how_to", "technical_depth": "high", "s14_specific": true, "has_resolution": true, "confidence": 0.9}\n```')]
        mock_client.messages.create.return_value = mock_response

        result = classify.classify_thread(sample_thread, forum_config, mock_client)
        assert result is not None
        assert result["thread_type"] == "how_to"

    def test_returns_none_on_api_failure(self, sample_thread, forum_config):
        """Returns None when all retries exhausted."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        result = classify.classify_thread(sample_thread, forum_config, mock_client)
        assert result is None

    def test_returns_none_on_invalid_json(self, sample_thread, forum_config):
        """Returns None when API returns invalid JSON."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="not valid json at all")]
        mock_client.messages.create.return_value = mock_response

        result = classify.classify_thread(sample_thread, forum_config, mock_client)
        assert result is None
