"""
Tests for 04_score.py - Quality Scoring

Tests scoring logic and composite calculation (no real API calls).
"""

import json
import pytest
from unittest.mock import MagicMock

from pipeline import score


# ============================================================================
# Test: build_thread_content
# ============================================================================

class TestBuildThreadContent:
    """Tests for thread content extraction."""

    def test_includes_title(self, sample_classified_thread):
        """Content includes thread title."""
        content = score.build_thread_content(sample_classified_thread, 4000)
        assert "Valve Adjustment" in content

    def test_includes_all_posts(self, sample_classified_thread):
        """Content includes all post authors."""
        content = score.build_thread_content(sample_classified_thread, 10000)
        assert "Split_S" in content
        assert "BimmerDude" in content

    def test_truncates(self, sample_classified_thread):
        """Content truncated at max_chars."""
        content = score.build_thread_content(sample_classified_thread, 100)
        assert len(content) <= 115  # Allow for truncation marker

    def test_includes_classification(self, sample_classified_thread):
        """Content includes classification type."""
        content = score.build_thread_content(sample_classified_thread, 4000)
        assert "how_to" in content


# ============================================================================
# Test: score_thread (mocked API)
# ============================================================================

class TestScoreThread:
    """Tests for thread scoring with mocked API."""

    def test_successful_scoring(self, sample_classified_thread, forum_config):
        """Returns scores with composite on success."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "factual_specificity": 4,
            "answer_completeness": 4,
            "procedural_clarity": 5,
            "s14_relevance": 5,
            "unique_knowledge": 3,
        }))]
        mock_client.messages.create.return_value = mock_response

        result = score.score_thread(sample_classified_thread, forum_config, mock_client)
        assert result is not None
        assert "composite" in result
        assert 1.0 <= result["composite"] <= 5.0

    def test_composite_calculation(self, sample_classified_thread, forum_config):
        """Composite is weighted average of dimensions."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        # All 5s should give composite of 5.0
        mock_response.content = [MagicMock(text=json.dumps({
            "factual_specificity": 5,
            "answer_completeness": 5,
            "procedural_clarity": 5,
            "s14_relevance": 5,
            "unique_knowledge": 5,
        }))]
        mock_client.messages.create.return_value = mock_response

        result = score.score_thread(sample_classified_thread, forum_config, mock_client)
        assert result["composite"] == 5.0

    def test_composite_all_ones(self, sample_classified_thread, forum_config):
        """All 1s gives composite of 1.0."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "factual_specificity": 1,
            "answer_completeness": 1,
            "procedural_clarity": 1,
            "s14_relevance": 1,
            "unique_knowledge": 1,
        }))]
        mock_client.messages.create.return_value = mock_response

        result = score.score_thread(sample_classified_thread, forum_config, mock_client)
        assert result["composite"] == 1.0

    def test_returns_none_on_failure(self, sample_classified_thread, forum_config):
        """Returns None when API fails."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        result = score.score_thread(sample_classified_thread, forum_config, mock_client)
        assert result is None

    def test_handles_markdown_json(self, sample_classified_thread, forum_config):
        """Handles JSON in markdown code blocks."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='```json\n{"factual_specificity": 4, "answer_completeness": 3, "procedural_clarity": 4, "s14_relevance": 5, "unique_knowledge": 3}\n```')]
        mock_client.messages.create.return_value = mock_response

        result = score.score_thread(sample_classified_thread, forum_config, mock_client)
        assert result is not None
        assert "composite" in result
