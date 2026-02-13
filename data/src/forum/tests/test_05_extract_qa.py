"""
Tests for 05_extract_qa.py - Q&A Extraction

Tests extraction logic and output schema (no real API calls).
"""

import json
import pytest
from unittest.mock import MagicMock

from pipeline import extract_qa


# ============================================================================
# Test: build_full_thread
# ============================================================================

class TestBuildFullThread:
    """Tests for building full thread content."""

    def test_includes_all_posts(self, sample_scored_thread):
        """Content includes all posts."""
        content, posts_used = extract_qa.build_full_thread(sample_scored_thread, 10000)
        assert "Split_S" in content
        assert "BimmerDude" in content
        assert posts_used == 3

    def test_includes_title_and_forum(self, sample_scored_thread):
        """Content includes metadata."""
        content, _ = extract_qa.build_full_thread(sample_scored_thread, 10000)
        assert "Valve Adjustment" in content
        assert "d-i-y" in content

    def test_truncation(self, sample_scored_thread):
        """Content is truncated."""
        content, _ = extract_qa.build_full_thread(sample_scored_thread, 100)
        assert len(content) <= 115


# ============================================================================
# Test: extract_qa (mocked API)
# ============================================================================

class TestExtractQA:
    """Tests for Q&A extraction with mocked API."""

    def _make_mock_client(self, qa_pairs):
        """Helper to create mock client returning given Q&A pairs."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({"qa_pairs": qa_pairs}))]
        mock_client.messages.create.return_value = mock_response
        return mock_client

    def test_successful_extraction(self, sample_scored_thread, forum_config):
        """Returns Q&A document on success."""
        qa_pairs = [
            {
                "question": "What is the valve shim clearance for the S14?",
                "answer": "0.28-0.33mm measured cold.",
                "question_type": "specification",
                "cross_reference_needed": True,
                "cross_reference_note": "Verify against Section 11",
            }
        ]
        mock_client = self._make_mock_client(qa_pairs)

        result = extract_qa.extract_qa(sample_scored_thread, forum_config, mock_client)
        assert result is not None
        assert result["source_type"] == "forum"
        assert len(result["qa_pairs"]) == 1

    def test_assigns_ids(self, sample_scored_thread, forum_config):
        """Q&A pairs get unique IDs."""
        qa_pairs = [
            {"question": "Q1?", "answer": "A1", "question_type": "specification",
             "cross_reference_needed": False, "cross_reference_note": None},
            {"question": "Q2?", "answer": "A2", "question_type": "procedural",
             "cross_reference_needed": False, "cross_reference_note": None},
        ]
        mock_client = self._make_mock_client(qa_pairs)

        result = extract_qa.extract_qa(sample_scored_thread, forum_config, mock_client)
        ids = [p["id"] for p in result["qa_pairs"]]
        assert ids == ["forum-1258841-q01", "forum-1258841-q02"]

    def test_output_schema(self, sample_scored_thread, forum_config):
        """Output document has all required fields."""
        qa_pairs = [
            {"question": "Q?", "answer": "A", "question_type": "specification",
             "cross_reference_needed": False, "cross_reference_note": None},
        ]
        mock_client = self._make_mock_client(qa_pairs)

        result = extract_qa.extract_qa(sample_scored_thread, forum_config, mock_client)

        required_keys = [
            "thread_id", "forum", "thread_title", "source_type",
            "content_type", "quality_score", "qa_pairs", "extraction",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

        assert result["thread_id"] == "1258841"
        assert result["forum"] == "d-i-y"
        assert result["source_type"] == "forum"

    def test_extraction_metadata(self, sample_scored_thread, forum_config):
        """Extraction metadata is populated."""
        qa_pairs = [{"question": "Q?", "answer": "A", "question_type": "specification",
                     "cross_reference_needed": False, "cross_reference_note": None}]
        mock_client = self._make_mock_client(qa_pairs)

        result = extract_qa.extract_qa(sample_scored_thread, forum_config, mock_client)
        ext = result["extraction"]
        assert "model" in ext
        assert "timestamp" in ext
        assert ext["thread_posts_total"] == 3

    def test_returns_none_on_failure(self, sample_scored_thread, forum_config):
        """Returns None when API fails."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")

        result = extract_qa.extract_qa(sample_scored_thread, forum_config, mock_client)
        assert result is None

    def test_handles_empty_qa(self, sample_scored_thread, forum_config):
        """Handles API returning empty qa_pairs."""
        mock_client = self._make_mock_client([])
        result = extract_qa.extract_qa(sample_scored_thread, forum_config, mock_client)
        assert result is not None
        assert len(result["qa_pairs"]) == 0
