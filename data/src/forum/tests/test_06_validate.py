"""
Tests for 06_validate.py - Cross-Reference & Filter

Tests quality filtering and numeric extraction.
"""

import json
import pytest
from pathlib import Path

from pipeline import validate


# ============================================================================
# Test: extract_numeric_claims
# ============================================================================

class TestExtractNumericClaims:
    """Tests for numeric value extraction."""

    def test_extracts_nm(self):
        """Extracts Newton-meters."""
        claims = validate.extract_numeric_claims("Torque to 85 Nm")
        assert (85.0, "nm") in claims

    def test_extracts_mm(self):
        """Extracts millimeters."""
        claims = validate.extract_numeric_claims("Clearance is 0.28mm")
        assert (0.28, "mm") in claims

    def test_extracts_bar(self):
        """Extracts pressure in bar."""
        claims = validate.extract_numeric_claims("Pressure: 3.5 bar")
        assert (3.5, "bar") in claims

    def test_extracts_celsius(self):
        """Extracts temperature."""
        claims = validate.extract_numeric_claims("Operating temp 90°C")
        assert (90.0, "°c") in claims

    def test_extracts_multiple(self):
        """Extracts multiple values."""
        claims = validate.extract_numeric_claims("Gap 0.28mm to 0.33mm at 85 Nm")
        assert len(claims) >= 3

    def test_no_units(self):
        """Returns empty for text without units."""
        claims = validate.extract_numeric_claims("Just some regular text")
        assert len(claims) == 0

    def test_empty_text(self):
        """Returns empty for empty string."""
        claims = validate.extract_numeric_claims("")
        assert len(claims) == 0


# ============================================================================
# Test: check_numeric_claims
# ============================================================================

class TestCheckNumericClaims:
    """Tests for numeric cross-reference against manual data."""

    def test_verified_when_match(self):
        """Returns 'verified' when forum claim matches manual."""
        manual = [{"answer": "Torque to 85 Nm"}]
        result = validate.check_numeric_claims("The torque spec is 85 Nm", manual)
        assert result == "verified"

    def test_unverified_when_no_match(self):
        """Returns 'unverified' when forum claim doesn't match manual."""
        manual = [{"answer": "Torque to 85 Nm"}]
        result = validate.check_numeric_claims("The torque spec is 95 Nm", manual)
        assert result == "unverified"

    def test_plausible_when_no_claims(self):
        """Returns 'plausible' when forum has no numeric claims."""
        manual = [{"answer": "Torque to 85 Nm"}]
        result = validate.check_numeric_claims("Remove the valve cover carefully", manual)
        assert result == "plausible"

    def test_unverified_when_no_manual(self):
        """Returns 'unverified' when no manual data available."""
        result = validate.check_numeric_claims("Clearance is 0.28mm", [])
        assert result == "unverified"

    def test_matches_across_multiple_manual_pairs(self):
        """Can verify against claims spread across multiple manual pairs."""
        manual = [
            {"answer": "Torque to 85 Nm"},
            {"answer": "Clearance is 0.28mm"},
        ]
        result = validate.check_numeric_claims("Set clearance to 0.28mm", manual)
        assert result == "verified"


# ============================================================================
# Test: filter_qa_pair
# ============================================================================

class TestFilterQAPair:
    """Tests for individual Q&A pair filtering."""

    def test_passes_valid_pair(self, forum_config):
        """Good Q&A pair passes."""
        pair = {
            "question": "What is the valve clearance spec for the S14?",
            "answer": "The BMW S14 valve shim clearance specification is 0.28mm to 0.33mm, measured cold.",
        }
        assert validate.filter_qa_pair(pair, forum_config) is None

    def test_rejects_short_answer(self, forum_config):
        """Answer below min_answer_length fails."""
        pair = {
            "question": "What is the valve clearance spec?",
            "answer": "0.28mm",
        }
        assert validate.filter_qa_pair(pair, forum_config) == "answer_too_short"

    def test_rejects_long_answer(self, forum_config):
        """Answer above max_answer_length fails."""
        pair = {
            "question": "What is the valve clearance spec?",
            "answer": "x" * 2001,
        }
        assert validate.filter_qa_pair(pair, forum_config) == "answer_too_long"

    def test_rejects_short_question(self, forum_config):
        """Question below min_question_length fails."""
        pair = {
            "question": "Clearance?",
            "answer": "The BMW S14 valve shim clearance specification is 0.28mm to 0.33mm, measured cold.",
        }
        assert validate.filter_qa_pair(pair, forum_config) == "question_too_short"

    def test_rejects_generic_answer(self, forum_config):
        """Answer with generic pattern fails."""
        pair = {
            "question": "What is the valve clearance spec?",
            "answer": "It depends on the specific engine variant and condition, so check the manual for details.",
        }
        result = validate.filter_qa_pair(pair, forum_config)
        assert result is not None
        assert "generic_answer" in result

    def test_rejects_vague_question(self, forum_config):
        """Question with vague pattern fails."""
        pair = {
            "question": "What do you think about valve clearances in general for older BMW engines?",
            "answer": "The BMW S14 valve shim clearance specification is 0.28mm to 0.33mm, measured cold.",
        }
        result = validate.filter_qa_pair(pair, forum_config)
        assert result is not None
        assert "vague_question" in result


# ============================================================================
# Test: load_manual_qa
# ============================================================================

class TestLoadManualQA:
    """Tests for loading manual Q&A pairs."""

    def test_loads_valid_jsonl(self, tmp_path):
        """Loads Q&A pairs from JSONL file."""
        manual_path = tmp_path / "manual_train.jsonl"
        record = {
            "conversations": [
                {"role": "user", "content": "What is the torque?"},
                {"role": "assistant", "content": "85 Nm"},
            ],
            "metadata": {"source": "manual"},
        }
        manual_path.write_text(json.dumps(record) + "\n")

        pairs = validate.load_manual_qa(manual_path)
        assert len(pairs) == 1
        assert pairs[0]["question"] == "What is the torque?"

    def test_returns_empty_for_missing_file(self, tmp_path):
        """Returns empty list for missing file."""
        pairs = validate.load_manual_qa(tmp_path / "missing.jsonl")
        assert pairs == []

    def test_handles_multiple_conversations(self, tmp_path):
        """Handles records with multiple Q&A turns."""
        manual_path = tmp_path / "manual_train.jsonl"
        record = {
            "conversations": [
                {"role": "user", "content": "Q1?"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2?"},
                {"role": "assistant", "content": "A2"},
            ],
            "metadata": {},
        }
        manual_path.write_text(json.dumps(record) + "\n")

        pairs = validate.load_manual_qa(manual_path)
        assert len(pairs) == 2


# ============================================================================
# Test: end-to-end validation pipeline
# ============================================================================

class TestValidationPipeline:
    """Integration tests for the validation pipeline."""

    def test_validates_qa_files(self, qa_raw_dir, tmp_path, config_file):
        """Processes Q&A files through validation."""
        output = tmp_path / "qa_validated"
        output.mkdir()
        report = tmp_path / "crossref_report.md"

        # Run via main() would need full CLI; test the pieces instead
        config = validate.load_config(config_file)

        qa_files = sorted(qa_raw_dir.glob("*.json"))
        assert len(qa_files) == 1

        # Load Q&A
        all_pairs = []
        for qa_file in qa_files:
            with open(qa_file) as f:
                doc = json.load(f)
            for idx, pair in enumerate(doc.get("qa_pairs", [])):
                all_pairs.append((0, idx, pair, doc))

        # Quality filter
        passing = []
        for i, (_, _, pair, _) in enumerate(all_pairs):
            reason = validate.filter_qa_pair(pair, config)
            if reason is None:
                passing.append(i)

        # Both pairs from sample_qa_doc should pass
        assert len(passing) == 2
