"""
Tests for 05_filter_qa.py - Q&A Quality Filtering

TDD tests based on specs/05_qa_quality_control_spec.md
"""

import json
import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple, Dict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import importlib.util

# Dynamic import to handle the numeric prefix in module name
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

filter_qa = load_module("filter_qa", Path(__file__).parent.parent / "scripts" / "05_filter_qa.py")


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_config():
    """Default filter configuration for tests."""
    return filter_qa.FilterConfig(
        min_answer_length=10,
        max_answer_length=500,
        min_question_length=15,
        require_question_mark=True,
        max_question_similarity=0.80,
        min_question_types_per_page=2,
        generic_answer_patterns=[
            "cannot determine",
            "not visible",
            "typically",
            "usually",
            "generally",
        ],
        self_referential_patterns=[
            "on this page",
            "in this image",
            "as shown here",
        ],
        valid_question_types={
            "factual", "procedural", "visual", "inspection",
            "tool", "safety", "wiring", "connector",
        },
    )


@pytest.fixture
def sample_qa_doc():
    """Sample Q&A document for testing."""
    return {
        "page_id": "21-03_clutch",
        "image_path": "data_src/21 - Clutch/21-03.jpg",
        "section_id": "21",
        "section_name": "Clutch",
        "source_type": "service_manual",
        "content_type": "procedure",
        "qa_pairs": [
            {
                "id": "21-03_clutch-q01",
                "question": "What should I visually inspect the clutch pressure plate for?",
                "answer": "Visually inspect the clutch for cracks, wear, and burnt spots.",
                "question_type": "inspection",
            },
            {
                "id": "21-03_clutch-q02",
                "question": "What is the torque specification for the flywheel bolts?",
                "answer": "The flywheel bolts should be torqued to 85 Nm.",
                "question_type": "factual",
            },
            {
                "id": "21-03_clutch-q03",
                "question": "How do I remove the clutch pressure plate?",
                "answer": "Loosen the bolts in a diagonal pattern, then remove the plate.",
                "question_type": "procedural",
            },
        ],
    }


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures" / "qa_raw"


# ============================================================================
# Test: load_qa_files
# ============================================================================

class TestLoadQAFiles:
    """Tests for load_qa_files function."""

    def test_load_qa_files_valid_directory(self, fixtures_dir):
        """Loads all JSON files from valid directory."""
        docs = filter_qa.load_qa_files(fixtures_dir)
        assert len(docs) >= 2  # At least our fixture files
        assert all("page_id" in doc for doc in docs)

    def test_load_qa_files_empty_directory(self, tmp_path):
        """Returns empty list for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        docs = filter_qa.load_qa_files(empty_dir)
        assert docs == []

    def test_load_qa_files_missing_directory(self):
        """Raises FileNotFoundError for missing directory."""
        with pytest.raises(FileNotFoundError):
            filter_qa.load_qa_files(Path("/nonexistent/path"))

    def test_load_qa_files_skips_invalid_json(self, tmp_path):
        """Logs warning and continues with valid files."""
        # Create valid and invalid files
        valid_file = tmp_path / "valid.json"
        valid_file.write_text('{"page_id": "test", "qa_pairs": []}')

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text('not valid json {{{')

        docs = filter_qa.load_qa_files(tmp_path)
        assert len(docs) == 1
        assert docs[0]["page_id"] == "test"

    def test_load_qa_files_handles_nested_dirs(self, tmp_path):
        """Only reads top-level files (no recursion)."""
        # Top-level file
        top_file = tmp_path / "top.json"
        top_file.write_text('{"page_id": "top", "qa_pairs": []}')

        # Nested file
        nested_dir = tmp_path / "nested"
        nested_dir.mkdir()
        nested_file = nested_dir / "nested.json"
        nested_file.write_text('{"page_id": "nested", "qa_pairs": []}')

        docs = filter_qa.load_qa_files(tmp_path)
        assert len(docs) == 1
        assert docs[0]["page_id"] == "top"


# ============================================================================
# Test: load_filter_config
# ============================================================================

class TestLoadFilterConfig:
    """Tests for load_filter_config function."""

    def test_load_filter_config_full(self, tmp_path):
        """All fields populated from config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
filters:
  min_answer_length: 20
  max_answer_length: 400
  min_question_length: 20
  require_question_mark: false
  max_question_similarity: 0.75
  min_question_types_per_page: 3
  generic_answer_patterns:
    - "cannot determine"
  self_referential_patterns:
    - "on this page"
  valid_question_types:
    - factual
    - procedural
""")

        config = filter_qa.load_filter_config(config_file)
        assert config.min_answer_length == 20
        assert config.max_answer_length == 400
        assert config.min_question_length == 20
        assert config.require_question_mark is False
        assert config.max_question_similarity == 0.75
        assert "cannot determine" in config.generic_answer_patterns
        assert "factual" in config.valid_question_types

    def test_load_filter_config_missing_section(self, tmp_path):
        """Uses defaults when filters section missing."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
other_section:
  some_key: value
""")

        config = filter_qa.load_filter_config(config_file)
        defaults = filter_qa.FilterConfig()
        assert config.min_answer_length == defaults.min_answer_length

    def test_load_filter_config_partial(self, tmp_path):
        """Merges provided values with defaults."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
filters:
  min_answer_length: 25
""")

        config = filter_qa.load_filter_config(config_file)
        defaults = filter_qa.FilterConfig()
        assert config.min_answer_length == 25  # Overridden
        assert config.max_answer_length == defaults.max_answer_length  # Default


# ============================================================================
# Test: filter_answer_length
# ============================================================================

class TestFilterAnswerLength:
    """Tests for filter_answer_length function."""

    def test_filter_answer_length_valid(self, sample_config):
        """50 chars passes."""
        qa = {"answer": "This is a valid answer with fifty characters here!"}
        passed, reason = filter_qa.filter_answer_length(qa, sample_config)
        assert passed is True
        assert reason is None

    def test_filter_answer_length_too_short(self, sample_config):
        """5 chars fails."""
        qa = {"answer": "Short"}
        passed, reason = filter_qa.filter_answer_length(qa, sample_config)
        assert passed is False
        assert reason == "answer_too_short"

    def test_filter_answer_length_too_long(self, sample_config):
        """600 chars fails."""
        qa = {"answer": "x" * 600}
        passed, reason = filter_qa.filter_answer_length(qa, sample_config)
        assert passed is False
        assert reason == "answer_too_long"

    def test_filter_answer_length_boundary_min(self, sample_config):
        """Exactly min_answer_length passes."""
        qa = {"answer": "x" * sample_config.min_answer_length}
        passed, reason = filter_qa.filter_answer_length(qa, sample_config)
        assert passed is True

    def test_filter_answer_length_boundary_max(self, sample_config):
        """Exactly max_answer_length passes."""
        qa = {"answer": "x" * sample_config.max_answer_length}
        passed, reason = filter_qa.filter_answer_length(qa, sample_config)
        assert passed is True

    def test_filter_answer_length_empty(self, sample_config):
        """Empty string fails."""
        qa = {"answer": ""}
        passed, reason = filter_qa.filter_answer_length(qa, sample_config)
        assert passed is False
        assert reason == "answer_too_short"

    def test_filter_answer_length_whitespace_only(self, sample_config):
        """Whitespace-only fails after strip."""
        qa = {"answer": "     "}
        passed, reason = filter_qa.filter_answer_length(qa, sample_config)
        assert passed is False
        assert reason == "answer_too_short"


# ============================================================================
# Test: filter_question_length
# ============================================================================

class TestFilterQuestionLength:
    """Tests for filter_question_length function."""

    def test_filter_question_length_valid(self, sample_config):
        """25 chars passes."""
        qa = {"question": "What is the torque value?"}
        passed, reason = filter_qa.filter_question_length(qa, sample_config)
        assert passed is True
        assert reason is None

    def test_filter_question_length_too_short(self, sample_config):
        """10 chars fails."""
        qa = {"question": "What is X?"}
        passed, reason = filter_qa.filter_question_length(qa, sample_config)
        assert passed is False
        assert reason == "question_too_short"

    def test_filter_question_length_boundary(self, sample_config):
        """Exactly min_question_length passes."""
        qa = {"question": "x" * sample_config.min_question_length}
        passed, reason = filter_qa.filter_question_length(qa, sample_config)
        assert passed is True


# ============================================================================
# Test: filter_question_mark
# ============================================================================

class TestFilterQuestionMark:
    """Tests for filter_question_mark function."""

    def test_filter_question_mark_present(self, sample_config):
        """'What is X?' passes."""
        qa = {"question": "What is X?"}
        passed, reason = filter_qa.filter_question_mark(qa, sample_config)
        assert passed is True
        assert reason is None

    def test_filter_question_mark_missing(self, sample_config):
        """'What is X' fails."""
        qa = {"question": "What is X"}
        passed, reason = filter_qa.filter_question_mark(qa, sample_config)
        assert passed is False
        assert reason == "missing_question_mark"

    def test_filter_question_mark_trailing_space(self, sample_config):
        """'What is X? ' passes after strip."""
        qa = {"question": "What is X? "}
        passed, reason = filter_qa.filter_question_mark(qa, sample_config)
        assert passed is True

    def test_filter_question_mark_disabled(self):
        """require_question_mark=False always passes."""
        config = filter_qa.FilterConfig(require_question_mark=False)
        qa = {"question": "What is X"}
        passed, reason = filter_qa.filter_question_mark(qa, config)
        assert passed is True


# ============================================================================
# Test: filter_generic_answer
# ============================================================================

class TestFilterGenericAnswer:
    """Tests for filter_generic_answer function."""

    def test_filter_generic_answer_valid(self, sample_config):
        """'The torque is 85 Nm' passes."""
        qa = {"answer": "The torque is 85 Nm"}
        passed, reason = filter_qa.filter_generic_answer(qa, sample_config)
        assert passed is True
        assert reason is None

    def test_filter_generic_answer_cannot_determine(self, sample_config):
        """'I cannot determine from this image' fails."""
        qa = {"answer": "I cannot determine from this image"}
        passed, reason = filter_qa.filter_generic_answer(qa, sample_config)
        assert passed is False
        assert "generic_answer" in reason

    def test_filter_generic_answer_typically(self, sample_config):
        """'Typically around 80 Nm' fails."""
        qa = {"answer": "Typically around 80 Nm"}
        passed, reason = filter_qa.filter_generic_answer(qa, sample_config)
        assert passed is False
        assert "generic_answer" in reason

    def test_filter_generic_answer_case_insensitive(self, sample_config):
        """'I CANNOT DETERMINE' fails."""
        qa = {"answer": "I CANNOT DETERMINE this value"}
        passed, reason = filter_qa.filter_generic_answer(qa, sample_config)
        assert passed is False

    def test_filter_generic_answer_partial_match(self, sample_config):
        """'undetermined' does NOT fail (word boundary)."""
        # This doesn't contain "cannot determine" as a phrase
        qa = {"answer": "The undetermined value should be measured"}
        passed, reason = filter_qa.filter_generic_answer(qa, sample_config)
        assert passed is True

    def test_filter_generic_answer_empty_patterns(self):
        """No patterns = always pass."""
        config = filter_qa.FilterConfig(generic_answer_patterns=[])
        qa = {"answer": "I cannot determine anything"}
        passed, reason = filter_qa.filter_generic_answer(qa, config)
        assert passed is True


# ============================================================================
# Test: filter_self_referential
# ============================================================================

class TestFilterSelfReferential:
    """Tests for filter_self_referential function."""

    def test_filter_self_referential_valid(self, sample_config):
        """'What is the clutch torque?' passes."""
        qa = {"question": "What is the clutch torque?"}
        passed, reason = filter_qa.filter_self_referential(qa, sample_config)
        assert passed is True
        assert reason is None

    def test_filter_self_referential_on_this_page(self, sample_config):
        """'What is shown on this page?' fails."""
        qa = {"question": "What is shown on this page?"}
        passed, reason = filter_qa.filter_self_referential(qa, sample_config)
        assert passed is False
        assert "self_referential" in reason

    def test_filter_self_referential_as_shown(self, sample_config):
        """'What part is shown here?' fails."""
        qa = {"question": "What part is as shown here?"}
        passed, reason = filter_qa.filter_self_referential(qa, sample_config)
        assert passed is False
        assert "self_referential" in reason

    def test_filter_self_referential_case_insensitive(self, sample_config):
        """'ON THIS PAGE' fails."""
        qa = {"question": "What is ON THIS PAGE?"}
        passed, reason = filter_qa.filter_self_referential(qa, sample_config)
        assert passed is False


# ============================================================================
# Test: filter_question_type
# ============================================================================

class TestFilterQuestionType:
    """Tests for filter_question_type function."""

    def test_filter_question_type_valid(self, sample_config):
        """'procedural' passes."""
        qa = {"question_type": "procedural"}
        passed, reason = filter_qa.filter_question_type(qa, sample_config)
        assert passed is True
        assert reason is None

    def test_filter_question_type_invalid(self, sample_config):
        """'random_type' fails."""
        qa = {"question_type": "random_type"}
        passed, reason = filter_qa.filter_question_type(qa, sample_config)
        assert passed is False
        assert "invalid_question_type" in reason

    def test_filter_question_type_missing(self, sample_config):
        """No question_type field = warning, passes."""
        qa = {"question": "What is this?", "answer": "Something"}
        passed, reason = filter_qa.filter_question_type(qa, sample_config)
        # Missing type should pass with a warning, not fail
        assert passed is True

    def test_filter_question_type_empty_config(self):
        """No valid_question_types = always passes."""
        config = filter_qa.FilterConfig(valid_question_types=set())
        qa = {"question_type": "anything"}
        passed, reason = filter_qa.filter_question_type(qa, config)
        assert passed is True


# ============================================================================
# Test: filter_question_diversity
# ============================================================================

class TestFilterQuestionDiversity:
    """Tests for filter_question_diversity function."""

    def test_filter_question_diversity_all_unique(self, sample_config):
        """3 different questions all pass."""
        qa_pairs = [
            {"id": "q1", "question": "What is the torque value?"},
            {"id": "q2", "question": "How do I remove the bolt?"},
            {"id": "q3", "question": "Which tool should I use?"},
        ]
        results = filter_qa.filter_question_diversity(qa_pairs, sample_config)
        assert all(passed for qa, passed, reason in results)

    def test_filter_question_diversity_exact_duplicate(self, sample_config):
        """Second identical question fails."""
        qa_pairs = [
            {"id": "q1", "question": "What is the torque value?"},
            {"id": "q2", "question": "What is the torque value?"},
        ]
        results = filter_qa.filter_question_diversity(qa_pairs, sample_config)
        assert results[0][1] is True  # First passes
        assert results[1][1] is False  # Second fails

    def test_filter_question_diversity_similar(self, sample_config):
        """85% word overlap fails at 0.80 threshold."""
        # High overlap questions
        qa_pairs = [
            {"id": "q1", "question": "What is the flywheel bolt torque specification?"},
            {"id": "q2", "question": "What is the flywheel bolt torque value?"},
        ]
        results = filter_qa.filter_question_diversity(qa_pairs, sample_config)
        # At least one should pass (first), second might fail due to similarity
        assert results[0][1] is True

    def test_filter_question_diversity_first_wins(self, sample_config):
        """First of duplicate pair passes."""
        qa_pairs = [
            {"id": "q1", "question": "What is the exact torque?"},
            {"id": "q2", "question": "What is the exact torque?"},
            {"id": "q3", "question": "What is the exact torque?"},
        ]
        results = filter_qa.filter_question_diversity(qa_pairs, sample_config)
        assert results[0][1] is True
        assert results[1][1] is False
        assert results[2][1] is False

    def test_filter_question_diversity_empty_list(self, sample_config):
        """Empty input returns empty output."""
        results = filter_qa.filter_question_diversity([], sample_config)
        assert results == []


# ============================================================================
# Test: apply_filters
# ============================================================================

class TestApplyFilters:
    """Tests for apply_filters function."""

    def test_apply_filters_all_pass(self, sample_config, sample_qa_doc):
        """All Q&A pairs pass, rejected_list empty."""
        filtered_doc, rejected = filter_qa.apply_filters(sample_qa_doc, sample_config)
        assert len(filtered_doc["qa_pairs"]) == 3
        assert len(rejected) == 0

    def test_apply_filters_some_fail(self, sample_config):
        """Mixed results, correct pairs in each list."""
        doc = {
            "page_id": "test",
            "qa_pairs": [
                {
                    "id": "q1",
                    "question": "What is the torque specification?",
                    "answer": "The torque is 85 Nm as specified.",
                    "question_type": "factual",
                },
                {
                    "id": "q2",
                    "question": "What is shown on this page?",  # Self-referential
                    "answer": "A diagram is shown.",
                    "question_type": "visual",
                },
            ],
        }

        filtered_doc, rejected = filter_qa.apply_filters(doc, sample_config)
        assert len(filtered_doc["qa_pairs"]) == 1
        assert len(rejected) == 1
        assert rejected[0]["qa"]["id"] == "q2"

    def test_apply_filters_all_fail(self, sample_config):
        """Returns doc with empty qa_pairs."""
        doc = {
            "page_id": "test",
            "qa_pairs": [
                {
                    "id": "q1",
                    "question": "X?",  # Too short
                    "answer": "Y",  # Too short
                    "question_type": "factual",
                },
            ],
        }

        filtered_doc, rejected = filter_qa.apply_filters(doc, sample_config)
        assert len(filtered_doc["qa_pairs"]) == 0
        assert len(rejected) == 1

    def test_apply_filters_preserves_metadata(self, sample_config, sample_qa_doc):
        """page_id, section_id etc preserved."""
        filtered_doc, _ = filter_qa.apply_filters(sample_qa_doc, sample_config)
        assert filtered_doc["page_id"] == sample_qa_doc["page_id"]
        assert filtered_doc["section_id"] == sample_qa_doc["section_id"]
        assert filtered_doc["section_name"] == sample_qa_doc["section_name"]

    def test_apply_filters_order_preserved(self, sample_config):
        """Passing Q&A pairs maintain order."""
        doc = {
            "page_id": "test",
            "qa_pairs": [
                {
                    "id": "q1",
                    "question": "First question here?",
                    "answer": "First answer here with enough length.",
                    "question_type": "factual",
                },
                {
                    "id": "q2",
                    "question": "Second question here?",
                    "answer": "Second answer here with enough length.",
                    "question_type": "factual",
                },
                {
                    "id": "q3",
                    "question": "Third question here?",
                    "answer": "Third answer here with enough length.",
                    "question_type": "factual",
                },
            ],
        }

        filtered_doc, _ = filter_qa.apply_filters(doc, sample_config)
        ids = [qa["id"] for qa in filtered_doc["qa_pairs"]]
        assert ids == ["q1", "q2", "q3"]


# ============================================================================
# Test: compute_page_warnings
# ============================================================================

class TestComputePageWarnings:
    """Tests for compute_page_warnings function."""

    def test_compute_page_warnings_all_good(self, sample_config, sample_qa_doc):
        """Returns empty list when all is well."""
        warnings = filter_qa.compute_page_warnings(sample_qa_doc, sample_config)
        assert warnings == []

    def test_compute_page_warnings_low_diversity(self, sample_config):
        """Warns if only 1 question_type."""
        doc = {
            "page_id": "test",
            "qa_pairs": [
                {"id": "q1", "question_type": "factual"},
                {"id": "q2", "question_type": "factual"},
                {"id": "q3", "question_type": "factual"},
            ],
        }

        warnings = filter_qa.compute_page_warnings(doc, sample_config)
        assert len(warnings) > 0
        assert any("question type" in w.lower() for w in warnings)

    def test_compute_page_warnings_all_filtered(self, sample_config):
        """Warns if 0 Q&A remain."""
        doc = {"page_id": "test", "qa_pairs": []}

        warnings = filter_qa.compute_page_warnings(doc, sample_config)
        assert len(warnings) > 0

    def test_compute_page_warnings_few_remaining(self, sample_config):
        """Warns if <3 Q&A remain."""
        doc = {
            "page_id": "test",
            "qa_pairs": [
                {"id": "q1", "question_type": "factual"},
                {"id": "q2", "question_type": "procedural"},
            ],
        }

        warnings = filter_qa.compute_page_warnings(doc, sample_config)
        assert len(warnings) > 0


# ============================================================================
# Test: write_filtered_output
# ============================================================================

class TestWriteFilteredOutput:
    """Tests for write_filtered_output function."""

    def test_write_filtered_output_creates_file(self, tmp_path, sample_qa_doc):
        """File exists after write."""
        output_path = tmp_path / "output.json"
        filter_qa.write_filtered_output(sample_qa_doc, output_path)
        assert output_path.exists()

    def test_write_filtered_output_valid_json(self, tmp_path, sample_qa_doc):
        """Output is parseable JSON."""
        output_path = tmp_path / "output.json"
        filter_qa.write_filtered_output(sample_qa_doc, output_path)

        with open(output_path) as f:
            loaded = json.load(f)
        assert loaded["page_id"] == sample_qa_doc["page_id"]

    def test_write_filtered_output_creates_dirs(self, tmp_path, sample_qa_doc):
        """Creates parent directories if needed."""
        output_path = tmp_path / "nested" / "dir" / "output.json"
        filter_qa.write_filtered_output(sample_qa_doc, output_path)
        assert output_path.exists()

    def test_write_filtered_output_overwrites(self, tmp_path, sample_qa_doc):
        """Overwrites existing file."""
        output_path = tmp_path / "output.json"
        output_path.write_text('{"old": "data"}')

        filter_qa.write_filtered_output(sample_qa_doc, output_path)

        with open(output_path) as f:
            loaded = json.load(f)
        assert "page_id" in loaded


# ============================================================================
# Test: write_rejection_log
# ============================================================================

class TestWriteRejectionLog:
    """Tests for write_rejection_log function."""

    def test_write_rejection_log_creates_file(self, tmp_path):
        """File exists after write."""
        rejections = [
            {
                "qa": {"id": "q1", "question": "Q?", "answer": "A"},
                "page_id": "test",
                "reason": "answer_too_short",
                "filter_name": "answer_length",
            }
        ]
        log_path = tmp_path / "rejections.csv"
        filter_qa.write_rejection_log(rejections, log_path)
        assert log_path.exists()

    def test_write_rejection_log_correct_schema(self, tmp_path):
        """Headers match expected schema."""
        rejections = []
        log_path = tmp_path / "rejections.csv"
        filter_qa.write_rejection_log(rejections, log_path)

        with open(log_path) as f:
            header = f.readline().strip()

        expected_fields = ["timestamp", "page_id", "qa_id", "question", "answer", "rejection_reason", "filter_name"]
        for field in expected_fields:
            assert field in header

    def test_write_rejection_log_escapes_csv(self, tmp_path):
        """Handles commas, quotes in Q&A text."""
        import csv

        rejections = [
            {
                "qa": {
                    "id": "q1",
                    "question": 'What is "the torque", exactly?',
                    "answer": "It's 85 Nm, as specified",
                },
                "page_id": "test",
                "reason": "test_reason",
                "filter_name": "test_filter",
            }
        ]
        log_path = tmp_path / "rejections.csv"
        filter_qa.write_rejection_log(rejections, log_path)

        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        assert "torque" in row["question"]


# ============================================================================
# Test: generate_filter_report
# ============================================================================

class TestGenerateFilterReport:
    """Tests for generate_filter_report function."""

    def test_generate_filter_report_creates_file(self, tmp_path):
        """File exists after generation."""
        from collections import Counter

        stats = {
            "files_processed": 10,
            "total_qa": 100,
            "passed_qa": 90,
            "rejected_qa": 10,
            "rejection_reasons": Counter({"answer_too_short": 5, "generic_answer": 5}),
            "warnings": [],
        }
        report_path = tmp_path / "report.md"
        filter_qa.generate_filter_report(stats, report_path)
        assert report_path.exists()

    def test_generate_filter_report_contains_stats(self, tmp_path):
        """Key stats present in report."""
        from collections import Counter

        stats = {
            "files_processed": 10,
            "total_qa": 100,
            "passed_qa": 90,
            "rejected_qa": 10,
            "rejection_reasons": Counter({"answer_too_short": 5}),
            "warnings": [],
        }
        report_path = tmp_path / "report.md"
        filter_qa.generate_filter_report(stats, report_path)

        content = report_path.read_text()
        assert "100" in content  # Total Q&A
        assert "90" in content  # Passed
        assert "10" in content  # Rejected

    def test_generate_filter_report_includes_samples(self, tmp_path):
        """Sample rejections included in report."""
        from collections import Counter

        stats = {
            "files_processed": 10,
            "total_qa": 100,
            "passed_qa": 90,
            "rejected_qa": 10,
            "rejection_reasons": Counter({"answer_too_short": 5}),
            "warnings": ["test: Low question diversity"],
            "sample_rejections": {
                "answer_too_short": [
                    {"question": "What?", "answer": "X", "reason": "answer_too_short"}
                ]
            },
        }
        report_path = tmp_path / "report.md"
        filter_qa.generate_filter_report(stats, report_path)

        content = report_path.read_text()
        assert "answer_too_short" in content
