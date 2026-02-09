"""
Tests for 08_validate_vlm.py - VLM Dataset Validation

TDD tests based on specs/06_emit_validate_spec.md
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import importlib.util

# Dynamic import to handle the numeric prefix in module name
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

validate_vlm = load_module("validate_vlm", Path(__file__).parent.parent / "scripts" / "08_validate_vlm.py")



# ============================================================================
# Test: load_validation_config
# ============================================================================

class TestLoadValidationConfig:
    """Tests for load_validation_config function."""

    def test_load_validation_config_full(self, tmp_path):
        """All fields populated from config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
vlm_validation:
  require_image_field: false
  require_conversations: true
  require_metadata: false
  check_image_exists: false
  check_image_readable: false
  min_image_width: 200
  min_image_height: 150
  min_question_length: 15
  min_answer_length: 10
  max_answer_length: 500
  min_qa_per_section: 10
  max_qa_per_section: 1000
  num_samples_per_split: 3
""")

        config = validate_vlm.load_validation_config(config_file)
        assert config.require_image_field is False
        assert config.min_image_width == 200
        assert config.max_answer_length == 500
        assert config.num_samples_per_split == 3

    def test_load_validation_config_defaults(self, tmp_path):
        """Uses defaults for missing values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
other_section:
  key: value
""")

        config = validate_vlm.load_validation_config(config_file)
        defaults = validate_vlm.ValidationConfig()
        assert config.require_image_field == defaults.require_image_field
        assert config.min_question_length == defaults.min_question_length


# ============================================================================
# Test: load_jsonl
# ============================================================================

class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_load_jsonl_valid(self, tmp_path):
        """Loads all records."""
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"id": "1"}\n')
            f.write('{"id": "2"}\n')
            f.write('{"id": "3"}\n')

        records = validate_vlm.load_jsonl(jsonl_file)
        assert len(records) == 3

    def test_load_jsonl_missing_file(self, tmp_path):
        """Raises FileNotFoundError."""
        missing_file = tmp_path / "missing.jsonl"

        with pytest.raises(FileNotFoundError):
            validate_vlm.load_jsonl(missing_file)

    def test_load_jsonl_invalid_json(self, tmp_path):
        """Raises ValueError with line number."""
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"id": "1"}\n')
            f.write('invalid json {{\n')
            f.write('{"id": "3"}\n')

        with pytest.raises(ValueError, match="line 2"):
            validate_vlm.load_jsonl(jsonl_file)

    def test_load_jsonl_empty_file(self, tmp_path):
        """Returns empty list."""
        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.touch()

        records = validate_vlm.load_jsonl(jsonl_file)
        assert records == []

    def test_load_jsonl_trailing_newline(self, tmp_path):
        """Handles trailing newline gracefully."""
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, "w") as f:
            f.write('{"id": "1"}\n')
            f.write('{"id": "2"}\n')
            f.write('\n')  # Trailing empty line

        records = validate_vlm.load_jsonl(jsonl_file)
        assert len(records) == 2


# ============================================================================
# Test: validate_schema
# ============================================================================

class TestValidateSchema:
    """Tests for validate_schema function."""

    def test_validate_schema_valid(self, sample_vlm_record):
        """Returns empty list for valid record."""
        config = validate_vlm.ValidationConfig()
        errors = validate_vlm.validate_schema(sample_vlm_record, config)
        assert errors == []

    def test_validate_schema_missing_image(self):
        """Error if require_image_field=True and missing."""
        config = validate_vlm.ValidationConfig(require_image_field=True)
        record = {
            "conversations": [{"role": "user", "content": "Q?"}],
            "metadata": {}
        }

        errors = validate_vlm.validate_schema(record, config)
        assert any("image" in e.lower() for e in errors)

    def test_validate_schema_missing_conversations(self):
        """Error if require_conversations=True and missing."""
        config = validate_vlm.ValidationConfig(require_conversations=True)
        record = {
            "image": "test.jpg",
            "metadata": {}
        }

        errors = validate_vlm.validate_schema(record, config)
        assert any("conversations" in e.lower() for e in errors)

    def test_validate_schema_invalid_conversations(self):
        """Error if conversations is not a list."""
        config = validate_vlm.ValidationConfig(require_conversations=True)
        record = {
            "image": "test.jpg",
            "conversations": "not a list",
            "metadata": {}
        }

        errors = validate_vlm.validate_schema(record, config)
        assert len(errors) > 0

    def test_validate_schema_bad_role(self):
        """Error if role not user/assistant."""
        config = validate_vlm.ValidationConfig(require_conversations=True)
        record = {
            "image": "test.jpg",
            "conversations": [
                {"role": "invalid_role", "content": "test"}
            ],
            "metadata": {}
        }

        errors = validate_vlm.validate_schema(record, config)
        assert any("role" in e.lower() for e in errors)

    def test_validate_schema_missing_metadata(self):
        """Error if require_metadata=True and missing."""
        config = validate_vlm.ValidationConfig(require_metadata=True)
        record = {
            "image": "test.jpg",
            "conversations": []
        }

        errors = validate_vlm.validate_schema(record, config)
        assert any("metadata" in e.lower() for e in errors)

    def test_validate_schema_null_image_allowed(self):
        """image=null is valid for text-only."""
        config = validate_vlm.ValidationConfig(require_image_field=True)
        record = {
            "image": None,
            "conversations": [
                {"role": "user", "content": "Q?"},
                {"role": "assistant", "content": "A."}
            ],
            "metadata": {}
        }

        errors = validate_vlm.validate_schema(record, config)
        assert errors == []


# ============================================================================
# Test: validate_conversations
# ============================================================================

class TestValidateConversations:
    """Tests for validate_conversations function."""

    def test_validate_conversations_valid(self):
        """No errors or warnings for valid conversations."""
        config = validate_vlm.ValidationConfig(
            min_question_length=5,
            min_answer_length=3,
            max_answer_length=100
        )
        record = {
            "conversations": [
                {"role": "user", "content": "What is the torque specification?"},
                {"role": "assistant", "content": "The torque is 85 Nm."}
            ]
        }

        errors, warnings = validate_vlm.validate_conversations(record, config)
        assert errors == []
        assert warnings == []

    def test_validate_conversations_short_question(self):
        """Warning for short question."""
        config = validate_vlm.ValidationConfig(min_question_length=20)
        record = {
            "conversations": [
                {"role": "user", "content": "Torque?"},
                {"role": "assistant", "content": "The torque is 85 Nm."}
            ]
        }

        errors, warnings = validate_vlm.validate_conversations(record, config)
        assert any("question" in w.lower() and "short" in w.lower() for w in warnings)

    def test_validate_conversations_short_answer(self):
        """Warning for short answer."""
        config = validate_vlm.ValidationConfig(min_answer_length=20)
        record = {
            "conversations": [
                {"role": "user", "content": "What is the torque specification?"},
                {"role": "assistant", "content": "85 Nm"}
            ]
        }

        errors, warnings = validate_vlm.validate_conversations(record, config)
        assert any("answer" in w.lower() and "short" in w.lower() for w in warnings)

    def test_validate_conversations_long_answer(self):
        """Warning for long answer."""
        config = validate_vlm.ValidationConfig(max_answer_length=20)
        record = {
            "conversations": [
                {"role": "user", "content": "What is the torque specification?"},
                {"role": "assistant", "content": "The torque specification for this component is 85 Nm and must be applied in a cross pattern."}
            ]
        }

        errors, warnings = validate_vlm.validate_conversations(record, config)
        assert any("answer" in w.lower() and "long" in w.lower() for w in warnings)

    def test_validate_conversations_wrong_order(self):
        """Error for wrong role order."""
        config = validate_vlm.ValidationConfig()
        record = {
            "conversations": [
                {"role": "assistant", "content": "Answer first"},
                {"role": "user", "content": "Question second"}
            ]
        }

        errors, warnings = validate_vlm.validate_conversations(record, config)
        assert any("order" in e.lower() or "role" in e.lower() for e in errors)

    def test_validate_conversations_empty(self):
        """Error for empty conversations."""
        config = validate_vlm.ValidationConfig()
        record = {"conversations": []}

        errors, warnings = validate_vlm.validate_conversations(record, config)
        assert any("empty" in e.lower() for e in errors)


# ============================================================================
# Test: validate_image
# ============================================================================

class TestValidateImage:
    """Tests for validate_image function."""

    def test_validate_image_exists(self, temp_image_dir):
        """No errors for existing image."""
        # Create test image in expected location
        images_dir = temp_image_dir.parent / "images"
        images_dir.mkdir(exist_ok=True)

        try:
            from PIL import Image
            img = Image.new("RGB", (800, 600), color="white")
            img.save(images_dir / "test.jpg", "JPEG")
        except ImportError:
            (images_dir / "test.jpg").write_bytes(b'\xff\xd8\xff\xe0')

        config = validate_vlm.ValidationConfig(check_image_exists=True, check_image_readable=False)
        record = {"image": "images/test.jpg"}

        errors, warnings = validate_vlm.validate_image(record, images_dir.parent, config)
        # Should find the image
        assert not any("not found" in e.lower() for e in errors)

    def test_validate_image_missing(self, tmp_path):
        """Error for missing image."""
        config = validate_vlm.ValidationConfig(check_image_exists=True)
        record = {"image": "images/missing.jpg"}

        errors, warnings = validate_vlm.validate_image(record, tmp_path, config)
        assert any("not found" in e.lower() or "missing" in e.lower() or "exist" in e.lower() for e in errors)

    def test_validate_image_null(self):
        """No checks for null image."""
        config = validate_vlm.ValidationConfig(check_image_exists=True)
        record = {"image": None}

        errors, warnings = validate_vlm.validate_image(record, Path("/tmp"), config)
        assert errors == []
        assert warnings == []

    def test_validate_image_skip_check(self, tmp_path):
        """Respects check_image_exists=False."""
        config = validate_vlm.ValidationConfig(check_image_exists=False)
        record = {"image": "images/missing.jpg"}

        errors, warnings = validate_vlm.validate_image(record, tmp_path, config)
        assert errors == []


# ============================================================================
# Test: compute_distribution_stats
# ============================================================================

class TestComputeDistributionStats:
    """Tests for compute_distribution_stats function."""

    def test_compute_distribution_stats_sections(self):
        """Correct section counts."""
        records = [
            {"metadata": {"section_id": "21", "source_type": "manual", "question_type": "factual", "content_type": "proc"}},
            {"metadata": {"section_id": "21", "source_type": "manual", "question_type": "factual", "content_type": "proc"}},
            {"metadata": {"section_id": "23", "source_type": "manual", "question_type": "factual", "content_type": "proc"}},
        ]

        stats = validate_vlm.compute_distribution_stats(records)
        assert stats["section_counts"]["21"] == 2
        assert stats["section_counts"]["23"] == 1

    def test_compute_distribution_stats_sources(self):
        """Correct source counts."""
        records = [
            {"metadata": {"section_id": "21", "source_type": "service_manual", "question_type": "factual", "content_type": "proc"}},
            {"metadata": {"section_id": "21", "source_type": "html_specs", "question_type": "factual", "content_type": "proc"}},
        ]

        stats = validate_vlm.compute_distribution_stats(records)
        assert stats["source_counts"]["service_manual"] == 1
        assert stats["source_counts"]["html_specs"] == 1

    def test_compute_distribution_stats_image_count(self):
        """Correct image/text-only counts."""
        records = [
            {"image": "img1.jpg", "metadata": {"section_id": "21", "source_type": "m", "question_type": "f", "content_type": "p"}},
            {"image": None, "metadata": {"section_id": "21", "source_type": "m", "question_type": "f", "content_type": "p"}},
            {"image": "img2.jpg", "metadata": {"section_id": "21", "source_type": "m", "question_type": "f", "content_type": "p"}},
        ]

        stats = validate_vlm.compute_distribution_stats(records)
        assert stats["image_count"] == 2
        assert stats["text_only_count"] == 1

    def test_compute_distribution_stats_empty(self):
        """Handles empty list."""
        stats = validate_vlm.compute_distribution_stats([])
        assert stats["image_count"] == 0
        assert stats["text_only_count"] == 0


# ============================================================================
# Test: check_distribution_issues
# ============================================================================

class TestCheckDistributionIssues:
    """Tests for check_distribution_issues function."""

    def test_check_distribution_issues_none(self):
        """No warnings for balanced data."""
        config = validate_vlm.ValidationConfig(min_qa_per_section=5, max_qa_per_section=100)
        stats = {
            "section_counts": {"21": 50, "23": 50}
        }

        warnings = validate_vlm.check_distribution_issues(stats, config)
        assert warnings == []

    def test_check_distribution_issues_sparse_section(self):
        """Warns for low-count section."""
        config = validate_vlm.ValidationConfig(min_qa_per_section=10)
        stats = {
            "section_counts": {"21": 5, "23": 50}
        }

        warnings = validate_vlm.check_distribution_issues(stats, config)
        assert any("21" in w for w in warnings)

    def test_check_distribution_issues_dense_section(self):
        """Warns for high-count section."""
        config = validate_vlm.ValidationConfig(max_qa_per_section=100)
        stats = {
            "section_counts": {"21": 500, "23": 50}
        }

        warnings = validate_vlm.check_distribution_issues(stats, config)
        assert any("21" in w for w in warnings)


# ============================================================================
# Test: sample_records
# ============================================================================

class TestSampleRecords:
    """Tests for sample_records function."""

    def test_sample_records_exact_n(self):
        """Returns exactly n records."""
        records = [{"id": str(i)} for i in range(100)]
        sample = validate_vlm.sample_records(records, 5)
        assert len(sample) == 5

    def test_sample_records_fewer_available(self):
        """Returns all if < n records."""
        records = [{"id": str(i)} for i in range(3)]
        sample = validate_vlm.sample_records(records, 10)
        assert len(sample) == 3

    def test_sample_records_reproducible(self):
        """Same seed = same sample."""
        records = [{"id": str(i)} for i in range(100)]
        sample1 = validate_vlm.sample_records(records, 5, seed=42)
        sample2 = validate_vlm.sample_records(records, 5, seed=42)
        assert sample1 == sample2


# ============================================================================
# Test: generate_validation_report
# ============================================================================

class TestGenerateValidationReport:
    """Tests for generate_validation_report function."""

    def test_generate_validation_report_passed(self, tmp_path):
        """Status shows PASSED."""
        results = {
            "passed": True,
            "train_count": 90,
            "val_count": 10,
            "train_stats": {"section_counts": {}, "source_counts": {}, "image_count": 50, "text_only_count": 0},
            "val_stats": {"section_counts": {}, "source_counts": {}, "image_count": 5, "text_only_count": 0},
            "errors": [],
            "warnings": [],
            "train_samples": [],
            "val_samples": []
        }

        report_path = tmp_path / "report.md"
        validate_vlm.generate_validation_report(results, report_path)

        content = report_path.read_text()
        assert "PASSED" in content

    def test_generate_validation_report_failed(self, tmp_path):
        """Status shows FAILED."""
        results = {
            "passed": False,
            "train_count": 90,
            "val_count": 10,
            "train_stats": {"section_counts": {}, "source_counts": {}, "image_count": 50, "text_only_count": 0},
            "val_stats": {"section_counts": {}, "source_counts": {}, "image_count": 5, "text_only_count": 0},
            "errors": ["Missing image: test.jpg"],
            "warnings": [],
            "train_samples": [],
            "val_samples": []
        }

        report_path = tmp_path / "report.md"
        validate_vlm.generate_validation_report(results, report_path)

        content = report_path.read_text()
        assert "FAILED" in content

    def test_generate_validation_report_includes_samples(self, tmp_path, sample_vlm_record):
        """Samples included in output."""
        results = {
            "passed": True,
            "train_count": 90,
            "val_count": 10,
            "train_stats": {"section_counts": {}, "source_counts": {}, "image_count": 50, "text_only_count": 0},
            "val_stats": {"section_counts": {}, "source_counts": {}, "image_count": 5, "text_only_count": 0},
            "errors": [],
            "warnings": [],
            "train_samples": [sample_vlm_record],
            "val_samples": []
        }

        report_path = tmp_path / "report.md"
        validate_vlm.generate_validation_report(results, report_path)

        content = report_path.read_text()
        assert "Sample" in content or "sample" in content

    def test_generate_validation_report_error_list(self, tmp_path):
        """Errors listed in report."""
        results = {
            "passed": False,
            "train_count": 90,
            "val_count": 10,
            "train_stats": {"section_counts": {}, "source_counts": {}, "image_count": 50, "text_only_count": 0},
            "val_stats": {"section_counts": {}, "source_counts": {}, "image_count": 5, "text_only_count": 0},
            "errors": ["Error 1: Missing field", "Error 2: Invalid format"],
            "warnings": ["Warning 1: Short answer"],
            "train_samples": [],
            "val_samples": []
        }

        report_path = tmp_path / "report.md"
        validate_vlm.generate_validation_report(results, report_path)

        content = report_path.read_text()
        assert "Error 1" in content or "Missing field" in content
