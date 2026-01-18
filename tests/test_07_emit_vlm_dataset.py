"""
Tests for 07_emit_vlm_dataset.py - VLM Dataset Emission

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

emit_vlm = load_module("emit_vlm", Path(__file__).parent.parent / "scripts" / "07_emit_vlm_dataset.py")



# ============================================================================
# Test: load_qa_documents
# ============================================================================

class TestLoadQADocuments:
    """Tests for load_qa_documents function."""

    def test_load_qa_documents_valid_directory(self, temp_qa_dir):
        """Loads all JSON files from directory."""
        docs = emit_vlm.load_qa_documents(temp_qa_dir)
        assert len(docs) == 2

    def test_load_qa_documents_empty_directory(self, tmp_path):
        """Raises ValueError for empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No JSON files found"):
            emit_vlm.load_qa_documents(empty_dir)

    def test_load_qa_documents_missing_directory(self, tmp_path):
        """Raises FileNotFoundError for missing directory."""
        missing_dir = tmp_path / "missing"

        with pytest.raises(FileNotFoundError):
            emit_vlm.load_qa_documents(missing_dir)

    def test_load_qa_documents_skips_invalid_json(self, tmp_path):
        """Logs warning and continues with valid files."""
        qa_dir = tmp_path / "qa"
        qa_dir.mkdir()

        # Valid JSON
        with open(qa_dir / "valid.json", "w") as f:
            json.dump({"page_id": "test", "qa_pairs": []}, f)

        # Invalid JSON
        with open(qa_dir / "invalid.json", "w") as f:
            f.write("not valid json {{{")

        docs = emit_vlm.load_qa_documents(qa_dir)
        assert len(docs) == 1

    def test_load_qa_documents_preserves_order(self, tmp_path):
        """Files loaded in consistent order by filename."""
        qa_dir = tmp_path / "qa"
        qa_dir.mkdir()

        # Create files in non-alphabetical order
        for name in ["c.json", "a.json", "b.json"]:
            with open(qa_dir / name, "w") as f:
                json.dump({"page_id": name.replace(".json", ""), "qa_pairs": []}, f)

        docs = emit_vlm.load_qa_documents(qa_dir)
        page_ids = [d["page_id"] for d in docs]
        assert page_ids == ["a", "b", "c"]


# ============================================================================
# Test: load_emit_config
# ============================================================================

class TestLoadEmitConfig:
    """Tests for load_emit_config function."""

    def test_load_emit_config_full(self, tmp_path):
        """All fields populated from config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
output:
  train_split: 0.85
  random_seed: 123
  image_copy_mode: copy
  image_output_dir: my_images
  normalize_image_names: false
  stratify_by: source_type
  min_stratum_size: 5
""")

        config = emit_vlm.load_emit_config(config_file)
        assert config.train_split == 0.85
        assert config.random_seed == 123
        assert config.image_copy_mode == "copy"
        assert config.image_output_dir == "my_images"
        assert config.normalize_image_names is False
        assert config.stratify_by == "source_type"
        assert config.min_stratum_size == 5

    def test_load_emit_config_defaults(self, tmp_path):
        """Uses defaults for missing values."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
other_section:
  key: value
""")

        config = emit_vlm.load_emit_config(config_file)
        defaults = emit_vlm.EmitConfig()
        assert config.train_split == defaults.train_split
        assert config.random_seed == defaults.random_seed

    def test_load_emit_config_invalid_split_low(self, tmp_path):
        """Raises ValueError for split < 0."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
output:
  train_split: -0.1
""")

        with pytest.raises(ValueError, match="train_split"):
            emit_vlm.load_emit_config(config_file)

    def test_load_emit_config_invalid_split_high(self, tmp_path):
        """Raises ValueError for split > 1."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
output:
  train_split: 1.5
""")

        with pytest.raises(ValueError, match="train_split"):
            emit_vlm.load_emit_config(config_file)

    def test_load_emit_config_invalid_mode(self, tmp_path):
        """Raises ValueError for unknown image_copy_mode."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
output:
  image_copy_mode: invalid_mode
""")

        with pytest.raises(ValueError, match="image_copy_mode"):
            emit_vlm.load_emit_config(config_file)


# ============================================================================
# Test: flatten_qa_pairs
# ============================================================================

class TestFlattenQAPairs:
    """Tests for flatten_qa_pairs function."""

    def test_flatten_qa_pairs_single_doc(self, sample_qa_doc):
        """Flattens one document with multiple Q&A pairs."""
        records = emit_vlm.flatten_qa_pairs([sample_qa_doc])
        assert len(records) == 2  # sample_qa_doc has 2 Q&A pairs

    def test_flatten_qa_pairs_multiple_docs(self, sample_qa_doc, sample_html_qa_doc):
        """Flattens multiple documents."""
        records = emit_vlm.flatten_qa_pairs([sample_qa_doc, sample_html_qa_doc])
        assert len(records) == 3  # 2 + 1 Q&A pairs

    def test_flatten_qa_pairs_empty_doc(self, sample_qa_doc):
        """Handles document with no qa_pairs."""
        empty_doc = {
            "page_id": "empty",
            "section_id": "00",
            "qa_pairs": []
        }

        records = emit_vlm.flatten_qa_pairs([sample_qa_doc, empty_doc])
        assert len(records) == 2  # Only from sample_qa_doc

    def test_flatten_qa_pairs_preserves_metadata(self, sample_qa_doc):
        """All metadata fields present in flattened records."""
        records = emit_vlm.flatten_qa_pairs([sample_qa_doc])

        for rec in records:
            assert "question" in rec
            assert "answer" in rec
            assert "image_path" in rec
            assert "page_id" in rec
            assert "section_id" in rec
            assert "section_name" in rec
            assert "source_type" in rec
            assert "content_type" in rec
            assert "question_type" in rec
            assert "qa_id" in rec

    def test_flatten_qa_pairs_handles_null_image(self, sample_html_qa_doc):
        """HTML sources have image_path=None."""
        records = emit_vlm.flatten_qa_pairs([sample_html_qa_doc])
        assert records[0]["image_path"] is None


# ============================================================================
# Test: normalize_image_filename
# ============================================================================

class TestNormalizeImageFilename:
    """Tests for normalize_image_filename function."""

    def test_normalize_image_filename_spaces(self):
        """Replaces spaces with underscores."""
        result = emit_vlm.normalize_image_filename("21 - Clutch/21-03.jpg")
        assert " " not in result

    def test_normalize_image_filename_special_chars(self):
        """Handles colons, parens, etc."""
        result = emit_vlm.normalize_image_filename("Bosch Motronic ML 3-1/page001.jpg")
        assert " " not in result
        assert "/" not in result

    def test_normalize_image_filename_preserves_extension(self):
        """Extension stays intact."""
        result = emit_vlm.normalize_image_filename("path/to/file.jpg")
        assert result.endswith(".jpg")

    def test_normalize_image_filename_nested_path(self):
        """Flattens directory structure."""
        result = emit_vlm.normalize_image_filename("data_src/21 - Clutch/21-03.jpg")
        assert "/" not in result

    def test_normalize_image_filename_already_clean(self):
        """No change if already normalized."""
        result = emit_vlm.normalize_image_filename("clean_filename.jpg")
        assert result == "clean_filename.jpg"


# ============================================================================
# Test: format_vlm_record
# ============================================================================

class TestFormatVLMRecord:
    """Tests for format_vlm_record function."""

    def test_format_vlm_record_with_image(self):
        """Image path included in output."""
        qa_record = {
            "question": "What is the torque?",
            "answer": "85 Nm",
            "page_id": "21-03",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "procedure",
            "question_type": "factual",
            "qa_id": "21-03-q01"
        }

        result = emit_vlm.format_vlm_record(qa_record, "images/21-03.jpg")
        assert result["image"] == "images/21-03.jpg"

    def test_format_vlm_record_without_image(self):
        """image=None for text-only records."""
        qa_record = {
            "question": "What is the displacement?",
            "answer": "2302 cc",
            "page_id": "html-spec",
            "section_id": "spec",
            "section_name": "Specifications",
            "source_type": "html_specs",
            "content_type": "specification",
            "question_type": "factual",
            "qa_id": "html-q01"
        }

        result = emit_vlm.format_vlm_record(qa_record, None)
        assert result["image"] is None

    def test_format_vlm_record_conversation_structure(self):
        """Correct role alternation."""
        qa_record = {
            "question": "What is X?",
            "answer": "It is Y.",
            "page_id": "p1",
            "section_id": "01",
            "section_name": "Test",
            "source_type": "test",
            "content_type": "test",
            "question_type": "factual",
            "qa_id": "q1"
        }

        result = emit_vlm.format_vlm_record(qa_record, None)
        assert len(result["conversations"]) == 2
        assert result["conversations"][0]["role"] == "user"
        assert result["conversations"][1]["role"] == "assistant"
        assert result["conversations"][0]["content"] == "What is X?"
        assert result["conversations"][1]["content"] == "It is Y."

    def test_format_vlm_record_metadata_complete(self):
        """All metadata fields present."""
        qa_record = {
            "question": "Q?",
            "answer": "A.",
            "page_id": "p1",
            "section_id": "01",
            "section_name": "Test",
            "source_type": "test",
            "content_type": "test",
            "question_type": "factual",
            "qa_id": "q1"
        }

        result = emit_vlm.format_vlm_record(qa_record, "img.jpg")
        meta = result["metadata"]
        assert meta["page_id"] == "p1"
        assert meta["section_id"] == "01"
        assert meta["section_name"] == "Test"
        assert meta["source_type"] == "test"
        assert meta["content_type"] == "test"
        assert meta["question_type"] == "factual"
        assert meta["qa_id"] == "q1"

    def test_format_vlm_record_escapes_content(self):
        """Handles quotes in Q&A text."""
        qa_record = {
            "question": 'What does "TDC" mean?',
            "answer": 'It means "Top Dead Center".',
            "page_id": "p1",
            "section_id": "01",
            "section_name": "Test",
            "source_type": "test",
            "content_type": "test",
            "question_type": "factual",
            "qa_id": "q1"
        }

        result = emit_vlm.format_vlm_record(qa_record, None)
        # Should be JSON serializable
        json_str = json.dumps(result)
        assert '"TDC"' in json_str or '\\"TDC\\"' in json_str


# ============================================================================
# Test: stratified_split
# ============================================================================

class TestStratifiedSplit:
    """Tests for stratified_split function."""

    def test_stratified_split_ratio(self):
        """Approximately 90/10 split."""
        config = emit_vlm.EmitConfig(train_split=0.9, random_seed=42, stratify_by=None)
        records = [{"section_id": "21", "id": f"q{i}"} for i in range(100)]

        train, val = emit_vlm.stratified_split(records, config)
        assert len(train) == 90
        assert len(val) == 10

    def test_stratified_split_reproducible(self):
        """Same seed = same split."""
        config = emit_vlm.EmitConfig(train_split=0.9, random_seed=42, stratify_by=None)
        records = [{"section_id": "21", "id": f"q{i}"} for i in range(100)]

        train1, val1 = emit_vlm.stratified_split(records, config)
        train2, val2 = emit_vlm.stratified_split(records, config)

        assert [r["id"] for r in train1] == [r["id"] for r in train2]
        assert [r["id"] for r in val1] == [r["id"] for r in val2]

    def test_stratified_split_different_seed(self):
        """Different seed = different split."""
        records = [{"section_id": "21", "id": f"q{i}"} for i in range(100)]

        config1 = emit_vlm.EmitConfig(train_split=0.9, random_seed=42, stratify_by=None)
        config2 = emit_vlm.EmitConfig(train_split=0.9, random_seed=123, stratify_by=None)

        train1, _ = emit_vlm.stratified_split(records, config1)
        train2, _ = emit_vlm.stratified_split(records, config2)

        # Very unlikely to be identical with different seeds
        assert [r["id"] for r in train1] != [r["id"] for r in train2]

    def test_stratified_split_by_section(self):
        """Each section has ~90/10 distribution."""
        config = emit_vlm.EmitConfig(train_split=0.9, random_seed=42, stratify_by="section_id", min_stratum_size=5)

        # 50 records in section 21, 50 in section 23
        records = (
            [{"section_id": "21", "id": f"s21-q{i}"} for i in range(50)] +
            [{"section_id": "23", "id": f"s23-q{i}"} for i in range(50)]
        )

        train, val = emit_vlm.stratified_split(records, config)

        # Check distribution per section
        train_21 = [r for r in train if r["section_id"] == "21"]
        train_23 = [r for r in train if r["section_id"] == "23"]
        val_21 = [r for r in val if r["section_id"] == "21"]
        val_23 = [r for r in val if r["section_id"] == "23"]

        # Each section should have ~90% in train
        assert 40 <= len(train_21) <= 50
        assert 40 <= len(train_23) <= 50
        assert 0 <= len(val_21) <= 10
        assert 0 <= len(val_23) <= 10

    def test_stratified_split_no_stratify(self):
        """Random split when stratify_by=None."""
        config = emit_vlm.EmitConfig(train_split=0.8, random_seed=42, stratify_by=None)
        records = [{"section_id": "21", "id": f"q{i}"} for i in range(100)]

        train, val = emit_vlm.stratified_split(records, config)
        assert len(train) == 80
        assert len(val) == 20

    def test_stratified_split_small_stratum(self):
        """Falls back to random for small strata."""
        config = emit_vlm.EmitConfig(train_split=0.9, random_seed=42, stratify_by="section_id", min_stratum_size=10)

        # Only 5 records in one section (below min_stratum_size)
        records = [{"section_id": "21", "id": f"q{i}"} for i in range(5)]

        train, val = emit_vlm.stratified_split(records, config)
        # Should still work, just not stratified
        assert len(train) + len(val) == 5

    def test_stratified_split_single_record(self):
        """Handles edge case of 1 record."""
        config = emit_vlm.EmitConfig(train_split=0.9, random_seed=42, stratify_by=None)
        records = [{"section_id": "21", "id": "q1"}]

        train, val = emit_vlm.stratified_split(records, config)
        assert len(train) + len(val) == 1


# ============================================================================
# Test: copy_or_link_image
# ============================================================================

class TestCopyOrLinkImage:
    """Tests for copy_or_link_image function."""

    def test_copy_or_link_image_copy(self, temp_image_dir, tmp_path):
        """File copied successfully."""
        src = temp_image_dir / "21 - Clutch" / "21-03.jpg"
        dst = tmp_path / "output" / "21-03.jpg"

        result = emit_vlm.copy_or_link_image(src, dst, "copy")
        assert result is True
        assert dst.exists()
        assert dst.is_file()

    def test_copy_or_link_image_symlink(self, temp_image_dir, tmp_path):
        """Symlink created."""
        src = temp_image_dir / "21 - Clutch" / "21-03.jpg"
        dst = tmp_path / "output" / "21-03.jpg"

        result = emit_vlm.copy_or_link_image(src, dst, "symlink")
        assert result is True
        assert dst.exists()
        assert dst.is_symlink()

    def test_copy_or_link_image_missing_source(self, tmp_path):
        """Returns False for missing source."""
        src = tmp_path / "missing.jpg"
        dst = tmp_path / "output" / "missing.jpg"

        result = emit_vlm.copy_or_link_image(src, dst, "copy")
        assert result is False

    def test_copy_or_link_image_creates_dirs(self, temp_image_dir, tmp_path):
        """Creates parent directories."""
        src = temp_image_dir / "21 - Clutch" / "21-03.jpg"
        dst = tmp_path / "deep" / "nested" / "path" / "21-03.jpg"

        result = emit_vlm.copy_or_link_image(src, dst, "copy")
        assert result is True
        assert dst.exists()

    def test_copy_or_link_image_idempotent(self, temp_image_dir, tmp_path):
        """Skips existing valid files."""
        src = temp_image_dir / "21 - Clutch" / "21-03.jpg"
        dst = tmp_path / "21-03.jpg"

        # First call
        emit_vlm.copy_or_link_image(src, dst, "copy")
        mtime1 = dst.stat().st_mtime

        # Second call should skip
        result = emit_vlm.copy_or_link_image(src, dst, "copy")
        assert result is True
        mtime2 = dst.stat().st_mtime
        assert mtime1 == mtime2  # File not modified


# ============================================================================
# Test: write_jsonl
# ============================================================================

class TestWriteJsonl:
    """Tests for write_jsonl function."""

    def test_write_jsonl_creates_file(self, tmp_path):
        """File exists after write."""
        records = [{"id": "1"}, {"id": "2"}]
        output_path = tmp_path / "output.jsonl"

        count = emit_vlm.write_jsonl(records, output_path)
        assert output_path.exists()
        assert count == 2

    def test_write_jsonl_valid_jsonl(self, tmp_path):
        """Each line is valid JSON."""
        records = [{"id": "1", "name": "test"}, {"id": "2", "name": "test2"}]
        output_path = tmp_path / "output.jsonl"

        emit_vlm.write_jsonl(records, output_path)

        with open(output_path) as f:
            for line in f:
                parsed = json.loads(line)
                assert "id" in parsed

    def test_write_jsonl_line_count(self, tmp_path):
        """Correct number of lines."""
        records = [{"id": str(i)} for i in range(10)]
        output_path = tmp_path / "output.jsonl"

        emit_vlm.write_jsonl(records, output_path)

        with open(output_path) as f:
            lines = f.readlines()
        assert len(lines) == 10

    def test_write_jsonl_utf8(self, tmp_path):
        """Handles unicode characters."""
        records = [{"text": "Temperature: 85°C", "symbol": "µm"}]
        output_path = tmp_path / "output.jsonl"

        emit_vlm.write_jsonl(records, output_path)

        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "85°C" in content
        assert "µm" in content

    def test_write_jsonl_no_trailing_newline_issues(self, tmp_path):
        """Proper line endings."""
        records = [{"id": "1"}, {"id": "2"}]
        output_path = tmp_path / "output.jsonl"

        emit_vlm.write_jsonl(records, output_path)

        with open(output_path) as f:
            content = f.read()

        # Should end with newline
        assert content.endswith("\n")
        # No double newlines
        assert "\n\n" not in content


# ============================================================================
# Test: collect_unique_images
# ============================================================================

class TestCollectUniqueImages:
    """Tests for collect_unique_images function."""

    def test_collect_unique_images_deduplicates(self):
        """Same image referenced twice = one entry."""
        records = [
            {"image_path": "img1.jpg"},
            {"image_path": "img1.jpg"},
            {"image_path": "img2.jpg"},
        ]

        unique = emit_vlm.collect_unique_images(records)
        assert len(unique) == 2
        assert "img1.jpg" in unique
        assert "img2.jpg" in unique

    def test_collect_unique_images_excludes_none(self):
        """HTML sources not included."""
        records = [
            {"image_path": "img1.jpg"},
            {"image_path": None},
            {"image_path": "img2.jpg"},
        ]

        unique = emit_vlm.collect_unique_images(records)
        assert len(unique) == 2
        assert None not in unique

    def test_collect_unique_images_empty(self):
        """Empty input = empty set."""
        unique = emit_vlm.collect_unique_images([])
        assert len(unique) == 0


# ============================================================================
# Test: generate_emit_report
# ============================================================================

class TestGenerateEmitReport:
    """Tests for generate_emit_report function."""

    def test_generate_emit_report_creates_file(self, tmp_path):
        """File exists after generation."""
        stats = {
            "total_qa": 100,
            "train_qa": 90,
            "val_qa": 10,
            "unique_images": 50,
            "text_only_qa": 5,
            "images_copied": 50,
            "images_missing": 0,
            "image_copy_mode": "symlink",
            "section_distribution": {},
            "source_distribution": {},
            "warnings": []
        }

        report_path = tmp_path / "report.md"
        emit_vlm.generate_emit_report(stats, report_path)
        assert report_path.exists()

    def test_generate_emit_report_contains_stats(self, tmp_path):
        """Key metrics present in report."""
        stats = {
            "total_qa": 100,
            "train_qa": 90,
            "val_qa": 10,
            "unique_images": 50,
            "text_only_qa": 5,
            "images_copied": 50,
            "images_missing": 2,
            "image_copy_mode": "copy",
            "section_distribution": {},
            "source_distribution": {},
            "warnings": ["Missing image: test.jpg"]
        }

        report_path = tmp_path / "report.md"
        emit_vlm.generate_emit_report(stats, report_path)

        content = report_path.read_text()
        assert "100" in content  # Total Q&A
        assert "90" in content  # Train
        assert "10" in content  # Val
        assert "Warning" in content or "warning" in content.lower()

    def test_generate_emit_report_distribution_tables(self, tmp_path):
        """Section/source tables present when data exists."""
        stats = {
            "total_qa": 100,
            "train_qa": 90,
            "val_qa": 10,
            "unique_images": 50,
            "text_only_qa": 5,
            "images_copied": 50,
            "images_missing": 0,
            "image_copy_mode": "symlink",
            "section_distribution": {("21", "train"): 45, ("21", "val"): 5},
            "source_distribution": {("service_manual", "train"): 90, ("service_manual", "val"): 10},
            "warnings": []
        }

        report_path = tmp_path / "report.md"
        emit_vlm.generate_emit_report(stats, report_path)

        content = report_path.read_text()
        assert "Section" in content or "section" in content.lower()
