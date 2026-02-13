"""
Tests for 07_emit.py - Emit Training Data

Tests VLM record conversion and train/val splitting.
"""

import json
import pytest
from pathlib import Path

from pipeline import emit


# ============================================================================
# Test: load_emit_config
# ============================================================================

class TestLoadEmitConfig:
    """Tests for configuration loading."""

    def test_load_config_full(self, config_file):
        """All fields populated."""
        config = emit.load_emit_config(config_file)
        assert config.train_split == 0.90
        assert config.random_seed == 42
        assert config.prefix == "forum"
        assert config.stratify_by == "forum"

    def test_load_config_defaults(self, tmp_path):
        """Uses defaults for missing config."""
        missing = tmp_path / "missing.yaml"
        config = emit.load_emit_config(missing)
        defaults = emit.EmitConfig()
        assert config.train_split == defaults.train_split

    def test_load_config_partial(self, tmp_path):
        """Merges provided with defaults."""
        import yaml
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"output": {"train_split": 0.80}}))

        config = emit.load_emit_config(config_file)
        assert config.train_split == 0.80
        assert config.random_seed == emit.EmitConfig().random_seed


# ============================================================================
# Test: qa_to_vlm_record
# ============================================================================

class TestQAToVLMRecord:
    """Tests for converting Q&A pairs to VLM format."""

    def test_basic_conversion(self, sample_qa_doc):
        """Produces valid VLM record."""
        pair = sample_qa_doc["qa_pairs"][0]
        record = emit.qa_to_vlm_record(pair, sample_qa_doc)

        assert "conversations" in record
        assert "metadata" in record
        assert "image" not in record  # Forum = text-only

    def test_conversation_structure(self, sample_qa_doc):
        """Correct user/assistant alternation."""
        pair = sample_qa_doc["qa_pairs"][0]
        record = emit.qa_to_vlm_record(pair, sample_qa_doc)

        convos = record["conversations"]
        assert len(convos) == 2
        assert convos[0]["role"] == "user"
        assert convos[1]["role"] == "assistant"
        assert convos[0]["content"] == pair["question"]
        assert convos[1]["content"] == pair["answer"]

    def test_metadata_fields(self, sample_qa_doc):
        """All metadata fields present."""
        pair = sample_qa_doc["qa_pairs"][0]
        record = emit.qa_to_vlm_record(pair, sample_qa_doc)

        meta = record["metadata"]
        assert meta["source"] == "forum"
        assert meta["source_id"] == pair["id"]
        assert meta["thread_id"] == sample_qa_doc["thread_id"]
        assert meta["forum"] == sample_qa_doc["forum"]
        assert meta["content_type"] == sample_qa_doc["content_type"]
        assert meta["question_type"] == pair["question_type"]
        assert meta["quality_score"] == sample_qa_doc["quality_score"]

    def test_factual_confidence_in_metadata(self, sample_qa_doc):
        """Factual confidence is included."""
        pair = sample_qa_doc["qa_pairs"][0]
        record = emit.qa_to_vlm_record(pair, sample_qa_doc)
        assert "factual_confidence" in record["metadata"]

    def test_no_image_field(self, sample_qa_doc):
        """Forum records have no image field."""
        pair = sample_qa_doc["qa_pairs"][0]
        record = emit.qa_to_vlm_record(pair, sample_qa_doc)
        assert "image" not in record


# ============================================================================
# Test: stratified_split
# ============================================================================

class TestStratifiedSplit:
    """Tests for train/val splitting."""

    def test_split_ratio(self):
        """Approximately correct train/val ratio."""
        records = [
            {"metadata": {"forum": "d-i-y"}, "conversations": []}
            for _ in range(100)
        ]
        train, val = emit.stratified_split(records, 0.90, 42, None)
        assert len(train) == 90
        assert len(val) == 10

    def test_reproducible(self):
        """Same seed gives same split."""
        records = [
            {"metadata": {"forum": "d-i-y"}, "conversations": [], "id": i}
            for i in range(100)
        ]
        train1, val1 = emit.stratified_split(records, 0.90, 42, None)
        train2, val2 = emit.stratified_split(records, 0.90, 42, None)
        assert train1 == train2
        assert val1 == val2

    def test_different_seed(self):
        """Different seed gives different split."""
        records = [
            {"metadata": {"forum": "d-i-y"}, "conversations": [], "id": i}
            for i in range(100)
        ]
        train1, _ = emit.stratified_split(records, 0.90, 42, None)
        train2, _ = emit.stratified_split(records, 0.90, 99, None)
        assert train1 != train2

    def test_stratified_by_forum(self):
        """Each forum has ~90/10 distribution."""
        records = (
            [{"metadata": {"forum": "d-i-y"}, "conversations": []} for _ in range(50)]
            + [{"metadata": {"forum": "no-start"}, "conversations": []} for _ in range(50)]
        )
        train, val = emit.stratified_split(records, 0.90, 42, "forum")

        train_diy = [r for r in train if r["metadata"]["forum"] == "d-i-y"]
        train_ns = [r for r in train if r["metadata"]["forum"] == "no-start"]
        assert 40 <= len(train_diy) <= 50
        assert 40 <= len(train_ns) <= 50

    def test_single_record(self):
        """Handles edge case of 1 record."""
        records = [{"metadata": {"forum": "d-i-y"}, "conversations": []}]
        train, val = emit.stratified_split(records, 0.90, 42, None)
        assert len(train) + len(val) == 1

    def test_empty_input(self):
        """Handles empty input."""
        train, val = emit.stratified_split([], 0.90, 42, None)
        assert train == []
        assert val == []


# ============================================================================
# Test: end-to-end emit
# ============================================================================

class TestEmitEndToEnd:
    """Integration tests for the full emit pipeline."""

    def test_emits_jsonl_files(self, qa_validated_dir, tmp_path, config_file):
        """Produces train and val JSONL files."""
        output = tmp_path / "prepared"
        output.mkdir()
        report = tmp_path / "emit_report.md"

        config = emit.load_emit_config(config_file)

        # Load Q&A
        records = []
        for qa_file in sorted(qa_validated_dir.glob("*.json")):
            with open(qa_file) as f:
                doc = json.load(f)
            for pair in doc.get("qa_pairs", []):
                records.append(emit.qa_to_vlm_record(pair, doc))

        # Split
        train, val = emit.stratified_split(
            records, config.train_split, config.random_seed, config.stratify_by,
        )

        # Write
        train_path = output / "forum_train.jsonl"
        val_path = output / "forum_val.jsonl"

        with open(train_path, "w") as f:
            for rec in train:
                f.write(json.dumps(rec) + "\n")
        with open(val_path, "w") as f:
            for rec in val:
                f.write(json.dumps(rec) + "\n")

        assert train_path.exists()
        assert val_path.exists()
        assert len(train) + len(val) == 2  # 2 Q&A pairs from fixture

    def test_output_schema(self, qa_validated_dir, tmp_path):
        """Output records match the data architecture contract."""
        for qa_file in qa_validated_dir.glob("*.json"):
            with open(qa_file) as f:
                doc = json.load(f)
            for pair in doc.get("qa_pairs", []):
                record = emit.qa_to_vlm_record(pair, doc)

                # Required fields per contract
                assert "conversations" in record
                assert len(record["conversations"]) == 2
                assert record["conversations"][0]["role"] == "user"
                assert record["conversations"][1]["role"] == "assistant"
                assert "metadata" in record
                assert record["metadata"]["source"] == "forum"

    def test_report_generation(self, tmp_path):
        """Report file is created with expected content."""
        from collections import Counter

        report_path = tmp_path / "report.md"
        emit._write_report(
            report_path,
            total=100,
            train_count=90,
            val_count=10,
            forum_counts=Counter({"d-i-y": 50, "no-start": 50}),
            type_counts=Counter({"specification": 60, "procedural": 40}),
        )

        assert report_path.exists()
        content = report_path.read_text()
        assert "100" in content
        assert "90" in content
        assert "d-i-y" in content
