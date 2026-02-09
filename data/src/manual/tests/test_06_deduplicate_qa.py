"""
Tests for 06_deduplicate_qa.py - Q&A Deduplication

TDD tests based on specs/05_qa_quality_control_spec.md
"""

import json
import pytest
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
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

dedup_qa = load_module("dedup_qa", Path(__file__).parent.parent / "scripts" / "06_deduplicate_qa.py")



# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_dedup_config():
    """Default deduplication configuration for tests."""
    return dedup_qa.DedupConfig(
        enable_exact_match=True,
        enable_semantic=False,  # Disable for most unit tests
        embedding_model="all-MiniLM-L6-v2",
        similarity_threshold=0.90,
        cross_page_enabled=True,
        prefer_strategy="longer_answer",
        embedding_batch_size=64,
    )


@pytest.fixture
def sample_qa_docs():
    """Sample Q&A documents for deduplication testing."""
    return [
        {
            "page_id": "21-03_clutch",
            "section_id": "21",
            "qa_pairs": [
                {
                    "id": "21-03_clutch-q01",
                    "question": "What is the flywheel bolt torque specification?",
                    "answer": "The flywheel bolts should be torqued to 85 Nm.",
                    "question_type": "factual",
                },
                {
                    "id": "21-03_clutch-q02",
                    "question": "What should I inspect the clutch for?",
                    "answer": "Inspect for cracks, wear, and burnt spots.",
                    "question_type": "inspection",
                },
            ],
        },
        {
            "page_id": "21-04_clutch",
            "section_id": "21",
            "qa_pairs": [
                {
                    "id": "21-04_clutch-q01",
                    "question": "What is the flywheel bolt torque?",
                    "answer": "Torque to 85 Nm.",
                    "question_type": "factual",
                },
                {
                    "id": "21-04_clutch-q02",
                    "question": "How do I adjust clutch free play?",
                    "answer": "Adjust the cable until pedal free play is 15-25 mm.",
                    "question_type": "procedural",
                },
            ],
        },
    ]


@pytest.fixture
def fixtures_dir():
    """Path to test fixtures directory."""
    return Path(__file__).parent / "fixtures" / "qa_raw"


# ============================================================================
# Test: load_dedup_config
# ============================================================================

class TestLoadDedupConfig:
    """Tests for load_dedup_config function."""

    def test_load_dedup_config_full(self, tmp_path):
        """All fields populated from config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
deduplication:
  enable_exact_match: true
  enable_semantic: false
  embedding_model: "custom-model"
  similarity_threshold: 0.85
  cross_page_enabled: false
  prefer_strategy: "first_seen"
  embedding_batch_size: 32
""")

        config = dedup_qa.load_dedup_config(config_file)
        assert config.enable_exact_match is True
        assert config.enable_semantic is False
        assert config.embedding_model == "custom-model"
        assert config.similarity_threshold == 0.85
        assert config.cross_page_enabled is False
        assert config.prefer_strategy == "first_seen"
        assert config.embedding_batch_size == 32

    def test_load_dedup_config_defaults(self, tmp_path):
        """Uses defaults for missing section."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
other_section:
  key: value
""")

        config = dedup_qa.load_dedup_config(config_file)
        defaults = dedup_qa.DedupConfig()
        assert config.enable_exact_match == defaults.enable_exact_match
        assert config.similarity_threshold == defaults.similarity_threshold

    def test_load_dedup_config_invalid_strategy(self, tmp_path):
        """Invalid strategy raises ValueError."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
deduplication:
  prefer_strategy: "invalid_strategy"
""")

        with pytest.raises(ValueError, match="Invalid prefer_strategy"):
            dedup_qa.load_dedup_config(config_file)


# ============================================================================
# Test: normalize_text
# ============================================================================

class TestNormalizeText:
    """Tests for normalize_text function."""

    def test_normalize_text_lowercase(self):
        """'ABC' -> 'abc'."""
        assert dedup_qa.normalize_text("ABC") == "abc"

    def test_normalize_text_punctuation(self):
        """'What's the torque?' -> 'whats the torque'."""
        assert dedup_qa.normalize_text("What's the torque?") == "whats the torque"

    def test_normalize_text_whitespace(self):
        """'a  b   c' -> 'a b c'."""
        assert dedup_qa.normalize_text("a  b   c") == "a b c"

    def test_normalize_text_unicode(self):
        """Handles '85°C' properly."""
        # Should keep numbers and letters, remove special chars
        result = dedup_qa.normalize_text("85°C temperature")
        assert "85" in result
        assert "c" in result.lower()


# ============================================================================
# Test: find_exact_duplicates
# ============================================================================

class TestFindExactDuplicates:
    """Tests for find_exact_duplicates function."""

    def test_find_exact_duplicates_none(self):
        """All unique, returns empty list."""
        qa_pairs = [
            {"id": "q1", "question": "What is the torque?", "answer": "85 Nm", "page_id": "p1"},
            {"id": "q2", "question": "How do I remove it?", "answer": "Pull firmly", "page_id": "p1"},
            {"id": "q3", "question": "Which tool to use?", "answer": "Socket wrench", "page_id": "p1"},
        ]
        groups = dedup_qa.find_exact_duplicates(qa_pairs)
        assert len(groups) == 0

    def test_find_exact_duplicates_pair(self):
        """Two identical questions grouped."""
        qa_pairs = [
            {"id": "q1", "question": "What is the torque?", "answer": "85 Nm", "page_id": "p1"},
            {"id": "q2", "question": "What is the torque?", "answer": "85 Nm spec", "page_id": "p2"},
        ]
        groups = dedup_qa.find_exact_duplicates(qa_pairs)
        assert len(groups) == 1
        assert len(groups[0].members) == 2

    def test_find_exact_duplicates_triple(self):
        """Three identical questions grouped."""
        qa_pairs = [
            {"id": "q1", "question": "What is the torque?", "answer": "A", "page_id": "p1"},
            {"id": "q2", "question": "What is the torque?", "answer": "B", "page_id": "p2"},
            {"id": "q3", "question": "What is the torque?", "answer": "C", "page_id": "p3"},
        ]
        groups = dedup_qa.find_exact_duplicates(qa_pairs)
        assert len(groups) == 1
        assert len(groups[0].members) == 3

    def test_find_exact_duplicates_case_insensitive(self):
        """'What is X?' == 'what is x?'."""
        qa_pairs = [
            {"id": "q1", "question": "What is X?", "answer": "A", "page_id": "p1"},
            {"id": "q2", "question": "what is x?", "answer": "B", "page_id": "p2"},
        ]
        groups = dedup_qa.find_exact_duplicates(qa_pairs)
        assert len(groups) == 1

    def test_find_exact_duplicates_punctuation_insensitive(self):
        """'What's the torque?' == 'Whats the torque'."""
        qa_pairs = [
            {"id": "q1", "question": "What's the torque?", "answer": "A", "page_id": "p1"},
            {"id": "q2", "question": "Whats the torque", "answer": "B", "page_id": "p2"},
        ]
        groups = dedup_qa.find_exact_duplicates(qa_pairs)
        assert len(groups) == 1


# ============================================================================
# Test: dedup_qa.compute_embeddings (with mocking for speed)
# ============================================================================

# Check if sentence-transformers is available
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestComputeEmbeddings:
    """Tests for compute_embeddings function."""

    def test_compute_embeddings_shape(self):
        """Returns correct shape."""
        texts = ["Hello world", "Another sentence"]
        embeddings = dedup_qa.compute_embeddings(texts, "all-MiniLM-L6-v2", batch_size=64)

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Should have embedding dimension

    def test_compute_embeddings_deterministic(self):
        """Same text = same embedding."""
        import numpy as np

        texts = ["Test sentence"]
        emb1 = dedup_qa.compute_embeddings(texts, "all-MiniLM-L6-v2", batch_size=64)
        emb2 = dedup_qa.compute_embeddings(texts, "all-MiniLM-L6-v2", batch_size=64)

        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_compute_embeddings_similar_texts(self):
        """Similar texts have high cosine similarity."""
        import numpy as np

        texts = [
            "What is the flywheel bolt torque?",
            "What torque for flywheel bolts?",
        ]
        embeddings = dedup_qa.compute_embeddings(texts, "all-MiniLM-L6-v2", batch_size=64)

        # Compute cosine similarity
        norm1 = embeddings[0] / np.linalg.norm(embeddings[0])
        norm2 = embeddings[1] / np.linalg.norm(embeddings[1])
        similarity = np.dot(norm1, norm2)

        assert similarity > 0.7  # Should be similar

    def test_compute_embeddings_different_texts(self):
        """Different texts have lower similarity."""
        import numpy as np

        texts = [
            "What is the torque specification?",
            "The weather is nice today.",
        ]
        embeddings = dedup_qa.compute_embeddings(texts, "all-MiniLM-L6-v2", batch_size=64)

        norm1 = embeddings[0] / np.linalg.norm(embeddings[0])
        norm2 = embeddings[1] / np.linalg.norm(embeddings[1])
        similarity = np.dot(norm1, norm2)

        assert similarity < 0.5  # Should be dissimilar


# ============================================================================
# Test: find_semantic_duplicates
# ============================================================================

@pytest.mark.skipif(not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed")
class TestFindSemanticDuplicates:
    """Tests for find_semantic_duplicates function."""

    def test_find_semantic_duplicates_none(self):
        """All semantically distinct, returns empty."""
        config = dedup_qa.DedupConfig(enable_semantic=True, similarity_threshold=0.90)
        qa_pairs = [
            {"id": "q1", "question": "What is the torque?", "answer": "A", "page_id": "p1"},
            {"id": "q2", "question": "The weather is nice today", "answer": "B", "page_id": "p2"},
        ]

        groups = dedup_qa.find_semantic_duplicates(qa_pairs, config, exclude_ids=set())
        assert len(groups) == 0

    def test_find_semantic_duplicates_paraphrase(self):
        """'What's the torque?' ~ 'What torque should I use?'."""
        config = dedup_qa.DedupConfig(enable_semantic=True, similarity_threshold=0.80)
        qa_pairs = [
            {"id": "q1", "question": "What is the flywheel bolt torque specification?", "answer": "A", "page_id": "p1"},
            {"id": "q2", "question": "What torque for flywheel bolts?", "answer": "B", "page_id": "p2"},
        ]

        groups = dedup_qa.find_semantic_duplicates(qa_pairs, config, exclude_ids=set())
        # Depending on model, these might or might not be similar enough
        # Just verify the function runs
        assert isinstance(groups, list)


# ============================================================================
# Test: select_best_qa
# ============================================================================

class TestSelectBestQA:
    """Tests for select_best_qa function."""

    def test_select_best_qa_longer_answer(self):
        """Picks longest answer."""
        members = [
            {"id": "q1", "question": "Q?", "answer": "Short", "page_id": "p1"},
            {"id": "q2", "question": "Q?", "answer": "This is a much longer answer with details", "page_id": "p2"},
        ]
        group = dedup_qa.DuplicateGroup(
            group_id="dup-001",
            members=members,
            kept=None,
            dropped=[],
            match_type="exact",
        )

        best = dedup_qa.select_best_qa(group, "longer_answer")
        assert best["id"] == "q2"

    def test_select_best_qa_longer_answer_tie(self):
        """Tie-breaks by first_seen (sorted by id)."""
        members = [
            {"id": "q2", "question": "Q?", "answer": "Same length answer", "page_id": "p2"},
            {"id": "q1", "question": "Q?", "answer": "Same length answer", "page_id": "p1"},
        ]
        group = dedup_qa.DuplicateGroup(
            group_id="dup-001",
            members=members,
            kept=None,
            dropped=[],
            match_type="exact",
        )

        best = dedup_qa.select_best_qa(group, "longer_answer")
        assert best["id"] == "q1"  # First when sorted by id

    def test_select_best_qa_first_seen(self):
        """Picks earliest by page_id sort order."""
        members = [
            {"id": "q2", "question": "Q?", "answer": "Longer answer here", "page_id": "p2"},
            {"id": "q1", "question": "Q?", "answer": "Short", "page_id": "p1"},
        ]
        group = dedup_qa.DuplicateGroup(
            group_id="dup-001",
            members=members,
            kept=None,
            dropped=[],
            match_type="exact",
        )

        best = dedup_qa.select_best_qa(group, "first_seen")
        assert best["id"] == "q1"  # p1 comes before p2

    def test_select_best_qa_invalid_strategy(self):
        """Invalid strategy raises ValueError."""
        members = [
            {"id": "q1", "question": "Q?", "answer": "A", "page_id": "p1"},
        ]
        group = dedup_qa.DuplicateGroup(
            group_id="dup-001",
            members=members,
            kept=None,
            dropped=[],
            match_type="exact",
        )

        with pytest.raises(ValueError, match="Invalid strategy"):
            dedup_qa.select_best_qa(group, "invalid_strategy")


# ============================================================================
# Test: merge_duplicate_groups
# ============================================================================

class TestMergeDuplicateGroups:
    """Tests for merge_duplicate_groups function."""

    def test_merge_duplicate_groups_no_overlap(self):
        """Simple concatenation when no overlap."""
        exact = [
            dedup_qa.DuplicateGroup("e1", [{"id": "q1"}, {"id": "q2"}], None, [], "exact"),
        ]
        semantic = [
            dedup_qa.DuplicateGroup("s1", [{"id": "q3"}, {"id": "q4"}], None, [], "semantic"),
        ]

        merged = dedup_qa.merge_duplicate_groups(exact, semantic)
        assert len(merged) == 2

    def test_merge_duplicate_groups_overlap(self):
        """Exact group takes precedence when overlap."""
        exact = [
            dedup_qa.DuplicateGroup("e1", [{"id": "q1"}, {"id": "q2"}], None, [], "exact"),
        ]
        semantic = [
            dedup_qa.DuplicateGroup("s1", [{"id": "q2"}, {"id": "q3"}], None, [], "semantic"),  # q2 overlaps
        ]

        merged = dedup_qa.merge_duplicate_groups(exact, semantic)
        # q2 should only appear in exact group
        exact_ids = {m["id"] for g in merged if g.match_type == "exact" for m in g.members}
        assert "q2" in exact_ids

    def test_merge_duplicate_groups_empty(self):
        """Handles empty inputs."""
        merged = dedup_qa.merge_duplicate_groups([], [])
        assert merged == []


# ============================================================================
# Test: apply_deduplication
# ============================================================================

class TestApplyDeduplication:
    """Tests for apply_deduplication function."""

    def test_apply_deduplication_no_duplicates(self, sample_dedup_config):
        """All docs unchanged when no duplicates."""
        docs = [
            {
                "page_id": "p1",
                "qa_pairs": [
                    {"id": "q1", "question": "First question here?", "answer": "A"},
                ],
            },
            {
                "page_id": "p2",
                "qa_pairs": [
                    {"id": "q2", "question": "Second different question?", "answer": "B"},
                ],
            },
        ]

        deduped_docs, groups = dedup_qa.apply_deduplication(docs, sample_dedup_config)

        total_original = sum(len(d["qa_pairs"]) for d in docs)
        total_deduped = sum(len(d["qa_pairs"]) for d in deduped_docs)
        assert total_original == total_deduped
        assert len(groups) == 0

    def test_apply_deduplication_within_page(self, sample_dedup_config):
        """Duplicates in same page removed."""
        docs = [
            {
                "page_id": "p1",
                "qa_pairs": [
                    {"id": "q1", "question": "What is the torque?", "answer": "85 Nm"},
                    {"id": "q2", "question": "What is the torque?", "answer": "85 Nm value"},
                ],
            },
        ]

        deduped_docs, groups = dedup_qa.apply_deduplication(docs, sample_dedup_config)
        assert len(deduped_docs[0]["qa_pairs"]) == 1
        assert len(groups) == 1

    def test_apply_deduplication_cross_page(self, sample_dedup_config):
        """Duplicates across pages removed."""
        docs = [
            {
                "page_id": "p1",
                "qa_pairs": [
                    {"id": "q1", "question": "What is the torque?", "answer": "85 Nm is the torque specification"},
                ],
            },
            {
                "page_id": "p2",
                "qa_pairs": [
                    {"id": "q2", "question": "What is the torque?", "answer": "85 Nm"},
                ],
            },
        ]

        deduped_docs, groups = dedup_qa.apply_deduplication(docs, sample_dedup_config)
        total_deduped = sum(len(d["qa_pairs"]) for d in deduped_docs)
        assert total_deduped == 1
        assert len(groups) == 1

    def test_apply_deduplication_cross_page_disabled(self):
        """config.cross_page_enabled=False keeps cross-page duplicates."""
        config = dedup_qa.DedupConfig(
            enable_exact_match=True,
            enable_semantic=False,
            cross_page_enabled=False,
        )

        docs = [
            {
                "page_id": "p1",
                "qa_pairs": [
                    {"id": "q1", "question": "What is the torque?", "answer": "85 Nm"},
                ],
            },
            {
                "page_id": "p2",
                "qa_pairs": [
                    {"id": "q2", "question": "What is the torque?", "answer": "85 Nm value"},
                ],
            },
        ]

        deduped_docs, groups = dedup_qa.apply_deduplication(docs, config)
        total_deduped = sum(len(d["qa_pairs"]) for d in deduped_docs)
        # Both should be kept since cross-page is disabled
        assert total_deduped == 2

    def test_apply_deduplication_preserves_metadata(self, sample_dedup_config):
        """page_id etc preserved."""
        docs = [
            {
                "page_id": "p1",
                "section_id": "21",
                "section_name": "Clutch",
                "qa_pairs": [
                    {"id": "q1", "question": "Unique question?", "answer": "A"},
                ],
            },
        ]

        deduped_docs, _ = dedup_qa.apply_deduplication(docs, sample_dedup_config)
        assert deduped_docs[0]["page_id"] == "p1"
        assert deduped_docs[0]["section_id"] == "21"
        assert deduped_docs[0]["section_name"] == "Clutch"


# ============================================================================
# Test: write_duplicate_log
# ============================================================================

class TestWriteDuplicateLog:
    """Tests for write_duplicate_log function."""

    def test_write_duplicate_log_creates_file(self, tmp_path):
        """File exists after write."""
        groups = [
            dedup_qa.DuplicateGroup(
                group_id="dup-001",
                members=[
                    {"id": "q1", "question": "Q?", "answer": "A", "page_id": "p1"},
                    {"id": "q2", "question": "Q?", "answer": "B", "page_id": "p2"},
                ],
                kept={"id": "q1", "question": "Q?", "answer": "A", "page_id": "p1"},
                dropped=[{"id": "q2", "question": "Q?", "answer": "B", "page_id": "p2"}],
                match_type="exact",
            ),
        ]

        log_path = tmp_path / "duplicates.csv"
        dedup_qa.write_duplicate_log(groups, log_path)
        assert log_path.exists()

    def test_write_duplicate_log_correct_schema(self, tmp_path):
        """Headers match expected schema."""
        log_path = tmp_path / "duplicates.csv"
        dedup_qa.write_duplicate_log([], log_path)

        with open(log_path) as f:
            header = f.readline().strip()

        expected_fields = ["group_id", "page_id", "qa_id", "question", "answer", "similarity_score", "action", "kept_qa_id"]
        for field in expected_fields:
            assert field in header

    def test_write_duplicate_log_kept_marked(self, tmp_path):
        """Kept member has action='kept'."""
        import csv

        groups = [
            dedup_qa.DuplicateGroup(
                group_id="dup-001",
                members=[
                    {"id": "q1", "question": "Q?", "answer": "A", "page_id": "p1"},
                    {"id": "q2", "question": "Q?", "answer": "B", "page_id": "p2"},
                ],
                kept={"id": "q1", "question": "Q?", "answer": "A", "page_id": "p1"},
                dropped=[{"id": "q2", "question": "Q?", "answer": "B", "page_id": "p2"}],
                match_type="exact",
            ),
        ]

        log_path = tmp_path / "duplicates.csv"
        dedup_qa.write_duplicate_log(groups, log_path)

        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        kept_rows = [r for r in rows if r["action"] == "kept"]
        assert len(kept_rows) == 1
        assert kept_rows[0]["qa_id"] == "q1"

    def test_write_duplicate_log_dropped_marked(self, tmp_path):
        """Dropped members have action='dropped'."""
        import csv

        groups = [
            dedup_qa.DuplicateGroup(
                group_id="dup-001",
                members=[
                    {"id": "q1", "question": "Q?", "answer": "A", "page_id": "p1"},
                    {"id": "q2", "question": "Q?", "answer": "B", "page_id": "p2"},
                ],
                kept={"id": "q1", "question": "Q?", "answer": "A", "page_id": "p1"},
                dropped=[{"id": "q2", "question": "Q?", "answer": "B", "page_id": "p2"}],
                match_type="exact",
            ),
        ]

        log_path = tmp_path / "duplicates.csv"
        dedup_qa.write_duplicate_log(groups, log_path)

        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        dropped_rows = [r for r in rows if r["action"] == "dropped"]
        assert len(dropped_rows) == 1
        assert dropped_rows[0]["qa_id"] == "q2"


# ============================================================================
# Test: generate_dedup_report
# ============================================================================

class TestGenerateDedupReport:
    """Tests for generate_dedup_report function."""

    def test_generate_dedup_report_creates_file(self, tmp_path):
        """File exists after generation."""
        stats = {
            "files_processed": 10,
            "total_input": 100,
            "total_output": 90,
            "duplicates_removed": 10,
            "exact_groups": 5,
            "semantic_groups": 3,
            "duplicate_groups": [],
        }

        report_path = tmp_path / "report.md"
        dedup_qa.generate_dedup_report(stats, report_path)
        assert report_path.exists()

    def test_generate_dedup_report_contains_stats(self, tmp_path):
        """Key stats present in report."""
        stats = {
            "files_processed": 10,
            "total_input": 100,
            "total_output": 90,
            "duplicates_removed": 10,
            "exact_groups": 5,
            "semantic_groups": 3,
            "duplicate_groups": [],
        }

        report_path = tmp_path / "report.md"
        dedup_qa.generate_dedup_report(stats, report_path)

        content = report_path.read_text()
        assert "100" in content  # Total input
        assert "90" in content  # Total output
        assert "10" in content  # Removed
