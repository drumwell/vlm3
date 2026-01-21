"""
DeepEval test cases for VLM evaluation.

Run with: pytest eval/test_vlm.py -v
"""

import os
from pathlib import Path

import pytest

from eval.metrics import (
    RougeL,
    NumericExactMatch,
    UnitConsistency,
    KeywordPresence,
    SafetyCriticalAccuracy,
    VisualGrounding,
    ProcedureStepOrdering,
)
from eval.conftest import extract_question_answer


# =============================================================================
# Metric Unit Tests
# =============================================================================

class TestRougeL:
    """Test ROUGE-L metric."""

    def test_identical_strings(self):
        metric = RougeL()
        result = metric.compute("The torque is 30 Nm", "The torque is 30 Nm")
        assert result.score == 1.0
        assert result.passed

    def test_partial_overlap(self):
        metric = RougeL()
        result = metric.compute(
            "The torque specification is 30 Nm",
            "Apply 30 Nm of torque"
        )
        assert 0.3 < result.score < 1.0

    def test_no_overlap(self):
        metric = RougeL()
        result = metric.compute(
            "Hello world",
            "Goodbye moon"
        )
        assert result.score < 0.3


class TestNumericExactMatch:
    """Test numeric extraction accuracy."""

    def test_exact_torque_match(self):
        metric = NumericExactMatch()
        result = metric.compute(
            "The specification is 30 Nm",
            "30 Nm"
        )
        assert result.score == 1.0
        assert result.passed

    def test_within_tolerance(self):
        metric = NumericExactMatch()
        result = metric.compute(
            "Apply 31 Nm of torque",
            "30 Nm"  # Within 1 Nm tolerance
        )
        assert result.score == 1.0

    def test_outside_tolerance(self):
        metric = NumericExactMatch()
        result = metric.compute(
            "Apply 35 Nm of torque",
            "30 Nm"  # Outside tolerance
        )
        assert result.score == 0.0

    def test_multiple_values(self):
        metric = NumericExactMatch()
        result = metric.compute(
            "Gap: 0.25 mm, Torque: 30 Nm",
            "0.25 mm and 30 Nm"
        )
        assert result.score == 1.0


class TestUnitConsistency:
    """Test unit consistency checking."""

    def test_canonical_units(self):
        metric = UnitConsistency()
        result = metric.compute("The torque is 30 Nm and clearance is 0.5 mm")
        assert result.score == 1.0
        assert result.passed

    def test_non_canonical_ftlbs(self):
        metric = UnitConsistency()
        result = metric.compute("Apply 22 ft-lbs of torque")
        assert result.score < 1.0
        assert not result.passed
        assert "ft-lbs" in str(result.details["issues"])

    def test_non_canonical_inches(self):
        metric = UnitConsistency()
        result = metric.compute("Gap should be 0.010 inch")
        assert result.score < 1.0

    def test_non_canonical_psi(self):
        metric = UnitConsistency()
        result = metric.compute("Pressure: 45 psi")
        assert result.score < 1.0


class TestKeywordPresence:
    """Test keyword presence checking."""

    def test_all_keywords_present(self):
        metric = KeywordPresence()
        result = metric.compute(
            "Remove the transmission and inspect the clutch plate",
            ["transmission", "clutch", "plate"]
        )
        assert result.score == 1.0
        assert result.passed

    def test_partial_keywords(self):
        metric = KeywordPresence()
        result = metric.compute(
            "Remove the transmission assembly",
            ["transmission", "clutch", "flywheel"]
        )
        assert result.score == pytest.approx(1/3, 0.01)

    def test_no_required_keywords(self):
        metric = KeywordPresence()
        result = metric.compute("Some text", [])
        assert result.score == 1.0


class TestSafetyCriticalAccuracy:
    """Test safety-critical threshold handling."""

    def test_safety_question_type(self):
        metric = SafetyCriticalAccuracy()
        assert metric.is_safety_critical("Any question", question_type="safety")

    def test_safety_keywords_in_question(self):
        metric = SafetyCriticalAccuracy()
        assert metric.is_safety_critical(
            "What is the torque specification for the brake caliper bolt?"
        )

    def test_non_safety_question(self):
        metric = SafetyCriticalAccuracy()
        assert not metric.is_safety_critical("What color is the wire?")


class TestVisualGrounding:
    """Test visual grounding detection."""

    def test_grounded_answer(self):
        metric = VisualGrounding()
        result = metric.compute(
            "As shown in the diagram, the connector is located in the upper right corner with a red indicator",
            question_type="visual"
        )
        assert result.score > 0.3

    def test_generic_answer(self):
        metric = VisualGrounding()
        result = metric.compute(
            "This component typically requires regular maintenance and is usually found in similar locations",
            question_type="visual"
        )
        assert result.score < 0.5


class TestProcedureStepOrdering:
    """Test procedure step ordering."""

    def test_sequential_steps(self):
        metric = ProcedureStepOrdering()
        result = metric.compute(
            "1. Remove cover. 2. Disconnect wires. 3. Extract component."
        )
        assert result.score == 1.0
        assert result.passed

    def test_missing_step(self):
        metric = ProcedureStepOrdering()
        result = metric.compute(
            "1. Remove cover. 3. Extract component."  # Missing step 2
        )
        assert result.score < 1.0

    def test_no_numbered_steps(self):
        metric = ProcedureStepOrdering()
        result = metric.compute(
            "First remove the cover, then disconnect wires."
        )
        assert result.score == 0.5  # Neutral score


# =============================================================================
# Integration Tests with Sample Data
# =============================================================================

@pytest.mark.slow
class TestEvaluationSample:
    """Tests using actual evaluation sample data."""

    def test_sample_loading(self, eval_sample):
        """Verify evaluation sample loads correctly."""
        assert len(eval_sample) > 0
        # Check first record structure
        first = eval_sample[0]
        assert "image" in first
        assert "conversations" in first
        assert "metadata" in first

    def test_extract_qa(self, eval_sample, question_answer_extractor):
        """Verify Q&A extraction from records."""
        for record in eval_sample[:10]:
            question, answer = question_answer_extractor(record)
            assert len(question) > 0
            assert len(answer) > 0

    def test_question_type_distribution(self, eval_sample):
        """Verify question types are distributed."""
        types = set()
        for record in eval_sample:
            qtype = record.get("metadata", {}).get("question_type")
            if qtype:
                types.add(qtype)

        # Should have multiple question types
        assert len(types) >= 3


# =============================================================================
# Manual Probe Tests
# =============================================================================

class TestManualProbes:
    """Tests for manual probe evaluation."""

    def test_probes_loading(self, manual_probes):
        """Verify manual probes load correctly."""
        assert len(manual_probes) > 0

        # Check probe structure
        first = manual_probes[0]
        assert "id" in first
        assert "category" in first
        assert "question" in first
        assert "image" in first

    def test_probe_categories(self, manual_probes):
        """Verify all expected categories are present."""
        categories = {p["category"] for p in manual_probes}
        expected = {"specification", "procedure", "safety", "visual_only"}
        assert categories & expected  # At least some overlap

    def test_critical_probes_flagged(self, manual_probes):
        """Verify critical probes are properly flagged."""
        critical = [p for p in manual_probes if p.get("is_critical")]
        assert len(critical) > 0


# =============================================================================
# Model Inference Tests (with mock)
# =============================================================================

class TestVLMInference:
    """Tests for VLM inference (using mock by default)."""

    def test_mock_inference(self, vlm_wrapper):
        """Test mock wrapper generates responses."""
        response = vlm_wrapper.generate(
            image_path="test.jpg",
            question="What is the torque specification?"
        )
        assert len(response) > 0
        assert "torque" in response.lower() or "nm" in response.lower()

    def test_mock_batch_inference(self, vlm_wrapper):
        """Test batch inference."""
        examples = [
            {"image": "test1.jpg", "question": "What is the torque spec?"},
            {"image": "test2.jpg", "question": "Describe the procedure."},
        ]
        responses = vlm_wrapper.batch_generate(examples)
        assert len(responses) == 2
        assert all(len(r) > 0 for r in responses)


# =============================================================================
# LLM-as-Judge Tests (requires ANTHROPIC_API_KEY)
# =============================================================================

@pytest.mark.llm_judge
class TestLLMJudge:
    """Tests requiring LLM-as-judge (Claude)."""

    def test_deepeval_metrics_available(self, deepeval_model):
        """Verify DeepEval metrics can be configured."""
        from eval.metrics import get_deepeval_metrics

        metrics = get_deepeval_metrics()
        assert metrics is not None
        assert "answer_relevancy" in metrics
        assert "correctness" in metrics
