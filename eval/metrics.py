"""
Custom evaluation metrics for BMW E30 M3 VLM evaluation.

Includes:
- Automated metrics (ROUGE-L, NumericExactMatch, UnitConsistency, KeywordPresence)
- Domain-specific metrics (SafetyCriticalAccuracy, VisualGrounding, ProcedureStepOrdering)
- DeepEval integration for LLM-as-judge metrics
"""

import re
from typing import Optional
from dataclasses import dataclass

from rouge_score import rouge_scorer


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MetricResult:
    """Result from a metric evaluation."""
    name: str
    score: float
    passed: bool
    details: dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


# =============================================================================
# Automated Metrics (No LLM Judge Required)
# =============================================================================

class RougeL:
    """ROUGE-L metric for general text quality."""

    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def compute(self, prediction: str, reference: str) -> MetricResult:
        """Compute ROUGE-L score between prediction and reference."""
        scores = self.scorer.score(reference, prediction)
        rouge_l = scores['rougeL'].fmeasure

        return MetricResult(
            name="ROUGE-L",
            score=rouge_l,
            passed=rouge_l >= 0.3,  # Threshold for basic quality
            details={
                "precision": scores['rougeL'].precision,
                "recall": scores['rougeL'].recall,
            }
        )


class NumericExactMatch:
    """
    Numeric extraction accuracy with tolerance.

    Handles specifications with tolerances:
    - Torque: +/- 1 Nm
    - Clearances: +/- 0.05 mm
    - Pressures: +/- 0.1 bar
    """

    # Tolerances by unit type
    TOLERANCES = {
        "nm": 1.0,       # Torque
        "mm": 0.05,      # Length/clearance
        "bar": 0.1,      # Pressure
        "l": 0.1,        # Volume
        "°c": 1.0,       # Temperature
    }

    # Pattern to extract numbers with optional units
    NUMBER_PATTERN = re.compile(
        r'(\d+(?:\.\d+)?)\s*'
        r'(nm|mm|bar|l|litre|liter|°c|celsius|psi|ft-?lbs?)?',
        re.IGNORECASE
    )

    def extract_numbers(self, text: str) -> list[tuple[float, str]]:
        """Extract all numbers with their units from text."""
        matches = self.NUMBER_PATTERN.findall(text.lower())
        return [(float(m[0]), m[1].lower() if m[1] else "") for m in matches]

    def compute(self, prediction: str, reference: str) -> MetricResult:
        """Compare numeric values with appropriate tolerances."""
        pred_nums = self.extract_numbers(prediction)
        ref_nums = self.extract_numbers(reference)

        if not ref_nums:
            # No reference numbers to check
            return MetricResult(
                name="NumericExactMatch",
                score=1.0,
                passed=True,
                details={"note": "No numeric values in reference"}
            )

        matches = 0
        comparisons = []

        for ref_val, ref_unit in ref_nums:
            tolerance = self.TOLERANCES.get(ref_unit, 0.01 * ref_val)  # Default 1% tolerance
            matched = False

            for pred_val, pred_unit in pred_nums:
                if abs(pred_val - ref_val) <= tolerance:
                    matches += 1
                    matched = True
                    comparisons.append({
                        "reference": f"{ref_val} {ref_unit}",
                        "predicted": f"{pred_val} {pred_unit}",
                        "within_tolerance": True
                    })
                    break

            if not matched:
                comparisons.append({
                    "reference": f"{ref_val} {ref_unit}",
                    "predicted": None,
                    "within_tolerance": False
                })

        score = matches / len(ref_nums) if ref_nums else 1.0

        return MetricResult(
            name="NumericExactMatch",
            score=score,
            passed=score >= 0.9,  # 90% threshold
            details={"comparisons": comparisons}
        )


class UnitConsistency:
    """
    Verify units match canonical forms.

    Based on config.yaml unit_canon:
    - torque: Nm (not ft-lbs)
    - pressure: bar (not psi)
    - length: mm (not inches)
    - volume: L (not gallons)
    - temp: C (not F)
    """

    # Canonical units
    CANONICAL = {
        "torque": "nm",
        "length": "mm",
        "pressure": "bar",
        "volume": "l",
        "temp": "c",
    }

    # Non-canonical patterns to flag
    NON_CANONICAL_PATTERNS = [
        (r'\b\d+\s*(?:ft[\s-]?lb|foot[\s-]?pound)', "ft-lbs (should be Nm)"),
        (r'\b\d+\s*(?:psi|pounds?\s+per\s+square)', "psi (should be bar)"),
        (r'\b\d+(?:\.\d+)?\s*(?:inch|in\.|")\b', "inches (should be mm)"),
        (r'\b\d+(?:\.\d+)?\s*(?:gal|gallon)', "gallons (should be L)"),
        (r'\b\d+\s*°?\s*F\b', "Fahrenheit (should be Celsius)"),
    ]

    def compute(self, prediction: str) -> MetricResult:
        """Check for non-canonical units in prediction."""
        issues = []

        for pattern, description in self.NON_CANONICAL_PATTERNS:
            if re.search(pattern, prediction, re.IGNORECASE):
                issues.append(description)

        score = 1.0 if not issues else max(0, 1.0 - (len(issues) * 0.25))

        return MetricResult(
            name="UnitConsistency",
            score=score,
            passed=len(issues) == 0,
            details={"issues": issues}
        )


class KeywordPresence:
    """Check for presence of required technical keywords in response."""

    def compute(self, prediction: str, required_keywords: list[str]) -> MetricResult:
        """Check what fraction of required keywords are present."""
        if not required_keywords:
            return MetricResult(
                name="KeywordPresence",
                score=1.0,
                passed=True,
                details={"note": "No required keywords specified"}
            )

        pred_lower = prediction.lower()
        found = []
        missing = []

        for keyword in required_keywords:
            if keyword.lower() in pred_lower:
                found.append(keyword)
            else:
                missing.append(keyword)

        score = len(found) / len(required_keywords)

        return MetricResult(
            name="KeywordPresence",
            score=score,
            passed=score >= 0.7,  # 70% keyword presence threshold
            details={"found": found, "missing": missing}
        )


# =============================================================================
# Domain-Specific Metrics
# =============================================================================

class SafetyCriticalAccuracy:
    """
    Higher threshold for safety-critical questions.

    Safety-critical includes:
    - question_type == "safety"
    - Questions about torque specs for critical components
    - Questions about brake systems
    - Questions about steering components
    """

    SAFETY_KEYWORDS = [
        "torque", "brake", "steering", "safety", "warning", "caution",
        "danger", "critical", "tighten", "bolt", "nut", "specification"
    ]

    def __init__(self, base_threshold: float = 0.7, safety_threshold: float = 0.9):
        self.base_threshold = base_threshold
        self.safety_threshold = safety_threshold

    def is_safety_critical(self, question: str, question_type: str = None) -> bool:
        """Determine if a question is safety-critical."""
        if question_type == "safety":
            return True

        question_lower = question.lower()
        safety_count = sum(1 for kw in self.SAFETY_KEYWORDS if kw in question_lower)
        return safety_count >= 2

    def compute(
        self,
        prediction: str,
        reference: str,
        question: str,
        question_type: str = None,
        correctness_score: float = None
    ) -> MetricResult:
        """
        Evaluate safety-critical accuracy.

        Uses provided correctness_score or falls back to simple overlap metric.
        """
        is_critical = self.is_safety_critical(question, question_type)
        threshold = self.safety_threshold if is_critical else self.base_threshold

        # If correctness_score provided (from LLM judge), use it
        if correctness_score is not None:
            score = correctness_score
        else:
            # Fallback: simple word overlap metric
            pred_words = set(prediction.lower().split())
            ref_words = set(reference.lower().split())
            if ref_words:
                overlap = len(pred_words & ref_words) / len(ref_words)
                score = min(overlap * 1.5, 1.0)  # Scale up slightly
            else:
                score = 1.0

        return MetricResult(
            name="SafetyCriticalAccuracy",
            score=score,
            passed=score >= threshold,
            details={
                "is_safety_critical": is_critical,
                "threshold_used": threshold,
            }
        )


class VisualGrounding:
    """
    Verify answer uses image-specific details.

    Checks for:
    - References to visual elements (colors, positions, labels)
    - Specific diagram callouts
    - Image-only answerable content
    """

    VISUAL_INDICATORS = [
        r'\b(shown|visible|depicted|illustrated|see|arrow|labeled|marked)\b',
        r'\b(top|bottom|left|right|center|upper|lower)\b',
        r'\b(red|blue|green|yellow|black|white|colored?)\b',
        r'\b(diagram|figure|image|photo|illustration)\b',
        r'\b(connector\s+\d+|pin\s+\d+|terminal\s+[a-z0-9]+)\b',
    ]

    # Patterns suggesting generic/not grounded answers
    GENERIC_PATTERNS = [
        r'\b(typically|usually|generally|often|commonly)\b',
        r'\b(should be|would be|might be|could be)\b',
        r'\b(in most cases|as a rule|normally)\b',
    ]

    def compute(
        self,
        prediction: str,
        question_type: str = None,
        is_visual_only: bool = False
    ) -> MetricResult:
        """Evaluate visual grounding of an answer."""
        pred_lower = prediction.lower()

        # Count visual indicators
        visual_count = sum(
            1 for pattern in self.VISUAL_INDICATORS
            if re.search(pattern, pred_lower)
        )

        # Count generic patterns (negative indicator)
        generic_count = sum(
            1 for pattern in self.GENERIC_PATTERNS
            if re.search(pattern, pred_lower)
        )

        # Base score from visual indicators
        score = min(visual_count * 0.2, 1.0)

        # Penalty for generic patterns
        score = max(0, score - (generic_count * 0.15))

        # Higher threshold for explicitly visual questions
        threshold = 0.5 if (question_type == "visual" or is_visual_only) else 0.3

        return MetricResult(
            name="VisualGrounding",
            score=score,
            passed=score >= threshold,
            details={
                "visual_indicators_found": visual_count,
                "generic_patterns_found": generic_count,
                "is_visual_question": question_type == "visual" or is_visual_only,
            }
        )


class ProcedureStepOrdering:
    """
    Check that procedural answers maintain correct step ordering.

    For procedural Q&A, verifies:
    - Steps are numbered correctly
    - Logical sequence is maintained
    - No missing intermediate steps
    """

    STEP_PATTERN = re.compile(r'(?:^|\n)\s*(\d+)[.\)]\s*(.+?)(?=(?:\n\s*\d+[.\)]|\Z))', re.DOTALL)

    def extract_steps(self, text: str) -> list[tuple[int, str]]:
        """Extract numbered steps from text."""
        matches = self.STEP_PATTERN.findall(text)
        return [(int(m[0]), m[1].strip()) for m in matches]

    def compute(self, prediction: str, reference: str = None) -> MetricResult:
        """Evaluate procedure step ordering."""
        pred_steps = self.extract_steps(prediction)

        if not pred_steps:
            # No numbered steps found
            return MetricResult(
                name="ProcedureStepOrdering",
                score=0.5,  # Neutral if no steps
                passed=True,
                details={"note": "No numbered steps found in prediction"}
            )

        # Check step numbers are sequential
        step_numbers = [s[0] for s in pred_steps]
        is_sequential = step_numbers == list(range(1, len(step_numbers) + 1))

        # Check for gaps
        max_step = max(step_numbers)
        expected_steps = set(range(1, max_step + 1))
        actual_steps = set(step_numbers)
        missing_steps = expected_steps - actual_steps

        # Calculate score
        if is_sequential:
            score = 1.0
        elif not missing_steps:
            score = 0.8  # Correct steps but not starting from 1
        else:
            score = max(0, 1.0 - (len(missing_steps) * 0.2))

        return MetricResult(
            name="ProcedureStepOrdering",
            score=score,
            passed=score >= 0.7,
            details={
                "step_count": len(pred_steps),
                "is_sequential": is_sequential,
                "missing_steps": list(missing_steps),
            }
        )


# =============================================================================
# LLM-as-Judge Integration (Claude via DeepEval)
# =============================================================================

def get_deepeval_metrics(threshold: float = 0.7, safety_threshold: float = 0.9):
    """
    Get configured DeepEval metrics for evaluation.

    Returns a dict of metric configurations for use with DeepEval.
    Uses Claude as the LLM judge.
    """
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            GEval,
        )
        from deepeval.models import AnthropicModel

        # Configure Claude as judge
        judge_model = AnthropicModel(model="claude-sonnet-4-20250514")

        return {
            "answer_relevancy": AnswerRelevancyMetric(
                threshold=threshold,
                model=judge_model,
            ),
            "faithfulness": FaithfulnessMetric(
                threshold=threshold,
                model=judge_model,
            ),
            "correctness": GEval(
                name="Correctness",
                criteria="Is the answer factually accurate for BMW E30 M3 service procedures? "
                         "Consider technical specifications, procedures, and safety information.",
                evaluation_steps=[
                    "Check if numeric values (torque specs, clearances) are accurate",
                    "Verify procedure steps match service manual conventions",
                    "Ensure component names and part numbers are correct",
                    "Check that safety warnings are not omitted for critical procedures"
                ],
                threshold=threshold,
                model=judge_model,
            ),
            "completeness": GEval(
                name="Completeness",
                criteria="Does the answer include all key information needed to answer the question? "
                         "For procedural questions, are all necessary steps included?",
                evaluation_steps=[
                    "Check if all relevant steps are mentioned for procedures",
                    "Verify all requested specifications are provided",
                    "Ensure prerequisites and warnings are included where needed"
                ],
                threshold=threshold,
                model=judge_model,
            ),
            # Higher threshold version for safety-critical questions
            "correctness_safety": GEval(
                name="Correctness_Safety",
                criteria="Is the answer factually accurate for safety-critical BMW E30 M3 information? "
                         "This requires higher accuracy for brake, steering, and torque specifications.",
                evaluation_steps=[
                    "Verify torque specifications are exactly correct",
                    "Check that safety warnings are complete and accurate",
                    "Ensure no potentially dangerous omissions"
                ],
                threshold=safety_threshold,
                model=judge_model,
            ),
        }
    except ImportError:
        return None


# =============================================================================
# Aggregated Evaluation
# =============================================================================

class VLMEvaluator:
    """
    Aggregated evaluator combining all metrics.

    Provides a unified interface for running all metrics on a single example.
    """

    def __init__(self, use_llm_judge: bool = True):
        # Automated metrics
        self.rouge_l = RougeL()
        self.numeric_match = NumericExactMatch()
        self.unit_consistency = UnitConsistency()
        self.keyword_presence = KeywordPresence()

        # Domain-specific metrics
        self.safety_accuracy = SafetyCriticalAccuracy()
        self.visual_grounding = VisualGrounding()
        self.procedure_ordering = ProcedureStepOrdering()

        # LLM-as-judge metrics (optional)
        self.use_llm_judge = use_llm_judge
        self.deepeval_metrics = None
        if use_llm_judge:
            self.deepeval_metrics = get_deepeval_metrics()

    def evaluate(
        self,
        prediction: str,
        reference: str,
        question: str,
        question_type: str = None,
        content_type: str = None,
        keywords: list[str] = None,
        is_visual_only: bool = False,
        is_safety_critical: bool = False,
    ) -> dict[str, MetricResult]:
        """
        Run all applicable metrics on a single example.

        Returns dict mapping metric name to MetricResult.
        """
        results = {}

        # Always run ROUGE-L
        results["rouge_l"] = self.rouge_l.compute(prediction, reference)

        # Always run unit consistency
        results["unit_consistency"] = self.unit_consistency.compute(prediction)

        # Run numeric match if reference contains numbers
        if re.search(r'\d', reference):
            results["numeric_match"] = self.numeric_match.compute(prediction, reference)

        # Run keyword presence if keywords provided
        if keywords:
            results["keyword_presence"] = self.keyword_presence.compute(prediction, keywords)

        # Run visual grounding for visual questions
        if question_type == "visual" or is_visual_only:
            results["visual_grounding"] = self.visual_grounding.compute(
                prediction, question_type, is_visual_only
            )

        # Run procedure ordering for procedural questions
        if question_type == "procedural":
            results["procedure_ordering"] = self.procedure_ordering.compute(prediction, reference)

        # Run safety-critical check
        if is_safety_critical or question_type == "safety":
            results["safety_accuracy"] = self.safety_accuracy.compute(
                prediction, reference, question, question_type
            )

        return results

    def compute_aggregate_scores(self, results: list[dict[str, MetricResult]]) -> dict:
        """
        Compute aggregate scores across multiple examples.

        Returns dict with mean scores and pass rates for each metric.
        """
        if not results:
            return {}

        aggregates = {}

        # Collect all metric names
        all_metrics = set()
        for r in results:
            all_metrics.update(r.keys())

        for metric_name in all_metrics:
            scores = []
            passes = []

            for r in results:
                if metric_name in r:
                    scores.append(r[metric_name].score)
                    passes.append(r[metric_name].passed)

            if scores:
                aggregates[metric_name] = {
                    "mean_score": sum(scores) / len(scores),
                    "pass_rate": sum(passes) / len(passes),
                    "count": len(scores),
                }

        return aggregates
