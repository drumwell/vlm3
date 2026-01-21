"""
Pytest/DeepEval configuration and fixtures for VLM evaluation.
"""

import json
import os
from pathlib import Path

import pytest


# =============================================================================
# Paths
# =============================================================================

EVAL_DIR = Path(__file__).parent
PROJECT_ROOT = EVAL_DIR.parent
TRAINING_DATA_DIR = PROJECT_ROOT / "training_data"
BENCHMARKS_DIR = EVAL_DIR / "benchmarks"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def training_data_dir():
    """Return training data directory."""
    return TRAINING_DATA_DIR


@pytest.fixture(scope="session")
def eval_sample_path():
    """Return path to evaluation sample."""
    return EVAL_DIR / "eval_sample.jsonl"


@pytest.fixture(scope="session")
def manual_probes_path():
    """Return path to manual probes."""
    return BENCHMARKS_DIR / "manual_probes.json"


@pytest.fixture(scope="session")
def eval_sample(eval_sample_path):
    """Load evaluation sample records."""
    if not eval_sample_path.exists():
        pytest.skip(f"Evaluation sample not found: {eval_sample_path}")

    records = []
    with open(eval_sample_path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


@pytest.fixture(scope="session")
def manual_probes(manual_probes_path):
    """Load manual probe test cases."""
    if not manual_probes_path.exists():
        pytest.skip(f"Manual probes not found: {manual_probes_path}")

    with open(manual_probes_path, "r") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def vlm_wrapper():
    """
    Create VLM wrapper for testing.

    Uses mock wrapper if no GPU available or MOCK_VLM env var is set.
    """
    from eval.model_wrapper import create_wrapper

    use_mock = os.environ.get("MOCK_VLM", "0") == "1"

    if use_mock:
        return create_wrapper(mock=True)

    # Try to create real wrapper, fall back to mock
    try:
        import torch
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            print("No GPU available, using mock VLM wrapper")
            return create_wrapper(mock=True)

        return create_wrapper()
    except ImportError:
        print("PyTorch not available, using mock VLM wrapper")
        return create_wrapper(mock=True)


@pytest.fixture(scope="session")
def evaluator():
    """Create VLM evaluator."""
    from eval.metrics import VLMEvaluator
    return VLMEvaluator(use_llm_judge=False)


# =============================================================================
# DeepEval Configuration
# =============================================================================

@pytest.fixture(scope="session")
def deepeval_model():
    """Configure DeepEval to use Claude as judge."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set, skipping LLM-as-judge tests")

    try:
        from deepeval.models import AnthropicModel
        return AnthropicModel(model="claude-sonnet-4-20250514")
    except ImportError:
        pytest.skip("deepeval not installed")


# =============================================================================
# Test Helpers
# =============================================================================

def extract_question_answer(record: dict) -> tuple[str, str]:
    """Extract question and reference answer from record."""
    conversations = record.get("conversations", [])
    question = ""
    answer = ""

    for turn in conversations:
        if turn.get("role") == "user":
            content = turn.get("content", "")
            if isinstance(content, str):
                question = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        question = item.get("text", "")
        elif turn.get("role") == "assistant":
            answer = turn.get("content", "")

    return question, answer


@pytest.fixture
def question_answer_extractor():
    """Return function to extract Q&A from records."""
    return extract_question_answer


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "llm_judge: marks tests requiring LLM-as-judge"
    )
    config.addinivalue_line(
        "markers",
        "gpu: marks tests requiring GPU"
    )
