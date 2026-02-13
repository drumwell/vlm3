"""
Shared test fixtures for all forum pipeline test modules.

Automatically available to all tests in tests/.
"""

import copy
import json
import sys
import tempfile

import pytest
import yaml
from pathlib import Path

# Add project root to path so we can import the pipeline package
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Raw Post Fixtures
# ============================================================================

@pytest.fixture
def sample_post():
    """A single raw forum post."""
    return {
        "post_id": "1258841",
        "thread_id": "1258841",
        "author": "Split_S",
        "author_id": "",
        "content_html": "<div>test</div>",
        "content_text": "#1 Valve Adjustment 05-29-2017, 11:37 AM This is a detailed post about valve shim adjustment on the S14 engine with specific measurements and procedures.",
        "created_at": "2017-05-29T15:37:50-07:00",
        "post_number": 1,
        "is_first_post": True,
        "images": [
            "https://s6.postimg.org/xrdaz63pd/IMG_1988.jpg",
            "https://example.com/member_sponsors/logo.jpg",
        ],
        "quotes": [],
        "thread_title": "Valve Adjustment",
    }


@pytest.fixture
def sample_reply():
    """A reply post in a thread."""
    return {
        "post_id": "1258900",
        "thread_id": "1258841",
        "author": "BimmerDude",
        "author_id": "",
        "content_html": "<div>reply</div>",
        "content_text": "#2 Great writeup! The clearance spec 0.28-0.33mm is correct. I measured mine at 0.35mm and needed new shims.",
        "created_at": "2017-05-30T10:00:00-07:00",
        "post_number": 2,
        "is_first_post": False,
        "images": [],
        "quotes": [],
        "thread_title": "Valve Adjustment",
    }


@pytest.fixture
def sample_noise_reply():
    """A social noise reply that should be filtered."""
    return {
        "post_id": "1258901",
        "thread_id": "1258841",
        "author": "Lurker",
        "author_id": "",
        "content_html": "<div>thanks</div>",
        "content_text": "thanks",
        "created_at": "2017-05-31T12:00:00-07:00",
        "post_number": 3,
        "is_first_post": False,
        "images": [],
        "quotes": [],
        "thread_title": "Valve Adjustment",
    }


# ============================================================================
# Thread Metadata Fixtures
# ============================================================================

@pytest.fixture
def sample_thread_metadata():
    """Thread metadata record."""
    return {
        "thread_id": "1258841",
        "title": "Valve Adjustment",
        "url": "https://s14net.vbulletin.net/forum/s14/topics-of-interest/d-i-y/1258841-valve-adjustment",
        "forum_id": "d-i-y",
        "author": "",
        "author_id": "",
        "created_at": "",
        "reply_count": 0,
        "view_count": 0,
        "last_post_at": "",
        "last_poster": "",
        "is_sticky": False,
        "is_locked": False,
        "page_count": 1,
    }


# ============================================================================
# Reconstructed Thread Fixtures
# ============================================================================

@pytest.fixture
def sample_thread():
    """A reconstructed thread (output of Stage 1)."""
    return {
        "thread_id": "1258841",
        "thread_title": "Valve Adjustment",
        "forum": "d-i-y",
        "author": "Split_S",
        "post_count": 3,
        "first_post_date": "2017-05-29T15:37:50-07:00",
        "last_post_date": "2017-05-31T12:00:00-07:00",
        "is_sticky": False,
        "is_locked": False,
        "posts": [
            {
                "post_id": "1258841",
                "author": "Split_S",
                "post_number": 1,
                "is_first_post": True,
                "content_text": "This is a detailed post about valve shim adjustment on the S14 engine. BMW spec is 0.28-0.33mm cold. Use a 36mm socket on the crank bolt for rotation.",
                "content_images": ["https://s6.postimg.org/xrdaz63pd/IMG_1988.jpg"],
                "quotes": [],
            },
            {
                "post_id": "1258900",
                "author": "BimmerDude",
                "post_number": 2,
                "is_first_post": False,
                "content_text": "Great writeup! The clearance spec 0.28-0.33mm is correct. I measured mine at 0.35mm and needed new shims.",
                "content_images": [],
                "quotes": [],
            },
            {
                "post_id": "1258901",
                "author": "Lurker",
                "post_number": 3,
                "is_first_post": False,
                "content_text": "Thanks for the info! Very helpful guide for S14 valve work.",
                "content_images": [],
                "quotes": [],
            },
        ],
    }


@pytest.fixture
def sample_classified_thread(sample_thread):
    """A classified thread (output of Stage 3)."""
    thread = copy.deepcopy(sample_thread)
    thread["classification"] = {
        "thread_type": "how_to",
        "technical_depth": "high",
        "s14_specific": True,
        "has_resolution": True,
        "confidence": 0.92,
        "timestamp": "2025-01-15T10:00:00Z",
    }
    return thread


@pytest.fixture
def sample_scored_thread(sample_classified_thread):
    """A scored thread (output of Stage 4)."""
    thread = copy.deepcopy(sample_classified_thread)
    thread["scores"] = {
        "factual_specificity": 4,
        "answer_completeness": 4,
        "procedural_clarity": 5,
        "s14_relevance": 5,
        "unique_knowledge": 3,
        "composite": 4.15,
        "timestamp": "2025-01-15T11:00:00Z",
    }
    return thread


# ============================================================================
# Q&A Document Fixtures
# ============================================================================

@pytest.fixture
def sample_qa_doc():
    """A forum Q&A document (output of Stage 5)."""
    return {
        "thread_id": "1258841",
        "forum": "d-i-y",
        "thread_title": "Valve Adjustment",
        "source_type": "forum",
        "content_type": "how_to",
        "quality_score": 4.15,
        "qa_pairs": [
            {
                "id": "forum-1258841-q01",
                "question": "What is the valve shim clearance specification for the BMW S14 engine?",
                "answer": "The BMW S14 valve shim clearance specification is 0.28mm to 0.33mm, measured cold. Shims come in gradations of 0.05mm. To calculate the required shim thickness: add the measured valve gap to the measured shim thickness, then subtract the desired gap.",
                "question_type": "specification",
                "cross_reference_needed": True,
                "cross_reference_note": "Verify 0.28-0.33mm spec against service manual Section 11",
                "factual_confidence": "plausible",
            },
            {
                "id": "forum-1258841-q02",
                "question": "What tools are needed for an S14 valve shim adjustment?",
                "answer": "You need a 36mm socket for crankshaft rotation, 10mm socket for the valve cover, feeler gauges (angled tips recommended), a Skyway Tools #1007 Double Ended Tappet Depressor, compressed air with nozzle, and a magnetic pickup tool for removing shims.",
                "question_type": "procedural",
                "cross_reference_needed": False,
                "cross_reference_note": None,
                "factual_confidence": "plausible",
            },
        ],
    }


@pytest.fixture
def sample_validated_qa_doc(sample_qa_doc):
    """A validated Q&A document (output of Stage 6)."""
    return sample_qa_doc


# ============================================================================
# VLM Record Fixtures
# ============================================================================

@pytest.fixture
def sample_vlm_record():
    """A VLM-formatted record (output of Stage 7)."""
    return {
        "conversations": [
            {"role": "user", "content": "What is the valve shim clearance specification for the BMW S14 engine?"},
            {"role": "assistant", "content": "The BMW S14 valve shim clearance specification is 0.28mm to 0.33mm, measured cold."},
        ],
        "metadata": {
            "source": "forum",
            "source_id": "forum-1258841-q01",
            "thread_id": "1258841",
            "forum": "d-i-y",
            "content_type": "how_to",
            "question_type": "specification",
            "quality_score": 4.15,
            "factual_confidence": "plausible",
        },
    }


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def raw_dir_with_posts(tmp_path, sample_post, sample_reply, sample_noise_reply, sample_thread_metadata):
    """Temporary raw directory with sample JSONL files."""
    raw = tmp_path / "raw"
    raw.mkdir()

    # Write posts
    with open(raw / "posts_d-i-y.jsonl", "w") as f:
        for post in [sample_post, sample_reply, sample_noise_reply]:
            f.write(json.dumps(post) + "\n")

    # Write thread metadata
    with open(raw / "threads_d-i-y.jsonl", "w") as f:
        f.write(json.dumps(sample_thread_metadata) + "\n")

    return raw


@pytest.fixture
def threads_dir(tmp_path, sample_thread):
    """Temporary directory with reconstructed threads."""
    threads = tmp_path / "threads"
    threads.mkdir()

    with open(threads / "1258841.json", "w") as f:
        json.dump(sample_thread, f)

    return threads


@pytest.fixture
def classified_threads_dir(tmp_path, sample_classified_thread):
    """Temporary directory with classified threads."""
    classified = tmp_path / "threads_classified"
    classified.mkdir()

    with open(classified / "1258841.json", "w") as f:
        json.dump(sample_classified_thread, f)

    return classified


@pytest.fixture
def scored_threads_dir(tmp_path, sample_scored_thread):
    """Temporary directory with scored threads."""
    scored = tmp_path / "threads_scored"
    scored.mkdir()

    with open(scored / "1258841.json", "w") as f:
        json.dump(sample_scored_thread, f)

    return scored


@pytest.fixture
def qa_raw_dir(tmp_path, sample_qa_doc):
    """Temporary directory with raw Q&A documents."""
    qa = tmp_path / "qa_raw"
    qa.mkdir()

    with open(qa / "1258841.json", "w") as f:
        json.dump(sample_qa_doc, f)

    return qa


@pytest.fixture
def qa_validated_dir(tmp_path, sample_validated_qa_doc):
    """Temporary directory with validated Q&A documents."""
    qa = tmp_path / "qa_validated"
    qa.mkdir()

    with open(qa / "1258841.json", "w") as f:
        json.dump(sample_validated_qa_doc, f)

    return qa


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def forum_config():
    """Forum pipeline configuration dict."""
    return {
        "forums": {"targets": ["d-i-y", "no-start"]},
        "api": {
            "provider": "anthropic",
            "max_retries": 3,
            "rate_limit_delay_seconds": 0.0,
            "models": {
                "classification": "claude-haiku-4-5-20251001",
                "scoring": "claude-sonnet-4-5-20250929",
                "extraction": "claude-sonnet-4-5-20250929",
            },
        },
        "filtering": {
            "min_post_length": 30,
            "min_first_post_length": 50,
            "min_thread_posts": 2,
            "social_noise_patterns": ["thanks", "bump", "+1", "subscribed", "great post"],
            "title_noise_patterns": ["WTB", "WTS", "FS:", "for sale"],
            "strip_post_number_markers": True,
            "strip_timestamps": True,
            "strip_signatures": True,
            "normalize_whitespace": True,
            "strip_html_artifacts": True,
        },
        "classification": {
            "max_preview_chars": 2000,
            "max_replies_for_preview": 3,
            "min_technical_depth": "medium",
            "require_s14_specific": True,
        },
        "scoring": {
            "max_thread_chars": 4000,
            "dimensions": {
                "factual_specificity": 0.30,
                "answer_completeness": 0.25,
                "procedural_clarity": 0.20,
                "s14_relevance": 0.15,
                "unique_knowledge": 0.10,
            },
            "min_composite_score": 3.0,
        },
        "extraction": {
            "max_qa_per_thread": 5,
            "max_thread_chars": 8000,
        },
        "cross_reference": {
            "manual_train_path": "../manual/prepared/manual_train.jsonl",
            "similarity_threshold": 0.85,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_batch_size": 64,
        },
        "filters": {
            "min_answer_length": 50,
            "max_answer_length": 2000,
            "min_question_length": 15,
            "generic_answer_patterns": ["it depends", "check the manual", "not sure"],
            "vague_question_patterns": ["what do you think", "any thoughts"],
        },
        "output": {
            "train_split": 0.90,
            "random_seed": 42,
            "prefix": "forum",
            "stratify_by": "forum",
        },
    }


@pytest.fixture
def config_file(tmp_path, forum_config):
    """Temporary config YAML file."""
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(forum_config, f)
    return config_path
