"""
Shared test fixtures for all test modules.

These fixtures are automatically available to all tests in the tests/ directory.
"""

import json
import sys
import pytest
import tempfile
from pathlib import Path

# Add project root to path so we can import the scripts package
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Q&A Document Fixtures (Stage 5 → Stage 6 interface)
# ============================================================================

@pytest.fixture
def sample_qa_doc():
    """Sample Q&A document matching Stage 5 output schema."""
    return {
        "page_id": "21-03_clutch",
        "image_path": "data_src/21 - Clutch/21-03.jpg",
        "section_id": "21",
        "section_name": "Clutch",
        "source_type": "service_manual",
        "content_type": "procedure",
        "procedures_covered": ["21 21 000"],
        "procedures_names": ["Clutch disc - remove and install"],
        "generation": {
            "model": "claude-sonnet-4-20250514",
            "timestamp": "2025-01-15T10:30:00Z",
            "prompt_template": "procedure"
        },
        "qa_pairs": [
            {
                "id": "21-03_clutch-q01",
                "question": "What should I visually inspect the clutch pressure plate for?",
                "answer": "Visually inspect the clutch for cracks, wear, and burnt spots.",
                "question_type": "inspection"
            },
            {
                "id": "21-03_clutch-q02",
                "question": "What is the torque specification for the pressure plate bolts?",
                "answer": "Tighten the pressure plate bolts to 25 Nm in a cross pattern.",
                "question_type": "factual"
            }
        ]
    }


@pytest.fixture
def sample_html_qa_doc():
    """Sample Q&A document for HTML source (no image)."""
    return {
        "page_id": "html-m3-techspec",
        "image_path": None,
        "section_id": "techspec",
        "section_name": "Technical Specifications",
        "source_type": "html_specs",
        "content_type": "specification",
        "generation": {
            "method": "html_parse",
            "timestamp": "2025-01-15T10:30:00Z"
        },
        "qa_pairs": [
            {
                "id": "html-m3-q01",
                "question": "What is the engine displacement for the BMW E30 M3?",
                "answer": "2302 cc",
                "question_type": "factual"
            }
        ]
    }


@pytest.fixture
def sample_vlm_record():
    """Sample VLM-formatted record."""
    return {
        "image": "images/21-03_clutch.jpg",
        "conversations": [
            {"role": "user", "content": "What should I inspect?"},
            {"role": "assistant", "content": "Inspect for cracks and wear."}
        ],
        "metadata": {
            "page_id": "21-03_clutch",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "procedure",
            "question_type": "inspection",
            "qa_id": "21-03_clutch-q01"
        }
    }


@pytest.fixture
def sample_vlm_record_no_image():
    """Sample VLM-formatted record without image (HTML source)."""
    return {
        "image": None,
        "conversations": [
            {"role": "user", "content": "What is the engine displacement for the BMW E30 M3?"},
            {"role": "assistant", "content": "2302 cc"}
        ],
        "metadata": {
            "page_id": "html-m3-techspec",
            "section_id": "techspec",
            "section_name": "Technical Specifications",
            "source_type": "html_specs",
            "content_type": "specification",
            "question_type": "factual",
            "qa_id": "html-m3-q01"
        }
    }


# ============================================================================
# Temporary Directory Fixtures
# ============================================================================

@pytest.fixture
def temp_qa_dir(sample_qa_doc, sample_html_qa_doc):
    """Temporary directory with sample Q&A JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qa_dir = Path(tmpdir) / "qa_unique"
        qa_dir.mkdir()

        # Write sample docs
        with open(qa_dir / "21-03_clutch.json", "w") as f:
            json.dump(sample_qa_doc, f)

        with open(qa_dir / "html-m3-techspec.json", "w") as f:
            json.dump(sample_html_qa_doc, f)

        yield qa_dir


@pytest.fixture
def temp_image_dir():
    """Temporary directory with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "data_src" / "21 - Clutch"
        img_dir.mkdir(parents=True)

        # Create a minimal valid JPEG using PIL
        try:
            from PIL import Image
            img = Image.new("RGB", (800, 600), color="white")
            img.save(img_dir / "21-03.jpg", "JPEG")
        except ImportError:
            # Fallback: create a minimal JPEG manually
            # This is a minimal valid JPEG (1x1 white pixel)
            minimal_jpeg = bytes([
                0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
                0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
                0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
                0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
                0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
                0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
                0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
                0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
                0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00,
                0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01, 0x03,
                0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01, 0x7D,
                0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06,
                0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08,
                0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72,
                0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x25, 0x26, 0x27, 0x28,
                0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45,
                0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
                0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
                0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
                0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
                0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6,
                0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9,
                0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2,
                0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4,
                0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
                0x00, 0x00, 0x3F, 0x00, 0xFB, 0xD5, 0xDB, 0x20, 0xA8, 0xF7, 0xFF, 0xD9
            ])
            with open(img_dir / "21-03.jpg", "wb") as f:
                f.write(minimal_jpeg)

        yield Path(tmpdir) / "data_src"


@pytest.fixture
def temp_output_dir():
    """Temporary output directory for training_data/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def sample_emit_config():
    """Sample emit configuration dict."""
    return {
        "output": {
            "train_split": 0.9,
            "random_seed": 42,
            "image_copy_mode": "symlink",
            "image_output_dir": "images",
            "normalize_image_names": True,
            "stratify_by": "section_id",
            "min_stratum_size": 10
        }
    }


@pytest.fixture
def sample_validation_config():
    """Sample validation configuration dict."""
    return {
        "vlm_validation": {
            "require_image_field": True,
            "require_conversations": True,
            "require_metadata": True,
            "check_image_exists": True,
            "check_image_readable": True,
            "min_image_width": 100,
            "min_image_height": 100,
            "min_question_length": 10,
            "min_answer_length": 5,
            "max_answer_length": 1000,
            "min_qa_per_section": 5,
            "max_qa_per_section": 2000,
            "num_samples_per_split": 5
        }
    }


@pytest.fixture
def sample_upload_config():
    """Sample upload configuration dict."""
    return {
        "upload": {
            "repo_id": "test/test-dataset",
            "private": False,
            "commit_message": "Test upload",
            "dataset_card": {
                "language": "en",
                "license": "cc-by-nc-4.0",
                "task_categories": ["visual-question-answering"],
                "tags": ["automotive", "bmw", "vlm"]
            }
        }
    }


@pytest.fixture
def sample_config(sample_emit_config, sample_validation_config, sample_upload_config):
    """Combined sample configuration."""
    config = {}
    config.update(sample_emit_config)
    config.update(sample_validation_config)
    config.update(sample_upload_config)
    return config


@pytest.fixture
def temp_config_file(sample_config):
    """Temporary config YAML file."""
    import yaml
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(sample_config, f)
        yield config_path


# ============================================================================
# JSONL Fixtures
# ============================================================================

@pytest.fixture
def temp_jsonl_files(sample_vlm_record, sample_vlm_record_no_image):
    """Temporary directory with sample JSONL files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create train JSONL
        train_records = [sample_vlm_record] * 9
        with open(data_dir / "vlm_train.jsonl", "w") as f:
            for rec in train_records:
                f.write(json.dumps(rec) + "\n")

        # Create val JSONL
        val_records = [sample_vlm_record_no_image]
        with open(data_dir / "vlm_val.jsonl", "w") as f:
            for rec in val_records:
                f.write(json.dumps(rec) + "\n")

        # Create images directory
        images_dir = data_dir / "images"
        images_dir.mkdir()

        # Create placeholder image
        try:
            from PIL import Image
            img = Image.new("RGB", (800, 600), color="white")
            img.save(images_dir / "21-03_clutch.jpg", "JPEG")
        except ImportError:
            # Create empty file as fallback
            (images_dir / "21-03_clutch.jpg").touch()

        yield data_dir
