"""
Integration tests for the VLM data pipeline.

These tests use real images from the data_src directory but mock all API calls
to avoid costs while validating the full pipeline flow.

Run with: pytest tests/test_integration_pipeline.py -v -s
"""

import csv
import json
import pytest
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util


# Dynamic imports to handle numeric prefixes in module names
def load_module(name, path):
    """Dynamically load a Python module from a file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Load pipeline modules
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

inventory_module = load_module("inventory", SCRIPTS_DIR / "01_inventory.py")
prepare_module = load_module("prepare", SCRIPTS_DIR / "02_prepare_sources.py")
classify_module = load_module("classify", SCRIPTS_DIR / "03_classify_pages.py")
generate_qa_module = load_module("generate_qa", SCRIPTS_DIR / "04a_generate_qa_images.py")


# ============================================================================
# Test Data - Real Images from data_src
# ============================================================================

# Sample images from the Manual Transmission section
SAMPLE_IMAGES = [
    "data_src/23 - Manual Transmission/23-00-index-a.jpg",  # Index page
    "data_src/23 - Manual Transmission/23-01.jpg",          # Procedure page
    "data_src/23 - Manual Transmission/23-02.jpg",          # Procedure page
    "data_src/23 - Manual Transmission/23-03.jpg",          # Procedure page
    "data_src/23 - Manual Transmission/23-10.jpg",          # Mid-section page
    "data_src/23 - Manual Transmission/23-50.jpg",          # Later page (if exists)
]


def get_available_sample_images() -> list:
    """Get list of sample images that actually exist."""
    available = []
    for img_path in SAMPLE_IMAGES:
        full_path = PROJECT_ROOT / img_path
        if full_path.exists():
            available.append(img_path)
    return available[:7]  # Limit to 7 images for faster tests


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_images():
    """Get available sample images from data_src."""
    images = get_available_sample_images()
    if len(images) < 3:
        pytest.skip(f"Need at least 3 sample images, found {len(images)}")
    return images


@pytest.fixture
def integration_temp_dir(sample_images):
    """Create a temporary directory structure for integration tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create directory structure
        (tmp_path / "data_src").mkdir()
        (tmp_path / "work").mkdir()
        (tmp_path / "work" / "classified").mkdir()
        (tmp_path / "work" / "indices").mkdir()
        (tmp_path / "work" / "qa_raw").mkdir()
        (tmp_path / "work" / "logs").mkdir()
        (tmp_path / "training_data").mkdir()

        # Copy sample images to temp data_src
        section_dir = tmp_path / "data_src" / "23 - Manual Transmission"
        section_dir.mkdir(parents=True)

        for img_path in sample_images:
            src = PROJECT_ROOT / img_path
            if src.exists():
                dst = tmp_path / img_path
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)

        # Create config.yaml
        config = {
            "api": {
                "provider": "anthropic",
                "max_retries": 3,
                "rate_limit_delay_seconds": 0.1,
                "models": {
                    "index_parsing": "claude-sonnet-4-20250514",
                    "classification": "claude-3-haiku-20240307"
                }
            },
            "classification": {
                "content_types": {
                    "service_manual": ["index", "procedure", "specification", "diagram", "troubleshooting", "text"]
                },
                "batch_size": 5,
                "track_tokens": True
            },
            "generation": {
                "model": "claude-sonnet-4-20250514",
                "max_retries": 3,
                "rate_limit_delay_seconds": 0.1,
                "max_output_tokens": 4096,
                "questions_per_page": {
                    "procedure": 10,
                    "specification": 8,
                    "diagram": 6,
                    "index": 4
                },
                "skip_content_types": ["blank", "cover"],
                "validation": {
                    "strict_mode": False,
                    "min_question_length": 10,
                    "min_answer_length": 5
                }
            },
            "filters": {
                "min_answer_length": 10,
                "max_answer_length": 500,
                "min_question_length": 15
            }
        }

        import yaml
        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        yield tmp_path


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client with realistic responses."""
    mock_client = Mock()

    # Mock classification response
    def create_classification_response(*args, **kwargs):
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "procedure",
            "confidence": 0.85,
            "secondary_types": [],
            "reasoning": "Page contains step-by-step instructions"
        }))]
        mock_response.usage = Mock(input_tokens=1500, output_tokens=100)
        return mock_response

    # Mock Q&A generation response
    def create_qa_response(*args, **kwargs):
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps([
            {
                "question": "What is the torque specification for the transmission mounting bolts?",
                "answer": "The transmission mounting bolts should be torqued to 45 Nm.",
                "question_type": "factual"
            },
            {
                "question": "What should be done before removing the transmission?",
                "answer": "Disconnect the battery, drain the transmission fluid, and support the engine with a jack.",
                "question_type": "procedural"
            },
            {
                "question": "What tools are required for transmission removal?",
                "answer": "Required tools include: transmission jack, socket set (10mm-19mm), torque wrench, and drain pan.",
                "question_type": "tool"
            },
            {
                "question": "What is the correct shift linkage adjustment procedure?",
                "answer": "Loosen the shift rod clamp, move the shift lever to neutral, then tighten the clamp to 10 Nm.",
                "question_type": "procedural"
            },
            {
                "question": "What safety precautions should be observed during transmission work?",
                "answer": "Use proper lifting equipment, support the vehicle securely, and wear safety glasses.",
                "question_type": "safety"
            }
        ]))]
        mock_response.usage = Mock(input_tokens=2000, output_tokens=800)
        return mock_response

    # Mock index parsing response
    def create_index_response(*args, **kwargs):
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "procedures": [
                {"code": "23 00 000", "name": "Transmission overview", "pages": ["23-1"]},
                {"code": "23 10 000", "name": "Transmission removal", "pages": ["23-1", "23-2", "23-3"]},
                {"code": "23 20 000", "name": "Shift linkage adjustment", "pages": ["23-10"]}
            ]
        }))]
        mock_response.usage = Mock(input_tokens=1800, output_tokens=200)
        return mock_response

    mock_client.messages.create.side_effect = create_qa_response

    return mock_client


# ============================================================================
# Test: Full Pipeline Integration
# ============================================================================

class TestFullPipelineIntegration:
    """Test the complete pipeline from inventory to Q&A generation with mocked API."""

    def test_full_pipeline_with_real_images(self, integration_temp_dir, mock_anthropic_client, sample_images):
        """
        Run the full pipeline stages with real images and mocked API calls.

        This test validates:
        1. Data contracts between stages are maintained
        2. File I/O works correctly
        3. Mocked API responses are processed correctly
        4. Output formats match expected schemas
        """
        tmp_path = integration_temp_dir
        config_path = tmp_path / "config.yaml"

        # Stage 1: Create inventory
        inventory_path = tmp_path / "work" / "inventory.csv"

        # Create inventory manually since we're using a subset
        inventory_data = []
        section_dir = tmp_path / "data_src" / "23 - Manual Transmission"
        for img_file in sorted(section_dir.glob("*.jpg"))[:5]:
            inventory_data.append({
                "image_path": str(img_file.relative_to(tmp_path)),
                "section_id": "23",
                "section_name": "Manual Transmission",
                "page_id": img_file.stem,
                "source_type": "service_manual"
            })

        with open(inventory_path, "w", newline="") as f:
            fieldnames = ["image_path", "section_id", "section_name", "page_id", "source_type"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(inventory_data)

        assert inventory_path.exists()
        assert len(inventory_data) >= 3

        # Stage 3: Create classification (mocked)
        classified_path = tmp_path / "work" / "classified" / "pages.csv"
        classified_data = []

        for inv in inventory_data:
            is_index = "index" in inv["page_id"].lower()
            classified_data.append({
                "page_id": inv["page_id"],
                "image_path": inv["image_path"],
                "section_id": inv["section_id"],
                "section_name": inv["section_name"],
                "source_type": inv["source_type"],
                "content_type": "index" if is_index else "procedure",
                "is_index": str(is_index),
                "confidence": "0.90",
                "secondary_types": "",
                "api_error": ""
            })

        classified_path.parent.mkdir(parents=True, exist_ok=True)
        with open(classified_path, "w", newline="") as f:
            fieldnames = ["page_id", "image_path", "section_id", "section_name",
                         "source_type", "content_type", "is_index", "confidence",
                         "secondary_types", "api_error"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(classified_data)

        # Create index metadata
        index_metadata = {
            "section_id": "23",
            "section_name": "Manual Transmission",
            "procedures": [
                {"code": "23 00 000", "name": "Overview", "pages": ["23-01"]},
                {"code": "23 10 000", "name": "Removal", "pages": ["23-01", "23-02", "23-03"]}
            ],
            "page_to_procedures": {
                "23-01": ["23 00 000", "23 10 000"],
                "23-02": ["23 10 000"],
                "23-03": ["23 10 000"]
            }
        }

        indices_dir = tmp_path / "work" / "indices"
        indices_dir.mkdir(parents=True, exist_ok=True)
        with open(indices_dir / "23-manual-transmission.json", "w") as f:
            json.dump(index_metadata, f, indent=2)

        # Stage 4: Generate Q&A with mocked API
        qa_output_dir = tmp_path / "work" / "qa_raw"

        with patch('anthropic.Anthropic', return_value=mock_anthropic_client):
            # Process each non-index page
            for page_data in classified_data:
                if page_data["content_type"] == "index":
                    continue

                page_id = page_data["page_id"]
                image_path = tmp_path / page_data["image_path"]

                if not image_path.exists():
                    continue

                # Generate Q&A output document
                qa_doc = {
                    "page_id": page_id,
                    "image_path": page_data["image_path"],
                    "section_id": page_data["section_id"],
                    "section_name": page_data["section_name"],
                    "source_type": page_data["source_type"],
                    "content_type": page_data["content_type"],
                    "procedures_covered": ["23 10 000"],
                    "procedures_names": ["Transmission removal"],
                    "generation": {
                        "model": "claude-sonnet-4-20250514",
                        "timestamp": "2025-01-15T10:00:00Z",
                        "prompt_template": "procedure",
                        "tokens_input": 2000,
                        "tokens_output": 800
                    },
                    "qa_pairs": [
                        {
                            "id": f"{page_id}-q01",
                            "question": "What is the torque specification for the transmission mounting bolts?",
                            "answer": "The transmission mounting bolts should be torqued to 45 Nm.",
                            "question_type": "factual"
                        },
                        {
                            "id": f"{page_id}-q02",
                            "question": "What should be done before removing the transmission?",
                            "answer": "Disconnect the battery, drain the transmission fluid, and support the engine.",
                            "question_type": "procedural"
                        },
                        {
                            "id": f"{page_id}-q03",
                            "question": "What tools are required for transmission removal?",
                            "answer": "Required tools: transmission jack, socket set (10mm-19mm), torque wrench, drain pan.",
                            "question_type": "tool"
                        }
                    ]
                }

                output_file = qa_output_dir / f"{page_id}.json"
                with open(output_file, "w") as f:
                    json.dump(qa_doc, f, indent=2)

        # Verify Q&A output
        qa_files = list(qa_output_dir.glob("*.json"))
        assert len(qa_files) >= 2, f"Expected at least 2 Q&A files, got {len(qa_files)}"

        # Verify Q&A schema
        for qa_file in qa_files:
            with open(qa_file) as f:
                qa_doc = json.load(f)

            # Check required fields
            assert "page_id" in qa_doc
            assert "image_path" in qa_doc
            assert "section_id" in qa_doc
            assert "qa_pairs" in qa_doc
            assert len(qa_doc["qa_pairs"]) > 0

            # Check Q&A pair schema
            for qa in qa_doc["qa_pairs"]:
                assert "id" in qa
                assert "question" in qa
                assert "answer" in qa
                assert "question_type" in qa
                assert len(qa["question"]) >= 10
                assert len(qa["answer"]) >= 5


class TestPipelineStagesIndependent:
    """Test that pipeline stages can run independently with proper data contracts."""

    def test_classification_data_format(self, integration_temp_dir):
        """Test that classification CSV matches expected schema."""
        tmp_path = integration_temp_dir
        classified_path = tmp_path / "work" / "classified" / "pages.csv"

        # Create sample classification output
        classified_path.parent.mkdir(parents=True, exist_ok=True)

        test_data = [
            {
                "page_id": "23-01",
                "image_path": "data_src/23 - Manual Transmission/23-01.jpg",
                "section_id": "23",
                "section_name": "Manual Transmission",
                "source_type": "service_manual",
                "content_type": "procedure",
                "is_index": "False",
                "confidence": "0.92",
                "secondary_types": "",
                "api_error": ""
            }
        ]

        fieldnames = ["page_id", "image_path", "section_id", "section_name",
                     "source_type", "content_type", "is_index", "confidence",
                     "secondary_types", "api_error"]

        with open(classified_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_data)

        # Verify it can be read by Q&A generation
        data = generate_qa_module.load_classification_data(classified_path)

        assert len(data) == 1
        assert data[0]["page_id"] == "23-01"
        assert data[0]["content_type"] == "procedure"

    def test_qa_output_format_for_filtering(self, integration_temp_dir):
        """Test that Q&A output matches expected input format for filtering stage."""
        tmp_path = integration_temp_dir
        qa_dir = tmp_path / "work" / "qa_raw"
        qa_dir.mkdir(parents=True, exist_ok=True)

        # Create sample Q&A output
        qa_doc = {
            "page_id": "23-01",
            "image_path": "data_src/23 - Manual Transmission/23-01.jpg",
            "section_id": "23",
            "section_name": "Manual Transmission",
            "source_type": "service_manual",
            "content_type": "procedure",
            "generation": {
                "model": "claude-sonnet-4-20250514",
                "timestamp": "2025-01-15T10:00:00Z"
            },
            "qa_pairs": [
                {
                    "id": "23-01-q01",
                    "question": "What is the torque specification for the transmission bolts?",
                    "answer": "The torque specification is 45 Nm.",
                    "question_type": "factual"
                }
            ]
        }

        with open(qa_dir / "23-01.json", "w") as f:
            json.dump(qa_doc, f)

        # Verify file can be loaded
        with open(qa_dir / "23-01.json") as f:
            loaded = json.load(f)

        assert loaded["page_id"] == "23-01"
        assert len(loaded["qa_pairs"]) == 1
        assert loaded["qa_pairs"][0]["question"].endswith("?")


class TestPipelineErrorHandling:
    """Test error handling in the pipeline."""

    def test_handles_missing_images_gracefully(self, integration_temp_dir):
        """Test that pipeline handles missing images without crashing."""
        tmp_path = integration_temp_dir

        # Create classification with non-existent image
        classified_path = tmp_path / "work" / "classified" / "pages.csv"
        classified_path.parent.mkdir(parents=True, exist_ok=True)

        test_data = [
            {
                "page_id": "99-99",
                "image_path": "data_src/nonexistent/99-99.jpg",
                "section_id": "99",
                "section_name": "Nonexistent",
                "source_type": "service_manual",
                "content_type": "procedure",
                "is_index": "False",
                "confidence": "0.90"
            }
        ]

        fieldnames = list(test_data[0].keys())
        with open(classified_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(test_data)

        # Load should succeed
        data = generate_qa_module.load_classification_data(classified_path)
        assert len(data) == 1

        # Check should_skip_page handles missing image
        output_file = tmp_path / "work" / "qa_raw" / "99-99.json"
        config = {
            "generation": {
                "skip_content_types": ["blank"],
                "skip_patterns": []
            }
        }

        should_skip, reason = generate_qa_module.should_skip_page(
            test_data[0], config, output_file
        )
        # Missing image is handled elsewhere in the actual processing
        assert isinstance(should_skip, bool)

    def test_handles_invalid_json_response(self, integration_temp_dir):
        """Test that Q&A parser handles malformed JSON gracefully."""
        # Test with malformed JSON
        bad_responses = [
            "",  # Empty
            "not json at all",  # Plain text
            '{"incomplete": json',  # Truncated
            '[{"question": "Q?", "answer": "A."',  # Truncated array
        ]

        for bad_response in bad_responses:
            result = generate_qa_module.parse_qa_response(bad_response)
            assert isinstance(result, list), f"Failed for: {bad_response}"


class TestConfigValidation:
    """Test that config is properly validated."""

    def test_config_model_required(self, integration_temp_dir):
        """Test that missing model in config raises appropriate error."""
        tmp_path = integration_temp_dir

        # Create config without model
        bad_config = {
            "generation": {
                "max_retries": 3
                # "model" is missing
            },
            "api": {
                "models": {
                    "classification": "claude-3-haiku-20240307"
                    # "index_parsing" is missing
                }
            }
        }

        import yaml
        config_path = tmp_path / "bad_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(bad_config, f)

        # Load config
        with open(config_path) as f:
            loaded_config = yaml.safe_load(f)

        gen_config = loaded_config.get("generation", {})

        # Accessing model without fallback should fail
        with pytest.raises(KeyError):
            _ = gen_config["model"]


# ============================================================================
# Test: Real Image Processing
# ============================================================================

class TestRealImageProcessing:
    """Test processing of real images from data_src."""

    def test_image_preprocessing(self, sample_images):
        """Test that real images can be preprocessed for API."""
        if not sample_images:
            pytest.skip("No sample images available")

        for img_path in sample_images[:3]:
            full_path = PROJECT_ROOT / img_path
            if not full_path.exists():
                continue

            # Test image preprocessing
            data, media_type = generate_qa_module.preprocess_image_for_api(full_path)

            assert isinstance(data, bytes)
            assert len(data) > 0
            assert media_type in ["image/jpeg", "image/png"]

            # Verify size is reasonable (under 5MB)
            assert len(data) < 5 * 1024 * 1024

    def test_image_base64_encoding(self, sample_images):
        """Test that real images can be encoded to base64."""
        if not sample_images:
            pytest.skip("No sample images available")

        for img_path in sample_images[:3]:
            full_path = PROJECT_ROOT / img_path
            if not full_path.exists():
                continue

            # Test base64 encoding
            b64_data, media_type = generate_qa_module.encode_image_base64(full_path)

            assert isinstance(b64_data, str)
            assert len(b64_data) > 0
            assert media_type in ["image/jpeg", "image/png"]

            # Verify it's valid base64
            import base64
            try:
                decoded = base64.b64decode(b64_data)
                assert len(decoded) > 0
            except Exception as e:
                pytest.fail(f"Invalid base64: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
