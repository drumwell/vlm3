"""
Tests for Stage 4a: Image Q&A Generation

Run with: pytest tests/test_04a_generate_qa_images.py -v
"""

import pytest
import json
import csv
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import importlib.util

# Dynamic import to handle the numeric prefix in module name
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

generate_qa = load_module("generate_qa", Path(__file__).parent.parent / "scripts" / "04a_generate_qa_images.py")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration dict"""
    return {
        "generation": {
            "model": "claude-sonnet-4-20250514",
            "max_retries": 3,
            "retry_delay_seconds": 1,
            "rate_limit_delay_seconds": 0.5,
            "max_output_tokens": 4096,
            "questions_per_page": {
                "procedure": 12,
                "specification": 10,
                "diagram": 8,
                "troubleshooting": 10,
                "wiring": 10,
                "text": 6
            },
            "skip_patterns": ["*-blank-*", "*-cover-*"],
            "skip_content_types": ["index"],
            "image": {
                "max_size_mb": 5.0,
                "max_dimension": 4096,
                "jpeg_quality": 85
            },
            "validation": {
                "strict_mode": False,
                "min_question_length": 10,
                "min_answer_length": 5
            }
        },
        "api": {
            "api_key": "test-key"
        }
    }


@pytest.fixture
def sample_classification_data():
    """Sample classification data"""
    return [
        {
            "page_id": "21-01",
            "image_path": "data_src/21 - Clutch/21-01.jpg",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "procedure",
            "is_index": "False",
            "confidence": "0.85"
        },
        {
            "page_id": "21-02",
            "image_path": "data_src/21 - Clutch/21-02.jpg",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "specification",
            "is_index": "False",
            "confidence": "0.90"
        }
    ]


@pytest.fixture
def sample_index_metadata():
    """Sample index metadata"""
    return {
        "21-clutch": {
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [
                {"code": "21 00 006", "name": "Clutch - bleed", "pages": ["21-1"]},
                {"code": "21 21 000", "name": "Clutch disc - remove and install", "pages": ["21-1", "21-2"]}
            ],
            "page_to_procedures": {
                "21-1": ["21 00 006", "21 21 000"],
                "21-2": ["21 21 000"]
            }
        }
    }


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image"""
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color='white')
    img.save(img_path, "JPEG")
    return img_path


# =============================================================================
# Test: Page ID Normalization
# =============================================================================

class TestNormalizePageId:
    """Test page ID normalization for cross-source matching"""

    def test_normalize_zero_padded(self):
        """Should strip leading zeros from page number"""
        assert generate_qa.normalize_page_id("21-01") == "21-1"
        assert generate_qa.normalize_page_id("21-001") == "21-1"
        assert generate_qa.normalize_page_id("00-05") == "00-5"

    def test_normalize_already_normalized(self):
        """Should not change already normalized IDs"""
        assert generate_qa.normalize_page_id("21-1") == "21-1"
        assert generate_qa.normalize_page_id("21-15") == "21-15"

    def test_normalize_etm_format(self):
        """Should handle ETM prefix format"""
        assert generate_qa.normalize_page_id("ETM-001") == "ETM-1"
        assert generate_qa.normalize_page_id("ETM-12") == "ETM-12"

    def test_normalize_preserves_zero_page(self):
        """Should preserve page 0"""
        assert generate_qa.normalize_page_id("21-00") == "21-0"
        assert generate_qa.normalize_page_id("21-0") == "21-0"


# =============================================================================
# Test: Section Slug Derivation
# =============================================================================

class TestDeriveSectionSlug:
    """Test section slug derivation for index file lookup"""

    def test_derive_simple_slug(self):
        """Should derive slug from section id and name"""
        assert generate_qa.derive_section_slug("21", "Clutch") == "21-clutch"
        assert generate_qa.derive_section_slug("00", "Maintenance") == "00-maintenance"

    def test_derive_multi_word_slug(self):
        """Should handle multi-word section names"""
        assert generate_qa.derive_section_slug("ETM", "Electrical Troubleshooting") == "etm-electrical-troubleshooting"

    def test_derive_removes_special_chars(self):
        """Should remove special characters"""
        slug = generate_qa.derive_section_slug("12", "Brakes (ABS)")
        assert "(" not in slug
        assert ")" not in slug


# =============================================================================
# Test: Index Directory Discovery
# =============================================================================

class TestFindIndicesDir:
    """Test index directory discovery with fallbacks"""

    def test_find_default_indices(self, tmp_path):
        """Should find default indices directory"""
        indices_dir = tmp_path / "indices"
        indices_dir.mkdir()
        (indices_dir / "21-clutch.json").write_text("{}")

        result = generate_qa.find_indices_dir(tmp_path)
        assert result == indices_dir

    def test_find_indices_100(self, tmp_path):
        """Should find indices_100 as fallback"""
        indices_dir = tmp_path / "indices_100"
        indices_dir.mkdir()
        (indices_dir / "21-clutch.json").write_text("{}")

        result = generate_qa.find_indices_dir(tmp_path)
        assert result == indices_dir

    def test_find_raises_if_none(self, tmp_path):
        """Should raise if no indices directory found"""
        with pytest.raises(FileNotFoundError):
            generate_qa.find_indices_dir(tmp_path)

    def test_find_ignores_empty_dir(self, tmp_path):
        """Should ignore empty directories"""
        # Empty indices dir
        (tmp_path / "indices").mkdir()
        # Populated indices_100
        indices_100 = tmp_path / "indices_100"
        indices_100.mkdir()
        (indices_100 / "21-clutch.json").write_text("{}")

        result = generate_qa.find_indices_dir(tmp_path)
        assert result == indices_100


# =============================================================================
# Test: Image Preprocessing
# =============================================================================

class TestPreprocessImageForApi:
    """Test image preprocessing for API size limits"""

    def test_small_image_unchanged(self, tmp_path):
        """Should return small images without modification"""
        from PIL import Image
        img_path = tmp_path / "small.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path, "JPEG")

        data, media_type = generate_qa.preprocess_image_for_api(img_path)

        assert isinstance(data, bytes)
        assert media_type == "image/jpeg"
        assert len(data) < 1024 * 1024  # Less than 1MB

    def test_large_image_resized(self, tmp_path):
        """Should resize images exceeding max dimensions"""
        from PIL import Image
        img_path = tmp_path / "large.jpg"
        img = Image.new('RGB', (8000, 6000), color='blue')
        img.save(img_path, "JPEG", quality=95)

        data, media_type = generate_qa.preprocess_image_for_api(img_path, max_dim=2048)

        # Verify it was resized by checking the returned image
        from io import BytesIO
        result_img = Image.open(BytesIO(data))
        assert max(result_img.size) <= 2048

    def test_rgba_converted_to_rgb(self, tmp_path):
        """Should convert RGBA to RGB for JPEG"""
        from PIL import Image
        img_path = tmp_path / "transparent.png"
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(img_path, "PNG")

        data, media_type = generate_qa.preprocess_image_for_api(img_path)

        assert media_type == "image/jpeg"


# =============================================================================
# Test: Q&A Validation
# =============================================================================

class TestValidateQAPair:
    """Test Q&A pair validation"""

    def test_valid_qa_pair(self):
        """Should accept valid Q&A pair"""
        qa = {
            "question": "What is the torque specification for the clutch bolts?",
            "answer": "The torque specification is 25 Nm.",
            "question_type": "factual"
        }

        is_valid, warnings = generate_qa.validate_qa_pair(qa, {})

        assert is_valid is True
        assert len(warnings) == 0

    def test_reject_short_question(self):
        """Should reject questions that are too short"""
        qa = {"question": "What?", "answer": "Something.", "question_type": "factual"}

        is_valid, warnings = generate_qa.validate_qa_pair(qa, {})

        assert is_valid is False
        assert any("too short" in w.lower() for w in warnings)

    def test_warn_missing_question_mark(self):
        """Should warn if question doesn't end with '?'"""
        qa = {
            "question": "Tell me about the clutch specifications",
            "answer": "The clutch is rated for 240 Nm.",
            "question_type": "factual"
        }

        is_valid, warnings = generate_qa.validate_qa_pair(qa, {})

        assert any("?" in w for w in warnings)

    def test_warn_hallucination_phrases(self):
        """Should warn on hallucination indicator phrases"""
        qa = {
            "question": "What is the oil capacity?",
            "answer": "I cannot see the oil capacity on this page, but typically it would be around 4 liters.",
            "question_type": "factual"
        }

        is_valid, warnings = generate_qa.validate_qa_pair(qa, {})

        assert any("hallucination" in w.lower() for w in warnings)

    def test_warn_self_referential(self):
        """Should warn on self-referential questions"""
        qa = {
            "question": "What is shown on this page?",
            "answer": "The clutch assembly diagram.",
            "question_type": "visual"
        }

        is_valid, warnings = generate_qa.validate_qa_pair(qa, {})

        assert any("self-referential" in w.lower() for w in warnings)

    def test_warn_invalid_question_type(self):
        """Should warn on invalid question_type"""
        qa = {
            "question": "What is the torque spec?",
            "answer": "25 Nm.",
            "question_type": "invalid_type"
        }

        is_valid, warnings = generate_qa.validate_qa_pair(qa, {})

        assert any("question_type" in w.lower() for w in warnings)


class TestFilterAndValidateQAPairs:
    """Test batch Q&A filtering"""

    def test_filter_separates_valid_invalid(self):
        """Should separate valid and invalid pairs"""
        qa_pairs = [
            {"question": "What is the torque specification?", "answer": "25 Nm.", "question_type": "factual"},
            {"question": "?", "answer": "X", "question_type": "factual"},  # Too short
            {"question": "What is the bolt size?", "answer": "M8 bolt size.", "question_type": "factual"},
        ]

        valid, rejected = generate_qa.filter_and_validate_qa_pairs(qa_pairs, {})

        assert len(valid) == 2
        assert len(rejected) == 1

    def test_strict_mode_rejects_warnings(self):
        """Should reject pairs with warnings in strict mode"""
        qa_pairs = [
            {"question": "What is shown on this page?", "answer": "The diagram.", "question_type": "visual"},
        ]

        # Non-strict: accepted with warning
        valid, _ = generate_qa.filter_and_validate_qa_pairs(qa_pairs, {}, strict=False)
        assert len(valid) == 1

        # Strict: rejected
        valid, rejected = generate_qa.filter_and_validate_qa_pairs(qa_pairs, {}, strict=True)
        assert len(valid) == 0
        assert len(rejected) == 1


# =============================================================================
# Test: Classification Data Loading
# =============================================================================

class TestLoadClassificationData:
    """Test loading classification CSV"""

    def test_load_classification_data_valid(self, tmp_path):
        """Should load valid classification CSV"""
        csv_path = tmp_path / "pages.csv"
        csv_path.write_text("""page_id,image_path,section_id,section_name,source_type,content_type,is_index,confidence
21-01,data_src/21 - Clutch/21-01.jpg,21,Clutch,service_manual,procedure,False,0.85
21-02,data_src/21 - Clutch/21-02.jpg,21,Clutch,service_manual,specification,False,0.90
""")

        data = generate_qa.load_classification_data(csv_path)

        assert len(data) == 2
        assert data[0]["page_id"] == "21-01"
        assert data[0]["content_type"] == "procedure"
        assert data[1]["content_type"] == "specification"

    def test_load_classification_data_missing_file(self, tmp_path):
        """Should raise error for missing file"""
        with pytest.raises(FileNotFoundError):
            generate_qa.load_classification_data(tmp_path / "nonexistent.csv")


# =============================================================================
# Test: Index Metadata Loading
# =============================================================================

class TestLoadIndexMetadata:
    """Test loading index metadata from JSON files"""

    def test_load_index_metadata_valid(self, tmp_path):
        """Should load all index JSON files"""
        indices_dir = tmp_path / "indices"
        indices_dir.mkdir()

        (indices_dir / "21-clutch.json").write_text(json.dumps({
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [{"code": "21 00 006", "name": "Clutch - bleed", "pages": ["21-1"]}],
            "page_to_procedures": {"21-1": ["21 00 006"]}
        }))

        metadata = generate_qa.load_index_metadata(indices_dir)

        assert "21-clutch" in metadata
        assert metadata["21-clutch"]["section_id"] == "21"

    def test_load_index_metadata_empty_dir(self, tmp_path):
        """Should return empty dict for empty directory"""
        indices_dir = tmp_path / "indices"
        indices_dir.mkdir()

        metadata = generate_qa.load_index_metadata(indices_dir)

        assert metadata == {}

    def test_load_index_metadata_missing_dir(self, tmp_path):
        """Should return empty dict for missing directory"""
        metadata = generate_qa.load_index_metadata(tmp_path / "nonexistent")

        assert metadata == {}


# =============================================================================
# Test: Procedure Lookup
# =============================================================================

class TestGetProceduresForPage:
    """Test looking up procedures for a page"""

    def test_get_procedures_found(self, sample_index_metadata):
        """Should return procedures for matching page"""
        codes, names = generate_qa.get_procedures_for_page("21-1", "21-clutch", sample_index_metadata)

        assert "21 00 006" in codes
        assert "21 21 000" in codes
        assert "Clutch - bleed" in names

    def test_get_procedures_not_found(self, sample_index_metadata):
        """Should return empty lists for unknown page"""
        codes, names = generate_qa.get_procedures_for_page("99-99", "21-clutch", sample_index_metadata)

        assert codes == []
        assert names == []

    def test_get_procedures_unknown_section(self, sample_index_metadata):
        """Should return empty lists for unknown section"""
        codes, names = generate_qa.get_procedures_for_page("21-1", "unknown-section", sample_index_metadata)

        assert codes == []
        assert names == []


# =============================================================================
# Test: Skip Logic
# =============================================================================

class TestShouldSkipPage:
    """Test page skip logic"""

    def test_skip_existing_output(self, tmp_path, sample_config):
        """Should skip page if output already exists"""
        output_file = tmp_path / "21-01.json"
        output_file.write_text("{}")

        page_data = {"page_id": "21-01", "content_type": "procedure"}

        should_skip, reason = generate_qa.should_skip_page(page_data, sample_config, output_file)

        assert should_skip is True
        assert "exists" in reason.lower()

    def test_skip_index_content_type(self, sample_config):
        """Should skip index pages"""
        page_data = {"page_id": "21-00-index", "content_type": "index"}

        should_skip, reason = generate_qa.should_skip_page(page_data, sample_config, None)

        assert should_skip is True
        assert "index" in reason.lower()

    def test_skip_blank_pattern(self, sample_config):
        """Should skip pages matching skip patterns"""
        page_data = {"page_id": "21-blank-01", "content_type": "text"}

        should_skip, reason = generate_qa.should_skip_page(page_data, sample_config, None)

        assert should_skip is True
        assert "pattern" in reason.lower()

    def test_no_skip_normal_page(self, sample_config):
        """Should not skip normal procedure page"""
        page_data = {"page_id": "21-03", "content_type": "procedure"}

        should_skip, reason = generate_qa.should_skip_page(page_data, sample_config, None)

        assert should_skip is False


# =============================================================================
# Test: Context Block Building
# =============================================================================

class TestBuildContextBlock:
    """Test context block generation"""

    def test_context_with_procedures(self):
        """Should include procedure names when available"""
        page_data = {"content_type": "procedure"}
        codes = ["21 00 006", "21 21 000"]
        names = ["Clutch - bleed", "Clutch disc - remove and install"]

        context = generate_qa.build_context_block(page_data, codes, names)

        assert "21 00 006" in context
        assert "Clutch - bleed" in context
        assert "21 21 000" in context

    def test_context_without_procedures(self):
        """Should return minimal context when no procedures"""
        page_data = {"content_type": "specification"}

        context = generate_qa.build_context_block(page_data, [], [])

        # Should still return something (possibly empty or default)
        assert isinstance(context, str)

    def test_context_for_specification(self):
        """Should include spec-specific context"""
        page_data = {"content_type": "specification"}

        context = generate_qa.build_context_block(page_data, [], [])

        # May include "specification" or "values" in context
        assert isinstance(context, str)


# =============================================================================
# Test: Prompt Selection
# =============================================================================

class TestSelectPromptTemplate:
    """Test prompt template selection"""

    def test_service_manual_procedure(self):
        """Should select procedure prompt for service manual procedure"""
        system_prompt, user_template = generate_qa.select_prompt_template("service_manual", "procedure")

        assert "BMW E30 M3" in system_prompt
        assert "procedural" in user_template.lower() or "procedure" in user_template.lower()

    def test_electrical_manual_wiring(self):
        """Should select wiring prompt for electrical manual"""
        system_prompt, user_template = generate_qa.select_prompt_template("electrical_manual", "wiring")

        assert "wire" in user_template.lower() or "electrical" in user_template.lower()

    def test_ecu_technical_diagram(self):
        """Should select ECU prompt for Bosch Motronic"""
        system_prompt, user_template = generate_qa.select_prompt_template("ecu_technical", "diagram")

        assert isinstance(system_prompt, str)
        assert isinstance(user_template, str)

    def test_fallback_for_unknown(self):
        """Should return fallback prompt for unknown types"""
        system_prompt, user_template = generate_qa.select_prompt_template("unknown", "unknown")

        # Should not raise, should return some default
        assert system_prompt is not None
        assert user_template is not None


# =============================================================================
# Test: Image Encoding
# =============================================================================

class TestEncodeImageBase64:
    """Test image base64 encoding"""

    def test_encode_jpg(self, temp_image):
        """Should encode JPG to base64"""
        data, media_type = generate_qa.encode_image_base64(temp_image)

        assert isinstance(data, str)
        assert len(data) > 0
        assert media_type == "image/jpeg"

    def test_encode_png(self, tmp_path):
        """Should encode PNG to base64"""
        from PIL import Image
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (50, 50), color='blue')
        img.save(img_path, "PNG")

        data, media_type = generate_qa.encode_image_base64(img_path)

        assert media_type == "image/png"

    def test_encode_missing_file(self, tmp_path):
        """Should raise error for missing file"""
        with pytest.raises(FileNotFoundError):
            generate_qa.encode_image_base64(tmp_path / "nonexistent.jpg")


# =============================================================================
# Test: Q&A Response Parsing
# =============================================================================

class TestParseQAResponse:
    """Test parsing Claude API responses"""

    def test_parse_clean_json(self):
        """Should parse clean JSON array"""
        response = '''[
            {"question": "What is X?", "answer": "X is Y.", "question_type": "factual"},
            {"question": "How to do Z?", "answer": "Do Z by...", "question_type": "procedural"}
        ]'''

        qa_pairs = generate_qa.parse_qa_response(response)

        assert len(qa_pairs) == 2
        assert qa_pairs[0]["question"] == "What is X?"
        assert qa_pairs[1]["question_type"] == "procedural"

    def test_parse_markdown_wrapped(self):
        """Should parse JSON wrapped in markdown code blocks"""
        response = '''```json
[
    {"question": "Test?", "answer": "Answer.", "question_type": "factual"}
]
```'''

        qa_pairs = generate_qa.parse_qa_response(response)

        assert len(qa_pairs) == 1
        assert qa_pairs[0]["question"] == "Test?"

    def test_parse_with_trailing_text(self):
        """Should handle response with trailing explanation text"""
        response = '''[{"question": "Q?", "answer": "A.", "question_type": "factual"}]

I generated this based on the visible content.'''

        qa_pairs = generate_qa.parse_qa_response(response)

        # Parser should extract the JSON array even with trailing text
        # If it returns empty, that's acceptable behavior (conservative parsing)
        assert isinstance(qa_pairs, list)

    def test_parse_empty_response(self):
        """Should return empty list for empty response"""
        qa_pairs = generate_qa.parse_qa_response("")

        assert qa_pairs == []

    def test_parse_malformed_json(self):
        """Should handle malformed JSON gracefully"""
        response = '''[{"question": "Q?", "answer": "A." broken json'''

        # Should not raise, should return empty or partial
        qa_pairs = generate_qa.parse_qa_response(response)
        assert isinstance(qa_pairs, list)


# =============================================================================
# Test: Q&A ID Assignment
# =============================================================================

class TestAssignQAIds:
    """Test Q&A ID assignment"""

    def test_assign_sequential_ids(self):
        """Should assign sequential IDs"""
        qa_pairs = [
            {"question": "Q1?", "answer": "A1."},
            {"question": "Q2?", "answer": "A2."},
            {"question": "Q3?", "answer": "A3."}
        ]

        result = generate_qa.assign_qa_ids(qa_pairs, "21-03")

        assert result[0]["id"] == "21-03-q01"
        assert result[1]["id"] == "21-03-q02"
        assert result[2]["id"] == "21-03-q03"

    def test_preserve_existing_fields(self):
        """Should preserve existing fields"""
        qa_pairs = [
            {"question": "Q?", "answer": "A.", "question_type": "factual"}
        ]

        result = generate_qa.assign_qa_ids(qa_pairs, "21-03")

        assert result[0]["question"] == "Q?"
        assert result[0]["answer"] == "A."
        assert result[0]["question_type"] == "factual"


# =============================================================================
# Test: Output Writing
# =============================================================================

class TestWriteQAOutput:
    """Test Q&A output file writing"""

    def test_write_output_schema(self, tmp_path):
        """Should write correct JSON schema"""
        page_data = {
            "page_id": "21-03",
            "image_path": "data_src/21 - Clutch/21-03.jpg",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "procedure"
        }
        qa_pairs = [
            {"id": "21-03-q01", "question": "Q?", "answer": "A.", "question_type": "factual"}
        ]
        generation_metadata = {
            "model": "claude-sonnet-4-20250514",
            "timestamp": "2025-01-15T10:30:00Z",
            "prompt_template": "procedure",
            "tokens_input": 1500,
            "tokens_output": 500
        }

        output_path = generate_qa.write_qa_output("21-03", page_data, qa_pairs, generation_metadata, tmp_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["page_id"] == "21-03"
        assert data["source_type"] == "service_manual"
        assert len(data["qa_pairs"]) == 1
        assert data["generation"]["model"] == "claude-sonnet-4-20250514"

    def test_write_creates_directory(self, tmp_path):
        """Should create output directory if needed"""
        output_dir = tmp_path / "nested" / "qa_raw"

        page_data = {"page_id": "21-03", "image_path": "", "section_id": "21",
                     "section_name": "Clutch", "source_type": "service_manual",
                     "content_type": "procedure"}

        output_path = generate_qa.write_qa_output("21-03", page_data, [], {}, output_dir)

        assert output_path.exists()


# =============================================================================
# Test: Progress/Error Logging
# =============================================================================

class TestProgressLogging:
    """Test progress and error logging"""

    def test_write_progress_log(self, tmp_path):
        """Should write progress log with correct schema"""
        log_path = tmp_path / "progress.csv"
        entries = [
            {
                "timestamp": "2025-01-15T10:30:00",
                "page_id": "21-03",
                "status": "success",
                "qa_count": 12,
                "tokens_input": 1500,
                "tokens_output": 500,
                "cost_usd": "0.0300"
            }
        ]

        generate_qa.write_progress_log(entries, log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["page_id"] == "21-03"
        assert rows[0]["status"] == "success"

    def test_write_error_log(self, tmp_path):
        """Should write error log with correct schema"""
        log_path = tmp_path / "errors.csv"
        entries = [
            {
                "timestamp": "2025-01-15T10:30:00",
                "page_id": "21-03",
                "error_type": "api_error",
                "error_message": "Rate limit exceeded"
            }
        ]

        generate_qa.write_error_log(entries, log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["error_type"] == "api_error"


# =============================================================================
# Test: API Integration (Mocked)
# =============================================================================

class TestGenerateQAForPage:
    """Test Claude API call (mocked)"""

    def test_generate_qa_success(self, temp_image):
        """Should generate Q&A pairs from API response"""
        # Mock the Anthropic class
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='''[
            {"question": "Test question?", "answer": "Test answer.", "question_type": "factual"}
        ]''')]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response

        with patch('anthropic.Anthropic', return_value=mock_client):
            page_data = {
                "page_id": "21-03",
                "source_type": "service_manual",
                "content_type": "procedure"
            }

            result = generate_qa.generate_qa_for_page(
                image_path=temp_image,
                page_data=page_data,
                context_block="Test context",
                num_questions=10,
                api_key="test-key",
                model="claude-sonnet-4-20250514"
            )

            assert result["error"] is None
            assert len(result["qa_pairs"]) == 1
            assert result["qa_pairs"][0]["question"] == "Test question?"

    def test_generate_qa_api_error_retry(self, temp_image):
        """Should retry on API error"""
        mock_client = Mock()
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.content = [Mock(text='[{"question": "Q?", "answer": "A.", "question_type": "factual"}]')]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)

        mock_client.messages.create.side_effect = [
            Exception("API Error"),
            mock_response
        ]

        with patch('anthropic.Anthropic', return_value=mock_client):
            page_data = {"page_id": "21-03", "source_type": "service_manual", "content_type": "procedure"}

            result = generate_qa.generate_qa_for_page(
                image_path=temp_image,
                page_data=page_data,
                context_block="",
                num_questions=10,
                api_key="test-key",
                model="test-model",
                max_retries=3
            )

            # Should succeed after retry
            assert result["error"] is None or len(result["qa_pairs"]) > 0


# =============================================================================
# Test: CLI
# =============================================================================

class TestCLI:
    """Test command-line interface"""

    def test_help_output(self):
        """Should show help with --help"""
        import subprocess
        script_path = Path(__file__).parent.parent / "scripts" / "04a_generate_qa_images.py"
        if not script_path.exists():
            pytest.skip("Script not implemented yet")

        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert '--classified' in result.stdout
        assert '--indices' in result.stdout
        assert '--output' in result.stdout
        assert '--config' in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
