# Stage 3: Classification & Index Parsing - Implementation Spec (TDD)

## Overview

**Script**: `scripts/03_classify_pages.py`

**Purpose**: Classify each page by content type and source type, extract structured metadata from index pages, and prepare context for Q&A generation.

**Architecture Reference**: See `pipeline_rearchitecture.md` lines 377-437 for full architectural context.

---

## Requirements Summary

### Inputs
- `work/inventory_prepared.csv` (from Stage 2)
- `config.yaml` (classification rules, API settings, source patterns)
- `data_src/` directory (for reading images)

### Outputs
- `work/classified/pages.csv` (all image pages with classification metadata)
- `work/indices/{section_id}-{section_name_slug}.json` (one per section with index data)
- `work/logs/classification_errors.csv` (failed classifications)
- `work/logs/stage3_classification_report.md` (summary report)

### Key Behaviors
1. **Source Type Detection**: Identify source material type from directory structure
2. **Index Page Detection**: Identify index/TOC pages by filename patterns
3. **Index Parsing**: Extract repair codes, procedure names, and page mappings using Claude API (Sonnet)
4. **Content Classification**: Classify pages using Claude Vision API (Haiku for cost efficiency)
5. **Context Preparation**: Create metadata for Q&A generation in Stage 4
6. **Error Handling**: Graceful failures; log and continue processing

---

## Test-Driven Development Approach

### Phase 1: Write Tests First

**Test File**: `tests/test_03_classify_pages.py`

Create tests for all functions BEFORE implementing them. Run tests and watch them fail. Then implement to make them pass.

#### Test Suite Structure

```python
import pytest
from pathlib import Path
import json
import csv
from unittest.mock import Mock, patch, MagicMock

# Import functions under test (will fail initially)
from scripts.classify_pages_03 import (
    detect_source_type,
    is_index_page,
    parse_index_page,
    classify_page_content,
    generate_page_id,
    generate_section_slug,  # NEW: section slug generation
    load_config,
    load_cached_results,    # NEW: cache loading for resume
    read_prepared_inventory,
    process_classification,
    merge_index_metadata,
    write_classification_csv,
    write_index_json,
    write_error_log,
    generate_report
)


class TestSourceTypeDetection:
    """Test source type detection from directory names"""

    def test_service_manual_numbered_section(self):
        """Should detect service manual from numbered section"""
        # Input: "21 - Clutch"
        # Expected: "service_manual"
        assert detect_source_type("21 - Clutch", "21-01.jpg") == "service_manual"
        assert detect_source_type("00 - Maintenance", "00-01.jpg") == "service_manual"
        assert detect_source_type("97 - Troubleshooting", "97-01.jpg") == "service_manual"

    def test_service_manual_getrag(self):
        """Should detect Getrag PDF conversion as service manual"""
        assert detect_source_type("Getrag265", "001.jpg") == "service_manual"

    def test_electrical_manual(self):
        """Should detect electrical troubleshooting manual"""
        assert detect_source_type(
            "1990 BMW M3 Electrical Troubleshooting Manual",
            "etm-001.jpg"
        ) == "electrical_manual"

    def test_ecu_technical(self):
        """Should detect Bosch Motronic ECU docs"""
        assert detect_source_type(
            "Bosch Motronic ML 3-1",
            "bosch-001.jpg"
        ) == "ecu_technical"

    def test_unknown_source(self):
        """Should return 'unknown' for unrecognized directories"""
        assert detect_source_type("Random Folder", "file.jpg") == "unknown"

    def test_html_specs(self):
        """Should detect HTML spec files"""
        assert detect_source_type("", "M3-techspec.html") == "html_specs"
        assert detect_source_type("", "320is-techspec.html") == "html_specs"


class TestIndexPageDetection:
    """Test index page identification by filename patterns"""

    def test_index_page_dash_index(self):
        """Should detect *-index-* pattern"""
        assert is_index_page("21-00-index-a.jpg") == True
        assert is_index_page("21-00-index-b.jpg") == True
        assert is_index_page("00-index.jpg") == True

    def test_index_page_toc(self):
        """Should detect *-toc-* pattern"""
        assert is_index_page("21-toc.jpg") == True
        assert is_index_page("section-toc-page1.jpg") == True

    def test_non_index_page(self):
        """Should return False for regular pages"""
        assert is_index_page("21-01.jpg") == False
        assert is_index_page("21-02.jpg") == False
        assert is_index_page("clutch-diagram.jpg") == False

    def test_case_insensitive(self):
        """Should be case-insensitive"""
        assert is_index_page("21-INDEX-A.jpg") == True
        assert is_index_page("21-TOC.jpg") == True


class TestIndexParsing:
    """Test index page parsing with Claude API"""

    @patch('anthropic.Anthropic')
    def test_parse_index_page_single_page(self, mock_anthropic):
        """Should extract repair codes and page mappings from index"""
        # Mock Claude API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [
                {
                    "code": "21 00 006",
                    "name": "Clutch - bleed",
                    "page": "21-1"
                },
                {
                    "code": "21 11 000",
                    "name": "Clutch housing - remove and install",
                    "page": "21-1"
                }
            ]
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = parse_index_page(
            image_path=Path("data_src/21 - Clutch/21-00-index-a.jpg"),
            section_dir="21 - Clutch",
            api_key="test_key",
            model="claude-sonnet-4"
        )

        assert result["section_id"] == "21"
        assert result["section_name"] == "Clutch"
        assert len(result["procedures"]) == 2
        assert result["procedures"][0]["code"] == "21 00 006"

    @patch('anthropic.Anthropic')
    def test_parse_index_page_multi_page(self, mock_anthropic):
        """Should handle multi-page indices by returning single-page result for merging later"""
        # Each index page is parsed independently; merging happens in merge_index_metadata()
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [
                {"code": "21 21 000", "name": "Clutch disc - remove and install", "page": "21-2"}
            ]
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Parse second index page (21-00-index-b.jpg)
        result = parse_index_page(
            image_path=Path("data_src/21 - Clutch/21-00-index-b.jpg"),
            section_dir="21 - Clutch",
            api_key="test_key",
            model="claude-sonnet-4"
        )

        assert result["section_id"] == "21"
        assert len(result["procedures"]) == 1
        assert result["procedures"][0]["code"] == "21 21 000"
        # Note: merge_index_metadata() combines results from multiple index pages

    @patch('anthropic.Anthropic')
    def test_parse_index_page_api_error(self, mock_anthropic):
        """Should handle API errors gracefully"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        result = parse_index_page(
            image_path=Path("data_src/21 - Clutch/21-00-index-a.jpg"),
            section_dir="21 - Clutch",
            api_key="test_key",
            model="claude-sonnet-4"
        )

        # Should return error result
        assert result["error"] is not None
        assert "API Error" in result["error"]

    @patch('anthropic.Anthropic')
    def test_parse_index_page_malformed_response(self, mock_anthropic):
        """Should handle malformed JSON responses"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Not valid JSON")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = parse_index_page(
            image_path=Path("data_src/21 - Clutch/21-00-index-a.jpg"),
            section_dir="21 - Clutch",
            api_key="test_key",
            model="claude-sonnet-4"
        )

        assert result["error"] is not None


class TestContentClassification:
    """Test content type classification using Claude Vision API"""

    def test_classify_index_page_skip_api(self):
        """Index pages detected by filename should skip API call"""
        result = classify_page_content(
            image_path=Path("data_src/21 - Clutch/21-00-index-a.jpg"),
            source_type="service_manual",
            is_index=True,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "index"
        assert result["confidence"] >= 0.95
        assert result["api_called"] == False  # No API call needed

    @patch('anthropic.Anthropic')
    def test_classify_procedure_page_via_api(self, mock_anthropic):
        """Should classify procedure page using vision API"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "procedure",
            "confidence": 0.92,
            "reasoning": "Multiple photos with numbered callouts and step-by-step text"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/21 - Clutch/21-01.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "procedure"
        assert result["confidence"] >= 0.9
        assert result["api_called"] == True

    @patch('anthropic.Anthropic')
    def test_classify_troubleshooting_page_via_api(self, mock_anthropic):
        """Should identify troubleshooting table format"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "troubleshooting",
            "confidence": 0.95,
            "reasoning": "Three-column table with Condition/Cause/Correction format"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/21 - Clutch/21-07.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "troubleshooting"
        assert result["confidence"] >= 0.9

    @patch('anthropic.Anthropic')
    def test_classify_diagram_page_via_api(self, mock_anthropic):
        """Should identify full-page technical diagrams"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "diagram",
            "confidence": 0.98,
            "reasoning": "Full-page technical illustration/cutaway drawing"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/11 - Engine/11-50.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "diagram"
        assert result["confidence"] >= 0.9

    @patch('anthropic.Anthropic')
    def test_classify_specification_page_via_api(self, mock_anthropic):
        """Should identify specification/data tables"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "specification",
            "confidence": 0.91,
            "reasoning": "Tabular data with values, units, and dimensions"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/00 - Torque Specs/00-01.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "specification"

    @patch('anthropic.Anthropic')
    def test_classify_wiring_page_via_api(self, mock_anthropic):
        """Should identify electrical/wiring pages"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "wiring",
            "confidence": 0.93,
            "reasoning": "Fuse data chart with circuit names and values"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/1990 BMW M3 Electrical Troubleshooting Manual/0670-01.jpg"),
            source_type="electrical_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "wiring"

    @patch('anthropic.Anthropic')
    def test_classify_api_error_returns_fallback(self, mock_anthropic):
        """Should return fallback classification on API error"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/21 - Clutch/21-01.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "unknown"
        assert result["confidence"] < 0.5
        assert result["error"] is not None

    @patch('anthropic.Anthropic')
    def test_classify_malformed_response_returns_fallback(self, mock_anthropic):
        """Should handle malformed API responses gracefully"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Not valid JSON")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/21 - Clutch/21-01.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "unknown"
        assert result["error"] is not None

    @patch('anthropic.Anthropic')
    def test_classify_mixed_content_page_returns_dominant_type(self, mock_anthropic):
        """Should return dominant content type with secondary_types for mixed pages"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "procedure",
            "confidence": 0.82,
            "secondary_types": ["specification"],
            "reasoning": "Page has 5 numbered steps with photos, plus small torque spec table"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/21 - Clutch/21-03.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )

        assert result["content_type"] == "procedure"
        assert result["secondary_types"] == ["specification"]
        assert result["confidence"] >= 0.8

    @patch('anthropic.Anthropic')
    def test_classify_returns_empty_secondary_types_for_single_type(self, mock_anthropic):
        """Should return empty secondary_types for pages with single content type"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "diagram",
            "confidence": 0.95,
            "secondary_types": [],
            "reasoning": "Full-page engine cutaway illustration"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        result = classify_page_content(
            image_path=Path("data_src/11 - Engine/11-50.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )

        assert result["content_type"] == "diagram"
        assert result["secondary_types"] == []


class TestPageIDGeneration:
    """Test page ID generation logic"""

    def test_generate_page_id_numbered_section(self):
        """Should extract page ID from numbered section filename"""
        # Input: "21-03.jpg" in "21 - Clutch"
        # Expected: "21-03"
        assert generate_page_id("21-03.jpg", "21 - Clutch") == "21-03"
        assert generate_page_id("00-01.jpg", "00 - Maintenance") == "00-01"

    def test_generate_page_id_index_page(self):
        """Should handle index page filenames"""
        assert generate_page_id("21-00-index-a.jpg", "21 - Clutch") == "21-00-index-a"

    def test_generate_page_id_electrical_manual(self):
        """Should use 'etm' prefix for electrical manual"""
        assert generate_page_id("etm-001.jpg", "1990 BMW M3 Electrical Troubleshooting Manual") == "etm-001"

    def test_generate_page_id_bosch(self):
        """Should use 'bosch' prefix for Bosch Motronic"""
        assert generate_page_id("bosch-001.jpg", "Bosch Motronic ML 3-1") == "bosch-001"

    def test_generate_page_id_getrag(self):
        """Should use 'getrag' prefix for Getrag PDF conversion"""
        assert generate_page_id("001.jpg", "Getrag265") == "getrag-001"

    def test_generate_page_id_fallback(self):
        """Should use filename stem as fallback"""
        assert generate_page_id("random-page.jpg", "Unknown Dir") == "random-page"


class TestSectionSlugGeneration:
    """Test section slug generation for filesystem-safe names"""

    def test_numbered_section(self):
        """Should handle numbered sections"""
        assert generate_section_slug("21 - Clutch") == "21-clutch"
        assert generate_section_slug("00 - Maintenance") == "00-maintenance"

    def test_multi_word_section(self):
        """Should handle multi-word section names"""
        assert generate_section_slug("00 - Torque Specs") == "00-torque-specs"
        assert generate_section_slug("12 - Engine Electrical Equipment") == "12-engine-electrical-equipment"

    def test_long_name(self):
        """Should handle long directory names"""
        assert generate_section_slug("1990 BMW M3 Electrical Troubleshooting Manual") == "1990-bmw-m3-electrical-troubleshooting-manual"

    def test_special_characters(self):
        """Should strip special characters except hyphens"""
        assert generate_section_slug("Bosch Motronic ML 3-1") == "bosch-motronic-ml-3-1"

    def test_no_spaces(self):
        """Should handle names without spaces"""
        assert generate_section_slug("Getrag265") == "getrag265"

    def test_already_slugified(self):
        """Should be idempotent"""
        assert generate_section_slug("21-clutch") == "21-clutch"


class TestCacheLoading:
    """Test loading cached classification results"""

    def test_load_cached_results_exists(self, tmp_path):
        """Should load existing cache file"""
        cache_path = tmp_path / "pages.csv"
        cache_path.write_text("""page_id,image_path,section_id,section_name,source_type,content_type,is_index,confidence,secondary_types,needs_review
21-01,data_src/21 - Clutch/21-01.jpg,21,Clutch,service_manual,procedure,False,0.85,[],False
21-02,data_src/21 - Clutch/21-02.jpg,21,Clutch,service_manual,specification,False,0.90,[],False
""")

        cached = load_cached_results(cache_path)

        assert len(cached) == 2
        assert "21-01" in cached
        assert "21-02" in cached
        assert cached["21-01"]["content_type"] == "procedure"
        assert cached["21-02"]["content_type"] == "specification"

    def test_load_cached_results_missing_file(self, tmp_path):
        """Should return empty dict for missing file"""
        cache_path = tmp_path / "nonexistent.csv"

        cached = load_cached_results(cache_path)

        assert cached == {}

    def test_load_cached_results_empty_file(self, tmp_path):
        """Should return empty dict for empty file"""
        cache_path = tmp_path / "empty.csv"
        cache_path.write_text("")

        cached = load_cached_results(cache_path)

        assert cached == {}

    def test_load_cached_results_header_only(self, tmp_path):
        """Should return empty dict for header-only file"""
        cache_path = tmp_path / "header_only.csv"
        cache_path.write_text("page_id,image_path,section_id,section_name,source_type,content_type,is_index,confidence,secondary_types,needs_review\n")

        cached = load_cached_results(cache_path)

        assert cached == {}


class TestConfigLoading:
    """Test configuration file loading"""

    def test_load_config_valid(self, tmp_path):
        """Should load valid YAML config"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
api:
  provider: anthropic
  model: claude-sonnet-4-20250514

sources:
  service_manual_patterns:
    - "^[0-9]{2} - "
    - "^Getrag"
  electrical_manual_patterns:
    - "Electrical Troubleshooting"
  ecu_technical_patterns:
    - "Bosch Motronic"

classification:
  content_types:
    service_manual: ["index", "procedure", "specification", "diagram", "troubleshooting", "text"]
    electrical_manual: ["wiring", "pinout", "flowchart", "text"]
    ecu_technical: ["diagram", "specification", "flowchart", "text"]
""")

        config = load_config(config_path)
        assert config["api"]["model"] == "claude-sonnet-4-20250514"
        assert "^[0-9]{2} - " in config["sources"]["service_manual_patterns"]

    def test_load_config_missing_file(self, tmp_path):
        """Should raise error for missing config file"""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_load_config_malformed_yaml(self, tmp_path):
        """Should raise error for malformed YAML"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("invalid: yaml: content:")

        with pytest.raises(Exception):
            load_config(config_path)


class TestInventoryReading:
    """Test reading prepared inventory from Stage 2"""

    def test_read_prepared_inventory_valid(self, tmp_path):
        """Should read valid prepared inventory CSV"""
        inventory_path = tmp_path / "inventory_prepared.csv"
        inventory_path.write_text("""file_path,file_type,section_dir,filename,original_source
data_src/21 - Clutch/21-00-index-a.jpg,jpg,21 - Clutch,21-00-index-a.jpg,
data_src/21 - Clutch/21-01.jpg,jpg,21 - Clutch,21-01.jpg,
data_src/Getrag265/001.jpg,jpg,Getrag265,001.jpg,data_src/Getrag265.pdf
""")

        records = read_prepared_inventory(inventory_path)
        assert len(records) == 3
        assert records[0]["file_path"] == "data_src/21 - Clutch/21-00-index-a.jpg"
        assert records[2]["original_source"] == "data_src/Getrag265.pdf"

    def test_read_prepared_inventory_filters_images_only(self, tmp_path):
        """Should filter to image files only (exclude HTML)"""
        inventory_path = tmp_path / "inventory_prepared.csv"
        inventory_path.write_text("""file_path,file_type,section_dir,filename,original_source
data_src/21 - Clutch/21-01.jpg,jpg,21 - Clutch,21-01.jpg,
data_src/M3-techspec.html,html,,M3-techspec.html,
""")

        records = read_prepared_inventory(inventory_path)
        assert len(records) == 1  # HTML filtered out
        assert records[0]["file_type"] == "jpg"


class TestClassificationProcessing:
    """Test main classification processing logic"""

    @patch('scripts.classify_pages_03.parse_index_page')
    @patch('scripts.classify_pages_03.classify_page_content')
    def test_process_classification_with_indices(self, mock_classify, mock_parse_index):
        """Should process pages and extract index metadata"""
        # Setup mocks
        mock_parse_index.return_value = {
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [{"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}],
            "error": None
        }
        mock_classify.return_value = {
            "content_type": "procedure",
            "confidence": 0.85
        }

        inventory_records = [
            {
                "file_path": "data_src/21 - Clutch/21-00-index-a.jpg",
                "file_type": "jpg",
                "section_dir": "21 - Clutch",
                "filename": "21-00-index-a.jpg",
                "original_source": ""
            },
            {
                "file_path": "data_src/21 - Clutch/21-01.jpg",
                "file_type": "jpg",
                "section_dir": "21 - Clutch",
                "filename": "21-01.jpg",
                "original_source": ""
            }
        ]

        config = {
            "api": {"provider": "anthropic", "model": "claude-sonnet-4", "api_key": "test"}
        }

        classification_records, index_metadata, errors = process_classification(
            inventory_records,
            config
        )

        # Should have 2 classification records
        assert len(classification_records) == 2

        # Should have 1 index metadata entry (for section 21)
        assert len(index_metadata) == 1
        assert "21-clutch" in index_metadata

        # Index page should be classified as 'index'
        index_record = [r for r in classification_records if r["is_index"]][0]
        assert index_record["content_type"] == "index"

    @patch('scripts.classify_pages_03.classify_page_content')
    def test_process_classification_handles_errors(self, mock_classify):
        """Should log errors and continue processing"""
        # First page succeeds, second fails
        mock_classify.side_effect = [
            {"content_type": "procedure", "confidence": 0.85},
            Exception("Classification failed")
        ]

        inventory_records = [
            {
                "file_path": "data_src/21 - Clutch/21-01.jpg",
                "file_type": "jpg",
                "section_dir": "21 - Clutch",
                "filename": "21-01.jpg",
                "original_source": ""
            },
            {
                "file_path": "data_src/21 - Clutch/21-02.jpg",
                "file_type": "jpg",
                "section_dir": "21 - Clutch",
                "filename": "21-02.jpg",
                "original_source": ""
            }
        ]

        config = {"api": {"provider": "anthropic", "model": "claude-sonnet-4", "api_key": "test"}}

        classification_records, index_metadata, errors = process_classification(
            inventory_records,
            config
        )

        # Should have 1 successful record
        assert len(classification_records) == 1

        # Should have 1 error logged
        assert len(errors) == 1
        assert "21-02.jpg" in errors[0]["filename"]


class TestIndexMetadataMerging:
    """Test merging of multi-page index metadata"""

    def test_merge_index_metadata_single_page(self):
        """Should handle single-page index"""
        index_results = {
            "21-clutch": {
                "section_id": "21",
                "section_name": "Clutch",
                "index_pages": ["21-00-index-a.jpg"],
                "procedures": [
                    {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}
                ]
            }
        }

        merged = merge_index_metadata(index_results)
        assert merged["21-clutch"]["section_id"] == "21"
        assert len(merged["21-clutch"]["procedures"]) == 1

    def test_merge_index_metadata_multi_page(self):
        """Should merge procedures from multiple index pages"""
        index_results = {
            "21-clutch": {
                "section_id": "21",
                "section_name": "Clutch",
                "index_pages": ["21-00-index-a.jpg", "21-00-index-b.jpg"],
                "procedures": [
                    {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"},
                    {"code": "21 11 000", "name": "Clutch housing", "page": "21-2"}
                ]
            }
        }

        merged = merge_index_metadata(index_results)
        assert len(merged["21-clutch"]["index_pages"]) == 2
        assert len(merged["21-clutch"]["procedures"]) == 2

    def test_merge_index_metadata_generates_page_mapping(self):
        """Should generate page_to_procedures mapping"""
        index_results = {
            "21-clutch": {
                "section_id": "21",
                "section_name": "Clutch",
                "index_pages": ["21-00-index-a.jpg"],
                "procedures": [
                    {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"},
                    {"code": "21 11 000", "name": "Clutch housing", "page": "21-1"},
                    {"code": "21 21 000", "name": "Clutch disc", "page": "21-2"}
                ]
            }
        }

        merged = merge_index_metadata(index_results)
        page_mapping = merged["21-clutch"]["page_to_procedures"]

        # Page 21-1 should have 2 procedures
        assert len(page_mapping["21-1"]) == 2
        assert "21 00 006" in page_mapping["21-1"]
        assert "21 11 000" in page_mapping["21-1"]

        # Page 21-2 should have 1 procedure
        assert len(page_mapping["21-2"]) == 1
        assert "21 21 000" in page_mapping["21-2"]


class TestOutputWriting:
    """Test writing output files"""

    def test_write_classification_csv_schema(self, tmp_path):
        """Should write classification CSV with correct schema"""
        output_path = tmp_path / "pages.csv"
        records = [
            {
                "page_id": "21-01",
                "image_path": "data_src/21 - Clutch/21-01.jpg",
                "section_id": "21",
                "section_name": "Clutch",
                "source_type": "service_manual",
                "content_type": "procedure",
                "is_index": False,
                "confidence": 0.85
            }
        ]

        write_classification_csv(records, output_path)

        # Read back and verify
        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["page_id"] == "21-01"
            assert rows[0]["content_type"] == "procedure"

    def test_write_classification_csv_sorted(self, tmp_path):
        """Should sort records by page_id for reproducibility"""
        output_path = tmp_path / "pages.csv"
        records = [
            {"page_id": "21-03", "image_path": "...", "section_id": "21", "section_name": "Clutch",
             "source_type": "service_manual", "content_type": "procedure", "is_index": False, "confidence": 0.85},
            {"page_id": "21-01", "image_path": "...", "section_id": "21", "section_name": "Clutch",
             "source_type": "service_manual", "content_type": "index", "is_index": True, "confidence": 0.95},
            {"page_id": "21-02", "image_path": "...", "section_id": "21", "section_name": "Clutch",
             "source_type": "service_manual", "content_type": "procedure", "is_index": False, "confidence": 0.85}
        ]

        write_classification_csv(records, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            # Should be sorted by page_id
            assert rows[0]["page_id"] == "21-01"
            assert rows[1]["page_id"] == "21-02"
            assert rows[2]["page_id"] == "21-03"

    def test_write_index_json_schema(self, tmp_path):
        """Should write index JSON with correct schema"""
        output_path = tmp_path / "21-clutch.json"
        metadata = {
            "section_id": "21",
            "section_name": "Clutch",
            "index_pages": ["21-00-index-a.jpg"],
            "procedures": [
                {
                    "code": "21 00 006",
                    "name": "Clutch - bleed",
                    "page": "21-1"
                }
            ],
            "page_to_procedures": {
                "21-1": ["21 00 006"]
            }
        }

        write_index_json(metadata, output_path)

        # Read back and verify
        with open(output_path) as f:
            data = json.load(f)
            assert data["section_id"] == "21"
            assert len(data["procedures"]) == 1
            assert "21-1" in data["page_to_procedures"]

    def test_write_error_log_schema(self, tmp_path):
        """Should write error log with correct schema"""
        log_path = tmp_path / "errors.csv"
        errors = [
            {
                "timestamp": "2025-01-15T10:30:00Z",
                "filename": "21-01.jpg",
                "operation": "classification",
                "error_message": "API timeout"
            }
        ]

        write_error_log(errors, log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == 1
            assert rows[0]["filename"] == "21-01.jpg"
            assert rows[0]["operation"] == "classification"


class TestReportGeneration:
    """Test summary report generation"""

    def test_generate_report_statistics(self, tmp_path):
        """Should generate report with correct statistics"""
        classification_records = [
            {"page_id": "21-01", "source_type": "service_manual", "content_type": "procedure", "confidence": 0.85},
            {"page_id": "21-02", "source_type": "service_manual", "content_type": "specification", "confidence": 0.90},
            {"page_id": "etm-01", "source_type": "electrical_manual", "content_type": "wiring", "confidence": 0.88}
        ]

        index_metadata = {
            "21-clutch": {"section_id": "21", "procedures": [{}, {}]},
            "11-engine": {"section_id": "11", "procedures": [{}, {}, {}]}
        }

        errors = []

        report_path = tmp_path / "report.md"
        generate_report(classification_records, index_metadata, errors, report_path)

        # Read report and verify contents
        report_text = report_path.read_text()
        assert "Total Pages Classified: 3" in report_text
        assert "service_manual" in report_text
        assert "electrical_manual" in report_text
        assert "Sections with Indices: 2" in report_text
        assert "Total Procedures Extracted: 5" in report_text

    def test_generate_report_includes_errors(self, tmp_path):
        """Should include error section if errors present"""
        classification_records = []
        index_metadata = {}
        errors = [
            {"filename": "21-01.jpg", "error_message": "API timeout"}
        ]

        report_path = tmp_path / "report.md"
        generate_report(classification_records, index_metadata, errors, report_path)

        report_text = report_path.read_text()
        assert "Errors Encountered: 1" in report_text
        assert "21-01.jpg" in report_text


class TestEndToEnd:
    """Integration tests for full classification workflow"""

    @patch('scripts.classify_pages_03.parse_index_page')
    @patch('scripts.classify_pages_03.classify_page_content')
    def test_full_pipeline_mixed_sources(self, mock_classify, mock_parse_index, tmp_path):
        """Should classify pages from multiple sources"""
        # Setup test data
        inventory_path = tmp_path / "inventory_prepared.csv"
        inventory_path.write_text("""file_path,file_type,section_dir,filename,original_source
data_src/21 - Clutch/21-00-index-a.jpg,jpg,21 - Clutch,21-00-index-a.jpg,
data_src/21 - Clutch/21-01.jpg,jpg,21 - Clutch,21-01.jpg,
data_src/etm/etm-001.jpg,jpg,1990 BMW M3 Electrical Troubleshooting Manual,etm-001.jpg,
""")

        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
api:
  provider: anthropic
  model: claude-sonnet-4
  api_key: test_key
sources:
  service_manual_patterns:
    - "^[0-9]{2} - "
  electrical_manual_patterns:
    - "Electrical Troubleshooting"
""")

        # Setup mocks
        mock_parse_index.return_value = {
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [{"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}],
            "error": None
        }
        mock_classify.side_effect = [
            {"content_type": "procedure", "confidence": 0.85},
            {"content_type": "wiring", "confidence": 0.88}
        ]

        # Run classification (would call main() in actual test)
        # For now, test individual components
        config = load_config(config_path)
        records = read_prepared_inventory(inventory_path)

        assert len(records) == 3
        assert records[0]["section_dir"] == "21 - Clutch"
        assert records[2]["section_dir"] == "1990 BMW M3 Electrical Troubleshooting Manual"

    @patch('scripts.classify_pages_03.classify_page_content')
    def test_idempotency_rerun_safe(self, mock_classify, tmp_path):
        """Should be safe to rerun on same inventory - uses cache file"""
        # Setup: Create a cache file from "previous run"
        cache_path = tmp_path / "classified" / "pages.csv"
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text("""page_id,image_path,section_id,section_name,source_type,content_type,is_index,confidence
21-01,data_src/21 - Clutch/21-01.jpg,21,Clutch,service_manual,procedure,False,0.85
""")

        inventory_path = tmp_path / "inventory_prepared.csv"
        inventory_path.write_text("""file_path,file_type,section_dir,filename,original_source
data_src/21 - Clutch/21-01.jpg,jpg,21 - Clutch,21-01.jpg,
data_src/21 - Clutch/21-02.jpg,jpg,21 - Clutch,21-02.jpg,
""")

        # Mock only called for 21-02 (21-01 is cached)
        mock_classify.return_value = {"content_type": "procedure", "confidence": 0.85}

        config = {"api": {"provider": "anthropic", "model": "claude-sonnet-4", "api_key": "test"}}
        records = read_prepared_inventory(inventory_path)

        # load_cached_results returns existing classifications
        cached = load_cached_results(cache_path)
        assert "21-01" in cached

        # process_classification should skip cached entries
        classification_records, _, _ = process_classification(
            records, config, cached_results=cached
        )

        # Should have called classify only once (for 21-02, not 21-01)
        assert mock_classify.call_count == 1

    @patch('scripts.classify_pages_03.classify_page_content')
    def test_handles_missing_images_gracefully(self, mock_classify, tmp_path):
        """Should continue processing if some images are missing"""
        inventory_path = tmp_path / "inventory_prepared.csv"
        inventory_path.write_text("""file_path,file_type,section_dir,filename,original_source
data_src/21 - Clutch/missing.jpg,jpg,21 - Clutch,missing.jpg,
data_src/21 - Clutch/21-01.jpg,jpg,21 - Clutch,21-01.jpg,
""")

        # Create only the second image
        img_dir = tmp_path / "data_src" / "21 - Clutch"
        img_dir.mkdir(parents=True)
        # Create a minimal valid JPEG
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_dir / "21-01.jpg", "JPEG")

        mock_classify.return_value = {"content_type": "procedure", "confidence": 0.85}

        config = {"api": {"provider": "anthropic", "model": "claude-sonnet-4", "api_key": "test"}}
        records = read_prepared_inventory(inventory_path)

        classification_records, _, errors = process_classification(records, config)

        # Should have 1 successful record (21-01) and 1 error (missing.jpg)
        assert len(classification_records) == 1
        assert len(errors) == 1
        assert "missing.jpg" in errors[0]["filename"]
        assert "not found" in errors[0]["error_message"].lower() or "does not exist" in errors[0]["error_message"].lower()
```

---

### Phase 2: Implement to Pass Tests

After writing tests, implement `scripts/03_classify_pages.py` to pass them.

---

## Implementation Specification

### Script Structure

```python
#!/usr/bin/env python3
"""
Stage 3: Classification & Index Parsing
Classifies pages by content type and source type, extracts index metadata.

This script processes the prepared inventory from Stage 2, detects source types,
identifies and parses index pages, classifies content types, and prepares
metadata for Q&A generation in Stage 4.

Usage:
    python scripts/03_classify_pages.py \\
        --inventory work/inventory_prepared.csv \\
        --output-csv work/classified/pages.csv \\
        --output-indices work/indices \\
        --config config.yaml
"""

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import yaml
from anthropic import Anthropic
from PIL import Image
import base64

# Configure logging (follow same pattern as Stages 1 & 2)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary

    Required config structure:
        api:
          provider: anthropic
          model: claude-sonnet-4-20250514
          api_key: ${ANTHROPIC_API_KEY}  # or direct value
        sources:
          service_manual_patterns: [...]
          electrical_manual_patterns: [...]
          ecu_technical_patterns: [...]
        classification:
          content_types:
            service_manual: [...]
            electrical_manual: [...]
            ecu_technical: [...]
    """
    pass


def detect_source_type(section_dir: str, filename: str) -> str:
    """
    Detect source type from directory name and filename.

    Args:
        section_dir: Section directory name (e.g., "21 - Clutch")
        filename: Base filename (e.g., "21-01.jpg")

    Returns:
        Source type: 'service_manual', 'electrical_manual', 'ecu_technical',
                     'html_specs', or 'unknown'

    Detection Rules:
        - HTML files → 'html_specs'
        - "XX - Name" (numbered sections) → 'service_manual'
        - "Getrag*" → 'service_manual'
        - "Electrical Troubleshooting" in name → 'electrical_manual'
        - "Bosch Motronic" in name → 'ecu_technical'
        - Otherwise → 'unknown'
    """
    pass


def is_index_page(filename: str) -> bool:
    """
    Determine if a page is an index/table of contents.

    Args:
        filename: Base filename (e.g., "21-00-index-a.jpg")

    Returns:
        True if filename matches index patterns

    Index Patterns:
        - *-index-* (e.g., "21-00-index-a.jpg")
        - *-toc-* (e.g., "section-toc.jpg")
        - Case insensitive
    """
    pass


def generate_page_id(filename: str, section_dir: str) -> str:
    """
    Generate unique page ID from filename and section.

    Args:
        filename: Base filename
        section_dir: Section directory name

    Returns:
        Page ID string

    Examples:
        - "21-03.jpg" in "21 - Clutch" → "21-03"
        - "etm-001.jpg" in "1990 BMW M3 Electrical..." → "etm-001"
        - "001.jpg" in "Getrag265" → "getrag-001"
        - "bosch-001.jpg" in "Bosch Motronic..." → "bosch-001"
    """
    pass


def encode_image_base64(image_path: Path) -> str:
    """
    Encode image to base64 for Claude API.

    Args:
        image_path: Path to image file

    Returns:
        Base64-encoded image string
    """
    pass


def parse_index_page(
    image_path: Path,
    section_dir: str,
    api_key: str,
    model: str,
    max_retries: int = 3
) -> Dict:
    """
    Parse index page to extract repair codes and page mappings using Claude API.

    Args:
        image_path: Path to index page image
        section_dir: Section directory name (e.g., "21 - Clutch")
        api_key: Anthropic API key
        model: Claude model name
        max_retries: Maximum retry attempts for API calls

    Returns:
        Dictionary with extracted metadata:
        {
            "section_id": "21",
            "section_name": "Clutch",
            "index_pages": ["21-00-index-a.jpg"],
            "procedures": [
                {
                    "code": "21 00 006",
                    "name": "Clutch - bleed",
                    "page": "21-1"
                },
                ...
            ],
            "error": None or error message
        }

    API Prompt Template:
        System: "You are extracting structured repair procedure data from a BMW E30
                 service manual index page. Extract repair codes, procedure names,
                 and page references. Output valid JSON only."

        User: [IMAGE] + "Extract all repair procedures from this index page.
               For each procedure, identify:
               - Repair code (e.g., '21 00 006')
               - Procedure name
               - Page reference (e.g., '21-1')

               Output as JSON:
               {
                 'section_id': 'XX',
                 'section_name': 'Name',
                 'procedures': [
                   {'code': 'XX YY ZZZ', 'name': 'Procedure name', 'page': 'XX-Y'},
                   ...
                 ]
               }"
    """
    pass


def classify_page_content(
    image_path: Path,
    source_type: str,
    is_index: bool,
    api_key: str,
    model: str = "claude-3-haiku-20240307",
    max_retries: int = 3
) -> Dict:
    """
    Classify page content type using Claude Vision API.

    Args:
        image_path: Path to page image
        source_type: Source type from detect_source_type()
        is_index: Whether page is an index page (skip API if True)
        api_key: Anthropic API key
        model: Claude model for classification (default: Haiku for cost efficiency)
        max_retries: Maximum retry attempts for API calls

    Returns:
        Dictionary with classification:
        {
            "content_type": "procedure" | "specification" | "diagram" | ...,
            "confidence": 0.0-1.0,
            "reasoning": "Brief explanation of classification",
            "api_called": True | False,
            "error": None | "error message"
        }

    Classification Logic:

        If is_index=True:
            → Return immediately: content_type='index', confidence=0.95, api_called=False

        Otherwise, call Claude Vision API to classify based on visual content.

    Valid Content Types by Source:

        service_manual:
            - 'index': Table of contents with repair codes and page references
            - 'procedure': Multiple photos with numbered callouts + step-by-step text
            - 'specification': Data tables with values, units, dimensions
            - 'diagram': Full-page technical illustrations, exploded views, cutaways
            - 'troubleshooting': 3-column table (Condition | Cause | Correction)
            - 'text': Dense paragraphs, introductory/reference text

        electrical_manual:
            - 'wiring': Wiring diagrams, circuit schematics
            - 'pinout': Connector pin assignments, terminal layouts
            - 'fuse_chart': Fuse/relay tables with circuit names
            - 'flowchart': Diagnostic flowcharts
            - 'text': Circuit descriptions

        ecu_technical:
            - 'diagram': Signal diagrams, system architecture
            - 'specification': Parameter tables, sensor values
            - 'flowchart': Control logic flowcharts
            - 'oscilloscope': Test procedures with scope traces
            - 'text': Technical descriptions

    Error Handling:
        - API errors: Return content_type='unknown', confidence=0.3, error=message
        - Malformed response: Return content_type='unknown', confidence=0.3, error=message
        - Retry on transient failures

    Cost Optimization:
        - Uses Haiku by default (~$0.25/M input, $1.25/M output tokens)
        - Single image + short prompt ≈ 1000-2000 tokens
        - Estimated cost: ~$0.01-0.02 per image
        - ~1400 images ≈ $15-30 total for classification
    """
    pass


def read_prepared_inventory(inventory_path: Path) -> List[Dict[str, str]]:
    """
    Read prepared inventory CSV from Stage 2.

    Args:
        inventory_path: Path to inventory_prepared.csv

    Returns:
        List of inventory records (dicts), filtered to image files only

    Expected columns: file_path, file_type, section_dir, filename, original_source

    Filters: Only include file_type='jpg' or 'png' (exclude HTML)
    """
    pass


def process_classification(
    inventory_records: List[Dict[str, str]],
    config: Dict
) -> Tuple[List[Dict], Dict[str, Dict], List[Dict]]:
    """
    Main classification processing logic.

    Args:
        inventory_records: Records from inventory_prepared.csv
        config: Configuration dictionary

    Returns:
        Tuple of (classification_records, index_metadata, errors)

    Logic:
        1. For each image in inventory:
           a. Detect source_type from directory
           b. Check if index page
           c. If index page:
              - Call parse_index_page() to extract metadata
              - Store in index_metadata dict (keyed by section slug)
           d. Classify content_type
           e. Create classification record

        2. Handle errors gracefully:
           - Log error
           - Add to errors list
           - Continue processing

    classification_records schema:
        - page_id: unique identifier
        - image_path: path to image
        - section_id: section number/code
        - section_name: section name
        - source_type: detected source type
        - content_type: classified content type
        - is_index: boolean
        - confidence: classification confidence

    index_metadata schema (keyed by "section_id-section_name_slug"):
        {
            "section_id": "21",
            "section_name": "Clutch",
            "index_pages": ["21-00-index-a.jpg", ...],
            "procedures": [
                {"code": "21 00 006", "name": "...", "page": "21-1"},
                ...
            ]
        }

    errors schema:
        - timestamp: ISO 8601
        - filename: filename
        - operation: 'index_parsing' | 'classification'
        - error_message: error details
    """
    pass


def merge_index_metadata(index_metadata: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Merge index metadata and generate page_to_procedures mapping.

    Args:
        index_metadata: Raw index metadata from process_classification()

    Returns:
        Enhanced index metadata with page_to_procedures mapping

    Adds to each section:
        "page_to_procedures": {
            "21-1": ["21 00 006", "21 11 000"],
            "21-2": ["21 21 000"],
            ...
        }

    This mapping enables Stage 4 to inject context: "This page covers procedures X, Y"
    """
    pass


def write_classification_csv(records: List[Dict], output_path: Path):
    """
    Write classification CSV.

    Args:
        records: Classification records
        output_path: Path to output CSV

    Schema: page_id, image_path, section_id, section_name, source_type,
            content_type, is_index, confidence

    Behavior:
        - Creates output directory if needed
        - Sorts by page_id for reproducibility
        - Writes header + data
    """
    pass


def write_index_json(metadata: Dict, output_path: Path):
    """
    Write index metadata JSON file.

    Args:
        metadata: Index metadata for one section
        output_path: Path to JSON file

    Schema: (see merge_index_metadata())

    Behavior:
        - Creates output directory if needed
        - Pretty-prints JSON (indent=2)
        - Overwrites existing file
    """
    pass


def write_error_log(errors: List[Dict], log_path: Path):
    """
    Write error log CSV.

    Args:
        errors: Error records
        log_path: Path to log CSV

    Schema: timestamp, filename, operation, error_message

    Behavior:
        - Creates log directory if needed
        - Appends to existing log if present
    """
    pass


def generate_report(
    classification_records: List[Dict],
    index_metadata: Dict[str, Dict],
    errors: List[Dict],
    report_path: Path
):
    """
    Generate summary report in Markdown.

    Args:
        classification_records: Classification records
        index_metadata: Index metadata
        errors: Error records
        report_path: Path to output report

    Report Contents:
        # Stage 3: Classification & Index Parsing Report

        ## Summary Statistics
        - Total Pages Classified: X
        - Index Pages: X
        - Sections with Indices: X
        - Total Procedures Extracted: X
        - Errors Encountered: X

        ## Source Type Distribution
        - service_manual: X pages
        - electrical_manual: X pages
        - ecu_technical: X pages
        - unknown: X pages

        ## Content Type Distribution
        (breakdown by source type)

        ## Classification Confidence
        - High (>0.8): X pages
        - Medium (0.5-0.8): X pages
        - Low (<0.5): X pages

        ## Index Metadata Summary
        (per section: number of procedures, pages covered)

        ## Errors (if any)
        (list of errors with details)
    """
    pass


def print_summary(
    classification_records: List[Dict],
    index_metadata: Dict[str, Dict],
    errors: List[Dict]
):
    """
    Print summary statistics to stdout.

    Args:
        classification_records: Classification records
        index_metadata: Index metadata
        errors: Error records

    Summary includes:
        - Total pages classified
        - Source type breakdown
        - Content type breakdown
        - Index pages processed
        - Procedures extracted
        - Errors encountered
    """
    pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stage 3: Classification & Index Parsing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/03_classify_pages.py \\
      --inventory work/inventory_prepared.csv \\
      --output-csv work/classified/pages.csv \\
      --output-indices work/indices \\
      --config config.yaml

  # With verbose logging
  python scripts/03_classify_pages.py \\
      --inventory work/inventory_prepared.csv \\
      --output-csv work/classified/pages.csv \\
      --output-indices work/indices \\
      --config config.yaml \\
      --verbose

Input CSV schema (from Stage 2):
  file_path, file_type, section_dir, filename, original_source

Output CSV schema:
  page_id, image_path, section_id, section_name, source_type,
  content_type, is_index, confidence

Index JSON schema:
  {
    "section_id": "21",
    "section_name": "Clutch",
    "index_pages": ["21-00-index-a.jpg"],
    "procedures": [
      {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"},
      ...
    ],
    "page_to_procedures": {
      "21-1": ["21 00 006", "21 11 000"],
      ...
    }
  }

Environment Variables:
  ANTHROPIC_API_KEY - Claude API key (required for index parsing)
        """
    )

    parser.add_argument(
        '--inventory',
        type=Path,
        required=True,
        help='Path to inventory_prepared.csv from Stage 2'
    )

    parser.add_argument(
        '--output-csv',
        type=Path,
        required=True,
        help='Path to output classification CSV'
    )

    parser.add_argument(
        '--output-indices',
        type=Path,
        required=True,
        help='Directory for index JSON files'
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to config.yaml'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )

    parser.add_argument(
        '--skip-index-parsing',
        action='store_true',
        help='Skip index parsing (for testing classification only)'
    )

    parser.add_argument(
        '--skip-classification',
        action='store_true',
        help='Skip content classification (for testing index parsing only)'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-processing even if output files exist'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit processing to first N images (for validation before full run)'
    )

    parser.add_argument(
        '--sample-validation',
        action='store_true',
        help='Run on 10 diverse sample images and print results for manual review'
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info("Starting classification & index parsing")

        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Check for API key if not skipping index parsing
        if not args.skip_index_parsing:
            api_key = config.get('api', {}).get('api_key')
            if not api_key or api_key.startswith('${'):
                # Try environment variable
                import os
                api_key = os.environ.get('ANTHROPIC_API_KEY')
                if not api_key:
                    logger.error("ANTHROPIC_API_KEY not found in config or environment")
                    return 1
            config['api']['api_key'] = api_key

        # Read inventory
        inventory_records = read_prepared_inventory(args.inventory)
        logger.info(f"Read {len(inventory_records)} image records from inventory")

        # Process classification
        classification_records, index_metadata, errors = process_classification(
            inventory_records,
            config
        )

        # Merge index metadata and generate page mappings
        index_metadata = merge_index_metadata(index_metadata)

        # Write outputs
        write_classification_csv(classification_records, args.output_csv)
        logger.info(f"Wrote {len(classification_records)} classification records to {args.output_csv}")

        # Write index JSON files
        args.output_indices.mkdir(parents=True, exist_ok=True)
        for section_key, metadata in index_metadata.items():
            json_path = args.output_indices / f"{section_key}.json"
            write_index_json(metadata, json_path)
        logger.info(f"Wrote {len(index_metadata)} index JSON files to {args.output_indices}")

        # Write error log if errors occurred
        if errors:
            error_log_path = args.output_csv.parent.parent / "logs" / "classification_errors.csv"
            write_error_log(errors, error_log_path)
            logger.warning(f"Encountered {len(errors)} errors; see {error_log_path}")

        # Generate report
        report_path = args.output_csv.parent.parent / "logs" / "stage3_classification_report.md"
        generate_report(classification_records, index_metadata, errors, report_path)
        logger.info(f"Generated report at {report_path}")

        # Print summary
        print_summary(classification_records, index_metadata, errors)

        logger.info("Classification & index parsing completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

## Acceptance Criteria

### Functional Requirements

- [ ] **Source Type Detection**
  - [ ] Service manual sections detected by "XX - Name" pattern
  - [ ] Getrag PDFs detected as service_manual
  - [ ] Electrical troubleshooting manual detected
  - [ ] Bosch Motronic detected as ecu_technical
  - [ ] HTML files detected as html_specs

- [ ] **Index Page Detection**
  - [ ] *-index-* patterns detected
  - [ ] *-toc-* patterns detected
  - [ ] Case-insensitive detection
  - [ ] Non-index pages correctly excluded

- [ ] **Index Parsing**
  - [ ] Repair codes extracted (>95% success rate)
  - [ ] Procedure names extracted
  - [ ] Page references extracted
  - [ ] Multi-page indices merged correctly
  - [ ] API errors handled gracefully
  - [ ] Malformed responses logged as errors

- [ ] **Content Classification (Vision-Based)**
  - [ ] Index pages detected by filename classified as 'index' (no API call)
  - [ ] All other pages classified via Claude Haiku Vision API
  - [ ] Service manual pages: procedure, specification, diagram, troubleshooting, text
  - [ ] Electrical manual pages: wiring, fuse_chart, pinout, flowchart, text
  - [ ] ECU technical pages: diagram, specification, flowchart, oscilloscope, text
  - [ ] API failures result in 'unknown' classification with error logged

- [ ] **Page Mapping Generation**
  - [ ] page_to_procedures mapping generated for all sections
  - [ ] Multiple procedures per page handled correctly
  - [ ] Page references normalized

- [ ] **Output Files**
  - [ ] work/classified/pages.csv contains all pages
  - [ ] CSV sorted by page_id
  - [ ] Correct schema
  - [ ] work/indices/*.json files created (one per section)
  - [ ] JSON files have correct schema
  - [ ] work/logs/classification_errors.csv logs failures
  - [ ] work/logs/stage3_classification_report.md generated

### Code Quality Requirements

- [ ] **Testing**
  - [ ] All unit tests pass
  - [ ] Integration tests pass
  - [ ] API mocking works correctly
  - [ ] Test coverage >85%

- [ ] **Style**
  - [ ] Follows same patterns as Stages 1 & 2
  - [ ] Type hints on all functions
  - [ ] Docstrings on all public functions
  - [ ] Clear logging at INFO level

- [ ] **Error Handling**
  - [ ] API errors handled gracefully
  - [ ] Missing images logged and skipped
  - [ ] Malformed API responses handled
  - [ ] Continues processing after individual failures

- [ ] **Documentation**
  - [ ] `--help` output is comprehensive
  - [ ] Examples in epilog
  - [ ] README updated if needed

### Performance Requirements

- [ ] **Efficiency**
  - [ ] API calls only for index pages
  - [ ] Reasonable rate limiting
  - [ ] Processes files in deterministic order
  - [ ] Memory-efficient

### Validation Checks

Before considering Stage 3 complete, run:

```bash
# Test on sample data
python scripts/03_classify_pages.py \
    --inventory work/inventory_prepared.csv \
    --output-csv work/classified/pages.csv \
    --output-indices work/indices \
    --config config.yaml \
    --verbose

# Verify outputs exist
test -f work/classified/pages.csv || echo "FAIL: pages.csv missing"
test -d work/indices || echo "FAIL: indices directory missing"

# Check classification CSV has entries
CLASSIFIED_COUNT=$(tail -n +2 work/classified/pages.csv | wc -l)
echo "Pages classified: $CLASSIFIED_COUNT"

# Verify index JSON files created
INDEX_COUNT=$(ls work/indices/*.json 2>/dev/null | wc -l)
echo "Index JSON files: $INDEX_COUNT"

# Check for source type distribution
echo "Source types:"
tail -n +2 work/classified/pages.csv | cut -d',' -f5 | sort | uniq -c

# Check for content type distribution
echo "Content types:"
tail -n +2 work/classified/pages.csv | cut -d',' -f6 | sort | uniq -c

# Verify index pages classified correctly
INDEX_PAGES=$(grep -c ",true," work/classified/pages.csv || echo 0)
echo "Index pages: $INDEX_PAGES"

# Check report generated
test -f work/logs/stage3_classification_report.md || echo "FAIL: Report missing"
```

---

## Dependencies

Add to `requirements.txt`:

```txt
# Existing dependencies
...

# Stage 3: Classification & Index Parsing
anthropic>=0.18.0       # Claude API for index parsing
PyYAML>=6.0            # Config file parsing
Pillow>=10.0.0         # Image processing (already in Stage 2)
```

Environment:

```bash
# Set API key
export ANTHROPIC_API_KEY="your-api-key-here"
```

---

## Implementation Checklist

### Step 1: Setup (before coding)
- [ ] Read this spec completely
- [ ] Read architecture doc (`pipeline_rearchitecture.md` lines 377-437)
- [ ] Install dependencies: `pip install anthropic pyyaml`
- [ ] Set ANTHROPIC_API_KEY environment variable
- [ ] Review Stage 1 & 2 code for patterns

### Step 2: Create Config File
- [ ] Create `config.yaml` with required structure
- [ ] Add source detection patterns
- [ ] Add API settings
- [ ] Add content type definitions

### Step 3: Write Tests (TDD Phase 1)
- [ ] Create `tests/test_03_classify_pages.py`
- [ ] Write all test cases from test suite above
- [ ] Create test fixtures (sample inventory, images, config)
- [ ] Setup API mocking
- [ ] Run tests: `pytest tests/test_03_classify_pages.py -v`
- [ ] **Expected: ALL TESTS FAIL** (functions don't exist yet)

### Step 4: Implement (TDD Phase 2)
- [ ] Create `scripts/03_classify_pages.py`
- [ ] Implement `load_config()` → run tests → fix until pass
- [ ] Implement `detect_source_type()` → run tests → fix until pass
- [ ] Implement `is_index_page()` → run tests → fix until pass
- [ ] Implement `generate_page_id()` → run tests → fix until pass
- [ ] Implement `encode_image_base64()` → run tests → fix until pass
- [ ] Implement `parse_index_page()` → run tests → fix until pass
- [ ] Implement `classify_page_content()` → run tests → fix until pass
- [ ] Implement `read_prepared_inventory()` → run tests → fix until pass
- [ ] Implement `process_classification()` → run tests → fix until pass
- [ ] Implement `merge_index_metadata()` → run tests → fix until pass
- [ ] Implement `write_classification_csv()` → run tests → fix until pass
- [ ] Implement `write_index_json()` → run tests → fix until pass
- [ ] Implement `write_error_log()` → run tests → fix until pass
- [ ] Implement `generate_report()` → verify output
- [ ] Implement `print_summary()` → verify output
- [ ] Implement `main()` → run integration tests
- [ ] Run all tests: `pytest tests/test_03_classify_pages.py -v`
- [ ] **Expected: ALL TESTS PASS**

### Step 5: Integration Testing
- [ ] Run on actual inventory: `python scripts/03_classify_pages.py --inventory work/inventory_prepared.csv --output-csv work/classified/pages.csv --output-indices work/indices --config config.yaml --verbose`
- [ ] Verify outputs created
- [ ] Inspect classification CSV
- [ ] Review index JSON files
- [ ] Check error log for issues
- [ ] Verify report is comprehensive

### Step 6: Validation
- [ ] Run all acceptance criteria checks (see above)
- [ ] Sample 10 index pages manually to verify extraction accuracy
- [ ] Sample 20 classified pages to verify content_type accuracy
- [ ] Code review against Stages 1 & 2 style
- [ ] Update Makefile
- [ ] Commit with message: `feat(stage3): implement classification and index parsing`

---

## Notes for Implementer

1. **Follow Stage 1 & 2 patterns**: Use same logging style, error handling, CLI structure
2. **Test-driven approach**: Write tests FIRST, then implement
3. **API rate limiting**: Add delays between Claude API calls (configurable in config.yaml)
4. **Graceful degradation**: If classification or index parsing fails, log error and continue
5. **Clear logging**: User should see progress (INFO) and details (DEBUG)
6. **Source type is critical**: Drives which content types are valid for classification
7. **Index metadata enables context**: Stage 4 will use page_to_procedures for richer Q&A
8. **Classification**: Use Claude Haiku Vision API for all non-index pages
9. **Section slug generation**: Convert "21 - Clutch" → "21-clutch" for filename compatibility
10. **Handle multi-page indices**: Combine all *-index-a, *-index-b, etc. into single JSON
11. **Cost optimization**: Skip API for index pages (detected by filename), use Haiku for classification
12. **Progress tracking**: Log every N images processed (configurable batch_size in config)
13. **Token tracking**: Optionally log input/output tokens for cost monitoring

## Cost Estimate

| Task | Model | Images | Est. Cost |
|------|-------|--------|-----------|
| Index Parsing | Sonnet | ~50 | ~$1-2 |
| Page Classification | Haiku | ~1,370 | ~$15-25 |
| **Total** | | ~1,420 | **~$16-27** |

Assumptions:
- Index parsing: ~2000 input tokens (image + prompt), ~500 output tokens per page
- Classification: ~1500 input tokens (image + prompt), ~100 output tokens per page
- Haiku pricing: $0.25/M input, $1.25/M output
- Sonnet pricing: $3/M input, $15/M output

---

## Validation Workflow (Run Before Full Processing)

**IMPORTANT**: Since this costs real money, validate on samples first.

### Step 1: Sample Validation (~$0.15)
```bash
# Run on 10 diverse sample images, manually review results
python scripts/03_classify_pages.py \
    --inventory work/inventory_prepared.csv \
    --output-csv work/classified/pages_sample.csv \
    --output-indices work/indices_sample \
    --config config.yaml \
    --sample-validation
```

This selects 10 images covering different content types:
- 2 procedure pages (photos + steps)
- 2 troubleshooting pages (3-column tables)
- 2 specification pages (data tables)
- 2 diagram pages (technical illustrations)
- 2 electrical pages (wiring/fuse charts)

**Review the output**: Check that classifications match what you see in the images.

### Step 2: Small Batch Test (~$1.50)
```bash
# Run on first 100 images
python scripts/03_classify_pages.py \
    --inventory work/inventory_prepared.csv \
    --output-csv work/classified/pages_100.csv \
    --output-indices work/indices_100 \
    --config config.yaml \
    --limit 100
```

**Review**: Spot-check 10-20 random results.

### Step 3: Full Run (~$16-27)
```bash
# Only after validation passes
python scripts/03_classify_pages.py \
    --inventory work/inventory_prepared.csv \
    --output-csv work/classified/pages.csv \
    --output-indices work/indices \
    --config config.yaml
```

### Sample Selection Logic

The `--sample-validation` flag selects diverse images:

```python
def select_validation_samples(inventory_records: List[Dict], n: int = 10) -> List[Dict]:
    """
    Select diverse sample images for validation.

    Strategy:
    - Pick from different sections (11-Engine, 21-Clutch, 34-Brakes, etc.)
    - Pick from different source types (service_manual, electrical_manual)
    - Avoid index pages (we know those work)
    - Pick pages that visually represent different content types

    Known good samples for this dataset:
    - 21-01.jpg: procedure (photos + steps)
    - 21-07.jpg: troubleshooting (3-column table)
    - 11-50.jpg: diagram (engine cutaway)
    - 00-01.jpg (Torque Specs): specification (data table)
    - 0670-01.jpg (ETM): fuse_chart (electrical table)
    """
    # Hardcoded diverse samples for initial validation
    sample_filenames = [
        "21-01.jpg",      # procedure
        "21-07.jpg",      # troubleshooting
        "34-10.jpg",      # procedure
        "11-50.jpg",      # diagram
        "00-01.jpg",      # specification (in Torque Specs)
        "0670-01.jpg",    # fuse_chart (ETM)
        "11-100.jpg",     # likely procedure or spec
        "34-01.jpg",      # likely procedure
        "13-01.jpg",      # fuel system - unknown
        "31-01.jpg",      # front axle - unknown
    ]

    return [r for r in inventory_records if r["filename"] in sample_filenames][:n]
```

---

## Claude API Integration

### Classification Prompt Template (Haiku)

```python
CLASSIFICATION_SYSTEM_PROMPT = """You are classifying pages from a BMW E30 M3 service manual.

Analyze the visual content and structure of the page to determine its type.

Valid content types:
- "index": Table of contents with repair codes, procedure names, and page references
- "procedure": Step-by-step instructions with multiple photos showing hands/tools, numbered callouts
- "specification": Data tables with values, units, torque specs, measurements
- "diagram": Full-page technical illustrations, exploded views, cutaway drawings (minimal text)
- "troubleshooting": Diagnostic table with columns for Condition/Symptom, Cause, and Correction
- "wiring": Electrical wiring diagrams, circuit schematics
- "fuse_chart": Fuse/relay tables listing circuits, amperage, and functions
- "pinout": Connector pin assignments, terminal diagrams
- "flowchart": Diagnostic decision flowcharts
- "oscilloscope": Test procedures showing oscilloscope/multimeter readings
- "text": Dense paragraphs of text, introductory content, reference material

Output ONLY valid JSON:
{
  "content_type": "procedure",
  "confidence": 0.92,
  "reasoning": "Brief explanation"
}"""

CLASSIFICATION_USER_PROMPT = """Classify this page from a {source_type}.

Look at the visual layout and content:
- Are there multiple photos with numbered callouts? → procedure
- Is it a 3-column table (Condition/Cause/Correction)? → troubleshooting
- Is it a full-page technical drawing with minimal text? → diagram
- Is it a data table with values and units? → specification
- Is it a table of contents with repair codes? → index

Return JSON only."""

# API call structure for classification
client = Anthropic(api_key=api_key)
response = client.messages.create(
    model="claude-3-haiku-20240307",  # Use Haiku for cost efficiency
    max_tokens=256,  # Classification needs minimal output
    system=CLASSIFICATION_SYSTEM_PROMPT,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image_data
                    }
                },
                {
                    "type": "text",
                    "text": CLASSIFICATION_USER_PROMPT.format(source_type=source_type)
                }
            ]
        }
    ]
)

# Parse response
json_text = response.content[0].text
data = json.loads(json_text)
```

---

### Index Parsing Prompt Template (Sonnet)

```python
SYSTEM_PROMPT = """You are extracting structured repair procedure data from a BMW E30 M3 service manual index page.

Your task is to extract:
1. Section ID (e.g., "21")
2. Section name (e.g., "Clutch")
3. All repair procedures listed with their:
   - Repair code (e.g., "21 00 006")
   - Procedure name (e.g., "Clutch - bleed")
   - Page reference (e.g., "21-1")

Output ONLY valid JSON in this exact format:
{
  "section_id": "21",
  "section_name": "Clutch",
  "procedures": [
    {
      "code": "21 00 006",
      "name": "Clutch - bleed",
      "page": "21-1"
    },
    {
      "code": "21 11 000",
      "name": "Clutch housing - remove and install",
      "page": "21-1"
    }
  ]
}

Do not include any explanatory text, only the JSON object."""

USER_PROMPT = """Extract all repair procedures from this BMW E30 M3 service manual index page.

Guidelines:
- Repair codes follow the pattern: XX YY ZZZ (e.g., "21 00 006")
- Some entries may be just numbers (e.g., "565") without the full code format
- Page references are usually "XX-Y" format (e.g., "21-1", "21-2")
- Include ALL procedures visible on this page
- If the section ID or name is not clearly visible, infer from the content

Return the JSON object only."""

# API call structure
client = Anthropic(api_key=api_key)
response = client.messages.create(
    model=model,
    max_tokens=4096,
    system=SYSTEM_PROMPT,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": base64_image_data
                    }
                },
                {
                    "type": "text",
                    "text": USER_PROMPT
                }
            ]
        }
    ]
)

# Parse response
json_text = response.content[0].text
data = json.loads(json_text)
```

---

## Example Test Fixtures

Create in `tests/fixtures/`:

### `sample_inventory_prepared.csv`
```csv
file_path,file_type,section_dir,filename,original_source
data_src/21 - Clutch/21-00-index-a.jpg,jpg,21 - Clutch,21-00-index-a.jpg,
data_src/21 - Clutch/21-01.jpg,jpg,21 - Clutch,21-01.jpg,
data_src/21 - Clutch/21-02.jpg,jpg,21 - Clutch,21-02.jpg,
data_src/etm/etm-001.jpg,jpg,1990 BMW M3 Electrical Troubleshooting Manual,etm-001.jpg,
data_src/bosch/bosch-001.jpg,jpg,Bosch Motronic ML 3-1,bosch-001.jpg,
data_src/Getrag265/001.jpg,jpg,Getrag265,001.jpg,data_src/Getrag265.pdf
```

### `sample_config.yaml`
```yaml
api:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}
  max_retries: 3
  rate_limit_delay_seconds: 0.5  # Delay between API calls

  # Use different models for different tasks
  models:
    index_parsing: claude-sonnet-4-20250514    # More capable for structured extraction
    classification: claude-3-haiku-20240307    # Cost-efficient for simple classification

sources:
  service_manual_patterns:
    - "^[0-9]{2} - "
    - "^Getrag"
  electrical_manual_patterns:
    - "Electrical Troubleshooting"
  ecu_technical_patterns:
    - "Bosch Motronic"

classification:
  # Valid content types by source type
  content_types:
    service_manual:
      - index
      - procedure
      - specification
      - diagram
      - troubleshooting
      - text
    electrical_manual:
      - index
      - wiring
      - pinout
      - fuse_chart
      - flowchart
      - text
    ecu_technical:
      - diagram
      - specification
      - flowchart
      - oscilloscope
      - text

  # Batch processing settings
  batch_size: 10  # Number of images to process before logging progress

  # Cost tracking
  track_tokens: true  # Log token usage for cost monitoring
```

### Sample Images
- Create small test images for different types
- Use real index page image if available for integration testing

### Mock API Responses
Create JSON fixtures for mocking Claude API responses in tests.

---

## Success Criteria

Stage 3 is complete when:

1. ✅ All tests pass (`pytest tests/test_03_classify_pages.py -v`)
2. ✅ Script runs successfully on actual inventory
3. ✅ All pages classified with source_type and content_type
4. ✅ Index pages detected and parsed
5. ✅ `work/classified/pages.csv` created with correct schema
6. ✅ `work/indices/*.json` files created (one per section)
7. ✅ page_to_procedures mapping generated for all sections
8. ✅ >95% of repair codes extracted from index pages
9. ✅ Classification confidence high (>90% of pages >0.8 confidence via vision API)
10. ✅ Error log tracks failures without crashing pipeline
11. ✅ Summary report is comprehensive
12. ✅ All acceptance criteria checked and passing
13. ✅ Code reviewed and committed
14. ✅ Ready for Stage 4 (Q&A Generation)

---

## Design Decisions

1. **Pages that don't fit any content type**: Return 'unknown' with low confidence; log for manual review

2. **Repair code validation**: Log warnings for codes that don't match "XX YY ZZZ" or number patterns

3. **Multiple index pages with conflicting data**: See "Multi-Page Index Merge Algorithm" below

4. **API failures**: Retry with exponential backoff (3 attempts), then log error, mark as 'unknown', continue processing

5. **Index metadata scope**: Extract codes/names/pages only; extend later if needed

6. **Cost optimization**: Haiku for classification, Sonnet for index extraction. Skip API for index pages (detected by filename).

7. **Caching/Resume**: See "Resume and Caching Mechanism" below

8. **Mixed-content pages**: See "Mixed-Content Page Classification" below

9. **Confidence score usage**: See "Confidence Score Usage" below

---

## Mixed-Content Page Classification

**Problem**: Some pages contain multiple content types (e.g., procedure steps with an embedded spec table).

**Decision**: Classify by **dominant content type** using this priority hierarchy:

```
Priority (highest to lowest):
1. troubleshooting - If 3-column Condition/Cause/Correction table present
2. procedure - If numbered steps with photos/callouts present
3. specification - If data tables with values/units dominate
4. diagram - If >60% of page is technical illustration
5. wiring/pinout/fuse_chart - Electrical-specific content
6. text - Fallback for text-heavy pages
```

**API Prompt Modification**: The classification prompt instructs Claude to:
1. Identify ALL content types present on the page
2. Return the dominant type based on visual area and semantic importance
3. Include `secondary_types` field for mixed pages

**Updated Classification Response Schema**:
```json
{
  "content_type": "procedure",
  "confidence": 0.85,
  "secondary_types": ["specification"],
  "reasoning": "Page shows 4 numbered procedure steps with photos; small torque spec table in corner"
}
```

**Downstream Impact**: Stage 4 can use `secondary_types` to generate additional Q&A pairs from the same page.

---

## Resume and Caching Mechanism

**Problem**: Full classification run takes ~1400 API calls. Need to resume after interruption.

**Solution**: Cache-based resume using output CSV as state file.

### Cache File Format
The output CSV (`work/classified/pages.csv`) serves as the cache. On re-run:

1. **Load existing cache**: Read `pages.csv` into dict keyed by `page_id`
2. **Skip cached entries**: For each inventory record, check if `page_id` exists in cache
3. **Process only new/missing**: Only call API for uncached pages
4. **Merge and write**: Combine cached + new results, write complete CSV

### New Functions Required

```python
def load_cached_results(cache_path: Path) -> Dict[str, Dict]:
    """
    Load existing classification results from cache file.

    Args:
        cache_path: Path to pages.csv

    Returns:
        Dict mapping page_id -> full classification record
        Empty dict if file doesn't exist
    """
    pass


def process_classification(
    inventory_records: List[Dict[str, str]],
    config: Dict,
    cached_results: Optional[Dict[str, Dict]] = None  # NEW PARAMETER
) -> Tuple[List[Dict], Dict[str, Dict], List[Dict]]:
    """
    Updated signature to accept cached results.

    If cached_results provided:
    - Skip API calls for page_ids already in cache
    - Return cached record directly
    - Only call API for new pages
    """
    pass
```

### CLI Flags

```bash
# Normal run (uses cache, skips existing)
python scripts/03_classify_pages.py --inventory ... --output-csv work/classified/pages.csv

# Force re-run (ignores cache, reprocesses all)
python scripts/03_classify_pages.py --inventory ... --output-csv work/classified/pages.csv --force

# Resume from specific point (for debugging)
python scripts/03_classify_pages.py --inventory ... --output-csv work/classified/pages.csv --start-from "21-05"
```

### Progress Tracking

During processing, write intermediate results every `batch_size` (default 10) pages:
```python
# After every batch_size pages processed
write_classification_csv(all_results_so_far, output_path)
logger.info(f"Checkpoint: {len(all_results_so_far)} pages saved")
```

This ensures minimal re-work on crash.

---

## Section Slug Algorithm

**Function**: `generate_section_slug(section_dir: str) -> str`

**Purpose**: Convert section directory name to filesystem-safe slug for JSON filenames.

### Algorithm

```python
def generate_section_slug(section_dir: str) -> str:
    """
    Convert section directory name to filesystem-safe slug.

    Args:
        section_dir: Section directory name (e.g., "21 - Clutch")

    Returns:
        Slug string (e.g., "21-clutch")

    Algorithm:
        1. Lowercase the entire string
        2. Replace " - " with "-"
        3. Replace remaining spaces with "-"
        4. Remove characters not in [a-z0-9-]
        5. Collapse multiple dashes to single dash
        6. Strip leading/trailing dashes

    Examples:
        "21 - Clutch" → "21-clutch"
        "00 - Maintenance" → "00-maintenance"
        "00 - Torque Specs" → "00-torque-specs"
        "1990 BMW M3 Electrical Troubleshooting Manual" → "1990-bmw-m3-electrical-troubleshooting-manual"
        "Bosch Motronic ML 3-1" → "bosch-motronic-ml-3-1"
        "Getrag265" → "getrag265"
    """
    import re

    slug = section_dir.lower()
    slug = slug.replace(" - ", "-")
    slug = slug.replace(" ", "-")
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')

    return slug
```

### Tests

```python
class TestSectionSlugGeneration:
    """Test section slug generation"""

    def test_numbered_section(self):
        assert generate_section_slug("21 - Clutch") == "21-clutch"
        assert generate_section_slug("00 - Maintenance") == "00-maintenance"

    def test_multi_word_section(self):
        assert generate_section_slug("00 - Torque Specs") == "00-torque-specs"
        assert generate_section_slug("12 - Engine Electrical Equipment") == "12-engine-electrical-equipment"

    def test_long_name(self):
        assert generate_section_slug("1990 BMW M3 Electrical Troubleshooting Manual") == "1990-bmw-m3-electrical-troubleshooting-manual"

    def test_special_characters(self):
        assert generate_section_slug("Bosch Motronic ML 3-1") == "bosch-motronic-ml-3-1"

    def test_no_spaces(self):
        assert generate_section_slug("Getrag265") == "getrag265"

    def test_already_slugified(self):
        assert generate_section_slug("21-clutch") == "21-clutch"
```

---

## Multi-Page Index Merge Algorithm

**Problem**: Sections like "21 - Clutch" may have multiple index pages (21-00-index-a.jpg, 21-00-index-b.jpg). Need to merge without data loss or duplication.

### Merge Rules

1. **Same repair code, same page reference**: Keep one (deduplicate)
2. **Same repair code, different page references**: Keep ALL page references (procedure spans pages)
3. **Same repair code, different names**: Log warning, keep the FIRST name encountered
4. **Different sections in same merge**: Error - should never happen (grouped by section_slug)

### Updated merge_index_metadata Function

```python
def merge_index_metadata(index_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Merge index metadata from multiple index pages per section.

    Args:
        index_results: Dict mapping section_slug -> list of parse results
                       Each parse result is from one index page

    Returns:
        Dict mapping section_slug -> merged metadata with:
        - section_id
        - section_name
        - index_pages: list of all index page filenames
        - procedures: deduplicated list with merged page references
        - page_to_procedures: mapping of page -> list of procedure codes
        - merge_warnings: list of any conflicts detected

    Merge Algorithm:
        1. Group all procedures by repair code
        2. For each code:
           a. Collect all page references (may be multiple)
           b. Use first name encountered (log warning if names differ)
           c. Create single procedure entry with pages as list
        3. Build page_to_procedures mapping from merged data
        4. Record any warnings for later review
    """
    merged = {}

    for section_slug, parse_results in index_results.items():
        # Combine all procedures from all index pages
        procedures_by_code = {}  # code -> {name, pages: set()}
        index_pages = []
        section_id = None
        section_name = None
        warnings = []

        for result in parse_results:
            if result.get("error"):
                continue

            section_id = section_id or result.get("section_id")
            section_name = section_name or result.get("section_name")
            index_pages.append(result.get("source_filename", "unknown"))

            for proc in result.get("procedures", []):
                code = proc["code"]
                name = proc["name"]
                page = proc["page"]

                if code in procedures_by_code:
                    existing = procedures_by_code[code]
                    # Add page reference
                    existing["pages"].add(page)
                    # Check for name conflict
                    if existing["name"] != name:
                        warnings.append(f"Code {code}: name mismatch '{existing['name']}' vs '{name}'")
                else:
                    procedures_by_code[code] = {
                        "name": name,
                        "pages": {page}
                    }

        # Convert to final format
        procedures = []
        for code, data in sorted(procedures_by_code.items()):
            procedures.append({
                "code": code,
                "name": data["name"],
                "pages": sorted(data["pages"])  # List, not set
            })

        # Build page_to_procedures mapping
        page_to_procedures = {}
        for proc in procedures:
            for page in proc["pages"]:
                if page not in page_to_procedures:
                    page_to_procedures[page] = []
                page_to_procedures[page].append(proc["code"])

        merged[section_slug] = {
            "section_id": section_id,
            "section_name": section_name,
            "index_pages": index_pages,
            "procedures": procedures,
            "page_to_procedures": page_to_procedures,
            "merge_warnings": warnings
        }

    return merged
```

### Updated Tests

```python
class TestIndexMetadataMerging:
    """Test merging of multi-page index metadata"""

    def test_merge_deduplicates_same_code_same_page(self):
        """Same code+page from multiple sources should deduplicate"""
        index_results = {
            "21-clutch": [
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-a.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}
                    ]
                },
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-b.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}  # Duplicate
                    ]
                }
            ]
        }

        merged = merge_index_metadata(index_results)

        # Should have exactly 1 procedure (deduplicated)
        assert len(merged["21-clutch"]["procedures"]) == 1
        assert merged["21-clutch"]["procedures"][0]["code"] == "21 00 006"
        assert merged["21-clutch"]["procedures"][0]["pages"] == ["21-1"]

    def test_merge_keeps_multiple_pages_for_same_code(self):
        """Same code with different pages should keep all pages"""
        index_results = {
            "21-clutch": [
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-a.jpg",
                    "procedures": [
                        {"code": "21 21 000", "name": "Clutch disc - remove", "page": "21-2"}
                    ]
                },
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-b.jpg",
                    "procedures": [
                        {"code": "21 21 000", "name": "Clutch disc - remove", "page": "21-3"}  # Different page
                    ]
                }
            ]
        }

        merged = merge_index_metadata(index_results)

        # Should have 1 procedure with 2 pages
        assert len(merged["21-clutch"]["procedures"]) == 1
        proc = merged["21-clutch"]["procedures"][0]
        assert proc["code"] == "21 21 000"
        assert sorted(proc["pages"]) == ["21-2", "21-3"]

    def test_merge_logs_warning_on_name_conflict(self):
        """Different names for same code should log warning, keep first"""
        index_results = {
            "21-clutch": [
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-a.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}
                    ]
                },
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-b.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch bleeding procedure", "page": "21-1"}  # Different name!
                    ]
                }
            ]
        }

        merged = merge_index_metadata(index_results)

        # Should keep first name
        assert merged["21-clutch"]["procedures"][0]["name"] == "Clutch - bleed"
        # Should have warning
        assert len(merged["21-clutch"]["merge_warnings"]) == 1
        assert "21 00 006" in merged["21-clutch"]["merge_warnings"][0]
```

---

## Confidence Score Usage

**Problem**: Confidence scores are captured but nothing uses them downstream.

**Decision**: Use confidence scores for:

### 1. Filtering in Stage 4 (Q&A Generation)

```python
# In Stage 4, skip low-confidence pages for Q&A generation
MIN_CONFIDENCE_FOR_QA = 0.7

def should_generate_qa(classification_record: Dict) -> bool:
    """Determine if page should be used for Q&A generation"""
    if classification_record["content_type"] == "unknown":
        return False
    if classification_record["confidence"] < MIN_CONFIDENCE_FOR_QA:
        return False
    return True
```

### 2. Manual Review Queue

Pages with confidence < 0.7 are flagged for manual review:

```python
# In report generation
low_confidence_pages = [r for r in records if r["confidence"] < 0.7]
# Write to work/logs/manual_review_queue.csv
```

### 3. Report Statistics

The stage 3 report includes confidence distribution:

```markdown
## Classification Confidence Distribution
- High (≥0.9): 1,150 pages (82%)
- Medium (0.7-0.9): 180 pages (13%)
- Low (<0.7): 74 pages (5%) ← flagged for review
```

### 4. Stage 4 Quality Weighting

Confidence can be passed to Q&A generation to:
- Weight training examples (higher confidence = higher weight)
- Generate more Q&A pairs from high-confidence pages
- Skip speculative Q&A for low-confidence classifications

### Updated Output Schema

Add to `work/classified/pages.csv`:
```csv
page_id,image_path,...,confidence,needs_review
21-01,...,0.92,False
21-07,...,0.65,True
```

### Updated Function

```python
def write_classification_csv(records: List[Dict], output_path: Path):
    """
    Write classification CSV with needs_review flag.

    Schema: page_id, image_path, section_id, section_name, source_type,
            content_type, is_index, confidence, secondary_types, needs_review

    needs_review = True if confidence < 0.7 or content_type == 'unknown'
    """
    pass
```

---

## Architecture Alignment Checklist

- [ ] Follows Stage 3 spec from `pipeline_rearchitecture.md`
- [ ] Input: `work/inventory_prepared.csv` (Stage 2 output)
- [ ] Output: `work/classified/pages.csv` (Stage 4 input)
- [ ] Output: `work/indices/*.json` (Stage 4 input)
- [ ] Source type detection for all manual types (service, electrical, ECU)
- [ ] Index parsing with Claude Sonnet API
- [ ] Content type classification with Claude Haiku Vision API
- [ ] page_to_procedures mapping generated
- [ ] Comprehensive error logging
- [ ] Cost tracking (optional token logging)
- [ ] Ready for Stage 4 Q&A generation

---

**IMPORTANT**: This stage is critical for Stage 4 Q&A generation. The source_type determines which prompts to use, and the index metadata provides context for richer Q&A pairs. Vision-based classification ensures accurate content type detection for routing pages to appropriate generation templates.

**Test thoroughly**: Sample manual review of classification results is essential before proceeding to Stage 4. Verify that:
- Procedure pages have photos with numbered callouts
- Troubleshooting pages have 3-column table format
- Specification pages have data tables with values/units
- Diagram pages are full-page technical illustrations
