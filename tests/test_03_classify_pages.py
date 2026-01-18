"""
Tests for Stage 3: Classification & Index Parsing

Run with: pytest tests/test_03_classify_pages.py -v
"""

import pytest
import json
import csv
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import importlib.util

# Dynamic import to handle the numeric prefix in module name
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

classify = load_module("classify", Path(__file__).parent.parent / "scripts" / "03_classify_pages.py")


class TestSourceTypeDetection:
    """Test source type detection from directory names"""

    def test_service_manual_numbered_section(self):
        """Should detect service manual from numbered section"""
        assert classify.detect_source_type("21 - Clutch", "21-01.jpg") == "service_manual"
        assert classify.detect_source_type("00 - Maintenance", "00-01.jpg") == "service_manual"
        assert classify.detect_source_type("97 - Troubleshooting", "97-01.jpg") == "service_manual"

    def test_service_manual_getrag(self):
        """Should detect Getrag PDF conversion as service manual"""
        assert classify.detect_source_type("Getrag265", "001.jpg") == "service_manual"

    def test_electrical_manual(self):
        """Should detect electrical troubleshooting manual"""
        assert classify.detect_source_type(
            "1990 BMW M3 Electrical Troubleshooting Manual",
            "etm-001.jpg"
        ) == "electrical_manual"

    def test_ecu_technical(self):
        """Should detect Bosch Motronic ECU docs"""
        assert classify.detect_source_type(
            "Bosch Motronic ML 3-1",
            "bosch-001.jpg"
        ) == "ecu_technical"

    def test_unknown_source(self):
        """Should return 'unknown' for unrecognized directories"""
        assert classify.detect_source_type("Random Folder", "file.jpg") == "unknown"

    def test_html_specs(self):
        """Should detect HTML spec files"""
        assert classify.detect_source_type("", "M3-techspec.html") == "html_specs"
        assert classify.detect_source_type("21 - Clutch", "index.html") == "html_specs"


class TestIndexPageDetection:
    """Test index page identification by filename patterns"""

    def test_index_page_dash_index(self):
        """Should detect *-index-* pattern"""
        assert classify.is_index_page("21-00-index-a.jpg") == True
        assert classify.is_index_page("21-00-index-b.jpg") == True
        assert classify.is_index_page("00-index.jpg") == True

    def test_index_page_toc(self):
        """Should detect *-toc-* pattern"""
        assert classify.is_index_page("21-toc.jpg") == True
        assert classify.is_index_page("section-toc-page1.jpg") == True

    def test_non_index_page(self):
        """Should return False for regular pages"""
        assert classify.is_index_page("21-01.jpg") == False
        assert classify.is_index_page("21-02.jpg") == False
        assert classify.is_index_page("clutch-diagram.jpg") == False

    def test_case_insensitive(self):
        """Should be case-insensitive"""
        assert classify.is_index_page("21-INDEX-A.jpg") == True
        assert classify.is_index_page("21-TOC.jpg") == True


class TestPageIDGeneration:
    """Test page ID generation logic"""

    def test_generate_page_id_numbered_section(self):
        """Should include section suffix for disambiguation"""
        assert classify.generate_page_id("21-03.jpg", "21 - Clutch") == "21-03_clutch"
        assert classify.generate_page_id("00-01.jpg", "00 - Maintenance") == "00-01_maintena"

    def test_generate_page_id_different_sections_same_prefix(self):
        """Different sections with same number prefix should have unique IDs"""
        id1 = classify.generate_page_id("00-01.jpg", "00 - Maintenance")
        id2 = classify.generate_page_id("00-01.jpg", "00 - Torque Specs")
        assert id1 != id2
        assert id1 == "00-01_maintena"
        assert id2 == "00-01_torque-s"

    def test_generate_page_id_index_page(self):
        """Should handle index page filenames"""
        assert classify.generate_page_id("21-00-index-a.jpg", "21 - Clutch") == "21-00-index-a_clutch"

    def test_generate_page_id_getrag(self):
        """Should use 'getrag' prefix for Getrag PDF conversion"""
        assert classify.generate_page_id("001.jpg", "Getrag265") == "getrag-001"

    def test_generate_page_id_non_standard(self):
        """Should use section slug for non-standard sections"""
        assert classify.generate_page_id("etm-001.jpg", "1990 BMW M3 Electrical Troubleshooting Manual").startswith("1990-bmw-m3")


class TestSectionSlugGeneration:
    """Test section slug generation for filesystem-safe names"""

    def test_numbered_section(self):
        """Should handle numbered sections"""
        assert classify.generate_section_slug("21 - Clutch") == "21-clutch"
        assert classify.generate_section_slug("00 - Maintenance") == "00-maintenance"

    def test_multi_word_section(self):
        """Should handle multi-word section names"""
        assert classify.generate_section_slug("00 - Torque Specs") == "00-torque-specs"
        assert classify.generate_section_slug("12 - Engine Electrical Equipment") == "12-engine-electrical-equipment"

    def test_long_name(self):
        """Should handle long directory names"""
        assert classify.generate_section_slug("1990 BMW M3 Electrical Troubleshooting Manual") == "1990-bmw-m3-electrical-troubleshooting-manual"

    def test_special_characters(self):
        """Should strip special characters except hyphens"""
        assert classify.generate_section_slug("Bosch Motronic ML 3-1") == "bosch-motronic-ml-3-1"

    def test_no_spaces(self):
        """Should handle names without spaces"""
        assert classify.generate_section_slug("Getrag265") == "getrag265"

    def test_already_slugified(self):
        """Should be idempotent"""
        assert classify.generate_section_slug("21-clutch") == "21-clutch"


class TestCacheLoading:
    """Test loading cached classification results"""

    def test_load_cached_results_exists(self, tmp_path):
        """Should load existing cache file"""
        cache_path = tmp_path / "pages.csv"
        cache_path.write_text("""page_id,image_path,section_id,section_name,source_type,content_type,is_index,confidence,secondary_types,needs_review
21-01,data_src/21 - Clutch/21-01.jpg,21,Clutch,service_manual,procedure,False,0.85,[],False
21-02,data_src/21 - Clutch/21-02.jpg,21,Clutch,service_manual,specification,False,0.90,[],False
""")

        cached = classify.load_cached_results(cache_path)

        assert len(cached) == 2
        assert "21-01" in cached
        assert "21-02" in cached
        assert cached["21-01"]["content_type"] == "procedure"
        assert cached["21-02"]["content_type"] == "specification"

    def test_load_cached_results_missing_file(self, tmp_path):
        """Should return empty dict for missing file"""
        cache_path = tmp_path / "nonexistent.csv"

        cached = classify.load_cached_results(cache_path)

        assert cached == {}

    def test_load_cached_results_empty_file(self, tmp_path):
        """Should return empty dict for empty file"""
        cache_path = tmp_path / "empty.csv"
        cache_path.write_text("")

        cached = classify.load_cached_results(cache_path)

        assert cached == {}


class TestConfigLoading:
    """Test configuration file loading"""

    def test_load_config_valid(self, tmp_path):
        """Should load valid YAML config"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text("""
api:
  provider: anthropic
  models:
    index_parsing: claude-sonnet-4-20250514
    classification: claude-3-haiku-20240307

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

        config = classify.load_config(config_path)
        assert config["api"]["models"]["index_parsing"] == "claude-sonnet-4-20250514"
        assert "^[0-9]{2} - " in config["sources"]["service_manual_patterns"]

    def test_load_config_missing_file(self, tmp_path):
        """Should raise error for missing config file"""
        with pytest.raises(FileNotFoundError):
            classify.load_config(tmp_path / "nonexistent.yaml")


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

        records = classify.read_prepared_inventory(inventory_path)
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

        records = classify.read_prepared_inventory(inventory_path)
        assert len(records) == 1  # HTML filtered out
        assert records[0]["file_type"] == "jpg"


class TestContentClassification:
    """Test content type classification"""

    def test_classify_index_page_skip_api(self):
        """Index pages detected by filename should skip API call"""
        result = classify.classify_page_content(
            image_path=Path("data_src/21 - Clutch/21-00-index-a.jpg"),
            source_type="service_manual",
            is_index=True,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "index"
        assert result["confidence"] >= 0.95
        assert result["api_called"] == False

    @patch('anthropic.Anthropic')
    def test_classify_procedure_page_via_api(self, mock_anthropic_class, tmp_path):
        """Should classify procedure page using vision API"""
        # Create a test image
        from PIL import Image
        img_path = tmp_path / "21-01.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path, "JPEG")

        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text=json.dumps({
            "content_type": "procedure",
            "confidence": 0.92,
            "secondary_types": [],
            "reasoning": "Multiple photos with numbered callouts and step-by-step text"
        }))]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        result = classify.classify_page_content(
            image_path=img_path,
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "procedure"
        assert result["confidence"] >= 0.9
        assert result["api_called"] == True

    def test_classify_missing_image_returns_error(self):
        """Should return error for missing image"""
        result = classify.classify_page_content(
            image_path=Path("/nonexistent/path/image.jpg"),
            source_type="service_manual",
            is_index=False,
            api_key="test_key",
            model="claude-haiku-3"
        )
        assert result["content_type"] == "unknown"
        assert result["error"] is not None


class TestIndexMetadataMerging:
    """Test merging of multi-page index metadata"""

    def test_merge_single_page(self):
        """Should handle single-page index"""
        index_results = {
            "21-clutch": [
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-a.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}
                    ],
                    "error": None
                }
            ]
        }

        merged = classify.merge_index_metadata(index_results)
        assert merged["21-clutch"]["section_id"] == "21"
        assert len(merged["21-clutch"]["procedures"]) == 1

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
                    ],
                    "error": None
                },
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-b.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"}
                    ],
                    "error": None
                }
            ]
        }

        merged = classify.merge_index_metadata(index_results)

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
                    ],
                    "error": None
                },
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-b.jpg",
                    "procedures": [
                        {"code": "21 21 000", "name": "Clutch disc - remove", "page": "21-3"}
                    ],
                    "error": None
                }
            ]
        }

        merged = classify.merge_index_metadata(index_results)

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
                    ],
                    "error": None
                },
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-b.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch bleeding procedure", "page": "21-1"}
                    ],
                    "error": None
                }
            ]
        }

        merged = classify.merge_index_metadata(index_results)

        assert merged["21-clutch"]["procedures"][0]["name"] == "Clutch - bleed"
        assert len(merged["21-clutch"]["merge_warnings"]) == 1
        assert "21 00 006" in merged["21-clutch"]["merge_warnings"][0]

    def test_merge_generates_page_to_procedures(self):
        """Should generate page_to_procedures mapping"""
        index_results = {
            "21-clutch": [
                {
                    "section_id": "21",
                    "section_name": "Clutch",
                    "source_filename": "21-00-index-a.jpg",
                    "procedures": [
                        {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"},
                        {"code": "21 11 000", "name": "Clutch housing", "page": "21-1"},
                        {"code": "21 21 000", "name": "Clutch disc", "page": "21-2"}
                    ],
                    "error": None
                }
            ]
        }

        merged = classify.merge_index_metadata(index_results)
        page_mapping = merged["21-clutch"]["page_to_procedures"]

        assert len(page_mapping["21-1"]) == 2
        assert "21 00 006" in page_mapping["21-1"]
        assert "21 11 000" in page_mapping["21-1"]
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
                "confidence": 0.85,
                "secondary_types": "[]",
                "needs_review": False
            }
        ]

        classify.write_classification_csv(records, output_path)

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
             "source_type": "service_manual", "content_type": "procedure", "is_index": False,
             "confidence": 0.85, "secondary_types": "[]", "needs_review": False},
            {"page_id": "21-01", "image_path": "...", "section_id": "21", "section_name": "Clutch",
             "source_type": "service_manual", "content_type": "index", "is_index": True,
             "confidence": 0.95, "secondary_types": "[]", "needs_review": False},
            {"page_id": "21-02", "image_path": "...", "section_id": "21", "section_name": "Clutch",
             "source_type": "service_manual", "content_type": "procedure", "is_index": False,
             "confidence": 0.85, "secondary_types": "[]", "needs_review": False}
        ]

        classify.write_classification_csv(records, output_path)

        with open(output_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
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
                    "pages": ["21-1"]
                }
            ],
            "page_to_procedures": {
                "21-1": ["21 00 006"]
            },
            "merge_warnings": []
        }

        classify.write_index_json(metadata, output_path)

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
                "timestamp": "2025-01-15T10:30:00",
                "filename": "21-01.jpg",
                "operation": "classification",
                "error_message": "API timeout"
            }
        ]

        classify.write_error_log(errors, log_path)

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
            {"page_id": "21-01", "source_type": "service_manual", "content_type": "procedure",
             "confidence": 0.85, "is_index": False},
            {"page_id": "21-02", "source_type": "service_manual", "content_type": "specification",
             "confidence": 0.90, "is_index": False},
            {"page_id": "etm-01", "source_type": "electrical_manual", "content_type": "wiring",
             "confidence": 0.88, "is_index": False}
        ]

        index_metadata = {
            "21-clutch": {"section_id": "21", "procedures": [{}, {}], "page_to_procedures": {}},
            "11-engine": {"section_id": "11", "procedures": [{}, {}, {}], "page_to_procedures": {}}
        }

        errors = []

        report_path = tmp_path / "report.md"
        classify.generate_report(classification_records, index_metadata, errors, report_path)

        report_text = report_path.read_text()
        assert "Total Pages Classified" in report_text
        assert "service_manual" in report_text
        assert "electrical_manual" in report_text
        assert "Sections with Indices**: 2" in report_text
        assert "Total Procedures Extracted**: 5" in report_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
