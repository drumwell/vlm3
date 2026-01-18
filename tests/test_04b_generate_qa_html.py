"""
Tests for Stage 4b: HTML Q&A Generation

Run with: pytest tests/test_04b_generate_qa_html.py -v
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import importlib.util

# Dynamic import to handle the numeric prefix in module name
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

generate_qa_html = load_module("generate_qa_html", Path(__file__).parent.parent / "scripts" / "04b_generate_qa_html.py")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_m3_html(tmp_path):
    """Create sample M3 techspec HTML"""
    html_content = """<!DOCTYPE html>
<html>
<head><title>BMW E30 M3 Technical Specifications</title></head>
<body>
<h1>BMW E30 M3 Technical Specifications</h1>
<table>
    <tr><th>Category</th><th>Specification</th><th>Value</th></tr>
    <tr><td>Engine</td><td>Displacement</td><td>2302 cc</td></tr>
    <tr><td>Engine</td><td>Bore x Stroke</td><td>93.4 x 84.0 mm</td></tr>
    <tr><td>Engine</td><td>Compression Ratio</td><td>10.5:1</td></tr>
    <tr><td>Engine</td><td>Power Output</td><td>195 hp @ 6750 rpm</td></tr>
    <tr><td>Engine</td><td>Torque</td><td>240 Nm @ 4750 rpm</td></tr>
    <tr><td>Transmission</td><td>Type</td><td>Getrag 265/5</td></tr>
    <tr><td>Transmission</td><td>Gear Ratios 1st</td><td>3.72:1</td></tr>
    <tr><td>Dimensions</td><td>Length</td><td>4360 mm</td></tr>
    <tr><td>Dimensions</td><td>Width</td><td>1680 mm</td></tr>
    <tr><td>Performance</td><td>Top Speed</td><td>235 km/h</td></tr>
    <tr><td>Performance</td><td>0-100 km/h</td><td>6.7 seconds</td></tr>
</table>
</body>
</html>"""

    html_path = tmp_path / "M3-techspec.html"
    html_path.write_text(html_content)
    return html_path


@pytest.fixture
def sample_320is_html(tmp_path):
    """Create sample 320is techspec HTML"""
    html_content = """<!DOCTYPE html>
<html>
<head><title>BMW E30 320is Technical Specifications</title></head>
<body>
<h1>BMW E30 320is Technical Specifications</h1>
<table>
    <tr><th>Specification</th><th>Value</th></tr>
    <tr><td>Displacement</td><td>1990 cc</td></tr>
    <tr><td>Power Output</td><td>192 hp @ 6900 rpm</td></tr>
</table>
</body>
</html>"""

    html_path = tmp_path / "320is-techspec.html"
    html_path.write_text(html_content)
    return html_path


# =============================================================================
# Test: HTML Parsing
# =============================================================================

class TestParseHTMLSpecs:
    """Test HTML spec table parsing"""

    def test_parse_m3_specs(self, sample_m3_html):
        """Should parse M3 techspec table"""
        specs = generate_qa_html.parse_html_specs(sample_m3_html)

        assert len(specs) >= 10

        # Check specific specs exist
        displacement = next((s for s in specs if "displacement" in s["spec_name"].lower()), None)
        assert displacement is not None
        assert displacement["spec_value"] == "2302 cc"

        power = next((s for s in specs if "power" in s["spec_name"].lower()), None)
        assert power is not None
        assert "195 hp" in power["spec_value"]

    def test_parse_specs_with_categories(self, sample_m3_html):
        """Should extract categories when present"""
        specs = generate_qa_html.parse_html_specs(sample_m3_html)

        engine_specs = [s for s in specs if s.get("category") == "Engine"]
        assert len(engine_specs) >= 4

    def test_parse_missing_file(self, tmp_path):
        """Should raise error for missing HTML file"""
        with pytest.raises(FileNotFoundError):
            generate_qa_html.parse_html_specs(tmp_path / "nonexistent.html")

    def test_parse_empty_table(self, tmp_path):
        """Should return empty list for HTML with no tables"""
        html_path = tmp_path / "empty.html"
        html_path.write_text("<html><body><p>No tables here</p></body></html>")

        specs = generate_qa_html.parse_html_specs(html_path)

        assert specs == []

    def test_parse_two_column_table(self, sample_320is_html):
        """Should parse 2-column tables (no category)"""
        specs = generate_qa_html.parse_html_specs(sample_320is_html)

        assert len(specs) >= 2
        # Check that spec_name and spec_value are populated
        for spec in specs:
            assert spec["spec_name"]
            assert spec["spec_value"]


# =============================================================================
# Test: Model Detection
# =============================================================================

class TestDetectModelFromFilename:
    """Test vehicle model detection from filename"""

    def test_detect_m3(self, sample_m3_html):
        """Should detect M3 from filename"""
        model = generate_qa_html.detect_model_from_filename(sample_m3_html)

        assert model == "M3"

    def test_detect_320is(self, sample_320is_html):
        """Should detect 320is from filename"""
        model = generate_qa_html.detect_model_from_filename(sample_320is_html)

        assert model == "320is"

    def test_detect_unknown(self, tmp_path):
        """Should return 'E30' for unknown filename pattern"""
        html_path = tmp_path / "specs.html"
        html_path.write_text("<html></html>")

        model = generate_qa_html.detect_model_from_filename(html_path)

        assert model in ["E30", "unknown", "BMW"]


# =============================================================================
# Test: Q&A Variation Generation
# =============================================================================

class TestGenerateQAVariations:
    """Test Q&A variation generation"""

    def test_generate_basic_variations(self):
        """Should generate 2-3 variations per spec"""
        qa_pairs = generate_qa_html.generate_qa_variations(
            spec_name="Displacement",
            spec_value="2302 cc",
            category="Engine",
            model="M3"
        )

        assert len(qa_pairs) >= 2
        assert len(qa_pairs) <= 4

    def test_variation_includes_model(self):
        """Should include model name in some variations"""
        qa_pairs = generate_qa_html.generate_qa_variations(
            spec_name="Power Output",
            spec_value="195 hp",
            category="Engine",
            model="M3"
        )

        # At least one variation should mention M3
        has_model = any("M3" in qa["question"] or "M3" in qa["answer"] for qa in qa_pairs)
        assert has_model

    def test_variation_question_types(self):
        """All variations should have question_type=factual"""
        qa_pairs = generate_qa_html.generate_qa_variations(
            spec_name="Torque",
            spec_value="240 Nm",
            category="Engine",
            model="M3"
        )

        assert all(qa.get("question_type") == "factual" for qa in qa_pairs)

    def test_variation_without_category(self):
        """Should work without category"""
        qa_pairs = generate_qa_html.generate_qa_variations(
            spec_name="Top Speed",
            spec_value="235 km/h",
            category="",
            model="M3"
        )

        assert len(qa_pairs) >= 2


# =============================================================================
# Test: Q&A ID Assignment
# =============================================================================

class TestAssignQAIds:
    """Test Q&A ID assignment"""

    def test_assign_sequential_ids(self):
        """Should assign sequential IDs"""
        qa_pairs = [
            {"question": "Q1?", "answer": "A1.", "question_type": "factual"},
            {"question": "Q2?", "answer": "A2.", "question_type": "factual"},
        ]

        result = generate_qa_html.assign_qa_ids(qa_pairs, "html-m3-techspec")

        assert result[0]["id"] == "html-m3-techspec-q01"
        assert result[1]["id"] == "html-m3-techspec-q02"


# =============================================================================
# Test: Full HTML Q&A Generation
# =============================================================================

class TestGenerateHTMLQA:
    """Test full HTML Q&A generation"""

    def test_generate_m3_qa(self, sample_m3_html, tmp_path):
        """Should generate Q&A from M3 techspec"""
        output_dir = tmp_path / "qa_raw"

        result = generate_qa_html.generate_html_qa(sample_m3_html, output_dir)

        assert result["page_id"] == "html-m3-techspec"
        assert result["qa_count"] >= 20  # 10+ specs * 2+ variations
        assert result["output_path"].exists()

    def test_output_schema(self, sample_m3_html, tmp_path):
        """Should write correct output schema"""
        output_dir = tmp_path / "qa_raw"
        result = generate_qa_html.generate_html_qa(sample_m3_html, output_dir)

        with open(result["output_path"]) as f:
            data = json.load(f)

        assert data["page_id"] == "html-m3-techspec"
        assert data["image_path"] is None
        assert data["source_type"] == "html_specs"
        assert data["content_type"] == "specification"
        assert "generation" in data
        assert data["generation"]["method"] == "html_parse"
        assert "qa_pairs" in data
        assert len(data["qa_pairs"]) > 0

    def test_qa_pairs_have_ids(self, sample_m3_html, tmp_path):
        """Should assign IDs to Q&A pairs"""
        output_dir = tmp_path / "qa_raw"
        result = generate_qa_html.generate_html_qa(sample_m3_html, output_dir)

        with open(result["output_path"]) as f:
            data = json.load(f)

        for qa in data["qa_pairs"]:
            assert "id" in qa
            assert qa["id"].startswith("html-m3-techspec-q")


# =============================================================================
# Test: Idempotency
# =============================================================================

class TestIdempotency:
    """Test idempotent behavior"""

    def test_skip_existing_output(self, sample_m3_html, tmp_path):
        """Should skip if output already exists"""
        output_dir = tmp_path / "qa_raw"

        # First run
        result1 = generate_qa_html.generate_html_qa(sample_m3_html, output_dir)
        mtime1 = result1["output_path"].stat().st_mtime

        # Wait a bit
        import time
        time.sleep(0.1)

        # Second run (should skip)
        result2 = generate_qa_html.generate_html_qa(sample_m3_html, output_dir)
        mtime2 = result2["output_path"].stat().st_mtime

        # File should not be modified
        assert mtime1 == mtime2

    def test_force_regenerate(self, sample_m3_html, tmp_path):
        """Should regenerate with force=True"""
        output_dir = tmp_path / "qa_raw"

        # First run
        result1 = generate_qa_html.generate_html_qa(sample_m3_html, output_dir)
        mtime1 = result1["output_path"].stat().st_mtime

        import time
        time.sleep(0.1)

        # Second run with force
        result2 = generate_qa_html.generate_html_qa(sample_m3_html, output_dir, force=True)
        mtime2 = result2["output_path"].stat().st_mtime

        assert mtime2 > mtime1


# =============================================================================
# Test: CLI
# =============================================================================

class TestCLI:
    """Test command-line interface"""

    def test_help_output(self):
        """Should show help with --help"""
        import subprocess
        script_path = Path(__file__).parent.parent / "scripts" / "04b_generate_qa_html.py"
        if not script_path.exists():
            pytest.skip("Script not implemented yet")

        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert '--data-src' in result.stdout
        assert '--output' in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
