#!/usr/bin/env python3
"""
Tests for Stage 2: Source Preparation

Following TDD approach - these tests define the expected behavior
of the source preparation script before implementation.
"""

import pytest
import csv
from pathlib import Path
from PIL import Image

# Import from scripts package using clean aliases
from scripts import prepare_sources


def get_func(name):
    """Get function from module or return None"""
    return getattr(prepare_sources, name, None)


# Helper fixtures
@pytest.fixture
def temp_data_src(tmp_path):
    """Create temporary data_src directory"""
    data_src = tmp_path / "data_src"
    data_src.mkdir()
    return data_src


@pytest.fixture
def temp_work(tmp_path):
    """Create temporary work directory"""
    work = tmp_path / "work"
    work.mkdir()
    (work / "logs").mkdir()
    return work


def create_test_image(path: Path, width: int = 100, height: int = 100):
    """Create a simple test image"""
    img = Image.new('RGB', (width, height), color='white')
    img.save(path)


def create_test_pdf(path: Path, num_pages: int = 1):
    """Create a simple test PDF using reportlab if available, otherwise use raw PDF"""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(str(path), pagesize=letter)
        for i in range(num_pages):
            c.drawString(100, 750, f"Test Page {i + 1}")
            if i < num_pages - 1:
                c.showPage()
        c.save()
    except ImportError:
        # Create minimal valid PDF without reportlab
        # This is a minimal valid PDF structure
        pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids ["""

        # Add page references
        page_refs = " ".join([f"{3 + i*2} 0 R" for i in range(num_pages)])
        pdf_content += page_refs.encode()
        pdf_content += f"""] /Count {num_pages} >>
endobj
""".encode()

        obj_num = 3
        for i in range(num_pages):
            # Page object
            pdf_content += f"""{obj_num} 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {obj_num + 1} 0 R >>
endobj
""".encode()
            obj_num += 1
            # Content stream
            content = f"BT /F1 12 Tf 100 700 Td (Test Page {i + 1}) Tj ET"
            pdf_content += f"""{obj_num} 0 obj
<< /Length {len(content)} >>
stream
{content}
endstream
endobj
""".encode()
            obj_num += 1

        pdf_content += b"""xref
0 1
0000000000 65535 f
trailer
<< /Size 1 /Root 1 0 R >>
startxref
0
%%EOF"""

        path.write_bytes(pdf_content)


def create_inventory_csv(path: Path, records: list):
    """Create an inventory CSV file"""
    fieldnames = ['file_path', 'file_type', 'section_dir', 'filename', 'needs_conversion']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def read_csv_as_list(csv_path: Path) -> list:
    """Helper to read CSV file as list of dicts"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


class TestImageValidation:
    """Test image validation logic"""

    def test_validate_image_valid_jpg(self, tmp_path):
        """Should return True for valid JPG"""
        validate_image = get_func('validate_image')
        if validate_image is None:
            pytest.skip("Module not implemented yet")

        # Create valid JPG
        img_path = tmp_path / "test.jpg"
        create_test_image(img_path)

        is_valid, error = validate_image(img_path)
        assert is_valid is True
        assert error is None

    def test_validate_image_valid_png(self, tmp_path):
        """Should return True for valid PNG"""
        validate_image = get_func('validate_image')
        if validate_image is None:
            pytest.skip("Module not implemented yet")

        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(img_path)

        is_valid, error = validate_image(img_path)
        assert is_valid is True
        assert error is None

    def test_validate_image_corrupted_file(self, tmp_path):
        """Should return False for corrupted image"""
        validate_image = get_func('validate_image')
        if validate_image is None:
            pytest.skip("Module not implemented yet")

        # Create file with .jpg extension but invalid data
        img_path = tmp_path / "corrupted.jpg"
        img_path.write_bytes(b"not a real image content here")

        is_valid, error = validate_image(img_path)
        assert is_valid is False
        assert error is not None

    def test_validate_image_missing_file(self, tmp_path):
        """Should return False for non-existent file"""
        validate_image = get_func('validate_image')
        if validate_image is None:
            pytest.skip("Module not implemented yet")

        img_path = tmp_path / "nonexistent.jpg"

        is_valid, error = validate_image(img_path)
        assert is_valid is False
        assert error is not None

    def test_validate_image_empty_file(self, tmp_path):
        """Should return False for 0-byte image file"""
        validate_image = get_func('validate_image')
        if validate_image is None:
            pytest.skip("Module not implemented yet")

        img_path = tmp_path / "empty.jpg"
        img_path.touch()  # Create empty file

        is_valid, error = validate_image(img_path)
        assert is_valid is False
        assert error is not None


class TestPDFConversion:
    """Test PDF to JPG conversion"""

    def test_convert_pdf_single_page(self, tmp_path):
        """Should convert 1-page PDF to 001.jpg"""
        convert_pdf_to_jpgs = get_func('convert_pdf_to_jpgs')
        if convert_pdf_to_jpgs is None:
            pytest.skip("Module not implemented yet")

        # Create a simple 1-page PDF
        pdf_path = tmp_path / "single.pdf"
        create_test_pdf(pdf_path, num_pages=1)

        output_dir = tmp_path / "output"

        success, output_paths, error, was_skipped = convert_pdf_to_jpgs(pdf_path, output_dir)

        assert success is True
        assert error is None
        assert was_skipped is False
        assert output_dir.exists()
        assert len(output_paths) == 1
        assert output_paths[0].name == "001.jpg"

        # Verify image is readable
        img = Image.open(output_paths[0])
        assert img.width > 0
        assert img.height > 0

    def test_convert_pdf_multi_page(self, tmp_path):
        """Should convert 3-page PDF to 001.jpg through 003.jpg"""
        convert_pdf_to_jpgs = get_func('convert_pdf_to_jpgs')
        if convert_pdf_to_jpgs is None:
            pytest.skip("Module not implemented yet")

        pdf_path = tmp_path / "multi.pdf"
        create_test_pdf(pdf_path, num_pages=3)

        output_dir = tmp_path / "output"

        success, output_paths, error, was_skipped = convert_pdf_to_jpgs(pdf_path, output_dir)

        assert success is True
        assert was_skipped is False
        assert len(output_paths) == 3

        expected_names = ["001.jpg", "002.jpg", "003.jpg"]
        actual_names = [p.name for p in output_paths]
        assert actual_names == expected_names

        # All should be readable
        for p in output_paths:
            img = Image.open(p)
            assert img.width > 0

    def test_convert_pdf_output_directory_created(self, tmp_path):
        """Should create output directory if it doesn't exist"""
        convert_pdf_to_jpgs = get_func('convert_pdf_to_jpgs')
        if convert_pdf_to_jpgs is None:
            pytest.skip("Module not implemented yet")

        pdf_path = tmp_path / "test.pdf"
        create_test_pdf(pdf_path, num_pages=1)

        # Nested output directory that doesn't exist
        output_dir = tmp_path / "deeply" / "nested" / "output"

        success, output_paths, error, was_skipped = convert_pdf_to_jpgs(pdf_path, output_dir)

        assert success is True
        assert output_dir.exists()
        assert len(list(output_dir.glob("*.jpg"))) == 1

    def test_convert_pdf_already_exists_skip(self, tmp_path):
        """Should skip conversion if output directory already has images"""
        convert_pdf_to_jpgs = get_func('convert_pdf_to_jpgs')
        if convert_pdf_to_jpgs is None:
            pytest.skip("Module not implemented yet")

        pdf_path = tmp_path / "test.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        output_dir = tmp_path / "output"

        # First conversion
        success1, paths1, error1, was_skipped1 = convert_pdf_to_jpgs(pdf_path, output_dir)
        assert success1 is True
        assert was_skipped1 is False

        # Record modification time
        first_mtime = paths1[0].stat().st_mtime

        # Second conversion (should skip)
        success2, paths2, error2, was_skipped2 = convert_pdf_to_jpgs(pdf_path, output_dir)

        # Should still succeed but indicate skipped
        assert success2 is True
        assert was_skipped2 is True
        # Original files should be unchanged
        assert paths2[0].stat().st_mtime == first_mtime

    def test_convert_pdf_force_reconvert(self, tmp_path):
        """Should reconvert if force=True even when output exists"""
        convert_pdf_to_jpgs = get_func('convert_pdf_to_jpgs')
        if convert_pdf_to_jpgs is None:
            pytest.skip("Module not implemented yet")

        pdf_path = tmp_path / "test.pdf"
        create_test_pdf(pdf_path, num_pages=1)

        output_dir = tmp_path / "output"

        # First conversion
        success1, paths1, _, was_skipped1 = convert_pdf_to_jpgs(pdf_path, output_dir)
        assert success1 is True
        assert was_skipped1 is False
        first_mtime = paths1[0].stat().st_mtime

        # Wait a tiny bit to ensure mtime difference
        import time
        time.sleep(0.1)

        # Second conversion with force
        success2, paths2, _, was_skipped2 = convert_pdf_to_jpgs(pdf_path, output_dir, force=True)
        assert success2 is True
        assert was_skipped2 is False
        # Files should be newer
        assert paths2[0].stat().st_mtime > first_mtime

    def test_convert_pdf_invalid_pdf_returns_error(self, tmp_path):
        """Should return error for corrupted PDF"""
        convert_pdf_to_jpgs = get_func('convert_pdf_to_jpgs')
        if convert_pdf_to_jpgs is None:
            pytest.skip("Module not implemented yet")

        # Create corrupted PDF file
        pdf_path = tmp_path / "corrupted.pdf"
        pdf_path.write_bytes(b"not a real PDF content")

        output_dir = tmp_path / "output"

        success, paths, error, was_skipped = convert_pdf_to_jpgs(pdf_path, output_dir)

        assert success is False
        assert error is not None
        assert len(paths) == 0


class TestInventoryReading:
    """Test inventory reading"""

    def test_read_inventory(self, temp_data_src, temp_work):
        """Should read inventory CSV correctly"""
        read_inventory = get_func('read_inventory')
        if read_inventory is None:
            pytest.skip("Module not implemented yet")

        # Create test inventory
        inventory_path = temp_work / "inventory.csv"
        records = [
            {'file_path': str(temp_data_src / "test.jpg"), 'file_type': 'jpg',
             'section_dir': '', 'filename': 'test.jpg', 'needs_conversion': 'false'},
            {'file_path': str(temp_data_src / "doc.pdf"), 'file_type': 'pdf',
             'section_dir': '', 'filename': 'doc.pdf', 'needs_conversion': 'true'},
        ]
        create_inventory_csv(inventory_path, records)

        result = read_inventory(inventory_path)

        assert len(result) == 2
        assert result[0]['file_type'] == 'jpg'
        assert result[1]['needs_conversion'] == 'true'


class TestInventoryProcessing:
    """Test inventory processing logic"""

    def test_process_inventory_validates_images(self, temp_data_src, temp_work):
        """Should validate existing JPG/PNG files"""
        process_inventory = get_func('process_inventory')
        if process_inventory is None:
            pytest.skip("Module not implemented yet")

        # Create test images
        img1 = temp_data_src / "image1.jpg"
        img2 = temp_data_src / "image2.png"
        create_test_image(img1)
        create_test_image(img2)

        records = [
            {'file_path': str(img1), 'file_type': 'jpg',
             'section_dir': '', 'filename': 'image1.jpg', 'needs_conversion': 'false'},
            {'file_path': str(img2), 'file_type': 'png',
             'section_dir': '', 'filename': 'image2.png', 'needs_conversion': 'false'},
        ]

        log_entries, prepared_records = process_inventory(records, temp_data_src)

        # Both images should be validated
        assert len(prepared_records) == 2

        # Check log entries exist for validations
        validation_logs = [e for e in log_entries if e['operation'] == 'image_validate']
        assert len(validation_logs) == 2
        assert all(e['status'] == 'success' for e in validation_logs)

    def test_process_inventory_converts_pdfs(self, temp_data_src, temp_work):
        """Should convert all PDFs marked needs_conversion=true"""
        process_inventory = get_func('process_inventory')
        if process_inventory is None:
            pytest.skip("Module not implemented yet")

        # Create test PDF
        pdf_path = temp_data_src / "document.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        records = [
            {'file_path': str(pdf_path), 'file_type': 'pdf',
             'section_dir': '', 'filename': 'document.pdf', 'needs_conversion': 'true'},
        ]

        log_entries, prepared_records = process_inventory(records, temp_data_src)

        # PDF should be converted to 2 JPGs
        assert len(prepared_records) == 2
        assert all(r['file_type'] == 'jpg' for r in prepared_records)

        # Check original_source is set
        assert all(r['original_source'] == str(pdf_path) for r in prepared_records)

        # Check log entries
        convert_logs = [e for e in log_entries if e['operation'] == 'pdf_convert']
        assert len(convert_logs) >= 1
        assert any(e['status'] == 'success' for e in convert_logs)

    def test_process_inventory_handles_failures_gracefully(self, temp_data_src, temp_work):
        """Should continue processing after individual failures"""
        process_inventory = get_func('process_inventory')
        if process_inventory is None:
            pytest.skip("Module not implemented yet")

        # Create one good image and one bad PDF
        good_img = temp_data_src / "good.jpg"
        create_test_image(good_img)

        bad_pdf = temp_data_src / "bad.pdf"
        bad_pdf.write_bytes(b"not a real PDF")

        records = [
            {'file_path': str(good_img), 'file_type': 'jpg',
             'section_dir': '', 'filename': 'good.jpg', 'needs_conversion': 'false'},
            {'file_path': str(bad_pdf), 'file_type': 'pdf',
             'section_dir': '', 'filename': 'bad.pdf', 'needs_conversion': 'true'},
        ]

        log_entries, prepared_records = process_inventory(records, temp_data_src)

        # Good image should still be in output
        assert len(prepared_records) >= 1
        assert any(r['filename'] == 'good.jpg' for r in prepared_records)

        # Failure should be logged
        failure_logs = [e for e in log_entries if e['status'] == 'failure']
        assert len(failure_logs) >= 1

    def test_process_inventory_html_passed_through(self, temp_data_src, temp_work):
        """Should pass HTML files through unchanged"""
        process_inventory = get_func('process_inventory')
        if process_inventory is None:
            pytest.skip("Module not implemented yet")

        # Create HTML file
        html_path = temp_data_src / "index.html"
        html_path.write_text("<html><body>Test</body></html>")

        records = [
            {'file_path': str(html_path), 'file_type': 'html',
             'section_dir': '', 'filename': 'index.html', 'needs_conversion': 'false'},
        ]

        log_entries, prepared_records = process_inventory(records, temp_data_src)

        # HTML should be in output unchanged
        assert len(prepared_records) == 1
        assert prepared_records[0]['file_type'] == 'html'
        assert prepared_records[0]['filename'] == 'index.html'
        assert prepared_records[0]['original_source'] == ''


class TestPreparedInventoryGeneration:
    """Test generation of inventory_prepared.csv"""

    def test_prepared_inventory_schema(self, temp_work):
        """Should output correct CSV schema"""
        write_prepared_inventory = get_func('write_prepared_inventory')
        if write_prepared_inventory is None:
            pytest.skip("Module not implemented yet")

        records = [
            {'file_path': 'data_src/test.jpg', 'file_type': 'jpg',
             'section_dir': '', 'filename': 'test.jpg', 'original_source': ''},
        ]

        output_path = temp_work / "inventory_prepared.csv"
        write_prepared_inventory(records, output_path)

        # Check header
        with open(output_path, 'r') as f:
            header = f.readline().strip()
            assert header == "file_path,file_type,section_dir,filename,original_source"

    def test_prepared_inventory_sorted_by_path(self, temp_work):
        """Should sort output by file_path for reproducibility"""
        write_prepared_inventory = get_func('write_prepared_inventory')
        if write_prepared_inventory is None:
            pytest.skip("Module not implemented yet")

        records = [
            {'file_path': 'data_src/zebra.jpg', 'file_type': 'jpg',
             'section_dir': '', 'filename': 'zebra.jpg', 'original_source': ''},
            {'file_path': 'data_src/apple.jpg', 'file_type': 'jpg',
             'section_dir': '', 'filename': 'apple.jpg', 'original_source': ''},
            {'file_path': 'data_src/mango.jpg', 'file_type': 'jpg',
             'section_dir': '', 'filename': 'mango.jpg', 'original_source': ''},
        ]

        output_path = temp_work / "inventory_prepared.csv"
        write_prepared_inventory(records, output_path)

        result = read_csv_as_list(output_path)
        file_paths = [r['file_path'] for r in result]
        assert file_paths == sorted(file_paths)


class TestConversionLog:
    """Test conversion logging"""

    def test_conversion_log_schema(self, temp_work):
        """Should write log with correct columns"""
        write_conversion_log = get_func('write_conversion_log')
        if write_conversion_log is None:
            pytest.skip("Module not implemented yet")

        log_entries = [
            {
                'timestamp': '2024-01-01T12:00:00',
                'source_file': 'data_src/test.pdf',
                'operation': 'pdf_convert',
                'status': 'success',
                'output_path': 'data_src/test/001.jpg',
                'error_message': ''
            }
        ]

        log_path = temp_work / "logs" / "source_preparation.csv"
        write_conversion_log(log_entries, log_path)

        # Check header
        with open(log_path, 'r') as f:
            header = f.readline().strip()
            expected = "timestamp,source_file,operation,status,output_path,error_message"
            assert header == expected

    def test_conversion_log_failure_entries(self, temp_work):
        """Should log failed conversions with error messages"""
        write_conversion_log = get_func('write_conversion_log')
        if write_conversion_log is None:
            pytest.skip("Module not implemented yet")

        log_entries = [
            {
                'timestamp': '2024-01-01T12:00:00',
                'source_file': 'data_src/bad.pdf',
                'operation': 'pdf_convert',
                'status': 'failure',
                'output_path': '',
                'error_message': 'Invalid PDF format'
            }
        ]

        log_path = temp_work / "logs" / "source_preparation.csv"
        write_conversion_log(log_entries, log_path)

        result = read_csv_as_list(log_path)
        assert len(result) == 1
        assert result[0]['status'] == 'failure'
        assert result[0]['error_message'] == 'Invalid PDF format'


class TestEndToEnd:
    """Integration tests for full workflow"""

    def test_full_pipeline_mixed_sources(self, temp_data_src, temp_work):
        """Should process inventory with mixed file types"""
        process_inventory = get_func('process_inventory')
        write_prepared_inventory = get_func('write_prepared_inventory')
        write_conversion_log = get_func('write_conversion_log')

        if any(f is None for f in [process_inventory, write_prepared_inventory, write_conversion_log]):
            pytest.skip("Module not implemented yet")

        # Create test files
        # 1 PDF with 2 pages
        pdf_path = temp_data_src / "doc.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        # 3 existing JPGs
        for i in range(3):
            img_path = temp_data_src / f"image{i}.jpg"
            create_test_image(img_path)

        # 1 HTML file
        html_path = temp_data_src / "index.html"
        html_path.write_text("<html></html>")

        # Create inventory
        records = [
            {'file_path': str(pdf_path), 'file_type': 'pdf',
             'section_dir': '', 'filename': 'doc.pdf', 'needs_conversion': 'true'},
        ]
        for i in range(3):
            records.append({
                'file_path': str(temp_data_src / f"image{i}.jpg"),
                'file_type': 'jpg',
                'section_dir': '',
                'filename': f'image{i}.jpg',
                'needs_conversion': 'false'
            })
        records.append({
            'file_path': str(html_path), 'file_type': 'html',
            'section_dir': '', 'filename': 'index.html', 'needs_conversion': 'false'
        })

        # Process
        log_entries, prepared_records = process_inventory(records, temp_data_src)

        # Should have: 2 (from PDF) + 3 (existing JPGs) + 1 (HTML) = 6 entries
        assert len(prepared_records) == 6

        # Count by type
        jpg_count = sum(1 for r in prepared_records if r['file_type'] == 'jpg')
        html_count = sum(1 for r in prepared_records if r['file_type'] == 'html')

        assert jpg_count == 5  # 2 from PDF + 3 existing
        assert html_count == 1

        # Write outputs
        output_path = temp_work / "inventory_prepared.csv"
        write_prepared_inventory(prepared_records, output_path)
        assert output_path.exists()

        log_path = temp_work / "logs" / "source_preparation.csv"
        write_conversion_log(log_entries, log_path)
        assert log_path.exists()

    def test_idempotency_rerun_safe(self, temp_data_src, temp_work):
        """Should be safe to rerun on same inventory"""
        process_inventory = get_func('process_inventory')
        write_prepared_inventory = get_func('write_prepared_inventory')

        if any(f is None for f in [process_inventory, write_prepared_inventory]):
            pytest.skip("Module not implemented yet")

        # Create test PDF
        pdf_path = temp_data_src / "test.pdf"
        create_test_pdf(pdf_path, num_pages=2)

        records = [
            {'file_path': str(pdf_path), 'file_type': 'pdf',
             'section_dir': '', 'filename': 'test.pdf', 'needs_conversion': 'true'},
        ]

        # First run
        log_entries1, prepared_records1 = process_inventory(records, temp_data_src)
        output_path = temp_work / "inventory_prepared.csv"
        write_prepared_inventory(prepared_records1, output_path)

        with open(output_path, 'r') as f:
            content1 = f.read()

        # Second run
        log_entries2, prepared_records2 = process_inventory(records, temp_data_src)
        write_prepared_inventory(prepared_records2, output_path)

        with open(output_path, 'r') as f:
            content2 = f.read()

        # Outputs should be identical
        assert content1 == content2

        # Second run should have skipped conversion
        convert_logs = [e for e in log_entries2 if e['operation'] == 'pdf_convert']
        skipped = any(e.get('status') == 'skipped' for e in convert_logs)
        # Either skipped or still success (idempotent)
        assert len(prepared_records1) == len(prepared_records2)


class TestCLI:
    """Test command-line interface"""

    def test_help_output(self):
        """Should show help with --help"""
        import subprocess
        script_path = Path(__file__).parent.parent / "scripts" / "02_prepare_sources.py"
        if not script_path.exists():
            pytest.skip("Script not implemented yet")

        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert '--inventory' in result.stdout
        assert '--output' in result.stdout
        assert '--log' in result.stdout
        assert '--force' in result.stdout
        assert '--data-src' in result.stdout
