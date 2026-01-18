#!/usr/bin/env python3
"""
Tests for Stage 1: Inventory script

Following TDD approach - these tests define the expected behavior
of the inventory script before implementation.
"""

import pytest
import csv
from pathlib import Path

# Import from scripts package using clean aliases
from scripts import inventory

# Get functions from the module
scan_directory = getattr(inventory, 'scan_directory', None)
normalize_file_type = getattr(inventory, 'normalize_file_type', None)
write_inventory_csv = getattr(inventory, 'write_inventory_csv', None)


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
    return work


def read_csv_as_list(csv_path: Path):
    """Helper to read CSV file as list of dicts"""
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)


# Test 1: Empty directory
def test_empty_directory(temp_data_src, temp_work):
    """Test that empty directory produces CSV with header only"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    # Should have header but no data rows
    assert output_csv.exists()
    rows = read_csv_as_list(output_csv)
    assert len(rows) == 0

    # Verify header is correct
    with open(output_csv, 'r') as f:
        header = f.readline().strip()
        assert header == "file_path,file_type,section_dir,filename,needs_conversion"


# Test 2: Single JPG file
def test_single_jpg_file(temp_data_src, temp_work):
    """Test cataloging a single JPG file"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create test file
    test_file = temp_data_src / "test.jpg"
    test_file.touch()

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    rows = read_csv_as_list(output_csv)
    assert len(rows) == 1

    row = rows[0]
    assert row['file_type'] == 'jpg'
    assert row['filename'] == 'test.jpg'
    assert row['needs_conversion'] == 'false'
    assert row['section_dir'] == ''


# Test 3: Mixed file types
def test_mixed_file_types(temp_data_src, temp_work):
    """Test handling multiple file types correctly"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create test files
    (temp_data_src / "image.jpg").touch()
    (temp_data_src / "document.pdf").touch()
    (temp_data_src / "page.html").touch()

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    rows = read_csv_as_list(output_csv)
    assert len(rows) == 3

    # Check each file type
    types = {row['filename']: row for row in rows}

    assert types['image.jpg']['file_type'] == 'jpg'
    assert types['image.jpg']['needs_conversion'] == 'false'

    assert types['document.pdf']['file_type'] == 'pdf'
    assert types['document.pdf']['needs_conversion'] == 'true'

    assert types['page.html']['file_type'] == 'html'
    assert types['page.html']['needs_conversion'] == 'false'


# Test 4: Nested directories
def test_nested_directories(temp_data_src, temp_work):
    """Test that section_dir is correctly extracted from nested structure"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create nested structure
    section = temp_data_src / "21 - Clutch"
    section.mkdir()
    (section / "21-01.jpg").touch()
    (section / "21-02.jpg").touch()

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    rows = read_csv_as_list(output_csv)
    assert len(rows) == 2

    for row in rows:
        assert row['section_dir'] == '21 - Clutch'
        assert '21 - Clutch' in row['file_path']


# Test 5: Case insensitivity
def test_case_insensitivity(temp_data_src, temp_work):
    """Test that file extensions are normalized regardless of case"""
    if normalize_file_type is None:
        pytest.skip("Module not implemented yet")

    # Test the normalize function directly
    assert normalize_file_type('.jpg') == 'jpg'
    assert normalize_file_type('.JPG') == 'jpg'
    assert normalize_file_type('.Jpg') == 'jpg'
    assert normalize_file_type('.jpeg') == 'jpg'
    assert normalize_file_type('.JPEG') == 'jpg'

    assert normalize_file_type('.png') == 'png'
    assert normalize_file_type('.PNG') == 'png'

    assert normalize_file_type('.pdf') == 'pdf'
    assert normalize_file_type('.PDF') == 'pdf'

    assert normalize_file_type('.html') == 'html'
    assert normalize_file_type('.HTML') == 'html'
    assert normalize_file_type('.htm') == 'html'
    assert normalize_file_type('.HTM') == 'html'


# Test 6: Unsupported file types
def test_unsupported_file_types(temp_data_src, temp_work):
    """Test that unsupported file types are ignored"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create mix of supported and unsupported files
    (temp_data_src / "image.jpg").touch()
    (temp_data_src / "readme.txt").touch()
    (temp_data_src / "archive.zip").touch()
    (temp_data_src / "notes.md").touch()
    (temp_data_src / ".DS_Store").touch()
    (temp_data_src / "document.pdf").touch()

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    rows = read_csv_as_list(output_csv)
    # Only jpg and pdf should be included
    assert len(rows) == 2

    filenames = [row['filename'] for row in rows]
    assert 'image.jpg' in filenames
    assert 'document.pdf' in filenames
    assert 'readme.txt' not in filenames
    assert 'archive.zip' not in filenames
    assert 'notes.md' not in filenames
    assert '.DS_Store' not in filenames


# Test 7: Files without parent directory
def test_files_without_parent_directory(temp_data_src, temp_work):
    """Test that files directly in data_src have empty section_dir"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create file directly in data_src
    (temp_data_src / "M3-techspec.html").touch()

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    rows = read_csv_as_list(output_csv)
    assert len(rows) == 1

    row = rows[0]
    assert row['filename'] == 'M3-techspec.html'
    assert row['section_dir'] == ''
    assert row['file_type'] == 'html'


# Test 8: Alphabetical sorting
def test_alphabetical_sorting(temp_data_src, temp_work):
    """Test that output is sorted alphabetically by file_path"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create files in non-alphabetical order
    (temp_data_src / "zebra.jpg").touch()
    (temp_data_src / "apple.jpg").touch()
    (temp_data_src / "mango.jpg").touch()

    section = temp_data_src / "01 - Section"
    section.mkdir()
    (section / "beta.jpg").touch()
    (section / "alpha.jpg").touch()

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    rows = read_csv_as_list(output_csv)

    # Extract file paths and verify they're sorted
    file_paths = [row['file_path'] for row in rows]
    assert file_paths == sorted(file_paths)


# Test 9: Idempotency
def test_idempotency(temp_data_src, temp_work):
    """Test that running twice produces identical output"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create test files
    (temp_data_src / "test1.jpg").touch()
    (temp_data_src / "test2.pdf").touch()

    section = temp_data_src / "Section"
    section.mkdir()
    (section / "nested.jpg").touch()

    output_csv = temp_work / "inventory.csv"

    # Run first time
    records1 = scan_directory(temp_data_src)
    write_inventory_csv(records1, output_csv)

    with open(output_csv, 'r') as f:
        content1 = f.read()

    # Run second time
    records2 = scan_directory(temp_data_src)
    write_inventory_csv(records2, output_csv)

    with open(output_csv, 'r') as f:
        content2 = f.read()

    # Output should be byte-for-byte identical
    assert content1 == content2


# Test 10: PDF conversion flag
def test_pdf_conversion_flag(temp_data_src, temp_work):
    """Test that only PDFs have needs_conversion=true"""
    if scan_directory is None:
        pytest.skip("Module not implemented yet")

    # Create mix of PDFs and images
    (temp_data_src / "image1.jpg").touch()
    (temp_data_src / "image2.png").touch()
    (temp_data_src / "doc1.pdf").touch()
    (temp_data_src / "doc2.pdf").touch()
    (temp_data_src / "page.html").touch()

    output_csv = temp_work / "inventory.csv"

    records = scan_directory(temp_data_src)
    write_inventory_csv(records, output_csv)

    rows = read_csv_as_list(output_csv)

    for row in rows:
        if row['file_type'] == 'pdf':
            assert row['needs_conversion'] == 'true', f"PDF {row['filename']} should need conversion"
        else:
            assert row['needs_conversion'] == 'false', f"Non-PDF {row['filename']} should not need conversion"


# Additional test: normalize_file_type with unsupported types
def test_normalize_file_type_unsupported():
    """Test that normalize_file_type returns None for unsupported types"""
    if normalize_file_type is None:
        pytest.skip("Module not implemented yet")

    assert normalize_file_type('.txt') is None
    assert normalize_file_type('.zip') is None
    assert normalize_file_type('.md') is None
    assert normalize_file_type('.py') is None
    assert normalize_file_type('') is None
    assert normalize_file_type('.') is None
