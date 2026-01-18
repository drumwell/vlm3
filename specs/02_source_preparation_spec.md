# Stage 2: Source Preparation - Implementation Spec (TDD)

## Overview

**Script**: `scripts/02_prepare_sources.py`
**Instruction**: **OVERWRITE** the existing `scripts/02_preprocess.py` with this new implementation.

**Purpose**: Convert PDFs to JPG images and validate all images are readable, producing an updated inventory that references converted files.

**Architecture Reference**: See `pipeline_rearchitecture.md` lines 338-375 for full architectural context.

---

## Requirements Summary

### Inputs
- `work/inventory.csv` (from Stage 1)

### Outputs
- `data_src/{pdf_stem}/` directories containing `001.jpg`, `002.jpg`, etc. (converted PDF pages)
- `work/inventory_prepared.csv` (updated inventory with converted files)
- `work/logs/source_preparation.csv` (conversion log with timestamps, statuses)

### Key Behaviors
1. **PDF Conversion**: Convert each PDF to a directory of JPG files at 200 DPI
2. **Image Validation**: Verify all images (original + converted) can be opened by PIL
3. **Idempotency**: Skip already-converted PDFs; safe to rerun
4. **Logging**: Detailed conversion log with success/failure tracking
5. **Error Handling**: Graceful failures; continue processing other files

---

## Test-Driven Development Approach

### Phase 1: Write Tests First

**Test File**: `tests/test_02_prepare_sources.py`

Create tests for all functions BEFORE implementing them. Run tests and watch them fail. Then implement to make them pass.

#### Test Suite Structure

```python
import pytest
from pathlib import Path
from PIL import Image
import tempfile
import csv

# Import functions under test (will fail initially)
from scripts.prepare_sources_02 import (
    convert_pdf_to_jpgs,
    validate_image,
    process_inventory,
    generate_prepared_inventory,
    write_conversion_log
)


class TestPDFConversion:
    """Test PDF to JPG conversion"""

    def test_convert_pdf_single_page(self, tmp_path):
        """Should convert 1-page PDF to 001.jpg"""
        # Create a simple 1-page PDF fixture
        # Convert it
        # Assert: output directory created
        # Assert: 001.jpg exists
        # Assert: image is readable
        # Assert: image dimensions reasonable (200 DPI)
        pass

    def test_convert_pdf_multi_page(self, tmp_path):
        """Should convert 5-page PDF to 001.jpg through 005.jpg"""
        # Create 5-page PDF fixture
        # Convert it
        # Assert: 5 JPGs created with correct names
        # Assert: all are readable
        pass

    def test_convert_pdf_output_directory_created(self, tmp_path):
        """Should create output directory if it doesn't exist"""
        # PDF stem should become directory name
        # Example: Getrag265_Rebuild.pdf -> data_src/Getrag265_Rebuild/
        pass

    def test_convert_pdf_already_exists_skip(self, tmp_path):
        """Should skip conversion if output directory already exists and has images"""
        # Create PDF
        # Convert once
        # Convert again
        # Assert: second conversion was skipped (log message or return value)
        pass

    def test_convert_pdf_invalid_pdf_raises_error(self, tmp_path):
        """Should raise appropriate error for corrupted PDF"""
        # Create corrupted PDF file
        # Attempt conversion
        # Assert: raises expected exception
        pass


class TestImageValidation:
    """Test image validation logic"""

    def test_validate_image_valid_jpg(self, tmp_path):
        """Should return True for valid JPG"""
        # Create valid JPG
        # Validate it
        # Assert: returns True
        pass

    def test_validate_image_valid_png(self, tmp_path):
        """Should return True for valid PNG"""
        pass

    def test_validate_image_corrupted_file(self, tmp_path):
        """Should return False for corrupted image"""
        # Create file with .jpg extension but invalid data
        # Validate it
        # Assert: returns False
        pass

    def test_validate_image_missing_file(self, tmp_path):
        """Should return False for non-existent file"""
        pass

    def test_validate_image_empty_file(self, tmp_path):
        """Should return False for 0-byte image file"""
        pass


class TestInventoryProcessing:
    """Test inventory processing logic"""

    def test_process_inventory_converts_pdfs(self, tmp_path):
        """Should convert all PDFs marked needs_conversion=true"""
        # Create inventory.csv with 2 PDFs, 3 JPGs
        # Process inventory
        # Assert: PDFs converted
        # Assert: JPGs validated
        # Assert: returns success records
        pass

    def test_process_inventory_validates_existing_images(self, tmp_path):
        """Should validate existing JPG/PNG files"""
        # Create inventory with existing images
        # Process inventory
        # Assert: all images validated
        pass

    def test_process_inventory_handles_failures_gracefully(self, tmp_path):
        """Should continue processing after individual failures"""
        # Create inventory with 1 bad PDF, 2 good JPGs
        # Process inventory
        # Assert: good JPGs still processed
        # Assert: failure logged for bad PDF
        pass

    def test_process_inventory_html_files_passed_through(self, tmp_path):
        """Should skip HTML files (no validation needed)"""
        # Create inventory with HTML files
        # Process inventory
        # Assert: HTML files appear in output inventory unchanged
        pass


class TestPreparedInventoryGeneration:
    """Test generation of inventory_prepared.csv"""

    def test_prepared_inventory_schema(self, tmp_path):
        """Should output correct CSV schema"""
        # Schema: file_path, file_type, section_dir, filename, original_source
        # Process some files
        # Generate prepared inventory
        # Assert: correct columns present
        pass

    def test_prepared_inventory_original_jpgs_unchanged(self, tmp_path):
        """Should include original JPGs with empty original_source"""
        # Process inventory with existing JPGs
        # Generate prepared inventory
        # Assert: original JPGs present
        # Assert: original_source column is empty for them
        pass

    def test_prepared_inventory_converted_pdfs_tracked(self, tmp_path):
        """Should replace PDFs with converted JPGs and track original_source"""
        # Input: data_src/Getrag265.pdf
        # Converts to: data_src/Getrag265/001.jpg, 002.jpg, 003.jpg
        # Output inventory should have:
        #   - data_src/Getrag265/001.jpg with original_source=data_src/Getrag265.pdf
        #   - data_src/Getrag265/002.jpg with original_source=data_src/Getrag265.pdf
        #   - etc.
        pass

    def test_prepared_inventory_html_passed_through(self, tmp_path):
        """Should include HTML files unchanged"""
        pass

    def test_prepared_inventory_sorted_by_path(self, tmp_path):
        """Should sort output by file_path for reproducibility"""
        pass


class TestConversionLog:
    """Test conversion logging"""

    def test_conversion_log_schema(self, tmp_path):
        """Should write log with correct columns"""
        # Columns: timestamp, source_file, operation, status, output_path, error_message
        pass

    def test_conversion_log_success_entries(self, tmp_path):
        """Should log successful conversions"""
        # Convert PDF successfully
        # Check log
        # Assert: success entry with correct paths
        pass

    def test_conversion_log_failure_entries(self, tmp_path):
        """Should log failed conversions with error messages"""
        # Attempt to convert corrupted PDF
        # Check log
        # Assert: failure entry with error message
        pass

    def test_conversion_log_skipped_entries(self, tmp_path):
        """Should log skipped conversions"""
        # Convert PDF that's already converted
        # Check log
        # Assert: skipped entry
        pass

    def test_conversion_log_validation_entries(self, tmp_path):
        """Should log image validation results"""
        # Validate existing images
        # Check log
        # Assert: validation entries present
        pass


class TestEndToEnd:
    """Integration tests for full workflow"""

    def test_full_pipeline_mixed_sources(self, tmp_path):
        """Should process inventory with mixed file types"""
        # Create inventory with:
        #   - 2 PDFs (one 3-page, one 5-page)
        #   - 10 existing JPGs
        #   - 2 HTML files
        # Run full pipeline
        # Assert: PDFs converted (8 total JPGs)
        # Assert: existing JPGs validated (10)
        # Assert: HTML passed through (2)
        # Assert: prepared inventory has 20 entries (8+10+2)
        # Assert: conversion log has all operations
        pass

    def test_idempotency_rerun_safe(self, tmp_path):
        """Should be safe to rerun on same inventory"""
        # Run pipeline once
        # Capture output
        # Run pipeline again
        # Assert: outputs identical
        # Assert: conversions skipped second time
        pass

    def test_real_world_inventory(self, tmp_path):
        """Should handle actual inventory.csv from Stage 1"""
        # Use fixtures/sample_inventory.csv (realistic data)
        # Run pipeline
        # Assert: all expected outputs created
        # Assert: no crashes
        pass
```

---

### Phase 2: Implement to Pass Tests

After writing tests, implement `scripts/02_prepare_sources.py` to pass them.

---

## Implementation Specification

### Script Structure

```python
#!/usr/bin/env python3
"""
Stage 2: Source Preparation
Converts PDFs to JPG images and validates all images are readable.

This script processes the inventory from Stage 1, converts any PDF files to
directories of JPG images, validates all images can be opened, and produces
an updated inventory that references the converted files.

Usage:
    python scripts/02_prepare_sources.py \\
        --inventory work/inventory.csv \\
        --output work/inventory_prepared.csv \\
        --log work/logs/source_preparation.csv
"""

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_path

# Configure logging (follow same pattern as Stage 1)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def convert_pdf_to_jpgs(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    quality: int = 95,
    force: bool = False
) -> Tuple[bool, List[Path], Optional[str]]:
    """
    Convert PDF to directory of JPG images.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to write JPG files (will be created)
        dpi: DPI for conversion (default 200)
        quality: JPEG quality 1-100 (default 95)
        force: If True, reconvert even if output exists

    Returns:
        Tuple of (success: bool, output_paths: List[Path], error_message: Optional[str])

    Behavior:
        - Creates output_dir if it doesn't exist
        - Names files: 001.jpg, 002.jpg, 003.jpg, etc.
        - If output_dir exists and contains JPGs, skips unless force=True
        - Returns list of created file paths
    """
    pass


def validate_image(image_path: Path) -> Tuple[bool, Optional[str]]:
    """
    Validate that an image file can be opened by PIL.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (is_valid: bool, error_message: Optional[str])

    Checks:
        - File exists
        - File is not empty (size > 0)
        - PIL can open and verify the image
    """
    pass


def read_inventory(inventory_path: Path) -> List[Dict[str, str]]:
    """
    Read inventory CSV from Stage 1.

    Args:
        inventory_path: Path to inventory.csv

    Returns:
        List of inventory records (dicts)

    Expected columns: file_path, file_type, section_dir, filename, needs_conversion
    """
    pass


def process_inventory(
    inventory_records: List[Dict[str, str]],
    data_src: Path,
    force: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """
    Process inventory: convert PDFs and validate images.

    Args:
        inventory_records: Records from inventory.csv
        data_src: Root data_src directory
        force: Force reconversion of already-converted PDFs

    Returns:
        Tuple of (conversion_log_entries, prepared_records)

    Logic:
        1. For each record where needs_conversion='true':
           - Convert PDF to JPG directory
           - Log conversion result
           - Create new records for each JPG

        2. For each image record (jpg, png):
           - Validate image is readable
           - Log validation result
           - Keep record if valid

        3. For HTML records:
           - Pass through unchanged

    conversion_log_entries schema:
        - timestamp: ISO 8601 timestamp
        - source_file: original file path
        - operation: 'pdf_convert' | 'image_validate' | 'skip'
        - status: 'success' | 'failure' | 'skipped'
        - output_path: path to output file(s)
        - error_message: error details if status='failure'

    prepared_records schema:
        - file_path: path to file
        - file_type: jpg | png | html
        - section_dir: section directory name
        - filename: base filename
        - original_source: path to original PDF (if converted) or empty
    """
    pass


def write_prepared_inventory(records: List[Dict], output_path: Path):
    """
    Write inventory_prepared.csv.

    Args:
        records: List of prepared records
        output_path: Path to output CSV

    Schema: file_path, file_type, section_dir, filename, original_source

    Behavior:
        - Creates output directory if needed
        - Sorts records by file_path for reproducibility
        - Writes CSV with header
    """
    pass


def write_conversion_log(log_entries: List[Dict], log_path: Path):
    """
    Write conversion log CSV.

    Args:
        log_entries: List of log entries
        log_path: Path to log CSV

    Schema: timestamp, source_file, operation, status, output_path, error_message

    Behavior:
        - Creates log directory if needed
        - Appends to existing log if present
        - Includes header if file is new
    """
    pass


def print_summary(log_entries: List[Dict], prepared_records: List[Dict]):
    """
    Print summary statistics to stdout.

    Args:
        log_entries: Conversion log entries
        prepared_records: Prepared inventory records

    Summary should include:
        - Total files processed
        - PDFs converted (count, total pages)
        - Images validated (pass/fail counts)
        - HTML files passed through
        - Any errors encountered
    """
    pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stage 2: Source Preparation - Convert PDFs and validate images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/02_prepare_sources.py \\
      --inventory work/inventory.csv \\
      --output work/inventory_prepared.csv \\
      --log work/logs/source_preparation.csv

  # Force reconversion of all PDFs
  python scripts/02_prepare_sources.py \\
      --inventory work/inventory.csv \\
      --output work/inventory_prepared.csv \\
      --log work/logs/source_preparation.csv \\
      --force

  # With verbose logging
  python scripts/02_prepare_sources.py \\
      --inventory work/inventory.csv \\
      --output work/inventory_prepared.csv \\
      --log work/logs/source_preparation.csv \\
      --verbose

Input CSV schema (from Stage 1):
  file_path, file_type, section_dir, filename, needs_conversion

Output CSV schema:
  file_path, file_type, section_dir, filename, original_source

Conversion log schema:
  timestamp, source_file, operation, status, output_path, error_message

PDF Conversion:
  - Converts at 200 DPI for quality
  - JPEG quality: 95 (high quality)
  - Output naming: 001.jpg, 002.jpg, etc.
  - Idempotent: skips already-converted PDFs unless --force used
        """
    )

    parser.add_argument(
        '--inventory',
        type=Path,
        required=True,
        help='Path to inventory.csv from Stage 1'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output inventory_prepared.csv'
    )

    parser.add_argument(
        '--log',
        type=Path,
        required=True,
        help='Path to conversion log CSV'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force reconversion of already-converted PDFs'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info("Starting source preparation")

        # Read inventory
        inventory_records = read_inventory(args.inventory)
        logger.info(f"Read {len(inventory_records)} records from inventory")

        # Determine data_src from inventory paths
        # Assume all paths start with same root directory
        if inventory_records:
            first_path = Path(inventory_records[0]['file_path'])
            data_src = first_path.parts[0]  # e.g., 'data_src'
            data_src_path = Path(data_src)
        else:
            logger.error("Empty inventory")
            return 1

        # Process inventory
        log_entries, prepared_records = process_inventory(
            inventory_records,
            data_src_path,
            force=args.force
        )

        # Write outputs
        write_prepared_inventory(prepared_records, args.output)
        write_conversion_log(log_entries, args.log)

        # Print summary
        print_summary(log_entries, prepared_records)

        logger.info("Source preparation completed successfully")
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

- [ ] **PDF Conversion**
  - [ ] All PDFs from inventory converted to JPG directories
  - [ ] JPG files named `001.jpg`, `002.jpg`, etc.
  - [ ] Conversion at 200 DPI, JPEG quality 95
  - [ ] Output directories created in `data_src/`

- [ ] **Image Validation**
  - [ ] All original JPG/PNG files validated
  - [ ] All converted JPG files validated
  - [ ] Invalid/corrupted images logged as failures

- [ ] **Idempotency**
  - [ ] Safe to rerun; skips already-converted PDFs
  - [ ] `--force` flag forces reconversion
  - [ ] Second run produces identical output

- [ ] **Output Files**
  - [ ] `work/inventory_prepared.csv` contains all image files
  - [ ] PDFs removed from prepared inventory
  - [ ] Converted JPGs include `original_source` reference
  - [ ] HTML files passed through unchanged

- [ ] **Logging**
  - [ ] `work/logs/source_preparation.csv` contains all operations
  - [ ] Success, failure, and skipped operations logged
  - [ ] Timestamps in ISO 8601 format
  - [ ] Error messages included for failures

### Code Quality Requirements

- [ ] **Testing**
  - [ ] All unit tests pass
  - [ ] Integration tests pass
  - [ ] Test coverage >90%

- [ ] **Style**
  - [ ] Follows same patterns as `01_inventory.py`
  - [ ] Type hints on all functions
  - [ ] Docstrings on all public functions
  - [ ] Clear logging at INFO level

- [ ] **Error Handling**
  - [ ] Graceful handling of corrupted PDFs
  - [ ] Continues processing after individual failures
  - [ ] Clear error messages for users

- [ ] **Documentation**
  - [ ] `--help` output is comprehensive
  - [ ] Examples in epilog
  - [ ] README updated if needed

### Performance Requirements

- [ ] **Efficiency**
  - [ ] Skips already-converted files (idempotency check is fast)
  - [ ] Processes files in deterministic order (sorted)
  - [ ] Memory-efficient (doesn't load all images simultaneously)

### Validation Checks

Before considering Stage 2 complete, run:

```bash
# Test on sample data
python scripts/02_prepare_sources.py \
    --inventory work/inventory.csv \
    --output work/inventory_prepared.csv \
    --log work/logs/source_preparation.csv \
    --verbose

# Verify outputs exist
test -f work/inventory_prepared.csv || echo "FAIL: inventory_prepared.csv missing"
test -f work/logs/source_preparation.csv || echo "FAIL: log missing"

# Check prepared inventory has more entries than original (PDFs expanded)
# Count PDFs in original inventory
PDF_COUNT=$(grep -c ",pdf," work/inventory.csv || echo 0)
# Count entries in prepared inventory
PREPARED_COUNT=$(tail -n +2 work/inventory_prepared.csv | wc -l)
echo "PDFs in original: $PDF_COUNT"
echo "Total prepared entries: $PREPARED_COUNT"

# Verify no PDFs in prepared inventory
if grep -q ",pdf," work/inventory_prepared.csv; then
    echo "FAIL: PDFs still present in prepared inventory"
else
    echo "PASS: No PDFs in prepared inventory"
fi

# Verify converted directories exist
# Example: if inventory had data_src/Getrag265.pdf, check for data_src/Getrag265/ directory

# Test idempotency
cp work/inventory_prepared.csv work/inventory_prepared_v1.csv
python scripts/02_prepare_sources.py \
    --inventory work/inventory.csv \
    --output work/inventory_prepared.csv \
    --log work/logs/source_preparation.csv
diff work/inventory_prepared_v1.csv work/inventory_prepared.csv || echo "FAIL: Not idempotent"
```

---

## Dependencies

Add to `requirements.txt`:

```txt
# Existing dependencies
...

# Stage 2: Source Preparation
pdf2image>=1.16.0        # PDF to image conversion
Pillow>=10.0.0          # Image validation and processing
```

System dependencies (install separately):

```bash
# macOS
brew install poppler

# Ubuntu/Debian
sudo apt-get install poppler-utils

# Note: pdf2image requires poppler-utils to be installed
```

---

## Implementation Checklist

### Step 1: Setup (before coding)
- [ ] Read this spec completely
- [ ] Read architecture doc (`pipeline_rearchitecture.md` lines 338-375)
- [ ] Install dependencies: `pip install pdf2image Pillow`
- [ ] Install poppler: `brew install poppler` (macOS)

### Step 2: Write Tests (TDD Phase 1)
- [ ] Create `tests/test_02_prepare_sources.py`
- [ ] Write all test cases from test suite above
- [ ] Create test fixtures (sample PDFs, images, CSVs)
- [ ] Run tests: `pytest tests/test_02_prepare_sources.py -v`
- [ ] **Expected: ALL TESTS FAIL** (functions don't exist yet)

### Step 3: Implement (TDD Phase 2)
- [ ] Create `scripts/02_prepare_sources.py`
- [ ] Implement `convert_pdf_to_jpgs()` → run tests → fix until pass
- [ ] Implement `validate_image()` → run tests → fix until pass
- [ ] Implement `read_inventory()` → run tests → fix until pass
- [ ] Implement `process_inventory()` → run tests → fix until pass
- [ ] Implement `write_prepared_inventory()` → run tests → fix until pass
- [ ] Implement `write_conversion_log()` → run tests → fix until pass
- [ ] Implement `print_summary()` → verify output
- [ ] Implement `main()` → run integration tests
- [ ] Run all tests: `pytest tests/test_02_prepare_sources.py -v`
- [ ] **Expected: ALL TESTS PASS**

### Step 4: Integration Testing
- [ ] Run on actual inventory: `python scripts/02_prepare_sources.py --inventory work/inventory.csv --output work/inventory_prepared.csv --log work/logs/source_preparation.csv --verbose`
- [ ] Verify outputs created
- [ ] Inspect conversion log for errors
- [ ] Check converted PDF directories exist
- [ ] Validate prepared inventory schema
- [ ] Test idempotency (run twice, compare outputs)

### Step 5: Validation
- [ ] Run all acceptance criteria checks (see above)
- [ ] Code review against Stage 1 style
- [ ] Update Makefile if needed
- [ ] Commit with message: `feat(stage2): implement source preparation with PDF conversion`

---

## Notes for Implementer

1. **Follow Stage 1 patterns**: Use same logging style, error handling, CLI structure
2. **Test-driven approach**: Write tests FIRST, then implement
3. **Idempotency is critical**: Must be safe to rerun
4. **Handle edge cases**: Empty PDFs, corrupted files, missing directories
5. **Clear logging**: User should see progress (INFO) and details (DEBUG)
6. **Performance**: Don't load all images into memory; process one at a time
7. **PDF stem logic**: `Getrag265_Rebuild.pdf` → `data_src/Getrag265_Rebuild/`
8. **Original source tracking**: Enables tracing converted files back to PDFs

---

## Example Test Fixtures

Create in `tests/fixtures/`:

### `sample_inventory.csv`
```csv
file_path,file_type,section_dir,filename,needs_conversion
data_src/test/sample.pdf,pdf,test,sample.pdf,true
data_src/test/image01.jpg,jpg,test,image01.jpg,false
data_src/test/image02.png,png,test,image02.png,false
data_src/specs.html,html,,specs.html,false
```

### Sample PDF
Use `PyPDF2` or `reportlab` to generate a simple multi-page PDF for testing.

### Sample Images
Create small test JPG/PNG files (10x10 pixels).

### Corrupted Files
Create files with `.jpg` extension but invalid content for negative testing.

---

## Success Criteria

Stage 2 is complete when:

1. ✅ All tests pass (`pytest tests/test_02_prepare_sources.py -v`)
2. ✅ Script runs successfully on actual inventory
3. ✅ All PDFs converted to JPG directories
4. ✅ All images validated
5. ✅ `work/inventory_prepared.csv` created with correct schema
6. ✅ `work/logs/source_preparation.csv` contains detailed log
7. ✅ Idempotency verified (rerun produces identical output)
8. ✅ No PDFs in prepared inventory
9. ✅ All acceptance criteria checked and passing
10. ✅ Code reviewed and committed

---

## Questions to Answer Before Starting

1. Should corrupted images be excluded from prepared inventory or fail the entire pipeline?
   - **Answer**: Exclude and log; continue processing

2. Should PDF conversion be parallelized for performance?
   - **Answer**: No, keep sequential for simplicity; optimize later if needed

3. What should happen if a PDF has 0 pages?
   - **Answer**: Log as error, exclude from prepared inventory

4. Should we validate image dimensions or just that they can be opened?
   - **Answer**: Just verify they can be opened; don't enforce dimensions

5. For `original_source` tracking, use absolute or relative paths?
   - **Answer**: Use same path format as input (relative from project root)

---

## Architecture Alignment Checklist

- [ ] Follows Stage 2 spec from `pipeline_rearchitecture.md`
- [ ] Input: `work/inventory.csv` (Stage 1 output)
- [ ] Output: `work/inventory_prepared.csv` (Stage 3 input)
- [ ] PDFs converted at 200 DPI, JPEG quality 95
- [ ] Converted files placed in `data_src/{pdf_stem}/` directories
- [ ] Schema matches architecture: `file_path, file_type, section_dir, filename, original_source`
- [ ] Idempotent operation
- [ ] Comprehensive logging
- [ ] Ready for Stage 3 classification

---

**IMPORTANT**: This script **replaces** the old `scripts/02_preprocess.py`. The new script is `scripts/02_prepare_sources.py` and follows the VLM architecture, not the old OCR-based approach.

**Overwrite instruction**: Delete or ignore `scripts/02_preprocess.py` and create `scripts/02_prepare_sources.py` from scratch following this spec.
