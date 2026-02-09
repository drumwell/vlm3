#!/usr/bin/env python3
"""
Stage 2: Source Preparation
Converts PDFs to JPG images and validates all images are readable.

This script processes the inventory from Stage 1, converts any PDF files to
directories of JPG images, validates all images can be opened, and produces
an updated inventory that references the converted files.

Usage:
    python scripts/02_prepare_sources.py \
        --inventory work/inventory.csv \
        --output work/inventory_prepared.csv \
        --log work/logs/source_preparation.csv
"""

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from PIL import Image
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class PrepareConfig:
    """Configuration for source preparation."""

    inventory: Path
    data_src: Path
    output: Path
    log: Path
    force: bool = False
    verbose: bool = False
    dpi: int = 200
    jpeg_quality: int = 95

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "PrepareConfig":
        """Create config from parsed CLI arguments."""
        return cls(
            inventory=args.inventory,
            data_src=args.data_src,
            output=args.output,
            log=args.log,
            force=args.force,
            verbose=args.verbose,
        )

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
) -> Tuple[bool, List[Path], Optional[str], bool]:
    """
    Convert PDF to directory of JPG images.

    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to write JPG files (will be created)
        dpi: DPI for conversion (default 200)
        quality: JPEG quality 1-100 (default 95)
        force: If True, reconvert even if output exists

    Returns:
        Tuple of (success: bool, output_paths: List[Path], error_message: Optional[str], was_skipped: bool)

    Behavior:
        - Creates output_dir if it doesn't exist
        - Names files: 001.jpg, 002.jpg, 003.jpg, etc.
        - If output_dir exists and contains JPGs, skips unless force=True
        - Returns list of created file paths
    """
    try:
        # Check if already converted (idempotency)
        if output_dir.exists() and not force:
            existing_jpgs = sorted(output_dir.glob("*.jpg"))
            if existing_jpgs:
                logger.debug(f"Skipping {pdf_path.name}: already converted ({len(existing_jpgs)} images)")
                return True, existing_jpgs, None, True  # was_skipped=True

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert PDF to images
        logger.debug(f"Converting {pdf_path.name} at {dpi} DPI")
        images = convert_from_path(str(pdf_path), dpi=dpi)

        if not images:
            return False, [], "PDF has no pages", False

        output_paths = []
        for i, image in enumerate(images, start=1):
            # Name format: 001.jpg, 002.jpg, etc.
            output_path = output_dir / f"{i:03d}.jpg"
            image.save(str(output_path), "JPEG", quality=quality)
            output_paths.append(output_path)
            logger.debug(f"  Created {output_path.name}")

        logger.info(f"Converted {pdf_path.name} -> {len(output_paths)} images")
        return True, output_paths, None, False  # was_skipped=False

    except (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError) as e:
        error_msg = f"PDF conversion error: {str(e)}"
        logger.error(f"Failed to convert {pdf_path.name}: {error_msg}")
        return False, [], error_msg, False
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Failed to convert {pdf_path.name}: {error_msg}")
        return False, [], error_msg, False


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
    try:
        # Check file exists
        if not image_path.exists():
            return False, f"File not found: {image_path}"

        # Check file is not empty
        if image_path.stat().st_size == 0:
            return False, f"File is empty: {image_path}"

        # Try to open and verify with PIL
        with Image.open(image_path) as img:
            img.verify()

        # Re-open to ensure it's fully readable (verify() can leave file in bad state)
        with Image.open(image_path) as img:
            # Access a pixel to ensure data is readable
            img.load()

        return True, None

    except FileNotFoundError:
        return False, f"File not found: {image_path}"
    except Exception as e:
        return False, f"Invalid image: {str(e)}"


def read_inventory(inventory_path: Path) -> List[Dict[str, str]]:
    """
    Read inventory CSV from Stage 1.

    Args:
        inventory_path: Path to inventory.csv

    Returns:
        List of inventory records (dicts)

    Expected columns: file_path, file_type, section_dir, filename, needs_conversion
    """
    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    records = []
    with open(inventory_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    return records


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
    """
    log_entries = []
    prepared_records = []

    for record in inventory_records:
        file_path = Path(record['file_path'])
        file_type = record['file_type']
        section_dir = record['section_dir']
        needs_conversion = record.get('needs_conversion', 'false') == 'true'

        timestamp = datetime.now().isoformat()

        if needs_conversion and file_type == 'pdf':
            # Convert PDF to JPGs
            pdf_stem = file_path.stem
            output_dir = file_path.parent / pdf_stem

            success, output_paths, error, was_skipped = convert_pdf_to_jpgs(
                file_path, output_dir, force=force
            )

            if success and output_paths:
                log_entries.append({
                    'timestamp': timestamp,
                    'source_file': str(file_path),
                    'operation': 'pdf_convert',
                    'status': 'skipped' if was_skipped else 'success',
                    'output_path': str(output_dir),
                    'error_message': ''
                })

                # Create records for each converted JPG
                for jpg_path in output_paths:
                    prepared_records.append({
                        'file_path': str(jpg_path),
                        'file_type': 'jpg',
                        'section_dir': section_dir,
                        'filename': jpg_path.name,
                        'original_source': str(file_path)
                    })
            else:
                # Conversion failed
                log_entries.append({
                    'timestamp': timestamp,
                    'source_file': str(file_path),
                    'operation': 'pdf_convert',
                    'status': 'failure',
                    'output_path': '',
                    'error_message': error or 'Unknown error'
                })

        elif file_type in ('jpg', 'png'):
            # Validate existing image
            is_valid, error = validate_image(file_path)

            log_entries.append({
                'timestamp': timestamp,
                'source_file': str(file_path),
                'operation': 'image_validate',
                'status': 'success' if is_valid else 'failure',
                'output_path': str(file_path) if is_valid else '',
                'error_message': error or ''
            })

            if is_valid:
                prepared_records.append({
                    'file_path': str(file_path),
                    'file_type': file_type,
                    'section_dir': section_dir,
                    'filename': record['filename'],
                    'original_source': ''
                })

        elif file_type == 'html':
            # Pass through HTML files unchanged
            log_entries.append({
                'timestamp': timestamp,
                'source_file': str(file_path),
                'operation': 'skip',
                'status': 'success',
                'output_path': str(file_path),
                'error_message': ''
            })

            prepared_records.append({
                'file_path': str(file_path),
                'file_type': file_type,
                'section_dir': section_dir,
                'filename': record['filename'],
                'original_source': ''
            })

        else:
            # Unknown file type - skip
            logger.warning(f"Unknown file type: {file_type} for {file_path}")

    return log_entries, prepared_records


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
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort records by file_path for reproducibility
    sorted_records = sorted(records, key=lambda r: r['file_path'])

    # Define CSV columns
    fieldnames = ['file_path', 'file_type', 'section_dir', 'filename', 'original_source']

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_records)

    logger.info(f"Wrote {len(sorted_records)} records to {output_path}")


def write_conversion_log(log_entries: List[Dict], log_path: Path):
    """
    Write conversion log CSV.

    Args:
        log_entries: List of log entries
        log_path: Path to log CSV

    Schema: timestamp, source_file, operation, status, output_path, error_message

    Behavior:
        - Creates log directory if needed
        - Writes CSV with header (overwrites existing)
    """
    # Ensure log directory exists
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    fieldnames = ['timestamp', 'source_file', 'operation', 'status', 'output_path', 'error_message']

    # Write CSV
    with open(log_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_entries)

    logger.info(f"Wrote {len(log_entries)} log entries to {log_path}")


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
    # Count operations by type and status
    pdf_converts = [e for e in log_entries if e['operation'] == 'pdf_convert']
    pdf_success = sum(1 for e in pdf_converts if e['status'] in ('success', 'skipped'))
    pdf_failure = sum(1 for e in pdf_converts if e['status'] == 'failure')
    pdf_skipped = sum(1 for e in pdf_converts if e['status'] == 'skipped')

    image_validates = [e for e in log_entries if e['operation'] == 'image_validate']
    image_success = sum(1 for e in image_validates if e['status'] == 'success')
    image_failure = sum(1 for e in image_validates if e['status'] == 'failure')

    html_skips = [e for e in log_entries if e['operation'] == 'skip']

    # Count prepared records by type
    type_counts = {}
    converted_count = 0
    for record in prepared_records:
        file_type = record['file_type']
        type_counts[file_type] = type_counts.get(file_type, 0) + 1
        if record['original_source']:
            converted_count += 1

    # Print summary
    logger.info("=== Source Preparation Summary ===")
    logger.info(f"Total operations: {len(log_entries)}")
    logger.info(f"PDF conversions: {len(pdf_converts)} ({pdf_success} success, {pdf_skipped} skipped, {pdf_failure} failed)")
    logger.info(f"Image validations: {len(image_validates)} ({image_success} valid, {image_failure} invalid)")
    logger.info(f"HTML pass-through: {len(html_skips)}")
    logger.info(f"")
    logger.info(f"Prepared inventory: {len(prepared_records)} files")
    for file_type in sorted(type_counts.keys()):
        logger.info(f"  {file_type}: {type_counts[file_type]}")
    logger.info(f"  (from PDF conversion: {converted_count})")

    if pdf_failure > 0 or image_failure > 0:
        logger.warning(f"Errors encountered: {pdf_failure + image_failure}")
        for entry in log_entries:
            if entry['status'] == 'failure':
                logger.warning(f"  {entry['source_file']}: {entry['error_message']}")

    logger.info("==================================")


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
      --data-src data_src \\
      --output work/inventory_prepared.csv \\
      --log work/logs/source_preparation.csv

  # Force reconversion of all PDFs
  python scripts/02_prepare_sources.py \\
      --inventory work/inventory.csv \\
      --data-src data_src \\
      --output work/inventory_prepared.csv \\
      --log work/logs/source_preparation.csv \\
      --force

  # With verbose logging
  python scripts/02_prepare_sources.py \\
      --inventory work/inventory.csv \\
      --data-src data_src \\
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
        '--data-src',
        type=Path,
        required=True,
        help='Path to data source directory (required for PDF output location)'
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

        # Validate data_src directory exists
        if not args.data_src.exists():
            logger.error(f"Data source directory not found: {args.data_src}")
            return 1
        if not args.data_src.is_dir():
            logger.error(f"Data source is not a directory: {args.data_src}")
            return 1

        if not inventory_records:
            logger.error("Empty inventory")
            return 1

        logger.info(f"Data source directory: {args.data_src}")

        # Process inventory
        log_entries, prepared_records = process_inventory(
            inventory_records,
            args.data_src,
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
