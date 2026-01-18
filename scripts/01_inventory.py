#!/usr/bin/env python3
"""
Stage 1: Inventory
Catalogs all source files in data_src/ and produces work/inventory.csv

This script recursively scans the data_src directory for supported file types
(jpg, png, pdf, html) and creates a CSV inventory with metadata about each file.

Usage:
    python scripts/01_inventory.py --data-src data_src --output work/inventory.csv
"""

import argparse
import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class InventoryConfig:
    """Configuration for inventory scanning."""

    data_src: Path
    output: Path
    section_filter: str = ""
    verbose: bool = False

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "InventoryConfig":
        """Create config from parsed CLI arguments."""
        return cls(
            data_src=args.data_src,
            output=args.output,
            section_filter=getattr(args, 'section_filter', "") or "",
            verbose=args.verbose,
        )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def normalize_file_type(extension: str) -> Optional[str]:
    """
    Normalize file extension to standard type.

    Args:
        extension: File extension including dot (e.g., '.jpg', '.PDF')

    Returns:
        Normalized type ('jpg', 'pdf', 'html', 'png') or None if unsupported

    Examples:
        >>> normalize_file_type('.jpg')
        'jpg'
        >>> normalize_file_type('.PDF')
        'pdf'
        >>> normalize_file_type('.txt')
        None
    """
    # Normalize to lowercase for case-insensitive matching
    ext_lower = extension.lower()

    # Map to standard types
    type_map = {
        '.jpg': 'jpg',
        '.jpeg': 'jpg',
        '.png': 'png',
        '.pdf': 'pdf',
        '.html': 'html',
        '.htm': 'html',
    }

    return type_map.get(ext_lower)


def scan_directory(data_src: Path) -> List[Dict[str, str]]:
    """
    Recursively scan data_src for supported files.

    Args:
        data_src: Path to the data source directory

    Returns:
        List of file records with keys: file_path, file_type, section_dir,
        filename, needs_conversion

    Raises:
        FileNotFoundError: If data_src doesn't exist
        PermissionError: If data_src is not readable
    """
    if not data_src.exists():
        raise FileNotFoundError(f"Data source directory not found: {data_src}")

    if not data_src.is_dir():
        raise NotADirectoryError(f"Data source is not a directory: {data_src}")

    records = []

    # Walk through all files recursively
    for file_path in sorted(data_src.rglob('*')):
        # Skip directories
        if not file_path.is_file():
            continue

        # Check if file type is supported
        file_type = normalize_file_type(file_path.suffix)
        if file_type is None:
            logger.debug(f"Skipping unsupported file: {file_path}")
            continue

        # Extract section directory (immediate parent)
        if file_path.parent == data_src:
            # File is directly in data_src
            section_dir = ''
        else:
            # Get the immediate parent directory name
            section_dir = file_path.parent.name

        # Determine if file needs conversion
        needs_conversion = 'true' if file_type == 'pdf' else 'false'

        # Create record
        record = {
            'file_path': str(file_path),
            'file_type': file_type,
            'section_dir': section_dir,
            'filename': file_path.name,
            'needs_conversion': needs_conversion
        }

        records.append(record)
        logger.debug(f"Cataloged: {file_path.name} ({file_type})")

    # Sort by file_path for reproducibility
    records.sort(key=lambda r: r['file_path'])

    return records


def write_inventory_csv(records: List[Dict[str, str]], output_path: Path):
    """
    Write inventory records to CSV file.

    Args:
        records: List of file records
        output_path: Path to output CSV file

    The output CSV has columns: file_path, file_type, section_dir, filename, needs_conversion
    """
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Define CSV columns
    fieldnames = ['file_path', 'file_type', 'section_dir', 'filename', 'needs_conversion']

    # Write CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    logger.info(f"Wrote {len(records)} records to {output_path}")


def print_summary(records: List[Dict[str, str]]):
    """
    Print summary statistics to stdout.

    Args:
        records: List of file records
    """
    if not records:
        logger.info("No supported files found")
        return

    # Count by type
    type_counts = {}
    conversion_count = 0

    for record in records:
        file_type = record['file_type']
        type_counts[file_type] = type_counts.get(file_type, 0) + 1

        if record['needs_conversion'] == 'true':
            conversion_count += 1

    # Print summary
    logger.info(f"=== Inventory Summary ===")
    logger.info(f"Total files cataloged: {len(records)}")
    logger.info(f"Files by type:")
    for file_type in sorted(type_counts.keys()):
        logger.info(f"  {file_type}: {type_counts[file_type]}")
    logger.info(f"Files needing conversion: {conversion_count}")
    logger.info(f"=========================")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stage 1: Inventory - Catalog all source files in data_src/',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/01_inventory.py --data-src data_src --output work/inventory.csv

  # With verbose logging
  python scripts/01_inventory.py --data-src data_src --output work/inventory.csv --verbose

Supported file types:
  - Images: .jpg, .jpeg, .png
  - Documents: .pdf
  - Web pages: .html, .htm

Output CSV schema:
  file_path        - Relative path from project root
  file_type        - Normalized type (jpg, png, pdf, html)
  section_dir      - Immediate parent directory name
  filename         - Base filename with extension
  needs_conversion - true for PDFs, false otherwise
        """
    )

    parser.add_argument(
        '--data-src',
        type=Path,
        required=True,
        help='Path to data source directory (read-only)'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Path to output CSV file (will be created/overwritten)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )

    parser.add_argument(
        '--section-filter',
        type=str,
        default="",
        help='Filter to specific section (e.g., "21" for Clutch)'
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info(f"Starting inventory scan of {args.data_src}")

        # Scan directory
        records = scan_directory(args.data_src)

        # Write output
        write_inventory_csv(records, args.output)

        # Print summary
        print_summary(records)

        logger.info("Inventory completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return 1
    except PermissionError as e:
        logger.error(f"Permission error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
