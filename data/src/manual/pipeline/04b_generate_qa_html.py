#!/usr/bin/env python3
"""
Stage 4b: HTML Q&A Generation

Generates question-answer pairs from HTML techspec files programmatically.
No API calls required - parses HTML tables and generates Q&A variations.

Usage:
    python scripts/04b_generate_qa_html.py \
        --data-src data_src \
        --output work/qa_raw \
        --config config.yaml
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# =============================================================================
# HTML Parsing
# =============================================================================

def parse_html_specs(html_path: Path) -> List[Dict]:
    """
    Parse HTML techspec file and extract spec rows.

    Returns list of dicts with:
    - category: str (e.g., "Engine", "Transmission")
    - spec_name: str (e.g., "Displacement")
    - spec_value: str (e.g., "2302 cc")
    """
    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_path}")

    with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    specs = []

    for table in soup.find_all('table', recursive=True):
        # Skip if this table is inside another table
        if table.find_parent('table'):
            continue

        rows = table.find_all('tr')
        if len(rows) < 2:
            continue

        # Detect structure from first data row
        first_data = None
        for row in rows:
            if not row.find('th'):
                first_data = row
                break
        if first_data is None and rows:
            first_data = rows[0]

        if first_data is None:
            continue

        cols = first_data.find_all(['td', 'th'])
        num_cols = len(cols)

        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if len(cells) < 2:
                continue

            if num_cols >= 3:
                # Assume: category, name, value [, unit]
                category = cells[0]
                spec_name = cells[1]
                spec_value = " ".join(cells[2:])  # Merge remaining
            else:
                # Assume: name, value
                category = ""
                spec_name = cells[0]
                spec_value = " ".join(cells[1:])

            # Skip header rows
            if spec_name.lower() in ('specification', 'spec', 'parameter', 'category', 'item', 'description'):
                continue

            # Skip empty values
            if not spec_value.strip():
                continue

            specs.append({
                "category": category,
                "spec_name": spec_name,
                "spec_value": spec_value
            })

    return specs


# =============================================================================
# Model Detection
# =============================================================================

def detect_model_from_filename(html_path: Path) -> str:
    """
    Detect vehicle model from HTML filename.

    M3-techspec.html -> "M3"
    320is-techspec.html -> "320is"
    """
    stem = html_path.stem.lower()

    if 'm3' in stem:
        return "M3"
    if '320is' in stem:
        return "320is"
    if '325is' in stem:
        return "325is"
    if '325i' in stem:
        return "325i"
    if '318i' in stem:
        return "318i"

    # Default fallback
    return "E30"


# =============================================================================
# Q&A Variation Generation
# =============================================================================

def generate_qa_variations(
    spec_name: str,
    spec_value: str,
    category: str,
    model: str
) -> List[Dict]:
    """
    Generate 2-3 question variations for a single spec.

    Variations:
    1. "What is the {spec_name} for the BMW E30 {model}?" -> "{spec_value}"
    2. "What is the {spec_name}?" -> "The {spec_name} for the E30 {model} is {spec_value}."
    3. (if category) "What is the {category} {spec_name}?" -> "{spec_value}"
    """
    qa_pairs = []

    # Clean up spec name for questions
    clean_name = spec_name.strip().lower()

    # Variation 1: Full context question with concise answer
    qa_pairs.append({
        "question": f"What is the {spec_name.lower()} for the BMW E30 {model}?",
        "answer": spec_value,
        "question_type": "factual"
    })

    # Variation 2: Simple question with full context answer
    qa_pairs.append({
        "question": f"What is the {spec_name.lower()}?",
        "answer": f"The {spec_name.lower()} for the BMW E30 {model} is {spec_value}.",
        "question_type": "factual"
    })

    # Variation 3: With category if available
    if category and category.lower() not in ('', 'general', 'specifications'):
        qa_pairs.append({
            "question": f"What is the {category.lower()} {spec_name.lower()} specification?",
            "answer": spec_value,
            "question_type": "factual"
        })

    return qa_pairs


def assign_qa_ids(qa_pairs: List[Dict], page_id: str) -> List[Dict]:
    """
    Assign unique IDs to Q&A pairs.

    Format: {page_id}-q{nn} (e.g., "html-m3-techspec-q01")
    """
    result = []
    for i, qa in enumerate(qa_pairs, start=1):
        qa_copy = qa.copy()
        qa_copy["id"] = f"{page_id}-q{i:02d}"
        result.append(qa_copy)
    return result


# =============================================================================
# Full HTML Q&A Generation
# =============================================================================

def generate_html_qa(
    html_path: Path,
    output_dir: Path,
    force: bool = False
) -> Dict:
    """
    Generate Q&A from HTML spec file.

    Returns summary dict with:
    - page_id: str
    - qa_count: int
    - output_path: Path
    """
    # Generate page_id from filename
    stem = html_path.stem.lower()
    page_id = f"html-{stem}"

    # Sanitize for filename
    safe_page_id = re.sub(r'[^\w\-]', '_', page_id)
    output_path = output_dir / f"{safe_page_id}.json"

    # Check if already exists (idempotent)
    if output_path.exists() and not force:
        logger.info(f"Skipping {html_path.name}: output already exists")
        # Read existing to get count
        with open(output_path, 'r') as f:
            existing = json.load(f)
        return {
            "page_id": page_id,
            "qa_count": len(existing.get("qa_pairs", [])),
            "output_path": output_path
        }

    # Parse HTML
    specs = parse_html_specs(html_path)
    logger.info(f"Parsed {len(specs)} specs from {html_path.name}")

    # Detect model
    model = detect_model_from_filename(html_path)
    logger.info(f"Detected model: {model}")

    # Generate Q&A variations
    all_qa_pairs = []
    for spec in specs:
        variations = generate_qa_variations(
            spec_name=spec["spec_name"],
            spec_value=spec["spec_value"],
            category=spec["category"],
            model=model
        )
        all_qa_pairs.extend(variations)

    # Assign IDs
    all_qa_pairs = assign_qa_ids(all_qa_pairs, page_id)

    # Build output data
    output_data = {
        "page_id": page_id,
        "image_path": None,
        "section_id": "techspec",
        "section_name": "Technical Specifications",
        "source_type": "html_specs",
        "content_type": "specification",
        "generation": {
            "method": "html_parse",
            "timestamp": datetime.now().isoformat(),
            "source_file": html_path.name,
            "model_detected": model,
            "specs_parsed": len(specs)
        },
        "qa_pairs": all_qa_pairs
    }

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs for {html_path.name}")

    return {
        "page_id": page_id,
        "qa_count": len(all_qa_pairs),
        "output_path": output_path
    }


# =============================================================================
# Find HTML Files
# =============================================================================

def find_html_spec_files(data_src: Path) -> List[Path]:
    """
    Find all HTML techspec files in data_src directory.
    """
    html_files = []

    # Look for *-techspec.html or *-specs.html patterns
    for pattern in ['*-techspec.html', '*-specs.html', '*techspec*.html']:
        html_files.extend(data_src.glob(pattern))
        # Also check subdirectories
        html_files.extend(data_src.glob(f'**/{pattern}'))

    # Also find any HTML files that might contain specs
    for html_path in data_src.glob('**/*.html'):
        if html_path not in html_files:
            # Check if it contains table elements (likely specs)
            try:
                with open(html_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(10000)  # Read first 10KB
                    if '<table' in content.lower():
                        html_files.append(html_path)
            except Exception:
                pass

    return sorted(set(html_files))


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stage 4b: Generate Q&A pairs from HTML techspec files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/04b_generate_qa_html.py \\
      --data-src data_src \\
      --output work/qa_raw \\
      --config config.yaml

  # Force regeneration
  python scripts/04b_generate_qa_html.py \\
      --data-src data_src \\
      --output work/qa_raw \\
      --config config.yaml \\
      --force
        """
    )

    parser.add_argument(
        '--data-src',
        type=Path,
        required=True,
        help='Path to data_src directory containing HTML files'
    )

    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for Q&A JSON files'
    )

    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to config.yaml'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regeneration even if output exists'
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
        logger.info("Starting Stage 4b: HTML Q&A Generation")

        # Load configuration (for future expansion)
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Find HTML files
        html_files = find_html_spec_files(args.data_src)
        logger.info(f"Found {len(html_files)} HTML spec files")

        if not html_files:
            logger.warning("No HTML spec files found")
            return 0

        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)

        # Process each HTML file
        total_qa = 0
        processed = 0
        skipped = 0

        for html_path in html_files:
            try:
                result = generate_html_qa(
                    html_path=html_path,
                    output_dir=args.output,
                    force=args.force
                )
                total_qa += result["qa_count"]
                if result["output_path"].exists():
                    processed += 1
            except Exception as e:
                logger.error(f"Error processing {html_path}: {e}")
                skipped += 1

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 4b: HTML Q&A GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"HTML files found: {len(html_files)}")
        logger.info(f"Processed: {processed}")
        logger.info(f"Skipped/Failed: {skipped}")
        logger.info(f"Total Q&A pairs: {total_qa}")
        logger.info("=" * 60)

        logger.info("Stage 4b completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
