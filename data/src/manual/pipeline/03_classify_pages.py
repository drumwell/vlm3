#!/usr/bin/env python3
"""
Stage 3: Classification & Index Parsing
Classifies pages by content type and source type, extracts index metadata.

This script processes the prepared inventory from Stage 2, detects source types,
identifies and parses index pages, classifies content types, and prepares
metadata for Q&A generation in Stage 4.

Usage:
    python scripts/03_classify_pages.py \
        --inventory work/inventory_prepared.csv \
        --output-csv work/classified/pages.csv \
        --output-indices work/indices \
        --config config.yaml
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
import base64
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv
import yaml

load_dotenv()
from PIL import Image

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
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# =============================================================================
# Source Type Detection
# =============================================================================

def detect_source_type(section_dir: str, filename: str) -> str:
    """
    Detect source type from directory name and filename.

    Args:
        section_dir: Section directory name (e.g., "21 - Clutch")
        filename: Base filename (e.g., "21-01.jpg")

    Returns:
        Source type: 'service_manual', 'electrical_manual', 'ecu_technical',
                     'html_specs', or 'unknown'
    """
    # Check for HTML files first
    if filename.lower().endswith('.html'):
        return 'html_specs'

    # Check for numbered section pattern (XX - Name)
    if re.match(r'^[0-9]{2} - ', section_dir):
        return 'service_manual'

    # Check for Getrag
    if section_dir.lower().startswith('getrag'):
        return 'service_manual'

    # Check for Electrical Troubleshooting Manual
    if 'electrical troubleshooting' in section_dir.lower():
        return 'electrical_manual'

    # Check for Bosch Motronic
    if 'bosch motronic' in section_dir.lower():
        return 'ecu_technical'

    return 'unknown'


# =============================================================================
# Index Page Detection
# =============================================================================

def is_index_page(filename: str) -> bool:
    """
    Determine if a page is an index/table of contents.

    Args:
        filename: Base filename (e.g., "21-00-index-a.jpg")

    Returns:
        True if filename matches index patterns
    """
    filename_lower = filename.lower()

    # Check for index pattern
    if '-index-' in filename_lower or '-index.' in filename_lower:
        return True

    # Check for toc pattern
    if '-toc-' in filename_lower or '-toc.' in filename_lower:
        return True

    return False


# =============================================================================
# Page ID and Section Slug Generation
# =============================================================================

def generate_page_id(filename: str, section_dir: str) -> str:
    """
    Generate unique page ID from filename and section.

    Args:
        filename: Base filename
        section_dir: Section directory name

    Returns:
        Page ID string (includes section slug for uniqueness)
    """
    # Remove extension
    stem = Path(filename).stem
    slug = generate_section_slug(section_dir)

    # Check for Getrag - add prefix
    if section_dir.lower().startswith('getrag') and re.match(r'^\d+$', stem):
        return f"getrag-{stem}"

    # For numbered sections (XX - Name), check if filename already has section prefix
    section_match = re.match(r'^(\d{2}) - ', section_dir)
    if section_match:
        section_num = section_match.group(1)
        # If filename starts with section number, use as-is
        if stem.startswith(f"{section_num}-"):
            # But need to disambiguate sections with same number prefix
            # e.g., "00 - Maintenance" vs "00 - Torque Specs"
            # Add short slug suffix for disambiguation
            if " - " in section_dir:
                section_name_slug = generate_section_slug(section_dir.split(" - ", 1)[1])[:8]
                return f"{stem}_{section_name_slug}"
            return stem
        else:
            # Use section slug as prefix
            return f"{slug}-{stem}"

    # Check for numbered files in non-standard sections
    if re.match(r'^\d+$', stem):
        # Generic numbered file, use full section slug as prefix
        return f"{slug}-{stem}"

    # For non-standard filenames, prepend section slug for uniqueness
    return f"{slug}-{stem}"


def generate_section_slug(section_dir: str) -> str:
    """
    Convert section directory name to filesystem-safe slug.

    Args:
        section_dir: Section directory name (e.g., "21 - Clutch")

    Returns:
        Slug string (e.g., "21-clutch")
    """
    slug = section_dir.lower()
    slug = slug.replace(" - ", "-")
    slug = slug.replace(" ", "-")
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    slug = re.sub(r'-+', '-', slug)
    slug = slug.strip('-')

    return slug


# =============================================================================
# Image Encoding
# =============================================================================

def encode_image_base64(image_path: Path) -> Tuple[str, str]:
    """
    Encode image to base64 for Claude API.

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (base64_data, media_type)

    Raises:
        FileNotFoundError: If image doesn't exist
        IOError: If image can't be read
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Determine media type
    suffix = image_path.suffix.lower()
    media_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_types.get(suffix, 'image/jpeg')

    with open(image_path, 'rb') as f:
        image_data = f.read()

    base64_data = base64.standard_b64encode(image_data).decode('utf-8')

    return base64_data, media_type


# =============================================================================
# Cache Loading
# =============================================================================

def load_cached_results(cache_path: Path) -> Dict[str, Dict]:
    """
    Load existing classification results from cache file.

    Args:
        cache_path: Path to pages.csv

    Returns:
        Dict mapping page_id -> full classification record
        Empty dict if file doesn't exist
    """
    if not cache_path.exists():
        return {}

    cached = {}
    try:
        with open(cache_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                page_id = row.get('page_id')
                if page_id:
                    # Convert string booleans
                    if 'is_index' in row:
                        row['is_index'] = row['is_index'].lower() == 'true'
                    if 'needs_review' in row:
                        row['needs_review'] = row['needs_review'].lower() == 'true'
                    if 'confidence' in row:
                        try:
                            row['confidence'] = float(row['confidence'])
                        except (ValueError, TypeError):
                            row['confidence'] = 0.0
                    cached[page_id] = row
    except Exception as e:
        logger.warning(f"Error loading cache: {e}")
        return {}

    return cached


# =============================================================================
# Inventory Reading
# =============================================================================

def read_prepared_inventory(inventory_path: Path) -> List[Dict[str, str]]:
    """
    Read prepared inventory CSV from Stage 2.

    Args:
        inventory_path: Path to inventory_prepared.csv

    Returns:
        List of inventory records (dicts), filtered to image files only
    """
    records = []

    with open(inventory_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter to image files only
            file_type = row.get('file_type', '').lower()
            if file_type in ('jpg', 'jpeg', 'png'):
                records.append(row)

    return records


# =============================================================================
# Index Parsing (Claude API)
# =============================================================================

INDEX_SYSTEM_PROMPT = """You are extracting structured repair procedure data from a BMW E30 M3 service manual index page.

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
    {"code": "21 00 006", "name": "Clutch - bleed", "page": "21-1"},
    {"code": "21 11 000", "name": "Clutch housing - remove and install", "page": "21-1"}
  ]
}

Do not include any explanatory text, only the JSON object."""

INDEX_USER_PROMPT = """Extract all repair procedures from this BMW E30 M3 service manual index page.

Guidelines:
- Repair codes follow the pattern: XX YY ZZZ (e.g., "21 00 006")
- Some entries may be just numbers (e.g., "565") without the full code format
- Page references are usually "XX-Y" format (e.g., "21-1", "21-2")
- Include ALL procedures visible on this page
- If the section ID or name is not clearly visible, infer from the content

Return the JSON object only."""


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
        section_dir: Section directory name
        api_key: Anthropic API key
        model: Claude model name
        max_retries: Maximum retry attempts

    Returns:
        Dictionary with extracted metadata
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        return {
            "section_id": None,
            "section_name": None,
            "procedures": [],
            "source_filename": image_path.name,
            "error": "anthropic package not installed"
        }

    try:
        base64_data, media_type = encode_image_base64(image_path)
    except Exception as e:
        return {
            "section_id": None,
            "section_name": None,
            "procedures": [],
            "source_filename": image_path.name,
            "error": f"Image read error: {str(e)}"
        }

    client = Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=INDEX_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data
                                }
                            },
                            {
                                "type": "text",
                                "text": INDEX_USER_PROMPT
                            }
                        ]
                    }
                ]
            )

            # Parse response
            json_text = response.content[0].text.strip()

            # Try to extract JSON if wrapped in markdown
            if json_text.startswith('```'):
                lines = json_text.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```') and not in_json:
                        in_json = True
                        continue
                    elif line.startswith('```') and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                json_text = '\n'.join(json_lines)

            data = json.loads(json_text)
            data["source_filename"] = image_path.name
            data["error"] = None

            return data

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "section_id": None,
                "section_name": None,
                "procedures": [],
                "source_filename": image_path.name,
                "error": f"JSON parse error: {str(e)}"
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "section_id": None,
                "section_name": None,
                "procedures": [],
                "source_filename": image_path.name,
                "error": f"API error: {str(e)}"
            }


# =============================================================================
# Content Classification (Claude Vision API)
# =============================================================================

CLASSIFICATION_SYSTEM_PROMPT = """You are classifying pages from a BMW E30 M3 service manual.

Analyze the visual content and structure of the page to determine its PRIMARY type.
If the page contains multiple content types, identify the DOMINANT type based on visual area and semantic importance.

Priority order for mixed-content pages:
1. troubleshooting - If 3-column Condition/Cause/Correction table present
2. procedure - If numbered steps with photos/callouts present
3. specification - If data tables with values/units dominate
4. diagram - If >60% of page is technical illustration
5. wiring/pinout/fuse_chart - Electrical-specific content
6. text - Fallback for text-heavy pages

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
  "secondary_types": ["specification"],
  "reasoning": "Brief explanation"
}"""

CLASSIFICATION_USER_PROMPT = """Classify this page from a {source_type}.

Look at the visual layout and content:
- Are there multiple photos with numbered callouts? → procedure
- Is it a 3-column table (Condition/Cause/Correction)? → troubleshooting
- Is it a full-page technical drawing with minimal text? → diagram
- Is it a data table with values and units? → specification
- Is it a table of contents with repair codes? → index

Return JSON only with content_type, confidence (0-1), secondary_types (list), and reasoning."""


def classify_page_content(
    image_path: Path,
    source_type: str,
    is_index: bool,
    api_key: str,
    model: str,
    max_retries: int = 3
) -> Dict:
    """
    Classify page content type using Claude Vision API.

    Args:
        image_path: Path to page image
        source_type: Source type from detect_source_type()
        is_index: Whether page is an index page (skip API if True)
        api_key: Anthropic API key
        model: Claude model for classification
        max_retries: Maximum retry attempts

    Returns:
        Dictionary with classification
    """
    # Skip API for index pages detected by filename
    if is_index:
        return {
            "content_type": "index",
            "confidence": 0.95,
            "secondary_types": [],
            "reasoning": "Detected by filename pattern",
            "api_called": False,
            "error": None
        }

    try:
        from anthropic import Anthropic
    except ImportError:
        return {
            "content_type": "unknown",
            "confidence": 0.0,
            "secondary_types": [],
            "reasoning": None,
            "api_called": False,
            "error": "anthropic package not installed"
        }

    try:
        base64_data, media_type = encode_image_base64(image_path)
    except FileNotFoundError as e:
        return {
            "content_type": "unknown",
            "confidence": 0.0,
            "secondary_types": [],
            "reasoning": None,
            "api_called": False,
            "error": str(e)
        }
    except Exception as e:
        return {
            "content_type": "unknown",
            "confidence": 0.0,
            "secondary_types": [],
            "reasoning": None,
            "api_called": False,
            "error": f"Image read error: {str(e)}"
        }

    client = Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                system=CLASSIFICATION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data
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
            json_text = response.content[0].text.strip()

            # Try to extract JSON if wrapped in markdown
            if json_text.startswith('```'):
                lines = json_text.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.startswith('```') and not in_json:
                        in_json = True
                        continue
                    elif line.startswith('```') and in_json:
                        break
                    elif in_json:
                        json_lines.append(line)
                json_text = '\n'.join(json_lines)

            data = json.loads(json_text)
            data["api_called"] = True
            data["error"] = None

            # Ensure secondary_types exists
            if "secondary_types" not in data:
                data["secondary_types"] = []

            return data

        except json.JSONDecodeError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "content_type": "unknown",
                "confidence": 0.3,
                "secondary_types": [],
                "reasoning": None,
                "api_called": True,
                "error": f"JSON parse error: {str(e)}"
            }
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return {
                "content_type": "unknown",
                "confidence": 0.3,
                "secondary_types": [],
                "reasoning": None,
                "api_called": True,
                "error": f"API error: {str(e)}"
            }


# =============================================================================
# Main Classification Processing
# =============================================================================

def process_classification(
    inventory_records: List[Dict[str, str]],
    config: Dict,
    cached_results: Optional[Dict[str, Dict]] = None,
    skip_index_parsing: bool = False,
    skip_classification: bool = False,
    limit: Optional[int] = None,
    start_from: Optional[str] = None,
    output_path: Optional[Path] = None
) -> Tuple[List[Dict], Dict[str, List[Dict]], List[Dict]]:
    """
    Main classification processing logic.

    Args:
        inventory_records: Records from inventory_prepared.csv
        config: Configuration dictionary
        cached_results: Optional dict of already-classified pages
        skip_index_parsing: Skip index page parsing
        skip_classification: Skip content classification
        limit: Limit to first N images
        start_from: Start processing from this page_id
        output_path: Path for checkpoint saves

    Returns:
        Tuple of (classification_records, index_results, errors)
    """
    if cached_results is None:
        cached_results = {}

    api_key = config.get('api', {}).get('api_key')
    if not api_key:
        api_key = os.environ.get('ANTHROPIC_API_KEY')

    index_model = config['api']['models']['index_parsing']
    classification_model = config['api']['models']['classification']
    rate_limit_delay = config.get('api', {}).get('rate_limit_delay_seconds', 0.5)
    batch_size = config.get('classification', {}).get('batch_size', 10)

    classification_records = []
    index_results = {}  # section_slug -> list of parse results
    errors = []

    # Apply limit
    if limit:
        inventory_records = inventory_records[:limit]

    # Find start point
    start_index = 0
    if start_from:
        for i, record in enumerate(inventory_records):
            page_id = generate_page_id(record['filename'], record['section_dir'])
            if page_id == start_from:
                start_index = i
                break

    total = len(inventory_records)
    processed = 0
    api_calls = 0

    for i, record in enumerate(inventory_records[start_index:], start=start_index):
        filename = record['filename']
        section_dir = record['section_dir']
        file_path = record['file_path']

        page_id = generate_page_id(filename, section_dir)

        # Check cache
        if page_id in cached_results:
            classification_records.append(cached_results[page_id])
            processed += 1
            continue

        # Detect source type
        source_type = detect_source_type(section_dir, filename)

        # Check if index page
        is_index = is_index_page(filename)

        # Parse section info from directory
        section_slug = generate_section_slug(section_dir)
        section_match = re.match(r'^(\d{2}) - (.+)$', section_dir)
        if section_match:
            section_id = section_match.group(1)
            section_name = section_match.group(2)
        else:
            section_id = section_slug.split('-')[0] if '-' in section_slug else ""
            section_name = section_dir

        # Parse index page if applicable
        if is_index and not skip_index_parsing and api_key:
            logger.info(f"Parsing index page: {filename}")
            image_path = Path(file_path)

            if image_path.exists():
                index_data = parse_index_page(
                    image_path=image_path,
                    section_dir=section_dir,
                    api_key=api_key,
                    model=index_model
                )
                api_calls += 1

                if section_slug not in index_results:
                    index_results[section_slug] = []
                index_results[section_slug].append(index_data)

                if index_data.get("error"):
                    errors.append({
                        "timestamp": datetime.now().isoformat(),
                        "filename": filename,
                        "operation": "index_parsing",
                        "error_message": index_data["error"]
                    })

                time.sleep(rate_limit_delay)
            else:
                errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "filename": filename,
                    "operation": "index_parsing",
                    "error_message": f"Image file does not exist: {file_path}"
                })

        # Classify content
        classification = None
        if not skip_classification:
            image_path = Path(file_path)

            if not image_path.exists():
                errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "filename": filename,
                    "operation": "classification",
                    "error_message": f"Image file does not exist: {file_path}"
                })
                continue

            if is_index:
                # Skip API call for index pages
                classification = classify_page_content(
                    image_path=image_path,
                    source_type=source_type,
                    is_index=True,
                    api_key=api_key,
                    model=classification_model
                )
            elif api_key:
                classification = classify_page_content(
                    image_path=image_path,
                    source_type=source_type,
                    is_index=False,
                    api_key=api_key,
                    model=classification_model
                )
                api_calls += 1
                time.sleep(rate_limit_delay)
            else:
                classification = {
                    "content_type": "unknown",
                    "confidence": 0.0,
                    "secondary_types": [],
                    "reasoning": "No API key provided",
                    "api_called": False,
                    "error": "No API key"
                }

            if classification.get("error") and classification.get("api_called"):
                errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "filename": filename,
                    "operation": "classification",
                    "error_message": classification["error"]
                })

        # Build classification record
        confidence = classification.get("confidence", 0.0) if classification else 0.0
        needs_review = confidence < 0.7 or (classification and classification.get("content_type") == "unknown")

        record_out = {
            "page_id": page_id,
            "image_path": file_path,
            "section_id": section_id,
            "section_name": section_name,
            "source_type": source_type,
            "content_type": classification.get("content_type", "unknown") if classification else "unknown",
            "is_index": is_index,
            "confidence": confidence,
            "secondary_types": json.dumps(classification.get("secondary_types", [])) if classification else "[]",
            "needs_review": needs_review
        }

        classification_records.append(record_out)
        processed += 1

        # Progress logging
        if processed % batch_size == 0:
            logger.info(f"Processed {processed}/{total} pages ({api_calls} API calls)")

            # Checkpoint save
            if output_path:
                write_classification_csv(classification_records, output_path)
                logger.debug(f"Checkpoint saved: {len(classification_records)} records")

    logger.info(f"Classification complete: {processed} pages, {api_calls} API calls, {len(errors)} errors")

    return classification_records, index_results, errors


# =============================================================================
# Index Metadata Merging
# =============================================================================

def merge_index_metadata(index_results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Merge index metadata from multiple index pages per section.

    Args:
        index_results: Dict mapping section_slug -> list of parse results

    Returns:
        Dict mapping section_slug -> merged metadata
    """
    merged = {}

    for section_slug, parse_results in index_results.items():
        procedures_by_code = {}
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
                code = proc.get("code", "")
                name = proc.get("name", "")
                page = proc.get("page", "")

                if not code:
                    continue

                if code in procedures_by_code:
                    existing = procedures_by_code[code]
                    existing["pages"].add(page)
                    if existing["name"] != name and name:
                        warnings.append(f"Code {code}: name mismatch '{existing['name']}' vs '{name}'")
                else:
                    procedures_by_code[code] = {
                        "name": name,
                        "pages": {page} if page else set()
                    }

        # Convert to final format
        procedures = []
        for code, data in sorted(procedures_by_code.items()):
            procedures.append({
                "code": code,
                "name": data["name"],
                "pages": sorted(data["pages"])
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


# =============================================================================
# Output Writing
# =============================================================================

def write_classification_csv(records: List[Dict], output_path: Path):
    """
    Write classification CSV with needs_review flag.

    Args:
        records: Classification records
        output_path: Path to output CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort by page_id for reproducibility
    sorted_records = sorted(records, key=lambda r: r.get("page_id", ""))

    fieldnames = [
        "page_id", "image_path", "section_id", "section_name",
        "source_type", "content_type", "is_index", "confidence",
        "secondary_types", "needs_review"
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(record)


def write_index_json(metadata: Dict, output_path: Path):
    """
    Write index metadata JSON file.

    Args:
        metadata: Index metadata for one section
        output_path: Path to JSON file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def write_error_log(errors: List[Dict], log_path: Path):
    """
    Write error log CSV.

    Args:
        errors: Error records
        log_path: Path to log CSV
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "filename", "operation", "error_message"]

    # Check if file exists to determine if we need header
    file_exists = log_path.exists()

    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for error in errors:
            writer.writerow(error)


def write_manual_review_queue(records: List[Dict], output_path: Path):
    """
    Write manual review queue for low-confidence classifications.

    Args:
        records: Classification records
        output_path: Path to output CSV
    """
    low_confidence = [r for r in records if r.get("needs_review", False)]

    if not low_confidence:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["page_id", "image_path", "content_type", "confidence", "secondary_types"]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(low_confidence, key=lambda r: r.get("confidence", 0)):
            writer.writerow({k: record.get(k) for k in fieldnames})

    logger.info(f"Wrote {len(low_confidence)} pages to manual review queue")


# =============================================================================
# Report Generation
# =============================================================================

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
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Calculate statistics
    total_pages = len(classification_records)
    index_pages = sum(1 for r in classification_records if r.get("is_index"))
    sections_with_indices = len(index_metadata)
    total_procedures = sum(
        len(m.get("procedures", []))
        for m in index_metadata.values()
    )

    # Source type distribution
    source_types = {}
    for r in classification_records:
        st = r.get("source_type", "unknown")
        source_types[st] = source_types.get(st, 0) + 1

    # Content type distribution
    content_types = {}
    for r in classification_records:
        ct = r.get("content_type", "unknown")
        content_types[ct] = content_types.get(ct, 0) + 1

    # Confidence distribution
    high_conf = sum(1 for r in classification_records if r.get("confidence", 0) >= 0.9)
    med_conf = sum(1 for r in classification_records if 0.7 <= r.get("confidence", 0) < 0.9)
    low_conf = sum(1 for r in classification_records if r.get("confidence", 0) < 0.7)

    # Generate report
    lines = [
        "# Stage 3: Classification & Index Parsing Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary Statistics",
        "",
        f"- **Total Pages Classified**: {total_pages}",
        f"- **Index Pages**: {index_pages}",
        f"- **Sections with Indices**: {sections_with_indices}",
        f"- **Total Procedures Extracted**: {total_procedures}",
        f"- **Errors Encountered**: {len(errors)}",
        "",
        "## Source Type Distribution",
        "",
    ]

    for st, count in sorted(source_types.items(), key=lambda x: -x[1]):
        pct = count / total_pages * 100 if total_pages > 0 else 0
        lines.append(f"- {st}: {count} pages ({pct:.1f}%)")

    lines.extend([
        "",
        "## Content Type Distribution",
        "",
    ])

    for ct, count in sorted(content_types.items(), key=lambda x: -x[1]):
        pct = count / total_pages * 100 if total_pages > 0 else 0
        lines.append(f"- {ct}: {count} pages ({pct:.1f}%)")

    lines.extend([
        "",
        "## Classification Confidence Distribution",
        "",
        f"- High (≥0.9): {high_conf} pages ({high_conf/total_pages*100:.1f}%)" if total_pages > 0 else "- High (≥0.9): 0 pages",
        f"- Medium (0.7-0.9): {med_conf} pages ({med_conf/total_pages*100:.1f}%)" if total_pages > 0 else "- Medium (0.7-0.9): 0 pages",
        f"- Low (<0.7): {low_conf} pages ({low_conf/total_pages*100:.1f}%) ← flagged for review" if total_pages > 0 else "- Low (<0.7): 0 pages",
        "",
        "## Index Metadata Summary",
        "",
    ])

    for section_slug, meta in sorted(index_metadata.items()):
        proc_count = len(meta.get("procedures", []))
        page_count = len(meta.get("page_to_procedures", {}))
        warnings = len(meta.get("merge_warnings", []))
        warning_str = f" ⚠️ {warnings} warnings" if warnings > 0 else ""
        lines.append(f"- **{section_slug}**: {proc_count} procedures, {page_count} pages{warning_str}")

    if errors:
        lines.extend([
            "",
            "## Errors",
            "",
        ])
        for error in errors[:20]:  # Limit to first 20
            lines.append(f"- [{error.get('operation')}] {error.get('filename')}: {error.get('error_message')}")
        if len(errors) > 20:
            lines.append(f"- ... and {len(errors) - 20} more errors")

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))


def print_summary(
    classification_records: List[Dict],
    index_metadata: Dict[str, Dict],
    errors: List[Dict]
):
    """
    Print summary statistics to stdout.
    """
    total_pages = len(classification_records)
    index_pages = sum(1 for r in classification_records if r.get("is_index"))
    total_procedures = sum(
        len(m.get("procedures", []))
        for m in index_metadata.values()
    )

    # Content type counts
    content_types = {}
    for r in classification_records:
        ct = r.get("content_type", "unknown")
        content_types[ct] = content_types.get(ct, 0) + 1

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3: CLASSIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total pages classified: {total_pages}")
    logger.info(f"Index pages: {index_pages}")
    logger.info(f"Sections with indices: {len(index_metadata)}")
    logger.info(f"Procedures extracted: {total_procedures}")
    logger.info(f"Errors: {len(errors)}")
    logger.info("")
    logger.info("Content type distribution:")
    for ct, count in sorted(content_types.items(), key=lambda x: -x[1]):
        logger.info(f"  {ct}: {count}")
    logger.info("=" * 60)


# =============================================================================
# Sample Selection for Validation
# =============================================================================

def select_validation_samples(inventory_records: List[Dict], n: int = 10) -> List[Dict]:
    """
    Select diverse sample images for validation.

    Args:
        inventory_records: All inventory records
        n: Number of samples to select

    Returns:
        List of sample records
    """
    # Try to get diverse samples from different sections
    samples = []
    sections_seen = set()

    # Prioritize specific filenames known to represent different types
    priority_patterns = [
        r'-01\.jpg$',      # First page of section (often procedure)
        r'-07\.jpg$',      # Often troubleshooting
        r'-50\.jpg$',      # Often diagram
        r'0670.*\.jpg$',   # Electrical manual
    ]

    for pattern in priority_patterns:
        for record in inventory_records:
            if re.search(pattern, record['filename']) and record['section_dir'] not in sections_seen:
                samples.append(record)
                sections_seen.add(record['section_dir'])
                if len(samples) >= n:
                    break
        if len(samples) >= n:
            break

    # Fill remaining with random diverse samples
    for record in inventory_records:
        if len(samples) >= n:
            break
        if record['section_dir'] not in sections_seen and not is_index_page(record['filename']):
            samples.append(record)
            sections_seen.add(record['section_dir'])

    return samples[:n]


# =============================================================================
# Main Entry Point
# =============================================================================

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

  # With limit for testing
  python scripts/03_classify_pages.py \\
      --inventory work/inventory_prepared.csv \\
      --output-csv work/classified/pages.csv \\
      --output-indices work/indices \\
      --config config.yaml \\
      --limit 100

  # Sample validation (10 images)
  python scripts/03_classify_pages.py \\
      --inventory work/inventory_prepared.csv \\
      --output-csv work/classified/pages_sample.csv \\
      --output-indices work/indices_sample \\
      --config config.yaml \\
      --sample-validation
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
        help='Limit processing to first N images'
    )

    parser.add_argument(
        '--start-from',
        type=str,
        default=None,
        help='Start processing from this page_id'
    )

    parser.add_argument(
        '--sample-validation',
        action='store_true',
        help='Run on 10 diverse sample images for manual review'
    )

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        logger.info("Starting Stage 3: Classification & Index Parsing")

        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Get API key
        api_key = config.get('api', {}).get('api_key')
        if not api_key or api_key.startswith('${'):
            api_key = os.environ.get('ANTHROPIC_API_KEY')
            if api_key:
                if 'api' not in config:
                    config['api'] = {}
                config['api']['api_key'] = api_key
            else:
                logger.warning("ANTHROPIC_API_KEY not found - will skip API calls")

        # Read inventory
        inventory_records = read_prepared_inventory(args.inventory)
        logger.info(f"Read {len(inventory_records)} image records from inventory")

        # Sample validation mode
        if args.sample_validation:
            inventory_records = select_validation_samples(inventory_records, 10)
            logger.info(f"Sample validation mode: selected {len(inventory_records)} diverse samples")

        # Load cache unless force mode
        cached_results = {}
        if not args.force and args.output_csv.exists():
            cached_results = load_cached_results(args.output_csv)
            logger.info(f"Loaded {len(cached_results)} cached results")

        # Process classification
        classification_records, index_results, errors = process_classification(
            inventory_records=inventory_records,
            config=config,
            cached_results=cached_results,
            skip_index_parsing=args.skip_index_parsing,
            skip_classification=args.skip_classification,
            limit=args.limit,
            start_from=args.start_from,
            output_path=args.output_csv
        )

        # Merge index metadata
        index_metadata = merge_index_metadata(index_results)

        # Write outputs
        write_classification_csv(classification_records, args.output_csv)
        logger.info(f"Wrote {len(classification_records)} classification records to {args.output_csv}")

        # Write index JSON files
        args.output_indices.mkdir(parents=True, exist_ok=True)
        for section_slug, metadata in index_metadata.items():
            json_path = args.output_indices / f"{section_slug}.json"
            write_index_json(metadata, json_path)
        logger.info(f"Wrote {len(index_metadata)} index JSON files to {args.output_indices}")

        # Write error log if errors occurred
        if errors:
            error_log_path = args.output_csv.parent.parent / "logs" / "classification_errors.csv"
            write_error_log(errors, error_log_path)
            logger.warning(f"Encountered {len(errors)} errors; see {error_log_path}")

        # Write manual review queue
        review_queue_path = args.output_csv.parent.parent / "logs" / "manual_review_queue.csv"
        write_manual_review_queue(classification_records, review_queue_path)

        # Generate report
        report_path = args.output_csv.parent.parent / "logs" / "stage3_classification_report.md"
        generate_report(classification_records, index_metadata, errors, report_path)
        logger.info(f"Generated report at {report_path}")

        # Print summary
        print_summary(classification_records, index_metadata, errors)

        logger.info("Stage 3 completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
