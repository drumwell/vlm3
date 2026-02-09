#!/usr/bin/env python3
"""
Stage 4a: Image Q&A Generation

Generates question-answer pairs from service manual images using Claude Vision API.
Uses context from Stage 3 classification and index metadata.

Usage:
    python scripts/04a_generate_qa_images.py \
        --classified work/classified/pages.csv \
        --indices work/indices \
        --output work/qa_raw \
        --config config.yaml
"""

import argparse
import base64
import csv
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# =============================================================================
# Page ID Normalization
# =============================================================================

def normalize_page_id(page_id: str) -> str:
    """
    Normalize page ID for matching across sources.
    Strips leading zeros from page number portion.

    Examples:
        "21-01" -> "21-1"
        "21-1"  -> "21-1"
        "ETM-001" -> "ETM-1"
    """
    parts = page_id.split("-")
    if len(parts) == 2:
        section, page_num = parts
        # Strip leading zeros from page number
        if page_num.isdigit():
            page_num = str(int(page_num))
        else:
            page_num = page_num.lstrip("0") or "0"
        return f"{section}-{page_num}"
    return page_id


# =============================================================================
# Section Slug Derivation
# =============================================================================

def derive_section_slug(section_id: str, section_name: str) -> str:
    """
    Derive section slug for index file lookup.

    Examples:
        ("21", "Clutch") -> "21-clutch"
        ("00", "Maintenance") -> "00-maintenance"
        ("ETM", "Electrical Troubleshooting") -> "etm-electrical-troubleshooting"
    """
    # Lowercase, replace spaces with hyphens, remove special chars
    slug = section_name.lower().replace(" ", "-")
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    return f"{section_id.lower()}-{slug}"


# =============================================================================
# Index Directory Discovery
# =============================================================================

def find_indices_dir(work_dir: Path) -> Path:
    """
    Find the indices directory, checking common locations.

    Checks: work/indices/, work/indices_100/, work/indices_50/
    Returns first directory that exists and contains .json files.
    Raises FileNotFoundError if none found.
    """
    candidates = [
        work_dir / "indices",
        work_dir / "indices_100",
        work_dir / "indices_50",
    ]
    for path in candidates:
        if path.exists() and list(path.glob("*.json")):
            return path
    raise FileNotFoundError(f"No index files found in {candidates}")


# =============================================================================
# Image Preprocessing
# =============================================================================

def preprocess_image_for_api(
    image_path: Path,
    max_size_mb: float = 5.0,
    max_dim: int = 4096
) -> Tuple[bytes, str]:
    """
    Preprocess image for Claude API.

    - Resize if dimensions exceed max_dim
    - Compress if file size exceeds max_size_mb
    - Convert RGBA/P to RGB for JPEG

    Returns (image_bytes, media_type)
    Raises ValueError if cannot compress below limit.
    """
    with Image.open(image_path) as img:
        # Resize if too large
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Convert to RGB if needed (for JPEG)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Compress to fit size limit
        quality = 85
        while quality > 20:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            if buffer.tell() <= max_size_mb * 1024 * 1024:
                buffer.seek(0)
                return buffer.read(), 'image/jpeg'
            quality -= 10

        raise ValueError(f"Cannot compress {image_path} below {max_size_mb}MB")


# =============================================================================
# Q&A Validation
# =============================================================================

def validate_qa_pair(qa: Dict, page_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a Q&A pair for quality issues.

    Checks:
    - Required fields present and minimum length
    - Question ends with '?'
    - No hallucination phrases in answer
    - No self-referential language in question
    - Valid question_type

    Returns (is_valid, list_of_warnings)
    """
    warnings = []

    # Required fields
    if not qa.get("question") or len(qa["question"]) < 10:
        return False, ["Question too short or missing"]
    if not qa.get("answer") or len(qa["answer"]) < 5:
        return False, ["Answer too short or missing"]

    # Question must be a question
    if not qa["question"].rstrip().endswith("?"):
        warnings.append("Question doesn't end with '?'")

    # Answer shouldn't be a question
    if qa["answer"].rstrip().endswith("?"):
        warnings.append("Answer ends with '?' (may be a question)")

    # Detect potential hallucination patterns
    hallucination_phrases = [
        "I cannot see", "I can't determine", "not visible",
        "image doesn't show", "unclear from the image",
        "I would need", "typically", "usually", "generally"
    ]
    answer_lower = qa["answer"].lower()
    for phrase in hallucination_phrases:
        if phrase in answer_lower:
            warnings.append(f"Potential hedge/hallucination: '{phrase}'")

    # Check for self-referential language (should be removed)
    bad_patterns = [
        "on this page", "in this image", "as shown",
        "the manual states", "according to the page"
    ]
    for pattern in bad_patterns:
        if pattern in qa["question"].lower():
            warnings.append(f"Self-referential question: '{pattern}'")

    # Validate question_type
    valid_types = {
        "factual", "procedural", "visual", "inspection", "tool",
        "safety", "navigation", "wiring", "connector", "component",
        "diagnostic", "troubleshooting", "signal", "parameter", "operation"
    }
    if qa.get("question_type") not in valid_types:
        warnings.append(f"Invalid question_type: {qa.get('question_type')}")

    return len(warnings) == 0 or not any("too short" in w or "missing" in w for w in warnings), warnings


def filter_and_validate_qa_pairs(
    qa_pairs: List[Dict],
    page_data: Dict,
    strict: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter Q&A pairs, returning (valid_pairs, rejected_pairs).

    If strict=True, reject any pair with warnings.
    If strict=False, only reject pairs that fail hard validation.
    """
    valid = []
    rejected = []

    for qa in qa_pairs:
        is_valid, warnings = validate_qa_pair(qa, page_data)

        if not is_valid or (strict and warnings):
            rejected.append({"qa": qa, "reasons": warnings})
        else:
            if warnings:
                qa["_warnings"] = warnings  # Attach for review
            valid.append(qa)

    return valid, rejected


# =============================================================================
# Classification Data Loading
# =============================================================================

def load_classification_data(csv_path: Path) -> List[Dict]:
    """
    Load classification data from pages.csv.

    Returns list of dicts with: page_id, image_path, section_id,
    section_name, source_type, content_type, is_index, confidence
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Classification CSV not found: {csv_path}")

    records = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert is_index to boolean
            row['is_index'] = row.get('is_index', '').lower() == 'true'
            # Convert confidence to float
            try:
                row['confidence'] = float(row.get('confidence', 0))
            except (ValueError, TypeError):
                row['confidence'] = 0.0
            records.append(row)

    return records


# =============================================================================
# Index Metadata Loading
# =============================================================================

def load_index_metadata(indices_dir: Path) -> Dict[str, Dict]:
    """
    Load all index metadata from work/indices/*.json.

    Returns dict mapping section_slug -> index metadata
    """
    if not indices_dir.exists():
        return {}

    metadata = {}
    for json_path in indices_dir.glob("*.json"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            section_slug = json_path.stem
            metadata[section_slug] = data
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Error loading index {json_path}: {e}")

    return metadata


# =============================================================================
# Procedure Lookup
# =============================================================================

def get_procedures_for_page(
    page_id: str,
    section_slug: str,
    index_metadata: Dict[str, Dict]
) -> Tuple[List[str], List[str]]:
    """
    Get procedures covered by this page from index metadata.

    Returns (procedure_codes, procedure_names)
    """
    if section_slug not in index_metadata:
        return [], []

    section_data = index_metadata[section_slug]
    page_to_procedures = section_data.get("page_to_procedures", {})
    procedures = section_data.get("procedures", [])

    # Normalize page_id for lookup
    normalized_id = normalize_page_id(page_id)

    # Try exact match and normalized match
    codes = page_to_procedures.get(normalized_id, [])
    if not codes:
        # Try with original page_id
        codes = page_to_procedures.get(page_id, [])

    # Get procedure names from codes
    code_to_name = {p["code"]: p["name"] for p in procedures}
    names = [code_to_name.get(code, "") for code in codes if code in code_to_name]

    return codes, names


# =============================================================================
# Skip Logic
# =============================================================================

def should_skip_page(
    page_data: Dict,
    config: Dict,
    existing_output: Optional[Path]
) -> Tuple[bool, str]:
    """
    Check if page should be skipped.

    Skip reasons:
    - Already processed (output file exists)
    - Matches skip_patterns
    - content_type in skip_content_types

    Returns (should_skip, reason)
    """
    gen_config = config.get("generation", {})

    # Check if output already exists
    if existing_output and existing_output.exists():
        return True, "Output file already exists"

    # Check skip patterns
    page_id = page_data.get("page_id", "")
    skip_patterns = gen_config.get("skip_patterns", [])
    for pattern in skip_patterns:
        if fnmatch(page_id, pattern):
            return True, f"Matches skip pattern: {pattern}"

    # Check skip content types
    content_type = page_data.get("content_type", "")
    skip_content_types = gen_config.get("skip_content_types", [])
    if content_type in skip_content_types:
        return True, f"Content type '{content_type}' is in skip list"

    return False, ""


# =============================================================================
# Context Block Building
# =============================================================================

def build_context_block(
    page_data: Dict,
    procedure_codes: List[str],
    procedure_names: List[str]
) -> str:
    """
    Build context block for prompt based on available metadata.
    """
    content_type = page_data.get("content_type", "")

    if procedure_codes and procedure_names:
        lines = ["According to the section index, this page covers:"]
        for code, name in zip(procedure_codes, procedure_names):
            lines.append(f"- {code}: {name}")
        return "\n".join(lines)

    if content_type == "specification":
        return "This page contains specification tables. Focus on extracting specific values with their units."

    return ""


# =============================================================================
# Prompt Templates
# =============================================================================

SYSTEM_PROMPT = """You are generating training data for an AI assistant that will answer questions about the BMW E30 M3 and 320is service manual. The AI will see the same manual page image and must answer questions accurately based on what's visible.

Generate question-answer pairs that a mechanic working on this car would actually ask. The questions should be natural and varied. The answers must be accurate and based only on what's visible on this page.

Output ONLY valid JSON in this exact format:
[
  {
    "question": "Your question here?",
    "answer": "Your answer here.",
    "question_type": "factual"
  }
]

Valid question_type values: factual, procedural, visual, inspection, tool, safety, navigation, wiring, connector, component, diagnostic, troubleshooting, signal, parameter, operation"""

PROMPT_TEMPLATES = {
    ("service_manual", "procedure"): """[IMAGE]

This is page {page_id} from the {section_name} section (Section {section_id}) of the BMW E30 M3 Service Manual.

{context_block}

Generate {num_questions} question-answer pairs for this page. Include a mix of:

1. **Factual questions** — Specific values, part names, specifications visible on the page
2. **Procedural questions** — How to perform tasks, what steps to follow, what sequence
3. **Visual questions** — What components are shown in diagrams, what callout numbers indicate
4. **Inspection questions** — What to look for, signs of wear, acceptance criteria
5. **Tool/supply questions** — What tools are needed, what lubricants or supplies are specified
6. **Safety/warning questions** — Important cautions, things to avoid, critical notes

Guidelines:
- Questions should be self-contained (don't say "on this page")
- Answers should be complete but concise (typically 1-3 sentences)
- For diagram callouts, describe what each number indicates
- For specifications, include the value AND units
- If something references another section, mention the cross-reference
- Don't invent information not visible on the page

Return JSON array only.""",

    ("service_manual", "specification"): """[IMAGE]

This is a specification page from the {section_name} section (Section {section_id}) of the BMW E30 M3 Service Manual.

{context_block}

Generate {num_questions} question-answer pairs for this page. Focus on:

1. **Factual questions** — Extract specific values with their units
2. **Comparison questions** — Different specifications for different variants
3. **Tolerance questions** — Acceptable ranges, min/max values
4. **Application questions** — What component uses this specification

Guidelines:
- Always include units in answers (Nm, mm, bar, °C, L)
- Be precise with numeric values
- Note any conditions (cold, warm, new, used)
- Don't invent values not visible on the page

Return JSON array only.""",

    ("service_manual", "troubleshooting"): """[IMAGE]

This is a troubleshooting page from the {section_name} section (Section {section_id}) of the BMW E30 M3 Service Manual.

{context_block}

Generate {num_questions} question-answer pairs for this page. Focus on:

1. **Symptom questions** — What symptoms indicate this problem?
2. **Diagnostic questions** — How to test or verify the cause?
3. **Cause questions** — What causes this symptom?
4. **Correction questions** — How to fix the problem?
5. **Cross-reference questions** — What related procedures are referenced?

Guidelines:
- Match symptoms to causes accurately
- Include specific test procedures when shown
- Reference repair codes if visible
- Don't invent diagnostic steps not on the page

Return JSON array only.""",

    ("service_manual", "diagram"): """[IMAGE]

This is a diagram page from the {section_name} section (Section {section_id}) of the BMW E30 M3 Service Manual.

{context_block}

Generate {num_questions} question-answer pairs for this page. Focus on:

1. **Component identification** — What parts are labeled?
2. **Callout questions** — What does each number reference?
3. **Assembly questions** — How components fit together
4. **Location questions** — Where components are positioned

Guidelines:
- Reference callout numbers when describing components
- Describe spatial relationships between parts
- Don't invent part names not visible

Return JSON array only.""",

    ("electrical_manual", "wiring"): """[IMAGE]

This is page {page_id} from the 1990 BMW M3 Electrical Troubleshooting Manual.

Generate {num_questions} question-answer pairs for this page. Include:

1. **Wiring questions** — Wire colors, routing, connector locations
2. **Connector questions** — Pin assignments, connector identification
3. **Component questions** — What components are shown, their functions
4. **Diagnostic questions** — How to test circuits, expected voltages
5. **Troubleshooting questions** — What to check if circuit doesn't work

Guidelines:
- Be specific about wire colors (e.g., "brown/white" not just "brown")
- Include connector designations (e.g., "X14 pin 3")
- Reference ground points by location when shown
- Don't invent pin assignments not visible on the page

Return JSON array only.""",

    ("electrical_manual", "pinout"): """[IMAGE]

This is a connector pinout page from the 1990 BMW M3 Electrical Troubleshooting Manual.

Generate {num_questions} question-answer pairs for this page. Focus on:

1. **Pin assignment questions** — What is connected to each pin?
2. **Wire color questions** — What color wire goes to which pin?
3. **Connector ID questions** — How is this connector identified?
4. **Function questions** — What is the purpose of this connector?

Guidelines:
- Be exact with pin numbers and wire colors
- Include connector type/size if shown
- Note viewing orientation (component side vs harness side)
- Don't invent assignments not visible

Return JSON array only.""",

    ("ecu_technical", "diagram"): """[IMAGE]

This is page {page_id} from the Bosch Motronic ML 3-1 technical documentation for the BMW E30 M3.

Generate {num_questions} question-answer pairs for this page. Include:

1. **Signal questions** — What signals are shown, their purposes
2. **Parameter questions** — Specific values, thresholds, timing
3. **Component questions** — Sensors, actuators, their functions
4. **Diagnostic questions** — How the ECU detects faults
5. **Operation questions** — How systems interact, control logic

Guidelines:
- Be precise with technical values (voltages, frequencies, timing)
- Reference sensor and actuator names as shown
- Include units for all measurements
- Don't invent values not visible on the page

Return JSON array only."""
}

# Default template for unmatched types
DEFAULT_TEMPLATE = """[IMAGE]

This is page {page_id} from the {section_name} section of the BMW E30 M3 documentation.

{context_block}

Generate {num_questions} question-answer pairs for this page. Include questions about:
- Specific values, specifications, or measurements visible
- Procedures or steps described
- Components or parts shown
- Any warnings or important notes

Guidelines:
- Questions should be self-contained
- Answers should be accurate based on visible content
- Include units for all measurements
- Don't invent information not on the page

Return JSON array only."""


def select_prompt_template(
    source_type: str,
    content_type: str
) -> Tuple[str, str]:
    """
    Select appropriate system and user prompt templates.

    Returns (system_prompt, user_prompt_template)
    """
    key = (source_type, content_type)
    user_template = PROMPT_TEMPLATES.get(key, DEFAULT_TEMPLATE)
    return SYSTEM_PROMPT, user_template


# =============================================================================
# Image Encoding
# =============================================================================

def encode_image_base64(image_path: Path) -> Tuple[str, str]:
    """
    Encode image to base64 for Claude API.

    Returns (base64_data, media_type)
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
# Q&A Response Parsing
# =============================================================================

def parse_qa_response(response_text: str) -> List[Dict]:
    """
    Parse Claude's JSON response into Q&A pairs.

    Handles:
    - Clean JSON array
    - JSON wrapped in markdown code blocks
    - Malformed JSON (best effort)

    Returns list of Q&A dicts with id, question, answer, question_type
    """
    if not response_text or not response_text.strip():
        return []

    text = response_text.strip()

    # Try to extract JSON if wrapped in markdown
    if text.startswith('```'):
        lines = text.split('\n')
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
        text = '\n'.join(json_lines)

    # Try to find JSON array in the text
    if not text.startswith('['):
        # Look for array start
        start_idx = text.find('[')
        if start_idx != -1:
            # Find matching end
            end_idx = text.rfind(']')
            if end_idx != -1:
                text = text[start_idx:end_idx + 1]

    try:
        qa_pairs = json.loads(text)
        if isinstance(qa_pairs, list):
            return qa_pairs
    except json.JSONDecodeError:
        pass

    return []


def assign_qa_ids(qa_pairs: List[Dict], page_id: str) -> List[Dict]:
    """
    Assign unique IDs to Q&A pairs.

    Format: {page_id}-q{nn} (e.g., "21-03-q01")
    """
    result = []
    for i, qa in enumerate(qa_pairs, start=1):
        qa_copy = qa.copy()
        qa_copy["id"] = f"{page_id}-q{i:02d}"
        result.append(qa_copy)
    return result


# =============================================================================
# API Call
# =============================================================================

def generate_qa_for_page(
    image_path: Path,
    page_data: Dict,
    context_block: str,
    num_questions: int,
    api_key: str,
    model: str,
    max_retries: int = 3,
    config: Optional[Dict] = None
) -> Dict:
    """
    Call Claude API to generate Q&A pairs for a single page.

    Returns dict with:
    - qa_pairs: List of Q&A dicts
    - tokens_input: int
    - tokens_output: int
    - error: Optional[str]
    """
    try:
        from anthropic import Anthropic
    except ImportError:
        return {
            "qa_pairs": [],
            "tokens_input": 0,
            "tokens_output": 0,
            "error": "anthropic package not installed"
        }

    # Get preprocessing settings
    gen_config = config.get("generation", {}) if config else {}
    image_config = gen_config.get("image", {})
    max_size_mb = image_config.get("max_size_mb", 5.0)
    max_dim = image_config.get("max_dimension", 4096)

    # Preprocess image
    try:
        image_bytes, media_type = preprocess_image_for_api(
            image_path, max_size_mb=max_size_mb, max_dim=max_dim
        )
        base64_data = base64.standard_b64encode(image_bytes).decode('utf-8')
    except Exception as e:
        # Fall back to direct encoding
        try:
            base64_data, media_type = encode_image_base64(image_path)
        except Exception as e2:
            return {
                "qa_pairs": [],
                "tokens_input": 0,
                "tokens_output": 0,
                "error": f"Image read error: {str(e2)}"
            }

    # Select prompts
    source_type = page_data.get("source_type", "service_manual")
    content_type = page_data.get("content_type", "procedure")
    system_prompt, user_template = select_prompt_template(source_type, content_type)

    # Format user prompt
    user_prompt = user_template.format(
        page_id=page_data.get("page_id", "unknown"),
        section_id=page_data.get("section_id", ""),
        section_name=page_data.get("section_name", ""),
        context_block=context_block,
        num_questions=num_questions
    )

    client = Anthropic(api_key=api_key)
    max_output_tokens = gen_config.get("max_output_tokens", 4096)
    retry_delay = gen_config.get("retry_delay_seconds", 2)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_output_tokens,
                system=system_prompt,
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
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            )

            # Parse response
            response_text = response.content[0].text if response.content else ""
            qa_pairs = parse_qa_response(response_text)

            return {
                "qa_pairs": qa_pairs,
                "tokens_input": response.usage.input_tokens,
                "tokens_output": response.usage.output_tokens,
                "error": None
            }

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))
                continue
            return {
                "qa_pairs": [],
                "tokens_input": 0,
                "tokens_output": 0,
                "error": f"API error: {str(e)}"
            }


# =============================================================================
# Output Writing
# =============================================================================

def write_qa_output(
    page_id: str,
    page_data: Dict,
    qa_pairs: List[Dict],
    generation_metadata: Dict,
    output_dir: Path
) -> Path:
    """
    Write Q&A output JSON for a single page.

    Returns path to written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "page_id": page_id,
        "image_path": page_data.get("image_path"),
        "section_id": page_data.get("section_id"),
        "section_name": page_data.get("section_name"),
        "source_type": page_data.get("source_type"),
        "content_type": page_data.get("content_type"),
        "procedures_covered": generation_metadata.get("procedure_codes", []),
        "procedures_names": generation_metadata.get("procedure_names", []),
        "generation": {
            "model": generation_metadata.get("model"),
            "timestamp": generation_metadata.get("timestamp"),
            "prompt_template": generation_metadata.get("prompt_template"),
            "tokens_input": generation_metadata.get("tokens_input", 0),
            "tokens_output": generation_metadata.get("tokens_output", 0)
        },
        "qa_pairs": qa_pairs
    }

    # Sanitize page_id for filename
    safe_page_id = re.sub(r'[^\w\-]', '_', page_id)
    output_path = output_dir / f"{safe_page_id}.json"

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return output_path


def write_progress_log(
    progress_entries: List[Dict],
    log_path: Path
):
    """
    Write/append progress log entries.

    Columns: timestamp, page_id, status, qa_count, tokens_input, tokens_output, cost_usd
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "page_id", "status", "qa_count", "tokens_input", "tokens_output", "cost_usd"]

    file_exists = log_path.exists()

    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for entry in progress_entries:
            writer.writerow(entry)


def write_error_log(
    error_entries: List[Dict],
    log_path: Path
):
    """
    Write/append error log entries.

    Columns: timestamp, page_id, error_type, error_message
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "page_id", "error_type", "error_message"]

    file_exists = log_path.exists()

    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for entry in error_entries:
            writer.writerow(entry)


# =============================================================================
# Main Processing
# =============================================================================

def process_all_pages(
    classification_data: List[Dict],
    index_metadata: Dict[str, Dict],
    config: Dict,
    output_dir: Path,
    progress_log_path: Path,
    error_log_path: Path,
    limit: Optional[int] = None,
    start_from: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False
) -> Dict:
    """
    Main processing loop for all pages.

    Returns summary dict with:
    - total_pages: int
    - processed: int
    - skipped: int
    - failed: int
    - total_qa_pairs: int
    - total_tokens: int
    - total_cost_usd: float
    - budget_exceeded: bool
    - dry_run: bool
    """
    gen_config = config.get("generation", {})

    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        api_key = config.get("api", {}).get("api_key")

    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    model = gen_config["model"]
    rate_limit_delay = gen_config.get("rate_limit_delay_seconds", 1.0)
    questions_per_page = gen_config.get("questions_per_page", {})
    default_questions = 8
    validation_config = gen_config.get("validation", {})
    strict_mode = validation_config.get("strict_mode", False)

    # Budget settings
    max_cost_per_page = gen_config.get("max_cost_per_page_usd", 0.50)
    daily_budget = gen_config.get("daily_budget_usd", 50.0)
    total_cost_incurred = 0.0
    budget_exceeded = False

    # Token pricing (per 1K tokens)
    input_price_per_1k = gen_config.get("input_token_price_per_1k", 0.003)
    output_price_per_1k = gen_config.get("output_token_price_per_1k", 0.015)

    # Apply limit
    if limit:
        classification_data = classification_data[:limit]

    # Find start point
    start_index = 0
    if start_from:
        for i, record in enumerate(classification_data):
            if record.get("page_id") == start_from:
                start_index = i
                break

    total_pages = len(classification_data)
    processed = 0
    skipped = 0
    failed = 0
    total_qa_pairs = 0
    total_tokens_input = 0
    total_tokens_output = 0

    progress_entries = []
    error_entries = []

    for i, page_data in enumerate(classification_data[start_index:], start=start_index):
        page_id = page_data.get("page_id", f"page_{i}")
        image_path = Path(page_data.get("image_path", ""))

        # Safe page_id for filename
        safe_page_id = re.sub(r'[^\w\-]', '_', page_id)
        output_path = output_dir / f"{safe_page_id}.json"

        # Check if should skip
        existing_output = output_path if not force else None
        should_skip, skip_reason = should_skip_page(page_data, config, existing_output)

        if should_skip:
            logger.debug(f"Skipping {page_id}: {skip_reason}")
            skipped += 1
            continue

        # Dry-run mode: log what would be processed without making API calls
        if dry_run:
            content_type = page_data.get("content_type", "unknown")
            source_type = page_data.get("source_type", "unknown")
            logger.info(f"[DRY-RUN] Would process: {page_id} ({source_type}/{content_type})")
            processed += 1
            continue

        # Validate image exists
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            error_entries.append({
                "timestamp": datetime.now().isoformat(),
                "page_id": page_id,
                "error_type": "file_not_found",
                "error_message": f"Image file does not exist: {image_path}"
            })
            failed += 1
            continue

        # Get procedures for context
        section_slug = derive_section_slug(
            page_data.get("section_id", ""),
            page_data.get("section_name", "")
        )
        procedure_codes, procedure_names = get_procedures_for_page(
            page_id, section_slug, index_metadata
        )

        # Build context block
        context_block = build_context_block(page_data, procedure_codes, procedure_names)

        # Determine number of questions
        content_type = page_data.get("content_type", "text")
        num_questions = questions_per_page.get(content_type, default_questions)

        # Generate Q&A
        logger.info(f"Processing {page_id} ({content_type})")
        result = generate_qa_for_page(
            image_path=image_path,
            page_data=page_data,
            context_block=context_block,
            num_questions=num_questions,
            api_key=api_key,
            model=model,
            max_retries=gen_config.get("max_retries", 3),
            config=config
        )

        if result.get("error"):
            logger.error(f"Error processing {page_id}: {result['error']}")
            error_entries.append({
                "timestamp": datetime.now().isoformat(),
                "page_id": page_id,
                "error_type": "api_error",
                "error_message": result["error"]
            })
            failed += 1
            continue

        # Assign IDs and validate
        qa_pairs = assign_qa_ids(result["qa_pairs"], page_id)
        valid_pairs, rejected_pairs = filter_and_validate_qa_pairs(
            qa_pairs, page_data, strict=strict_mode
        )

        # Write output
        generation_metadata = {
            "model": model,
            "timestamp": datetime.now().isoformat(),
            "prompt_template": f"{page_data.get('source_type', 'unknown')}/{content_type}",
            "tokens_input": result["tokens_input"],
            "tokens_output": result["tokens_output"],
            "procedure_codes": procedure_codes,
            "procedure_names": procedure_names
        }

        write_qa_output(page_id, page_data, valid_pairs, generation_metadata, output_dir)

        # Calculate cost (approximate)
        input_cost = result["tokens_input"] * input_price_per_1k / 1000
        output_cost = result["tokens_output"] * output_price_per_1k / 1000
        total_cost = input_cost + output_cost
        total_cost_incurred += total_cost

        # Check per-page cost
        if total_cost > max_cost_per_page:
            logger.warning(f"Page {page_id} cost ${total_cost:.4f} exceeds max ${max_cost_per_page}")

        # Check daily budget
        if total_cost_incurred >= daily_budget:
            logger.error(f"Daily budget ${daily_budget:.2f} reached (spent ${total_cost_incurred:.2f}). Stopping.")
            error_entries.append({
                "timestamp": datetime.now().isoformat(),
                "page_id": page_id,
                "error_type": "budget_exceeded",
                "error_message": f"Daily budget ${daily_budget:.2f} exceeded (total: ${total_cost_incurred:.2f})"
            })
            budget_exceeded = True
            # Still count this page as processed since we did process it
            processed += 1
            total_qa_pairs += len(valid_pairs)
            total_tokens_input += result["tokens_input"]
            total_tokens_output += result["tokens_output"]
            break  # Exit the processing loop

        # Log progress
        progress_entries.append({
            "timestamp": datetime.now().isoformat(),
            "page_id": page_id,
            "status": "success",
            "qa_count": len(valid_pairs),
            "tokens_input": result["tokens_input"],
            "tokens_output": result["tokens_output"],
            "cost_usd": f"{total_cost:.4f}"
        })

        processed += 1
        total_qa_pairs += len(valid_pairs)
        total_tokens_input += result["tokens_input"]
        total_tokens_output += result["tokens_output"]

        # Log rejected pairs if any
        if rejected_pairs:
            logger.debug(f"Rejected {len(rejected_pairs)} Q&A pairs for {page_id}")

        # Progress logging
        if processed % 10 == 0:
            logger.info(f"Progress: {processed}/{total_pages - skipped} processed, {total_qa_pairs} Q&A pairs generated")

        # Write logs periodically
        if processed % 50 == 0:
            write_progress_log(progress_entries, progress_log_path)
            progress_entries = []
            if error_entries:
                write_error_log(error_entries, error_log_path)
                error_entries = []

        # Rate limiting
        time.sleep(rate_limit_delay)

    # Write remaining logs
    if progress_entries:
        write_progress_log(progress_entries, progress_log_path)
    if error_entries:
        write_error_log(error_entries, error_log_path)

    return {
        "total_pages": total_pages,
        "processed": processed,
        "skipped": skipped,
        "failed": failed,
        "total_qa_pairs": total_qa_pairs,
        "total_tokens_input": total_tokens_input,
        "total_tokens_output": total_tokens_output,
        "total_cost_usd": total_cost_incurred,
        "budget_exceeded": budget_exceeded,
        "dry_run": dry_run
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Stage 4a: Generate Q&A pairs from service manual images using Claude Vision API',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/04a_generate_qa_images.py \\
      --classified work/classified/pages.csv \\
      --indices work/indices \\
      --output work/qa_raw \\
      --config config.yaml

  # With limit for testing
  python scripts/04a_generate_qa_images.py \\
      --classified work/classified/pages.csv \\
      --indices work/indices \\
      --output work/qa_raw \\
      --config config.yaml \\
      --limit 10

  # Resume from specific page
  python scripts/04a_generate_qa_images.py \\
      --classified work/classified/pages.csv \\
      --indices work/indices \\
      --output work/qa_raw \\
      --config config.yaml \\
      --start-from 21-05
        """
    )

    parser.add_argument(
        '--classified',
        type=Path,
        required=True,
        help='Path to classification CSV from Stage 3'
    )

    parser.add_argument(
        '--indices',
        type=Path,
        default=None,
        help='Path to indices directory (auto-detected if not specified)'
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
        '--data-src',
        type=Path,
        default=None,
        help='Base directory for resolving image paths (if paths in CSV are relative)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit processing to first N pages'
    )

    parser.add_argument(
        '--start-from',
        type=str,
        default=None,
        help='Start processing from this page_id'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-processing even if output exists'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without making API calls'
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
        logger.info("Starting Stage 4a: Image Q&A Generation")

        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")

        # Load classification data
        classification_data = load_classification_data(args.classified)
        logger.info(f"Loaded {len(classification_data)} classification records")

        # Resolve image paths relative to data_src if provided
        if args.data_src:
            data_src_str = str(args.data_src)
            for record in classification_data:
                img_path = record.get("image_path", "")
                # Avoid double-prefixing if path already starts with data_src
                if not Path(img_path).is_absolute() and not img_path.startswith(data_src_str):
                    record["image_path"] = str(args.data_src / img_path)
            logger.info(f"Resolved image paths relative to {args.data_src}")

        # Find indices directory
        work_dir = args.classified.parent.parent  # work/classified/pages.csv -> work/
        if args.indices:
            indices_dir = args.indices
        else:
            try:
                indices_dir = find_indices_dir(work_dir)
            except FileNotFoundError:
                logger.warning("No indices directory found; proceeding without context")
                indices_dir = None

        # Load index metadata
        index_metadata = {}
        if indices_dir:
            index_metadata = load_index_metadata(indices_dir)
            logger.info(f"Loaded {len(index_metadata)} section indices from {indices_dir}")

        # Create output directory
        args.output.mkdir(parents=True, exist_ok=True)

        # Set up log paths
        log_dir = work_dir / "logs"
        progress_log_path = log_dir / "generation_progress.csv"
        error_log_path = log_dir / "generation_errors.csv"

        # Process pages
        summary = process_all_pages(
            classification_data=classification_data,
            index_metadata=index_metadata,
            config=config,
            output_dir=args.output,
            progress_log_path=progress_log_path,
            error_log_path=error_log_path,
            limit=args.limit,
            start_from=args.start_from,
            force=args.force,
            dry_run=args.dry_run
        )

        # Print summary
        logger.info("")
        logger.info("=" * 60)
        if summary.get("dry_run"):
            logger.info("STAGE 4a: Q&A GENERATION SUMMARY (DRY RUN)")
            logger.info("*** No API calls were made ***")
        else:
            logger.info("STAGE 4a: Q&A GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total pages: {summary['total_pages']}")
        logger.info(f"Processed: {summary['processed']}")
        logger.info(f"Skipped: {summary['skipped']}")
        logger.info(f"Failed: {summary['failed']}")
        if not summary.get("dry_run"):
            logger.info(f"Total Q&A pairs: {summary['total_qa_pairs']}")
            logger.info(f"Total tokens (input): {summary['total_tokens_input']}")
            logger.info(f"Total tokens (output): {summary['total_tokens_output']}")
            logger.info(f"Total cost: ${summary.get('total_cost_usd', 0):.4f}")
            if summary.get("budget_exceeded"):
                logger.warning("*** BUDGET LIMIT REACHED - Processing stopped early ***")
        logger.info("=" * 60)

        logger.info("Stage 4a completed successfully")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
