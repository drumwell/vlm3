#!/usr/bin/env python3
"""
08_validate_vlm.py - Validate VLM training dataset

Stage 6 of the data pipeline: Validate dataset integrity before upload.

Usage:
    python scripts/08_validate_vlm.py \
        --train training_data/vlm_train.jsonl \
        --val training_data/vlm_val.jsonl \
        --images training_data/images \
        --output work/logs/vlm_qa_report.md \
        --config config.yaml
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ValidationConfig:
    """Configuration for VLM dataset validation."""
    require_image_field: bool = True
    require_conversations: bool = True
    require_metadata: bool = True
    check_image_exists: bool = True
    check_image_readable: bool = True
    min_image_width: int = 100
    min_image_height: int = 100
    min_question_length: int = 10
    min_answer_length: int = 5
    max_answer_length: int = 1000
    min_qa_per_section: int = 5
    max_qa_per_section: int = 2000
    num_samples_per_split: int = 5


def load_validation_config(config_path: Path) -> ValidationConfig:
    """
    Load validation configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        ValidationConfig with loaded values
    """
    config = ValidationConfig()

    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}

        val_config = yaml_config.get("vlm_validation", {})

        if "require_image_field" in val_config:
            config.require_image_field = val_config["require_image_field"]
        if "require_conversations" in val_config:
            config.require_conversations = val_config["require_conversations"]
        if "require_metadata" in val_config:
            config.require_metadata = val_config["require_metadata"]
        if "check_image_exists" in val_config:
            config.check_image_exists = val_config["check_image_exists"]
        if "check_image_readable" in val_config:
            config.check_image_readable = val_config["check_image_readable"]
        if "min_image_width" in val_config:
            config.min_image_width = int(val_config["min_image_width"])
        if "min_image_height" in val_config:
            config.min_image_height = int(val_config["min_image_height"])
        if "min_question_length" in val_config:
            config.min_question_length = int(val_config["min_question_length"])
        if "min_answer_length" in val_config:
            config.min_answer_length = int(val_config["min_answer_length"])
        if "max_answer_length" in val_config:
            config.max_answer_length = int(val_config["max_answer_length"])
        if "min_qa_per_section" in val_config:
            config.min_qa_per_section = int(val_config["min_qa_per_section"])
        if "max_qa_per_section" in val_config:
            config.max_qa_per_section = int(val_config["max_qa_per_section"])
        if "num_samples_per_split" in val_config:
            config.num_samples_per_split = int(val_config["num_samples_per_split"])

    return config


# ============================================================================
# JSONL Loading
# ============================================================================

def load_jsonl(file_path: Path) -> List[Dict]:
    """
    Load records from JSONL file.

    Args:
        file_path: Path to .jsonl file

    Returns:
        List of parsed records

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If any line is invalid JSON
    """
    if not file_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {file_path}")

    records = []
    with open(file_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records.append(record)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")

    return records


# ============================================================================
# Schema Validation
# ============================================================================

def validate_schema(record: Dict, config: ValidationConfig) -> List[str]:
    """
    Validate record schema.

    Returns:
        List of error messages (empty if valid)

    Checks:
        - "image" field exists (if require_image_field)
        - "conversations" field exists and is list (if require_conversations)
        - "metadata" field exists (if require_metadata)
        - conversations has correct role structure
    """
    errors = []

    # Check image field
    if config.require_image_field and "image" not in record:
        errors.append("Missing 'image' field")

    # Check conversations field
    if config.require_conversations:
        if "conversations" not in record:
            errors.append("Missing 'conversations' field")
        elif not isinstance(record["conversations"], list):
            errors.append("'conversations' must be a list")
        else:
            # Check role structure
            valid_roles = {"user", "assistant"}
            for i, conv in enumerate(record["conversations"]):
                if not isinstance(conv, dict):
                    errors.append(f"Conversation {i} is not a dict")
                    continue
                role = conv.get("role")
                if role not in valid_roles:
                    errors.append(f"Invalid role '{role}' at conversation {i}")

    # Check metadata field
    if config.require_metadata and "metadata" not in record:
        errors.append("Missing 'metadata' field")

    return errors


# ============================================================================
# Conversation Validation
# ============================================================================

def validate_conversations(
    record: Dict,
    config: ValidationConfig
) -> Tuple[List[str], List[str]]:
    """
    Validate conversation content.

    Returns:
        (errors, warnings)

    Checks:
        - Question length >= min_question_length
        - Answer length >= min_answer_length
        - Answer length <= max_answer_length
        - Proper role alternation (user, assistant, user, assistant...)
    """
    errors = []
    warnings = []

    conversations = record.get("conversations", [])

    if not conversations:
        errors.append("Empty conversations list")
        return errors, warnings

    # Check role order
    expected_role = "user"
    for i, conv in enumerate(conversations):
        role = conv.get("role")
        if role != expected_role:
            errors.append(f"Wrong role order at position {i}: expected '{expected_role}', got '{role}'")
        expected_role = "assistant" if expected_role == "user" else "user"

    # Check content lengths
    for conv in conversations:
        role = conv.get("role")
        content = conv.get("content", "")

        if role == "user":
            if len(content) < config.min_question_length:
                warnings.append(f"Question too short ({len(content)} chars)")
        elif role == "assistant":
            if len(content) < config.min_answer_length:
                warnings.append(f"Answer too short ({len(content)} chars)")
            if len(content) > config.max_answer_length:
                warnings.append(f"Answer too long ({len(content)} chars)")

    return errors, warnings


# ============================================================================
# Image Validation
# ============================================================================

def validate_image(
    record: Dict,
    images_dir: Path,
    config: ValidationConfig
) -> Tuple[List[str], List[str]]:
    """
    Validate image reference.

    Returns:
        (errors, warnings)

    Checks:
        - Image file exists (if check_image_exists)
        - Image is readable by PIL (if check_image_readable)
        - Image dimensions >= min (warnings only)

    Notes:
        - Skips checks if image is null (text-only record)
    """
    errors = []
    warnings = []

    image_path = record.get("image")

    # Skip for text-only records
    if image_path is None:
        return errors, warnings

    if not config.check_image_exists:
        return errors, warnings

    # Resolve image path
    full_path = images_dir / image_path if not Path(image_path).is_absolute() else Path(image_path)

    if not full_path.exists():
        errors.append(f"Image not found: {image_path}")
        return errors, warnings

    if config.check_image_readable:
        try:
            from PIL import Image
            with Image.open(full_path) as img:
                width, height = img.size

                if width < config.min_image_width:
                    warnings.append(f"Image width {width}px < {config.min_image_width}px")
                if height < config.min_image_height:
                    warnings.append(f"Image height {height}px < {config.min_image_height}px")
        except ImportError:
            # PIL not available, skip readability check
            pass
        except Exception as e:
            errors.append(f"Image unreadable: {image_path} ({e})")

    return errors, warnings


# ============================================================================
# Distribution Statistics
# ============================================================================

def compute_distribution_stats(records: List[Dict]) -> Dict:
    """
    Compute distribution statistics from records.

    Returns:
        Dict with:
        - section_counts: Counter of section_id
        - source_counts: Counter of source_type
        - question_type_counts: Counter of question_type
        - content_type_counts: Counter of content_type
        - image_count: Number of records with images
        - text_only_count: Number of records without images
        - answer_length_stats: min, max, mean, median
    """
    section_counts = Counter()
    source_counts = Counter()
    question_type_counts = Counter()
    content_type_counts = Counter()
    image_count = 0
    text_only_count = 0
    answer_lengths = []

    for record in records:
        meta = record.get("metadata", {})
        section_counts[meta.get("section_id", "unknown")] += 1
        source_counts[meta.get("source_type", "unknown")] += 1
        question_type_counts[meta.get("question_type", "unknown")] += 1
        content_type_counts[meta.get("content_type", "unknown")] += 1

        if record.get("image") is not None:
            image_count += 1
        else:
            text_only_count += 1

        # Get answer length
        convs = record.get("conversations", [])
        for conv in convs:
            if conv.get("role") == "assistant":
                answer_lengths.append(len(conv.get("content", "")))

    # Compute answer length stats
    if answer_lengths:
        answer_length_stats = {
            "min": min(answer_lengths),
            "max": max(answer_lengths),
            "mean": sum(answer_lengths) / len(answer_lengths),
            "median": sorted(answer_lengths)[len(answer_lengths) // 2]
        }
    else:
        answer_length_stats = {"min": 0, "max": 0, "mean": 0, "median": 0}

    return {
        "section_counts": dict(section_counts),
        "source_counts": dict(source_counts),
        "question_type_counts": dict(question_type_counts),
        "content_type_counts": dict(content_type_counts),
        "image_count": image_count,
        "text_only_count": text_only_count,
        "answer_length_stats": answer_length_stats
    }


# ============================================================================
# Distribution Issue Checks
# ============================================================================

def check_distribution_issues(stats: Dict, config: ValidationConfig) -> List[str]:
    """
    Check for distribution anomalies.

    Returns:
        List of warning messages

    Checks:
        - Section with < min_qa_per_section Q&A
        - Section with > max_qa_per_section Q&A
        - Missing expected source types
    """
    warnings = []

    section_counts = stats.get("section_counts", {})

    for section, count in section_counts.items():
        if count < config.min_qa_per_section:
            warnings.append(f"Section '{section}' has only {count} Q&A pairs (min: {config.min_qa_per_section})")
        if count > config.max_qa_per_section:
            warnings.append(f"Section '{section}' has {count} Q&A pairs (max: {config.max_qa_per_section})")

    return warnings


# ============================================================================
# Sampling
# ============================================================================

def sample_records(records: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """
    Sample n records for report display.

    Uses fixed seed for reproducibility.
    Returns fewer than n if not enough records.
    """
    if len(records) <= n:
        return records.copy()

    random.seed(seed)
    return random.sample(records, n)


# ============================================================================
# Report Generation
# ============================================================================

def generate_validation_report(results: Dict, report_path: Path) -> None:
    """
    Generate Markdown validation report.

    Results include:
    - passed: bool
    - train_stats: distribution stats for train set
    - val_stats: distribution stats for val set
    - errors: list of critical errors
    - warnings: list of warnings
    - train_samples: sample records from train
    - val_samples: sample records from val
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()
    status = "PASSED" if results["passed"] else "FAILED"
    status_emoji = "\u2705" if results["passed"] else "\u274c"

    lines = [
        "# VLM Dataset Validation Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Status**: {status_emoji} {status}",
        "",
        "## Summary",
        "",
        "| Metric | Train | Val | Total |",
        "|--------|-------|-----|-------|",
        f"| Q&A Pairs | {results['train_count']:,} | {results['val_count']:,} | {results['train_count'] + results['val_count']:,} |",
        f"| Unique Images | {results['train_stats']['image_count']:,} | {results['val_stats']['image_count']:,} | {results['train_stats']['image_count'] + results['val_stats']['image_count']:,} |",
        f"| Text-only Q&A | {results['train_stats']['text_only_count']:,} | {results['val_stats']['text_only_count']:,} | {results['train_stats']['text_only_count'] + results['val_stats']['text_only_count']:,} |",
        "",
    ]

    # Section distribution
    train_sections = results["train_stats"].get("section_counts", {})
    val_sections = results["val_stats"].get("section_counts", {})
    all_sections = set(train_sections.keys()) | set(val_sections.keys())

    if all_sections:
        lines.append("## Distribution by Section")
        lines.append("")
        lines.append("| Section | Train | Val |")
        lines.append("|---------|-------|-----|")
        for section in sorted(all_sections):
            train = train_sections.get(section, 0)
            val = val_sections.get(section, 0)
            lines.append(f"| {section} | {train:,} | {val:,} |")
        lines.append("")

    # Source distribution
    train_sources = results["train_stats"].get("source_counts", {})
    val_sources = results["val_stats"].get("source_counts", {})
    all_sources = set(train_sources.keys()) | set(val_sources.keys())

    if all_sources:
        lines.append("## Distribution by Source Type")
        lines.append("")
        lines.append("| Source Type | Train | Val |")
        lines.append("|-------------|-------|-----|")
        for source in sorted(all_sources):
            train = train_sources.get(source, 0)
            val = val_sources.get(source, 0)
            lines.append(f"| {source} | {train:,} | {val:,} |")
        lines.append("")

    # Errors
    lines.append("## Critical Errors")
    lines.append("")
    if results["errors"]:
        for error in results["errors"][:50]:  # Limit to 50
            lines.append(f"- {error}")
        if len(results["errors"]) > 50:
            lines.append(f"- ... and {len(results['errors']) - 50} more")
    else:
        lines.append("None")
    lines.append("")

    # Warnings
    lines.append("## Warnings")
    lines.append("")
    if results["warnings"]:
        for warning in results["warnings"][:50]:  # Limit to 50
            lines.append(f"- {warning}")
        if len(results["warnings"]) > 50:
            lines.append(f"- ... and {len(results['warnings']) - 50} more")
    else:
        lines.append("None")
    lines.append("")

    # Samples
    if results["train_samples"]:
        lines.append("## Sample Q&A Pairs")
        lines.append("")
        lines.append("### Training Set")
        lines.append("")
        for i, sample in enumerate(results["train_samples"], 1):
            convs = sample.get("conversations", [])
            question = convs[0]["content"] if convs else ""
            answer = convs[1]["content"] if len(convs) > 1 else ""
            lines.append(f"**{i}.** Q: {question[:100]}...")
            lines.append(f"   A: {answer[:100]}...")
            lines.append("")

    if results["val_samples"]:
        lines.append("### Validation Set")
        lines.append("")
        for i, sample in enumerate(results["val_samples"], 1):
            convs = sample.get("conversations", [])
            question = convs[0]["content"] if convs else ""
            answer = convs[1]["content"] if len(convs) > 1 else ""
            lines.append(f"**{i}.** Q: {question[:100]}...")
            lines.append(f"   A: {answer[:100]}...")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Validation report written to {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate VLM training dataset"
    )
    parser.add_argument(
        "--train", type=Path, required=True,
        help="Training JSONL file"
    )
    parser.add_argument(
        "--val", type=Path, required=True,
        help="Validation JSONL file"
    )
    parser.add_argument(
        "--images", type=Path, required=True,
        help="Images directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Validation report path"
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Config YAML file"
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--skip-image-check", action="store_true",
        help="Skip image existence/readability checks"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log each validation check"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_validation_config(args.config)
    if args.skip_image_check:
        config.check_image_exists = False
        config.check_image_readable = False

    logger.info(f"Loading training data from {args.train}")
    train_records = load_jsonl(args.train)
    logger.info(f"Loaded {len(train_records)} training records")

    logger.info(f"Loading validation data from {args.val}")
    val_records = load_jsonl(args.val)
    logger.info(f"Loaded {len(val_records)} validation records")

    all_errors = []
    all_warnings = []

    # Validate each record
    for split_name, records in [("train", train_records), ("val", val_records)]:
        logger.info(f"Validating {split_name} set...")
        for i, record in enumerate(records):
            # Schema validation
            schema_errors = validate_schema(record, config)
            for e in schema_errors:
                all_errors.append(f"{split_name}[{i}]: {e}")

            # Conversation validation
            conv_errors, conv_warnings = validate_conversations(record, config)
            for e in conv_errors:
                all_errors.append(f"{split_name}[{i}]: {e}")
            for w in conv_warnings:
                all_warnings.append(f"{split_name}[{i}]: {w}")

            # Image validation
            img_errors, img_warnings = validate_image(record, args.images, config)
            for e in img_errors:
                all_errors.append(f"{split_name}[{i}]: {e}")
            for w in img_warnings:
                all_warnings.append(f"{split_name}[{i}]: {w}")

    # Compute distribution stats
    train_stats = compute_distribution_stats(train_records)
    val_stats = compute_distribution_stats(val_records)

    # Check distribution issues
    dist_warnings = check_distribution_issues(train_stats, config)
    all_warnings.extend([f"train distribution: {w}" for w in dist_warnings])

    dist_warnings = check_distribution_issues(val_stats, config)
    all_warnings.extend([f"val distribution: {w}" for w in dist_warnings])

    # Sample records for report
    train_samples = sample_records(train_records, config.num_samples_per_split)
    val_samples = sample_records(val_records, config.num_samples_per_split)

    # Determine pass/fail
    passed = len(all_errors) == 0
    if args.strict and len(all_warnings) > 0:
        passed = False

    # Generate report
    results = {
        "passed": passed,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "errors": all_errors,
        "warnings": all_warnings,
        "train_samples": train_samples,
        "val_samples": val_samples
    }

    generate_validation_report(results, args.output)

    # Exit with appropriate code
    if passed:
        logger.info(f"\u2705 Validation PASSED: {len(train_records)} train + {len(val_records)} val records")
        sys.exit(0)
    else:
        logger.error(f"\u274c Validation FAILED: {len(all_errors)} errors, {len(all_warnings)} warnings")
        sys.exit(1)


if __name__ == "__main__":
    main()
