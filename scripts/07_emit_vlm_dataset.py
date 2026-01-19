#!/usr/bin/env python3
"""
07_emit_vlm_dataset.py - Convert Q&A pairs to VLM training format

Stage 6 of the data pipeline: Transform filtered and deduplicated Q&A pairs
into the final VLM training format with train/val split.

Usage:
    python scripts/07_emit_vlm_dataset.py \
        --qa work/qa_unique \
        --data-src data_src \
        --output training_data \
        --report work/logs/emit_report.md \
        --config config.yaml
"""

import argparse
import json
import logging
import os
import random
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
class EmitConfig:
    """Configuration for VLM dataset emission."""
    train_split: float = 0.90
    random_seed: int = 42
    image_copy_mode: str = "symlink"  # symlink, copy, relative
    image_output_dir: str = "images"
    normalize_image_names: bool = True
    stratify_by: Optional[str] = "section_id"  # section_id, source_type, None
    min_stratum_size: int = 10


def load_emit_config(config_path: Path) -> EmitConfig:
    """
    Load emit configuration from YAML file.

    Uses defaults for any missing values.

    Args:
        config_path: Path to config YAML file

    Returns:
        EmitConfig with loaded values

    Raises:
        ValueError: If train_split is invalid or image_copy_mode is unknown
    """
    config = EmitConfig()

    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}

        output_config = yaml_config.get("output", {})

        if "train_split" in output_config:
            config.train_split = float(output_config["train_split"])
        if "random_seed" in output_config:
            config.random_seed = int(output_config["random_seed"])
        if "image_copy_mode" in output_config:
            config.image_copy_mode = output_config["image_copy_mode"]
        if "image_output_dir" in output_config:
            config.image_output_dir = output_config["image_output_dir"]
        if "normalize_image_names" in output_config:
            config.normalize_image_names = output_config["normalize_image_names"]
        if "stratify_by" in output_config:
            val = output_config["stratify_by"]
            config.stratify_by = val if val != "none" else None
        if "min_stratum_size" in output_config:
            config.min_stratum_size = int(output_config["min_stratum_size"])

    # Validate
    if config.train_split < 0 or config.train_split > 1:
        raise ValueError(f"train_split must be between 0 and 1, got {config.train_split}")

    valid_modes = {"symlink", "copy", "relative"}
    if config.image_copy_mode not in valid_modes:
        raise ValueError(f"image_copy_mode must be one of {valid_modes}, got {config.image_copy_mode}")

    return config


# ============================================================================
# Q&A Document Loading
# ============================================================================

def load_qa_documents(input_dir: Path) -> List[Dict]:
    """
    Load all Q&A JSON files from input directory.

    Args:
        input_dir: Path to directory containing *.json files

    Returns:
        List of parsed Q&A documents (one per file)

    Raises:
        FileNotFoundError: If input_dir doesn't exist
        ValueError: If no JSON files found
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        raise ValueError(f"No JSON files found in {input_dir}")

    docs = []
    for json_file in json_files:
        try:
            with open(json_file) as f:
                doc = json.load(f)
                docs.append(doc)
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping invalid JSON file {json_file}: {e}")
            continue

    return docs


# ============================================================================
# Q&A Flattening
# ============================================================================

def flatten_qa_pairs(qa_docs: List[Dict]) -> List[Dict]:
    """
    Flatten all Q&A pairs from documents into a single list.

    Each output record contains:
    - question: str
    - answer: str
    - image_path: Optional[str] (original path from source doc)
    - page_id: str
    - section_id: str
    - section_name: str
    - source_type: str
    - content_type: str
    - question_type: str
    - qa_id: str

    Args:
        qa_docs: List of Q&A documents from Stage 5

    Returns:
        List of flattened Q&A records
    """
    records = []

    for doc in qa_docs:
        page_id = doc.get("page_id", "")
        image_path = doc.get("image_path")
        section_id = doc.get("section_id", "")
        section_name = doc.get("section_name", "")
        source_type = doc.get("source_type", "")
        content_type = doc.get("content_type", "")

        for qa in doc.get("qa_pairs", []):
            record = {
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "image_path": image_path,
                "page_id": page_id,
                "section_id": section_id,
                "section_name": section_name,
                "source_type": source_type,
                "content_type": content_type,
                "question_type": qa.get("question_type", ""),
                "qa_id": qa.get("id", "")
            }
            records.append(record)

    return records


# ============================================================================
# Image Filename Normalization
# ============================================================================

def normalize_image_filename(image_path: str) -> str:
    """
    Normalize image filename for output.

    - Replaces spaces with underscores
    - Replaces special characters with underscores
    - Preserves extension
    - Handles paths with directories (flattens to single filename)

    Args:
        image_path: Original image path (may include directories)

    Returns:
        Normalized filename (no directories)

    Examples:
        "21 - Clutch/21-03.jpg" -> "21_-_Clutch_21-03.jpg"
        "Bosch Motronic ML 3-1/page001.jpg" -> "Bosch_Motronic_ML_3-1_page001.jpg"
    """
    # Get just the path components and flatten
    parts = Path(image_path).parts

    # Join all parts with underscore, replacing problematic characters
    flattened = "_".join(parts)

    # Replace spaces and special characters
    normalized = re.sub(r'[\s/\\:*?"<>|]+', '_', flattened)

    # Clean up multiple underscores
    normalized = re.sub(r'_+', '_', normalized)

    # Remove leading/trailing underscores
    normalized = normalized.strip('_')

    return normalized


# ============================================================================
# VLM Record Formatting
# ============================================================================

def format_vlm_record(qa_record: Dict, image_output_path: Optional[str]) -> Dict:
    """
    Format a Q&A record for VLM training output.

    Args:
        qa_record: Flattened Q&A record
        image_output_path: Relative path to image in output dir, or None for text-only

    Returns:
        Dict with structure:
        {
            "image": "images/21-03_clutch.jpg" or null,
            "conversations": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "metadata": {...}
        }
    """
    return {
        "image": image_output_path,
        "conversations": [
            {"role": "user", "content": qa_record["question"]},
            {"role": "assistant", "content": qa_record["answer"]}
        ],
        "metadata": {
            "page_id": qa_record["page_id"],
            "section_id": qa_record["section_id"],
            "section_name": qa_record["section_name"],
            "source_type": qa_record["source_type"],
            "content_type": qa_record["content_type"],
            "question_type": qa_record["question_type"],
            "qa_id": qa_record["qa_id"]
        }
    }


# ============================================================================
# Train/Val Split
# ============================================================================

def stratified_split(
    records: List[Dict],
    config: EmitConfig
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split records into train/val sets with stratification.

    Args:
        records: List of flattened Q&A records
        config: Emit configuration with split ratio and stratify_by

    Returns:
        (train_records, val_records)

    Notes:
        - Uses config.random_seed for reproducibility
        - If stratify_by is set, maintains distribution across strata
        - Falls back to random split if stratum size < min_stratum_size
    """
    random.seed(config.random_seed)

    if not records:
        return [], []

    if len(records) == 1:
        # Single record goes to train or val based on random
        if random.random() < config.train_split:
            return records.copy(), []
        else:
            return [], records.copy()

    if config.stratify_by is None:
        # Simple random split
        shuffled = records.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * config.train_split)
        return shuffled[:split_idx], shuffled[split_idx:]

    # Stratified split
    # Group records by stratum
    strata = {}
    for rec in records:
        key = rec.get(config.stratify_by, "unknown")
        if key not in strata:
            strata[key] = []
        strata[key].append(rec)

    train = []
    val = []

    for stratum_key, stratum_records in strata.items():
        if len(stratum_records) < config.min_stratum_size:
            # Too small for stratification, add to pool for random assignment
            shuffled = stratum_records.copy()
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * config.train_split)
            train.extend(shuffled[:split_idx])
            val.extend(shuffled[split_idx:])
        else:
            # Stratified split within this stratum
            shuffled = stratum_records.copy()
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * config.train_split)
            train.extend(shuffled[:split_idx])
            val.extend(shuffled[split_idx:])

    return train, val


# ============================================================================
# Image Copying/Linking
# ============================================================================

def copy_or_link_image(src_path: Path, dst_path: Path, mode: str) -> bool:
    """
    Copy or symlink an image file.

    Args:
        src_path: Source image path
        dst_path: Destination path
        mode: "copy", "symlink", or "relative"

    Returns:
        True if successful, False if source doesn't exist

    Notes:
        - Creates parent directories if needed
        - "relative" mode creates relative symlinks
        - Skips if destination already exists and is valid
    """
    if not src_path.exists():
        logger.warning(f"Source image not found: {src_path}")
        return False

    # Create parent directories
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if destination already exists
    if dst_path.exists():
        return True

    try:
        if mode == "copy":
            shutil.copy2(src_path, dst_path)
        elif mode == "symlink":
            dst_path.symlink_to(src_path.resolve())
        elif mode == "relative":
            rel_path = os.path.relpath(src_path, dst_path.parent)
            dst_path.symlink_to(rel_path)
        return True
    except Exception as e:
        logger.warning(f"Failed to {mode} image {src_path} -> {dst_path}: {e}")
        return False


# ============================================================================
# JSONL Writing
# ============================================================================

def write_jsonl(records: List[Dict], output_path: Path) -> int:
    """
    Write records to JSONL file.

    Args:
        records: List of VLM-formatted records
        output_path: Path to output .jsonl file

    Returns:
        Number of records written

    Notes:
        - One JSON object per line
        - UTF-8 encoding
        - Creates parent directories if needed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


# ============================================================================
# Image Collection
# ============================================================================

def collect_unique_images(records: List[Dict]) -> Set[str]:
    """
    Collect unique image paths from records.

    Args:
        records: List of flattened Q&A records

    Returns:
        Set of unique image_path values (excluding None)
    """
    unique = set()
    for rec in records:
        img_path = rec.get("image_path")
        if img_path is not None:
            unique.add(img_path)
    return unique


# ============================================================================
# Report Generation
# ============================================================================

def generate_emit_report(stats: Dict, report_path: Path) -> None:
    """
    Generate Markdown summary report.

    Stats include:
    - Total Q&A pairs emitted (train/val)
    - Unique images copied/linked
    - Distribution by section and source_type
    - Any warnings (missing images, etc.)
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    lines = [
        "# VLM Dataset Emit Report",
        "",
        f"**Generated**: {timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Total Q&A Pairs | {stats['total_qa']:,} |",
        f"| Training Set | {stats['train_qa']:,} ({100*stats['train_qa']/max(stats['total_qa'],1):.1f}%) |",
        f"| Validation Set | {stats['val_qa']:,} ({100*stats['val_qa']/max(stats['total_qa'],1):.1f}%) |",
        f"| Unique Images | {stats['unique_images']:,} |",
        f"| Text-only Q&A | {stats['text_only_qa']:,} |",
        "",
        "## Images",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Images Copied/Linked | {stats['images_copied']:,} |",
        f"| Missing Images | {stats['images_missing']:,} |",
        f"| Mode | {stats.get('image_copy_mode', 'unknown')} |",
        "",
    ]

    # Section distribution
    section_dist = stats.get("section_distribution", {})
    if section_dist:
        lines.append("## Distribution by Section")
        lines.append("")
        lines.append("| Section | Train | Val | Total |")
        lines.append("|---------|-------|-----|-------|")

        # Aggregate by section
        sections = {}
        for (section, split), count in section_dist.items():
            if section not in sections:
                sections[section] = {"train": 0, "val": 0}
            sections[section][split] = count

        for section in sorted(sections.keys()):
            train = sections[section]["train"]
            val = sections[section]["val"]
            total = train + val
            lines.append(f"| {section} | {train:,} | {val:,} | {total:,} |")

        lines.append("")

    # Source distribution
    source_dist = stats.get("source_distribution", {})
    if source_dist:
        lines.append("## Distribution by Source Type")
        lines.append("")
        lines.append("| Source Type | Train | Val | Total |")
        lines.append("|-------------|-------|-----|-------|")

        # Aggregate by source
        sources = {}
        for (source, split), count in source_dist.items():
            if source not in sources:
                sources[source] = {"train": 0, "val": 0}
            sources[source][split] = count

        for source in sorted(sources.keys()):
            train = sources[source]["train"]
            val = sources[source]["val"]
            total = train + val
            lines.append(f"| {source} | {train:,} | {val:,} | {total:,} |")

        lines.append("")

    # Warnings
    warnings = stats.get("warnings", [])
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("None")
    lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Report written to {report_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Emit VLM training dataset from Q&A pairs"
    )
    parser.add_argument(
        "--qa", type=Path, required=True,
        help="Input directory with qa_unique/*.json"
    )
    parser.add_argument(
        "--data-src", type=Path, required=True,
        help="Source images directory"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for training_data/"
    )
    parser.add_argument(
        "--report", type=Path, required=True,
        help="Markdown summary report path"
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Config YAML file"
    )
    parser.add_argument(
        "--train-split", type=float,
        help="Override train split ratio"
    )
    parser.add_argument(
        "--seed", type=int,
        help="Override random seed"
    )
    parser.add_argument(
        "--copy-images", action="store_true",
        help="Force copy instead of symlink"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be emitted without writing"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log each Q&A pair"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config with overrides
    config = load_emit_config(args.config)
    if args.train_split is not None:
        config.train_split = args.train_split
    if args.seed is not None:
        config.random_seed = args.seed
    if args.copy_images:
        config.image_copy_mode = "copy"

    logger.info(f"Loading Q&A documents from {args.qa}")

    # Load and flatten Q&A documents
    qa_docs = load_qa_documents(args.qa)
    logger.info(f"Loaded {len(qa_docs)} Q&A documents")

    all_records = flatten_qa_pairs(qa_docs)
    logger.info(f"Flattened to {len(all_records)} Q&A pairs")

    # Split into train/val
    train_records, val_records = stratified_split(all_records, config)
    logger.info(f"Split: {len(train_records)} train, {len(val_records)} val")

    # Collect unique images
    unique_images = collect_unique_images(all_records)
    logger.info(f"Found {len(unique_images)} unique images")

    # Track stats
    stats = {
        "total_qa": len(all_records),
        "train_qa": len(train_records),
        "val_qa": len(val_records),
        "unique_images": len(unique_images),
        "text_only_qa": len([r for r in all_records if r.get("image_path") is None]),
        "images_copied": 0,
        "images_missing": 0,
        "image_copy_mode": config.image_copy_mode,
        "section_distribution": Counter(),
        "source_distribution": Counter(),
        "warnings": []
    }

    # Update distributions
    for r in train_records:
        stats["section_distribution"][(r["section_id"], "train")] += 1
        stats["source_distribution"][(r["source_type"], "train")] += 1
    for r in val_records:
        stats["section_distribution"][(r["section_id"], "val")] += 1
        stats["source_distribution"][(r["source_type"], "val")] += 1

    if args.dry_run:
        logger.info(f"[DRY RUN] Would emit {len(train_records)} train + {len(val_records)} val Q&A pairs")
        logger.info(f"[DRY RUN] Would copy/link {len(unique_images)} images")
        generate_emit_report(stats, args.report)
        return

    # Create output directories
    output_dir = args.output
    images_dir = output_dir / config.image_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy/link images
    image_mapping = {}  # original path -> output path
    for img_path in unique_images:
        # Resolve source path
        if Path(img_path).is_absolute():
            src = Path(img_path)
        else:
            # Try relative to data_src parent (for paths like "data_src/...")
            if img_path.startswith("data_src/"):
                src = args.data_src.parent / img_path
            else:
                src = args.data_src / img_path

        # Determine destination filename
        if config.normalize_image_names:
            dst_name = normalize_image_filename(img_path)
        else:
            dst_name = Path(img_path).name

        dst = images_dir / dst_name

        if copy_or_link_image(src, dst, config.image_copy_mode):
            stats["images_copied"] += 1
            image_mapping[img_path] = f"{config.image_output_dir}/{dst_name}"
        else:
            stats["images_missing"] += 1
            stats["warnings"].append(f"Missing image: {img_path}")

    # Format records for output
    def format_record(r):
        output_img = image_mapping.get(r.get("image_path")) if r.get("image_path") else None
        return format_vlm_record(r, output_img)

    train_formatted = [format_record(r) for r in train_records]
    val_formatted = [format_record(r) for r in val_records]

    # Write JSONL files
    train_path = output_dir / "vlm_train.jsonl"
    val_path = output_dir / "vlm_val.jsonl"

    write_jsonl(train_formatted, train_path)
    write_jsonl(val_formatted, val_path)

    logger.info(f"Wrote {len(train_formatted)} records to {train_path}")
    logger.info(f"Wrote {len(val_formatted)} records to {val_path}")

    # Generate report
    generate_emit_report(stats, args.report)

    logger.info(f"Emitted {len(train_records)} train + {len(val_records)} val Q&A pairs")
    logger.info(f"Copied/linked {stats['images_copied']} images ({stats['images_missing']} missing)")


if __name__ == "__main__":
    main()
