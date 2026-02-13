#!/usr/bin/env python3
"""
07_emit.py - Emit forum Q&A pairs as VLM training data

Stage 7 of the forum pipeline: converts validated Q&A pairs into the prepared
output format (JSONL) with train/val split.

Usage:
    python pipeline/07_emit.py \
        --input work/qa_validated \
        --output prepared \
        --prefix forum \
        --report work/logs/emit_report.md \
        --config config.yaml
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EmitConfig:
    """Configuration for training data emission."""
    train_split: float = 0.90
    random_seed: int = 42
    prefix: str = "forum"
    stratify_by: Optional[str] = "forum"


def load_emit_config(config_path: Path) -> EmitConfig:
    """Load emit configuration from YAML file."""
    config = EmitConfig()
    if not config_path.exists():
        return config

    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    output = data.get("output", {})
    if "train_split" in output:
        config.train_split = float(output["train_split"])
    if "random_seed" in output:
        config.random_seed = int(output["random_seed"])
    if "prefix" in output:
        config.prefix = output["prefix"]
    if "stratify_by" in output:
        config.stratify_by = output["stratify_by"]

    return config


# ============================================================================
# Record Conversion
# ============================================================================

def qa_to_vlm_record(pair: Dict, doc: Dict) -> Dict:
    """Convert a Q&A pair to VLM training format (text-only, no image field)."""
    return {
        "conversations": [
            {"role": "user", "content": pair["question"]},
            {"role": "assistant", "content": pair["answer"]},
        ],
        "metadata": {
            "source": "forum",
            "source_id": pair.get("id", ""),
            "thread_id": doc.get("thread_id", ""),
            "forum": doc.get("forum", ""),
            "content_type": doc.get("content_type", ""),
            "question_type": pair.get("question_type", ""),
            "quality_score": doc.get("quality_score", 0),
            "factual_confidence": pair.get("factual_confidence", "unverified"),
        },
    }


# ============================================================================
# Train/Val Split
# ============================================================================

def stratified_split(
    records: List[Dict],
    train_ratio: float,
    seed: int,
    stratify_key: Optional[str],
) -> tuple[List[Dict], List[Dict]]:
    """Split records into train/val with optional stratification."""
    rng = random.Random(seed)
    records = list(records)  # Copy to avoid mutating input

    if not stratify_key:
        rng.shuffle(records)
        split_idx = int(len(records) * train_ratio)
        return records[:split_idx], records[split_idx:]

    # Group by stratum
    strata = defaultdict(list)
    for record in records:
        key = record.get("metadata", {}).get(stratify_key, "unknown")
        strata[key].append(record)

    train, val = [], []
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        split_idx = max(1, int(len(group) * train_ratio))
        train.extend(group[:split_idx])
        val.extend(group[split_idx:])

    # Shuffle final sets
    rng.shuffle(train)
    rng.shuffle(val)

    return train, val


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Emit forum Q&A as VLM training data"
    )
    parser.add_argument("--input", type=Path, required=True, help="Validated Q&A directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory (prepared/)")
    parser.add_argument("--prefix", type=str, default="forum", help="Output file prefix")
    parser.add_argument("--report", type=Path, required=True, help="Emit report path")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    emit_config = load_emit_config(args.config)

    # Override prefix from CLI if provided
    prefix = args.prefix or emit_config.prefix

    # Load all validated Q&A
    qa_files = sorted(args.input.glob("*.json"))
    logger.info(f"Loading Q&A from {len(qa_files)} files")

    records = []
    forum_counts = Counter()
    type_counts = Counter()

    for qa_file in qa_files:
        with open(qa_file) as f:
            doc = json.load(f)

        for pair in doc.get("qa_pairs", []):
            record = qa_to_vlm_record(pair, doc)
            records.append(record)
            forum_counts[doc.get("forum", "unknown")] += 1
            type_counts[pair.get("question_type", "unknown")] += 1

    logger.info(f"Total records: {len(records)}")

    # Split
    train, val = stratified_split(
        records,
        emit_config.train_split,
        emit_config.random_seed,
        emit_config.stratify_by,
    )

    logger.info(f"Train: {len(train)}, Val: {len(val)}")

    # Write output
    args.output.mkdir(parents=True, exist_ok=True)
    train_path = args.output / f"{prefix}_train.jsonl"
    val_path = args.output / f"{prefix}_val.jsonl"

    with open(train_path, "w") as f:
        for record in train:
            f.write(json.dumps(record) + "\n")

    with open(val_path, "w") as f:
        for record in val:
            f.write(json.dumps(record) + "\n")

    logger.info(f"Written: {train_path} ({len(train)} records)")
    logger.info(f"Written: {val_path} ({len(val)} records)")

    # Write report
    _write_report(args.report, len(records), len(train), len(val), forum_counts, type_counts)


def _write_report(
    report_path: Path,
    total: int,
    train_count: int,
    val_count: int,
    forum_counts: Counter,
    type_counts: Counter,
):
    """Write emit summary report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Forum Pipeline — Emit Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Total Q&A pairs: {total}",
        f"- Train: {train_count} ({train_count/max(total,1)*100:.1f}%)",
        f"- Val: {val_count} ({val_count/max(total,1)*100:.1f}%)",
        "",
        "## By Forum",
        "",
        "| Forum | Count |",
        "|-------|-------|",
    ]
    for forum, count in forum_counts.most_common():
        lines.append(f"| {forum} | {count} |")

    lines += [
        "",
        "## By Question Type",
        "",
        "| Type | Count |",
        "|------|-------|",
    ]
    for qtype, count in type_counts.most_common():
        lines.append(f"| {qtype} | {count} |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
