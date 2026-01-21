#!/usr/bin/env python3
"""
Create stratified evaluation sample from validation dataset.

Samples ~300 examples from vlm_val.jsonl with stratification by question_type
to ensure coverage across all categories while maintaining statistical validity.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# Target sample sizes by question type (from evaluation plan)
# Total: ~300 samples for ~5% margin of error at 95% confidence
TARGET_SAMPLES = {
    "procedural": 80,      # Most common (31%)
    "factual": 70,         # Second most (27%)
    "visual": 25,          # Tests image comprehension
    "component": 20,       # Tests part identification
    "navigation": 20,      # Tests manual navigation
    "diagnostic": 15,      # Tests troubleshooting reasoning
    "safety": 15,          # Critical - higher accuracy threshold
    "parameter": 15,       # Tests numeric precision
    "wiring": 15,          # Tests electrical diagram reading
    # Remaining types pooled into "other"
}

# Types to pool into "other" if count is small
POOL_THRESHOLD = 20
POOL_SAMPLE_SIZE = 25


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into list of records."""
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path):
    """Save list of records to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def get_question_type(record: dict) -> str:
    """Extract question_type from record metadata."""
    metadata = record.get("metadata", {})
    return metadata.get("question_type", "unknown")


def get_content_type(record: dict) -> str:
    """Extract content_type from record metadata."""
    metadata = record.get("metadata", {})
    return metadata.get("content_type", "unknown")


def stratified_sample(
    records: list[dict],
    target_samples: dict[str, int],
    pool_threshold: int = POOL_THRESHOLD,
    pool_sample_size: int = POOL_SAMPLE_SIZE,
    seed: int = 42,
) -> tuple[list[dict], dict]:
    """
    Create stratified sample from records.

    Args:
        records: Full list of records to sample from.
        target_samples: Dict mapping question_type to target sample count.
        pool_threshold: Types with fewer than this count are pooled.
        pool_sample_size: Sample size for pooled "other" category.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (sampled records, statistics dict).
    """
    random.seed(seed)

    # Group records by question_type
    by_type = defaultdict(list)
    for record in records:
        qtype = get_question_type(record)
        by_type[qtype].append(record)

    # Calculate statistics
    stats = {
        "total_records": len(records),
        "types_found": dict(Counter(get_question_type(r) for r in records)),
        "sampled_by_type": {},
        "pooled_types": [],
    }

    sampled = []
    pooled_records = []

    # Sample from each specified type
    for qtype, target_count in target_samples.items():
        available = by_type.get(qtype, [])

        if len(available) < pool_threshold:
            # Pool small categories
            pooled_records.extend(available)
            stats["pooled_types"].append(qtype)
            continue

        # Sample up to target or all available
        sample_count = min(target_count, len(available))
        type_sample = random.sample(available, sample_count)
        sampled.extend(type_sample)

        stats["sampled_by_type"][qtype] = {
            "available": len(available),
            "sampled": sample_count,
        }

    # Handle types not in target_samples
    for qtype, type_records in by_type.items():
        if qtype not in target_samples:
            if len(type_records) < pool_threshold:
                pooled_records.extend(type_records)
                stats["pooled_types"].append(qtype)
            else:
                # Sample proportionally for unexpected types
                sample_count = min(10, len(type_records))
                sampled.extend(random.sample(type_records, sample_count))
                stats["sampled_by_type"][qtype] = {
                    "available": len(type_records),
                    "sampled": sample_count,
                }

    # Sample from pooled "other" category
    if pooled_records:
        pool_sample = random.sample(
            pooled_records,
            min(pool_sample_size, len(pooled_records))
        )
        sampled.extend(pool_sample)
        stats["sampled_by_type"]["other_pooled"] = {
            "available": len(pooled_records),
            "sampled": len(pool_sample),
        }

    # Shuffle final sample
    random.shuffle(sampled)

    stats["total_sampled"] = len(sampled)

    return sampled, stats


def print_distribution(records: list[dict], title: str = "Distribution"):
    """Print question type distribution."""
    type_counts = Counter(get_question_type(r) for r in records)
    content_counts = Counter(get_content_type(r) for r in records)

    print(f"\n{title}")
    print("=" * 50)
    print(f"Total records: {len(records)}")

    print("\nBy question_type:")
    for qtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(records)
        print(f"  {qtype:20s}: {count:5d} ({pct:5.1f}%)")

    print("\nBy content_type:")
    for ctype, count in sorted(content_counts.items(), key=lambda x: -x[1])[:10]:
        pct = 100 * count / len(records)
        print(f"  {ctype:20s}: {count:5d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Create stratified evaluation sample from validation dataset"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("training_data/vlm_val.jsonl"),
        help="Input validation JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("eval/eval_sample.jsonl"),
        help="Output sample JSONL file"
    )
    parser.add_argument(
        "--n-samples", "-n",
        type=int,
        default=300,
        help="Target total sample size (approximate)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Optional path to save sampling statistics JSON"
    )

    args = parser.parse_args()

    # Load validation data
    print(f"Loading validation data from {args.input}")
    records = load_jsonl(args.input)
    print_distribution(records, "Input Distribution")

    # Scale target samples if different total requested
    scale_factor = args.n_samples / sum(TARGET_SAMPLES.values())
    scaled_targets = {
        k: max(5, int(v * scale_factor))
        for k, v in TARGET_SAMPLES.items()
    }

    # Create stratified sample
    print(f"\nCreating stratified sample (target: {args.n_samples})...")
    sampled, stats = stratified_sample(
        records,
        scaled_targets,
        seed=args.seed,
    )

    print_distribution(sampled, "Sampled Distribution")

    # Save sample
    save_jsonl(sampled, args.output)
    print(f"\nSaved {len(sampled)} samples to {args.output}")

    # Save statistics if requested
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_output, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {args.stats_output}")

    # Print summary
    print("\nSampling Summary:")
    print(f"  Input records:  {stats['total_records']}")
    print(f"  Output samples: {stats['total_sampled']}")
    print(f"  Pooled types:   {stats['pooled_types']}")


if __name__ == "__main__":
    main()
