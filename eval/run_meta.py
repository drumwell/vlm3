#!/usr/bin/env python3
"""
Create or update run_meta.json for archived evaluation runs.

Extracts metadata from existing result JSONs or accepts explicit overrides.
Idempotent: merges new fields without overwriting existing ones.

Usage:
    # Auto-extract from existing results
    python eval/run_meta.py eval/reports/archive/run_20260207_134206 --auto

    # Explicit metadata
    python eval/run_meta.py eval/reports/archive/run_20260207_134206 \
        --label v1-manual-only \
        --description "First fine-tune, manual data only"

    # Combine: auto-extract then override specific fields
    python eval/run_meta.py eval/reports/archive/run_20260207_134206 \
        --auto --label v1-manual-only
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def extract_auto_metadata(run_dir: Path) -> dict:
    """Extract metadata from existing baseline.json and finetuned.json."""
    meta = {}

    # Try finetuned.json first (has adapter info)
    finetuned_path = run_dir / "finetuned.json"
    if finetuned_path.exists():
        with open(finetuned_path) as f:
            ft = json.load(f)
        meta["model"] = ft.get("model") or ft.get("model_info", {}).get("model_path")
        meta["adapter_repo"] = ft.get("adapter_repo") or ft.get("model_info", {}).get("adapter_path")
        meta["eval_samples"] = len(ft.get("samples", []))
        if ft.get("timestamp"):
            meta["created"] = ft["timestamp"]

    # Baseline for timestamp fallback and sample count cross-check
    baseline_path = run_dir / "baseline.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            bl = json.load(f)
        if not meta.get("model"):
            meta["model"] = bl.get("model") or bl.get("model_info", {}).get("model_path")
        if not meta.get("eval_samples"):
            meta["eval_samples"] = len(bl.get("samples", []))
        if not meta.get("created") and bl.get("timestamp"):
            meta["created"] = bl["timestamp"]

    # Sample stats for additional context
    stats_path = run_dir / "sample_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        if not meta.get("eval_samples"):
            meta["eval_samples"] = stats.get("total_sampled")

    # Clean up None values
    return {k: v for k, v in meta.items() if v is not None}


def load_existing_meta(run_dir: Path) -> dict:
    """Load existing run_meta.json if present."""
    meta_path = run_dir / "run_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {}


def save_meta(run_dir: Path, meta: dict):
    """Write run_meta.json."""
    meta_path = run_dir / "run_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create or update run_meta.json for an archived eval run"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to archived run directory (e.g. eval/reports/archive/run_20260207_134206)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-extract metadata from baseline.json/finetuned.json"
    )
    parser.add_argument("--label", type=str, help="Human-readable run label")
    parser.add_argument("--description", type=str, help="Run description")
    parser.add_argument("--dataset-repo", type=str, help="HuggingFace dataset repo")
    parser.add_argument("--adapter-repo", type=str, help="HuggingFace adapter repo")

    args = parser.parse_args()

    if not args.run_dir.is_dir():
        print(f"Error: {args.run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Start with existing metadata (preserve what's there)
    meta = load_existing_meta(args.run_dir)

    # Layer on auto-extracted metadata (don't overwrite existing)
    if args.auto:
        auto = extract_auto_metadata(args.run_dir)
        for k, v in auto.items():
            if k not in meta:
                meta[k] = v

    # Layer on explicit flags (always overwrite)
    if args.label is not None:
        meta["label"] = args.label
    if args.description is not None:
        meta["description"] = args.description
    if args.dataset_repo is not None:
        meta["dataset_repo"] = args.dataset_repo
    if args.adapter_repo is not None:
        meta["adapter_repo"] = args.adapter_repo

    if not meta:
        print("Warning: no metadata to write (use --auto or provide flags)", file=sys.stderr)
        sys.exit(1)

    save_meta(args.run_dir, meta)

    # Print summary
    print(f"\nMetadata for {args.run_dir.name}:")
    for k, v in sorted(meta.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
