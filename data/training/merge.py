#!/usr/bin/env python3
"""
Merge prepared outputs from all data sources into unified training set.

Reads from each source's prepared/ directory, applies weighting,
copies images, and emits merged_train.jsonl and merged_val.jsonl.

Usage:
    python merge.py --config config.yaml --output-dir .
    python merge.py --config config.yaml --output-dir . --dry-run
    python merge.py --config config.yaml --output-dir . --verbose
"""

import argparse
import json
import logging
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SourceConfig:
    name: str
    path: Path
    weight: float


@dataclass
class MergeConfig:
    sources: list[SourceConfig] = field(default_factory=list)
    upsample_to_balance: bool = True
    shuffle_seed: int = 42


def load_config(config_path: Path, base_dir: Path) -> MergeConfig:
    """Load merge config from YAML file."""
    import yaml

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    sources = []
    for name, src in raw.get("sources", {}).items():
        # Resolve path relative to config file directory
        src_path = (base_dir / src["path"]).resolve()
        sources.append(SourceConfig(
            name=name,
            path=src_path,
            weight=float(src.get("weight", 1.0)),
        ))

    return MergeConfig(
        sources=sources,
        upsample_to_balance=raw.get("upsample_to_balance", True),
        shuffle_seed=raw.get("shuffle_seed", 42),
    )


# ============================================================================
# Data loading
# ============================================================================

def load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_source(source: SourceConfig) -> dict[str, list[dict]]:
    """Load train and val records from a source's prepared/ directory.

    Returns {"train": [...], "val": [...]} or empty lists if files are missing.
    """
    result = {"train": [], "val": []}

    train_path = source.path / f"{source.name}_train.jsonl"
    val_path = source.path / f"{source.name}_val.jsonl"

    if train_path.exists():
        result["train"] = load_jsonl(train_path)
        logging.info("  %s train: %d records from %s", source.name, len(result["train"]), train_path)
    else:
        logging.warning("  %s train: not found at %s (skipping)", source.name, train_path)

    if val_path.exists():
        result["val"] = load_jsonl(val_path)
        logging.info("  %s val: %d records from %s", source.name, len(result["val"]), val_path)
    else:
        logging.warning("  %s val: not found at %s (skipping)", source.name, val_path)

    return result


# ============================================================================
# Image handling
# ============================================================================

def create_blank_image(output_dir: Path) -> Path:
    """Create a 28x28 solid gray placeholder image for text-only records."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    blank_path = images_dir / "blank.jpg"

    img = Image.new("RGB", (28, 28), color=(128, 128, 128))
    img.save(blank_path, "JPEG")
    logging.info("Created blank placeholder image: %s", blank_path)

    return blank_path


def copy_source_images(source: SourceConfig, output_dir: Path, dry_run: bool = False) -> int:
    """Copy images from a source's prepared/images/ to the output images/ directory.

    Returns count of images copied.
    """
    src_images = source.path / "images"
    if not src_images.exists():
        logging.info("  %s: no images/ directory", source.name)
        return 0

    dst_images = output_dir / "images"

    copied = 0
    for img_path in src_images.iterdir():
        if img_path.is_file():
            if not dry_run:
                dst_images.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, dst_images / img_path.name)
            copied += 1

    logging.info("  %s: %d images %s", source.name, copied,
                 "(dry run)" if dry_run else "copied")
    return copied


# ============================================================================
# Weighting / upsampling
# ============================================================================

def apply_weighting(
    source_records: dict[str, list[dict]],
    config: MergeConfig,
) -> dict[str, list[dict]]:
    """Apply source weights via upsampling the smaller source(s).

    When upsample_to_balance is True, oversample smaller sources using
    random.choices with replacement so the mix approaches the target
    weight ratio.
    """
    if not config.upsample_to_balance:
        # No weighting — just concatenate
        combined = {}
        for split in ("train", "val"):
            combined[split] = []
            for name, data in source_records.items():
                combined[split].extend(data[split])
        return combined

    rng = random.Random(config.shuffle_seed)

    # Build weight lookup
    weight_by_name = {s.name: s.weight for s in config.sources}
    total_weight = sum(weight_by_name.values())

    combined = {}
    for split in ("train", "val"):
        # Find the largest source (by count / target_fraction)
        # to set the scale — we only upsample, never downsample
        source_sizes = {}
        for name, data in source_records.items():
            count = len(data[split])
            if count > 0:
                target_frac = weight_by_name.get(name, 1.0) / total_weight
                source_sizes[name] = (count, target_frac)

        if not source_sizes:
            combined[split] = []
            continue

        # Determine target count for each source
        # Scale factor: largest (count / fraction) sets the total
        scale = max(count / frac for count, frac in source_sizes.values())

        records = []
        for name, data in source_records.items():
            src_records = data[split]
            if not src_records:
                continue

            target_frac = weight_by_name.get(name, 1.0) / total_weight
            target_count = int(round(scale * target_frac))

            if target_count <= len(src_records):
                # Source is already at or above target — use as-is
                records.extend(src_records)
                logging.info("  %s %s: %d records (no upsampling needed)", split, name, len(src_records))
            else:
                # Upsample with replacement
                upsampled = rng.choices(src_records, k=target_count)
                records.extend(upsampled)
                logging.info("  %s %s: %d → %d records (upsampled)", split, name, len(src_records), target_count)

        combined[split] = records

    return combined


# ============================================================================
# Text-only record handling
# ============================================================================

def inject_blank_image(records: list[dict]) -> tuple[list[dict], int]:
    """For records without an 'image' field, inject 'image': 'images/blank.jpg'.

    Returns (updated_records, text_only_count).
    """
    text_only_count = 0
    for record in records:
        if not record.get("image"):
            record["image"] = "images/blank.jpg"
            text_only_count += 1
    return records, text_only_count


# ============================================================================
# Output
# ============================================================================

def write_jsonl(records: list[dict], path: Path) -> None:
    """Write records to a JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def log_composition(records: list[dict], split: str) -> None:
    """Log composition breakdown for a split."""
    source_counts = Counter()
    content_type_counts = Counter()
    text_only_count = 0

    for r in records:
        meta = r.get("metadata", {})
        source = meta.get("source", meta.get("source_type", "unknown"))
        source_counts[source] += 1
        ct = meta.get("content_type", "unknown")
        content_type_counts[ct] += 1
        if r.get("image", "").endswith("blank.jpg"):
            text_only_count += 1

    logging.info("")
    logging.info("=== %s composition (%d total) ===", split, len(records))
    logging.info("By source:")
    for src, count in source_counts.most_common():
        logging.info("  %-25s %d", src, count)
    logging.info("By content_type:")
    for ct, count in content_type_counts.most_common():
        logging.info("  %-25s %d", ct, count)
    logging.info("Text-only (blank image):    %d", text_only_count)


# ============================================================================
# Main
# ============================================================================

def merge(config_path: Path, output_dir: Path, dry_run: bool = False) -> dict:
    """Run the merge pipeline. Returns summary dict."""
    base_dir = config_path.parent
    config = load_config(config_path, base_dir)

    logging.info("Merge config: %d sources, upsample=%s, seed=%d",
                 len(config.sources), config.upsample_to_balance, config.shuffle_seed)
    for s in config.sources:
        logging.info("  %s: weight=%.2f path=%s", s.name, s.weight, s.path)

    # --- Load all sources ---
    logging.info("")
    logging.info("Loading sources...")
    source_records = {}
    for source in config.sources:
        data = load_source(source)
        if data["train"] or data["val"]:
            source_records[source.name] = data
        else:
            logging.warning("  %s: no data found, skipping", source.name)

    if not source_records:
        logging.error("No data found from any source. Nothing to merge.")
        return {"status": "empty", "train_count": 0, "val_count": 0}

    # --- Copy images ---
    logging.info("")
    logging.info("Copying images...")
    total_images = 0
    for source in config.sources:
        if source.name in source_records:
            total_images += copy_source_images(source, output_dir, dry_run)

    # --- Create blank placeholder ---
    if not dry_run:
        create_blank_image(output_dir)

    # --- Apply weighting ---
    logging.info("")
    logging.info("Applying weights...")
    combined = apply_weighting(source_records, config)

    # --- Inject blank image for text-only records ---
    for split in ("train", "val"):
        combined[split], text_only = inject_blank_image(combined[split])
        if text_only > 0:
            logging.info("  %s: %d text-only records got blank placeholder", split, text_only)

    # --- Shuffle ---
    rng = random.Random(config.shuffle_seed + 1)  # Different seed than upsampling
    rng.shuffle(combined["train"])
    rng.shuffle(combined["val"])

    # --- Emit ---
    train_path = output_dir / "merged_train.jsonl"
    val_path = output_dir / "merged_val.jsonl"

    if dry_run:
        logging.info("")
        logging.info("DRY RUN — would write:")
        logging.info("  %s (%d records)", train_path, len(combined["train"]))
        logging.info("  %s (%d records)", val_path, len(combined["val"]))
    else:
        write_jsonl(combined["train"], train_path)
        write_jsonl(combined["val"], val_path)
        logging.info("")
        logging.info("Wrote %s (%d records)", train_path, len(combined["train"]))
        logging.info("Wrote %s (%d records)", val_path, len(combined["val"]))

    # --- Log composition ---
    log_composition(combined["train"], "train")
    log_composition(combined["val"], "val")

    return {
        "status": "ok",
        "train_count": len(combined["train"]),
        "val_count": len(combined["val"]),
        "images_copied": total_images,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge prepared data source outputs into unified training set"
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to merge config.yaml"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: same as config file directory)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be merged without writing files"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    output_dir = args.output_dir or args.config.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    result = merge(args.config, output_dir, dry_run=args.dry_run)

    logging.info("")
    logging.info("Done. train=%d, val=%d, images=%d",
                 result["train_count"], result["val_count"], result["images_copied"])

    if result["status"] == "empty":
        sys.exit(1)


if __name__ == "__main__":
    main()
