#!/usr/bin/env python3
"""
09_upload_vlm.py - Upload VLM dataset to HuggingFace Hub

Stage 6 of the data pipeline: Upload validated dataset to HuggingFace.

Usage:
    python scripts/09_upload_vlm.py \
        --train training_data/vlm_train.jsonl \
        --val training_data/vlm_val.jsonl \
        --images training_data/images \
        --repo drumwell/vlm3 \
        --report work/logs/upload_report.md \
        --config config.yaml
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
import yaml

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class UploadConfig:
    """Configuration for HuggingFace upload."""
    repo_id: Optional[str] = None
    private: bool = False
    commit_message: str = "Update VLM training dataset"
    language: str = "en"
    license: str = "cc-by-nc-4.0"
    task_categories: List[str] = field(default_factory=lambda: ["visual-question-answering"])
    tags: List[str] = field(default_factory=lambda: ["automotive", "bmw", "vlm"])


def load_upload_config(config_path: Path) -> UploadConfig:
    """
    Load upload configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        UploadConfig with loaded values
    """
    config = UploadConfig()

    if config_path.exists():
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}

        upload_config = yaml_config.get("upload", {})

        if "repo_id" in upload_config:
            config.repo_id = upload_config["repo_id"]
        if "private" in upload_config:
            config.private = upload_config["private"]
        if "commit_message" in upload_config:
            config.commit_message = upload_config["commit_message"]

        # Dataset card metadata
        card_config = upload_config.get("dataset_card", {})
        if "language" in card_config:
            config.language = card_config["language"]
        if "license" in card_config:
            config.license = card_config["license"]
        if "task_categories" in card_config:
            config.task_categories = card_config["task_categories"]
        if "tags" in card_config:
            config.tags = card_config["tags"]

    return config


# ============================================================================
# Token Handling
# ============================================================================

def get_hf_token(cli_token: Optional[str] = None) -> str:
    """
    Get HuggingFace token from CLI arg or environment.

    Priority:
    1. CLI argument
    2. HF_TOKEN environment variable
    3. HUGGING_FACE_HUB_TOKEN environment variable

    Raises:
        ValueError: If no token found
    """
    if cli_token:
        return cli_token

    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    raise ValueError(
        "No HuggingFace token found. Provide via --token argument, "
        "HF_TOKEN, or HUGGING_FACE_HUB_TOKEN environment variable."
    )


# ============================================================================
# Dataset Card Generation
# ============================================================================

def create_dataset_card(config: UploadConfig, stats: Dict) -> str:
    """
    Create dataset card (README.md) content.

    Includes:
    - Title and description
    - Dataset statistics
    - Usage examples
    - License information
    - Citation format

    Args:
        config: Upload configuration
        stats: Dataset statistics

    Returns:
        Markdown content for README.md
    """
    # YAML frontmatter
    frontmatter = {
        "language": [config.language] if isinstance(config.language, str) else config.language,
        "license": config.license,
        "task_categories": config.task_categories,
        "tags": config.tags
    }

    # Build frontmatter string
    fm_lines = ["---"]
    for key, value in frontmatter.items():
        if isinstance(value, list):
            fm_lines.append(f"{key}:")
            for item in value:
                fm_lines.append(f"- {item}")
        else:
            fm_lines.append(f"{key}: {value}")
    fm_lines.append("---")

    # Format numbers with commas
    train_count = f"{stats.get('train_count', 0):,}"
    val_count = f"{stats.get('val_count', 0):,}"
    image_count = f"{stats.get('image_count', 0):,}"

    # Build markdown content
    content = f"""
# BMW E30 M3 Service Manual VLM Dataset

Visual question-answering dataset for BMW E30 M3 and 320is service procedures.

## Dataset Statistics

| Split | Q&A Pairs | Unique Images |
|-------|-----------|---------------|
| Train | {train_count} | {image_count} |
| Val | {val_count} | - |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{config.repo_id or 'your-username/dataset-name'}")
```

## Data Format

Each record contains:
- `image`: Path to service manual page image (or null for text-only)
- `conversations`: List of user/assistant message pairs
- `metadata`: Source information (section, page, content type)

## License

{config.license.upper()}

## Citation

```bibtex
@dataset{{bmw_e30_m3_vlm,
  title={{BMW E30 M3 Service Manual VLM Dataset}},
  year={{2025}},
  publisher={{HuggingFace}}
}}
```
"""

    return "\n".join(fm_lines) + "\n" + content.strip()


# ============================================================================
# File Preparation
# ============================================================================

def prepare_dataset_files(
    train_path: Path,
    val_path: Path,
    images_dir: Path
) -> Dict[str, Path]:
    """
    Prepare file mapping for upload.

    Args:
        train_path: Path to training JSONL
        val_path: Path to validation JSONL
        images_dir: Path to images directory

    Returns:
        Dict mapping relative paths to local paths:
        {
            "train.jsonl": Path("/path/to/vlm_train.jsonl"),
            "val.jsonl": Path("/path/to/vlm_val.jsonl"),
            "images/21-03.jpg": Path("/path/to/images/21-03.jpg"),
            ...
        }
    """
    files = {
        "train.jsonl": train_path,
        "val.jsonl": val_path
    }

    # Add images
    if images_dir.exists():
        for img_file in images_dir.glob("*"):
            if img_file.is_file():
                files[f"images/{img_file.name}"] = img_file

    return files


# ============================================================================
# HuggingFace Upload
# ============================================================================

# Import HfApi lazily to avoid import errors if huggingface_hub not installed
try:
    from huggingface_hub import HfApi
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False
    HfApi = None


def upload_to_huggingface(
    files: Dict[str, Path],
    config: UploadConfig,
    token: str,
    dataset_card: str
) -> str:
    """
    Upload dataset to HuggingFace Hub.

    Args:
        files: Mapping of relative paths to local paths
        config: Upload configuration
        token: HuggingFace API token
        dataset_card: README.md content

    Returns:
        URL of uploaded dataset

    Uses huggingface_hub library for upload.
    """
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub is required for upload. Install with: pip install huggingface_hub")

    api = HfApi()

    # Create repo if needed
    try:
        api.create_repo(
            repo_id=config.repo_id,
            token=token,
            repo_type="dataset",
            private=config.private,
            exist_ok=True
        )
    except Exception as e:
        logger.warning(f"Could not create repo (may already exist): {e}")

    # Create a temporary directory for upload
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # Create README.md
        readme_path = tmp_path / "README.md"
        with open(readme_path, "w") as f:
            f.write(dataset_card)

        # Copy/link files to temp directory
        for rel_path, local_path in files.items():
            dst = tmp_path / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)
            if local_path.exists():
                import shutil
                shutil.copy2(local_path, dst)

        # Upload folder
        api.upload_folder(
            folder_path=str(tmp_path),
            repo_id=config.repo_id,
            repo_type="dataset",
            token=token,
            commit_message=config.commit_message
        )

    url = f"https://huggingface.co/datasets/{config.repo_id}"
    return url


# ============================================================================
# Report Generation
# ============================================================================

def generate_upload_report(stats: Dict, url: str, report_path: Path) -> None:
    """
    Generate upload confirmation report.

    Args:
        stats: Upload statistics
        url: Dataset URL
        report_path: Path to write report
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().isoformat()

    lines = [
        "# VLM Dataset Upload Report",
        "",
        f"**Generated**: {timestamp}",
        f"**Status**: \u2705 SUCCESS",
        "",
        "## Upload Details",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| URL | {url} |",
        f"| Visibility | {'Private' if stats.get('private', False) else 'Public'} |",
        "",
        "## Files Uploaded",
        "",
        "| File Type | Count | Size |",
        "|-----------|-------|------|",
        f"| JSONL | 2 | - |",
        f"| Images | {stats.get('image_count', 0):,} | {stats.get('total_size_mb', 0):.1f} MB |",
        f"| README | 1 | - |",
        "",
        "## Dataset Statistics",
        "",
        "| Split | Q&A Pairs |",
        "|-------|-----------|",
        f"| Train | {stats.get('train_count', 0):,} |",
        f"| Val | {stats.get('val_count', 0):,} |",
        ""
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Upload report written to {report_path}")


# ============================================================================
# JSONL Loading (for stats)
# ============================================================================

def load_jsonl_count(file_path: Path) -> int:
    """Count records in JSONL file."""
    count = 0
    with open(file_path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Upload VLM dataset to HuggingFace"
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
        "--repo", type=str, required=True,
        help="HuggingFace repo ID (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--report", type=Path, required=True,
        help="Upload report path"
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Config YAML file"
    )
    parser.add_argument(
        "--token", type=str,
        help="HuggingFace token (or use HF_TOKEN env var)"
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Make dataset private"
    )
    parser.add_argument(
        "--message", type=str,
        help="Custom commit message"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded without actually uploading"
    )

    args = parser.parse_args()

    config = load_upload_config(args.config)
    config.repo_id = args.repo
    if args.private:
        config.private = True
    if args.message:
        config.commit_message = args.message

    # Prepare files
    files = prepare_dataset_files(args.train, args.val, args.images)

    # Compute stats
    train_count = load_jsonl_count(args.train)
    val_count = load_jsonl_count(args.val)
    image_count = len([f for f in files if f.startswith("images/")])

    total_size = sum(f.stat().st_size for f in files.values() if f.exists())
    total_size_mb = total_size / (1024 * 1024)

    stats = {
        "train_count": train_count,
        "val_count": val_count,
        "image_count": image_count,
        "total_size_mb": total_size_mb,
        "private": config.private
    }

    # Create dataset card
    dataset_card = create_dataset_card(config, stats)

    if args.dry_run:
        logger.info(f"[DRY RUN] Would upload to {config.repo_id}")
        logger.info(f"[DRY RUN] Files: {len(files)} ({total_size_mb:.1f} MB)")
        logger.info(f"[DRY RUN] Train: {train_count}, Val: {val_count}")
        logger.info(f"[DRY RUN] Dataset card preview:")
        logger.info("-" * 40)
        logger.info(dataset_card[:500])
        logger.info("-" * 40)
        return

    # Get token
    token = get_hf_token(args.token)

    # Upload
    logger.info(f"Uploading to {config.repo_id}...")
    url = upload_to_huggingface(files, config, token, dataset_card)

    # Generate report
    generate_upload_report(stats, url, args.report)

    logger.info(f"\u2705 Dataset uploaded: {url}")


if __name__ == "__main__":
    main()
