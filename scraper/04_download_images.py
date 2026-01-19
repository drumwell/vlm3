#!/usr/bin/env python3
"""
Stage 4: Image Downloader

Downloads images referenced in posts and updates post records with local paths.

Usage:
    # Download images from posts in a specific forum
    python scraper/13_download_images.py --forum-id 42

    # Download all images
    python scraper/13_download_images.py --all

    # Dry run (list images without downloading)
    python scraper/13_download_images.py --all --dry-run
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from urllib.parse import urlparse, unquote

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.core import (
    load_forum_config,
    ScraperSession,
    Checkpoint,
    setup_logging,
    save_json,
    load_json,
)


def get_image_filename(url: str, post_id: str, index: int) -> str:
    """
    Generate a safe filename for an image.

    Uses hash of URL to ensure uniqueness while keeping it short.
    """
    # Parse URL
    parsed = urlparse(url)
    path = unquote(parsed.path)

    # Get extension
    ext = Path(path).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
        ext = ".jpg"  # Default

    # Create hash of full URL for uniqueness
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]

    return f"{post_id}_{index}_{url_hash}{ext}"


def collect_images_from_posts(data_dir: Path, forum_id: str = None) -> List[Dict]:
    """
    Collect all unique image URLs from post files.

    Returns list of dicts with: url, post_id, thread_id, forum_id, index
    """
    images = []
    seen_urls: Set[str] = set()

    # Find post files
    if forum_id:
        post_files = [data_dir / f"posts_{forum_id}.jsonl"]
    else:
        post_files = list(data_dir.glob("posts_*.jsonl"))

    for post_file in post_files:
        if not post_file.exists():
            continue

        fid = post_file.stem.replace("posts_", "")

        with open(post_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                post = json.loads(line)
                post_images = post.get("images", [])

                for i, url in enumerate(post_images):
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    images.append({
                        "url": url,
                        "post_id": post["post_id"],
                        "thread_id": post["thread_id"],
                        "forum_id": fid,
                        "index": i,
                    })

    return images


def download_images(
    session: ScraperSession,
    images: List[Dict],
    output_dir: Path,
    checkpoint: Checkpoint,
    logger,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """
    Download images to output directory.

    Returns:
        Tuple of (successful_downloads, failed_downloads)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0

    for i, img in enumerate(images):
        url = img["url"]
        post_id = img["post_id"]

        # Check if already downloaded
        if checkpoint.is_completed(url):
            logger.debug(f"Skipping already downloaded: {url}")
            continue

        # Generate filename
        filename = get_image_filename(url, post_id, img["index"])
        dest_path = output_dir / filename

        if dry_run:
            logger.info(f"[DRY RUN] Would download: {url}")
            logger.info(f"          -> {dest_path}")
            continue

        logger.info(f"[{i+1}/{len(images)}] Downloading: {url[:80]}...")

        if session.download_file(url, dest_path):
            success += 1
            checkpoint.mark_completed(url)
            # Store local path mapping
            checkpoint.metadata[url] = str(dest_path)
            logger.debug(f"  Saved to: {dest_path}")
        else:
            failed += 1
            checkpoint.mark_failed(url)
            logger.warning(f"  Failed to download")

        # Save checkpoint periodically
        if (i + 1) % 50 == 0:
            checkpoint.save(Path("data_src/forum/checkpoints/images.json"))

    return success, failed


def create_image_manifest(
    images: List[Dict],
    checkpoint: Checkpoint,
    output_path: Path,
):
    """
    Create a manifest mapping image URLs to local paths and post references.
    """
    manifest = {
        "total_images": len(images),
        "downloaded": len(checkpoint.completed_ids),
        "failed": len(checkpoint.failed_ids),
        "mappings": [],
    }

    for img in images:
        url = img["url"]
        local_path = checkpoint.metadata.get(url, "")

        manifest["mappings"].append({
            "url": url,
            "local_path": local_path,
            "post_id": img["post_id"],
            "thread_id": img["thread_id"],
            "forum_id": img["forum_id"],
            "downloaded": url in checkpoint.completed_ids,
        })

    save_json(manifest, output_path)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Stage 4: Download images from posts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    target_group = arg_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--forum-id",
        type=str,
        help="Download images from this forum's posts",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Download all images from all forums",
    )

    arg_parser.add_argument(
        "--config",
        type=Path,
        default=Path("scraper/scraper_config.yaml"),
        help="Configuration file",
    )

    arg_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_src/forum/data"),
        help="Data directory with post JSONL files",
    )

    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_src/forum/raw/images"),
        help="Output directory for downloaded images",
    )

    arg_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List images without downloading",
    )

    arg_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint",
    )

    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = arg_parser.parse_args()

    # Setup
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(
        "download_images",
        log_dir=Path("data_src/forum/logs"),
        level=log_level,
    )

    config = load_forum_config(args.config)
    session = ScraperSession(config, logger)

    try:
        # Collect images
        logger.info("Collecting image URLs from posts...")
        forum_id = args.forum_id if args.forum_id else None
        images = collect_images_from_posts(args.data_dir, forum_id)
        logger.info(f"Found {len(images)} unique images")

        if not images:
            logger.info("No images to download")
            return 0

        # Load checkpoint
        checkpoint_path = Path("data_src/forum/checkpoints/images.json")
        if args.resume:
            checkpoint = Checkpoint.load_or_create(checkpoint_path, "images")
        else:
            checkpoint = Checkpoint(stage="images")

        # Download
        logger.info(f"Downloading images to {args.output_dir}...")
        success, failed = download_images(
            session, images, args.output_dir, checkpoint, logger,
            dry_run=args.dry_run,
        )

        # Save checkpoint
        checkpoint.save(checkpoint_path)

        # Create manifest
        manifest_path = args.data_dir / "image_manifest.json"
        create_image_manifest(images, checkpoint, manifest_path)
        logger.info(f"Created manifest: {manifest_path}")

        # Summary
        logger.info(f"\n=== Summary ===")
        logger.info(f"Total images: {len(images)}")
        logger.info(f"Downloaded: {success}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Previously downloaded: {len(checkpoint.completed_ids) - success}")

        stats = session.get_stats()
        logger.info(f"Requests: {stats['requests']}, Errors: {stats['errors']}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted - saving checkpoint")
        checkpoint.save(checkpoint_path)
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
