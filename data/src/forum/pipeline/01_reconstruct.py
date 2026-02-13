#!/usr/bin/env python3
"""
01_reconstruct.py - Reconstruct threads from raw forum posts

Stage 1 of the forum pipeline: groups individual posts into complete thread
conversations, ordered chronologically. Enriches with thread metadata.

Usage:
    python pipeline/01_reconstruct.py \
        --raw raw \
        --output work/threads \
        --config config.yaml
"""

import argparse
import json
import logging
import re
import statistics
import sys
from collections import defaultdict
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

def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Image URL filtering
# ============================================================================

SPONSOR_PATTERNS = [
    "member_sponsors/",
    "memberlogos/",
    "/avatar",
    "smilies/",
    "statusicon/",
    "buttons/",
]


def filter_images(images: List[str]) -> List[str]:
    """Strip forum chrome images (signatures, avatars, sponsor logos)."""
    if not images:
        return []
    return [
        url for url in images
        if not any(pat in url.lower() for pat in SPONSOR_PATTERNS)
    ]


# ============================================================================
# Thread reconstruction
# ============================================================================

def load_posts(raw_dir: Path, target_forums: List[str]) -> Dict[str, List[Dict]]:
    """Load posts from target forum JSONL files, grouped by thread_id."""
    threads = defaultdict(list)
    total_posts = 0

    for forum in target_forums:
        posts_file = raw_dir / f"posts_{forum}.jsonl"
        if not posts_file.exists():
            logger.warning(f"Posts file not found: {posts_file}")
            continue

        count = 0
        with open(posts_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                post = json.loads(line)
                post["forum"] = forum
                post["images"] = filter_images(post.get("images", []))
                threads[post["thread_id"]].append(post)
                count += 1

        logger.info(f"  {forum}: {count} posts")
        total_posts += count

    logger.info(f"Total: {total_posts} posts across {len(threads)} threads")
    return dict(threads)


def load_thread_metadata(raw_dir: Path, target_forums: List[str]) -> Dict[str, Dict]:
    """Load thread metadata from threads_*.jsonl files."""
    metadata = {}

    for forum in target_forums:
        threads_file = raw_dir / f"threads_{forum}.jsonl"
        if not threads_file.exists():
            logger.warning(f"Threads file not found: {threads_file}")
            continue

        with open(threads_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                thread = json.loads(line)
                metadata[thread["thread_id"]] = thread

    logger.info(f"Loaded metadata for {len(metadata)} threads")
    return metadata


def reconstruct_thread(
    thread_id: str,
    posts: List[Dict],
    metadata: Dict[str, Dict],
) -> Optional[Dict]:
    """Reconstruct a single thread from its posts and metadata.

    Returns None if posts list is empty.
    """
    if not posts:
        logger.warning(f"Thread {thread_id} has no posts, skipping")
        return None

    # Sort posts by post_number
    posts_sorted = sorted(posts, key=lambda p: int(p.get("post_number", 0)))

    first_post = posts_sorted[0]
    last_post = posts_sorted[-1]
    forum = first_post.get("forum", "unknown")

    # Get thread metadata
    meta = metadata.get(thread_id, {})

    thread = {
        "thread_id": thread_id,
        "thread_title": meta.get("title", first_post.get("thread_title", "")),
        "forum": forum,
        "author": first_post.get("author", ""),
        "post_count": len(posts_sorted),
        "first_post_date": first_post.get("created_at", ""),
        "last_post_date": last_post.get("created_at", ""),
        "is_sticky": meta.get("is_sticky", False),
        "is_locked": meta.get("is_locked", False),
        "posts": [],
    }

    for post in posts_sorted:
        thread["posts"].append({
            "post_id": post["post_id"],
            "author": post.get("author", ""),
            "post_number": int(post.get("post_number", 0)),
            "is_first_post": post.get("is_first_post", False),
            "content_text": post.get("content_text", ""),
            "content_images": post.get("images", []),
            "quotes": post.get("quotes", []),
        })

    return thread


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reconstruct threads from raw forum posts"
    )
    parser.add_argument("--raw", type=Path, required=True, help="Raw data directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for threads")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    target_forums = config["forums"]["targets"]

    logger.info(f"Reconstructing threads from {len(target_forums)} forums")

    # Load posts and metadata
    posts_by_thread = load_posts(args.raw, target_forums)
    metadata = load_thread_metadata(args.raw, target_forums)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Reconstruct and write threads
    forum_counts = defaultdict(int)
    thread_lengths = []

    for thread_id, posts in posts_by_thread.items():
        thread = reconstruct_thread(thread_id, posts, metadata)
        if thread is None:
            continue

        output_file = args.output / f"{thread_id}.json"
        with open(output_file, "w") as f:
            json.dump(thread, f, indent=2)

        forum_counts[thread["forum"]] += 1
        thread_lengths.append(len(posts))

    # Log summary
    logger.info("")
    logger.info("=== Reconstruction Summary ===")
    logger.info(f"Total threads: {len(posts_by_thread)}")
    for forum in sorted(forum_counts):
        logger.info(f"  {forum}: {forum_counts[forum]} threads")
    logger.info(f"Total posts: {sum(thread_lengths)}")
    if thread_lengths:
        logger.info(f"Thread length: mean={statistics.mean(thread_lengths):.1f}, "
                     f"median={statistics.median(thread_lengths):.1f}, "
                     f"min={min(thread_lengths)}, max={max(thread_lengths)}")


if __name__ == "__main__":
    main()
