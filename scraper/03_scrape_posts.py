#!/usr/bin/env python3
"""
Stage 3: Post Content Scraper

Scrapes all posts from threads, handling pagination.
Stores both raw HTML and parsed structured data.

Usage:
    # Scrape posts from threads in a specific forum
    python scraper/12_scrape_posts.py --forum-id 42

    # Scrape a specific thread
    python scraper/12_scrape_posts.py --thread-id 12345

    # Scrape all threads from all forums
    python scraper/12_scrape_posts.py --all

    # Limit for testing
    python scraper/12_scrape_posts.py --forum-id 42 --max-threads 5
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Generator
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.core import (
    load_forum_config,
    ScraperSession,
    Checkpoint,
    setup_logging,
    save_json,
    load_json,
    save_html,
    append_jsonl,
)
from scraper.parser import VBulletinParser, Post, Thread


def load_threads_for_forum(forum_id: str, data_dir: Path) -> Generator[dict, None, None]:
    """Load threads from JSONL file for a forum."""
    threads_file = data_dir / f"threads_{forum_id}.jsonl"
    if not threads_file.exists():
        return

    with open(threads_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def scrape_thread_posts(
    session: ScraperSession,
    parser: VBulletinParser,
    thread: dict,
    checkpoint: Checkpoint,
    logger,
    max_pages: Optional[int] = None,
) -> List[Post]:
    """
    Scrape all posts from a single thread.

    Args:
        session: HTTP session
        parser: HTML parser
        thread: Thread dict with thread_id and url
        checkpoint: Checkpoint for resume
        logger: Logger instance
        max_pages: Optional page limit

    Returns:
        List of Post objects
    """
    thread_id = thread["thread_id"]
    thread_url = thread["url"]
    posts: List[Post] = []
    current_url = thread_url
    page = 1

    while current_url:
        if max_pages and page > max_pages:
            break

        logger.debug(f"  Page {page}: {current_url}")

        html = session.get_html(current_url)
        if not html:
            logger.warning(f"  Failed to fetch page {page}")
            break

        # Save raw HTML
        html_path = Path(f"data_src/forum/raw/threads/{thread_id}_page{page}.html")
        save_html(html, html_path)

        # Parse posts
        page_posts, pagination = parser.parse_thread_page(html, thread_id)

        for post in page_posts:
            if not checkpoint.is_completed(post.post_id):
                posts.append(post)

        # Next page
        if pagination.has_next and pagination.next_url:
            current_url = pagination.next_url
            page += 1
        else:
            break

    return posts


def main():
    arg_parser = argparse.ArgumentParser(
        description="Stage 3: Scrape post content from threads",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target specification
    target_group = arg_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--forum-id",
        type=str,
        help="Scrape all threads from this forum",
    )
    target_group.add_argument(
        "--thread-id",
        type=str,
        help="Scrape a specific thread ID",
    )
    target_group.add_argument(
        "--thread-url",
        type=str,
        help="Scrape a specific thread URL",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Scrape all threads from all forums",
    )

    arg_parser.add_argument(
        "--config",
        type=Path,
        default=Path("scraper/scraper_config.yaml"),
        help="Configuration file path",
    )

    arg_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data_src/forum/data"),
        help="Data directory with thread JSONL files",
    )

    arg_parser.add_argument(
        "--max-threads",
        type=int,
        default=None,
        help="Maximum threads to scrape (for testing)",
    )

    arg_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages per thread (for testing)",
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
        "scrape_posts",
        log_dir=Path("data_src/forum/logs"),
        level=log_level,
    )

    config = load_forum_config(args.config)
    session = ScraperSession(config, logger)
    parser = VBulletinParser(config.base_url)

    try:
        # Determine threads to scrape
        threads_to_scrape = []

        if args.thread_url:
            thread_id = parser._extract_id(args.thread_url)
            threads_to_scrape.append({
                "thread_id": thread_id,
                "url": args.thread_url,
                "forum_id": "unknown",
            })

        elif args.thread_id:
            thread_url = f"{config.base_url}/threads/{args.thread_id}"
            threads_to_scrape.append({
                "thread_id": args.thread_id,
                "url": thread_url,
                "forum_id": "unknown",
            })

        elif args.forum_id:
            # Load threads from JSONL
            for thread in load_threads_for_forum(args.forum_id, args.data_dir):
                threads_to_scrape.append(thread)
                if args.max_threads and len(threads_to_scrape) >= args.max_threads:
                    break

        elif args.all:
            # Find all thread JSONL files
            for threads_file in args.data_dir.glob("threads_*.jsonl"):
                forum_id = threads_file.stem.replace("threads_", "")
                for thread in load_threads_for_forum(forum_id, args.data_dir):
                    threads_to_scrape.append(thread)
                    if args.max_threads and len(threads_to_scrape) >= args.max_threads:
                        break
                if args.max_threads and len(threads_to_scrape) >= args.max_threads:
                    break

        logger.info(f"Will scrape {len(threads_to_scrape)} thread(s)")

        # Load checkpoint
        checkpoint_path = Path("data_src/forum/checkpoints/posts.json")
        if args.resume:
            checkpoint = Checkpoint.load_or_create(checkpoint_path, "posts")
        else:
            checkpoint = Checkpoint(stage="posts")

        # Scrape each thread
        total_posts = 0
        for i, thread in enumerate(threads_to_scrape):
            thread_id = thread["thread_id"]
            forum_id = thread.get("forum_id", "unknown")

            # Skip if thread already fully processed
            if checkpoint.is_completed(f"thread_{thread_id}"):
                logger.debug(f"Skipping already completed thread {thread_id}")
                continue

            logger.info(f"[{i+1}/{len(threads_to_scrape)}] Thread {thread_id}: {thread.get('title', 'Unknown')[:50]}")

            posts = scrape_thread_posts(
                session, parser, thread, checkpoint, logger,
                max_pages=args.max_pages,
            )

            # Save posts to JSONL
            output_file = args.data_dir / f"posts_{forum_id}.jsonl"
            for post in posts:
                post_data = post.to_dict()
                post_data["thread_title"] = thread.get("title", "")
                append_jsonl(post_data, output_file)
                checkpoint.mark_completed(post.post_id)

            # Mark thread as completed
            checkpoint.mark_completed(f"thread_{thread_id}")
            total_posts += len(posts)

            logger.info(f"  Scraped {len(posts)} posts")

            # Save checkpoint periodically
            if (i + 1) % config.checkpoint_interval == 0:
                checkpoint.save(checkpoint_path)

        # Final checkpoint save
        checkpoint.save(checkpoint_path)

        # Summary
        logger.info(f"\n=== Summary ===")
        logger.info(f"Threads processed: {len(threads_to_scrape)}")
        logger.info(f"Total posts scraped: {total_posts}")

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
