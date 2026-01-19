#!/usr/bin/env python3
"""
Stage 2: Thread Listing Scraper

Scrapes thread listings from specified forums.
Handles pagination to collect all threads in a forum.

Usage:
    # Scrape specific forum by ID
    python scraper/11_scrape_threads.py --forum-id 42

    # Scrape specific forum by URL
    python scraper/02_scrape_threads.py --forum-url "https://example.com/forum/42-subforum"

    # Scrape all forums from discovery output
    python scraper/11_scrape_threads.py --all

    # Limit pages for testing
    python scraper/11_scrape_threads.py --forum-id 42 --max-pages 2
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.core import (
    ScraperConfig,
    ScraperSession,
    Checkpoint,
    setup_logging,
    save_json,
    load_json,
    save_html,
    append_jsonl,
)
from scraper.parser import VBulletinParser, Thread


def scrape_forum_threads(
    session: ScraperSession,
    parser: VBulletinParser,
    forum_url: str,
    forum_id: str,
    checkpoint: Checkpoint,
    logger,
    max_pages: Optional[int] = None,
) -> List[Thread]:
    """
    Scrape all threads from a forum, handling pagination.

    Args:
        session: HTTP session
        parser: HTML parser
        forum_url: URL of the forum
        forum_id: Forum ID
        checkpoint: Checkpoint for resume
        logger: Logger instance
        max_pages: Optional limit on pages to scrape

    Returns:
        List of Thread objects
    """
    threads: List[Thread] = []
    current_url = forum_url
    page = 1

    while current_url:
        # Check page limit
        if max_pages and page > max_pages:
            logger.info(f"Reached max pages limit ({max_pages})")
            break

        logger.info(f"Scraping forum {forum_id} page {page}: {current_url}")

        # Fetch page
        html = session.get_html(current_url)
        if not html:
            logger.error(f"Failed to fetch {current_url}")
            break

        # Save raw HTML
        html_path = Path(f"forum_archive/raw/forums/{forum_id}_page{page}.html")
        save_html(html, html_path)

        # Parse threads
        subforums, page_threads, pagination = parser.parse_forum_page(html, forum_id)

        logger.info(f"  Found {len(page_threads)} threads on page {page}")

        for thread in page_threads:
            if not checkpoint.is_completed(thread.thread_id):
                threads.append(thread)
                # Save thread to JSONL incrementally
                append_jsonl(
                    thread.to_dict(),
                    Path(f"forum_archive/data/threads_{forum_id}.jsonl"),
                )
                checkpoint.mark_completed(thread.thread_id)

        # Save checkpoint periodically
        if page % 5 == 0:
            checkpoint.save(Path(f"forum_archive/checkpoints/threads_{forum_id}.json"))

        # Move to next page
        if pagination.has_next and pagination.next_url:
            current_url = pagination.next_url
            page += 1
        else:
            logger.info(f"  No more pages (total: {pagination.total_pages})")
            break

    return threads


def main():
    arg_parser = argparse.ArgumentParser(
        description="Stage 2: Scrape thread listings from forums",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target specification (mutually exclusive)
    target_group = arg_parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--forum-id",
        type=str,
        help="Forum ID to scrape",
    )
    target_group.add_argument(
        "--forum-url",
        type=str,
        help="Forum URL to scrape",
    )
    target_group.add_argument(
        "--all",
        action="store_true",
        help="Scrape all forums from forums.json",
    )

    arg_parser.add_argument(
        "--config",
        type=Path,
        default=Path("scraper/scraper_config.yaml"),
        help="Path to configuration file",
    )

    arg_parser.add_argument(
        "--forums-file",
        type=Path,
        default=Path("forum_archive/data/forums.json"),
        help="Path to forums.json (for --all mode)",
    )

    arg_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("forum_archive/data"),
        help="Output directory for thread data",
    )

    arg_parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum pages to scrape per forum (for testing)",
    )

    arg_parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoint (default: True)",
    )

    arg_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = arg_parser.parse_args()

    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(
        "scrape_threads",
        log_dir=Path("forum_archive/logs"),
        level=log_level,
    )

    # Load config
    config = ScraperConfig.from_yaml(args.config)
    session = ScraperSession(config, logger)
    parser = VBulletinParser(config.base_url)

    try:
        # Determine forums to scrape
        forums_to_scrape = []

        if args.forum_url:
            forum_id = parser._extract_id(args.forum_url)
            forums_to_scrape.append({"id": forum_id, "url": args.forum_url})

        elif args.forum_id:
            forum_url = f"{config.base_url}/forum/{args.forum_id}"
            forums_to_scrape.append({"id": args.forum_id, "url": forum_url})

        elif args.all:
            forums_data = load_json(args.forums_file)
            if not forums_data:
                logger.error(f"Could not load {args.forums_file}")
                return 1
            # Flatten forum tree
            def flatten_forums(forums):
                for f in forums:
                    yield {"id": f["forum_id"], "url": f["url"]}
                    if f.get("subforums"):
                        yield from flatten_forums(f["subforums"])
            forums_to_scrape = list(flatten_forums(forums_data["forums"]))

        logger.info(f"Will scrape {len(forums_to_scrape)} forum(s)")

        # Scrape each forum
        all_threads = []
        for forum_info in forums_to_scrape:
            forum_id = forum_info["id"]
            forum_url = forum_info["url"]

            logger.info(f"\n=== Scraping forum {forum_id} ===")

            # Load or create checkpoint
            checkpoint_path = Path(f"forum_archive/checkpoints/threads_{forum_id}.json")
            if args.resume:
                checkpoint = Checkpoint.load_or_create(checkpoint_path, f"threads_{forum_id}")
            else:
                checkpoint = Checkpoint(stage=f"threads_{forum_id}")

            threads = scrape_forum_threads(
                session, parser, forum_url, forum_id, checkpoint, logger,
                max_pages=args.max_pages,
            )

            all_threads.extend(threads)

            # Save final checkpoint
            checkpoint.save(checkpoint_path)
            logger.info(f"Scraped {len(threads)} threads from forum {forum_id}")

        # Save combined summary
        summary = {
            "total_threads": len(all_threads),
            "forums_scraped": len(forums_to_scrape),
            "threads_by_forum": {},
        }
        for thread in all_threads:
            fid = thread.forum_id
            if fid not in summary["threads_by_forum"]:
                summary["threads_by_forum"][fid] = 0
            summary["threads_by_forum"][fid] += 1

        save_json(summary, args.output_dir / "threads_summary.json")
        logger.info(f"\nTotal threads scraped: {len(all_threads)}")

        # Print stats
        stats = session.get_stats()
        logger.info(f"Requests: {stats['requests']}, Errors: {stats['errors']}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
