#!/usr/bin/env python3
"""
Stage 1: Forum Discovery

Discovers the forum structure by crawling from the index page.
Outputs a JSON file with all forums and their hierarchy.

Usage:
    python scraper/10_discover_forums.py
    python scraper/10_discover_forums.py --output forum_archive/data/forums.json
"""

import argparse
import sys
from pathlib import Path
from typing import List, Set

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scraper.core import (
    ScraperConfig,
    ScraperSession,
    Checkpoint,
    setup_logging,
    save_json,
    save_html,
)
from scraper.parser import VBulletinParser, Forum


def discover_forums(
    session: ScraperSession,
    parser: VBulletinParser,
    start_url: str,
    logger,
) -> List[Forum]:
    """
    Recursively discover all forums starting from the index.

    Args:
        session: HTTP session for requests
        parser: HTML parser
        start_url: Starting URL (forum index)
        logger: Logger instance

    Returns:
        List of top-level Forum objects with nested subforums
    """
    all_forums: List[Forum] = []
    visited_ids: Set[str] = set()

    def crawl_forum(url: str, parent_id: str = None, depth: int = 0) -> List[Forum]:
        """Recursively crawl a forum and its subforums."""
        indent = "  " * depth
        logger.info(f"{indent}Crawling: {url}")

        html = session.get_html(url)
        if not html:
            logger.warning(f"{indent}Failed to fetch {url}")
            return []

        # Save raw HTML
        forum_id = parser._extract_id(url) or "index"
        html_path = Path("forum_archive/raw/forums") / f"{forum_id}.html"
        save_html(html, html_path)

        forums = []

        if depth == 0:
            # Parse index page
            found_forums = parser.parse_forum_index(html)
            logger.info(f"{indent}Found {len(found_forums)} top-level forums")
        else:
            # Parse forum page for subforums
            subforums, threads, pagination = parser.parse_forum_page(html, forum_id)
            found_forums = subforums
            logger.info(f"{indent}Found {len(subforums)} subforums, {len(threads)} threads")

        for forum in found_forums:
            if forum.forum_id in visited_ids:
                continue
            visited_ids.add(forum.forum_id)

            forum.parent_id = parent_id
            logger.info(f"{indent}  Forum: {forum.title} (ID: {forum.forum_id})")

            # Recursively crawl subforums (limit depth to avoid infinite loops)
            if depth < 3:
                forum.subforums = crawl_forum(
                    forum.url, parent_id=forum.forum_id, depth=depth + 1
                )

            forums.append(forum)

        return forums

    all_forums = crawl_forum(start_url, depth=0)
    return all_forums


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Discover forum structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("scraper/scraper_config.yaml"),
        help="Path to configuration file",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("forum_archive/data/forums.json"),
        help="Output JSON file for forum structure",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Setup
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logging(
        "discover_forums",
        log_dir=Path("forum_archive/logs"),
        level=log_level,
    )

    # Load config
    config = ScraperConfig.from_yaml(args.config)
    logger.info(f"Base URL: {config.base_url}")

    # Initialize session and parser
    session = ScraperSession(config, logger)
    html_parser = VBulletinParser(config.base_url)

    try:
        # Discover forums
        logger.info("Starting forum discovery...")
        forums = discover_forums(session, html_parser, config.base_url, logger)

        # Save results
        forum_data = {
            "base_url": config.base_url,
            "forums": [f.to_dict() for f in forums],
            "total_forums": sum(1 for _ in _count_forums(forums)),
        }
        save_json(forum_data, args.output)
        logger.info(f"Saved {forum_data['total_forums']} forums to {args.output}")

        # Print summary
        logger.info("\n=== Forum Structure ===")
        _print_forum_tree(forums, logger)

        # Print stats
        stats = session.get_stats()
        logger.info(f"\nRequests: {stats['requests']}, Errors: {stats['errors']}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


def _count_forums(forums: List[Forum]):
    """Generator to count all forums including nested."""
    for forum in forums:
        yield forum
        yield from _count_forums(forum.subforums)


def _print_forum_tree(forums: List[Forum], logger, indent: int = 0):
    """Print forum tree structure."""
    for forum in forums:
        prefix = "  " * indent
        logger.info(f"{prefix}- {forum.title} (ID: {forum.forum_id})")
        _print_forum_tree(forum.subforums, logger, indent + 1)


if __name__ == "__main__":
    sys.exit(main())
