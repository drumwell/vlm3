#!/usr/bin/env python3
"""
Test Scrape Runner

A convenience script to test the scraper on a small subforum.
Runs all stages in sequence with appropriate parameters for testing.

Usage:
    # Run full test on No-Start FAQ subforum
    python scraper/run_test_scrape.py

    # Run specific stage
    python scraper/run_test_scrape.py --stage threads

    # Dry run (show what would be done)
    python scraper/run_test_scrape.py --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Test target: No-Start FAQ subforum
# You'll need to discover the actual forum ID by running stage 1 first
# or by manually finding it on the forum
TEST_FORUM_ID = None  # Will be set after forum discovery
TEST_FORUM_URL = None  # Will be discovered


def run_command(cmd: list, dry_run: bool = False) -> int:
    """Run a command, optionally as dry run."""
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")
    print("-" * 60)

    if dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run test scrape on a small subforum",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  discover  - Stage 1: Discover forum structure
  threads   - Stage 2: Scrape thread listings
  posts     - Stage 3: Scrape post content
  images    - Stage 4: Download images
  all       - Run all stages in sequence

Example workflow:
  1. Run 'discover' first to find forum IDs
  2. Edit this script to set TEST_FORUM_ID
  3. Run 'threads', 'posts', 'images' in sequence

Or run 'all' after setting TEST_FORUM_ID.
""",
    )

    parser.add_argument(
        "--stage",
        choices=["discover", "threads", "posts", "images", "all"],
        default="discover",
        help="Stage to run (default: discover)",
    )

    parser.add_argument(
        "--forum-id",
        type=str,
        default=TEST_FORUM_ID,
        help="Forum ID to scrape (overrides TEST_FORUM_ID)",
    )

    parser.add_argument(
        "--max-threads",
        type=int,
        default=5,
        help="Max threads to scrape (for testing)",
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=2,
        help="Max pages per forum/thread (for testing)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent.parent
    python = sys.executable
    verbose_flag = ["--verbose"] if args.verbose else []

    stages_to_run = []
    if args.stage == "all":
        stages_to_run = ["discover", "threads", "posts", "images"]
    else:
        stages_to_run = [args.stage]

    for stage in stages_to_run:
        if stage == "discover":
            # Stage 1: Discover forums
            cmd = [
                python, str(base_dir / "scraper/01_discover_forums.py"),
                *verbose_flag,
            ]
            ret = run_command(cmd, args.dry_run)
            if ret != 0:
                print(f"Stage 'discover' failed with code {ret}")
                return ret

            if not args.dry_run:
                print("\n" + "=" * 60)
                print("Forum discovery complete!")
                print("Check data_src/forum/data/forums.json for forum IDs")
                print("Then set --forum-id and run threads stage")
                print("=" * 60)

        elif stage == "threads":
            forum_id = args.forum_id
            if not forum_id:
                print("ERROR: --forum-id is required for threads stage")
                print("Run 'discover' stage first, then set the forum ID")
                return 1

            # Stage 2: Scrape thread listings
            cmd = [
                python, str(base_dir / "scraper/02_scrape_threads.py"),
                "--forum-id", forum_id,
                "--max-pages", str(args.max_pages),
                *verbose_flag,
            ]
            ret = run_command(cmd, args.dry_run)
            if ret != 0:
                print(f"Stage 'threads' failed with code {ret}")
                return ret

        elif stage == "posts":
            forum_id = args.forum_id
            if not forum_id:
                print("ERROR: --forum-id is required for posts stage")
                return 1

            # Stage 3: Scrape posts
            cmd = [
                python, str(base_dir / "scraper/03_scrape_posts.py"),
                "--forum-id", forum_id,
                "--max-threads", str(args.max_threads),
                "--max-pages", str(args.max_pages),
                *verbose_flag,
            ]
            ret = run_command(cmd, args.dry_run)
            if ret != 0:
                print(f"Stage 'posts' failed with code {ret}")
                return ret

        elif stage == "images":
            forum_id = args.forum_id
            if forum_id:
                # Download images for specific forum
                cmd = [
                    python, str(base_dir / "scraper/04_download_images.py"),
                    "--forum-id", forum_id,
                    *verbose_flag,
                ]
            else:
                # Download all images
                cmd = [
                    python, str(base_dir / "scraper/04_download_images.py"),
                    "--all",
                    *verbose_flag,
                ]
            ret = run_command(cmd, args.dry_run)
            if ret != 0:
                print(f"Stage 'images' failed with code {ret}")
                return ret

    print("\n" + "=" * 60)
    print("Scraping complete!")
    print("=" * 60)
    print("\nOutput locations:")
    print("  Raw HTML:   data_src/forum/raw/")
    print("  Data:       data_src/forum/data/")
    print("  Images:     data_src/forum/raw/images/")
    print("  Logs:       data_src/forum/logs/")
    print("  Checkpoints: data_src/forum/checkpoints/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
