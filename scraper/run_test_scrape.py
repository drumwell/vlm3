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

    # Run without S3 sync (local testing)
    python scraper/run_test_scrape.py --no-sync
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Test target: No-Start FAQ subforum
# You'll need to discover the actual forum ID by running stage 1 first
# or by manually finding it on the forum
TEST_FORUM_ID = None  # Will be set after forum discovery
TEST_FORUM_URL = None  # Will be discovered

# S3 sync configuration
S3_BUCKET = os.environ.get("S3_BUCKET", "")
S3_SYNC_INTERVAL = 50  # Sync every N items during scraping (via checkpoint)


def run_command(cmd: list, dry_run: bool = False) -> int:
    """Run a command, optionally as dry run."""
    cmd_str = " ".join(cmd)
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running: {cmd_str}")
    print("-" * 60)

    if dry_run:
        return 0

    result = subprocess.run(cmd)
    return result.returncode


def sync_to_s3(dry_run: bool = False, stage: str = "") -> int:
    """Sync local data to S3 bucket.

    Args:
        dry_run: If True, only print what would be done
        stage: Current stage name (for logging)

    Returns:
        0 on success, non-zero on failure
    """
    if not S3_BUCKET:
        print("[S3 Sync] Skipped: S3_BUCKET environment variable not set")
        return 0

    source = "data_src/forum"
    dest = f"s3://{S3_BUCKET}/data_src/forum"

    cmd = [
        "aws", "s3", "sync",
        source, dest,
        "--exclude", "*.log",  # Logs can be large and change frequently
    ]

    stage_info = f" (after {stage})" if stage else ""
    print(f"\n[S3 Sync{stage_info}] Syncing {source} -> {dest}")

    if dry_run:
        print(f"[DRY RUN] Would run: {' '.join(cmd)}")
        return 0

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            # Count uploaded files from output
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            upload_count = sum(1 for l in lines if l.startswith('upload:'))
            print(f"[S3 Sync] Complete: {upload_count} files uploaded")
        else:
            print(f"[S3 Sync] Warning: {result.stderr}")
        return result.returncode
    except FileNotFoundError:
        print("[S3 Sync] Warning: aws CLI not found, skipping S3 sync")
        return 0
    except Exception as e:
        print(f"[S3 Sync] Error: {e}")
        return 1


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
  2. Run '--stage all' to scrape all discovered forums
  3. Or use '--forum-id X' to scrape a specific forum

Running '--stage all' without --forum-id will scrape ALL discovered forums.
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

    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Disable S3 sync (for local testing)",
    )

    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Only sync to S3, don't run any stages",
    )

    args = parser.parse_args()

    # Handle sync-only mode
    if args.sync_only:
        if not S3_BUCKET:
            print("ERROR: S3_BUCKET environment variable not set")
            return 1
        return sync_to_s3(args.dry_run, "manual")

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

            # Sync after discover stage
            if not args.no_sync:
                sync_to_s3(args.dry_run, "discover")

        elif stage == "threads":
            forum_id = args.forum_id
            # Determine if we're running all stages (use --all) or single stage (require forum-id)
            running_all_stages = args.stage == "all"

            if forum_id:
                # Specific forum requested
                cmd = [
                    python, str(base_dir / "scraper/02_scrape_threads.py"),
                    "--forum-id", forum_id,
                    "--max-pages", str(args.max_pages),
                    *verbose_flag,
                ]
            elif running_all_stages:
                # Running all stages - use --all to scrape all discovered forums
                cmd = [
                    python, str(base_dir / "scraper/02_scrape_threads.py"),
                    "--all",
                    "--max-pages", str(args.max_pages),
                    *verbose_flag,
                ]
            else:
                print("ERROR: --forum-id is required for threads stage")
                print("Run 'discover' stage first, then set the forum ID")
                print("Or use '--stage all' to scrape all discovered forums")
                return 1

            ret = run_command(cmd, args.dry_run)
            if ret != 0:
                print(f"Stage 'threads' failed with code {ret}")
                return ret

            # Sync after threads stage
            if not args.no_sync:
                sync_to_s3(args.dry_run, "threads")

        elif stage == "posts":
            forum_id = args.forum_id
            running_all_stages = args.stage == "all"

            if forum_id:
                # Specific forum requested
                cmd = [
                    python, str(base_dir / "scraper/03_scrape_posts.py"),
                    "--forum-id", forum_id,
                    "--max-threads", str(args.max_threads),
                    "--max-pages", str(args.max_pages),
                    *verbose_flag,
                ]
            elif running_all_stages:
                # Running all stages - use --all to scrape all forums
                cmd = [
                    python, str(base_dir / "scraper/03_scrape_posts.py"),
                    "--all",
                    "--max-threads", str(args.max_threads),
                    "--max-pages", str(args.max_pages),
                    *verbose_flag,
                ]
            else:
                print("ERROR: --forum-id is required for posts stage")
                print("Or use '--stage all' to scrape all discovered forums")
                return 1

            ret = run_command(cmd, args.dry_run)
            if ret != 0:
                print(f"Stage 'posts' failed with code {ret}")
                return ret

            # Sync after posts stage
            if not args.no_sync:
                sync_to_s3(args.dry_run, "posts")

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
                # Download all images (works for both --stage all and --stage images without forum-id)
                cmd = [
                    python, str(base_dir / "scraper/04_download_images.py"),
                    "--all",
                    *verbose_flag,
                ]
            ret = run_command(cmd, args.dry_run)
            if ret != 0:
                print(f"Stage 'images' failed with code {ret}")
                return ret

            # Sync after images stage
            if not args.no_sync:
                sync_to_s3(args.dry_run, "images")

    # Final sync to ensure all data is uploaded
    if not args.no_sync and S3_BUCKET:
        print("\n" + "=" * 60)
        print("Running final S3 sync...")
        print("=" * 60)
        sync_to_s3(args.dry_run, "final")

    print("\n" + "=" * 60)
    print("Scraping complete!")
    print("=" * 60)
    print("\nOutput locations:")
    print("  Raw HTML:   data_src/forum/raw/")
    print("  Data:       data_src/forum/data/")
    print("  Images:     data_src/forum/raw/images/")
    print("  Logs:       data_src/forum/logs/")
    print("  Checkpoints: data_src/forum/checkpoints/")
    if S3_BUCKET and not args.no_sync:
        print(f"  S3 Bucket:  s3://{S3_BUCKET}/data_src/forum/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
