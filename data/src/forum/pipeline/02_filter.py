#!/usr/bin/env python3
"""
02_filter.py - Mechanical filtering of reconstructed threads

Stage 2 of the forum pipeline: removes noise at post and thread level using
deterministic rules. No API calls.

Usage:
    python pipeline/02_filter.py \
        --input work/threads \
        --output work/threads_clean \
        --report work/logs/filter_report.md \
        --config config.yaml \
        [--dry-run] [--verbose]
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class FilterConfig:
    """Configuration for mechanical filtering."""
    min_post_length: int = 30
    min_first_post_length: int = 50
    min_thread_posts: int = 2
    social_noise_patterns: List[str] = field(default_factory=list)
    title_noise_patterns: List[str] = field(default_factory=list)
    strip_post_number_markers: bool = True
    strip_timestamps: bool = True
    strip_signatures: bool = True
    normalize_whitespace: bool = True
    strip_html_artifacts: bool = True


def load_filter_config(config_path: Path) -> FilterConfig:
    """Load filter configuration from YAML file."""
    config = FilterConfig()
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}, using defaults")
        return config

    with open(config_path) as f:
        data = yaml.safe_load(f)

    filt = data.get("filtering", {})
    for key in ["min_post_length", "min_first_post_length", "min_thread_posts",
                "strip_post_number_markers", "strip_timestamps", "strip_signatures",
                "normalize_whitespace", "strip_html_artifacts"]:
        if key in filt:
            setattr(config, key, filt[key])

    config.social_noise_patterns = filt.get("social_noise_patterns", config.social_noise_patterns)
    config.title_noise_patterns = filt.get("title_noise_patterns", config.title_noise_patterns)
    return config


# ============================================================================
# Content Cleaning
# ============================================================================

# Regex for leading post number markers: #1, #16, etc.
POST_NUMBER_RE = re.compile(r"^#\d+\s*")

# Regex for timestamp lines: 06-20-2006, 12:17 AM or similar
TIMESTAMP_RE = re.compile(
    r"^\d{2}-\d{2}-\d{4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s*$",
    re.MULTILINE,
)

# Signature dividers
SIG_DIVIDERS = ["--", "___", "---", "____", "═══", "───"]

# HTML artifacts
HTML_ARTIFACT_RE = re.compile(r"<[^>]+>")
HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|nbsp|#\d+);")


def clean_content(text: str, config: FilterConfig) -> str:
    """Apply content cleaning rules to post text."""
    if not text:
        return ""

    if config.strip_post_number_markers:
        text = POST_NUMBER_RE.sub("", text)

    if config.strip_timestamps:
        text = TIMESTAMP_RE.sub("", text)

    if config.strip_html_artifacts:
        text = HTML_ARTIFACT_RE.sub("", text)
        text = HTML_ENTITY_RE.sub(" ", text)

    if config.strip_signatures:
        # Find signature divider in the bottom half of the post and truncate there.
        # Only match dividers on standalone lines. This avoids cutting on em-dashes
        # (--) that appear in mid-content technical text.
        lines = text.split("\n")
        if len(lines) > 2:
            earliest_sig_line = len(lines) // 2
            sig_line = None
            for i in range(earliest_sig_line, len(lines)):
                if lines[i].strip() in SIG_DIVIDERS:
                    sig_line = i
                    break
            if sig_line is not None:
                text = "\n".join(lines[:sig_line])

    if config.normalize_whitespace:
        # Collapse multiple blank lines to one
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip trailing whitespace per line
        text = "\n".join(line.rstrip() for line in text.split("\n"))
        text = text.strip()

    return text


# ============================================================================
# Post-Level Filters
# ============================================================================

def is_social_noise(text: str, patterns: List[str]) -> bool:
    """Check if post is pure social noise.

    Matches exact phrases (after stripping punctuation) and short posts
    that start with a social pattern (catches "thanks for the info", etc.).
    """
    normalized = text.strip().lower().rstrip("!.?,")
    patterns_lower = [p.lower() for p in patterns]

    # Exact match
    if normalized in patterns_lower:
        return True

    # Short text starting with social pattern
    if len(normalized) < 60:
        for p in patterns_lower:
            if normalized.startswith(p):
                return True

    return False


def is_quote_only(post: Dict) -> bool:
    """Check if post is entirely quoted content with no original text."""
    text = post.get("content_text", "").strip()
    quotes = post.get("quotes", [])
    if not quotes:
        return False

    # Remove quoted text from content and check what remains
    remaining = text
    for quote in quotes:
        quote_text = quote if isinstance(quote, str) else quote.get("text", "")
        if quote_text:
            remaining = remaining.replace(quote_text, "")

    remaining = remaining.strip()
    return len(remaining) < 10


def is_url_only(text: str) -> bool:
    """Check if post contains only URLs with no explanatory text."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return True
    url_pattern = re.compile(r"^https?://\S+$")
    return all(url_pattern.match(line) for line in lines)


def is_image_only(post: Dict) -> bool:
    """Check if post has images but no meaningful text."""
    text = post.get("content_text", "").strip()
    images = post.get("content_images", [])
    # Has images but text is trivially short
    return bool(images) and len(text) < 10


def filter_post(post: Dict, config: FilterConfig) -> Optional[str]:
    """
    Check if a post should be filtered. Returns reason string or None if it passes.
    """
    text = post.get("content_text", "").strip()

    if len(text) < config.min_post_length:
        return "too_short"

    if is_social_noise(text, config.social_noise_patterns):
        return "social_noise"

    if is_quote_only(post):
        return "quote_only"

    if is_url_only(text):
        return "url_only"

    if is_image_only(post):
        return "image_only"

    return None


# ============================================================================
# Thread-Level Filters
# ============================================================================

def filter_thread_title(title: str, patterns: List[str]) -> Optional[str]:
    """Check if thread title matches noise patterns."""
    title_lower = title.lower()
    for pattern in patterns:
        if pattern.lower() in title_lower:
            return f"title_noise:{pattern}"
    return None


def filter_thread(thread: Dict, config: FilterConfig) -> Optional[str]:
    """
    Check if entire thread should be filtered. Returns reason or None.
    Called AFTER post-level filtering.
    """
    posts = thread.get("posts", [])

    if len(posts) < config.min_thread_posts:
        return "too_few_posts"

    # Check first post length
    if posts:
        first_text = posts[0].get("content_text", "").strip()
        if len(first_text) < config.min_first_post_length:
            return "first_post_too_short"

    # Check title
    title = thread.get("thread_title", "")
    title_reason = filter_thread_title(title, config.title_noise_patterns)
    if title_reason:
        return title_reason

    return None


# ============================================================================
# Main Pipeline
# ============================================================================

def process_threads(
    input_dir: Path,
    output_dir: Path,
    config: FilterConfig,
    dry_run: bool = False,
) -> Dict:
    """Process all threads through filtering pipeline. Returns stats."""
    stats = {
        "total_threads": 0,
        "total_posts": 0,
        "posts_filtered": Counter(),
        "threads_filtered": Counter(),
        "threads_surviving": 0,
        "posts_surviving": 0,
        "by_forum": defaultdict(lambda: {"threads_in": 0, "threads_out": 0, "posts_in": 0, "posts_out": 0}),
    }

    thread_files = sorted(input_dir.glob("*.json"))
    stats["total_threads"] = len(thread_files)

    for thread_file in thread_files:
        with open(thread_file) as f:
            thread = json.load(f)

        forum = thread.get("forum", "unknown")
        original_post_count = len(thread.get("posts", []))
        stats["total_posts"] += original_post_count
        stats["by_forum"][forum]["threads_in"] += 1
        stats["by_forum"][forum]["posts_in"] += original_post_count

        # Post-level filtering: clean content BEFORE filtering so length
        # checks apply to cleaned text (prevents noise passing length gate
        # on raw text then cleaning down to trivial content).
        surviving_posts = []
        for post in thread.get("posts", []):
            post["content_text"] = clean_content(post["content_text"], config)
            reason = filter_post(post, config)
            if reason:
                stats["posts_filtered"][reason] += 1
            else:
                surviving_posts.append(post)

        thread["posts"] = surviving_posts
        thread["post_count"] = len(surviving_posts)

        # Thread-level filtering
        reason = filter_thread(thread, config)
        if reason:
            stats["threads_filtered"][reason] += 1
            continue

        stats["threads_surviving"] += 1
        stats["posts_surviving"] += len(surviving_posts)
        stats["by_forum"][forum]["threads_out"] += 1
        stats["by_forum"][forum]["posts_out"] += len(surviving_posts)

        if not dry_run:
            output_file = output_dir / thread_file.name
            with open(output_file, "w") as f:
                json.dump(thread, f, indent=2)

    return stats


def write_report(stats: Dict, report_path: Path):
    """Write markdown filter report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Forum Pipeline — Filter Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Threads in: {stats['total_threads']}",
        f"- Threads out: {stats['threads_surviving']} ({stats['threads_surviving']/max(stats['total_threads'],1)*100:.1f}%)",
        f"- Posts in: {stats['total_posts']}",
        f"- Posts out: {stats['posts_surviving']} ({stats['posts_surviving']/max(stats['total_posts'],1)*100:.1f}%)",
        "",
        "## Post-Level Filters",
        "",
        "| Filter | Posts Removed |",
        "|--------|-------------|",
    ]
    for reason, count in stats["posts_filtered"].most_common():
        lines.append(f"| {reason} | {count} |")

    lines += [
        "",
        "## Thread-Level Filters",
        "",
        "| Filter | Threads Removed |",
        "|--------|----------------|",
    ]
    for reason, count in stats["threads_filtered"].most_common():
        lines.append(f"| {reason} | {count} |")

    lines += [
        "",
        "## Per-Forum Survival",
        "",
        "| Forum | Threads In | Threads Out | Posts In | Posts Out |",
        "|-------|-----------|------------|---------|----------|",
    ]
    for forum in sorted(stats["by_forum"]):
        f = stats["by_forum"][forum]
        lines.append(f"| {forum} | {f['threads_in']} | {f['threads_out']} | {f['posts_in']} | {f['posts_out']} |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Report written to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Mechanical filtering of reconstructed threads"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input threads directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for clean threads")
    parser.add_argument("--report", type=Path, required=True, help="Filter report path")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_filter_config(args.config)

    if not args.input.exists():
        logger.error(f"Input directory not found: {args.input}")
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    logger.info(f"Filtering threads from {args.input}")
    stats = process_threads(args.input, args.output, config, dry_run=args.dry_run)

    logger.info(f"Threads: {stats['total_threads']} → {stats['threads_surviving']}")
    logger.info(f"Posts: {stats['total_posts']} → {stats['posts_surviving']}")

    write_report(stats, args.report)


if __name__ == "__main__":
    main()
