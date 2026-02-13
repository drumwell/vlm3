#!/usr/bin/env python3
"""
03_classify.py - Thread classification via Claude API

Stage 3 of the forum pipeline: classifies each thread by type and technical
relevance. Gates on technical_depth >= medium AND s14_specific == true.

Usage:
    python pipeline/03_classify.py \
        --input work/threads_clean \
        --output work/threads_classified \
        --config config.yaml \
        [--verbose]
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
import yaml

load_dotenv()

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


DEPTH_ORDER = {"high": 3, "medium": 2, "low": 1}


def passes_gate(classification: Dict, config: Dict) -> bool:
    """Check if classification passes the quality gate."""
    cls_config = config.get("classification", {})
    min_depth = cls_config.get("min_technical_depth", "medium")
    require_s14 = cls_config.get("require_s14_specific", True)

    depth = classification.get("technical_depth", "low")
    s14 = classification.get("s14_specific", False)

    depth_ok = DEPTH_ORDER.get(depth, 0) >= DEPTH_ORDER.get(min_depth, 2)
    s14_ok = s14 or not require_s14

    return depth_ok and s14_ok


# ============================================================================
# Thread Preview
# ============================================================================

def build_thread_preview(thread: Dict, max_chars: int, max_replies: int) -> str:
    """Build a truncated preview of thread for classification."""
    parts = []
    title = thread.get("thread_title", "")
    parts.append(f"Thread title: {title}")

    posts = thread.get("posts", [])
    if not posts:
        return "\n\n".join(parts)

    # First post
    first = posts[0].get("content_text", "")
    parts.append(f"First post by {posts[0].get('author', 'unknown')}:\n{first}")

    # First N replies
    for post in posts[1:max_replies + 1]:
        parts.append(f"Reply by {post.get('author', 'unknown')}:\n{post.get('content_text', '')}")

    preview = "\n\n".join(parts)
    if len(preview) > max_chars:
        preview = preview[:max_chars] + "\n[truncated]"

    return preview


# ============================================================================
# Claude API
# ============================================================================

CLASSIFICATION_PROMPT = """You are classifying forum threads from the s14.net BMW E30 M3 community.

Analyze this thread and return a JSON classification:

{preview}

Return ONLY valid JSON with these fields:
{{
  "thread_type": "<one of: troubleshooting, how_to, modification, specification, parts_discussion, experience_report, general_question, off_topic>",
  "technical_depth": "<one of: high, medium, low>",
  "s14_specific": <true if specifically about BMW E30 M3 / S14 engine, false if generic>,
  "has_resolution": <true if the thread reaches a clear answer or resolution>,
  "confidence": <0.0-1.0 your confidence in this classification>
}}

Classification guidelines:
- "troubleshooting": Problem described with diagnostic steps or resolution attempted
- "how_to": Step-by-step procedure or guide
- "modification": Aftermarket parts, performance mods, custom work
- "specification": Discussion of specific measurements, torque values, clearances
- "parts_discussion": Which parts to use, part numbers, suppliers
- "experience_report": First-hand experience with a repair or modification
- "general_question": Technical question without diagnostic framing
- "off_topic": Not relevant to E30 M3 maintenance or modification

Technical depth:
- "high": Contains specific numbers, part numbers, detailed procedures
- "medium": Contains useful technical information but less specific
- "low": Vague, opinion-based, or social content"""


def classify_thread(
    thread: Dict,
    config: Dict,
    client,
) -> Optional[Dict]:
    """Classify a single thread via Claude API."""
    cls_config = config.get("classification", {})
    max_chars = cls_config.get("max_preview_chars", 2000)
    max_replies = cls_config.get("max_replies_for_preview", 3)

    preview = build_thread_preview(thread, max_chars, max_replies)
    prompt = CLASSIFICATION_PROMPT.format(preview=preview)

    api_config = config.get("api", {})
    model = api_config.get("models", {}).get("classification", "claude-haiku-4-5-20251001")
    max_retries = api_config.get("max_retries", 3)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            # Extract JSON from response (handle markdown code blocks)
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            classification = json.loads(text)
            return classification

        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(api_config.get("rate_limit_delay_seconds", 0.5))
        except Exception as e:
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classify forum threads via Claude API"
    )
    parser.add_argument("--input", type=Path, required=True, help="Clean threads directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)

    # Initialize Anthropic client
    import anthropic
    client = anthropic.Anthropic()

    args.output.mkdir(parents=True, exist_ok=True)

    thread_files = sorted(args.input.glob("*.json"))
    logger.info(f"Classifying {len(thread_files)} threads")

    api_config = config.get("api", {})
    delay = api_config.get("rate_limit_delay_seconds", 0.5)

    passed = 0
    failed = 0
    skipped = 0
    errors = 0

    for i, thread_file in enumerate(thread_files):
        output_file = args.output / thread_file.name
        skip_file = args.output / f"{thread_file.stem}.skip"

        # Idempotent: skip if already classified or previously rejected
        if output_file.exists() or skip_file.exists():
            skipped += 1
            continue

        with open(thread_file) as f:
            thread = json.load(f)

        classification = classify_thread(thread, config, client)
        if classification is None:
            errors += 1
            logger.warning(f"Failed to classify: {thread.get('thread_title', thread_file.name)}")
            continue

        thread["classification"] = classification
        thread["classification"]["timestamp"] = datetime.now().isoformat()

        if passes_gate(classification, config):
            with open(output_file, "w") as f:
                json.dump(thread, f, indent=2)
            passed += 1
        else:
            # Write rejection marker for idempotency (avoids re-classifying)
            skip_file.write_text(
                f"rejected:{classification.get('technical_depth', 'low')},"
                f"s14={classification.get('s14_specific', False)}\n"
            )
            failed += 1

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(thread_files)} "
                         f"(passed={passed}, filtered={failed}, errors={errors})")

        time.sleep(delay)

    logger.info("")
    logger.info("=== Classification Summary ===")
    logger.info(f"Total: {len(thread_files)}")
    logger.info(f"Passed gate: {passed}")
    logger.info(f"Filtered out: {failed}")
    logger.info(f"Skipped (existing): {skipped}")
    logger.info(f"Errors: {errors}")


if __name__ == "__main__":
    main()
