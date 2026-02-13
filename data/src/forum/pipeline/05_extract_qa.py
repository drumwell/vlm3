#!/usr/bin/env python3
"""
05_extract_qa.py - Extract Q&A pairs from scored threads via Claude API

Stage 5 of the forum pipeline: extracts structured Q&A pairs from qualifying
threads. Synthesizes answers from multiple community replies.

Usage:
    python pipeline/05_extract_qa.py \
        --input work/threads_scored \
        --output work/qa_raw \
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


# ============================================================================
# Thread Content
# ============================================================================

def build_full_thread(thread: Dict, max_chars: int) -> str:
    """Build full thread content for extraction."""
    parts = []
    title = thread.get("thread_title", "")
    parts.append(f"Thread: {title}")
    parts.append(f"Forum: {thread.get('forum', 'unknown')}")

    classification = thread.get("classification", {})
    parts.append(f"Type: {classification.get('thread_type', 'unknown')}")
    parts.append("")

    posts_used = 0
    for post in thread.get("posts", []):
        author = post.get("author", "unknown")
        num = post.get("post_number", "?")
        text = post.get("content_text", "")
        parts.append(f"Post #{num} by {author}:\n{text}")
        posts_used += 1

    content = "\n\n".join(parts)
    if len(content) > max_chars:
        content = content[:max_chars] + "\n[truncated]"

    return content, posts_used


# ============================================================================
# Claude API
# ============================================================================

EXTRACTION_PROMPT = """You are extracting Q&A training pairs from a BMW E30 M3 forum thread.

{content}

Extract 1-{max_qa} high-quality Q&A pairs from this thread. For each pair:

1. Identify the core question being asked or implied
2. Synthesize the best answer from ALL replies — do NOT just copy one reply
3. Resolve contradictions by favoring:
   - Posts with specific measurements/part numbers over vague claims
   - Posts corroborated by multiple authors over single opinions
   - Posts describing first-hand experience over speculation
4. Write questions as standalone (a reader should understand without seeing the thread)
5. Write answers as authoritative technical responses (not conversational forum style)
6. Flag any factual claims that should be verified against the BMW service manual

Return ONLY valid JSON:
{{
  "qa_pairs": [
    {{
      "question": "<standalone technical question>",
      "answer": "<synthesized authoritative answer>",
      "question_type": "<one of: troubleshooting, procedural, specification, diagnostic, modification, maintenance, parts, experience>",
      "cross_reference_needed": <true/false>,
      "cross_reference_note": "<what to verify, or null>"
    }}
  ]
}}"""


def extract_qa(
    thread: Dict,
    config: Dict,
    client,
) -> Optional[Dict]:
    """Extract Q&A pairs from a single thread."""
    ext_config = config.get("extraction", {})
    max_chars = ext_config.get("max_thread_chars", 8000)
    max_qa = ext_config.get("max_qa_per_thread", 5)

    content, posts_used = build_full_thread(thread, max_chars)
    prompt = EXTRACTION_PROMPT.format(content=content, max_qa=max_qa)

    api_config = config.get("api", {})
    model = api_config.get("models", {}).get("extraction", "claude-sonnet-4-5-20250929")
    max_retries = api_config.get("max_retries", 3)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            result = json.loads(text)

            # Assign IDs and build output document
            thread_id = thread.get("thread_id", "unknown")
            qa_pairs = result.get("qa_pairs", [])

            for idx, pair in enumerate(qa_pairs):
                pair["id"] = f"forum-{thread_id}-q{idx + 1:02d}"

            return {
                "thread_id": thread_id,
                "forum": thread.get("forum", "unknown"),
                "thread_title": thread.get("thread_title", ""),
                "source_type": "forum",
                "content_type": thread.get("classification", {}).get("thread_type", "unknown"),
                "quality_score": thread.get("scores", {}).get("composite", 0),
                "qa_pairs": qa_pairs,
                "extraction": {
                    "model": model,
                    "timestamp": datetime.now().isoformat(),
                    "thread_posts_used": posts_used,
                    "thread_posts_total": len(thread.get("posts", [])),
                },
            }

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
        description="Extract Q&A pairs from scored threads"
    )
    parser.add_argument("--input", type=Path, required=True, help="Scored threads directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for Q&A")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)

    import anthropic
    client = anthropic.Anthropic()

    args.output.mkdir(parents=True, exist_ok=True)

    thread_files = sorted(args.input.glob("*.json"))
    logger.info(f"Extracting Q&A from {len(thread_files)} scored threads")

    api_config = config.get("api", {})
    delay = api_config.get("rate_limit_delay_seconds", 0.5)

    total_qa = 0
    processed = 0
    skipped = 0
    errors = 0

    for i, thread_file in enumerate(thread_files):
        output_file = args.output / thread_file.name

        # Idempotent: skip if already extracted
        if output_file.exists():
            skipped += 1
            continue

        with open(thread_file) as f:
            thread = json.load(f)

        result = extract_qa(thread, config, client)
        if result is None:
            errors += 1
            logger.warning(f"Failed to extract: {thread.get('thread_title', thread_file.name)}")
            continue

        qa_count = len(result.get("qa_pairs", []))
        total_qa += qa_count
        processed += 1

        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        logger.debug(f"  {thread.get('thread_title', '')}: {qa_count} Q&A pairs")

        if (i + 1) % 25 == 0:
            logger.info(f"Progress: {i + 1}/{len(thread_files)} "
                         f"(processed={processed}, qa={total_qa}, errors={errors})")

        time.sleep(delay)

    logger.info("")
    logger.info("=== Extraction Summary ===")
    logger.info(f"Total threads: {len(thread_files)}")
    logger.info(f"Processed: {processed}")
    logger.info(f"Skipped (existing): {skipped}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Total Q&A pairs: {total_qa}")
    if processed > 0:
        logger.info(f"Mean Q&A per thread: {total_qa / processed:.1f}")


if __name__ == "__main__":
    main()
