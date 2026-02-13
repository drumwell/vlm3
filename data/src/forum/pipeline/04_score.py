#!/usr/bin/env python3
"""
04_score.py - Quality scoring of classified threads via Claude API

Stage 4 of the forum pipeline: scores qualifying threads on 5 dimensions
that predict training value. Gates on composite score >= 3.0.

Usage:
    python pipeline/04_score.py \
        --input work/threads_classified \
        --output work/threads_scored \
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
# Thread Content Extraction
# ============================================================================

def build_thread_content(thread: Dict, max_chars: int) -> str:
    """Build full thread content for scoring, truncated to max_chars."""
    parts = []
    title = thread.get("thread_title", "")
    parts.append(f"Thread: {title}")
    parts.append(f"Forum: {thread.get('forum', 'unknown')}")
    parts.append(f"Type: {thread.get('classification', {}).get('thread_type', 'unknown')}")
    parts.append("")

    for post in thread.get("posts", []):
        author = post.get("author", "unknown")
        text = post.get("content_text", "")
        parts.append(f"[{author}]:\n{text}")

    content = "\n\n".join(parts)
    if len(content) > max_chars:
        content = content[:max_chars] + "\n[truncated]"

    return content


# ============================================================================
# Claude API
# ============================================================================

SCORING_PROMPT = """You are scoring forum threads from the s14.net BMW E30 M3 community for training data quality.

Rate this thread on 5 dimensions, each 1-5:

{content}

Return ONLY valid JSON:
{{
  "factual_specificity": <1-5, contains specific numbers, part numbers, measurements, tool names>,
  "answer_completeness": <1-5, thread reaches a clear resolution or complete answer>,
  "procedural_clarity": <1-5, steps described clearly enough to follow>,
  "s14_relevance": <1-5, specific to E30 M3 / S14, not generic BMW>,
  "unique_knowledge": <1-5, contains knowledge unlikely in the factory service manual>
}}

Scoring guide:
- 1: None/minimal
- 2: Some but vague
- 3: Moderate, useful but not detailed
- 4: Good, specific and actionable
- 5: Excellent, highly detailed and authoritative"""


def score_thread(
    thread: Dict,
    config: Dict,
    client,
) -> Optional[Dict]:
    """Score a single thread via Claude API."""
    scoring_config = config.get("scoring", {})
    max_chars = scoring_config.get("max_thread_chars", 4000)

    content = build_thread_content(thread, max_chars)
    prompt = SCORING_PROMPT.format(content=content)

    api_config = config.get("api", {})
    model = api_config.get("models", {}).get("scoring", "claude-sonnet-4-5-20250929")
    max_retries = api_config.get("max_retries", 3)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=256,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            scores = json.loads(text)

            # Compute weighted composite
            dimensions = scoring_config.get("dimensions", {
                "factual_specificity": 0.30,
                "answer_completeness": 0.25,
                "procedural_clarity": 0.20,
                "s14_relevance": 0.15,
                "unique_knowledge": 0.10,
            })

            composite = sum(
                scores.get(dim, 1) * weight
                for dim, weight in dimensions.items()
            )
            scores["composite"] = round(composite, 2)

            return scores

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
        description="Score classified threads via Claude API"
    )
    parser.add_argument("--input", type=Path, required=True, help="Classified threads directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
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
    logger.info(f"Scoring {len(thread_files)} classified threads")

    api_config = config.get("api", {})
    scoring_config = config.get("scoring", {})
    delay = api_config.get("rate_limit_delay_seconds", 0.5)
    min_score = scoring_config.get("min_composite_score", 3.0)

    passed = 0
    failed = 0
    skipped = 0
    errors = 0
    score_sum = 0.0

    for i, thread_file in enumerate(thread_files):
        output_file = args.output / thread_file.name
        skip_file = args.output / f"{thread_file.stem}.skip"

        # Idempotent: skip if already scored or previously rejected
        if output_file.exists() or skip_file.exists():
            skipped += 1
            continue

        with open(thread_file) as f:
            thread = json.load(f)

        scores = score_thread(thread, config, client)
        if scores is None:
            errors += 1
            logger.warning(f"Failed to score: {thread.get('thread_title', thread_file.name)}")
            continue

        thread["scores"] = scores
        thread["scores"]["timestamp"] = datetime.now().isoformat()
        composite = scores.get("composite", 0)
        score_sum += composite

        if composite >= min_score:
            with open(output_file, "w") as f:
                json.dump(thread, f, indent=2)
            passed += 1
        else:
            # Write rejection marker for idempotency (avoids re-scoring)
            skip_file.write_text(f"rejected:composite={composite:.2f}\n")
            failed += 1
            logger.debug(f"Below threshold ({composite:.2f}): {thread.get('thread_title', '')}")

        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i + 1}/{len(thread_files)} "
                         f"(passed={passed}, filtered={failed}, errors={errors})")

        time.sleep(delay)

    scored_total = passed + failed
    logger.info("")
    logger.info("=== Scoring Summary ===")
    logger.info(f"Total: {len(thread_files)}")
    logger.info(f"Passed (>= {min_score}): {passed}")
    logger.info(f"Filtered out: {failed}")
    logger.info(f"Skipped (existing): {skipped}")
    logger.info(f"Errors: {errors}")
    if scored_total > 0:
        logger.info(f"Mean composite score: {score_sum / scored_total:.2f}")


if __name__ == "__main__":
    main()
