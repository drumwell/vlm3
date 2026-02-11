#!/usr/bin/env python3
"""
Generate manual evaluation probes using Claude API.

For each curated image, sends it to Claude Vision to generate validated probe
questions across multiple evaluation categories. Outputs structured JSON
compatible with eval/modal_eval.py.

Usage:
    python eval/scripts/generate_probes.py
    python eval/scripts/generate_probes.py --images eval/probe_images.txt --output eval/benchmarks/manual_probes.json
"""

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Prompt
# =============================================================================

SYSTEM_PROMPT = """\
You are an evaluation engineer creating test probes for a BMW E30 M3 / 320is \
Vision-Language Model. The VLM was fine-tuned on scanned service manual pages \
and must answer mechanic questions accurately from the image alone.

Your job: look at the provided manual page image and generate 3-4 evaluation \
probes. Each probe tests a specific capability. Output ONLY a JSON array."""

USER_PROMPT = """\
Look at this BMW E30 M3 service manual image and generate 3-4 evaluation probes.

Generate probes from the following categories AS APPLICABLE to this image:

1. **specification** — Ask for a specific numeric value, torque, measurement, \
or tolerance visible on this page. Provide the exact expected answer.
2. **procedural** — Ask about a step sequence, prerequisite, or required tool \
shown on this page. Provide expected keywords.
3. **diagnostic** — Ask a symptom-cause or troubleshooting question IF the \
page contains diagnostic content. Provide expected keywords.
4. **safety** — Ask about a warning, caution, or critical precaution IF one \
is visible. Set is_critical to true.
5. **component** — Ask to identify a part, material, or assembly IF the \
page shows components. Provide expected keywords.
6. **wiring** — Ask about wire colors, connector pins, or circuit tracing IF \
this is an electrical diagram. Provide expected keywords.
7. **visual_grounding** — Ask a question that can ONLY be answered by looking \
at THIS specific image (e.g. layout, callout numbers, visible items). \
Provide expected keywords.
8. **hallucination_detection** — Ask about content that is NOT on this page. \
The correct model behavior is to decline. Set expected_behavior to a string \
containing "decline".
9. **out_of_distribution** — Ask about a different vehicle or era (e.g. \
2020 M3, Toyota). Limit to 1 per image, optional. Set expected_behavior to \
a string containing "decline".

Rules:
- For factual / specification probes: set "expected" to the exact answer AND/OR \
set "expected_keywords" to key terms.
- For hallucination / out_of_distribution probes: set "expected_behavior" to a \
string that contains the word "decline" (e.g. "decline - not shown in image").
- For safety probes: set "is_critical" to true.
- Every probe MUST have at least one of: "expected", "expected_keywords", or \
"expected_behavior".
- Do NOT reference "this image" or "this page" in the question text.

Output a JSON array where each element has this exact schema:
```json
{{
  "category": "<category>",
  "question": "<question text>",
  "expected": "<exact answer or omit>",
  "expected_keywords": ["<keyword>", "..."],
  "expected_behavior": "<decline string or omit>",
  "is_critical": false,
  "notes": "<why this probe matters>"
}}
```

Return ONLY the JSON array, no other text."""


# =============================================================================
# Image loading
# =============================================================================

def load_image_list(images_path: Path) -> list[str]:
    """Load image paths from a text file, skipping comments and blanks."""
    paths = []
    with open(images_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(line)
    return paths


def encode_image_base64(image_path: Path) -> tuple[str, str]:
    """Encode image to base64, return (data, media_type)."""
    suffix = image_path.suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }
    media_type = media_types.get(suffix, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


# =============================================================================
# Response parsing
# =============================================================================

def parse_json_response(text: str) -> list[dict]:
    """Extract JSON array from Claude response, handling markdown wrapping."""
    text = text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            elif line.startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        text = "\n".join(json_lines)

    # Find array bounds
    if not text.startswith("["):
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1:
            text = text[start : end + 1]

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error: {e}")

    return []


# =============================================================================
# API call
# =============================================================================

def generate_probes_for_image(
    image_path: Path,
    api_key: str,
    model: str,
    max_retries: int = 3,
) -> list[dict]:
    """Call Claude API with image to generate probes. Returns list of probe dicts."""
    from anthropic import Anthropic

    base64_data, media_type = encode_image_base64(image_path)
    client = Anthropic(api_key=api_key)

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": base64_data,
                                },
                            },
                            {
                                "type": "text",
                                "text": USER_PROMPT,
                            },
                        ],
                    }
                ],
            )

            response_text = response.content[0].text if response.content else ""
            probes = parse_json_response(response_text)

            tokens_in = response.usage.input_tokens
            tokens_out = response.usage.output_tokens
            logger.info(
                f"  Got {len(probes)} probes "
                f"(tokens: {tokens_in} in / {tokens_out} out)"
            )
            return probes

        except Exception as e:
            wait = (2 ** attempt)
            logger.warning(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"  Retrying in {wait}s...")
                time.sleep(wait)

    logger.error(f"  All {max_retries} attempts failed for {image_path.name}")
    return []


# =============================================================================
# Validation
# =============================================================================

def validate_probe(probe: dict) -> list[str]:
    """Return list of validation errors (empty = valid)."""
    errors = []

    if not probe.get("question"):
        errors.append("missing question")
    if not probe.get("category"):
        errors.append("missing category")

    has_eval_field = (
        probe.get("expected")
        or probe.get("expected_keywords")
        or probe.get("expected_behavior")
    )
    if not has_eval_field:
        errors.append("no evaluation field (expected/expected_keywords/expected_behavior)")

    return errors


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate manual evaluation probes using Claude API"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("eval/probe_images.txt"),
        help="Text file listing image paths (default: eval/probe_images.txt)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/src/manual/prepared"),
        help="Base directory for image paths (default: data/src/manual/prepared)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/benchmarks/manual_probes.json"),
        help="Output JSON file (default: eval/benchmarks/manual_probes.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY not set")
        return 1

    # Load image list
    if not args.images.exists():
        logger.error(f"Image list not found: {args.images}")
        return 1
    image_paths = load_image_list(args.images)
    logger.info(f"Loaded {len(image_paths)} image paths from {args.images}")

    # Validate all images exist
    missing = []
    for rel_path in image_paths:
        full_path = args.image_dir / rel_path
        if not full_path.exists():
            missing.append(rel_path)
    if missing:
        logger.error(f"{len(missing)} images not found:")
        for m in missing:
            logger.error(f"  {m}")
        return 1
    logger.info("All image paths verified")

    # Generate probes
    all_probes = []
    counters = {}  # category -> count, for ID assignment
    errors_by_image = {}

    for i, rel_path in enumerate(image_paths, 1):
        full_path = args.image_dir / rel_path
        logger.info(f"[{i}/{len(image_paths)}] Processing {rel_path}")

        probes = generate_probes_for_image(full_path, api_key, args.model)

        if not probes:
            errors_by_image[rel_path] = "no probes returned"
            continue

        for probe in probes:
            # Validate
            errs = validate_probe(probe)
            if errs:
                logger.warning(f"  Skipping invalid probe: {errs}")
                continue

            # Assign ID
            cat = probe["category"]
            counters[cat] = counters.get(cat, 0) + 1
            probe["id"] = f"{cat}_{counters[cat]:03d}"

            # Attach image path
            probe["image"] = rel_path

            # Ensure boolean default
            if "is_critical" not in probe:
                probe["is_critical"] = False

            all_probes.append(probe)

        # Rate limiting
        if i < len(image_paths):
            time.sleep(1)

    # Post-generation validation
    logger.info("")
    logger.info("=" * 60)
    logger.info("GENERATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Images processed: {len(image_paths)}")
    logger.info(f"Total probes: {len(all_probes)}")

    # Category distribution
    cat_counts = {}
    for p in all_probes:
        cat = p.get("category", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    logger.info("\nBy category:")
    for cat, count in sorted(cat_counts.items()):
        logger.info(f"  {cat}: {count}")

    critical_count = sum(1 for p in all_probes if p.get("is_critical"))
    logger.info(f"\nCritical probes: {critical_count}")

    if errors_by_image:
        logger.warning(f"\nImages with errors: {len(errors_by_image)}")
        for img, err in errors_by_image.items():
            logger.warning(f"  {img}: {err}")

    # Check for images that produced zero probes
    if errors_by_image:
        logger.error("Some images produced 0 probes")
        return 1

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_probes, f, indent=2)
    logger.info(f"\nWrote {len(all_probes)} probes to {args.output}")
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
