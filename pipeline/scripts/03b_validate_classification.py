#!/usr/bin/env python3
"""
Quick validation script to test classification prompt on sample images.
Run this before the full classification to validate the approach works.

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python scripts/03b_validate_classification.py --config config.yaml

Expected cost: ~$0.10-0.15 for 6 images
"""

import argparse
import anthropic
import base64
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import yaml

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Sample images with known/expected content types
SAMPLES = [
    ("data_src/e30-m3-320is/21 - Clutch/21-01.jpg", "procedure", "Photos with numbered callouts + steps"),
    ("data_src/e30-m3-320is/21 - Clutch/21-07.jpg", "troubleshooting", "3-column Condition/Cause/Correction table"),
    ("data_src/e30-m3-320is/11 - Engine/11-50.jpg", "diagram", "Full-page engine cutaway illustration"),
    ("data_src/e30-m3-320is/00 - Torque Specs/00-01.jpg", "specification", "Data table with values/units"),
    ("data_src/e30-m3-320is/1990 BMW M3 Electrical Troubleshooting Manual/0670-01.jpg", "fuse_chart", "Fuse data chart"),
    ("data_src/e30-m3-320is/34 - Brakes/34-10.jpg", "procedure", "Photos with steps"),
]

SYSTEM_PROMPT = """You are classifying pages from a BMW E30 M3 service manual.

Analyze the visual content and structure of the page to determine its type.

Valid content types:
- "index": Table of contents with repair codes, procedure names, and page references
- "procedure": Step-by-step instructions with multiple photos showing hands/tools, numbered callouts
- "specification": Data tables with values, units, torque specs, measurements
- "diagram": Full-page technical illustrations, exploded views, cutaway drawings (minimal text)
- "troubleshooting": Diagnostic table with columns for Condition/Symptom, Cause, and Correction
- "wiring": Electrical wiring diagrams, circuit schematics
- "fuse_chart": Fuse/relay tables listing circuits, amperage, and functions
- "pinout": Connector pin assignments, terminal diagrams
- "flowchart": Diagnostic decision flowcharts
- "text": Dense paragraphs of text, introductory content, reference material

Output ONLY valid JSON:
{
  "content_type": "procedure",
  "confidence": 0.92,
  "reasoning": "Brief explanation"
}"""

USER_PROMPT = """Classify this page. Look at the visual layout:
- Multiple photos with numbered callouts? → procedure
- 3-column table (Condition/Cause/Correction)? → troubleshooting
- Full-page technical drawing, minimal text? → diagram
- Data table with values and units? → specification
- Fuse/circuit table with amperage? → fuse_chart

Return JSON only."""


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def encode_image(image_path: Path) -> tuple[str, str]:
    """Encode image to base64 and determine media type."""
    suffix = image_path.suffix.lower()
    media_type = "image/jpeg" if suffix in [".jpg", ".jpeg"] else "image/png"

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return data, media_type


def classify_image(client: anthropic.Anthropic, image_path: Path, model: str) -> dict:
    """Classify a single image using Claude."""
    base64_data, media_type = encode_image(image_path)

    response = client.messages.create(
        model=model,
        max_tokens=256,
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

    # Parse response
    text = response.content[0].text
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{[^}]+\}', text)
        if match:
            result = json.loads(match.group())
        else:
            result = {"content_type": "unknown", "confidence": 0, "reasoning": f"Failed to parse: {text}"}

    # Add token usage
    result["input_tokens"] = response.usage.input_tokens
    result["output_tokens"] = response.usage.output_tokens

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Validate classification prompt on sample images'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config.yaml'),
        help='Path to config.yaml'
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    classification_model = config['api']['models']['classification']

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    logger.info("=" * 70)
    logger.info("CLASSIFICATION VALIDATION TEST")
    logger.info("=" * 70)
    logger.info(f"Using model: {classification_model}")
    logger.info("")

    total_input_tokens = 0
    total_output_tokens = 0
    correct = 0
    tested = 0

    for image_path, expected_type, description in SAMPLES:
        path = Path(image_path)
        if not path.exists():
            logger.info(f"SKIP: {image_path} (file not found)")
            continue

        tested += 1
        logger.info(f"Image: {image_path}")
        logger.info(f"Expected: {expected_type} ({description})")

        result = classify_image(client, path, classification_model)

        actual_type = result.get("content_type", "unknown")
        confidence = result.get("confidence", 0)
        reasoning = result.get("reasoning", "")

        match = "✓" if actual_type == expected_type else "✗"
        if actual_type == expected_type:
            correct += 1

        logger.info(f"Got: {actual_type} (confidence: {confidence:.2f}) {match}")
        logger.info(f"Reasoning: {reasoning}")
        logger.info(f"Tokens: {result['input_tokens']} in, {result['output_tokens']} out")
        logger.info("-" * 70)

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

    # Cost estimate (Haiku pricing)
    input_cost = (total_input_tokens / 1_000_000) * 0.25
    output_cost = (total_output_tokens / 1_000_000) * 1.25
    total_cost = input_cost + output_cost

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    if tested > 0:
        logger.info(f"Accuracy: {correct}/{tested} ({100*correct/tested:.0f}%)")
    else:
        logger.info("No samples tested (files not found)")
    logger.info(f"Total tokens: {total_input_tokens} input, {total_output_tokens} output")
    logger.info(f"Estimated cost: ${total_cost:.4f}")
    logger.info("")

    if tested == 0:
        logger.warning("No samples tested - check that sample files exist")
    elif correct == tested:
        logger.info("✓ All classifications correct! Safe to proceed with full run.")
    elif correct >= tested - 1:
        logger.warning("⚠ Minor misclassification. Review above and decide if acceptable.")
    else:
        logger.error("✗ Multiple misclassifications. Review prompt before full run.")


if __name__ == "__main__":
    main()
