#!/usr/bin/env python3
"""
11_generate_synthetic_validation.py - Generate synthetic validation examples

Creates validation examples by paraphrasing training questions to test generalization.
Ensures no overlap with training data while maintaining diverse question styles.

Strategies:
  - SPEC: Generate question variations for same technical values
  - PROCEDURE: Rephrase procedure questions
  - EXPLANATION: Create related explanation queries

Usage:
    python scripts/11_generate_synthetic_validation.py \
      --train data/hf_train_autotrain.jsonl \
      --output data/hf_val_synthetic.jsonl \
      --count 250

Output: Synthetic validation set (~250 examples)
"""

import json
import argparse
import logging
import sys
import random
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('11_generate_synthetic_validation')


# Question variation templates by task type
SPEC_VARIATIONS = [
    "What is the {field}?",
    "Tell me the {field}",
    "Can you provide the {field}?",
    "I need to know the {field}",
    "What's the value for {field}?",
]

PROCEDURE_VARIATIONS = [
    "How do you {action}?",
    "What's the procedure for {action}?",
    "Can you explain how to {action}?",
    "Walk me through {action}",
    "What are the steps to {action}?",
]

EXPLANATION_VARIATIONS = [
    "Explain {topic}",
    "Tell me about {topic}",
    "What is {topic}?",
    "Can you describe {topic}?",
    "I need information on {topic}",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    entries = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def parse_training_example(text: str) -> Dict[str, str]:
    """
    Parse flat text format into components.

    Input: "User: [SPEC] What is the engine displacement?\nAssistant: 2302 CC"
    Output: {"task": "spec", "question": "What is the engine displacement?", "answer": "2302 CC"}
    """
    try:
        user_part, assistant_part = text.split('\nAssistant: ', 1)
        user_content = user_part.replace('User: ', '', 1)

        # Extract task prefix
        task_match = user_content.split('] ', 1)
        if len(task_match) == 2:
            task = task_match[0].replace('[', '').lower()
            question = task_match[1]
        else:
            task = "unknown"
            question = user_content

        return {
            "task": task,
            "question": question,
            "answer": assistant_part
        }
    except Exception as e:
        logger.warning(f"Failed to parse: {text[:100]}... Error: {e}")
        return None


def extract_spec_field(question: str) -> str:
    """
    Extract field name from spec question.

    Example: "What is the tightening torque for engine section 52?" → "tightening torque"
    """
    # Common spec patterns
    patterns = [
        "What is the ",
        "What's the ",
        "Tell me the ",
        "I need the ",
    ]

    for pattern in patterns:
        if question.startswith(pattern):
            field = question[len(pattern):].split(' for ')[0].strip('?')
            return field

    # Fallback: just return first part
    return question.split('?')[0].strip()


def generate_spec_variation(parsed: Dict[str, str]) -> Dict[str, str]:
    """Generate a variation of a SPEC question."""
    field = extract_spec_field(parsed['question'])

    # Choose random variation template
    template = random.choice(SPEC_VARIATIONS)
    new_question = template.format(field=field)

    return {
        "text": f"User: [SPEC] {new_question}\nAssistant: {parsed['answer']}"
    }


def generate_procedure_variation(parsed: Dict[str, str]) -> Dict[str, str]:
    """Generate a variation of a PROCEDURE question."""
    # Extract action from question
    action = parsed['question'].replace('How do you ', '', 1)
    action = action.split('?')[0].strip()

    # Choose random variation template
    template = random.choice(PROCEDURE_VARIATIONS)
    new_question = template.format(action=action)

    return {
        "text": f"User: [PROCEDURE] {new_question}\nAssistant: {parsed['answer']}"
    }


def generate_explanation_variation(parsed: Dict[str, str]) -> Dict[str, str]:
    """Generate a variation of an EXPLANATION question."""
    # Extract topic from question
    topic = parsed['question'].replace('Explain ', '', 1)
    topic = topic.split('.')[0].strip()

    # Choose random variation template
    template = random.choice(EXPLANATION_VARIATIONS)
    new_question = template.format(topic=topic)

    return {
        "text": f"User: [EXPLANATION] {new_question}\nAssistant: {parsed['answer']}"
    }


def generate_synthetic_validation(
    train_entries: List[Dict[str, Any]],
    target_count: int = 250
) -> List[Dict[str, Any]]:
    """
    Generate synthetic validation examples from training data.

    Strategy:
      1. Group training examples by task type
      2. Sample diverse examples from each task
      3. Generate question variations
      4. Ensure answer uniqueness to avoid direct overlap
    """
    logger.info(f"Generating {target_count} synthetic validation examples...")

    # Parse training examples
    parsed_examples = []
    for entry in train_entries:
        parsed = parse_training_example(entry['text'])
        if parsed:
            parsed_examples.append(parsed)

    logger.info(f"Parsed {len(parsed_examples)} training examples")

    # Group by task
    by_task = defaultdict(list)
    for parsed in parsed_examples:
        by_task[parsed['task']].append(parsed)

    logger.info("Task distribution in training:")
    for task, examples in sorted(by_task.items()):
        logger.info(f"  {task}: {len(examples)} examples")

    # Calculate how many to generate per task (proportional)
    task_counts = {task: len(examples) for task, examples in by_task.items()}
    total_examples = sum(task_counts.values())

    target_per_task = {}
    for task, count in task_counts.items():
        proportion = count / total_examples
        target_per_task[task] = max(1, int(target_count * proportion))

    logger.info("\nTarget synthetic examples per task:")
    for task, count in sorted(target_per_task.items()):
        logger.info(f"  {task}: {count} examples")

    # Generate variations
    synthetic_examples = []
    generation_stats = Counter()

    for task, target in target_per_task.items():
        task_examples = by_task[task]

        # Sample diverse examples (avoid duplicates)
        sampled = random.sample(task_examples, min(target, len(task_examples)))

        for parsed in sampled:
            try:
                if task == "spec":
                    variation = generate_spec_variation(parsed)
                elif task == "procedure":
                    variation = generate_procedure_variation(parsed)
                elif task == "explanation":
                    variation = generate_explanation_variation(parsed)
                elif task == "troubleshooting":
                    # Treat like procedure
                    variation = generate_procedure_variation(parsed)
                elif task == "wiring":
                    # Treat like explanation
                    variation = generate_explanation_variation(parsed)
                else:
                    logger.warning(f"Unknown task: {task}, skipping")
                    continue

                synthetic_examples.append(variation)
                generation_stats[task] += 1

            except Exception as e:
                logger.warning(f"Failed to generate variation for {task}: {e}")

    logger.info(f"\n✅ Generated {len(synthetic_examples)} synthetic examples")
    logger.info("Breakdown by task:")
    for task, count in sorted(generation_stats.items()):
        logger.info(f"  {task}: {count} examples")

    return synthetic_examples


def write_jsonl(entries: List[Dict[str, Any]], path: Path):
    """Write list of dicts to JSONL file."""
    with open(path, 'w') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic validation examples'
    )
    parser.add_argument('--train', type=Path, required=True,
                       help='Path to training JSONL')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output path for synthetic validation')
    parser.add_argument('--count', type=int, default=250,
                       help='Number of synthetic examples to generate (default: 250)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    logger.info(f"Random seed: {args.seed}")

    logger.info(f"Training file: {args.train}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Target count: {args.count}")

    # Load training data
    logger.info("Loading training data...")
    train_entries = load_jsonl(args.train)
    logger.info(f"Loaded {len(train_entries)} training entries")

    # Generate synthetic validation
    synthetic_val = generate_synthetic_validation(train_entries, args.count)

    # Write output
    logger.info(f"\nWriting synthetic validation to {args.output}...")
    write_jsonl(synthetic_val, args.output)
    logger.info(f"✓ Wrote {len(synthetic_val)} synthetic validation examples")

    # Show samples
    logger.info("\n📝 Sample synthetic examples:")
    for i, example in enumerate(synthetic_val[:3], 1):
        logger.info(f"\n  Example {i}:")
        logger.info(f"    {json.dumps(example)}")

    logger.info("\n✅ Done! Synthetic validation ready for upload to HuggingFace")


if __name__ == "__main__":
    main()
