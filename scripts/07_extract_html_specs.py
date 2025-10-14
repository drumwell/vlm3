#!/usr/bin/env python3
"""
Extract technical specifications from HTML files (M3-techspec.html, 320is-techspec.html).

This script parses structured HTML tables and creates JSONL training examples
for general vehicle specifications that were missing from the OCR pipeline.

Output: Appends to data/hf_train.jsonl and data/hf_val.jsonl (90/10 split)
"""

import json
import random
from pathlib import Path
from bs4 import BeautifulSoup
from typing import List, Dict

# Set random seed for reproducible train/val split
random.seed(42)

def parse_tech_spec_html(html_path: Path, model_name: str) -> List[Dict]:
    """
    Parse technical specification HTML file.

    Args:
        html_path: Path to HTML file
        model_name: "M3" or "320is" for labeling

    Returns:
        List of training examples in HF format
    """
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    examples = []
    table = soup.find('table', id='techTable')

    if not table:
        print(f"⚠️  No techTable found in {html_path}")
        return examples

    rows = table.find_all('tr')
    current_section = "General"

    for row in rows:
        cells = row.find_all('td')

        if len(cells) < 2:
            continue

        # Check if this is a section header
        header = row.find('h2')
        if header:
            current_section = header.text.strip()
            continue

        # Extract spec label and value
        label_cell = cells[0]
        value_cell = cells[1] if len(cells) > 1 else None

        label = label_cell.get_text(strip=True)

        if not value_cell:
            continue

        value_span = value_cell.find('span', class_='tableCellA')
        if not value_span:
            continue

        value = value_span.get_text(separator=' ', strip=True)

        # Skip empty values or headers
        if not value or value == '???' or len(label) < 3:
            continue

        # Create multiple question variations for better generalization
        variations = generate_question_variations(label, value, model_name, current_section)
        examples.extend(variations)

    return examples


def generate_question_variations(label: str, value: str, model: str, section: str) -> List[Dict]:
    """
    Generate multiple question phrasings for the same spec.
    This helps the model generalize to different question formats.
    """
    examples = []

    # Normalize label
    label_lower = label.lower()

    # Generate 2-3 variations per spec
    questions = []

    # Direct question
    questions.append(f"What is the {label_lower}?")
    questions.append(f"[SPEC] What is the {label_lower}?")

    # Model-specific question
    if model == "M3":
        questions.append(f"What is the {label_lower} for the E30 M3?")
        questions.append(f"What is the {label_lower} for the S14 engine?")
    elif model == "320is":
        questions.append(f"What is the {label_lower} for the 320is?")

    # Section-specific question
    if section:
        questions.append(f"What is the {label_lower} in the {section} section?")

    # Special variations for common specs
    if "displacement" in label_lower:
        questions.extend([
            "How many liters is the engine?",
            "What is the cubic capacity?",
            "[SPEC] What is the engine displacement?",
            "What is the engine size?"
        ])

    if "bore" in label_lower and "stroke" in label_lower:
        questions.extend([
            "What are the cylinder dimensions?",
            "[SPEC] What is the bore and stroke?",
            "What are the bore and stroke measurements?"
        ])

    if "compression" in label_lower:
        questions.extend([
            "What is the compression?",
            "[SPEC] What is the compression ratio?",
            "What is the engine compression ratio?"
        ])

    if "power" in label_lower or "bhp" in value.lower():
        questions.extend([
            "How much power does the engine make?",
            "[SPEC] What is the power output?",
            "What is the horsepower?",
            "How much HP does it have?"
        ])

    if "torque" in label_lower and "engine" in label_lower:
        questions.extend([
            "How much torque does the engine make?",
            "[SPEC] What is the max engine torque?",
            "What is the peak torque?"
        ])

    # Limit to 4-5 variations to avoid over-representation
    questions = list(set(questions))[:5]

    # Create training examples in instruction/output format
    for question in questions:
        # Remove [SPEC] prefix if present for instruction field
        instruction = question.replace("[SPEC] ", "")

        example = {
            "instruction": instruction,
            "input": "",
            "output": value,
            "meta": {
                "task": "spec",
                "source": f"{model}_techspec_html",
                "section": section,
                "topic": label,
                "original_label": label,
                "validation": {"valid": True, "errors": []},
                "token_count": len(question.split()) + len(value.split())
            }
        }
        examples.append(example)

    return examples


def main():
    """Extract specs from HTML files and append to existing datasets."""

    data_src = Path('data_src')
    data_dir = Path('data')

    # Parse both HTML files
    print("🔍 Parsing HTML tech spec files...")

    m3_html = data_src / 'M3-techspec.html'
    is320_html = data_src / '320is-techspec.html'

    all_examples = []

    if m3_html.exists():
        print(f"  📄 Parsing {m3_html}")
        m3_examples = parse_tech_spec_html(m3_html, "M3")
        print(f"     ✅ Extracted {len(m3_examples)} examples from M3 tech specs")
        all_examples.extend(m3_examples)
    else:
        print(f"  ⚠️  {m3_html} not found")

    if is320_html.exists():
        print(f"  📄 Parsing {is320_html}")
        is320_examples = parse_tech_spec_html(is320_html, "320is")
        print(f"     ✅ Extracted {len(is320_examples)} examples from 320is tech specs")
        all_examples.extend(is320_examples)
    else:
        print(f"  ⚠️  {is320_html} not found")

    if not all_examples:
        print("❌ No examples extracted. Check HTML files.")
        return

    # Append to consolidated dataset
    dataset_file = data_dir / 'dataset.jsonl'

    # Count existing examples
    existing_count = 0
    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            existing_count = sum(1 for _ in f)

    print(f"\n📝 Appending to consolidated dataset:")
    print(f"   Existing: {existing_count} examples")
    print(f"   Adding:   {len(all_examples)} HTML specs")
    print(f"   Total:    {existing_count + len(all_examples)} examples")

    # Append all HTML examples
    with open(dataset_file, 'a') as f:
        for example in all_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\n✅ Successfully appended HTML tech spec examples!")

    # Show sample
    print(f"\n📝 Sample extracted example:")
    sample = all_examples[0]
    print(f"   Question: {sample['instruction']}")
    print(f"   Answer: {sample['output']}")
    print(f"   Section: {sample['meta']['section']}")
    print(f"   Topic: {sample['meta']['topic']}")


if __name__ == '__main__':
    main()
