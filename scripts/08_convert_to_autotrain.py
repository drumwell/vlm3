#!/usr/bin/env python3
"""
Convert dataset.jsonl to AutoTrain flat text format.
Reads instruction/output format and produces {"text": "User: [TASK] Q\nAssistant: A"}
"""

import json
from pathlib import Path

def main():
    dataset_path = Path("data/dataset.jsonl")
    output_path = Path("data/hf_train_autotrain.jsonl")

    if not dataset_path.exists():
        print(f"❌ Error: {dataset_path} not found!")
        print(f"   Run: make emit extract_html")
        return

    converted = []

    # Convert all examples to AutoTrain format
    for line in open(dataset_path):
        if line.strip():
            item = json.loads(line)
            task = item['meta'].get('task', 'unknown')
            instruction = item['instruction']
            output = item['output']
            text = f"User: [{task.upper()}] {instruction}\nAssistant: {output}"
            converted.append({"text": text})

    # Write training file
    output_path.write_text("\n".join(json.dumps(e) for e in converted))

    print(f"✅ Converted {len(converted)} examples to AutoTrain format")
    print(f"   File: {output_path} ({output_path.stat().st_size // 1024} KB)")
    print(f"\n📝 Next: Run 'make synthetic_val' to generate validation set")

if __name__ == '__main__':
    main()
