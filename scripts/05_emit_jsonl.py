#!/usr/bin/env python3
"""
05_emit_jsonl.py - Transform blocks into instruction-tuning JSONL.
Creates instruction/input/output/meta format for LoRA/QLoRA finetuning.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

rgx_step_good = re.compile(r'^\s*\d+[.)]\s+\S')  # e.g., "1. Text"
def is_valid_procedure(output: str) -> bool:
    if not output: return False
    lines = [l for l in output.splitlines() if l.strip()]
    return any(rgx_step_good.match(l) for l in lines)

def format_spec_entry(block):
    """
    Format spec block into instruction-tuning entry.
    instruction: "What is the {spec_name} for {topic}?"
    input: "" (or short excerpt)
    output: "{value} {units}" (value-only)
    """
    spec_name = block['value']['name']
    answer = block['value']['answer']
    
    # Topic from page context (simplified)
    topic = f"engine section {block['section_id']}"
    
    instruction = f"What is the {spec_name.lower()} for {topic}?"
    input_text = ""  # Empty for spec (pure value lookup)
    output_text = answer  # Value-only format
    
    meta = {
        'task': 'spec',
        'section_dir': block['section_dir'],
        'section_id': block['section_id'],
        'topic': spec_name,
        'source_path': block['source_path'],
        'page_no': block['page_no']
    }
    
    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text,
        'meta': meta
    }


def format_procedure_entry(block):
    """
    Format procedure block into instruction-tuning entry.
    instruction: "How do you {topic}?"
    input: {ocr_excerpt} (context)
    output: numbered steps, 1 sentence each
    """
    topic = block.get('topic', 'perform this procedure')[:100]
    steps = block.get('steps', [])
    ocr_excerpt = block.get('ocr_excerpt', '')[:200]
    
    instruction = f"How do you {topic.lower()}?"
    input_text = ocr_excerpt if ocr_excerpt else ""
    
    # Format steps as numbered lines
    output_lines = []
    for step in steps:
        step_num = step['step_num']
        step_text = step['text'].strip()
        # Ensure it's one sentence (take first sentence if multiple)
        first_sentence = step_text.split('.')[0].strip()
        if first_sentence:
            output_lines.append(f"{step_num}. {first_sentence}.")
    
    output_text = '\n'.join(output_lines) if output_lines else "No steps available."
    
    meta = {
        'task': 'procedure',
        'section_dir': block['section_dir'],
        'section_id': block['section_id'],
        'topic': topic,
        'source_path': block['source_path'],
        'page_no': block['page_no']
    }
    
    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text,
        'meta': meta
    }


def format_troubleshooting_entry(block):
    """
    Format troubleshooting block into instruction-tuning entry.
    instruction: "What checks should be performed for {symptom}?"
    input: {symptom} (optional)
    output: numbered checks
    """
    symptom = block.get('symptom', 'this issue')[:150]
    checks = block.get('checks', [])
    
    instruction = f"What checks should be performed for {symptom.lower()}?"
    input_text = ""  # Optional - symptom already in instruction
    
    # Format checks as numbered lines
    output_lines = []
    for check in checks:
        check_num = check['check_num']
        check_text = check['text'].strip()
        output_lines.append(f"{check_num}. {check_text}")
    
    output_text = '\n'.join(output_lines) if output_lines else "No checks available."
    
    meta = {
        'task': 'troubleshooting',
        'section_dir': block['section_dir'],
        'section_id': block['section_id'],
        'topic': symptom,
        'source_path': block['source_path'],
        'page_no': block['page_no']
    }
    
    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text,
        'meta': meta
    }


def format_explanation_entry(block):
    """
    Format explanation block into instruction-tuning entry.
    instruction: "Explain {topic}."
    input: ""
    output: 2-4 sentence paraphrase
    """
    explain_text = block.get('explain', '')
    
    # Extract topic from first few words
    topic_words = explain_text.split()[:8]
    topic = ' '.join(topic_words) if topic_words else 'this procedure'
    
    # Limit to 2-4 sentences
    sentences = re.split(r'[.!?]+', explain_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    output_sentences = sentences[:4]  # Max 4 sentences
    
    if len(output_sentences) < 2 and len(sentences) > 0:
        output_sentences = sentences[:2]  # At least 2 if available
    
    output_text = '. '.join(output_sentences)
    if output_text and not output_text.endswith('.'):
        output_text += '.'
    
    instruction = f"Explain {topic.lower()}."
    input_text = ""
    
    meta = {
        'task': 'explanation',
        'section_dir': block['section_dir'],
        'section_id': block['section_id'],
        'topic': topic,
        'source_path': block['source_path'],
        'page_no': block['page_no']
    }
    
    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text,
        'meta': meta
    }


def format_wiring_entry(block):
    """
    Format wiring block into instruction-tuning entry.
    instruction: "What are the wiring details for {topic}?"
    input: ""
    output: key-value pairs
    """
    wiring_kv = block.get('wiring_kv', {})
    
    # Create topic from first key
    topic_key = list(wiring_kv.keys())[0] if wiring_kv else 'this component'
    topic = topic_key[:50]
    
    instruction = f"What are the wiring details for {topic.lower()}?"
    input_text = ""
    
    # Format as key: value pairs
    output_lines = [f"{k}: {v}" for k, v in wiring_kv.items()]
    output_text = '\n'.join(output_lines) if output_lines else "No wiring details available."
    
    meta = {
        'task': 'wiring',
        'section_dir': block['section_dir'],
        'section_id': block['section_id'],
        'topic': topic,
        'source_path': block['source_path'],
        'page_no': block['page_no']
    }
    
    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text,
        'meta': meta
    }


def transform_block(block):
    """Transform a block into instruction-tuning format based on task type."""
    task = block['task']

    if task == 'spec':
        return format_spec_entry(block)
    elif task == 'procedure':
        # Format first, then validate the output
        entry = format_procedure_entry(block)
        if entry and is_valid_procedure(entry['output']):
            return entry
        else:
            # Skip invalid procedure (no valid steps)
            return None
    elif task == 'troubleshooting':
        return format_troubleshooting_entry(block)
    elif task == 'explanation':
        return format_explanation_entry(block)
    elif task == 'wiring':
        return format_wiring_entry(block)
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Transform blocks to instruction-tuning JSONL")
    parser.add_argument('--blocks-dir', default='work/blocks',
                        help='Input blocks directory')
    parser.add_argument('--section-filter', default='.*',
                        help='Regex to filter by section_id in blocks')
    parser.add_argument('--output-dir', default='data',
                        help='Output directory for JSONL files')
    parser.add_argument('--output-prefix', default='slice',
                        help='Prefix for output files (e.g., "slice" -> spec.slice.jsonl)')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    blocks_dir = project_root / args.blocks_dir
    output_dir = project_root / args.output_dir
    
    # Compile section filter
    section_pattern = re.compile(args.section_filter)
    
    print(f"[05_emit_jsonl] Blocks directory: {blocks_dir}")
    print(f"[05_emit_jsonl] Section filter: {args.section_filter}")
    print(f"[05_emit_jsonl] Output directory: {output_dir}")
    print(f"[05_emit_jsonl] Output prefix: {args.output_prefix}")
    
    # Load all blocks
    block_files = sorted(blocks_dir.glob('*.json'))
    print(f"[05_emit_jsonl] Found {len(block_files)} block files")
    
    # Group entries by task
    entries_by_task = defaultdict(list)
    skipped_filter = 0
    skipped_invalid = 0

    for block_file in block_files:
        with open(block_file) as f:
            block = json.load(f)

        # Check section filter
        section_id = block.get('section_id', '')
        if not section_pattern.match(section_id):
            skipped_filter += 1
            continue

        # Transform block
        entry = transform_block(block)
        if entry:
            entries_by_task[entry['meta']['task']].append(entry)
        elif block.get('task') == 'procedure':
            # Track skipped invalid procedures
            skipped_invalid += 1
            print(f"[05_emit_jsonl] Skipped invalid procedure from {block.get('page_no', '?')}")

    print(f"[05_emit_jsonl] Skipped {skipped_filter} blocks (section filter)")
    print(f"[05_emit_jsonl] Skipped {skipped_invalid} invalid procedures (no valid steps)")
    
    # Write JSONL files per task (for organization/debugging)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_counts = {}
    all_entries = []

    for task, entries in entries_by_task.items():
        output_file = output_dir / f"{task}.{args.output_prefix}.jsonl"

        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        task_counts[task] = len(entries)
        all_entries.extend(entries)
        print(f"[05_emit_jsonl] ✓ Wrote {len(entries)} entries to {output_file.name}")

    # Write consolidated file (all tasks combined)
    consolidated_file = output_dir / "dataset.jsonl"
    with open(consolidated_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"[05_emit_jsonl] ✓ Wrote {len(all_entries)} total entries to {consolidated_file.name}")
    
    # Summary
    print(f"\n[05_emit_jsonl] Summary:")
    print(f"  Total entries: {sum(task_counts.values())}")
    print(f"\n[05_emit_jsonl] Per-task counts:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task:20s}: {count}")
    
    # Show 2 examples per file (redacted)
    print(f"\n[05_emit_jsonl] Sample entries (2 per task, redacted):")
    for task in sorted(entries_by_task.keys()):
        entries = entries_by_task[task]
        print(f"\n  [{task}]:")
        for i, entry in enumerate(entries[:2], 1):
            instruction = entry['instruction'][:80] + '...' if len(entry['instruction']) > 80 else entry['instruction']
            output_preview = entry['output'][:100] + '...' if len(entry['output']) > 100 else entry['output']
            print(f"    {i}. instruction: {instruction}")
            print(f"       output: {output_preview}")
    
    print(f"\n[05_emit_jsonl] Done!")


if __name__ == '__main__':
    main()
