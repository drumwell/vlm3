#!/usr/bin/env python3
"""
04_parse_blocks.py - Parse OCR data into atomic task blocks.
Combines text + tables, applies regex fixes, classifies by task type,
and extracts structured data per task type.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

import yaml


def load_config(config_path):
    """Load config.yaml."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def apply_regex_fixes(text, regex_fixes):
    """Apply regex replacements from config."""
    for fix in regex_fixes:
        pattern = fix['pattern']
        replace = fix['replace']
        text = re.sub(pattern, replace, text)
    return text


def canonicalize_unit(unit, unit_canon):
    """Canonicalize unit using config mappings."""
    unit_lower = unit.lower().strip()
    
    for canonical_type, canonical_unit in unit_canon.items():
        if canonical_type == 'torque' and unit_lower in ['nm', 'n-m', 'mm']:
            return canonical_unit
        elif canonical_type == 'pressure' and unit_lower in ['bar', 'psi']:
            return 'bar' if 'bar' in unit_lower else canonical_unit
        elif canonical_type == 'length' and unit_lower in ['mm', 'cm', 'm']:
            return 'mm' if 'mm' in unit_lower else canonical_unit
        elif canonical_type == 'volume' and unit_lower in ['l', 'litre', 'liter']:
            return canonical_unit
        elif canonical_type == 'temp' and ('°c' in unit_lower or 'celsius' in unit_lower):
            return canonical_unit
    
    return unit


def classify_task_type(text_full, ocr_data, table_rows, type_keywords):
    """
    Classify page task type based on keywords and content.
    Returns task type: spec, procedure, troubleshooting, wiring, explanation
    """
    text_lower = text_full.lower()
    
    # Check for tables -> likely spec
    if table_rows:
        return 'spec'
    
    # Check keywords for each type
    for task_type, keywords in type_keywords.items():
        if task_type == 'spec_table':
            continue  # Already checked tables
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                if task_type == 'spec_table':
                    return 'spec'
                return task_type
    
    # Check for numbered steps/procedures
    step_pattern = re.compile(r'^\s*\d+[\)\.]\s', re.MULTILINE)
    if step_pattern.search(text_full):
        return 'procedure'
    
    # Default to explanation
    return 'explanation'


def is_valid_spec(value_raw, unit_raw):
    """
    Validate that a spec entry has numeric content.
    Filters out text-only entries like "HWB" (supplier codes), "- intake", etc.
    """
    if not value_raw or not value_raw.strip():
        return False

    # Check if value contains at least one digit or numeric symbol
    has_numeric = re.search(r'[\d~≈><.,/\-]', value_raw)

    # Text-only values are not valid specs (e.g., "HWB", "- intake")
    if not has_numeric:
        return False

    # If unit is same as value (e.g., "HWB" copied to both), skip it
    if value_raw.strip() == unit_raw.strip():
        return False

    return True


def extract_spec_blocks(ocr_data, table_rows, config):
    """Extract spec value blocks from table rows."""
    blocks = []
    skipped = 0

    for row in table_rows:
        spec_name = row['spec_name']
        value_raw = row['value_raw']
        unit_raw = row['unit_raw']
        notes = row['notes']

        # === FILTER NON-NUMERIC SPECS ===
        if not is_valid_spec(value_raw, unit_raw):
            skipped += 1
            continue

        # Apply regex fixes
        value_clean = apply_regex_fixes(value_raw, config['regex_fixes'])
        unit_clean = canonicalize_unit(unit_raw, config['unit_canon'])

        block = {
            'task': 'spec',
            'value': {
                'name': spec_name,
                'answer': f"{value_clean} {unit_clean}".strip(),
                'units': unit_clean,
                'alt_units': unit_raw if unit_raw != unit_clean else None
            },
            'ocr_excerpt': f"{spec_name}: {value_raw} {unit_raw} {notes}".strip()
        }
        blocks.append(block)

    if skipped > 0:
        print(f"    [Filtered {skipped} non-numeric spec entries]")

    return blocks


def extract_procedure_blocks(ocr_data, config):
    """
    Extract procedure step blocks from numbered lists.
    Converts bullets to numbered steps, enforces 1..N numbering, merges continuation lines.
    """
    blocks = []
    text_full = ocr_data['text_full']

    # Find procedure title/topic from first few lines
    topic_lines = ocr_data['lines'][:5]
    topic = ' '.join([l['text'] for l in topic_lines if len(l['text']) > 10])[:100]

    # === ROBUST STEP EXTRACTION ===
    lines = [l.rstrip() for l in text_full.splitlines()]
    rgx_num = re.compile(r'^\s*(\d+)[.)]\s+(.+)')
    rgx_bul = re.compile(r'^\s*[-•]\s+(.+)')
    rgx_head = re.compile(r'^\s*(steps?|procedure)[:：]\s*', re.I)

    steps = []
    for l in lines:
        if not l.strip():
            continue
        if rgx_head.match(l):
            # Drop standalone headings like "Steps:" / "Procedure:"
            continue
        m = rgx_num.match(l)
        if m and m.group(2).strip():
            steps.append(m.group(2).strip())
            continue
        b = rgx_bul.match(l)
        if b:
            steps.append(b.group(1).strip())
            continue
        # Continuation lines: attach to previous step if any
        if steps:
            steps[-1] = (steps[-1] + " " + l.strip()).strip()

    # Fallback: if no numbered steps, check for action-verb sentences
    if not steps:
        action_verbs = ['remove', 'install', 'unscrew', 'check', 'adjust', 'replace',
                        'tighten', 'loosen', 'disconnect', 'connect', 'lift', 'pull',
                        'push', 'turn', 'align', 'lubricate', 'clean']

        text_lower = text_full.lower()
        has_action = any(verb in text_lower for verb in action_verbs)

        if has_action and len(text_full) > 100:
            # Split by common sentence/step separators
            potential_steps = []
            for line in lines:
                line_stripped = line.strip()
                if not line_stripped or len(line_stripped) < 10:
                    continue
                # Look for lines that start with action verbs or capitals
                if any(line_stripped.lower().startswith(verb) for verb in action_verbs):
                    potential_steps.append(line_stripped)

            # If we found action steps, use them
            if potential_steps:
                steps = potential_steps[:config['task_rules']['procedure']['max_steps']]

    if not steps:
        return []

    # Normalize to strict 1..N numbering
    normalized_steps = [{'step_num': i, 'text': s} for i, s in enumerate(steps, 1)]

    if len(normalized_steps) <= config['task_rules']['procedure']['max_steps']:
        block = {
            'task': 'procedure',
            'steps': normalized_steps,
            'topic': topic,
            'ocr_excerpt': text_full[:300]
        }
        blocks.append(block)

    return blocks


def extract_troubleshooting_blocks(ocr_data, config):
    """Extract troubleshooting check blocks."""
    blocks = []
    text_full = ocr_data['text_full']
    
    # Look for symptom/check patterns
    lines = ocr_data['lines']
    
    # Try to find symptom and checks structure
    symptom_line = None
    check_lines = []
    
    for line in lines:
        text = line['text'].strip()
        if 'symptom' in text.lower():
            symptom_line = text
        elif re.match(r'^\s*\d+[\)\.]', text):
            check_lines.append(text)
    
    if symptom_line and check_lines:
        checks = []
        for check_text in check_lines[:config['task_rules']['troubleshooting']['max_checks']]:
            match = re.match(r'^\s*(\d+)[\)\.]\s+(.+)', check_text)
            if match:
                checks.append({'check_num': int(match.group(1)), 'text': match.group(2).strip()})
        
        if checks:
            block = {
                'task': 'troubleshooting',
                'checks': checks,
                'symptom': symptom_line,
                'ocr_excerpt': text_full[:300]
            }
            blocks.append(block)
    
    return blocks


def extract_explanation_blocks(ocr_data, config):
    """Extract explanation/description blocks."""
    blocks = []
    text_full = ocr_data['text_full']
    
    # Get first few sentences (up to max_sentences)
    sentences = re.split(r'[.!?]+', text_full)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    
    max_sentences = config['task_rules']['explanation']['max_sentences']
    explain_text = '. '.join(sentences[:max_sentences])
    
    if explain_text:
        block = {
            'task': 'explanation',
            'explain': explain_text,
            'ocr_excerpt': text_full[:200]
        }
        blocks.append(block)
    
    return blocks


def extract_wiring_blocks(ocr_data, config):
    """Extract wiring diagram metadata blocks."""
    blocks = []
    text_full = ocr_data['text_full']
    
    # Simple key-value extraction for wiring diagrams
    lines = ocr_data['lines']
    wiring_kv = {}
    
    for line in lines[:20]:  # First 20 lines likely contain diagram info
        text = line['text'].strip()
        # Look for "Label: Value" patterns
        if ':' in text:
            parts = text.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                if len(key) < 50 and len(value) < 100:
                    wiring_kv[key] = value
    
    if wiring_kv:
        block = {
            'task': 'wiring',
            'wiring_kv': wiring_kv,
            'ocr_excerpt': text_full[:200]
        }
        blocks.append(block)
    
    return blocks


def parse_page(ocr_json_path, table_rows_for_page, config):
    """
    Parse a single page into blocks.
    Returns list of block dicts.
    """
    # Load OCR data
    with open(ocr_json_path) as f:
        ocr_data = json.load(f)
    
    text_full = ocr_data['text_full']
    
    # Skip diagram pages
    if ocr_data.get('is_diagram', False):
        return []
    
    # Classify task type
    task_type = classify_task_type(text_full, ocr_data, table_rows_for_page, config['type_keywords'])
    
    # Extract blocks based on task type
    blocks = []
    
    if task_type == 'spec' and table_rows_for_page:
        blocks = extract_spec_blocks(ocr_data, table_rows_for_page, config)
    elif task_type == 'procedure':
        blocks = extract_procedure_blocks(ocr_data, config)
    elif task_type == 'troubleshooting':
        blocks = extract_troubleshooting_blocks(ocr_data, config)
    elif task_type == 'wiring':
        blocks = extract_wiring_blocks(ocr_data, config)
    else:  # explanation
        blocks = extract_explanation_blocks(ocr_data, config)
    
    # Add common metadata to all blocks
    for block in blocks:
        block['section_dir'] = ocr_data['section_dir']
        block['section_id'] = ocr_data['section_id']
        block['source_path'] = ocr_data['source_path']
        block['page_no'] = ocr_data['page_no']
    
    return blocks


def main():
    parser = argparse.ArgumentParser(description="Parse OCR data into task blocks")
    parser.add_argument('--ocr-dir', default='work/ocr_raw',
                        help='OCR JSON directory')
    parser.add_argument('--tables-dir', default='work/ocr_tables',
                        help='Tables CSV directory')
    parser.add_argument('--section-filter', default='.*',
                        help='Regex to filter sections')
    parser.add_argument('--output-dir', default='work/blocks',
                        help='Output directory for block JSON files')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    ocr_dir = project_root / args.ocr_dir
    tables_dir = project_root / args.tables_dir
    output_dir = project_root / args.output_dir
    config_path = project_root / args.config
    
    # Load config
    config = load_config(config_path)
    
    # Compile section filter
    section_pattern = re.compile(args.section_filter)
    
    print(f"[04_parse_blocks] OCR directory: {ocr_dir}")
    print(f"[04_parse_blocks] Tables directory: {tables_dir}")
    print(f"[04_parse_blocks] Section filter: {args.section_filter}")
    print(f"[04_parse_blocks] Output directory: {output_dir}")
    
    # Load table data
    table_data = defaultdict(list)  # page_no -> [rows]
    
    for tables_csv in tables_dir.glob('*.csv'):
        section_id = tables_csv.stem
        with open(tables_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['section_id'] == section_id:
                    table_data[row['page_no']].append(row)
    
    print(f"[04_parse_blocks] Loaded {sum(len(v) for v in table_data.values())} table rows")
    
    # Find section directories
    section_dirs = [d for d in ocr_dir.iterdir() if d.is_dir()]
    
    all_blocks = []
    task_counts = defaultdict(int)
    block_counter = 0
    
    for section_dir in section_dirs:
        section_id = section_dir.name
        
        # Check filter (approximate match)
        json_files = list(section_dir.glob('*.json'))
        if not json_files:
            continue
        
        # Get section_dir from first JSON
        with open(json_files[0]) as f:
            sample = json.load(f)
            section_dir_name = sample['section_dir']
        
        if not section_pattern.match(section_dir_name):
            continue
        
        print(f"\n[04_parse_blocks] Processing section {section_id}...")
        
        for json_file in sorted(json_files):
            page_no = json_file.stem
            table_rows = table_data.get(page_no, [])
            
            try:
                blocks = parse_page(json_file, table_rows, config)
                
                if blocks:
                    # Write each block to separate file
                    for block in blocks:
                        block_counter += 1
                        block_filename = f"{section_id}-block{block_counter:03d}.json"
                        output_path = output_dir / block_filename
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(block, f, indent=2, ensure_ascii=False)
                        
                        all_blocks.append(block)
                        task_counts[block['task']] += 1
                    
                    print(f"  {page_no}: {len(blocks)} blocks ({', '.join(b['task'] for b in blocks)})")
                
            except Exception as e:
                print(f"  {page_no}: ✗ {str(e)}")
    
    # Summary
    print(f"\n[04_parse_blocks] Summary:")
    print(f"  Total blocks: {len(all_blocks)}")
    print(f"\n[04_parse_blocks] Per-task histogram:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task:20s}: {count}")
    
    # Sample blocks
    print(f"\n[04_parse_blocks] Sample blocks (first 5, truncated to 200 chars):")
    for i, block in enumerate(all_blocks[:5], 1):
        block_str = json.dumps(block, ensure_ascii=False)
        truncated = block_str[:200] + '...' if len(block_str) > 200 else block_str
        print(f"  {i}. [{block['task']}] {truncated}")
    
    print(f"\n[04_parse_blocks] Done!")


if __name__ == '__main__':
    main()
