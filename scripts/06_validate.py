#!/usr/bin/env python3
"""
validate.py - Validate consolidated dataset against config rules.
Generates QA report with validation results and error counts.
"""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import yaml


def load_config(config_path):
    """Load config.yaml."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def count_tokens_approx(text):
    """Approximate token count (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def validate_spec_output(output, config):
    """
    Validate spec output matches value-only regex.
    Returns (is_valid: bool, error_msg: str)
    """
    spec_regex = config['validation']['spec_output_regex']
    pattern = re.compile(spec_regex)
    
    if pattern.match(output.strip()):
        return True, None
    else:
        return False, f"Output doesn't match value-only regex: {output[:50]}"


def validate_numbered_output(output, config):
    """
    Validate procedure/troubleshooting output has numbered lines.
    Each non-blank line must match '^[0-9]+[.)] <text>'.
    Returns (is_valid: bool, error_msg: str)
    """
    # Explicit pattern: digit(s) + period/paren + space + non-whitespace
    rgx_step_good = re.compile(r'^\s*\d+[.)]\s+\S')

    if not output or not output.strip():
        return False, "Procedure output is empty (require '1. <text>')"

    lines = output.splitlines()
    bad_lines = []
    for ln, l in enumerate(lines, 1):
        if l.strip() and not rgx_step_good.match(l):
            bad_lines.append(ln)

    if bad_lines:
        return False, (
            f"Invalid step lines at {bad_lines[:8]} "
            f"(each non-blank line must match '^[0-9]+[.)] <text>')"
        )

    return True, None


def validate_token_count(output, config):
    """
    Validate output is within token limit.
    Returns (is_valid: bool, error_msg: str, token_count: int)
    """
    max_tokens = config['validation']['max_output_tokens']
    token_count = count_tokens_approx(output)
    
    if token_count <= max_tokens:
        return True, None, token_count
    else:
        return False, f"Exceeds {max_tokens} tokens: {token_count}", token_count


def validate_meta(meta):
    """
    Validate meta fields are present.
    Returns (is_valid: bool, error_msg: str)
    """
    required_fields = ['section_id', 'source_path', 'task']
    missing = [f for f in required_fields if f not in meta or not meta[f]]
    
    if missing:
        return False, f"Missing meta fields: {missing}"
    
    return True, None


def validate_entry(entry, config):
    """
    Validate a single JSONL entry.
    Returns dict of validation results.
    """
    results = {
        'critical_errors': [],
        'warnings': [],
        'token_count': 0
    }
    
    # Check required top-level fields
    required_fields = ['instruction', 'input', 'output', 'meta']
    for field in required_fields:
        if field not in entry:
            results['critical_errors'].append(f"Missing field: {field}")
            return results
    
    task = entry['meta'].get('task', 'unknown')
    output = entry['output']
    
    # Validate meta
    meta_valid, meta_err = validate_meta(entry['meta'])
    if not meta_valid:
        results['critical_errors'].append(f"Meta: {meta_err}")
    
    # Validate token count
    token_valid, token_err, token_count = validate_token_count(output, config)
    results['token_count'] = token_count
    if not token_valid:
        results['warnings'].append(f"Token count: {token_err}")
    
    # Task-specific validation
    if task == 'spec':
        spec_valid, spec_err = validate_spec_output(output, config)
        if not spec_valid:
            results['critical_errors'].append(f"Spec format: {spec_err}")
    
    elif task in ['procedure', 'troubleshooting']:
        numbered_valid, numbered_err = validate_numbered_output(output, config)
        if not numbered_valid:
            results['critical_errors'].append(f"Numbering: {numbered_err}")
    
    return results


def generate_qa_report(validation_results, output_path):
    """Generate markdown QA report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Aggregate stats
    total_entries = sum(len(entries) for entries in validation_results.values())
    total_critical = sum(
        sum(1 for e in entries if e['results']['critical_errors'])
        for entries in validation_results.values()
    )
    total_warnings = sum(
        sum(len(e['results']['warnings']) for e in entries)
        for entries in validation_results.values()
    )
    
    # Start report
    lines = [
        "# QA Validation Report",
        f"**Generated**: {timestamp}",
        "",
        "## Summary",
        "",
        f"- **Total entries validated**: {total_entries}",
        f"- **Critical errors**: {total_critical}",
        f"- **Warnings**: {total_warnings}",
        "",
    ]
    
    # Per-task breakdown
    lines.append("## Per-Task Results")
    lines.append("")
    
    for task, entries in sorted(validation_results.items()):
        task_critical = sum(1 for e in entries if e['results']['critical_errors'])
        task_warnings = sum(len(e['results']['warnings']) for e in entries)
        avg_tokens = sum(e['results']['token_count'] for e in entries) / len(entries) if entries else 0
        
        lines.append(f"### {task.capitalize()}")
        lines.append(f"- Entries: {len(entries)}")
        lines.append(f"- Critical errors: {task_critical}")
        lines.append(f"- Warnings: {task_warnings}")
        lines.append(f"- Avg tokens/output: {avg_tokens:.1f}")
        lines.append("")
    
    # Critical errors detail
    if total_critical > 0:
        lines.append("## Critical Errors Detail")
        lines.append("")
        
        for task, entries in sorted(validation_results.items()):
            errors_in_task = [e for e in entries if e['results']['critical_errors']]
            if errors_in_task:
                lines.append(f"### {task.capitalize()}")
                for i, entry_data in enumerate(errors_in_task[:5], 1):
                    lines.append(f"{i}. Entry {entry_data['index']}:")
                    for err in entry_data['results']['critical_errors']:
                        lines.append(f"   - {err}")
                lines.append("")
    
    # Warnings detail (top 10)
    if total_warnings > 0:
        lines.append("## Warnings (Top 10)")
        lines.append("")
        
        all_warnings = []
        for task, entries in validation_results.items():
            for entry_data in entries:
                if entry_data['results']['warnings']:
                    all_warnings.append({
                        'task': task,
                        'index': entry_data['index'],
                        'warnings': entry_data['results']['warnings']
                    })
        
        for i, warn_data in enumerate(all_warnings[:10], 1):
            lines.append(f"{i}. [{warn_data['task']}] Entry {warn_data['index']}:")
            for w in warn_data['warnings']:
                lines.append(f"   - {w}")
        lines.append("")
    
    # Acceptance status
    lines.append("## Acceptance Status")
    lines.append("")
    if total_critical == 0:
        lines.append("✅ **PASS**: 0 critical errors")
    else:
        lines.append(f"❌ **FAIL**: {total_critical} critical error(s) found")
        lines.append("")
        lines.append("### Required Fixes:")
        lines.append("- Review critical errors above")
        lines.append("- Fix formatting issues in JSONL entries")
        lines.append("- Re-run validation")
    
    lines.append("")
    
    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return total_critical, total_warnings


def main():
    parser = argparse.ArgumentParser(description="Validate JSONL entries and generate QA report")
    parser.add_argument('--data-dir', default='data',
                        help='Directory containing JSONL files')
    parser.add_argument('--file', default='dataset.jsonl',
                        help='JSONL file to validate (default: dataset.jsonl)')
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config.yaml')
    parser.add_argument('--output', default='work/logs/qa_report.md',
                        help='Output QA report path')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    config_path = project_root / args.config
    output_path = project_root / args.output
    
    # Load config
    config = load_config(config_path)

    # Validate consolidated dataset file
    dataset_file = data_dir / args.file

    print(f"[validate] Validating file: {dataset_file}")
    print(f"[validate] Output: {output_path}")

    if not dataset_file.exists():
        print(f"[validate] ❌ File not found: {dataset_file}")
        return

    # Validate all entries
    validation_results = defaultdict(list)

    print(f"\n[validate] Reading and validating entries...")

    with open(dataset_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        if not line.strip():
            continue

        try:
            entry = json.loads(line)
            task = entry['meta'].get('task', 'unknown')
            results = validate_entry(entry, config)

            validation_results[task].append({
                'index': i,
                'entry': entry,
                'results': results
            })

            # Print progress every 100 entries
            if i % 100 == 0:
                print(f"  Validated {i} entries...")

        except json.JSONDecodeError as e:
            print(f"  ✗ Entry {i}: JSON parse error - {e}")
            validation_results['unknown'].append({
                'index': i,
                'entry': None,
                'results': {
                    'critical_errors': [f"JSON parse error: {e}"],
                    'warnings': [],
                    'token_count': 0
                }
            })

    # Generate report
    print(f"\n[validate] Generating QA report...")
    total_critical, total_warnings = generate_qa_report(validation_results, output_path)

    print(f"\n[validate] Summary:")
    print(f"  Total entries: {sum(len(e) for e in validation_results.values())}")
    print(f"  Critical errors: {total_critical}")
    print(f"  Warnings: {total_warnings}")
    print(f"\n[validate] ✓ QA report written to {output_path}")

    if total_critical == 0:
        print(f"[validate] ✅ PASS: 0 critical errors")
    else:
        print(f"[validate] ❌ FAIL: {total_critical} critical error(s)")

    print(f"\n[validate] Done!")


if __name__ == '__main__':
    main()
