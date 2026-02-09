#!/usr/bin/env python3
"""
05_filter_qa.py - Filter low-quality Q&A pairs

Stage 5 of the data pipeline: Quality control for Q&A pairs generated in Stage 4.
Filters out malformed, generic, or low-quality Q&A pairs before deduplication.

Usage:
    python scripts/05_filter_qa.py \
        --input work/qa_raw \
        --output work/qa_filtered \
        --log work/logs/qa_filtered_out.csv \
        --report work/logs/qa_filter_report.md \
        --config config.yaml \
        [--dry-run] [--verbose]
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class FilterConfig:
    """Configuration for Q&A filtering."""

    min_answer_length: int = 10
    max_answer_length: int = 500
    min_question_length: int = 15
    require_question_mark: bool = True
    max_question_similarity: float = 0.80
    min_question_types_per_page: int = 2
    generic_answer_patterns: List[str] = field(default_factory=list)
    self_referential_patterns: List[str] = field(default_factory=list)
    valid_question_types: Set[str] = field(default_factory=set)


def load_filter_config(config_path: Path) -> FilterConfig:
    """
    Load filter configuration from YAML file.

    Uses defaults for any missing values.

    Args:
        config_path: Path to config YAML file

    Returns:
        FilterConfig with loaded or default values
    """
    defaults = FilterConfig()

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return defaults

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    filters = config_data.get("filters", {})

    return FilterConfig(
        min_answer_length=filters.get("min_answer_length", defaults.min_answer_length),
        max_answer_length=filters.get("max_answer_length", defaults.max_answer_length),
        min_question_length=filters.get("min_question_length", defaults.min_question_length),
        require_question_mark=filters.get("require_question_mark", defaults.require_question_mark),
        max_question_similarity=filters.get("max_question_similarity", defaults.max_question_similarity),
        min_question_types_per_page=filters.get("min_question_types_per_page", defaults.min_question_types_per_page),
        generic_answer_patterns=filters.get("generic_answer_patterns", defaults.generic_answer_patterns),
        self_referential_patterns=filters.get("self_referential_patterns", defaults.self_referential_patterns),
        valid_question_types=set(filters.get("valid_question_types", defaults.valid_question_types)),
    )


# ============================================================================
# File I/O
# ============================================================================


def load_qa_files(input_dir: Path) -> List[Dict]:
    """
    Load all Q&A JSON files from input directory.

    Args:
        input_dir: Path to directory containing *.json files

    Returns:
        List of parsed Q&A documents (one per file)

    Raises:
        FileNotFoundError: If input_dir doesn't exist
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    docs = []

    # Only read top-level files (no recursion)
    for json_file in sorted(input_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                doc = json.load(f)
                docs.append(doc)
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping invalid JSON file {json_file}: {e}")
        except Exception as e:
            logger.warning(f"Error reading {json_file}: {e}")

    return docs


def write_filtered_output(qa_doc: Dict, output_path: Path) -> None:
    """
    Write filtered Q&A document to JSON file.

    Creates parent directories if needed.
    Pretty-prints JSON with indent=2.

    Args:
        qa_doc: Q&A document to write
        output_path: Path to output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(qa_doc, f, indent=2)


def write_rejection_log(rejections: List[Dict], log_path: Path) -> None:
    """
    Write CSV log of rejected Q&A pairs.

    Schema: timestamp,page_id,qa_id,question,answer,rejection_reason,filter_name

    Args:
        rejections: List of rejection records
        log_path: Path to CSV log file
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["timestamp", "page_id", "qa_id", "question", "answer", "rejection_reason", "filter_name"]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        timestamp = datetime.utcnow().isoformat() + "Z"

        for rejection in rejections:
            qa = rejection["qa"]
            writer.writerow({
                "timestamp": timestamp,
                "page_id": rejection.get("page_id", ""),
                "qa_id": qa.get("id", ""),
                "question": qa.get("question", ""),
                "answer": qa.get("answer", ""),
                "rejection_reason": rejection.get("reason", ""),
                "filter_name": rejection.get("filter_name", ""),
            })


def generate_filter_report(stats: Dict, report_path: Path) -> None:
    """
    Generate Markdown summary report.

    Args:
        stats: Dictionary with filtering statistics
        report_path: Path to output report file
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat() + "Z"

    total = stats.get("total_qa", 0)
    passed = stats.get("passed_qa", 0)
    rejected = stats.get("rejected_qa", 0)

    pass_pct = (passed / total * 100) if total > 0 else 0
    reject_pct = (rejected / total * 100) if total > 0 else 0

    lines = [
        "# Q&A Filter Report",
        "",
        f"**Generated**: {timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Files Processed | {stats.get('files_processed', 0)} |",
        f"| Total Q&A Pairs | {total:,} |",
        f"| Passed | {passed:,} ({pass_pct:.1f}%) |",
        f"| Rejected | {rejected:,} ({reject_pct:.1f}%) |",
        "",
    ]

    # Rejection reasons breakdown
    rejection_reasons = stats.get("rejection_reasons", Counter())
    if rejection_reasons:
        lines.extend([
            "## Rejection Reasons",
            "",
            "| Reason | Count | % of Rejected |",
            "|--------|-------|---------------|",
        ])

        for reason, count in rejection_reasons.most_common():
            pct = (count / rejected * 100) if rejected > 0 else 0
            lines.append(f"| {reason} | {count} | {pct:.1f}% |")

        lines.append("")

    # Warnings
    warnings = stats.get("warnings", [])
    if warnings:
        lines.extend([
            "## Warnings",
            "",
        ])
        for warning in warnings[:20]:  # Limit to 20 warnings
            lines.append(f"- {warning}")
        lines.append("")

    # Sample rejections
    sample_rejections = stats.get("sample_rejections", {})
    if sample_rejections:
        lines.extend([
            "## Sample Rejections",
            "",
        ])
        for reason, samples in sample_rejections.items():
            lines.append(f"### {reason} ({len(samples)} samples)")
            lines.append("")
            for i, sample in enumerate(samples[:3], 1):
                q = sample.get("question", "")[:50]
                a = sample.get("answer", "")[:50]
                lines.append(f"{i}. Q: \"{q}...\" A: \"{a}...\"")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Filter Functions
# ============================================================================


def filter_answer_length(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check answer length constraints.

    Returns:
        (True, None) if passed
        (False, "answer_too_short") if < min_answer_length
        (False, "answer_too_long") if > max_answer_length
    """
    answer = qa.get("answer", "").strip()
    length = len(answer)

    if length < config.min_answer_length:
        return False, "answer_too_short"

    if length > config.max_answer_length:
        return False, "answer_too_long"

    return True, None


def filter_question_length(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question length constraints.

    Returns:
        (True, None) if passed
        (False, "question_too_short") if < min_question_length
    """
    question = qa.get("question", "").strip()
    length = len(question)

    if length < config.min_question_length:
        return False, "question_too_short"

    return True, None


def filter_question_mark(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question ends with '?'.

    Returns:
        (True, None) if ends with '?' or require_question_mark=False
        (False, "missing_question_mark") otherwise
    """
    if not config.require_question_mark:
        return True, None

    question = qa.get("question", "").strip()

    if not question.endswith("?"):
        return False, "missing_question_mark"

    return True, None


def filter_generic_answer(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check answer doesn't contain generic/evasive patterns.

    Returns:
        (True, None) if no patterns matched
        (False, "generic_answer: <matched_pattern>") if pattern found
    """
    if not config.generic_answer_patterns:
        return True, None

    answer = qa.get("answer", "").lower()

    for pattern in config.generic_answer_patterns:
        if pattern.lower() in answer:
            return False, f"generic_answer: {pattern}"

    return True, None


def filter_self_referential(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question doesn't contain self-referential language.

    Returns:
        (True, None) if no patterns matched
        (False, "self_referential: <matched_pattern>") if pattern found
    """
    if not config.self_referential_patterns:
        return True, None

    question = qa.get("question", "").lower()

    for pattern in config.self_referential_patterns:
        if pattern.lower() in question:
            return False, f"self_referential: {pattern}"

    return True, None


def filter_question_type(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question_type is valid.

    Returns:
        (True, None) if question_type in valid_question_types or no validation
        (False, "invalid_question_type: <type>") if invalid
    """
    if not config.valid_question_types:
        return True, None

    question_type = qa.get("question_type")

    if question_type is None:
        # Missing type is allowed with warning
        logger.debug(f"Q&A missing question_type: {qa.get('id', 'unknown')}")
        return True, None

    if question_type not in config.valid_question_types:
        return False, f"invalid_question_type: {question_type}"

    return True, None


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate word-level Jaccard similarity between two texts."""
    words1 = set(re.findall(r'\w+', text1.lower()))
    words2 = set(re.findall(r'\w+', text2.lower()))

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


def filter_question_diversity(
    qa_pairs: List[Dict],
    config: FilterConfig
) -> List[Tuple[Dict, bool, Optional[str]]]:
    """
    Check for duplicate/similar questions within same page.

    Uses word-level Jaccard similarity.

    Returns:
        List of (qa, passed, reason) for each input qa
        First occurrence of similar pair passes; subsequent fail
    """
    if not qa_pairs:
        return []

    results = []
    seen_questions = []  # List of (normalized_question, original_qa)

    for qa in qa_pairs:
        question = qa.get("question", "").strip()

        # Check against all seen questions
        is_duplicate = False
        for seen_q, seen_qa in seen_questions:
            similarity = _jaccard_similarity(question, seen_q)
            if similarity >= config.max_question_similarity:
                is_duplicate = True
                results.append((qa, False, f"question_similarity: {similarity:.2f} with {seen_qa.get('id', 'unknown')}"))
                break

        if not is_duplicate:
            results.append((qa, True, None))
            seen_questions.append((question, qa))

    return results


# ============================================================================
# Main Processing
# ============================================================================


def apply_filters(
    qa_doc: Dict,
    config: FilterConfig
) -> Tuple[Dict, List[Dict]]:
    """
    Apply all filters to a Q&A document.

    Args:
        qa_doc: Full Q&A document with qa_pairs list
        config: Filter configuration

    Returns:
        (filtered_doc, rejected_list)
        - filtered_doc: Same schema, only passing qa_pairs
        - rejected_list: List of {qa, page_id, reason, filter_name}
    """
    page_id = qa_doc.get("page_id", "unknown")
    qa_pairs = qa_doc.get("qa_pairs", [])

    passed_pairs = []
    rejected_list = []

    # Individual filters (applied per Q&A pair)
    individual_filters = [
        ("answer_length", filter_answer_length),
        ("question_length", filter_question_length),
        ("question_mark", filter_question_mark),
        ("generic_answer", filter_generic_answer),
        ("self_referential", filter_self_referential),
        ("question_type", filter_question_type),
    ]

    # First pass: apply individual filters
    individually_passed = []
    for qa in qa_pairs:
        passed = True
        for filter_name, filter_func in individual_filters:
            ok, reason = filter_func(qa, config)
            if not ok:
                rejected_list.append({
                    "qa": qa,
                    "page_id": page_id,
                    "reason": reason,
                    "filter_name": filter_name,
                })
                passed = False
                break

        if passed:
            individually_passed.append(qa)

    # Second pass: apply diversity filter
    diversity_results = filter_question_diversity(individually_passed, config)
    for qa, ok, reason in diversity_results:
        if ok:
            passed_pairs.append(qa)
        else:
            rejected_list.append({
                "qa": qa,
                "page_id": page_id,
                "reason": reason,
                "filter_name": "question_diversity",
            })

    # Create filtered document (preserve all metadata)
    filtered_doc = {k: v for k, v in qa_doc.items() if k != "qa_pairs"}
    filtered_doc["qa_pairs"] = passed_pairs

    return filtered_doc, rejected_list


def compute_page_warnings(qa_doc: Dict, config: FilterConfig) -> List[str]:
    """
    Generate warnings for page-level quality issues.

    Warnings (non-blocking):
    - Fewer than min_question_types_per_page distinct types
    - All Q&A pairs filtered out
    - Very few Q&A pairs remaining (<3)

    Args:
        qa_doc: Q&A document to check
        config: Filter configuration

    Returns:
        List of warning strings
    """
    page_id = qa_doc.get("page_id", "unknown")
    qa_pairs = qa_doc.get("qa_pairs", [])
    warnings = []

    # Check if all filtered out
    if len(qa_pairs) == 0:
        warnings.append(f"{page_id}: All Q&A pairs filtered out")
        return warnings

    # Check if very few remaining
    if len(qa_pairs) < 3:
        warnings.append(f"{page_id}: Only {len(qa_pairs)} Q&A pairs remaining")

    # Check question type diversity
    question_types = set()
    for qa in qa_pairs:
        qt = qa.get("question_type")
        if qt:
            question_types.add(qt)

    if len(question_types) < config.min_question_types_per_page:
        types_str = ", ".join(sorted(question_types)) if question_types else "none"
        warnings.append(f"{page_id}: Only {len(question_types)} question type(s): {types_str}")

    return warnings


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Filter low-quality Q&A pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with qa_raw/*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for filtered files",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="CSV log of rejected Q&A",
    )
    parser.add_argument(
        "--report",
        type=Path,
        required=True,
        help="Markdown summary report",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Config YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be filtered without writing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each rejection",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    config = load_filter_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # Load all Q&A files
    try:
        qa_docs = load_qa_files(args.input)
        logger.info(f"Loaded {len(qa_docs)} Q&A files from {args.input}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not qa_docs:
        logger.warning("No Q&A files found, creating empty output")

    # Process each document
    all_rejections = []
    sample_rejections: Dict[str, List[Dict]] = {}
    stats = {
        "files_processed": 0,
        "total_qa": 0,
        "passed_qa": 0,
        "rejected_qa": 0,
        "rejection_reasons": Counter(),
        "warnings": [],
    }

    for qa_doc in qa_docs:
        filtered_doc, rejections = apply_filters(qa_doc, config)
        warnings = compute_page_warnings(filtered_doc, config)

        # Update stats
        stats["files_processed"] += 1
        stats["total_qa"] += len(qa_doc.get("qa_pairs", []))
        stats["passed_qa"] += len(filtered_doc.get("qa_pairs", []))
        stats["rejected_qa"] += len(rejections)

        for r in rejections:
            filter_name = r["filter_name"]
            stats["rejection_reasons"][filter_name] += 1

            # Collect samples for report
            if filter_name not in sample_rejections:
                sample_rejections[filter_name] = []
            if len(sample_rejections[filter_name]) < 5:
                sample_rejections[filter_name].append({
                    "question": r["qa"].get("question", ""),
                    "answer": r["qa"].get("answer", ""),
                    "reason": r["reason"],
                })

            if args.verbose:
                logger.debug(f"Rejected {r['qa'].get('id', 'unknown')}: {r['reason']}")

        stats["warnings"].extend(warnings)
        all_rejections.extend(rejections)

        # Write output (unless dry-run)
        if not args.dry_run:
            output_file = args.output / f"{qa_doc['page_id']}.json"
            write_filtered_output(filtered_doc, output_file)

    stats["sample_rejections"] = sample_rejections

    # Write logs and report
    if not args.dry_run:
        write_rejection_log(all_rejections, args.log)
        generate_filter_report(stats, args.report)
        logger.info(f"Wrote {len(qa_docs)} filtered files to {args.output}")
        logger.info(f"Wrote rejection log to {args.log}")
        logger.info(f"Wrote report to {args.report}")
    else:
        logger.info(f"[DRY RUN] Would filter {stats['rejected_qa']}/{stats['total_qa']} Q&A pairs")
        logger.info(f"[DRY RUN] Pass rate: {stats['passed_qa'] / stats['total_qa'] * 100:.1f}%" if stats['total_qa'] > 0 else "[DRY RUN] No Q&A pairs")

    # Summary
    logger.info(
        f"Summary: {stats['passed_qa']}/{stats['total_qa']} passed "
        f"({stats['passed_qa'] / stats['total_qa'] * 100:.1f}%)" if stats['total_qa'] > 0 else "No Q&A pairs processed"
    )


if __name__ == "__main__":
    main()
