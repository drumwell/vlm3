#!/usr/bin/env python3
"""
06_validate.py - Cross-reference forum Q&A against manual data and filter

Stage 6 of the forum pipeline: validates forum Q&A pairs against the manual
dataset, detects semantic overlap, tags factual confidence, and applies
quality filters.

Usage:
    python pipeline/06_validate.py \
        --input work/qa_raw \
        --output work/qa_validated \
        --report work/logs/crossref_report.md \
        --config config.yaml \
        [--verbose]
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
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

def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Manual Data Loading
# ============================================================================

def load_manual_qa(manual_path: Path) -> List[Dict]:
    """Load manual Q&A pairs for cross-reference."""
    if not manual_path.exists():
        logger.warning(f"Manual data not found: {manual_path}. Skipping cross-reference.")
        return []

    pairs = []
    with open(manual_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            convos = record.get("conversations", [])
            for i in range(0, len(convos) - 1, 2):
                if convos[i].get("role") == "user" and convos[i + 1].get("role") == "assistant":
                    pairs.append({
                        "question": convos[i]["content"],
                        "answer": convos[i + 1]["content"],
                        "metadata": record.get("metadata", {}),
                    })

    logger.info(f"Loaded {len(pairs)} manual Q&A pairs for cross-reference")
    return pairs


# ============================================================================
# Numeric Extraction for Cross-Reference
# ============================================================================

UNIT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(Nm|nm|mm|cm|bar|psi|°C|°F|kg|lbs?|ft[- ]?lbs?|liters?|L|ml|cc|rpm|hp|kw|V|A|ohms?|Ω)",
    re.IGNORECASE,
)


def extract_numeric_claims(text: str) -> List[Tuple[float, str]]:
    """Extract numeric values with units from text."""
    matches = UNIT_PATTERN.findall(text)
    return [(float(v), u.lower()) for v, u in matches]


def check_numeric_claims(forum_answer: str, manual_pairs: List[Dict]) -> str:
    """Check forum numeric claims against manual data.

    Returns 'verified' if any forum claim matches a manual claim,
    'unverified' if claims exist but no match found, or 'plausible'
    if no numeric claims to check.
    """
    forum_claims = extract_numeric_claims(forum_answer)
    if not forum_claims:
        return "plausible"

    # Build set of (value, unit) from manual answers
    manual_claims = set()
    for pair in manual_pairs:
        for claim in extract_numeric_claims(pair.get("answer", "")):
            manual_claims.add(claim)

    if not manual_claims:
        return "unverified"

    # Check for exact matches (same value + same unit)
    for value, unit in forum_claims:
        if (value, unit) in manual_claims:
            return "verified"

    return "unverified"


# ============================================================================
# Semantic Similarity (optional)
# ============================================================================

def compute_embeddings(texts: List[str], model_name: str, batch_size: int):
    """Compute sentence embeddings. Returns None if sentence-transformers unavailable."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.warning("sentence-transformers not installed. Skipping semantic dedup.")
        return None

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return embeddings


def find_similar_pairs(
    forum_questions: List[str],
    manual_questions: List[str],
    threshold: float,
    model_name: str,
    batch_size: int,
) -> Set[int]:
    """Find forum questions semantically similar to manual questions."""
    try:
        import numpy as np
    except ImportError:
        return set()

    if not manual_questions:
        return set()

    all_texts = forum_questions + manual_questions
    embeddings = compute_embeddings(all_texts, model_name, batch_size)
    if embeddings is None:
        return set()

    n_forum = len(forum_questions)
    forum_emb = embeddings[:n_forum]
    manual_emb = embeddings[n_forum:]

    # Cosine similarity
    from numpy.linalg import norm
    redundant = set()
    for i, fe in enumerate(forum_emb):
        for me in manual_emb:
            sim = np.dot(fe, me) / (norm(fe) * norm(me) + 1e-8)
            if sim > threshold:
                redundant.add(i)
                break

    return redundant


def find_internal_duplicates(
    questions: List[str],
    threshold: float,
    model_name: str,
    batch_size: int,
) -> Set[int]:
    """Find duplicate questions within the forum dataset itself."""
    try:
        import numpy as np
        from numpy.linalg import norm
    except ImportError:
        return set()

    embeddings = compute_embeddings(questions, model_name, batch_size)
    if embeddings is None:
        return set()

    duplicates = set()
    for i in range(len(embeddings)):
        if i in duplicates:
            continue
        for j in range(i + 1, len(embeddings)):
            if j in duplicates:
                continue
            sim = np.dot(embeddings[i], embeddings[j]) / (
                norm(embeddings[i]) * norm(embeddings[j]) + 1e-8
            )
            if sim > threshold:
                duplicates.add(j)  # Keep first, remove later

    return duplicates


# ============================================================================
# Quality Filters
# ============================================================================

def filter_qa_pair(pair: Dict, config: Dict) -> Optional[str]:
    """Check if a Q&A pair should be filtered. Returns reason or None."""
    filters = config.get("filters", {})

    answer = pair.get("answer", "")
    question = pair.get("question", "")

    # Answer length
    min_len = filters.get("min_answer_length", 50)
    max_len = filters.get("max_answer_length", 1000)
    if len(answer) < min_len:
        return "answer_too_short"
    if len(answer) > max_len:
        return "answer_too_long"

    # Question length
    min_q_len = filters.get("min_question_length", 15)
    if len(question) < min_q_len:
        return "question_too_short"

    # Generic answer patterns
    answer_lower = answer.lower()
    for pattern in filters.get("generic_answer_patterns", []):
        if pattern.lower() in answer_lower:
            return f"generic_answer:{pattern}"

    # Vague question patterns
    question_lower = question.lower()
    for pattern in filters.get("vague_question_patterns", []):
        if pattern.lower() in question_lower:
            return f"vague_question:{pattern}"

    return None


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-reference and filter forum Q&A pairs"
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw Q&A directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--report", type=Path, required=True, help="Cross-reference report path")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)
    xref_config = config.get("cross_reference", {})

    args.output.mkdir(parents=True, exist_ok=True)

    # Load manual Q&A for cross-reference
    manual_path = Path(xref_config.get("manual_train_path", "../manual/prepared/manual_train.jsonl"))
    if not manual_path.is_absolute():
        manual_path = args.config.parent / manual_path
    manual_qa = load_manual_qa(manual_path)
    manual_questions = [p["question"] for p in manual_qa]

    # Load all forum Q&A pairs
    qa_files = sorted(args.input.glob("*.json"))
    logger.info(f"Validating Q&A from {len(qa_files)} files")

    all_pairs = []  # (file_idx, pair_idx, pair, parent_doc)
    all_questions = []

    for file_idx, qa_file in enumerate(qa_files):
        with open(qa_file) as f:
            doc = json.load(f)
        for pair_idx, pair in enumerate(doc.get("qa_pairs", [])):
            all_pairs.append((file_idx, pair_idx, pair, doc))
            all_questions.append(pair.get("question", ""))

    logger.info(f"Total Q&A pairs: {len(all_pairs)}")

    # Quality filtering
    filter_reasons = Counter()
    quality_pass = set()
    for i, (_, _, pair, _) in enumerate(all_pairs):
        reason = filter_qa_pair(pair, config)
        if reason:
            filter_reasons[reason] += 1
        else:
            quality_pass.add(i)

    logger.info(f"Quality filter: {len(all_pairs)} → {len(quality_pass)}")

    # Semantic dedup against manual
    model_name = xref_config.get("embedding_model", "all-MiniLM-L6-v2")
    batch_size = xref_config.get("embedding_batch_size", 64)
    threshold = xref_config.get("similarity_threshold", 0.85)

    quality_questions = [all_questions[i] for i in sorted(quality_pass)]
    quality_indices = sorted(quality_pass)

    redundant_in_quality = find_similar_pairs(
        quality_questions, manual_questions, threshold, model_name, batch_size
    )
    redundant_global = {quality_indices[i] for i in redundant_in_quality}
    logger.info(f"Redundant with manual: {len(redundant_global)}")

    # Internal dedup
    internal_dups_in_quality = find_internal_duplicates(
        quality_questions, threshold, model_name, batch_size
    )
    internal_dups_global = {quality_indices[i] for i in internal_dups_in_quality}
    logger.info(f"Internal duplicates: {len(internal_dups_global)}")

    # Final surviving set
    surviving = quality_pass - redundant_global - internal_dups_global

    # Tag factual confidence using numeric cross-reference
    confidence_counts = Counter()
    for i in surviving:
        _, _, pair, _ = all_pairs[i]
        answer = pair.get("answer", "")
        if manual_qa:
            pair["factual_confidence"] = check_numeric_claims(answer, manual_qa)
        else:
            pair["factual_confidence"] = "unverified"
        confidence_counts[pair["factual_confidence"]] += 1

    logger.info(f"Final surviving Q&A: {len(surviving)}")

    # Write output grouped by source file
    surviving_by_file = defaultdict(list)
    for i in surviving:
        file_idx, pair_idx, pair, doc = all_pairs[i]
        surviving_by_file[file_idx].append(pair)

    output_count = 0
    for file_idx, pairs in surviving_by_file.items():
        _, _, _, doc = all_pairs[next(i for i in surviving if all_pairs[i][0] == file_idx)]
        output_doc = {
            "thread_id": doc.get("thread_id"),
            "forum": doc.get("forum"),
            "thread_title": doc.get("thread_title"),
            "source_type": "forum",
            "content_type": doc.get("content_type"),
            "quality_score": doc.get("quality_score"),
            "qa_pairs": pairs,
        }
        output_file = args.output / qa_files[file_idx].name
        with open(output_file, "w") as f:
            json.dump(output_doc, f, indent=2)
        output_count += 1

    logger.info(f"Wrote {output_count} validated Q&A files")

    # Write report
    _write_report(
        args.report, len(all_pairs), filter_reasons, len(redundant_global),
        len(internal_dups_global), len(surviving), confidence_counts,
    )


def _write_report(
    report_path: Path,
    total: int,
    filter_reasons: Counter,
    redundant: int,
    internal_dups: int,
    surviving: int,
    confidence_counts: Counter,
):
    """Write cross-reference report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Forum Pipeline — Cross-Reference & Filter Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        "",
        f"- Total Q&A pairs in: {total}",
        f"- Quality filtered: {sum(filter_reasons.values())}",
        f"- Redundant with manual: {redundant}",
        f"- Internal duplicates: {internal_dups}",
        f"- **Surviving Q&A pairs: {surviving}** ({surviving/max(total,1)*100:.1f}%)",
        "",
        "## Quality Filters",
        "",
        "| Filter | Count |",
        "|--------|-------|",
    ]
    for reason, count in filter_reasons.most_common():
        lines.append(f"| {reason} | {count} |")

    lines += [
        "",
        "## Factual Confidence",
        "",
        "| Level | Count |",
        "|-------|-------|",
    ]
    for level, count in confidence_counts.most_common():
        lines.append(f"| {level} | {count} |")

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    logger.info(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
