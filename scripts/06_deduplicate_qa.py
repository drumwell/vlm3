#!/usr/bin/env python3
"""
06_deduplicate_qa.py - Remove duplicate Q&A pairs

Stage 5 of the data pipeline: Deduplication of Q&A pairs.
Removes exact and semantic duplicates across pages.

Usage:
    python scripts/06_deduplicate_qa.py \
        --input work/qa_filtered \
        --output work/qa_unique \
        --log work/logs/qa_duplicates.csv \
        --report work/logs/qa_dedup_report.md \
        --config config.yaml \
        [--no-semantic] [--dry-run] [--verbose]
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

VALID_STRATEGIES = {"longer_answer", "higher_confidence", "first_seen"}


@dataclass
class DedupConfig:
    """Configuration for Q&A deduplication."""

    enable_exact_match: bool = True
    enable_semantic: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.90
    cross_page_enabled: bool = True
    prefer_strategy: str = "longer_answer"
    embedding_batch_size: int = 64


def load_dedup_config(config_path: Path) -> DedupConfig:
    """
    Load deduplication configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        DedupConfig with loaded or default values

    Raises:
        ValueError: If prefer_strategy is invalid
    """
    defaults = DedupConfig()

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return defaults

    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    dedup = config_data.get("deduplication", {})

    prefer_strategy = dedup.get("prefer_strategy", defaults.prefer_strategy)
    if prefer_strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid prefer_strategy: {prefer_strategy}. "
            f"Must be one of: {VALID_STRATEGIES}"
        )

    return DedupConfig(
        enable_exact_match=dedup.get("enable_exact_match", defaults.enable_exact_match),
        enable_semantic=dedup.get("enable_semantic", defaults.enable_semantic),
        embedding_model=dedup.get("embedding_model", defaults.embedding_model),
        similarity_threshold=dedup.get("similarity_threshold", defaults.similarity_threshold),
        cross_page_enabled=dedup.get("cross_page_enabled", defaults.cross_page_enabled),
        prefer_strategy=prefer_strategy,
        embedding_batch_size=dedup.get("embedding_batch_size", defaults.embedding_batch_size),
    )


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class DuplicateGroup:
    """A group of duplicate Q&A pairs."""

    group_id: str
    members: List[Dict]
    kept: Optional[Dict]
    dropped: List[Dict]
    match_type: str  # "exact" or "semantic"
    similarity_score: float = 1.0


# ============================================================================
# Text Processing
# ============================================================================


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Remove punctuation except alphanumeric and spaces
    - Collapse whitespace
    - Strip leading/trailing whitespace

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()

    # Keep only alphanumeric and spaces
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


# ============================================================================
# File I/O
# ============================================================================


def load_qa_files(input_dir: Path) -> List[Dict]:
    """
    Load all Q&A JSON files from input directory.

    Args:
        input_dir: Path to directory containing *.json files

    Returns:
        List of parsed Q&A documents

    Raises:
        FileNotFoundError: If input_dir doesn't exist
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    docs = []

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
    """Write Q&A document to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(qa_doc, f, indent=2)


def write_duplicate_log(groups: List[DuplicateGroup], log_path: Path) -> None:
    """
    Write CSV log of duplicate groups.

    Schema: group_id,page_id,qa_id,question,answer,similarity_score,action,kept_qa_id
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "group_id", "page_id", "qa_id", "question", "answer",
        "similarity_score", "action", "kept_qa_id"
    ]

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for group in groups:
            kept_id = group.kept["id"] if group.kept else ""

            # Write kept member
            if group.kept:
                writer.writerow({
                    "group_id": group.group_id,
                    "page_id": group.kept.get("page_id", ""),
                    "qa_id": group.kept.get("id", ""),
                    "question": group.kept.get("question", ""),
                    "answer": group.kept.get("answer", ""),
                    "similarity_score": group.similarity_score,
                    "action": "kept",
                    "kept_qa_id": kept_id,
                })

            # Write dropped members
            for dropped in group.dropped:
                writer.writerow({
                    "group_id": group.group_id,
                    "page_id": dropped.get("page_id", ""),
                    "qa_id": dropped.get("id", ""),
                    "question": dropped.get("question", ""),
                    "answer": dropped.get("answer", ""),
                    "similarity_score": group.similarity_score,
                    "action": "dropped",
                    "kept_qa_id": kept_id,
                })


def generate_dedup_report(stats: Dict, report_path: Path) -> None:
    """Generate Markdown summary report."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow().isoformat() + "Z"

    total_input = stats.get("total_input", 0)
    total_output = stats.get("total_output", 0)
    removed = stats.get("duplicates_removed", 0)
    removed_pct = (removed / total_input * 100) if total_input > 0 else 0

    lines = [
        "# Q&A Deduplication Report",
        "",
        f"**Generated**: {timestamp}",
        "",
        "## Summary",
        "",
        "| Metric | Count |",
        "|--------|-------|",
        f"| Files Processed | {stats.get('files_processed', 0)} |",
        f"| Total Q&A Pairs (input) | {total_input:,} |",
        f"| Unique Q&A Pairs (output) | {total_output:,} |",
        f"| Duplicates Removed | {removed:,} ({removed_pct:.1f}%) |",
        "",
        "## Duplicate Analysis",
        "",
        "| Type | Groups |",
        "|------|--------|",
        f"| Exact Match | {stats.get('exact_groups', 0)} |",
        f"| Semantic Similar | {stats.get('semantic_groups', 0)} |",
        "",
    ]

    # Sample duplicate groups
    duplicate_groups = stats.get("duplicate_groups", [])
    if duplicate_groups:
        lines.extend([
            "## Sample Duplicate Groups",
            "",
        ])

        for group in duplicate_groups[:10]:
            lines.append(f"### {group.group_id} ({group.match_type})")
            lines.append("")
            if group.kept:
                lines.append(f"- **Kept**: {group.kept.get('id', 'unknown')}")
                lines.append(f"  - Q: \"{group.kept.get('question', '')[:60]}...\"")
            for dropped in group.dropped[:3]:
                lines.append(f"- Dropped: {dropped.get('id', 'unknown')}")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))


# ============================================================================
# Embedding Functions
# ============================================================================


_embedding_model = None


def get_embedding_model(model_name: str):
    """Lazy-load the sentence transformer model."""
    global _embedding_model

    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise

    return _embedding_model


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64
) -> np.ndarray:
    """
    Compute sentence embeddings for texts.

    Args:
        texts: List of text strings to embed
        model_name: Name of sentence-transformers model
        batch_size: Batch size for embedding computation

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False)
    return np.array(embeddings)


# ============================================================================
# Duplicate Detection
# ============================================================================


def find_exact_duplicates(qa_pairs: List[Dict]) -> List[DuplicateGroup]:
    """
    Find Q&A pairs with identical normalized questions.

    Groups are formed by exact question match.

    Args:
        qa_pairs: List of Q&A pairs with page_id, id, question, answer

    Returns:
        List of DuplicateGroup for each set of duplicates
    """
    # Group by normalized question
    question_groups: Dict[str, List[Dict]] = defaultdict(list)

    for qa in qa_pairs:
        normalized = normalize_text(qa.get("question", ""))
        question_groups[normalized].append(qa)

    # Create duplicate groups for groups with >1 member
    groups = []
    group_counter = 0

    for normalized_q, members in question_groups.items():
        if len(members) > 1:
            group_counter += 1
            groups.append(DuplicateGroup(
                group_id=f"exact-{group_counter:03d}",
                members=members,
                kept=None,
                dropped=[],
                match_type="exact",
                similarity_score=1.0,
            ))

    return groups


def find_semantic_duplicates(
    qa_pairs: List[Dict],
    config: DedupConfig,
    exclude_ids: Set[str]
) -> List[DuplicateGroup]:
    """
    Find Q&A pairs with semantically similar questions.

    Uses cosine similarity on embeddings.

    Args:
        qa_pairs: List of Q&A pairs
        config: Deduplication configuration
        exclude_ids: Set of Q&A IDs already in exact duplicate groups

    Returns:
        List of DuplicateGroup for semantic duplicates
    """
    if not config.enable_semantic:
        return []

    # Filter out already-grouped Q&A pairs
    filtered_pairs = [qa for qa in qa_pairs if qa.get("id") not in exclude_ids]

    if len(filtered_pairs) < 2:
        return []

    # Get questions and compute embeddings
    questions = [qa.get("question", "") for qa in filtered_pairs]

    try:
        embeddings = compute_embeddings(
            questions,
            config.embedding_model,
            config.embedding_batch_size
        )
    except Exception as e:
        logger.warning(f"Failed to compute embeddings: {e}")
        return []

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normalized = embeddings / norms

    # Compute pairwise cosine similarity
    similarity_matrix = np.dot(normalized, normalized.T)

    # Find pairs above threshold
    groups = []
    used_indices: Set[int] = set()
    group_counter = 0

    for i in range(len(filtered_pairs)):
        if i in used_indices:
            continue

        # Find all similar pairs
        similar_indices = []
        for j in range(i + 1, len(filtered_pairs)):
            if j in used_indices:
                continue
            if similarity_matrix[i, j] >= config.similarity_threshold:
                similar_indices.append(j)

        if similar_indices:
            group_counter += 1
            members = [filtered_pairs[i]] + [filtered_pairs[j] for j in similar_indices]
            avg_similarity = np.mean([similarity_matrix[i, j] for j in similar_indices])

            groups.append(DuplicateGroup(
                group_id=f"semantic-{group_counter:03d}",
                members=members,
                kept=None,
                dropped=[],
                match_type="semantic",
                similarity_score=float(avg_similarity),
            ))

            used_indices.add(i)
            used_indices.update(similar_indices)

    return groups


def select_best_qa(group: DuplicateGroup, strategy: str) -> Dict:
    """
    Select the best Q&A pair to keep from a duplicate group.

    Strategies:
    - "longer_answer": Keep the one with longest answer
    - "higher_confidence": Keep one from higher-confidence page (if available)
    - "first_seen": Keep first occurrence (by id sort order)

    Args:
        group: DuplicateGroup to select from
        strategy: Selection strategy

    Returns:
        The selected Q&A pair to keep

    Raises:
        ValueError: If strategy is invalid
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}")

    members = group.members

    if strategy == "longer_answer":
        # Sort by answer length (desc), then by id (asc) for tie-breaking
        sorted_members = sorted(
            members,
            key=lambda m: (-len(m.get("answer", "")), m.get("id", ""))
        )
        return sorted_members[0]

    elif strategy == "first_seen":
        # Sort by page_id, then by id
        sorted_members = sorted(
            members,
            key=lambda m: (m.get("page_id", ""), m.get("id", ""))
        )
        return sorted_members[0]

    elif strategy == "higher_confidence":
        # For now, fall back to longer_answer
        # Could be extended to use confidence scores if available
        sorted_members = sorted(
            members,
            key=lambda m: (-len(m.get("answer", "")), m.get("id", ""))
        )
        return sorted_members[0]

    return members[0]


def merge_duplicate_groups(
    exact: List[DuplicateGroup],
    semantic: List[DuplicateGroup]
) -> List[DuplicateGroup]:
    """
    Merge exact and semantic duplicate groups.

    If a Q&A appears in both exact and semantic groups,
    prefer the exact match grouping.

    Args:
        exact: List of exact duplicate groups
        semantic: List of semantic duplicate groups

    Returns:
        Merged list of duplicate groups
    """
    if not exact and not semantic:
        return []

    # Get all IDs in exact groups
    exact_ids: Set[str] = set()
    for group in exact:
        for member in group.members:
            exact_ids.add(member.get("id", ""))

    # Filter semantic groups to remove members already in exact groups
    filtered_semantic = []
    for group in semantic:
        filtered_members = [m for m in group.members if m.get("id", "") not in exact_ids]
        if len(filtered_members) > 1:
            filtered_semantic.append(DuplicateGroup(
                group_id=group.group_id,
                members=filtered_members,
                kept=group.kept,
                dropped=group.dropped,
                match_type=group.match_type,
                similarity_score=group.similarity_score,
            ))

    return exact + filtered_semantic


# ============================================================================
# Main Processing
# ============================================================================


def apply_deduplication(
    qa_docs: List[Dict],
    config: DedupConfig
) -> Tuple[List[Dict], List[DuplicateGroup]]:
    """
    Apply deduplication across all Q&A documents.

    Args:
        qa_docs: List of Q&A documents (each with qa_pairs)
        config: Deduplication configuration

    Returns:
        (deduplicated_docs, duplicate_groups)
    """
    # Flatten all Q&A pairs with page context
    all_qa_pairs = []
    for doc in qa_docs:
        page_id = doc.get("page_id", "unknown")
        for qa in doc.get("qa_pairs", []):
            qa_with_page = {**qa, "page_id": page_id}
            all_qa_pairs.append(qa_with_page)

    # Find duplicates based on cross-page setting
    if config.cross_page_enabled:
        # Global deduplication
        exact_groups = find_exact_duplicates(all_qa_pairs) if config.enable_exact_match else []

        # Get IDs already in exact groups
        exact_ids = {m.get("id") for g in exact_groups for m in g.members}

        semantic_groups = find_semantic_duplicates(
            all_qa_pairs, config, exact_ids
        ) if config.enable_semantic else []

    else:
        # Per-page deduplication only
        exact_groups = []
        semantic_groups = []

        for doc in qa_docs:
            page_id = doc.get("page_id", "unknown")
            page_pairs = [{**qa, "page_id": page_id} for qa in doc.get("qa_pairs", [])]

            if config.enable_exact_match:
                exact_groups.extend(find_exact_duplicates(page_pairs))

            if config.enable_semantic:
                exact_ids = {m.get("id") for g in exact_groups for m in g.members}
                semantic_groups.extend(find_semantic_duplicates(page_pairs, config, exact_ids))

    # Merge groups
    all_groups = merge_duplicate_groups(exact_groups, semantic_groups)

    # Select best from each group
    for group in all_groups:
        group.kept = select_best_qa(group, config.prefer_strategy)
        group.dropped = [m for m in group.members if m.get("id") != group.kept.get("id")]

    # Get IDs to drop
    drop_ids: Set[str] = set()
    for group in all_groups:
        for dropped in group.dropped:
            drop_ids.add(dropped.get("id", ""))

    # Create deduplicated documents
    deduped_docs = []
    for doc in qa_docs:
        kept_pairs = [
            qa for qa in doc.get("qa_pairs", [])
            if qa.get("id") not in drop_ids
        ]

        deduped_doc = {k: v for k, v in doc.items() if k != "qa_pairs"}
        deduped_doc["qa_pairs"] = kept_pairs
        deduped_docs.append(deduped_doc)

    return deduped_docs, all_groups


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Deduplicate Q&A pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with qa_filtered/*.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for deduplicated files",
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="CSV log of duplicates",
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
        "--no-semantic",
        action="store_true",
        help="Skip semantic similarity (exact match only)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show duplicates without removing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log each duplicate group",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    try:
        config = load_dedup_config(args.config)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    if args.no_semantic:
        config.enable_semantic = False

    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Semantic deduplication: {'enabled' if config.enable_semantic else 'disabled'}")

    # Load all Q&A files
    try:
        qa_docs = load_qa_files(args.input)
        logger.info(f"Loaded {len(qa_docs)} Q&A files from {args.input}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not qa_docs:
        logger.warning("No Q&A files found, creating empty output")

    # Apply deduplication
    deduped_docs, duplicate_groups = apply_deduplication(qa_docs, config)

    # Compute stats
    total_input = sum(len(d.get("qa_pairs", [])) for d in qa_docs)
    total_output = sum(len(d.get("qa_pairs", [])) for d in deduped_docs)

    exact_groups = len([g for g in duplicate_groups if g.match_type == "exact"])
    semantic_groups = len([g for g in duplicate_groups if g.match_type == "semantic"])

    stats = {
        "files_processed": len(qa_docs),
        "total_input": total_input,
        "total_output": total_output,
        "duplicates_removed": total_input - total_output,
        "exact_groups": exact_groups,
        "semantic_groups": semantic_groups,
        "duplicate_groups": duplicate_groups,
    }

    if args.verbose:
        for group in duplicate_groups:
            logger.debug(
                f"Duplicate group {group.group_id}: "
                f"kept={group.kept.get('id') if group.kept else 'none'}, "
                f"dropped={[d.get('id') for d in group.dropped]}"
            )

    # Write outputs
    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)

        for doc in deduped_docs:
            output_file = args.output / f"{doc['page_id']}.json"
            write_filtered_output(doc, output_file)

        write_duplicate_log(duplicate_groups, args.log)
        generate_dedup_report(stats, args.report)

        logger.info(f"Wrote {len(deduped_docs)} deduplicated files to {args.output}")
        logger.info(f"Wrote duplicate log to {args.log}")
        logger.info(f"Wrote report to {args.report}")
    else:
        logger.info(f"[DRY RUN] Would remove {stats['duplicates_removed']} duplicates")
        logger.info(f"[DRY RUN] Exact groups: {exact_groups}, Semantic groups: {semantic_groups}")

    # Summary
    if total_input > 0:
        pct = (stats['duplicates_removed'] / total_input) * 100
        logger.info(
            f"Summary: {total_output}/{total_input} unique "
            f"({stats['duplicates_removed']} removed, {pct:.1f}%)"
        )
    else:
        logger.info("No Q&A pairs processed")


if __name__ == "__main__":
    main()
