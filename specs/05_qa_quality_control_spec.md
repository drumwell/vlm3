# Stage 5: Q&A Quality Control - Implementation Spec (TDD)

## Overview

**Scripts**:
- `scripts/05_filter_qa.py` - Filter out low-quality Q&A pairs
- `scripts/06_deduplicate_qa.py` - Remove duplicate Q&A pairs across pages

**Purpose**: Ensure training data quality by removing malformed, generic, or duplicate Q&A pairs before VLM dataset emission.

**Architecture Reference**: See `pipeline_rearchitecture.md` lines 111-122, 645-709.

---

## Input/Output Contracts

### 05_filter_qa.py

**Inputs**:
- `work/qa_raw/*.json` — Q&A files from Stage 4 (one per page)
- `config.yaml` — Filter thresholds and patterns

**Outputs**:
- `work/qa_filtered/*.json` — Filtered Q&A files (same schema, fewer pairs)
- `work/logs/qa_filter_report.md` — Summary report with statistics
- `work/logs/qa_filtered_out.csv` — Rejected Q&A pairs with rejection reasons

### 06_deduplicate_qa.py

**Inputs**:
- `work/qa_filtered/*.json` — Filtered Q&A files from 05_filter_qa.py
- `config.yaml` — Deduplication thresholds

**Outputs**:
- `work/qa_unique/*.json` — Deduplicated Q&A files
- `work/logs/qa_dedup_report.md` — Summary report with statistics
- `work/logs/qa_duplicates.csv` — Duplicate groups with kept/dropped decisions

---

## Data Schemas

### Input Q&A Schema (`work/qa_raw/*.json`)

```json
{
  "page_id": "21-03_clutch",
  "image_path": "data_src/21 - Clutch/21-03.jpg",
  "section_id": "21",
  "section_name": "Clutch",
  "source_type": "service_manual",
  "content_type": "procedure",
  "procedures_covered": ["21 21 000"],
  "procedures_names": ["Clutch disc - remove and install"],

  "generation": {
    "model": "claude-sonnet-4-20250514",
    "timestamp": "2025-01-15T10:30:00Z",
    "prompt_template": "procedure"
  },

  "qa_pairs": [
    {
      "id": "21-03_clutch-q01",
      "question": "What should I visually inspect the clutch pressure plate for?",
      "answer": "Visually inspect the clutch for cracks, wear, and burnt spots. The pressure contact surface must be level.",
      "question_type": "inspection"
    }
  ]
}
```

### Filtered Out Log Schema (`work/logs/qa_filtered_out.csv`)

```csv
timestamp,page_id,qa_id,question,answer,rejection_reason,filter_name
2025-01-15T10:30:00Z,21-03_clutch,21-03_clutch-q05,"What is shown?","I cannot determine from this image.",answer_too_generic,generic_answer_filter
```

### Duplicates Log Schema (`work/logs/qa_duplicates.csv`)

```csv
group_id,page_id,qa_id,question,answer,similarity_score,action,kept_qa_id
dup-001,21-03_clutch,21-03_clutch-q01,"What torque for flywheel?","85 Nm",1.0,dropped,21-02_clutch-q03
dup-001,21-02_clutch,21-02_clutch-q03,"What torque for flywheel bolts?","85 Nm",1.0,kept,21-02_clutch-q03
```

---

## Configuration Schema

Add to `config.yaml`:

```yaml
# Q&A Quality Filters (Stage 5)
filters:
  # Answer constraints
  min_answer_length: 10          # Minimum characters
  max_answer_length: 500         # Maximum characters (prevent rambling)

  # Question constraints
  min_question_length: 15        # Minimum characters
  require_question_mark: true    # Must end with "?"

  # Question diversity (per page)
  max_question_similarity: 0.80  # Reject if >80% word overlap with another Q on same page
  min_question_types_per_page: 2 # Warn if fewer than 2 different question_types

  # Generic answer patterns (reject if answer contains these)
  generic_answer_patterns:
    - "cannot determine"
    - "not visible"
    - "unclear from"
    - "I don't see"
    - "I cannot see"
    - "I can't determine"
    - "the image doesn't show"
    - "not specified"
    - "please refer to"
    - "typically"
    - "usually"
    - "generally"

  # Self-referential patterns (reject if question contains these)
  self_referential_patterns:
    - "on this page"
    - "in this image"
    - "as shown here"
    - "the manual states"
    - "according to the page"
    - "depicted in"

  # Valid question types
  valid_question_types:
    - factual
    - procedural
    - visual
    - inspection
    - tool
    - safety
    - navigation
    - wiring
    - connector
    - component
    - diagnostic
    - troubleshooting
    - signal
    - parameter
    - operation

# Deduplication settings (Stage 5)
deduplication:
  # Exact match detection
  enable_exact_match: true

  # Semantic similarity (requires sentence-transformers)
  enable_semantic: true
  embedding_model: "all-MiniLM-L6-v2"
  similarity_threshold: 0.90     # Questions with >0.90 cosine similarity are duplicates

  # Cross-page duplicate handling
  cross_page_enabled: true
  prefer_strategy: "longer_answer"  # Options: longer_answer, higher_confidence, first_seen

  # Batch processing for embeddings
  embedding_batch_size: 64
```

---

## Script 1: 05_filter_qa.py

### CLI Interface

```bash
python scripts/05_filter_qa.py \
  --input work/qa_raw \
  --output work/qa_filtered \
  --log work/logs/qa_filtered_out.csv \
  --report work/logs/qa_filter_report.md \
  --config config.yaml \
  [--dry-run]              # Show what would be filtered without writing
  [--verbose]              # Log each rejection
```

### Core Functions

#### 1. `load_qa_files(input_dir: Path) -> List[Dict]`

Load all Q&A JSON files from input directory.

```python
def load_qa_files(input_dir: Path) -> List[Dict]:
    """
    Load all Q&A JSON files from input directory.

    Args:
        input_dir: Path to directory containing *.json files

    Returns:
        List of parsed Q&A documents (one per file)

    Raises:
        FileNotFoundError: If input_dir doesn't exist
        ValueError: If no JSON files found
    """
```

**Test Cases**:
- `test_load_qa_files_valid_directory` — Loads all JSON files
- `test_load_qa_files_empty_directory` — Returns empty list (or raises ValueError)
- `test_load_qa_files_missing_directory` — Raises FileNotFoundError
- `test_load_qa_files_skips_invalid_json` — Logs warning, continues with valid files
- `test_load_qa_files_handles_nested_dirs` — Only reads top-level files (no recursion)

---

#### 2. `load_filter_config(config_path: Path) -> FilterConfig`

Load and validate filter configuration.

```python
@dataclass
class FilterConfig:
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
    """
```

**Test Cases**:
- `test_load_filter_config_full` — All fields populated
- `test_load_filter_config_missing_section` — Uses defaults
- `test_load_filter_config_partial` — Merges provided + defaults

---

#### 3. Filter Functions

Each filter returns `(passed: bool, reason: Optional[str])`.

##### 3a. `filter_answer_length(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]`

```python
def filter_answer_length(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check answer length constraints.

    Returns:
        (True, None) if passed
        (False, "answer_too_short") if < min_answer_length
        (False, "answer_too_long") if > max_answer_length
    """
```

**Test Cases**:
- `test_filter_answer_length_valid` — 50 chars passes
- `test_filter_answer_length_too_short` — 5 chars fails
- `test_filter_answer_length_too_long` — 600 chars fails
- `test_filter_answer_length_boundary_min` — Exactly min_answer_length passes
- `test_filter_answer_length_boundary_max` — Exactly max_answer_length passes
- `test_filter_answer_length_empty` — Empty string fails
- `test_filter_answer_length_whitespace_only` — "   " fails (after strip)

---

##### 3b. `filter_question_length(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]`

```python
def filter_question_length(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question length constraints.

    Returns:
        (True, None) if passed
        (False, "question_too_short") if < min_question_length
    """
```

**Test Cases**:
- `test_filter_question_length_valid` — 25 chars passes
- `test_filter_question_length_too_short` — 10 chars fails
- `test_filter_question_length_boundary` — Exactly min_question_length passes

---

##### 3c. `filter_question_mark(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]`

```python
def filter_question_mark(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question ends with '?'.

    Returns:
        (True, None) if ends with '?' or require_question_mark=False
        (False, "missing_question_mark") otherwise
    """
```

**Test Cases**:
- `test_filter_question_mark_present` — "What is X?" passes
- `test_filter_question_mark_missing` — "What is X" fails
- `test_filter_question_mark_trailing_space` — "What is X? " passes (strip before check)
- `test_filter_question_mark_disabled` — require_question_mark=False always passes

---

##### 3d. `filter_generic_answer(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]`

```python
def filter_generic_answer(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check answer doesn't contain generic/evasive patterns.

    Returns:
        (True, None) if no patterns matched
        (False, "generic_answer: <matched_pattern>") if pattern found
    """
```

**Test Cases**:
- `test_filter_generic_answer_valid` — "The torque is 85 Nm" passes
- `test_filter_generic_answer_cannot_determine` — "I cannot determine from this image" fails
- `test_filter_generic_answer_typically` — "Typically around 80 Nm" fails
- `test_filter_generic_answer_case_insensitive` — "I CANNOT DETERMINE" fails
- `test_filter_generic_answer_partial_match` — "undetermined" does NOT fail (word boundary)
- `test_filter_generic_answer_empty_patterns` — No patterns = always pass

---

##### 3e. `filter_self_referential(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]`

```python
def filter_self_referential(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question doesn't contain self-referential language.

    Returns:
        (True, None) if no patterns matched
        (False, "self_referential: <matched_pattern>") if pattern found
    """
```

**Test Cases**:
- `test_filter_self_referential_valid` — "What is the clutch torque?" passes
- `test_filter_self_referential_on_this_page` — "What is shown on this page?" fails
- `test_filter_self_referential_as_shown` — "What part is shown here?" fails
- `test_filter_self_referential_case_insensitive` — "ON THIS PAGE" fails

---

##### 3f. `filter_question_type(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]`

```python
def filter_question_type(qa: Dict, config: FilterConfig) -> Tuple[bool, Optional[str]]:
    """
    Check question_type is valid.

    Returns:
        (True, None) if question_type in valid_question_types or no validation
        (False, "invalid_question_type: <type>") if invalid
    """
```

**Test Cases**:
- `test_filter_question_type_valid` — "procedural" passes
- `test_filter_question_type_invalid` — "random_type" fails
- `test_filter_question_type_missing` — No question_type field = warning, passes
- `test_filter_question_type_empty_config` — No valid_question_types = always passes

---

##### 3g. `filter_question_diversity(qa_pairs: List[Dict], config: FilterConfig) -> List[Tuple[Dict, bool, Optional[str]]]`

```python
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
```

**Test Cases**:
- `test_filter_question_diversity_all_unique` — 3 different questions all pass
- `test_filter_question_diversity_exact_duplicate` — Second identical question fails
- `test_filter_question_diversity_similar` — 85% word overlap fails (at 0.80 threshold)
- `test_filter_question_diversity_first_wins` — First of duplicate pair passes
- `test_filter_question_diversity_empty_list` — Empty input returns empty output

---

#### 4. `apply_filters(qa_doc: Dict, config: FilterConfig) -> Tuple[Dict, List[Dict]]`

```python
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
```

**Test Cases**:
- `test_apply_filters_all_pass` — All Q&A pairs pass, rejected_list empty
- `test_apply_filters_some_fail` — Mixed results, correct pairs in each list
- `test_apply_filters_all_fail` — Returns doc with empty qa_pairs
- `test_apply_filters_preserves_metadata` — page_id, section_id etc preserved
- `test_apply_filters_order_preserved` — Passing Q&A pairs maintain order

---

#### 5. `compute_page_warnings(qa_doc: Dict, config: FilterConfig) -> List[str]`

```python
def compute_page_warnings(qa_doc: Dict, config: FilterConfig) -> List[str]:
    """
    Generate warnings for page-level quality issues.

    Warnings (non-blocking):
    - Fewer than min_question_types_per_page distinct types
    - All Q&A pairs filtered out
    - Very few Q&A pairs remaining (<3)
    """
```

**Test Cases**:
- `test_compute_page_warnings_all_good` — Returns empty list
- `test_compute_page_warnings_low_diversity` — Warns if only 1 question_type
- `test_compute_page_warnings_all_filtered` — Warns if 0 Q&A remain
- `test_compute_page_warnings_few_remaining` — Warns if <3 Q&A remain

---

#### 6. `write_filtered_output(qa_doc: Dict, output_path: Path) -> None`

```python
def write_filtered_output(qa_doc: Dict, output_path: Path) -> None:
    """
    Write filtered Q&A document to JSON file.

    Creates parent directories if needed.
    Pretty-prints JSON with indent=2.
    """
```

**Test Cases**:
- `test_write_filtered_output_creates_file` — File exists after write
- `test_write_filtered_output_valid_json` — Output is parseable JSON
- `test_write_filtered_output_creates_dirs` — Creates parent directories
- `test_write_filtered_output_overwrites` — Overwrites existing file

---

#### 7. `write_rejection_log(rejections: List[Dict], log_path: Path) -> None`

```python
def write_rejection_log(rejections: List[Dict], log_path: Path) -> None:
    """
    Write CSV log of rejected Q&A pairs.

    Schema: timestamp,page_id,qa_id,question,answer,rejection_reason,filter_name
    """
```

**Test Cases**:
- `test_write_rejection_log_creates_file` — File exists
- `test_write_rejection_log_correct_schema` — Headers match expected
- `test_write_rejection_log_escapes_csv` — Handles commas, quotes in Q&A text
- `test_write_rejection_log_appends` — Multiple calls append (not overwrite)

---

#### 8. `generate_filter_report(stats: Dict, report_path: Path) -> None`

```python
def generate_filter_report(stats: Dict, report_path: Path) -> None:
    """
    Generate Markdown summary report.

    Stats include:
    - Total Q&A pairs processed
    - Pairs passed / rejected
    - Rejection reasons breakdown
    - Per-page summary
    - Warnings
    """
```

Report template:

```markdown
# Q&A Filter Report

**Generated**: 2025-01-15T10:30:00Z

## Summary

| Metric | Count |
|--------|-------|
| Files Processed | 150 |
| Total Q&A Pairs | 1,500 |
| Passed | 1,350 (90.0%) |
| Rejected | 150 (10.0%) |

## Rejection Reasons

| Reason | Count | % of Rejected |
|--------|-------|---------------|
| answer_too_short | 45 | 30.0% |
| generic_answer | 40 | 26.7% |
| question_similarity | 35 | 23.3% |
| ... | ... | ... |

## Warnings

- **21-03_clutch**: Only 1 question type (procedural)
- **etm-045**: All Q&A pairs filtered out

## Sample Rejections

### answer_too_short (3 samples)
1. Q: "What is the torque?" A: "85 Nm" (9 chars, min=10)
...
```

**Test Cases**:
- `test_generate_filter_report_creates_file` — File exists
- `test_generate_filter_report_contains_stats` — Key stats present
- `test_generate_filter_report_includes_samples` — Sample rejections included

---

### Main Function

```python
def main():
    parser = argparse.ArgumentParser(description="Filter low-quality Q&A pairs")
    parser.add_argument("--input", type=Path, required=True, help="Input directory with qa_raw/*.json")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for filtered files")
    parser.add_argument("--log", type=Path, required=True, help="CSV log of rejected Q&A")
    parser.add_argument("--report", type=Path, required=True, help="Markdown summary report")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be filtered")
    parser.add_argument("--verbose", action="store_true", help="Log each rejection")

    args = parser.parse_args()

    # Load config
    config = load_filter_config(args.config)

    # Load all Q&A files
    qa_docs = load_qa_files(args.input)

    # Process each document
    all_rejections = []
    stats = {
        "files_processed": 0,
        "total_qa": 0,
        "passed_qa": 0,
        "rejected_qa": 0,
        "rejection_reasons": Counter(),
        "warnings": []
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
            stats["rejection_reasons"][r["filter_name"]] += 1
        stats["warnings"].extend(warnings)

        all_rejections.extend(rejections)

        # Write output (unless dry-run)
        if not args.dry_run:
            output_file = args.output / f"{qa_doc['page_id']}.json"
            write_filtered_output(filtered_doc, output_file)

    # Write logs and report
    if not args.dry_run:
        write_rejection_log(all_rejections, args.log)
        generate_filter_report(stats, args.report)
    else:
        print(f"[DRY RUN] Would filter {stats['rejected_qa']}/{stats['total_qa']} Q&A pairs")
```

---

## Script 2: 06_deduplicate_qa.py

### CLI Interface

```bash
python scripts/06_deduplicate_qa.py \
  --input work/qa_filtered \
  --output work/qa_unique \
  --log work/logs/qa_duplicates.csv \
  --report work/logs/qa_dedup_report.md \
  --config config.yaml \
  [--no-semantic]          # Skip semantic similarity (exact match only)
  [--dry-run]              # Show duplicates without removing
  [--verbose]              # Log each duplicate group
```

### Core Functions

#### 1. `load_dedup_config(config_path: Path) -> DedupConfig`

```python
@dataclass
class DedupConfig:
    enable_exact_match: bool = True
    enable_semantic: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    similarity_threshold: float = 0.90
    cross_page_enabled: bool = True
    prefer_strategy: str = "longer_answer"  # longer_answer, higher_confidence, first_seen
    embedding_batch_size: int = 64

def load_dedup_config(config_path: Path) -> DedupConfig:
    """Load deduplication configuration from YAML file."""
```

**Test Cases**:
- `test_load_dedup_config_full` — All fields populated
- `test_load_dedup_config_defaults` — Uses defaults for missing
- `test_load_dedup_config_invalid_strategy` — Raises ValueError

---

#### 2. `normalize_text(text: str) -> str`

```python
def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    - Lowercase
    - Remove punctuation except alphanumeric and spaces
    - Collapse whitespace
    - Strip leading/trailing whitespace
    """
```

**Test Cases**:
- `test_normalize_text_lowercase` — "ABC" -> "abc"
- `test_normalize_text_punctuation` — "What's the torque?" -> "whats the torque"
- `test_normalize_text_whitespace` — "a  b   c" -> "a b c"
- `test_normalize_text_unicode` — Handles "85°C" properly

---

#### 3. `find_exact_duplicates(qa_pairs: List[Dict]) -> List[DuplicateGroup]`

```python
@dataclass
class DuplicateGroup:
    group_id: str
    members: List[Dict]      # List of {page_id, qa_id, question, answer}
    kept: Dict               # The member to keep
    dropped: List[Dict]      # Members to drop

def find_exact_duplicates(qa_pairs: List[Dict]) -> List[DuplicateGroup]:
    """
    Find Q&A pairs with identical normalized questions.

    Groups are formed by exact question match.
    """
```

**Test Cases**:
- `test_find_exact_duplicates_none` — All unique, returns empty list
- `test_find_exact_duplicates_pair` — Two identical questions grouped
- `test_find_exact_duplicates_triple` — Three identical questions grouped
- `test_find_exact_duplicates_case_insensitive` — "What is X?" == "what is x?"
- `test_find_exact_duplicates_punctuation_insensitive` — "What's the torque?" == "Whats the torque"

---

#### 4. `compute_embeddings(texts: List[str], model_name: str, batch_size: int) -> np.ndarray`

```python
def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64
) -> np.ndarray:
    """
    Compute sentence embeddings for texts.

    Uses sentence-transformers library.

    Args:
        texts: List of text strings to embed
        model_name: Name of sentence-transformers model
        batch_size: Batch size for embedding computation

    Returns:
        numpy array of shape (len(texts), embedding_dim)
    """
```

**Test Cases**:
- `test_compute_embeddings_shape` — Returns correct shape
- `test_compute_embeddings_deterministic` — Same text = same embedding
- `test_compute_embeddings_similar_texts` — Similar texts have high cosine similarity
- `test_compute_embeddings_different_texts` — Different texts have lower similarity
- `test_compute_embeddings_batching` — Works with texts > batch_size

---

#### 5. `find_semantic_duplicates(qa_pairs: List[Dict], config: DedupConfig) -> List[DuplicateGroup]`

```python
def find_semantic_duplicates(
    qa_pairs: List[Dict],
    config: DedupConfig
) -> List[DuplicateGroup]:
    """
    Find Q&A pairs with semantically similar questions.

    Uses cosine similarity on embeddings.
    Threshold from config.similarity_threshold.

    Note: Excludes pairs already identified as exact duplicates.
    """
```

**Test Cases**:
- `test_find_semantic_duplicates_none` — All semantically distinct, returns empty
- `test_find_semantic_duplicates_paraphrase` — "What's the torque?" ~ "What torque should I use?"
- `test_find_semantic_duplicates_threshold` — 0.89 similarity at 0.90 threshold = not duplicate
- `test_find_semantic_duplicates_excludes_exact` — Doesn't double-count exact matches

---

#### 6. `select_best_qa(group: DuplicateGroup, strategy: str) -> Dict`

```python
def select_best_qa(group: DuplicateGroup, strategy: str) -> Dict:
    """
    Select the best Q&A pair to keep from a duplicate group.

    Strategies:
    - "longer_answer": Keep the one with longest answer
    - "higher_confidence": Keep one from higher-confidence page (if available)
    - "first_seen": Keep first occurrence (by page_id sort order)

    Returns:
        The selected Q&A pair to keep
    """
```

**Test Cases**:
- `test_select_best_qa_longer_answer` — Picks longest answer
- `test_select_best_qa_longer_answer_tie` — Tie-breaks by first_seen
- `test_select_best_qa_first_seen` — Picks earliest by page_id
- `test_select_best_qa_invalid_strategy` — Raises ValueError

---

#### 7. `merge_duplicate_groups(exact: List[DuplicateGroup], semantic: List[DuplicateGroup]) -> List[DuplicateGroup]`

```python
def merge_duplicate_groups(
    exact: List[DuplicateGroup],
    semantic: List[DuplicateGroup]
) -> List[DuplicateGroup]:
    """
    Merge exact and semantic duplicate groups.

    If a Q&A appears in both exact and semantic groups,
    prefer the exact match grouping.
    """
```

**Test Cases**:
- `test_merge_duplicate_groups_no_overlap` — Simple concatenation
- `test_merge_duplicate_groups_overlap` — Exact group takes precedence
- `test_merge_duplicate_groups_empty` — Handles empty inputs

---

#### 8. `apply_deduplication(qa_docs: List[Dict], config: DedupConfig) -> Tuple[List[Dict], List[DuplicateGroup]]`

```python
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
        - deduplicated_docs: Same structure, duplicates removed
        - duplicate_groups: All duplicate groups found
    """
```

**Test Cases**:
- `test_apply_deduplication_no_duplicates` — All docs unchanged
- `test_apply_deduplication_within_page` — Duplicates in same page removed
- `test_apply_deduplication_cross_page` — Duplicates across pages removed
- `test_apply_deduplication_cross_page_disabled` — config.cross_page_enabled=False
- `test_apply_deduplication_preserves_metadata` — page_id etc preserved

---

#### 9. `write_duplicate_log(groups: List[DuplicateGroup], log_path: Path) -> None`

```python
def write_duplicate_log(groups: List[DuplicateGroup], log_path: Path) -> None:
    """
    Write CSV log of duplicate groups.

    Schema: group_id,page_id,qa_id,question,answer,similarity_score,action,kept_qa_id
    """
```

**Test Cases**:
- `test_write_duplicate_log_creates_file` — File exists
- `test_write_duplicate_log_correct_schema` — Headers match
- `test_write_duplicate_log_kept_marked` — Kept member has action="kept"
- `test_write_duplicate_log_dropped_marked` — Dropped members have action="dropped"

---

#### 10. `generate_dedup_report(stats: Dict, report_path: Path) -> None`

```python
def generate_dedup_report(stats: Dict, report_path: Path) -> None:
    """
    Generate Markdown summary report.
    """
```

Report template:

```markdown
# Q&A Deduplication Report

**Generated**: 2025-01-15T10:30:00Z

## Summary

| Metric | Count |
|--------|-------|
| Files Processed | 150 |
| Total Q&A Pairs (input) | 1,350 |
| Unique Q&A Pairs (output) | 1,200 |
| Duplicates Removed | 150 (11.1%) |

## Duplicate Analysis

| Type | Groups | Pairs Removed |
|------|--------|---------------|
| Exact Match | 50 | 60 |
| Semantic Similar | 70 | 90 |

## Cross-Page Duplicates

Found 45 question clusters spanning multiple pages:

1. **"What torque for flywheel bolts?"** (3 occurrences)
   - Kept: 21-02_clutch-q03 (answer: 85 Nm, 6 words)
   - Dropped: 21-03_clutch-q01, 21-04_clutch-q02

2. ...

## Sample Duplicate Groups

### Exact Match (3 samples)
...

### Semantic Similar (3 samples)
...
```

---

### Main Function

```python
def main():
    parser = argparse.ArgumentParser(description="Deduplicate Q&A pairs")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--no-semantic", action="store_true", help="Skip semantic similarity")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = load_dedup_config(args.config)
    if args.no_semantic:
        config.enable_semantic = False

    qa_docs = load_qa_files(args.input)

    deduped_docs, duplicate_groups = apply_deduplication(qa_docs, config)

    # Compute stats
    total_input = sum(len(d.get("qa_pairs", [])) for d in qa_docs)
    total_output = sum(len(d.get("qa_pairs", [])) for d in deduped_docs)

    stats = {
        "files_processed": len(qa_docs),
        "total_input": total_input,
        "total_output": total_output,
        "duplicates_removed": total_input - total_output,
        "exact_groups": len([g for g in duplicate_groups if g.match_type == "exact"]),
        "semantic_groups": len([g for g in duplicate_groups if g.match_type == "semantic"]),
        "duplicate_groups": duplicate_groups
    }

    if not args.dry_run:
        # Write deduplicated files
        args.output.mkdir(parents=True, exist_ok=True)
        for doc in deduped_docs:
            output_file = args.output / f"{doc['page_id']}.json"
            write_filtered_output(doc, output_file)

        write_duplicate_log(duplicate_groups, args.log)
        generate_dedup_report(stats, args.report)
    else:
        print(f"[DRY RUN] Would remove {stats['duplicates_removed']} duplicates")
```

---

## Dependencies

Add to `requirements.txt`:

```
sentence-transformers>=2.2.0   # For semantic similarity
numpy>=1.21.0                  # Array operations
```

---

## Acceptance Criteria

### Per-Script Criteria

1. ✅ CLI with `--help` output
2. ✅ Idempotent (safe to rerun)
3. ✅ Handles empty input gracefully
4. ✅ Logs operations to stdout/stderr
5. ✅ Creates output directories if missing
6. ✅ Preserves Q&A document metadata

### Quality Criteria

1. ✅ Filter pass rate > 90% (most Q&A from Stage 4 should be good)
2. ✅ Deduplication removes < 15% (not too aggressive)
3. ✅ No false positives in semantic similarity (verify manually)
4. ✅ Cross-page duplicates correctly identified
5. ✅ Reports are human-readable and actionable

### Integration Criteria

1. ✅ Output schema matches Stage 4 input schema (for Stage 6)
2. ✅ Config integrates with existing config.yaml
3. ✅ Logs use consistent format with other stages

---

## Error Handling

### 05_filter_qa.py

| Error | Handling |
|-------|----------|
| Missing input directory | Exit with error code 1, clear message |
| Empty input directory | Warning, continue with empty output |
| Invalid JSON file | Log warning, skip file, continue |
| Missing config file | Exit with error code 1 |
| Invalid config values | Use defaults, log warning |

### 06_deduplicate_qa.py

| Error | Handling |
|-------|----------|
| Missing sentence-transformers | Log error, fall back to exact-only |
| Embedding model download fails | Log error, fall back to exact-only |
| Out of memory (large corpus) | Process in chunks, log warning |
| Invalid similarity threshold | Clamp to [0.0, 1.0], log warning |

---

## Performance Considerations

### 05_filter_qa.py

- **Time complexity**: O(n * m) where n=files, m=Q&A per file
- **Memory**: Loads one file at a time, low memory footprint
- **Expected runtime**: ~1 second per 1000 Q&A pairs

### 06_deduplicate_qa.py

- **Time complexity**: O(n²) for pairwise similarity (mitigated by batching)
- **Memory**: Embeddings for all questions in memory
- **Expected runtime**: ~30 seconds for 10,000 Q&A pairs (with GPU)
- **Optimization**: Use FAISS for large corpora (>50,000 Q&A)

---

## Test File Location

Tests should be placed in:
- `tests/test_05_filter_qa.py`
- `tests/test_06_deduplicate_qa.py`

Run with:
```bash
pytest tests/test_05_filter_qa.py tests/test_06_deduplicate_qa.py -v
```

---

## Implementation Order

1. **05_filter_qa.py** — Implement filters first
   - Start with `filter_answer_length` (simplest)
   - Add `filter_question_length`, `filter_question_mark`
   - Add `filter_generic_answer`, `filter_self_referential`
   - Add `filter_question_type`
   - Add `filter_question_diversity` (most complex)
   - Wire up `apply_filters` and main()

2. **06_deduplicate_qa.py** — Implement deduplication
   - Start with exact match (no dependencies)
   - Add semantic similarity (requires sentence-transformers)
   - Add `select_best_qa` strategies
   - Wire up `apply_deduplication` and main()

3. **Integration testing** — Test full pipeline
   - Run on Clutch section (21) sample
   - Verify output can be consumed by Stage 6
