# Stage 6: Emit & Validate - Implementation Spec (TDD)

## Overview

**Scripts**:
- `scripts/07_emit_vlm_dataset.py` - Convert Q&A pairs to VLM training format with train/val split
- `scripts/08_validate_vlm.py` - Validate VLM training dataset
- `scripts/09_upload_vlm.py` - Upload dataset to HuggingFace Hub

**Purpose**: Transform filtered and deduplicated Q&A pairs into the final VLM training format, validate dataset integrity, and optionally upload to HuggingFace Hub for distribution.

**Architecture Reference**: See `pipeline_rearchitecture.md` lines 125-140, 711-826.

---

## Input/Output Contracts

### 07_emit_vlm_dataset.py

**Inputs**:
- `work/qa_unique/*.json` — Deduplicated Q&A files from Stage 5
- `data_src/` — Source images (for copying/symlinking)
- `config.yaml` — Output settings (split ratio, random seed, image copy mode)

**Outputs**:
- `data/vlm_train.jsonl` — Training dataset (90% by default)
- `data/vlm_val.jsonl` — Validation dataset (10% by default)
- `data/images/` — Directory containing referenced images (copied or symlinked)
- `work/logs/emit_report.md` — Summary report with statistics

### 08_validate_vlm.py

**Inputs**:
- `data/vlm_train.jsonl` — Training dataset
- `data/vlm_val.jsonl` — Validation dataset
- `data/images/` — Image directory

**Outputs**:
- `work/logs/vlm_qa_report.md` — Validation report with statistics and errors
- Exit code 0 (success) or 1 (critical errors found)

### 09_upload_vlm.py

**Inputs**:
- `data/vlm_train.jsonl` — Training dataset
- `data/vlm_val.jsonl` — Validation dataset
- `data/images/` — Image directory
- HuggingFace token (environment variable or CLI arg)

**Outputs**:
- HuggingFace dataset at specified repository
- `work/logs/upload_report.md` — Upload confirmation with dataset URL

---

## Data Schemas

### Input Q&A Schema (`work/qa_unique/*.json`)

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

### Output VLM Training Schema (`data/vlm_train.jsonl`)

**For image-based Q&A:**
```json
{"image": "images/21-03_clutch.jpg", "conversations": [{"role": "user", "content": "What should I visually inspect the clutch pressure plate for?"}, {"role": "assistant", "content": "Visually inspect the clutch for cracks, wear, and burnt spots. The pressure contact surface must be level."}], "metadata": {"page_id": "21-03_clutch", "section_id": "21", "section_name": "Clutch", "source_type": "service_manual", "content_type": "procedure", "question_type": "inspection", "qa_id": "21-03_clutch-q01"}}
```

**For HTML-based Q&A (text-only, no image):**
```json
{"image": null, "conversations": [{"role": "user", "content": "What is the engine displacement for the BMW E30 M3?"}, {"role": "assistant", "content": "2302 cc"}], "metadata": {"page_id": "html-m3-techspec", "section_id": "techspec", "section_name": "Technical Specifications", "source_type": "html_specs", "content_type": "specification", "question_type": "factual", "qa_id": "html-m3-q01"}}
```

Each line is one Q&A pair. Multiple lines can reference the same image.

### Validation Report Schema (`work/logs/vlm_qa_report.md`)

```markdown
# VLM Dataset Validation Report

**Generated**: 2025-01-15T10:30:00Z
**Status**: ✅ PASSED / ❌ FAILED

## Summary

| Metric | Train | Val | Total |
|--------|-------|-----|-------|
| Q&A Pairs | 9,000 | 1,000 | 10,000 |
| Unique Images | 450 | 50 | 500 |
| Text-only Q&A | 200 | 22 | 222 |

## Distribution by Section

| Section | Train | Val |
|---------|-------|-----|
| 21 - Clutch | 150 | 17 |
| 23 - Engine | 500 | 55 |
| ... | ... | ... |

## Distribution by Source Type

| Source Type | Train | Val |
|-------------|-------|-----|
| service_manual | 7,500 | 833 |
| electrical_manual | 1,200 | 134 |
| ecu_technical | 100 | 11 |
| html_specs | 200 | 22 |

## Critical Errors

None / List of errors...

## Warnings

- 3 images have low resolution (<500px width)
- ...

## Sample Q&A Pairs

### Training Set (5 samples)
...

### Validation Set (5 samples)
...
```

---

## Configuration Schema

Add to `config.yaml`:

```yaml
# Output settings (Stage 6)
output:
  # Train/validation split
  train_split: 0.90              # 90% train, 10% val
  random_seed: 42                # For reproducible splits

  # Image handling
  image_copy_mode: symlink       # Options: symlink, copy, relative
  image_output_dir: images       # Subdirectory within data/

  # Image filename normalization
  normalize_image_names: true    # Convert spaces/special chars to underscores

  # Stratification
  stratify_by: section_id        # Options: section_id, source_type, none

  # Minimum samples per stratum for stratification
  min_stratum_size: 10           # Fall back to random if stratum too small

# Validation settings (Stage 6)
validation:
  # Schema checks
  require_image_field: true      # Every record must have "image" key
  require_conversations: true    # Every record must have "conversations" key
  require_metadata: true         # Every record must have "metadata" key

  # Image validation
  check_image_exists: true       # Verify image files exist
  check_image_readable: true     # Verify images can be opened by PIL
  min_image_width: 100           # Warn if image width < this
  min_image_height: 100          # Warn if image height < this

  # Content validation
  min_question_length: 10        # Warn if question < this
  min_answer_length: 5           # Warn if answer < this
  max_answer_length: 1000        # Warn if answer > this

  # Distribution checks
  min_qa_per_section: 5          # Warn if section has fewer Q&A
  max_qa_per_section: 2000       # Warn if section has more Q&A (possible duplication)

  # Sample output
  num_samples_per_split: 5       # Number of samples to include in report

# HuggingFace upload settings (Stage 6)
upload:
  repo_id: vlm3                  # e.g., "drumwell/vlm3" (required for upload)
  private: false                 # Whether dataset is private
  commit_message: "Update VLM training dataset"

  # Dataset card metadata
  dataset_card:
    language: en
    license: cc-by-nc-4.0
    task_categories:
      - visual-question-answering
      - image-to-text
    tags:
      - automotive
      - bmw
      - service-manual
      - vlm
```

---

## Script 1: 07_emit_vlm_dataset.py

### CLI Interface

```bash
python scripts/07_emit_vlm_dataset.py \
  --qa work/qa_unique \
  --data-src data_src \
  --output data \
  --report work/logs/emit_report.md \
  --config config.yaml \
  [--train-split 0.9]        # Override train/val ratio
  [--seed 42]                # Override random seed
  [--copy-images]            # Force copy instead of symlink
  [--dry-run]                # Show what would be emitted without writing
  [--verbose]                # Log each Q&A pair
```

### Core Functions

#### 1. `load_qa_documents(input_dir: Path) -> List[Dict]`

Load all Q&A JSON files from input directory.

```python
def load_qa_documents(input_dir: Path) -> List[Dict]:
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
- `test_load_qa_documents_valid_directory` — Loads all JSON files
- `test_load_qa_documents_empty_directory` — Raises ValueError
- `test_load_qa_documents_missing_directory` — Raises FileNotFoundError
- `test_load_qa_documents_skips_invalid_json` — Logs warning, continues with valid
- `test_load_qa_documents_preserves_order` — Consistent ordering by filename

---

#### 2. `load_emit_config(config_path: Path) -> EmitConfig`

```python
@dataclass
class EmitConfig:
    train_split: float = 0.90
    random_seed: int = 42
    image_copy_mode: str = "symlink"    # symlink, copy, relative
    image_output_dir: str = "images"
    normalize_image_names: bool = True
    stratify_by: Optional[str] = "section_id"  # section_id, source_type, None
    min_stratum_size: int = 10

def load_emit_config(config_path: Path) -> EmitConfig:
    """
    Load emit configuration from YAML file.

    Uses defaults for any missing values.
    """
```

**Test Cases**:
- `test_load_emit_config_full` — All fields populated
- `test_load_emit_config_defaults` — Uses defaults for missing
- `test_load_emit_config_invalid_split` — Raises ValueError for split < 0 or > 1
- `test_load_emit_config_invalid_mode` — Raises ValueError for unknown image_copy_mode

---

#### 3. `flatten_qa_pairs(qa_docs: List[Dict]) -> List[Dict]`

```python
def flatten_qa_pairs(qa_docs: List[Dict]) -> List[Dict]:
    """
    Flatten all Q&A pairs from documents into a single list.

    Each output record contains:
    - question: str
    - answer: str
    - image_path: Optional[str] (original path from source doc)
    - page_id: str
    - section_id: str
    - section_name: str
    - source_type: str
    - content_type: str
    - question_type: str
    - qa_id: str

    Returns:
        List of flattened Q&A records
    """
```

**Test Cases**:
- `test_flatten_qa_pairs_single_doc` — Flattens one document with 5 Q&A
- `test_flatten_qa_pairs_multiple_docs` — Flattens 3 documents
- `test_flatten_qa_pairs_empty_doc` — Handles doc with no qa_pairs
- `test_flatten_qa_pairs_preserves_metadata` — All metadata fields present
- `test_flatten_qa_pairs_handles_null_image` — HTML sources have image_path=None

---

#### 4. `normalize_image_filename(image_path: str) -> str`

```python
def normalize_image_filename(image_path: str) -> str:
    """
    Normalize image filename for output.

    - Replaces spaces with underscores
    - Replaces special characters with underscores
    - Preserves extension
    - Handles paths with directories

    Examples:
        "21 - Clutch/21-03.jpg" -> "21_Clutch_21-03.jpg"
        "Bosch Motronic ML 3-1/page001.jpg" -> "Bosch_Motronic_ML_3-1_page001.jpg"
    """
```

**Test Cases**:
- `test_normalize_image_filename_spaces` — Replaces spaces
- `test_normalize_image_filename_special_chars` — Handles colons, parens
- `test_normalize_image_filename_preserves_extension` — .jpg stays .jpg
- `test_normalize_image_filename_nested_path` — Flattens directory structure
- `test_normalize_image_filename_already_clean` — No change if already normalized

---

#### 5. `format_vlm_record(qa_record: Dict, image_output_path: Optional[str]) -> Dict`

```python
def format_vlm_record(qa_record: Dict, image_output_path: Optional[str]) -> Dict:
    """
    Format a Q&A record for VLM training output.

    Args:
        qa_record: Flattened Q&A record
        image_output_path: Relative path to image in output dir, or None for text-only

    Returns:
        Dict with structure:
        {
            "image": "images/21-03_clutch.jpg" or null,
            "conversations": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "metadata": {...}
        }
    """
```

**Test Cases**:
- `test_format_vlm_record_with_image` — Image path included
- `test_format_vlm_record_without_image` — image=null for HTML sources
- `test_format_vlm_record_conversation_structure` — Correct role alternation
- `test_format_vlm_record_metadata_complete` — All metadata fields present
- `test_format_vlm_record_escapes_content` — Handles quotes in Q&A text

---

#### 6. `stratified_split(records: List[Dict], config: EmitConfig) -> Tuple[List[Dict], List[Dict]]`

```python
def stratified_split(
    records: List[Dict],
    config: EmitConfig
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split records into train/val sets with stratification.

    Args:
        records: List of flattened Q&A records
        config: Emit configuration with split ratio and stratify_by

    Returns:
        (train_records, val_records)

    Notes:
        - Uses config.random_seed for reproducibility
        - If stratify_by is set, maintains distribution across strata
        - Falls back to random split if stratum size < min_stratum_size
    """
```

**Test Cases**:
- `test_stratified_split_ratio` — Approximately 90/10 split
- `test_stratified_split_reproducible` — Same seed = same split
- `test_stratified_split_different_seed` — Different seed = different split
- `test_stratified_split_by_section` — Each section has ~90/10 distribution
- `test_stratified_split_by_source_type` — Each source_type has ~90/10 distribution
- `test_stratified_split_no_stratify` — Random split when stratify_by=None
- `test_stratified_split_small_stratum` — Falls back to random for small strata
- `test_stratified_split_single_record` — Handles edge case of 1 record

---

#### 7. `copy_or_link_image(src_path: Path, dst_path: Path, mode: str) -> bool`

```python
def copy_or_link_image(src_path: Path, dst_path: Path, mode: str) -> bool:
    """
    Copy or symlink an image file.

    Args:
        src_path: Source image path
        dst_path: Destination path
        mode: "copy", "symlink", or "relative"

    Returns:
        True if successful, False if source doesn't exist

    Notes:
        - Creates parent directories if needed
        - "relative" mode creates relative symlinks
        - Skips if destination already exists and is valid
    """
```

**Test Cases**:
- `test_copy_or_link_image_copy` — File copied successfully
- `test_copy_or_link_image_symlink` — Symlink created
- `test_copy_or_link_image_relative` — Relative symlink created
- `test_copy_or_link_image_missing_source` — Returns False, logs warning
- `test_copy_or_link_image_creates_dirs` — Creates parent directories
- `test_copy_or_link_image_idempotent` — Skips existing valid files

---

#### 8. `write_jsonl(records: List[Dict], output_path: Path) -> int`

```python
def write_jsonl(records: List[Dict], output_path: Path) -> int:
    """
    Write records to JSONL file.

    Args:
        records: List of VLM-formatted records
        output_path: Path to output .jsonl file

    Returns:
        Number of records written

    Notes:
        - One JSON object per line
        - UTF-8 encoding
        - Creates parent directories if needed
    """
```

**Test Cases**:
- `test_write_jsonl_creates_file` — File exists after write
- `test_write_jsonl_valid_jsonl` — Each line is valid JSON
- `test_write_jsonl_line_count` — Correct number of lines
- `test_write_jsonl_utf8` — Handles unicode characters
- `test_write_jsonl_no_trailing_newline_issues` — Proper line endings

---

#### 9. `collect_unique_images(records: List[Dict]) -> Set[str]`

```python
def collect_unique_images(records: List[Dict]) -> Set[str]:
    """
    Collect unique image paths from records.

    Args:
        records: List of flattened Q&A records

    Returns:
        Set of unique image_path values (excluding None)
    """
```

**Test Cases**:
- `test_collect_unique_images_deduplicates` — Same image referenced twice = one entry
- `test_collect_unique_images_excludes_none` — HTML sources not included
- `test_collect_unique_images_empty` — Empty input = empty set

---

#### 10. `generate_emit_report(stats: Dict, report_path: Path) -> None`

```python
def generate_emit_report(stats: Dict, report_path: Path) -> None:
    """
    Generate Markdown summary report.

    Stats include:
    - Total Q&A pairs emitted (train/val)
    - Unique images copied/linked
    - Distribution by section and source_type
    - Any warnings (missing images, etc.)
    """
```

Report template:

```markdown
# VLM Dataset Emit Report

**Generated**: 2025-01-15T10:30:00Z

## Summary

| Metric | Count |
|--------|-------|
| Total Q&A Pairs | 10,000 |
| Training Set | 9,000 (90.0%) |
| Validation Set | 1,000 (10.0%) |
| Unique Images | 500 |
| Text-only Q&A | 222 |

## Images

| Metric | Count |
|--------|-------|
| Images Copied/Linked | 500 |
| Missing Images | 0 |
| Mode | symlink |

## Distribution by Section

| Section | Train | Val | Total |
|---------|-------|-----|-------|
| 21 - Clutch | 135 | 15 | 150 |
| 23 - Engine | 450 | 50 | 500 |
| ... | ... | ... | ... |

## Distribution by Source Type

| Source Type | Train | Val | Total |
|-------------|-------|-----|-------|
| service_manual | 7,500 | 833 | 8,333 |
| electrical_manual | 1,080 | 120 | 1,200 |
| ... | ... | ... | ... |

## Warnings

None / List of warnings...
```

**Test Cases**:
- `test_generate_emit_report_creates_file` — File exists
- `test_generate_emit_report_contains_stats` — Key metrics present
- `test_generate_emit_report_distribution_tables` — Section/source tables present

---

### Main Function

```python
def main():
    parser = argparse.ArgumentParser(description="Emit VLM training dataset")
    parser.add_argument("--qa", type=Path, required=True, help="Input directory with qa_unique/*.json")
    parser.add_argument("--data-src", type=Path, required=True, help="Source images directory")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for data/")
    parser.add_argument("--report", type=Path, required=True, help="Markdown summary report")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--train-split", type=float, help="Override train split ratio")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--copy-images", action="store_true", help="Force copy instead of symlink")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be emitted")
    parser.add_argument("--verbose", action="store_true", help="Log each Q&A pair")

    args = parser.parse_args()

    # Load config with overrides
    config = load_emit_config(args.config)
    if args.train_split is not None:
        config.train_split = args.train_split
    if args.seed is not None:
        config.random_seed = args.seed
    if args.copy_images:
        config.image_copy_mode = "copy"

    # Load and flatten Q&A documents
    qa_docs = load_qa_documents(args.qa)
    all_records = flatten_qa_pairs(qa_docs)

    # Split into train/val
    train_records, val_records = stratified_split(all_records, config)

    # Collect unique images
    unique_images = collect_unique_images(all_records)

    # Track stats
    stats = {
        "total_qa": len(all_records),
        "train_qa": len(train_records),
        "val_qa": len(val_records),
        "unique_images": len(unique_images),
        "text_only_qa": len([r for r in all_records if r.get("image_path") is None]),
        "images_copied": 0,
        "images_missing": 0,
        "section_distribution": Counter(),
        "source_distribution": Counter(),
        "warnings": []
    }

    # Update distributions
    for r in train_records:
        stats["section_distribution"][(r["section_id"], "train")] += 1
        stats["source_distribution"][(r["source_type"], "train")] += 1
    for r in val_records:
        stats["section_distribution"][(r["section_id"], "val")] += 1
        stats["source_distribution"][(r["source_type"], "val")] += 1

    if args.dry_run:
        print(f"[DRY RUN] Would emit {len(train_records)} train + {len(val_records)} val Q&A pairs")
        print(f"[DRY RUN] Would copy/link {len(unique_images)} images")
        return

    # Create output directories
    output_dir = args.output
    images_dir = output_dir / config.image_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Copy/link images
    image_mapping = {}  # original path -> output path
    for img_path in unique_images:
        src = args.data_src.parent / img_path if not Path(img_path).is_absolute() else Path(img_path)
        if config.normalize_image_names:
            dst_name = normalize_image_filename(img_path)
        else:
            dst_name = Path(img_path).name
        dst = images_dir / dst_name

        if copy_or_link_image(src, dst, config.image_copy_mode):
            stats["images_copied"] += 1
            image_mapping[img_path] = f"{config.image_output_dir}/{dst_name}"
        else:
            stats["images_missing"] += 1
            stats["warnings"].append(f"Missing image: {img_path}")

    # Format records for output
    def format_record(r):
        output_img = image_mapping.get(r.get("image_path")) if r.get("image_path") else None
        return format_vlm_record(r, output_img)

    train_formatted = [format_record(r) for r in train_records]
    val_formatted = [format_record(r) for r in val_records]

    # Write JSONL files
    write_jsonl(train_formatted, output_dir / "vlm_train.jsonl")
    write_jsonl(val_formatted, output_dir / "vlm_val.jsonl")

    # Generate report
    generate_emit_report(stats, args.report)

    print(f"Emitted {len(train_records)} train + {len(val_records)} val Q&A pairs")
    print(f"Copied/linked {stats['images_copied']} images ({stats['images_missing']} missing)")
```

---

## Script 2: 08_validate_vlm.py

### CLI Interface

```bash
python scripts/08_validate_vlm.py \
  --train data/vlm_train.jsonl \
  --val data/vlm_val.jsonl \
  --images data/images \
  --output work/logs/vlm_qa_report.md \
  --config config.yaml \
  [--strict]                 # Treat warnings as errors
  [--skip-image-check]       # Skip image existence/readability checks
  [--verbose]                # Log each validation check
```

### Core Functions

#### 1. `load_validation_config(config_path: Path) -> ValidationConfig`

```python
@dataclass
class ValidationConfig:
    require_image_field: bool = True
    require_conversations: bool = True
    require_metadata: bool = True
    check_image_exists: bool = True
    check_image_readable: bool = True
    min_image_width: int = 100
    min_image_height: int = 100
    min_question_length: int = 10
    min_answer_length: int = 5
    max_answer_length: int = 1000
    min_qa_per_section: int = 5
    max_qa_per_section: int = 2000
    num_samples_per_split: int = 5

def load_validation_config(config_path: Path) -> ValidationConfig:
    """Load validation configuration from YAML file."""
```

**Test Cases**:
- `test_load_validation_config_full` — All fields populated
- `test_load_validation_config_defaults` — Uses defaults for missing

---

#### 2. `load_jsonl(file_path: Path) -> List[Dict]`

```python
def load_jsonl(file_path: Path) -> List[Dict]:
    """
    Load records from JSONL file.

    Args:
        file_path: Path to .jsonl file

    Returns:
        List of parsed records

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If any line is invalid JSON
    """
```

**Test Cases**:
- `test_load_jsonl_valid` — Loads all records
- `test_load_jsonl_missing_file` — Raises FileNotFoundError
- `test_load_jsonl_invalid_json` — Raises ValueError with line number
- `test_load_jsonl_empty_file` — Returns empty list
- `test_load_jsonl_trailing_newline` — Handles trailing newline gracefully

---

#### 3. `validate_schema(record: Dict, config: ValidationConfig) -> List[str]`

```python
def validate_schema(record: Dict, config: ValidationConfig) -> List[str]:
    """
    Validate record schema.

    Returns:
        List of error messages (empty if valid)

    Checks:
        - "image" field exists (if require_image_field)
        - "conversations" field exists and is list (if require_conversations)
        - "metadata" field exists (if require_metadata)
        - conversations has correct role structure
    """
```

**Test Cases**:
- `test_validate_schema_valid` — Returns empty list
- `test_validate_schema_missing_image` — Error if require_image_field=True
- `test_validate_schema_missing_conversations` — Error if require_conversations=True
- `test_validate_schema_invalid_conversations` — Not a list
- `test_validate_schema_bad_role` — Role not user/assistant
- `test_validate_schema_missing_metadata` — Error if require_metadata=True
- `test_validate_schema_null_image_allowed` — image=null is valid for text-only

---

#### 4. `validate_conversations(record: Dict, config: ValidationConfig) -> Tuple[List[str], List[str]]`

```python
def validate_conversations(
    record: Dict,
    config: ValidationConfig
) -> Tuple[List[str], List[str]]:
    """
    Validate conversation content.

    Returns:
        (errors, warnings)

    Checks:
        - Question length >= min_question_length
        - Answer length >= min_answer_length
        - Answer length <= max_answer_length
        - Proper role alternation (user, assistant, user, assistant...)
    """
```

**Test Cases**:
- `test_validate_conversations_valid` — No errors or warnings
- `test_validate_conversations_short_question` — Warning for short question
- `test_validate_conversations_short_answer` — Warning for short answer
- `test_validate_conversations_long_answer` — Warning for long answer
- `test_validate_conversations_wrong_order` — Error for wrong role order
- `test_validate_conversations_empty` — Error for empty conversations

---

#### 5. `validate_image(record: Dict, images_dir: Path, config: ValidationConfig) -> Tuple[List[str], List[str]]`

```python
def validate_image(
    record: Dict,
    images_dir: Path,
    config: ValidationConfig
) -> Tuple[List[str], List[str]]:
    """
    Validate image reference.

    Returns:
        (errors, warnings)

    Checks:
        - Image file exists (if check_image_exists)
        - Image is readable by PIL (if check_image_readable)
        - Image dimensions >= min (warnings only)

    Notes:
        - Skips checks if image is null (text-only record)
    """
```

**Test Cases**:
- `test_validate_image_exists` — No errors for existing image
- `test_validate_image_missing` — Error for missing image
- `test_validate_image_unreadable` — Error for corrupted image
- `test_validate_image_small` — Warning for small dimensions
- `test_validate_image_null` — No checks for null image
- `test_validate_image_skip_check` — Respects check_image_exists=False

---

#### 6. `compute_distribution_stats(records: List[Dict]) -> Dict`

```python
def compute_distribution_stats(records: List[Dict]) -> Dict:
    """
    Compute distribution statistics from records.

    Returns:
        Dict with:
        - section_counts: Counter of section_id
        - source_counts: Counter of source_type
        - question_type_counts: Counter of question_type
        - content_type_counts: Counter of content_type
        - image_count: Number of records with images
        - text_only_count: Number of records without images
        - answer_length_stats: min, max, mean, median
    """
```

**Test Cases**:
- `test_compute_distribution_stats_sections` — Correct section counts
- `test_compute_distribution_stats_sources` — Correct source counts
- `test_compute_distribution_stats_answer_lengths` — Correct length stats
- `test_compute_distribution_stats_empty` — Handles empty list

---

#### 7. `check_distribution_issues(stats: Dict, config: ValidationConfig) -> List[str]`

```python
def check_distribution_issues(stats: Dict, config: ValidationConfig) -> List[str]:
    """
    Check for distribution anomalies.

    Returns:
        List of warning messages

    Checks:
        - Section with < min_qa_per_section Q&A
        - Section with > max_qa_per_section Q&A
        - Missing expected source types
    """
```

**Test Cases**:
- `test_check_distribution_issues_none` — No warnings for balanced data
- `test_check_distribution_issues_sparse_section` — Warns for low-count section
- `test_check_distribution_issues_dense_section` — Warns for high-count section

---

#### 8. `sample_records(records: List[Dict], n: int, seed: int = 42) -> List[Dict]`

```python
def sample_records(records: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """
    Sample n records for report display.

    Uses fixed seed for reproducibility.
    Returns fewer than n if not enough records.
    """
```

**Test Cases**:
- `test_sample_records_exact_n` — Returns exactly n records
- `test_sample_records_fewer_available` — Returns all if < n records
- `test_sample_records_reproducible` — Same seed = same sample

---

#### 9. `generate_validation_report(results: Dict, report_path: Path) -> None`

```python
def generate_validation_report(results: Dict, report_path: Path) -> None:
    """
    Generate Markdown validation report.

    Results include:
    - passed: bool
    - train_stats: distribution stats for train set
    - val_stats: distribution stats for val set
    - errors: list of critical errors
    - warnings: list of warnings
    - train_samples: sample records from train
    - val_samples: sample records from val
    """
```

**Test Cases**:
- `test_generate_validation_report_passed` — Status shows PASSED
- `test_generate_validation_report_failed` — Status shows FAILED
- `test_generate_validation_report_includes_samples` — Samples in output
- `test_generate_validation_report_error_list` — Errors listed

---

### Main Function

```python
def main():
    parser = argparse.ArgumentParser(description="Validate VLM training dataset")
    parser.add_argument("--train", type=Path, required=True, help="Training JSONL file")
    parser.add_argument("--val", type=Path, required=True, help="Validation JSONL file")
    parser.add_argument("--images", type=Path, required=True, help="Images directory")
    parser.add_argument("--output", type=Path, required=True, help="Validation report path")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("--skip-image-check", action="store_true", help="Skip image validation")
    parser.add_argument("--verbose", action="store_true", help="Log each check")

    args = parser.parse_args()

    config = load_validation_config(args.config)
    if args.skip_image_check:
        config.check_image_exists = False
        config.check_image_readable = False

    # Load datasets
    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)

    all_errors = []
    all_warnings = []

    # Validate each record
    for split_name, records in [("train", train_records), ("val", val_records)]:
        for i, record in enumerate(records):
            # Schema validation
            schema_errors = validate_schema(record, config)
            for e in schema_errors:
                all_errors.append(f"{split_name}[{i}]: {e}")

            # Conversation validation
            conv_errors, conv_warnings = validate_conversations(record, config)
            for e in conv_errors:
                all_errors.append(f"{split_name}[{i}]: {e}")
            for w in conv_warnings:
                all_warnings.append(f"{split_name}[{i}]: {w}")

            # Image validation
            img_errors, img_warnings = validate_image(record, args.images, config)
            for e in img_errors:
                all_errors.append(f"{split_name}[{i}]: {e}")
            for w in img_warnings:
                all_warnings.append(f"{split_name}[{i}]: {w}")

    # Compute distribution stats
    train_stats = compute_distribution_stats(train_records)
    val_stats = compute_distribution_stats(val_records)

    # Check distribution issues
    dist_warnings = check_distribution_issues(train_stats, config)
    all_warnings.extend([f"train distribution: {w}" for w in dist_warnings])

    dist_warnings = check_distribution_issues(val_stats, config)
    all_warnings.extend([f"val distribution: {w}" for w in dist_warnings])

    # Sample records for report
    train_samples = sample_records(train_records, config.num_samples_per_split)
    val_samples = sample_records(val_records, config.num_samples_per_split)

    # Determine pass/fail
    passed = len(all_errors) == 0
    if args.strict and len(all_warnings) > 0:
        passed = False

    # Generate report
    results = {
        "passed": passed,
        "train_count": len(train_records),
        "val_count": len(val_records),
        "train_stats": train_stats,
        "val_stats": val_stats,
        "errors": all_errors,
        "warnings": all_warnings,
        "train_samples": train_samples,
        "val_samples": val_samples
    }

    generate_validation_report(results, args.output)

    # Exit with appropriate code
    if passed:
        print(f"✅ Validation PASSED: {len(train_records)} train + {len(val_records)} val records")
        sys.exit(0)
    else:
        print(f"❌ Validation FAILED: {len(all_errors)} errors, {len(all_warnings)} warnings")
        sys.exit(1)
```

---

## Script 3: 09_upload_vlm.py

### CLI Interface

```bash
python scripts/09_upload_vlm.py \
  --train data/vlm_train.jsonl \
  --val data/vlm_val.jsonl \
  --images data/images \
  --repo drumwell/vlm3 \
  --report work/logs/upload_report.md \
  --config config.yaml \
  [--token HF_TOKEN]         # HuggingFace token (or use HF_TOKEN env var)
  [--private]                # Make dataset private
  [--message "commit msg"]   # Custom commit message
  [--dry-run]                # Show what would be uploaded
```

### Core Functions

#### 1. `load_upload_config(config_path: Path) -> UploadConfig`

```python
@dataclass
class UploadConfig:
    repo_id: Optional[str] = None
    private: bool = False
    commit_message: str = "Update VLM training dataset"
    language: str = "en"
    license: str = "cc-by-nc-4.0"
    task_categories: List[str] = field(default_factory=lambda: ["visual-question-answering"])
    tags: List[str] = field(default_factory=lambda: ["automotive", "bmw", "vlm"])

def load_upload_config(config_path: Path) -> UploadConfig:
    """Load upload configuration from YAML file."""
```

**Test Cases**:
- `test_load_upload_config_full` — All fields populated
- `test_load_upload_config_defaults` — Uses defaults for missing

---

#### 2. `get_hf_token(cli_token: Optional[str]) -> str`

```python
def get_hf_token(cli_token: Optional[str] = None) -> str:
    """
    Get HuggingFace token from CLI arg or environment.

    Priority:
    1. CLI argument
    2. HF_TOKEN environment variable
    3. HUGGING_FACE_HUB_TOKEN environment variable

    Raises:
        ValueError: If no token found
    """
```

**Test Cases**:
- `test_get_hf_token_cli` — CLI arg takes priority
- `test_get_hf_token_env` — Falls back to HF_TOKEN env
- `test_get_hf_token_missing` — Raises ValueError

---

#### 3. `create_dataset_card(config: UploadConfig, stats: Dict) -> str`

```python
def create_dataset_card(config: UploadConfig, stats: Dict) -> str:
    """
    Create dataset card (README.md) content.

    Includes:
    - Title and description
    - Dataset statistics
    - Usage examples
    - License information
    - Citation format
    """
```

Returns Markdown content for README.md:

```markdown
---
language:
- en
license: cc-by-nc-4.0
task_categories:
- visual-question-answering
- image-to-text
tags:
- automotive
- bmw
- service-manual
- vlm
---

# BMW E30 M3 Service Manual VLM Dataset

Visual question-answering dataset for BMW E30 M3 and 320is service procedures.

## Dataset Statistics

| Split | Q&A Pairs | Unique Images |
|-------|-----------|---------------|
| Train | 9,000 | 450 |
| Val | 1,000 | 50 |

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("drumwell/vlm3")
```

## Data Format

Each record contains:
- `image`: Path to service manual page image (or null for text-only)
- `conversations`: List of user/assistant message pairs
- `metadata`: Source information (section, page, content type)

## License

CC BY-NC 4.0

## Citation

```bibtex
@dataset{bmw_e30_m3_vlm,
  title={BMW E30 M3 Service Manual VLM Dataset},
  year={2025},
  publisher={HuggingFace}
}
```
```

**Test Cases**:
- `test_create_dataset_card_format` — Valid Markdown
- `test_create_dataset_card_stats` — Contains correct stats
- `test_create_dataset_card_metadata` — YAML frontmatter valid

---

#### 4. `prepare_dataset_files(train_path: Path, val_path: Path, images_dir: Path) -> Dict[str, Path]`

```python
def prepare_dataset_files(
    train_path: Path,
    val_path: Path,
    images_dir: Path
) -> Dict[str, Path]:
    """
    Prepare file mapping for upload.

    Returns:
        Dict mapping relative paths to local paths:
        {
            "train.jsonl": Path("/path/to/vlm_train.jsonl"),
            "val.jsonl": Path("/path/to/vlm_val.jsonl"),
            "images/21-03.jpg": Path("/path/to/images/21-03.jpg"),
            ...
        }
    """
```

**Test Cases**:
- `test_prepare_dataset_files_jsonl` — Includes train and val JSONL
- `test_prepare_dataset_files_images` — Includes all images
- `test_prepare_dataset_files_relative_paths` — Paths are relative

---

#### 5. `upload_to_huggingface(files: Dict[str, Path], config: UploadConfig, token: str, dataset_card: str) -> str`

```python
def upload_to_huggingface(
    files: Dict[str, Path],
    config: UploadConfig,
    token: str,
    dataset_card: str
) -> str:
    """
    Upload dataset to HuggingFace Hub.

    Args:
        files: Mapping of relative paths to local paths
        config: Upload configuration
        token: HuggingFace API token
        dataset_card: README.md content

    Returns:
        URL of uploaded dataset

    Uses huggingface_hub library for upload.
    """
```

**Test Cases**:
- `test_upload_to_huggingface_creates_repo` — Repo created if doesn't exist
- `test_upload_to_huggingface_uploads_files` — All files uploaded
- `test_upload_to_huggingface_private` — Respects private flag
- `test_upload_to_huggingface_dry_run` — Returns URL without uploading (mock)

---

#### 6. `generate_upload_report(stats: Dict, url: str, report_path: Path) -> None`

```python
def generate_upload_report(stats: Dict, url: str, report_path: Path) -> None:
    """
    Generate upload confirmation report.
    """
```

Report template:

```markdown
# VLM Dataset Upload Report

**Generated**: 2025-01-15T10:30:00Z
**Status**: ✅ SUCCESS

## Upload Details

| Field | Value |
|-------|-------|
| Repository | drumwell/vlm3 |
| URL | https://huggingface.co/datasets/drumwell/vlm3 |
| Visibility | Public |
| Commit | abc123... |

## Files Uploaded

| File Type | Count | Size |
|-----------|-------|------|
| JSONL | 2 | 5.2 MB |
| Images | 500 | 450 MB |
| README | 1 | 2 KB |

## Dataset Statistics

| Split | Q&A Pairs |
|-------|-----------|
| Train | 9,000 |
| Val | 1,000 |
```

**Test Cases**:
- `test_generate_upload_report_success` — Contains success status
- `test_generate_upload_report_url` — Contains dataset URL

---

### Main Function

```python
def main():
    parser = argparse.ArgumentParser(description="Upload VLM dataset to HuggingFace")
    parser.add_argument("--train", type=Path, required=True, help="Training JSONL file")
    parser.add_argument("--val", type=Path, required=True, help="Validation JSONL file")
    parser.add_argument("--images", type=Path, required=True, help="Images directory")
    parser.add_argument("--repo", type=str, required=True, help="HuggingFace repo ID")
    parser.add_argument("--report", type=Path, required=True, help="Upload report path")
    parser.add_argument("--config", type=Path, required=True, help="Config YAML file")
    parser.add_argument("--token", type=str, help="HuggingFace token")
    parser.add_argument("--private", action="store_true", help="Make dataset private")
    parser.add_argument("--message", type=str, help="Commit message")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")

    args = parser.parse_args()

    config = load_upload_config(args.config)
    config.repo_id = args.repo
    if args.private:
        config.private = True
    if args.message:
        config.commit_message = args.message

    # Get token
    if not args.dry_run:
        token = get_hf_token(args.token)
    else:
        token = "dry-run-token"

    # Prepare files
    files = prepare_dataset_files(args.train, args.val, args.images)

    # Compute stats for dataset card
    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.val)

    stats = {
        "train_count": len(train_records),
        "val_count": len(val_records),
        "image_count": len([f for f in files if f.startswith("images/")]),
        "total_size_mb": sum(f.stat().st_size for f in files.values()) / (1024 * 1024)
    }

    # Create dataset card
    dataset_card = create_dataset_card(config, stats)

    if args.dry_run:
        print(f"[DRY RUN] Would upload to {config.repo_id}")
        print(f"[DRY RUN] Files: {len(files)} ({stats['total_size_mb']:.1f} MB)")
        print(f"[DRY RUN] Train: {stats['train_count']}, Val: {stats['val_count']}")
        return

    # Upload
    url = upload_to_huggingface(files, config, token, dataset_card)

    # Generate report
    generate_upload_report(stats, url, args.report)

    print(f"✅ Dataset uploaded: {url}")
```

---

## Dependencies

Add to `requirements.txt`:

```
huggingface_hub>=0.20.0     # HuggingFace upload
Pillow>=9.0.0               # Image validation
PyYAML>=6.0                 # Config loading
```

---

## Acceptance Criteria

### Per-Script Criteria

1. ✅ CLI with `--help` output
2. ✅ Idempotent (safe to rerun)
3. ✅ Handles empty input gracefully
4. ✅ Logs operations to stdout/stderr
5. ✅ Creates output directories if missing
6. ✅ Preserves Q&A metadata through pipeline

### Quality Criteria (07_emit)

1. ✅ Train/val split matches configured ratio (±1%)
2. ✅ Split is reproducible with same seed
3. ✅ All referenced images exist in output
4. ✅ JSONL files are valid JSON-per-line
5. ✅ Stratification maintains distribution across sections

### Quality Criteria (08_validate)

1. ✅ Zero false positives in schema validation
2. ✅ All critical errors cause non-zero exit
3. ✅ Image validation catches corrupted files
4. ✅ Report is human-readable

### Quality Criteria (09_upload)

1. ✅ Dataset card contains accurate statistics
2. ✅ All files uploaded successfully
3. ✅ Token handling is secure (not logged)
4. ✅ Private flag respected

### Integration Criteria

1. ✅ Output from 08_validate can gate 09_upload
2. ✅ Dataset format compatible with standard VLM training frameworks
3. ✅ Can be loaded with `datasets.load_dataset()`

---

## Error Handling

### 07_emit_vlm_dataset.py

| Error | Handling |
|-------|----------|
| Missing input directory | Exit with error code 1, clear message |
| Empty input directory | Warning, continue with empty output |
| Invalid JSON file | Log warning, skip file, continue |
| Missing source image | Log warning, continue, track in report |
| Symlink creation fails | Fall back to copy, log warning |
| Invalid split ratio | Exit with error, suggest valid range |

### 08_validate_vlm.py

| Error | Handling |
|-------|----------|
| Missing JSONL file | Exit with error code 1 |
| Invalid JSON line | Critical error in report |
| Missing image | Critical error (unless --skip-image-check) |
| Corrupted image | Critical error (unless --skip-image-check) |
| Schema violation | Critical error |

### 09_upload_vlm.py

| Error | Handling |
|-------|----------|
| Missing HF token | Exit with error, explain how to set |
| Auth failure | Exit with error, suggest token refresh |
| Network error | Retry 3 times, then exit with error |
| Repo creation fails | Exit with error, check permissions |
| Rate limit | Wait and retry with exponential backoff |

---

## Performance Considerations

### 07_emit_vlm_dataset.py

- **Time complexity**: O(n) where n = total Q&A pairs
- **Memory**: Loads all Q&A into memory for stratified split
- **Expected runtime**: ~5 seconds for 10,000 Q&A pairs + image linking time
- **Optimization**: Parallel image copying for large datasets

### 08_validate_vlm.py

- **Time complexity**: O(n) for schema checks, O(n * image_load_time) for image validation
- **Memory**: Loads one image at a time
- **Expected runtime**: ~30 seconds for 10,000 Q&A with 500 images
- **Optimization**: Skip image validation with `--skip-image-check` for quick runs

### 09_upload_vlm.py

- **Time complexity**: O(total_file_size / bandwidth)
- **Memory**: Streams files, low memory footprint
- **Expected runtime**: Depends on network, ~5-10 minutes for 500MB
- **Optimization**: HuggingFace Hub handles chunked uploads automatically

---

## Test File Location

Tests should be placed in:
- `tests/test_07_emit_vlm_dataset.py`
- `tests/test_08_validate_vlm.py`
- `tests/test_09_upload_vlm.py`

Run with:
```bash
pytest tests/test_07_emit_vlm_dataset.py tests/test_08_validate_vlm.py tests/test_09_upload_vlm.py -v
```

---

## Test Fixtures

### Common Fixtures

```python
# tests/conftest.py

import pytest
import json
from pathlib import Path
import tempfile

@pytest.fixture
def sample_qa_doc():
    """Sample Q&A document matching Stage 5 output schema."""
    return {
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
                "answer": "Visually inspect the clutch for cracks, wear, and burnt spots.",
                "question_type": "inspection"
            },
            {
                "id": "21-03_clutch-q02",
                "question": "What is the torque specification for the pressure plate bolts?",
                "answer": "Tighten the pressure plate bolts to 25 Nm in a cross pattern.",
                "question_type": "factual"
            }
        ]
    }

@pytest.fixture
def sample_html_qa_doc():
    """Sample Q&A document for HTML source (no image)."""
    return {
        "page_id": "html-m3-techspec",
        "image_path": None,
        "section_id": "techspec",
        "section_name": "Technical Specifications",
        "source_type": "html_specs",
        "content_type": "specification",
        "generation": {
            "method": "html_parse",
            "timestamp": "2025-01-15T10:30:00Z"
        },
        "qa_pairs": [
            {
                "id": "html-m3-q01",
                "question": "What is the engine displacement for the BMW E30 M3?",
                "answer": "2302 cc",
                "question_type": "factual"
            }
        ]
    }

@pytest.fixture
def sample_vlm_record():
    """Sample VLM-formatted record."""
    return {
        "image": "images/21-03_clutch.jpg",
        "conversations": [
            {"role": "user", "content": "What should I inspect?"},
            {"role": "assistant", "content": "Inspect for cracks and wear."}
        ],
        "metadata": {
            "page_id": "21-03_clutch",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "procedure",
            "question_type": "inspection",
            "qa_id": "21-03_clutch-q01"
        }
    }

@pytest.fixture
def temp_qa_dir(sample_qa_doc, sample_html_qa_doc):
    """Temporary directory with sample Q&A JSON files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        qa_dir = Path(tmpdir) / "qa_unique"
        qa_dir.mkdir()

        # Write sample docs
        with open(qa_dir / "21-03_clutch.json", "w") as f:
            json.dump(sample_qa_doc, f)

        with open(qa_dir / "html-m3-techspec.json", "w") as f:
            json.dump(sample_html_qa_doc, f)

        yield qa_dir

@pytest.fixture
def temp_image_dir():
    """Temporary directory with sample images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = Path(tmpdir) / "data_src" / "21 - Clutch"
        img_dir.mkdir(parents=True)

        # Create a minimal valid JPEG
        from PIL import Image
        img = Image.new("RGB", (800, 600), color="white")
        img.save(img_dir / "21-03.jpg", "JPEG")

        yield Path(tmpdir) / "data_src"

@pytest.fixture
def sample_config():
    """Sample configuration dict."""
    return {
        "output": {
            "train_split": 0.9,
            "random_seed": 42,
            "image_copy_mode": "symlink",
            "image_output_dir": "images",
            "normalize_image_names": True,
            "stratify_by": "section_id",
            "min_stratum_size": 10
        },
        "validation": {
            "require_image_field": True,
            "require_conversations": True,
            "require_metadata": True,
            "check_image_exists": True,
            "check_image_readable": True,
            "min_image_width": 100,
            "min_image_height": 100,
            "min_question_length": 10,
            "min_answer_length": 5,
            "max_answer_length": 1000,
            "min_qa_per_section": 5,
            "max_qa_per_section": 2000,
            "num_samples_per_split": 5
        },
        "upload": {
            "repo_id": "test/test-dataset",
            "private": False,
            "commit_message": "Test upload"
        }
    }
```

---

## Implementation Order

1. **07_emit_vlm_dataset.py** — Core emission
   - Start with `load_qa_documents`, `flatten_qa_pairs` (data loading)
   - Add `normalize_image_filename`, `format_vlm_record` (transformation)
   - Add `stratified_split` (train/val splitting)
   - Add `copy_or_link_image`, `write_jsonl` (output)
   - Wire up main() with CLI

2. **08_validate_vlm.py** — Validation
   - Start with `load_jsonl`, `validate_schema` (basic checks)
   - Add `validate_conversations`, `validate_image` (content checks)
   - Add `compute_distribution_stats`, `check_distribution_issues` (analysis)
   - Add report generation
   - Wire up main() with exit codes

3. **09_upload_vlm.py** — Upload (optional, lower priority)
   - Start with `get_hf_token`, `load_upload_config`
   - Add `prepare_dataset_files`, `create_dataset_card`
   - Add `upload_to_huggingface` (requires HF account for testing)
   - Wire up main()

4. **Integration testing**
   - Run full Stage 5 → Stage 6 pipeline on Clutch section
   - Verify output can be loaded with `datasets.load_dataset()`
   - Manual review of sample Q&A pairs

---

## Integration with Makefile

Add to `Makefile`:

```makefile
# Stage 6: Emit & Validate
emit:
	python scripts/07_emit_vlm_dataset.py \
		--qa work/qa_unique \
		--data-src data_src \
		--output data \
		--report work/logs/emit_report.md \
		--config config.yaml

validate:
	python scripts/08_validate_vlm.py \
		--train data/vlm_train.jsonl \
		--val data/vlm_val.jsonl \
		--images data/images \
		--output work/logs/vlm_qa_report.md \
		--config config.yaml

upload:
	python scripts/09_upload_vlm.py \
		--train data/vlm_train.jsonl \
		--val data/vlm_val.jsonl \
		--images data/images \
		--repo drumwell/vlm3 \
		--report work/logs/upload_report.md \
		--config config.yaml

# Validate before upload (safe upload)
safe-upload: validate upload
```
