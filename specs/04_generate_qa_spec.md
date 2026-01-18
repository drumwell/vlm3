# Stage 4: Q&A Generation - Implementation Spec (TDD)

## Overview

**Scripts**:
- `scripts/04a_generate_qa_images.py` - Generate Q&A from image pages using Claude API
- `scripts/04b_generate_qa_html.py` - Generate Q&A from HTML techspec files (no API)

**Purpose**: Generate question-answer pairs for VLM training. Image pages use Claude Vision API with context-aware prompts. HTML specs are parsed programmatically.

**Architecture Reference**: See `pipeline_rearchitecture.md` lines 91-108, 440-599.

---

## Known Edge Cases & Clarifications

> **Note:** This section addresses gaps identified during spec review (7/10 score).
> Each subsection includes implementation code that should be incorporated into the scripts.

### 1. Page ID Normalization

Page IDs may differ between classification CSV and index metadata:
- CSV: `21-01`, `21-02` (zero-padded)
- Index: `21-1`, `21-2` (no zero-padding)

**Normalization Logic:**
```python
def normalize_page_id(page_id: str) -> str:
    """
    Normalize page ID for matching across sources.

    Examples:
        "21-01" -> "21-1"
        "21-1"  -> "21-1"
        "ETM-001" -> "ETM-1"
    """
    parts = page_id.split("-")
    if len(parts) == 2:
        section, page_num = parts
        # Strip leading zeros from page number
        page_num = str(int(page_num)) if page_num.isdigit() else page_num.lstrip("0") or "0"
        return f"{section}-{page_num}"
    return page_id
```

All lookups in `page_to_procedures` MUST use normalized IDs.

### 2. Section Slug Derivation

Section slug is derived deterministically from section_id and section_name:

```python
def derive_section_slug(section_id: str, section_name: str) -> str:
    """
    Derive section slug for index file lookup.

    Examples:
        ("21", "Clutch") -> "21-clutch"
        ("00", "Maintenance") -> "00-maintenance"
        ("ETM", "Electrical Troubleshooting") -> "etm-electrical-troubleshooting"
    """
    # Lowercase, replace spaces with hyphens, remove special chars
    slug = section_name.lower().replace(" ", "-")
    slug = re.sub(r'[^a-z0-9-]', '', slug)
    return f"{section_id.lower()}-{slug}"
```

### 3. Index Directory Path

The index directory path may vary based on Stage 3 configuration:
- Default: `work/indices/`
- With batch size: `work/indices_100/` (if Stage 3 used `--batch-size 100`)

**Resolution:** Accept `--indices` as CLI argument with fallback logic:
```python
def find_indices_dir(work_dir: Path) -> Path:
    """Find the indices directory, checking common locations."""
    candidates = [
        work_dir / "indices",
        work_dir / "indices_100",
        work_dir / "indices_50",
    ]
    for path in candidates:
        if path.exists() and list(path.glob("*.json")):
            return path
    raise FileNotFoundError(f"No index files found in {candidates}")
```

### 4. Token Budget & Cost Estimation

**API Limits:**
- Max input tokens: 128K (claude-sonnet-4-20250514)
- Max output tokens: 8192 (configurable)
- Typical image: 1,000-3,000 tokens depending on size/complexity

**Cost Estimation (per 1K tokens, approximate):**
- Input: $0.003
- Output: $0.015

**Budget Configuration:**
```yaml
generation:
  # Token limits
  max_output_tokens: 4096  # Enough for 12 Q&A pairs

  # Cost controls
  max_cost_per_page_usd: 0.10  # Abort if estimated cost exceeds
  daily_budget_usd: 50.00      # Optional daily limit

  # Estimation (logged, not enforced)
  estimated_input_tokens_per_page: 2000
  estimated_output_tokens_per_page: 800
```

**Progress Log Enhancement:**
```csv
timestamp,page_id,status,qa_count,tokens_input,tokens_output,cost_usd
```

### 5. Image Size Preprocessing

Claude Vision API has size limits. Large scans must be preprocessed:

**Limits:**
- Max file size: 20MB
- Recommended: < 5MB for faster processing
- Max dimensions: 8192 x 8192 pixels

**Preprocessing Logic:**
```python
def preprocess_image_for_api(image_path: Path, max_size_mb: float = 5.0, max_dim: int = 4096) -> Tuple[bytes, str]:
    """
    Preprocess image for Claude API.

    - Resize if dimensions exceed max_dim
    - Reduce quality if file size exceeds max_size_mb
    - Convert to JPEG for smaller size (unless PNG needed for diagrams)

    Returns (image_bytes, media_type)
    """
    from PIL import Image
    import io

    with Image.open(image_path) as img:
        # Resize if too large
        if max(img.size) > max_dim:
            ratio = max_dim / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)

        # Convert to RGB if needed (for JPEG)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')

        # Compress to fit size limit
        quality = 85
        while quality > 20:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            if buffer.tell() <= max_size_mb * 1024 * 1024:
                buffer.seek(0)
                return buffer.read(), 'image/jpeg'
            quality -= 10

        raise ValueError(f"Cannot compress {image_path} below {max_size_mb}MB")
```

### 6. HTML Table Structure Variations

HTML techspec files may have different table structures:

**Known Variations:**
```html
<!-- 3-column with category -->
<tr><td>Engine</td><td>Displacement</td><td>2302 cc</td></tr>

<!-- 2-column (no category) -->
<tr><td>Displacement</td><td>2302 cc</td></tr>

<!-- With units in separate column -->
<tr><td>Displacement</td><td>2302</td><td>cc</td></tr>

<!-- Nested tables -->
<table><tr><td><table>...</table></td></tr></table>
```

**Robust Parsing Strategy:**
```python
def parse_html_specs(html_path: Path) -> List[Dict]:
    """
    Parse HTML with flexible table detection.

    Strategy:
    1. Find all tables (skip nested tables)
    2. Detect structure from first data row
    3. Handle 2-column, 3-column, and 4-column formats
    4. Merge value+unit columns if separated
    """
    soup = BeautifulSoup(html_path.read_text(), 'html.parser')
    specs = []

    for table in soup.find_all('table', recursive=True):
        # Skip if this table is inside another table
        if table.find_parent('table'):
            continue

        rows = table.find_all('tr')
        if len(rows) < 2:
            continue

        # Detect structure from first data row
        first_data = rows[1] if rows[0].find('th') else rows[0]
        cols = first_data.find_all(['td', 'th'])
        num_cols = len(cols)

        for row in rows:
            cells = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            if len(cells) < 2:
                continue

            if num_cols >= 3:
                # Assume: category, name, value [, unit]
                category = cells[0]
                spec_name = cells[1]
                spec_value = " ".join(cells[2:])  # Merge remaining
            else:
                # Assume: name, value
                category = ""
                spec_name = cells[0]
                spec_value = " ".join(cells[1:])

            # Skip header rows
            if spec_name.lower() in ('specification', 'spec', 'parameter', 'category'):
                continue

            specs.append({
                "category": category,
                "spec_name": spec_name,
                "spec_value": spec_value
            })

    return specs
```

### 7. Quality Validation

Q&A pairs should be validated before saving:

**Validation Rules:**
```python
def validate_qa_pair(qa: Dict, page_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a Q&A pair for quality issues.

    Returns (is_valid, list_of_warnings)
    """
    warnings = []

    # Required fields
    if not qa.get("question") or len(qa["question"]) < 10:
        return False, ["Question too short or missing"]
    if not qa.get("answer") or len(qa["answer"]) < 5:
        return False, ["Answer too short or missing"]

    # Question must be a question
    if not qa["question"].rstrip().endswith("?"):
        warnings.append("Question doesn't end with '?'")

    # Answer shouldn't be a question
    if qa["answer"].rstrip().endswith("?"):
        warnings.append("Answer ends with '?' (may be a question)")

    # Detect potential hallucination patterns
    hallucination_phrases = [
        "I cannot see", "I can't determine", "not visible",
        "image doesn't show", "unclear from the image",
        "I would need", "typically", "usually", "generally"
    ]
    answer_lower = qa["answer"].lower()
    for phrase in hallucination_phrases:
        if phrase in answer_lower:
            warnings.append(f"Potential hedge/hallucination: '{phrase}'")

    # Check for self-referential language (should be removed)
    bad_patterns = [
        "on this page", "in this image", "as shown",
        "the manual states", "according to the page"
    ]
    for pattern in bad_patterns:
        if pattern in qa["question"].lower():
            warnings.append(f"Self-referential question: '{pattern}'")

    # Validate question_type
    valid_types = {
        "factual", "procedural", "visual", "inspection", "tool",
        "safety", "navigation", "wiring", "connector", "component",
        "diagnostic", "troubleshooting", "signal", "parameter", "operation"
    }
    if qa.get("question_type") not in valid_types:
        warnings.append(f"Invalid question_type: {qa.get('question_type')}")

    return len(warnings) == 0 or not any("too short" in w or "missing" in w for w in warnings), warnings


def filter_and_validate_qa_pairs(
    qa_pairs: List[Dict],
    page_data: Dict,
    strict: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter Q&A pairs, returning (valid_pairs, rejected_pairs).

    If strict=True, reject any pair with warnings.
    If strict=False, only reject pairs that fail hard validation.
    """
    valid = []
    rejected = []

    for qa in qa_pairs:
        is_valid, warnings = validate_qa_pair(qa, page_data)

        if not is_valid or (strict and warnings):
            rejected.append({"qa": qa, "reasons": warnings})
        else:
            if warnings:
                qa["_warnings"] = warnings  # Attach for review
            valid.append(qa)

    return valid, rejected
```

**Validation Output:**
```json
{
  "qa_pairs": [...],
  "validation": {
    "total_generated": 12,
    "accepted": 10,
    "rejected": 2,
    "warnings": 3
  },
  "rejected_pairs": [
    {"qa": {...}, "reasons": ["Answer too short"]}
  ]
}
```

---

## Requirements Summary

### 04a: Image Q&A Generation

#### Inputs
- `data_src/` (source images)
- `work/classified/pages.csv` (from Stage 3)
- `work/indices/*.json` (from Stage 3)
- `config.yaml` (prompt templates, rate limits, API settings)

#### Outputs
- `work/qa_raw/{page_id}.json` (one file per page)
- `work/logs/generation_progress.csv` (progress tracking)
- `work/logs/generation_errors.csv` (failed pages)

#### Key Behaviors
1. **Context Loading**: Load index metadata to inject procedure names into prompts
2. **Prompt Selection**: Route to different prompts based on `source_type` + `content_type`
3. **API Calls**: Send image + prompt to Claude, parse JSON response
4. **Caching**: Skip pages already processed (idempotent)
5. **Rate Limiting**: Configurable delay between API calls
6. **Error Handling**: Retry with backoff; log failures; continue processing

### 04b: HTML Q&A Generation

#### Inputs
- `data_src/*.html` (M3-techspec.html, 320is-techspec.html)
- `config.yaml`

#### Outputs
- `work/qa_raw/html-{filename_stem}.json`

#### Key Behaviors
1. **HTML Parsing**: Extract spec tables with BeautifulSoup
2. **Q&A Generation**: Programmatic question variations (no API)
3. **Schema Compatibility**: Same output schema as 04a

---

## Data Schemas

### Input: Classification CSV (`work/classified/pages.csv`)

```csv
page_id,image_path,section_id,section_name,source_type,content_type,is_index,confidence
21-01,data_src/21 - Clutch/21-01.jpg,21,Clutch,service_manual,procedure,False,0.85
```

### Input: Index Metadata (`work/indices/{section_slug}.json`)

```json
{
  "section_id": "21",
  "section_name": "Clutch",
  "procedures": [
    {"code": "21 00 006", "name": "Clutch - bleed", "pages": ["21-1"]}
  ],
  "page_to_procedures": {
    "21-1": ["21 00 006", "21 11 000"]
  }
}
```

### Output: Q&A Raw JSON (`work/qa_raw/{page_id}.json`)

```json
{
  "page_id": "21-03",
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
    "prompt_template": "procedure",
    "tokens_input": 1500,
    "tokens_output": 800
  },

  "qa_pairs": [
    {
      "id": "21-03-q01",
      "question": "What should I visually inspect the clutch pressure plate for?",
      "answer": "Visually inspect the clutch for cracks, wear, and burnt spots. The pressure contact surface must be level.",
      "question_type": "inspection"
    }
  ]
}
```

### Output: HTML Q&A JSON (`work/qa_raw/html-m3-techspec.json`)

```json
{
  "page_id": "html-m3-techspec",
  "image_path": null,
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
```

---

## Configuration (`config.yaml` additions)

```yaml
# Q&A Generation settings
generation:
  # API settings
  model: claude-sonnet-4-20250514
  max_retries: 3
  retry_delay_seconds: 2
  rate_limit_delay_seconds: 1.0

  # Token limits
  max_output_tokens: 4096  # Enough for 12 Q&A pairs

  # Cost controls
  max_cost_per_page_usd: 0.10  # Abort if estimated cost exceeds
  daily_budget_usd: 50.00      # Optional daily limit (null to disable)
  warn_at_budget_percent: 80   # Warn when this % of daily budget used

  # Estimation (for logging, not enforced)
  estimated_input_tokens_per_page: 2000
  estimated_output_tokens_per_page: 800

  # Image preprocessing
  image:
    max_size_mb: 5.0       # Compress if larger
    max_dimension: 4096    # Resize if larger
    jpeg_quality: 85       # Initial quality (reduces on retry)
    min_jpeg_quality: 30   # Minimum quality before failing

  # Validation
  validation:
    strict_mode: false     # If true, reject any Q&A with warnings
    min_question_length: 10
    min_answer_length: 5

  # Questions per page by content type
  questions_per_page:
    # Service manual
    index: 6
    procedure: 12
    specification: 10
    diagram: 8
    troubleshooting: 10
    text: 6
    # Electrical manual
    wiring: 10
    pinout: 8
    fuse_chart: 8
    flowchart: 8
    # ECU technical
    signal: 8
    oscilloscope: 8

  # Skip patterns
  skip_patterns:
    - "*-blank-*"
    - "*-cover-*"
    - "*-title-*"

  # Skip content types (no useful Q&A)
  skip_content_types:
    - "index"  # Index pages don't generate Q&A

  # Batch processing
  batch_size: 10
  checkpoint_interval: 50
```

---

## Prompt Templates

### System Prompt (all sources)

```
You are generating training data for an AI assistant that will answer questions about the BMW E30 M3 and 320is service manual. The AI will see the same manual page image and must answer questions accurately based on what's visible.

Generate question-answer pairs that a mechanic working on this car would actually ask. The questions should be natural and varied. The answers must be accurate and based only on what's visible on this page.

Output ONLY valid JSON in this exact format:
[
  {
    "question": "Your question here?",
    "answer": "Your answer here.",
    "question_type": "factual"
  }
]

Valid question_type values: factual, procedural, visual, inspection, tool, safety, navigation, wiring, connector, component, diagnostic, troubleshooting, signal, parameter, operation
```

### User Prompt - Service Manual (procedure)

```
[IMAGE]

This is page {page_id} from the {section_name} section (Section {section_id}) of the BMW E30 M3 Service Manual.

{context_block}

Generate {num_questions} question-answer pairs for this page. Include a mix of:

1. **Factual questions** — Specific values, part names, specifications visible on the page
2. **Procedural questions** — How to perform tasks, what steps to follow, what sequence
3. **Visual questions** — What components are shown in diagrams, what callout numbers indicate
4. **Inspection questions** — What to look for, signs of wear, acceptance criteria
5. **Tool/supply questions** — What tools are needed, what lubricants or supplies are specified
6. **Safety/warning questions** — Important cautions, things to avoid, critical notes

Guidelines:
- Questions should be self-contained (don't say "on this page")
- Answers should be complete but concise (typically 1-3 sentences)
- For diagram callouts, describe what each number indicates
- For specifications, include the value AND units
- If something references another section, mention the cross-reference
- Don't invent information not visible on the page

Return JSON array only.
```

### User Prompt - Service Manual (specification)

```
[IMAGE]

This is a specification page from the {section_name} section (Section {section_id}) of the BMW E30 M3 Service Manual.

{context_block}

Generate {num_questions} question-answer pairs for this page. Focus on:

1. **Factual questions** — Extract specific values with their units
2. **Comparison questions** — Different specifications for different variants
3. **Tolerance questions** — Acceptable ranges, min/max values
4. **Application questions** — What component uses this specification

Guidelines:
- Always include units in answers (Nm, mm, bar, °C, L)
- Be precise with numeric values
- Note any conditions (cold, warm, new, used)
- Don't invent values not visible on the page

Return JSON array only.
```

### User Prompt - Service Manual (troubleshooting)

```
[IMAGE]

This is a troubleshooting page from the {section_name} section (Section {section_id}) of the BMW E30 M3 Service Manual.

{context_block}

Generate {num_questions} question-answer pairs for this page. Focus on:

1. **Symptom questions** — What symptoms indicate this problem?
2. **Diagnostic questions** — How to test or verify the cause?
3. **Cause questions** — What causes this symptom?
4. **Correction questions** — How to fix the problem?
5. **Cross-reference questions** — What related procedures are referenced?

Guidelines:
- Match symptoms to causes accurately
- Include specific test procedures when shown
- Reference repair codes if visible
- Don't invent diagnostic steps not on the page

Return JSON array only.
```

### User Prompt - Electrical Manual (wiring)

```
[IMAGE]

This is page {page_id} from the 1990 BMW M3 Electrical Troubleshooting Manual.

Generate {num_questions} question-answer pairs for this page. Include:

1. **Wiring questions** — Wire colors, routing, connector locations
2. **Connector questions** — Pin assignments, connector identification
3. **Component questions** — What components are shown, their functions
4. **Diagnostic questions** — How to test circuits, expected voltages
5. **Troubleshooting questions** — What to check if circuit doesn't work

Guidelines:
- Be specific about wire colors (e.g., "brown/white" not just "brown")
- Include connector designations (e.g., "X14 pin 3")
- Reference ground points by location when shown
- Don't invent pin assignments not visible on the page

Return JSON array only.
```

### User Prompt - Electrical Manual (pinout)

```
[IMAGE]

This is a connector pinout page from the 1990 BMW M3 Electrical Troubleshooting Manual.

Generate {num_questions} question-answer pairs for this page. Focus on:

1. **Pin assignment questions** — What is connected to each pin?
2. **Wire color questions** — What color wire goes to which pin?
3. **Connector ID questions** — How is this connector identified?
4. **Function questions** — What is the purpose of this connector?

Guidelines:
- Be exact with pin numbers and wire colors
- Include connector type/size if shown
- Note viewing orientation (component side vs harness side)
- Don't invent assignments not visible

Return JSON array only.
```

### User Prompt - ECU Technical (Bosch Motronic)

```
[IMAGE]

This is page {page_id} from the Bosch Motronic ML 3-1 technical documentation for the BMW E30 M3.

Generate {num_questions} question-answer pairs for this page. Include:

1. **Signal questions** — What signals are shown, their purposes
2. **Parameter questions** — Specific values, thresholds, timing
3. **Component questions** — Sensors, actuators, their functions
4. **Diagnostic questions** — How the ECU detects faults
5. **Operation questions** — How systems interact, control logic

Guidelines:
- Be precise with technical values (voltages, frequencies, timing)
- Reference sensor and actuator names as shown
- Include units for all measurements
- Don't invent values not visible on the page

Return JSON array only.
```

### Context Block Templates

**For pages with index metadata:**
```
According to the section index, this page covers:
- {code}: {name}
- {code}: {name}
```

**For specification pages:**
```
This page contains specification tables. Focus on extracting specific values with their units.
```

**For pages without context:**
```
(No context block)
```

---

## Function Signatures

### 04a: Image Q&A Generation

```python
def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file."""
    pass

def normalize_page_id(page_id: str) -> str:
    """
    Normalize page ID for matching across sources.
    Strips leading zeros from page number portion.

    Examples:
        "21-01" -> "21-1"
        "21-1"  -> "21-1"
        "ETM-001" -> "ETM-1"
    """
    pass

def derive_section_slug(section_id: str, section_name: str) -> str:
    """
    Derive section slug for index file lookup.

    Examples:
        ("21", "Clutch") -> "21-clutch"
        ("00", "Maintenance") -> "00-maintenance"
    """
    pass

def find_indices_dir(work_dir: Path) -> Path:
    """
    Find the indices directory, checking common locations.

    Checks: work/indices/, work/indices_100/, work/indices_50/
    Returns first directory that exists and contains .json files.
    Raises FileNotFoundError if none found.
    """
    pass

def preprocess_image_for_api(
    image_path: Path,
    max_size_mb: float = 5.0,
    max_dim: int = 4096
) -> Tuple[bytes, str]:
    """
    Preprocess image for Claude API.

    - Resize if dimensions exceed max_dim
    - Compress if file size exceeds max_size_mb
    - Convert RGBA/P to RGB for JPEG

    Returns (image_bytes, media_type)
    Raises ValueError if cannot compress below limit.
    """
    pass

def validate_qa_pair(qa: Dict, page_data: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a Q&A pair for quality issues.

    Checks:
    - Required fields present and minimum length
    - Question ends with '?'
    - No hallucination phrases in answer
    - No self-referential language in question
    - Valid question_type

    Returns (is_valid, list_of_warnings)
    """
    pass

def filter_and_validate_qa_pairs(
    qa_pairs: List[Dict],
    page_data: Dict,
    strict: bool = False
) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter Q&A pairs, returning (valid_pairs, rejected_pairs).

    If strict=True, reject any pair with warnings.
    If strict=False, only reject pairs that fail hard validation.
    """
    pass

def load_classification_data(csv_path: Path) -> List[Dict]:
    """
    Load classification data from pages.csv.

    Returns list of dicts with: page_id, image_path, section_id,
    section_name, source_type, content_type, is_index, confidence
    """
    pass

def load_index_metadata(indices_dir: Path) -> Dict[str, Dict]:
    """
    Load all index metadata from work/indices/*.json.

    Returns dict mapping section_slug -> index metadata
    """
    pass

def get_procedures_for_page(
    page_id: str,
    section_slug: str,
    index_metadata: Dict[str, Dict]
) -> Tuple[List[str], List[str]]:
    """
    Get procedures covered by this page from index metadata.

    Returns (procedure_codes, procedure_names)
    """
    pass

def should_skip_page(
    page_data: Dict,
    config: Dict,
    existing_output: Optional[Path]
) -> Tuple[bool, str]:
    """
    Check if page should be skipped.

    Skip reasons:
    - Already processed (output file exists)
    - Matches skip_patterns
    - content_type in skip_content_types

    Returns (should_skip, reason)
    """
    pass

def build_context_block(
    page_data: Dict,
    procedure_codes: List[str],
    procedure_names: List[str]
) -> str:
    """
    Build context block for prompt based on available metadata.
    """
    pass

def select_prompt_template(
    source_type: str,
    content_type: str
) -> Tuple[str, str]:
    """
    Select appropriate system and user prompt templates.

    Returns (system_prompt, user_prompt_template)
    """
    pass

def encode_image_base64(image_path: Path) -> Tuple[str, str]:
    """
    Encode image to base64 for Claude API.

    Returns (base64_data, media_type)
    """
    pass

def generate_qa_for_page(
    image_path: Path,
    page_data: Dict,
    context_block: str,
    num_questions: int,
    api_key: str,
    model: str,
    max_retries: int = 3
) -> Dict:
    """
    Call Claude API to generate Q&A pairs for a single page.

    Returns dict with:
    - qa_pairs: List of Q&A dicts
    - tokens_input: int
    - tokens_output: int
    - error: Optional[str]
    """
    pass

def parse_qa_response(response_text: str) -> List[Dict]:
    """
    Parse Claude's JSON response into Q&A pairs.

    Handles:
    - Clean JSON array
    - JSON wrapped in markdown code blocks
    - Malformed JSON (best effort)

    Returns list of Q&A dicts with id, question, answer, question_type
    """
    pass

def assign_qa_ids(qa_pairs: List[Dict], page_id: str) -> List[Dict]:
    """
    Assign unique IDs to Q&A pairs.

    Format: {page_id}-q{nn} (e.g., "21-03-q01")
    """
    pass

def write_qa_output(
    page_id: str,
    page_data: Dict,
    qa_pairs: List[Dict],
    generation_metadata: Dict,
    output_dir: Path
) -> Path:
    """
    Write Q&A output JSON for a single page.

    Returns path to written file.
    """
    pass

def write_progress_log(
    progress_entries: List[Dict],
    log_path: Path
):
    """
    Write/append progress log entries.

    Columns: timestamp, page_id, status, qa_count, tokens_input, tokens_output
    """
    pass

def write_error_log(
    error_entries: List[Dict],
    log_path: Path
):
    """
    Write/append error log entries.

    Columns: timestamp, page_id, error_type, error_message
    """
    pass

def process_all_pages(
    classification_data: List[Dict],
    index_metadata: Dict[str, Dict],
    config: Dict,
    output_dir: Path,
    progress_log_path: Path,
    error_log_path: Path,
    limit: Optional[int] = None,
    start_from: Optional[str] = None,
    force: bool = False
) -> Dict:
    """
    Main processing loop for all pages.

    Returns summary dict with:
    - total_pages: int
    - processed: int
    - skipped: int
    - failed: int
    - total_qa_pairs: int
    - total_tokens: int
    """
    pass
```

### 04b: HTML Q&A Generation

```python
def parse_html_specs(html_path: Path) -> List[Dict]:
    """
    Parse HTML techspec file and extract spec rows.

    Returns list of dicts with:
    - category: str (e.g., "Engine", "Transmission")
    - spec_name: str (e.g., "Displacement")
    - spec_value: str (e.g., "2302 cc")
    """
    pass

def detect_model_from_filename(html_path: Path) -> str:
    """
    Detect vehicle model from HTML filename.

    M3-techspec.html -> "M3"
    320is-techspec.html -> "320is"
    """
    pass

def generate_qa_variations(
    spec_name: str,
    spec_value: str,
    category: str,
    model: str
) -> List[Dict]:
    """
    Generate 2-3 question variations for a single spec.

    Variations:
    1. "What is the {spec_name} for the BMW E30 {model}?" -> "{spec_value}"
    2. "What is the {spec_name}?" -> "The {spec_name} for the E30 {model} is {spec_value}."
    3. (if category) "What is the {category} {spec_name}?" -> "{spec_value}"
    """
    pass

def generate_html_qa(
    html_path: Path,
    output_dir: Path
) -> Dict:
    """
    Generate Q&A from HTML spec file.

    Returns summary dict with:
    - page_id: str
    - qa_count: int
    - output_path: Path
    """
    pass
```

---

## Test-Driven Development

### Test File: `tests/test_04a_generate_qa_images.py`

```python
"""
Tests for Stage 4a: Image Q&A Generation

Run with: pytest tests/test_04a_generate_qa_images.py -v
"""

import pytest
import json
import csv
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def load_module():
    """Load the module dynamically"""
    module_path = Path(__file__).parent.parent / "scripts" / "04a_generate_qa_images.py"
    if not module_path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_qa_images", str(module_path))
    if spec and spec.loader:
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Error loading module: {e}")
            return None
    return None


generate_qa = load_module()


def get_func(name):
    """Get function from module or return None"""
    if generate_qa is None:
        return None
    return getattr(generate_qa, name, None)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_config():
    """Sample configuration dict"""
    return {
        "generation": {
            "model": "claude-sonnet-4-20250514",
            "max_retries": 3,
            "retry_delay_seconds": 1,
            "rate_limit_delay_seconds": 0.5,
            "questions_per_page": {
                "procedure": 12,
                "specification": 10,
                "diagram": 8,
                "troubleshooting": 10,
                "wiring": 10,
                "text": 6
            },
            "skip_patterns": ["*-blank-*", "*-cover-*"],
            "skip_content_types": ["index"]
        },
        "api": {
            "api_key": "test-key"
        }
    }


@pytest.fixture
def sample_classification_data():
    """Sample classification data"""
    return [
        {
            "page_id": "21-01",
            "image_path": "data_src/21 - Clutch/21-01.jpg",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "procedure",
            "is_index": "False",
            "confidence": "0.85"
        },
        {
            "page_id": "21-02",
            "image_path": "data_src/21 - Clutch/21-02.jpg",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "specification",
            "is_index": "False",
            "confidence": "0.90"
        }
    ]


@pytest.fixture
def sample_index_metadata():
    """Sample index metadata"""
    return {
        "21-clutch": {
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [
                {"code": "21 00 006", "name": "Clutch - bleed", "pages": ["21-1"]},
                {"code": "21 21 000", "name": "Clutch disc - remove and install", "pages": ["21-1", "21-2"]}
            ],
            "page_to_procedures": {
                "21-1": ["21 00 006", "21 21 000"],
                "21-2": ["21 21 000"]
            }
        }
    }


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image"""
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    img = Image.new('RGB', (100, 100), color='white')
    img.save(img_path, "JPEG")
    return img_path


# =============================================================================
# Test: Page ID Normalization
# =============================================================================

class TestNormalizePageId:
    """Test page ID normalization for cross-source matching"""

    def test_normalize_zero_padded(self):
        """Should strip leading zeros from page number"""
        normalize_page_id = get_func('normalize_page_id')
        if normalize_page_id is None:
            pytest.skip("Module not implemented yet")

        assert normalize_page_id("21-01") == "21-1"
        assert normalize_page_id("21-001") == "21-1"
        assert normalize_page_id("00-05") == "00-5"

    def test_normalize_already_normalized(self):
        """Should not change already normalized IDs"""
        normalize_page_id = get_func('normalize_page_id')
        if normalize_page_id is None:
            pytest.skip("Module not implemented yet")

        assert normalize_page_id("21-1") == "21-1"
        assert normalize_page_id("21-15") == "21-15"

    def test_normalize_etm_format(self):
        """Should handle ETM prefix format"""
        normalize_page_id = get_func('normalize_page_id')
        if normalize_page_id is None:
            pytest.skip("Module not implemented yet")

        assert normalize_page_id("ETM-001") == "ETM-1"
        assert normalize_page_id("ETM-12") == "ETM-12"

    def test_normalize_preserves_zero_page(self):
        """Should preserve page 0"""
        normalize_page_id = get_func('normalize_page_id')
        if normalize_page_id is None:
            pytest.skip("Module not implemented yet")

        assert normalize_page_id("21-00") == "21-0"
        assert normalize_page_id("21-0") == "21-0"


# =============================================================================
# Test: Section Slug Derivation
# =============================================================================

class TestDeriveSectionSlug:
    """Test section slug derivation for index file lookup"""

    def test_derive_simple_slug(self):
        """Should derive slug from section id and name"""
        derive_section_slug = get_func('derive_section_slug')
        if derive_section_slug is None:
            pytest.skip("Module not implemented yet")

        assert derive_section_slug("21", "Clutch") == "21-clutch"
        assert derive_section_slug("00", "Maintenance") == "00-maintenance"

    def test_derive_multi_word_slug(self):
        """Should handle multi-word section names"""
        derive_section_slug = get_func('derive_section_slug')
        if derive_section_slug is None:
            pytest.skip("Module not implemented yet")

        assert derive_section_slug("ETM", "Electrical Troubleshooting") == "etm-electrical-troubleshooting"

    def test_derive_removes_special_chars(self):
        """Should remove special characters"""
        derive_section_slug = get_func('derive_section_slug')
        if derive_section_slug is None:
            pytest.skip("Module not implemented yet")

        slug = derive_section_slug("12", "Brakes (ABS)")
        assert "(" not in slug
        assert ")" not in slug


# =============================================================================
# Test: Index Directory Discovery
# =============================================================================

class TestFindIndicesDir:
    """Test index directory discovery with fallbacks"""

    def test_find_default_indices(self, tmp_path):
        """Should find default indices directory"""
        find_indices_dir = get_func('find_indices_dir')
        if find_indices_dir is None:
            pytest.skip("Module not implemented yet")

        indices_dir = tmp_path / "indices"
        indices_dir.mkdir()
        (indices_dir / "21-clutch.json").write_text("{}")

        result = find_indices_dir(tmp_path)
        assert result == indices_dir

    def test_find_indices_100(self, tmp_path):
        """Should find indices_100 as fallback"""
        find_indices_dir = get_func('find_indices_dir')
        if find_indices_dir is None:
            pytest.skip("Module not implemented yet")

        indices_dir = tmp_path / "indices_100"
        indices_dir.mkdir()
        (indices_dir / "21-clutch.json").write_text("{}")

        result = find_indices_dir(tmp_path)
        assert result == indices_dir

    def test_find_raises_if_none(self, tmp_path):
        """Should raise if no indices directory found"""
        find_indices_dir = get_func('find_indices_dir')
        if find_indices_dir is None:
            pytest.skip("Module not implemented yet")

        with pytest.raises(FileNotFoundError):
            find_indices_dir(tmp_path)

    def test_find_ignores_empty_dir(self, tmp_path):
        """Should ignore empty directories"""
        find_indices_dir = get_func('find_indices_dir')
        if find_indices_dir is None:
            pytest.skip("Module not implemented yet")

        # Empty indices dir
        (tmp_path / "indices").mkdir()
        # Populated indices_100
        indices_100 = tmp_path / "indices_100"
        indices_100.mkdir()
        (indices_100 / "21-clutch.json").write_text("{}")

        result = find_indices_dir(tmp_path)
        assert result == indices_100


# =============================================================================
# Test: Image Preprocessing
# =============================================================================

class TestPreprocessImageForApi:
    """Test image preprocessing for API size limits"""

    def test_small_image_unchanged(self, tmp_path):
        """Should return small images without modification"""
        preprocess_image_for_api = get_func('preprocess_image_for_api')
        if preprocess_image_for_api is None:
            pytest.skip("Module not implemented yet")

        from PIL import Image
        img_path = tmp_path / "small.jpg"
        img = Image.new('RGB', (100, 100), color='white')
        img.save(img_path, "JPEG")

        data, media_type = preprocess_image_for_api(img_path)

        assert isinstance(data, bytes)
        assert media_type == "image/jpeg"
        assert len(data) < 1024 * 1024  # Less than 1MB

    def test_large_image_resized(self, tmp_path):
        """Should resize images exceeding max dimensions"""
        preprocess_image_for_api = get_func('preprocess_image_for_api')
        if preprocess_image_for_api is None:
            pytest.skip("Module not implemented yet")

        from PIL import Image
        img_path = tmp_path / "large.jpg"
        img = Image.new('RGB', (8000, 6000), color='blue')
        img.save(img_path, "JPEG", quality=95)

        data, media_type = preprocess_image_for_api(img_path, max_dim=2048)

        # Verify it was resized by checking the returned image
        from io import BytesIO
        result_img = Image.open(BytesIO(data))
        assert max(result_img.size) <= 2048

    def test_rgba_converted_to_rgb(self, tmp_path):
        """Should convert RGBA to RGB for JPEG"""
        preprocess_image_for_api = get_func('preprocess_image_for_api')
        if preprocess_image_for_api is None:
            pytest.skip("Module not implemented yet")

        from PIL import Image
        img_path = tmp_path / "transparent.png"
        img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        img.save(img_path, "PNG")

        data, media_type = preprocess_image_for_api(img_path)

        assert media_type == "image/jpeg"


# =============================================================================
# Test: Q&A Validation
# =============================================================================

class TestValidateQAPair:
    """Test Q&A pair validation"""

    def test_valid_qa_pair(self):
        """Should accept valid Q&A pair"""
        validate_qa_pair = get_func('validate_qa_pair')
        if validate_qa_pair is None:
            pytest.skip("Module not implemented yet")

        qa = {
            "question": "What is the torque specification for the clutch bolts?",
            "answer": "The torque specification is 25 Nm.",
            "question_type": "factual"
        }

        is_valid, warnings = validate_qa_pair(qa, {})

        assert is_valid is True
        assert len(warnings) == 0

    def test_reject_short_question(self):
        """Should reject questions that are too short"""
        validate_qa_pair = get_func('validate_qa_pair')
        if validate_qa_pair is None:
            pytest.skip("Module not implemented yet")

        qa = {"question": "What?", "answer": "Something.", "question_type": "factual"}

        is_valid, warnings = validate_qa_pair(qa, {})

        assert is_valid is False
        assert any("too short" in w.lower() for w in warnings)

    def test_warn_missing_question_mark(self):
        """Should warn if question doesn't end with '?'"""
        validate_qa_pair = get_func('validate_qa_pair')
        if validate_qa_pair is None:
            pytest.skip("Module not implemented yet")

        qa = {
            "question": "Tell me about the clutch specifications",
            "answer": "The clutch is rated for 240 Nm.",
            "question_type": "factual"
        }

        is_valid, warnings = validate_qa_pair(qa, {})

        assert any("?" in w for w in warnings)

    def test_warn_hallucination_phrases(self):
        """Should warn on hallucination indicator phrases"""
        validate_qa_pair = get_func('validate_qa_pair')
        if validate_qa_pair is None:
            pytest.skip("Module not implemented yet")

        qa = {
            "question": "What is the oil capacity?",
            "answer": "I cannot see the oil capacity on this page, but typically it would be around 4 liters.",
            "question_type": "factual"
        }

        is_valid, warnings = validate_qa_pair(qa, {})

        assert any("hallucination" in w.lower() for w in warnings)

    def test_warn_self_referential(self):
        """Should warn on self-referential questions"""
        validate_qa_pair = get_func('validate_qa_pair')
        if validate_qa_pair is None:
            pytest.skip("Module not implemented yet")

        qa = {
            "question": "What is shown on this page?",
            "answer": "The clutch assembly diagram.",
            "question_type": "visual"
        }

        is_valid, warnings = validate_qa_pair(qa, {})

        assert any("self-referential" in w.lower() for w in warnings)

    def test_warn_invalid_question_type(self):
        """Should warn on invalid question_type"""
        validate_qa_pair = get_func('validate_qa_pair')
        if validate_qa_pair is None:
            pytest.skip("Module not implemented yet")

        qa = {
            "question": "What is the torque spec?",
            "answer": "25 Nm.",
            "question_type": "invalid_type"
        }

        is_valid, warnings = validate_qa_pair(qa, {})

        assert any("question_type" in w.lower() for w in warnings)


class TestFilterAndValidateQAPairs:
    """Test batch Q&A filtering"""

    def test_filter_separates_valid_invalid(self):
        """Should separate valid and invalid pairs"""
        filter_and_validate = get_func('filter_and_validate_qa_pairs')
        if filter_and_validate is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = [
            {"question": "What is the torque specification?", "answer": "25 Nm.", "question_type": "factual"},
            {"question": "?", "answer": "X", "question_type": "factual"},  # Too short
            {"question": "What is the bolt size?", "answer": "M8.", "question_type": "factual"},
        ]

        valid, rejected = filter_and_validate(qa_pairs, {})

        assert len(valid) == 2
        assert len(rejected) == 1

    def test_strict_mode_rejects_warnings(self):
        """Should reject pairs with warnings in strict mode"""
        filter_and_validate = get_func('filter_and_validate_qa_pairs')
        if filter_and_validate is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = [
            {"question": "What is shown on this page?", "answer": "The diagram.", "question_type": "visual"},
        ]

        # Non-strict: accepted with warning
        valid, _ = filter_and_validate(qa_pairs, {}, strict=False)
        assert len(valid) == 1

        # Strict: rejected
        valid, rejected = filter_and_validate(qa_pairs, {}, strict=True)
        assert len(valid) == 0
        assert len(rejected) == 1


# =============================================================================
# Test: Classification Data Loading
# =============================================================================

class TestLoadClassificationData:
    """Test loading classification CSV"""

    def test_load_classification_data_valid(self, tmp_path):
        """Should load valid classification CSV"""
        load_classification_data = get_func('load_classification_data')
        if load_classification_data is None:
            pytest.skip("Module not implemented yet")

        csv_path = tmp_path / "pages.csv"
        csv_path.write_text("""page_id,image_path,section_id,section_name,source_type,content_type,is_index,confidence
21-01,data_src/21 - Clutch/21-01.jpg,21,Clutch,service_manual,procedure,False,0.85
21-02,data_src/21 - Clutch/21-02.jpg,21,Clutch,service_manual,specification,False,0.90
""")

        data = load_classification_data(csv_path)

        assert len(data) == 2
        assert data[0]["page_id"] == "21-01"
        assert data[0]["content_type"] == "procedure"
        assert data[1]["content_type"] == "specification"

    def test_load_classification_data_missing_file(self, tmp_path):
        """Should raise error for missing file"""
        load_classification_data = get_func('load_classification_data')
        if load_classification_data is None:
            pytest.skip("Module not implemented yet")

        with pytest.raises(FileNotFoundError):
            load_classification_data(tmp_path / "nonexistent.csv")


# =============================================================================
# Test: Index Metadata Loading
# =============================================================================

class TestLoadIndexMetadata:
    """Test loading index metadata from JSON files"""

    def test_load_index_metadata_valid(self, tmp_path):
        """Should load all index JSON files"""
        load_index_metadata = get_func('load_index_metadata')
        if load_index_metadata is None:
            pytest.skip("Module not implemented yet")

        indices_dir = tmp_path / "indices"
        indices_dir.mkdir()

        (indices_dir / "21-clutch.json").write_text(json.dumps({
            "section_id": "21",
            "section_name": "Clutch",
            "procedures": [{"code": "21 00 006", "name": "Clutch - bleed", "pages": ["21-1"]}],
            "page_to_procedures": {"21-1": ["21 00 006"]}
        }))

        metadata = load_index_metadata(indices_dir)

        assert "21-clutch" in metadata
        assert metadata["21-clutch"]["section_id"] == "21"

    def test_load_index_metadata_empty_dir(self, tmp_path):
        """Should return empty dict for empty directory"""
        load_index_metadata = get_func('load_index_metadata')
        if load_index_metadata is None:
            pytest.skip("Module not implemented yet")

        indices_dir = tmp_path / "indices"
        indices_dir.mkdir()

        metadata = load_index_metadata(indices_dir)

        assert metadata == {}

    def test_load_index_metadata_missing_dir(self, tmp_path):
        """Should return empty dict for missing directory"""
        load_index_metadata = get_func('load_index_metadata')
        if load_index_metadata is None:
            pytest.skip("Module not implemented yet")

        metadata = load_index_metadata(tmp_path / "nonexistent")

        assert metadata == {}


# =============================================================================
# Test: Procedure Lookup
# =============================================================================

class TestGetProceduresForPage:
    """Test looking up procedures for a page"""

    def test_get_procedures_found(self, sample_index_metadata):
        """Should return procedures for matching page"""
        get_procedures_for_page = get_func('get_procedures_for_page')
        if get_procedures_for_page is None:
            pytest.skip("Module not implemented yet")

        codes, names = get_procedures_for_page("21-1", "21-clutch", sample_index_metadata)

        assert "21 00 006" in codes
        assert "21 21 000" in codes
        assert "Clutch - bleed" in names

    def test_get_procedures_not_found(self, sample_index_metadata):
        """Should return empty lists for unknown page"""
        get_procedures_for_page = get_func('get_procedures_for_page')
        if get_procedures_for_page is None:
            pytest.skip("Module not implemented yet")

        codes, names = get_procedures_for_page("99-99", "21-clutch", sample_index_metadata)

        assert codes == []
        assert names == []

    def test_get_procedures_unknown_section(self, sample_index_metadata):
        """Should return empty lists for unknown section"""
        get_procedures_for_page = get_func('get_procedures_for_page')
        if get_procedures_for_page is None:
            pytest.skip("Module not implemented yet")

        codes, names = get_procedures_for_page("21-1", "unknown-section", sample_index_metadata)

        assert codes == []
        assert names == []


# =============================================================================
# Test: Skip Logic
# =============================================================================

class TestShouldSkipPage:
    """Test page skip logic"""

    def test_skip_existing_output(self, tmp_path, sample_config):
        """Should skip page if output already exists"""
        should_skip_page = get_func('should_skip_page')
        if should_skip_page is None:
            pytest.skip("Module not implemented yet")

        output_file = tmp_path / "21-01.json"
        output_file.write_text("{}")

        page_data = {"page_id": "21-01", "content_type": "procedure"}

        should_skip, reason = should_skip_page(page_data, sample_config, output_file)

        assert should_skip is True
        assert "exists" in reason.lower()

    def test_skip_index_content_type(self, sample_config):
        """Should skip index pages"""
        should_skip_page = get_func('should_skip_page')
        if should_skip_page is None:
            pytest.skip("Module not implemented yet")

        page_data = {"page_id": "21-00-index", "content_type": "index"}

        should_skip, reason = should_skip_page(page_data, sample_config, None)

        assert should_skip is True
        assert "index" in reason.lower()

    def test_skip_blank_pattern(self, sample_config):
        """Should skip pages matching skip patterns"""
        should_skip_page = get_func('should_skip_page')
        if should_skip_page is None:
            pytest.skip("Module not implemented yet")

        page_data = {"page_id": "21-blank-01", "content_type": "text"}

        should_skip, reason = should_skip_page(page_data, sample_config, None)

        assert should_skip is True
        assert "pattern" in reason.lower()

    def test_no_skip_normal_page(self, sample_config):
        """Should not skip normal procedure page"""
        should_skip_page = get_func('should_skip_page')
        if should_skip_page is None:
            pytest.skip("Module not implemented yet")

        page_data = {"page_id": "21-03", "content_type": "procedure"}

        should_skip, reason = should_skip_page(page_data, sample_config, None)

        assert should_skip is False


# =============================================================================
# Test: Context Block Building
# =============================================================================

class TestBuildContextBlock:
    """Test context block generation"""

    def test_context_with_procedures(self):
        """Should include procedure names when available"""
        build_context_block = get_func('build_context_block')
        if build_context_block is None:
            pytest.skip("Module not implemented yet")

        page_data = {"content_type": "procedure"}
        codes = ["21 00 006", "21 21 000"]
        names = ["Clutch - bleed", "Clutch disc - remove and install"]

        context = build_context_block(page_data, codes, names)

        assert "21 00 006" in context
        assert "Clutch - bleed" in context
        assert "21 21 000" in context

    def test_context_without_procedures(self):
        """Should return minimal context when no procedures"""
        build_context_block = get_func('build_context_block')
        if build_context_block is None:
            pytest.skip("Module not implemented yet")

        page_data = {"content_type": "specification"}

        context = build_context_block(page_data, [], [])

        # Should still return something (possibly empty or default)
        assert isinstance(context, str)

    def test_context_for_specification(self):
        """Should include spec-specific context"""
        build_context_block = get_func('build_context_block')
        if build_context_block is None:
            pytest.skip("Module not implemented yet")

        page_data = {"content_type": "specification"}

        context = build_context_block(page_data, [], [])

        # May include "specification" or "values" in context
        assert isinstance(context, str)


# =============================================================================
# Test: Prompt Selection
# =============================================================================

class TestSelectPromptTemplate:
    """Test prompt template selection"""

    def test_service_manual_procedure(self):
        """Should select procedure prompt for service manual procedure"""
        select_prompt_template = get_func('select_prompt_template')
        if select_prompt_template is None:
            pytest.skip("Module not implemented yet")

        system_prompt, user_template = select_prompt_template("service_manual", "procedure")

        assert "BMW E30 M3" in system_prompt
        assert "procedural" in user_template.lower() or "procedure" in user_template.lower()

    def test_electrical_manual_wiring(self):
        """Should select wiring prompt for electrical manual"""
        select_prompt_template = get_func('select_prompt_template')
        if select_prompt_template is None:
            pytest.skip("Module not implemented yet")

        system_prompt, user_template = select_prompt_template("electrical_manual", "wiring")

        assert "wire" in user_template.lower() or "electrical" in user_template.lower()

    def test_ecu_technical_diagram(self):
        """Should select ECU prompt for Bosch Motronic"""
        select_prompt_template = get_func('select_prompt_template')
        if select_prompt_template is None:
            pytest.skip("Module not implemented yet")

        system_prompt, user_template = select_prompt_template("ecu_technical", "diagram")

        assert isinstance(system_prompt, str)
        assert isinstance(user_template, str)

    def test_fallback_for_unknown(self):
        """Should return fallback prompt for unknown types"""
        select_prompt_template = get_func('select_prompt_template')
        if select_prompt_template is None:
            pytest.skip("Module not implemented yet")

        system_prompt, user_template = select_prompt_template("unknown", "unknown")

        # Should not raise, should return some default
        assert system_prompt is not None
        assert user_template is not None


# =============================================================================
# Test: Image Encoding
# =============================================================================

class TestEncodeImageBase64:
    """Test image base64 encoding"""

    def test_encode_jpg(self, temp_image):
        """Should encode JPG to base64"""
        encode_image_base64 = get_func('encode_image_base64')
        if encode_image_base64 is None:
            pytest.skip("Module not implemented yet")

        data, media_type = encode_image_base64(temp_image)

        assert isinstance(data, str)
        assert len(data) > 0
        assert media_type == "image/jpeg"

    def test_encode_png(self, tmp_path):
        """Should encode PNG to base64"""
        encode_image_base64 = get_func('encode_image_base64')
        if encode_image_base64 is None:
            pytest.skip("Module not implemented yet")

        from PIL import Image
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (50, 50), color='blue')
        img.save(img_path, "PNG")

        data, media_type = encode_image_base64(img_path)

        assert media_type == "image/png"

    def test_encode_missing_file(self, tmp_path):
        """Should raise error for missing file"""
        encode_image_base64 = get_func('encode_image_base64')
        if encode_image_base64 is None:
            pytest.skip("Module not implemented yet")

        with pytest.raises(FileNotFoundError):
            encode_image_base64(tmp_path / "nonexistent.jpg")


# =============================================================================
# Test: Q&A Response Parsing
# =============================================================================

class TestParseQAResponse:
    """Test parsing Claude API responses"""

    def test_parse_clean_json(self):
        """Should parse clean JSON array"""
        parse_qa_response = get_func('parse_qa_response')
        if parse_qa_response is None:
            pytest.skip("Module not implemented yet")

        response = '''[
            {"question": "What is X?", "answer": "X is Y.", "question_type": "factual"},
            {"question": "How to do Z?", "answer": "Do Z by...", "question_type": "procedural"}
        ]'''

        qa_pairs = parse_qa_response(response)

        assert len(qa_pairs) == 2
        assert qa_pairs[0]["question"] == "What is X?"
        assert qa_pairs[1]["question_type"] == "procedural"

    def test_parse_markdown_wrapped(self):
        """Should parse JSON wrapped in markdown code blocks"""
        parse_qa_response = get_func('parse_qa_response')
        if parse_qa_response is None:
            pytest.skip("Module not implemented yet")

        response = '''```json
[
    {"question": "Test?", "answer": "Answer.", "question_type": "factual"}
]
```'''

        qa_pairs = parse_qa_response(response)

        assert len(qa_pairs) == 1
        assert qa_pairs[0]["question"] == "Test?"

    def test_parse_with_trailing_text(self):
        """Should handle response with trailing explanation text"""
        parse_qa_response = get_func('parse_qa_response')
        if parse_qa_response is None:
            pytest.skip("Module not implemented yet")

        response = '''[{"question": "Q?", "answer": "A.", "question_type": "factual"}]

I generated this based on the visible content.'''

        qa_pairs = parse_qa_response(response)

        assert len(qa_pairs) == 1

    def test_parse_empty_response(self):
        """Should return empty list for empty response"""
        parse_qa_response = get_func('parse_qa_response')
        if parse_qa_response is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = parse_qa_response("")

        assert qa_pairs == []

    def test_parse_malformed_json(self):
        """Should handle malformed JSON gracefully"""
        parse_qa_response = get_func('parse_qa_response')
        if parse_qa_response is None:
            pytest.skip("Module not implemented yet")

        response = '''[{"question": "Q?", "answer": "A." broken json'''

        # Should not raise, should return empty or partial
        qa_pairs = parse_qa_response(response)
        assert isinstance(qa_pairs, list)


# =============================================================================
# Test: Q&A ID Assignment
# =============================================================================

class TestAssignQAIds:
    """Test Q&A ID assignment"""

    def test_assign_sequential_ids(self):
        """Should assign sequential IDs"""
        assign_qa_ids = get_func('assign_qa_ids')
        if assign_qa_ids is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = [
            {"question": "Q1?", "answer": "A1."},
            {"question": "Q2?", "answer": "A2."},
            {"question": "Q3?", "answer": "A3."}
        ]

        result = assign_qa_ids(qa_pairs, "21-03")

        assert result[0]["id"] == "21-03-q01"
        assert result[1]["id"] == "21-03-q02"
        assert result[2]["id"] == "21-03-q03"

    def test_preserve_existing_fields(self):
        """Should preserve existing fields"""
        assign_qa_ids = get_func('assign_qa_ids')
        if assign_qa_ids is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = [
            {"question": "Q?", "answer": "A.", "question_type": "factual"}
        ]

        result = assign_qa_ids(qa_pairs, "21-03")

        assert result[0]["question"] == "Q?"
        assert result[0]["answer"] == "A."
        assert result[0]["question_type"] == "factual"


# =============================================================================
# Test: Output Writing
# =============================================================================

class TestWriteQAOutput:
    """Test Q&A output file writing"""

    def test_write_output_schema(self, tmp_path):
        """Should write correct JSON schema"""
        write_qa_output = get_func('write_qa_output')
        if write_qa_output is None:
            pytest.skip("Module not implemented yet")

        page_data = {
            "page_id": "21-03",
            "image_path": "data_src/21 - Clutch/21-03.jpg",
            "section_id": "21",
            "section_name": "Clutch",
            "source_type": "service_manual",
            "content_type": "procedure"
        }
        qa_pairs = [
            {"id": "21-03-q01", "question": "Q?", "answer": "A.", "question_type": "factual"}
        ]
        generation_metadata = {
            "model": "claude-sonnet-4-20250514",
            "timestamp": "2025-01-15T10:30:00Z",
            "prompt_template": "procedure",
            "tokens_input": 1500,
            "tokens_output": 500
        }

        output_path = write_qa_output("21-03", page_data, qa_pairs, generation_metadata, tmp_path)

        assert output_path.exists()

        with open(output_path) as f:
            data = json.load(f)

        assert data["page_id"] == "21-03"
        assert data["source_type"] == "service_manual"
        assert len(data["qa_pairs"]) == 1
        assert data["generation"]["model"] == "claude-sonnet-4-20250514"

    def test_write_creates_directory(self, tmp_path):
        """Should create output directory if needed"""
        write_qa_output = get_func('write_qa_output')
        if write_qa_output is None:
            pytest.skip("Module not implemented yet")

        output_dir = tmp_path / "nested" / "qa_raw"

        page_data = {"page_id": "21-03", "image_path": "", "section_id": "21",
                     "section_name": "Clutch", "source_type": "service_manual",
                     "content_type": "procedure"}

        output_path = write_qa_output("21-03", page_data, [], {}, output_dir)

        assert output_path.exists()


# =============================================================================
# Test: Progress/Error Logging
# =============================================================================

class TestProgressLogging:
    """Test progress and error logging"""

    def test_write_progress_log(self, tmp_path):
        """Should write progress log with correct schema"""
        write_progress_log = get_func('write_progress_log')
        if write_progress_log is None:
            pytest.skip("Module not implemented yet")

        log_path = tmp_path / "progress.csv"
        entries = [
            {
                "timestamp": "2025-01-15T10:30:00",
                "page_id": "21-03",
                "status": "success",
                "qa_count": 12,
                "tokens_input": 1500,
                "tokens_output": 500
            }
        ]

        write_progress_log(entries, log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["page_id"] == "21-03"
        assert rows[0]["status"] == "success"

    def test_write_error_log(self, tmp_path):
        """Should write error log with correct schema"""
        write_error_log = get_func('write_error_log')
        if write_error_log is None:
            pytest.skip("Module not implemented yet")

        log_path = tmp_path / "errors.csv"
        entries = [
            {
                "timestamp": "2025-01-15T10:30:00",
                "page_id": "21-03",
                "error_type": "api_error",
                "error_message": "Rate limit exceeded"
            }
        ]

        write_error_log(entries, log_path)

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["error_type"] == "api_error"


# =============================================================================
# Test: API Integration (Mocked)
# =============================================================================

class TestGenerateQAForPage:
    """Test Claude API call (mocked)"""

    @patch('anthropic.Anthropic')
    def test_generate_qa_success(self, mock_anthropic_class, temp_image):
        """Should generate Q&A pairs from API response"""
        generate_qa_for_page = get_func('generate_qa_for_page')
        if generate_qa_for_page is None:
            pytest.skip("Module not implemented yet")

        # Mock API response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text='''[
            {"question": "Test question?", "answer": "Test answer.", "question_type": "factual"}
        ]''')]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        page_data = {
            "page_id": "21-03",
            "source_type": "service_manual",
            "content_type": "procedure"
        }

        result = generate_qa_for_page(
            image_path=temp_image,
            page_data=page_data,
            context_block="Test context",
            num_questions=10,
            api_key="test-key",
            model="claude-sonnet-4-20250514"
        )

        assert result["error"] is None
        assert len(result["qa_pairs"]) == 1
        assert result["qa_pairs"][0]["question"] == "Test question?"

    @patch('anthropic.Anthropic')
    def test_generate_qa_api_error_retry(self, mock_anthropic_class, temp_image):
        """Should retry on API error"""
        generate_qa_for_page = get_func('generate_qa_for_page')
        if generate_qa_for_page is None:
            pytest.skip("Module not implemented yet")

        mock_client = Mock()
        # First call fails, second succeeds
        mock_response = Mock()
        mock_response.content = [Mock(text='[{"question": "Q?", "answer": "A.", "question_type": "factual"}]')]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)

        mock_client.messages.create.side_effect = [
            Exception("API Error"),
            mock_response
        ]
        mock_anthropic_class.return_value = mock_client

        page_data = {"page_id": "21-03", "source_type": "service_manual", "content_type": "procedure"}

        result = generate_qa_for_page(
            image_path=temp_image,
            page_data=page_data,
            context_block="",
            num_questions=10,
            api_key="test-key",
            model="test-model",
            max_retries=3
        )

        # Should succeed after retry
        assert result["error"] is None or len(result["qa_pairs"]) > 0


# =============================================================================
# Test: CLI
# =============================================================================

class TestCLI:
    """Test command-line interface"""

    def test_help_output(self):
        """Should show help with --help"""
        import subprocess
        script_path = Path(__file__).parent.parent / "scripts" / "04a_generate_qa_images.py"
        if not script_path.exists():
            pytest.skip("Script not implemented yet")

        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert '--classified' in result.stdout or '--classification' in result.stdout
        assert '--indices' in result.stdout
        assert '--output' in result.stdout
        assert '--config' in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Test File: `tests/test_04b_generate_qa_html.py`

```python
"""
Tests for Stage 4b: HTML Q&A Generation

Run with: pytest tests/test_04b_generate_qa_html.py -v
"""

import pytest
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def load_module():
    """Load the module dynamically"""
    module_path = Path(__file__).parent.parent / "scripts" / "04b_generate_qa_html.py"
    if not module_path.exists():
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_qa_html", str(module_path))
    if spec and spec.loader:
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Error loading module: {e}")
            return None
    return None


generate_qa_html = load_module()


def get_func(name):
    """Get function from module or return None"""
    if generate_qa_html is None:
        return None
    return getattr(generate_qa_html, name, None)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_m3_html(tmp_path):
    """Create sample M3 techspec HTML"""
    html_content = """<!DOCTYPE html>
<html>
<head><title>BMW E30 M3 Technical Specifications</title></head>
<body>
<h1>BMW E30 M3 Technical Specifications</h1>
<table>
    <tr><th>Category</th><th>Specification</th><th>Value</th></tr>
    <tr><td>Engine</td><td>Displacement</td><td>2302 cc</td></tr>
    <tr><td>Engine</td><td>Bore x Stroke</td><td>93.4 x 84.0 mm</td></tr>
    <tr><td>Engine</td><td>Compression Ratio</td><td>10.5:1</td></tr>
    <tr><td>Engine</td><td>Power Output</td><td>195 hp @ 6750 rpm</td></tr>
    <tr><td>Engine</td><td>Torque</td><td>240 Nm @ 4750 rpm</td></tr>
    <tr><td>Transmission</td><td>Type</td><td>Getrag 265/5</td></tr>
    <tr><td>Transmission</td><td>Gear Ratios 1st</td><td>3.72:1</td></tr>
    <tr><td>Dimensions</td><td>Length</td><td>4360 mm</td></tr>
    <tr><td>Dimensions</td><td>Width</td><td>1680 mm</td></tr>
    <tr><td>Performance</td><td>Top Speed</td><td>235 km/h</td></tr>
    <tr><td>Performance</td><td>0-100 km/h</td><td>6.7 seconds</td></tr>
</table>
</body>
</html>"""

    html_path = tmp_path / "M3-techspec.html"
    html_path.write_text(html_content)
    return html_path


@pytest.fixture
def sample_320is_html(tmp_path):
    """Create sample 320is techspec HTML"""
    html_content = """<!DOCTYPE html>
<html>
<head><title>BMW E30 320is Technical Specifications</title></head>
<body>
<h1>BMW E30 320is Technical Specifications</h1>
<table>
    <tr><th>Specification</th><th>Value</th></tr>
    <tr><td>Displacement</td><td>1990 cc</td></tr>
    <tr><td>Power Output</td><td>192 hp @ 6900 rpm</td></tr>
</table>
</body>
</html>"""

    html_path = tmp_path / "320is-techspec.html"
    html_path.write_text(html_content)
    return html_path


# =============================================================================
# Test: HTML Parsing
# =============================================================================

class TestParseHTMLSpecs:
    """Test HTML spec table parsing"""

    def test_parse_m3_specs(self, sample_m3_html):
        """Should parse M3 techspec table"""
        parse_html_specs = get_func('parse_html_specs')
        if parse_html_specs is None:
            pytest.skip("Module not implemented yet")

        specs = parse_html_specs(sample_m3_html)

        assert len(specs) >= 10

        # Check specific specs exist
        displacement = next((s for s in specs if "displacement" in s["spec_name"].lower()), None)
        assert displacement is not None
        assert displacement["spec_value"] == "2302 cc"

        power = next((s for s in specs if "power" in s["spec_name"].lower()), None)
        assert power is not None
        assert "195 hp" in power["spec_value"]

    def test_parse_specs_with_categories(self, sample_m3_html):
        """Should extract categories when present"""
        parse_html_specs = get_func('parse_html_specs')
        if parse_html_specs is None:
            pytest.skip("Module not implemented yet")

        specs = parse_html_specs(sample_m3_html)

        engine_specs = [s for s in specs if s.get("category") == "Engine"]
        assert len(engine_specs) >= 4

    def test_parse_missing_file(self, tmp_path):
        """Should raise error for missing HTML file"""
        parse_html_specs = get_func('parse_html_specs')
        if parse_html_specs is None:
            pytest.skip("Module not implemented yet")

        with pytest.raises(FileNotFoundError):
            parse_html_specs(tmp_path / "nonexistent.html")

    def test_parse_empty_table(self, tmp_path):
        """Should return empty list for HTML with no tables"""
        parse_html_specs = get_func('parse_html_specs')
        if parse_html_specs is None:
            pytest.skip("Module not implemented yet")

        html_path = tmp_path / "empty.html"
        html_path.write_text("<html><body><p>No tables here</p></body></html>")

        specs = parse_html_specs(html_path)

        assert specs == []


# =============================================================================
# Test: Model Detection
# =============================================================================

class TestDetectModelFromFilename:
    """Test vehicle model detection from filename"""

    def test_detect_m3(self, sample_m3_html):
        """Should detect M3 from filename"""
        detect_model = get_func('detect_model_from_filename')
        if detect_model is None:
            pytest.skip("Module not implemented yet")

        model = detect_model(sample_m3_html)

        assert model == "M3"

    def test_detect_320is(self, sample_320is_html):
        """Should detect 320is from filename"""
        detect_model = get_func('detect_model_from_filename')
        if detect_model is None:
            pytest.skip("Module not implemented yet")

        model = detect_model(sample_320is_html)

        assert model == "320is"

    def test_detect_unknown(self, tmp_path):
        """Should return 'E30' for unknown filename pattern"""
        detect_model = get_func('detect_model_from_filename')
        if detect_model is None:
            pytest.skip("Module not implemented yet")

        html_path = tmp_path / "specs.html"
        html_path.write_text("<html></html>")

        model = detect_model(html_path)

        assert model in ["E30", "unknown", "BMW"]


# =============================================================================
# Test: Q&A Variation Generation
# =============================================================================

class TestGenerateQAVariations:
    """Test Q&A variation generation"""

    def test_generate_basic_variations(self):
        """Should generate 2-3 variations per spec"""
        generate_qa_variations = get_func('generate_qa_variations')
        if generate_qa_variations is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = generate_qa_variations(
            spec_name="Displacement",
            spec_value="2302 cc",
            category="Engine",
            model="M3"
        )

        assert len(qa_pairs) >= 2
        assert len(qa_pairs) <= 4

    def test_variation_includes_model(self):
        """Should include model name in some variations"""
        generate_qa_variations = get_func('generate_qa_variations')
        if generate_qa_variations is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = generate_qa_variations(
            spec_name="Power Output",
            spec_value="195 hp",
            category="Engine",
            model="M3"
        )

        # At least one variation should mention M3
        has_model = any("M3" in qa["question"] or "M3" in qa["answer"] for qa in qa_pairs)
        assert has_model

    def test_variation_question_types(self):
        """All variations should have question_type=factual"""
        generate_qa_variations = get_func('generate_qa_variations')
        if generate_qa_variations is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = generate_qa_variations(
            spec_name="Torque",
            spec_value="240 Nm",
            category="Engine",
            model="M3"
        )

        assert all(qa.get("question_type") == "factual" for qa in qa_pairs)

    def test_variation_without_category(self):
        """Should work without category"""
        generate_qa_variations = get_func('generate_qa_variations')
        if generate_qa_variations is None:
            pytest.skip("Module not implemented yet")

        qa_pairs = generate_qa_variations(
            spec_name="Top Speed",
            spec_value="235 km/h",
            category="",
            model="M3"
        )

        assert len(qa_pairs) >= 2


# =============================================================================
# Test: Full HTML Q&A Generation
# =============================================================================

class TestGenerateHTMLQA:
    """Test full HTML Q&A generation"""

    def test_generate_m3_qa(self, sample_m3_html, tmp_path):
        """Should generate Q&A from M3 techspec"""
        generate_html_qa = get_func('generate_html_qa')
        if generate_html_qa is None:
            pytest.skip("Module not implemented yet")

        output_dir = tmp_path / "qa_raw"

        result = generate_html_qa(sample_m3_html, output_dir)

        assert result["page_id"] == "html-m3-techspec"
        assert result["qa_count"] >= 20  # 10+ specs * 2+ variations
        assert result["output_path"].exists()

    def test_output_schema(self, sample_m3_html, tmp_path):
        """Should write correct output schema"""
        generate_html_qa = get_func('generate_html_qa')
        if generate_html_qa is None:
            pytest.skip("Module not implemented yet")

        output_dir = tmp_path / "qa_raw"
        result = generate_html_qa(sample_m3_html, output_dir)

        with open(result["output_path"]) as f:
            data = json.load(f)

        assert data["page_id"] == "html-m3-techspec"
        assert data["image_path"] is None
        assert data["source_type"] == "html_specs"
        assert data["content_type"] == "specification"
        assert "generation" in data
        assert data["generation"]["method"] == "html_parse"
        assert "qa_pairs" in data
        assert len(data["qa_pairs"]) > 0

    def test_qa_pairs_have_ids(self, sample_m3_html, tmp_path):
        """Should assign IDs to Q&A pairs"""
        generate_html_qa = get_func('generate_html_qa')
        if generate_html_qa is None:
            pytest.skip("Module not implemented yet")

        output_dir = tmp_path / "qa_raw"
        result = generate_html_qa(sample_m3_html, output_dir)

        with open(result["output_path"]) as f:
            data = json.load(f)

        for qa in data["qa_pairs"]:
            assert "id" in qa
            assert qa["id"].startswith("html-m3-techspec-q")


# =============================================================================
# Test: Idempotency
# =============================================================================

class TestIdempotency:
    """Test idempotent behavior"""

    def test_skip_existing_output(self, sample_m3_html, tmp_path):
        """Should skip if output already exists"""
        generate_html_qa = get_func('generate_html_qa')
        if generate_html_qa is None:
            pytest.skip("Module not implemented yet")

        output_dir = tmp_path / "qa_raw"

        # First run
        result1 = generate_html_qa(sample_m3_html, output_dir)
        mtime1 = result1["output_path"].stat().st_mtime

        # Wait a bit
        import time
        time.sleep(0.1)

        # Second run (should skip)
        result2 = generate_html_qa(sample_m3_html, output_dir)
        mtime2 = result2["output_path"].stat().st_mtime

        # File should not be modified
        assert mtime1 == mtime2

    def test_force_regenerate(self, sample_m3_html, tmp_path):
        """Should regenerate with force=True"""
        generate_html_qa = get_func('generate_html_qa')
        if generate_html_qa is None:
            pytest.skip("Module not implemented yet")

        output_dir = tmp_path / "qa_raw"

        # First run
        result1 = generate_html_qa(sample_m3_html, output_dir)
        mtime1 = result1["output_path"].stat().st_mtime

        import time
        time.sleep(0.1)

        # Second run with force
        result2 = generate_html_qa(sample_m3_html, output_dir, force=True)
        mtime2 = result2["output_path"].stat().st_mtime

        assert mtime2 > mtime1


# =============================================================================
# Test: CLI
# =============================================================================

class TestCLI:
    """Test command-line interface"""

    def test_help_output(self):
        """Should show help with --help"""
        import subprocess
        script_path = Path(__file__).parent.parent / "scripts" / "04b_generate_qa_html.py"
        if not script_path.exists():
            pytest.skip("Script not implemented yet")

        result = subprocess.run(
            ['python', str(script_path), '--help'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert '--data-src' in result.stdout or '--input' in result.stdout
        assert '--output' in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Implementation Guidance

### 04a: Image Q&A Generation

#### CLI Interface

```bash
python scripts/04a_generate_qa_images.py \
    --classified work/classified/pages.csv \
    --indices work/indices \
    --output work/qa_raw \
    --config config.yaml \
    [--limit N] \
    [--start-from PAGE_ID] \
    [--force] \
    [--verbose]
```

#### Main Processing Loop

```python
def main():
    # 1. Parse arguments
    # 2. Load config
    # 3. Load classification data
    # 4. Load index metadata
    # 5. Create output directories

    # 6. Process each page:
    for page_data in classification_data:
        # a. Check if should skip
        if should_skip_page(page_data, config, output_path):
            continue

        # b. Get procedure context
        codes, names = get_procedures_for_page(...)
        context_block = build_context_block(...)

        # c. Select prompt
        system_prompt, user_template = select_prompt_template(...)

        # d. Call API
        result = generate_qa_for_page(...)

        # e. Assign IDs
        qa_pairs = assign_qa_ids(result["qa_pairs"], page_id)

        # f. Write output
        write_qa_output(...)

        # g. Log progress
        write_progress_log(...)

        # h. Rate limit delay
        time.sleep(rate_limit_delay)

    # 7. Print summary
```

#### Error Handling

- Retry on API errors with exponential backoff
- Log all errors to `work/logs/generation_errors.csv`
- Continue processing other pages after failure
- Save checkpoint every N pages (configurable)

### 04b: HTML Q&A Generation

#### CLI Interface

```bash
python scripts/04b_generate_qa_html.py \
    --data-src data_src \
    --output work/qa_raw \
    --config config.yaml \
    [--force]
```

#### HTML Parsing Strategy

1. Find all `<table>` elements
2. Detect header row (first row or `<th>` elements)
3. Parse each data row to extract:
   - Category (if 3+ columns)
   - Spec name
   - Spec value
4. Generate Q&A variations for each spec

---

## Acceptance Criteria

### 04a: Image Q&A Generation

1. **Processes all classified pages** (except skipped)
2. **Uses correct prompts** based on source_type + content_type
3. **Injects context** from index metadata when available
4. **Handles API errors** gracefully with retry
5. **Is idempotent** (skips existing outputs)
6. **Logs progress** and errors (including cost_usd per page)
7. **Generates 6-12 Q&A pairs per page** (varies by content type)
8. **Output matches schema** in pipeline_rearchitecture.md
9. **Page ID normalization** works correctly (21-01 matches 21-1 in index)
10. **Section slug derivation** finds correct index files
11. **Image preprocessing** keeps all images under 5MB
12. **Q&A validation** catches hallucinations and self-referential language

### 04b: HTML Q&A Generation

1. **Parses all spec tables** from HTML files (handles 2/3/4 column variations)
2. **Generates 2-3 Q&A variations** per spec
3. **Detects model** from filename (M3, 320is, or E30 fallback)
4. **Output matches schema** (image_path: null)
5. **No API calls required**
6. **Is idempotent**
7. **Handles nested tables** correctly (skips inner tables)

### Overall

1. All tests pass: `pytest tests/test_04a_generate_qa_images.py tests/test_04b_generate_qa_html.py -v`
2. Output files validate against schema
3. >98% of pages successfully processed
4. Reasonable token usage (track in logs)
5. Total cost logged per run
6. Validation stats included in output (accepted/rejected/warnings counts)

---

## Dependencies

```txt
# requirements.txt additions
anthropic>=0.18.0
beautifulsoup4>=4.12.0
Pillow>=10.0.0
PyYAML>=6.0
```

---

## Next Steps After Implementation

1. **Test on sample pages**: Run on 10-20 diverse pages for manual Q&A review
2. **Validate prompt effectiveness**: Check Q&A quality for each content type
3. **Tune questions_per_page**: Adjust based on actual page content density
4. **Run full pipeline**: Generate Q&A for all pages
5. **Proceed to Stage 5**: Filter and deduplicate Q&A pairs
