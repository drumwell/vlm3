# BMW E30 M3 Service Manual - VLM Training Dataset

> Convert scanned service manual pages into Vision-Language Model (VLM) training data using Claude API for Q&A generation directly from images.

## Overview

This pipeline generates training data for fine-tuning Vision-Language Models on automotive technical knowledge. Instead of traditional OCR-based text extraction, it uses Claude's vision capabilities to generate context-aware Q&A pairs directly from service manual images—preserving visual semantics like diagrams, callouts, and spatial relationships that text extraction loses.

## Quick Start

### 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Required for Q&A generation (Stage 3-4)
export ANTHROPIC_API_KEY=your_key_here
```

### 2. Run Pipeline

```bash
# Run full pipeline
make all

# This runs:
make inventory        # Stage 1: Catalog source files
make prepare          # Stage 2: Convert PDFs, validate images
make classify         # Stage 3: Classify pages, parse indices (Claude API)
make generate-qa      # Stage 4: Generate Q&A pairs (Claude API)
make quality-control  # Stage 5: Filter and deduplicate
make emit             # Stage 6a: Emit VLM JSONL
make validate         # Stage 6b: Validate dataset
```

**Output**:
- `data/vlm_train.jsonl` - Training examples (90%)
- `data/vlm_val.jsonl` - Validation examples (10%)
- `data/images/` - Referenced images

### 3. Upload to HuggingFace

```bash
huggingface-cli login
make upload
# Or: python scripts/09_upload_vlm.py --repo your-username/vlm3
```

## Pipeline Architecture

```
data_src/ (JPG/PDF/HTML)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: INVENTORY                                          │
│ 01_inventory.py → work/inventory.csv                        │
│ Catalogs all source files (JPG, PDF, HTML)                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 2: SOURCE PREPARATION                                 │
│ 02_prepare_sources.py → work/inventory_prepared.csv         │
│ Converts PDFs to JPG, validates all images                  │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 3: CLASSIFICATION & INDEX PARSING (Claude API)        │
│ 03_classify_pages.py → work/classified/pages.csv            │
│                      → work/indices/*.json                  │
│ Classifies content type, extracts repair codes from indices │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 4: Q&A GENERATION                                     │
│ 04a_generate_qa_images.py → work/qa_raw/*.json (Claude API) │
│ 04b_generate_qa_html.py   → work/qa_raw/*.json (no API)     │
│ Generates Q&A pairs with source-specific prompts            │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 5: QUALITY CONTROL                                    │
│ 05_filter_qa.py      → work/qa_filtered/*.json              │
│ 06_deduplicate_qa.py → work/qa_unique/*.json                │
│ Filters low-quality, removes duplicates                     │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│ Stage 6: EMIT & VALIDATE                                    │
│ 07_emit_vlm_dataset.py → data/vlm_train.jsonl               │
│                        → data/vlm_val.jsonl                 │
│ 08_validate_vlm.py     → work/logs/vlm_qa_report.md         │
│ 09_upload_vlm.py       → HuggingFace Hub                    │
└─────────────────────────────────────────────────────────────┘
```

## Source Materials

| Source | Format | Handling |
|--------|--------|----------|
| Service Manual sections (00-97) | JPG scans | Main pipeline with procedure/spec prompts |
| 1990 Electrical Troubleshooting Manual | JPG scans | Wiring-specific prompts |
| Bosch Motronic ML 3-1 | JPG scans | ECU technical prompts |
| M3-techspec.html, 320is-techspec.html | HTML | Direct parsing, no API needed |
| Getrag 265/5 Rebuild PDF | PDF | Converted to JPG in Stage 2 |

## Output Format

**VLM Training Record** (`data/vlm_train.jsonl`):
```json
{
  "image": "images/21-03.jpg",
  "conversations": [
    {"role": "user", "content": "What should I visually inspect the clutch pressure plate for?"},
    {"role": "assistant", "content": "Visually inspect the clutch for cracks, wear, and burnt spots. The pressure contact surface must be level."}
  ],
  "metadata": {
    "page_id": "21-03",
    "section_id": "21",
    "section_name": "Clutch",
    "source_type": "service_manual",
    "content_type": "procedure",
    "question_type": "inspection"
  }
}
```

**Question Types**:
- `factual` - Specific values, part names, specifications
- `procedural` - How to perform tasks, step sequences
- `visual` - Diagram callouts, component identification
- `inspection` - What to look for, acceptance criteria
- `tool` - Required tools and supplies
- `safety` - Cautions and warnings
- `navigation` - Repair codes, page references

## Make Targets

| Target | Description |
|--------|-------------|
| `make all` | Run complete pipeline (Stages 1-6) |
| `make status` | Show pipeline progress |
| `make quick` | Skip Stages 1-2 (sources already prepared) |
| `make regen-qa` | Regenerate from Stage 4 |
| `make refilter` | Rerun from Stage 5 |
| `make finalize` | Just emit and validate |
| `make clean` | Clean intermediate files |
| `make clean-all` | Clean everything including outputs |

## Project Structure

```
vlm3/
├── data_src/           # Source images, PDFs, HTML (read-only input)
│   ├── 00 - Maintenance/
│   ├── 11 - Engine/
│   ├── 21 - Clutch/
│   ├── ...
│   ├── M3-techspec.html
│   └── 320is-techspec.html
├── work/               # Intermediate artifacts
│   ├── inventory.csv
│   ├── inventory_prepared.csv
│   ├── classified/pages.csv
│   ├── indices/*.json
│   ├── qa_raw/, qa_filtered/, qa_unique/
│   └── logs/
├── data/               # Final outputs
│   ├── vlm_train.jsonl
│   ├── vlm_val.jsonl
│   └── images/
├── scripts/            # Pipeline scripts (01-09)
├── specs/              # Detailed specifications per stage
├── tests/              # pytest test suite
├── config.yaml         # Pipeline configuration
├── Makefile            # Orchestration
└── CLAUDE.md           # Claude Code project brief
```

## Configuration

All settings in `config.yaml`:

- **API**: Model selection, rate limits, retries
- **Classification**: Content type patterns, source detection
- **Generation**: Questions per page by type, skip patterns, cost controls
- **Filters**: Answer length, generic patterns, similarity thresholds
- **Output**: Train/val split ratio, image handling mode

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `anthropic` - Claude API for classification and Q&A generation
- `pillow`, `opencv-python` - Image processing
- `pdf2image` - PDF to JPG conversion
- `sentence-transformers` - Semantic deduplication
- `pyyaml` - Configuration
- `datasets`, `huggingface_hub` - Dataset upload

## Testing

```bash
pytest tests/                      # Run all tests
pytest tests/test_01_inventory.py  # Single test file
pytest tests/ -k "classify"        # Tests matching pattern
pytest tests/ -v                   # Verbose output
```

## License

This dataset is for research/educational purposes only. Check original BMW service manual licensing.

## Acknowledgments

- BMW E30 M3 service manual (original source)
- Anthropic Claude for vision-based Q&A generation
- HuggingFace for dataset hosting
- BMW enthusiast community
