# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM training data pipeline for BMW E30 M3 service manuals. Converts scanned service manual pages into Q&A pairs for fine-tuning Vision-Language Models. Uses Claude API to generate context-aware questions directly from images—no OCR needed.

## Commands

### Pipeline
```bash
make all                    # Run complete pipeline (Stages 1-6)
make status                 # Show pipeline progress
make quick                  # Skip Stages 1-2 (sources already prepared)
make regen-qa               # Regenerate from Stage 4
make refilter               # Rerun from Stage 5
make clean                  # Clean intermediate files
```

### Individual Stages
```bash
make inventory              # Stage 1: Catalog source files
make prepare                # Stage 2: Convert PDFs, validate images
make classify               # Stage 3: Classify pages, parse indices
make generate-qa            # Stage 4: Generate Q&A pairs
make quality-control        # Stage 5: Filter and deduplicate
make emit                   # Stage 6a: Emit VLM JSONL
make validate               # Stage 6b: Validate dataset
make upload                 # Stage 6c: Upload to HuggingFace
```

### Testing
```bash
pytest tests/                           # All tests
pytest tests/test_01_inventory.py       # Single file
pytest tests/ -k "classify"             # Pattern match
pytest tests/ -v                        # Verbose
```

### Environment
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key      # Required for Stages 3-4
```

## Architecture

### Pipeline Flow
```
data_src/ (JPG/PDF/HTML)
    ↓
01_inventory.py        → work/inventory.csv
    ↓
02_prepare_sources.py  → work/inventory_prepared.csv (+ PDF→JPG)
    ↓
03_classify_pages.py   → work/classified/pages.csv + work/indices/*.json
    ↓
04a_generate_qa_images.py  ┐
04b_generate_qa_html.py    ┘→ work/qa_raw/*.json
    ↓
05_filter_qa.py        → work/qa_filtered/*.json
06_deduplicate_qa.py   → work/qa_unique/*.json
    ↓
07_emit_vlm_dataset.py → data/vlm_train.jsonl + data/vlm_val.jsonl + data/images/
08_validate_vlm.py     → work/logs/vlm_qa_report.md
09_upload_vlm.py       → HuggingFace Hub
```

### Source Types (different prompt templates)
- `service_manual` — Sections 00-97, Getrag transmission
- `electrical_manual` — 1990 Electrical Troubleshooting Manual
- `ecu_technical` — Bosch Motronic documentation
- `html_specs` — HTML techspec files (no API, direct parsing)

### Content Types (classification)
- Service: `index`, `procedure`, `specification`, `diagram`, `troubleshooting`, `text`
- Electrical: `wiring`, `pinout`, `flowchart`, `fuse_chart`
- ECU: `signal`, `oscilloscope`

### Key Data Schemas

**Classified page** (`work/classified/pages.csv`):
```
page_id, image_path, section_id, section_name, source_type, content_type, is_index, confidence
```

**Q&A document** (`work/qa_raw/*.json`):
```json
{"page_id": "21-03", "image_path": "...", "source_type": "service_manual",
 "content_type": "procedure", "qa_pairs": [{"question": "...", "answer": "...", "question_type": "inspection"}]}
```

**VLM output** (`data/vlm_train.jsonl`):
```json
{"image": "images/21-03.jpg", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {...}}
```

## Configuration

`config.yaml` contains all pipeline settings:
- `api` — Model selection, rate limits, retries
- `classification` — Content type patterns
- `generation` — Questions per page, skip patterns, cost controls
- `filters` — Answer length, generic patterns, similarity thresholds
- `output` — Train/val split, image handling

## Directory Layout

```
data_src/       # Source images/PDFs/HTML (read-only)
work/           # Intermediate artifacts
data/           # Final outputs (vlm_train.jsonl, vlm_val.jsonl, images/)
scripts/        # Pipeline scripts 01-09
specs/          # Detailed stage specifications
tests/          # pytest suite with fixtures in conftest.py
```

## Script Pattern

All scripts follow consistent conventions:
- CLI with argparse and `--help`
- Idempotent (safe to rerun)
- Config loaded from `config.yaml`
- Logging to stdout and `work/logs/`
