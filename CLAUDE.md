# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM training data pipeline for BMW E30 M3 service manuals. Converts scanned service manual pages into Q&A pairs for fine-tuning Vision-Language Models. Uses Claude API to generate context-aware questions directly from images—no OCR needed.

## Commands

### Data Pipeline
```bash
make data                   # Run all data source pipelines
make data-manual            # Run manual pipeline only
make data-status            # Show pipeline progress
make data-clean             # Clean pipeline artifacts
```

### Manual Pipeline (from data/src/manual/)
```bash
make -C data/src/manual all              # Full pipeline
make -C data/src/manual status           # Show progress
make -C data/src/manual quick            # Skip Stages 1-2
make -C data/src/manual regen-qa         # Regenerate from Stage 4
make -C data/src/manual refilter         # Rerun from Stage 5
make -C data/src/manual clean            # Clean intermediate files
```

### Individual Stages (from data/src/manual/)
```bash
make -C data/src/manual inventory        # Stage 1: Catalog source files
make -C data/src/manual prepare          # Stage 2: Convert PDFs, validate images
make -C data/src/manual classify         # Stage 3: Classify pages, parse indices
make -C data/src/manual generate-qa      # Stage 4: Generate Q&A pairs
make -C data/src/manual quality-control  # Stage 5: Filter and deduplicate
make -C data/src/manual emit             # Stage 6a: Emit VLM JSONL
make -C data/src/manual validate         # Stage 6b: Validate dataset
make -C data/src/manual upload           # Stage 6c: Upload to HuggingFace
```

### Testing
```bash
pytest data/src/manual/tests/                           # All tests
pytest data/src/manual/tests/test_01_inventory.py       # Single file
pytest data/src/manual/tests/ -k "classify"             # Pattern match
pytest data/src/manual/tests/ -v                        # Verbose
```

### Environment
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key      # Required for Stages 3-4
```

## Architecture

### Data Source Convention

Every data source lives under `data/src/<name>/` and is self-contained:

```
data/src/<name>/
├── Makefile            # Source-level targets: all, status, clean
├── config.yaml         # Source-specific configuration
├── raw/                # Immutable input data
├── pipeline/           # Numbered scripts: 01_*.py, 02_*.py, ...
├── work/               # Intermediate outputs (safe to delete)
├── prepared/           # Final output: <name>_train.jsonl, <name>_val.jsonl
└── tests/              # Pipeline tests
```

### Manual Pipeline Flow
```
data/src/manual/raw/ (JPG/PDF/HTML)
    ↓
pipeline/01_inventory.py        → work/inventory.csv
    ↓
pipeline/02_prepare_sources.py  → work/inventory_prepared.csv (+ PDF→JPG)
    ↓
pipeline/03_classify_pages.py   → work/classified/pages.csv + work/indices/*.json
    ↓
pipeline/04a_generate_qa_images.py  ┐
pipeline/04b_generate_qa_html.py    ┘→ work/qa_raw/*.json
    ↓
pipeline/05_filter_qa.py        → work/qa_filtered/*.json
pipeline/06_deduplicate_qa.py   → work/qa_unique/*.json
    ↓
pipeline/07_emit_vlm_dataset.py → prepared/manual_train.jsonl + prepared/manual_val.jsonl + prepared/images/
pipeline/08_validate_vlm.py     → work/logs/vlm_qa_report.md
pipeline/09_upload_vlm.py       → HuggingFace Hub
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

**VLM output** (`prepared/manual_train.jsonl`):
```json
{"image": "images/21-03.jpg", "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "metadata": {...}}
```

## Configuration

`data/src/manual/config.yaml` contains all pipeline settings:
- `api` — Model selection, rate limits, retries
- `classification` — Content type patterns
- `generation` — Questions per page, skip patterns, cost controls
- `filters` — Answer length, generic patterns, similarity thresholds
- `output` — Train/val split, image handling

## Directory Layout

```
vlm3/
├── data/
│   ├── src/
│   │   ├── manual/                 # Service manual data source
│   │   │   ├── Makefile            # Source-level targets
│   │   │   ├── config.yaml         # Pipeline configuration
│   │   │   ├── raw/                # Scanned manual pages
│   │   │   ├── pipeline/           # Scripts 01-09
│   │   │   ├── work/               # Intermediate artifacts
│   │   │   ├── prepared/           # Final outputs (manual_train.jsonl, etc.)
│   │   │   └── tests/              # pytest suite
│   │   └── forum/                  # Forum data source (future)
│   ├── training/                   # Merge layer (config + merge.py)
│   └── Makefile                    # Data orchestrator
├── training/                       # VLM fine-tuning
├── eval/                           # Model evaluation
├── scraper/                        # Data collection
├── specs/                          # Project specifications
└── Makefile                        # Root: delegates to data/, training/, eval/
```

## Script Pattern

All scripts follow consistent conventions:
- CLI with argparse and `--help`
- Idempotent (safe to rerun)
- Config loaded from `config.yaml`
- Logging to stdout and `work/logs/`
