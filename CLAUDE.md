# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Build a Vision-Language Model fine-tuned on BMW E30 M3 service documentation. The project covers the full stack: scraping community knowledge, processing service manuals into VLM training data, fine-tuning Qwen2-VL-7B with LoRA on Modal, and evaluating results.

**Current state**: The manual data pipeline is complete (12,410 Q&A pairs). Training infrastructure exists and has been run on Modal. Evaluation framework is implemented but not yet run against a fine-tuned model. A forum data pipeline is planned but not yet built.

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
make -C data/src/manual all              # Full pipeline (Stages 1-6)
make -C data/src/manual status           # Show progress
make -C data/src/manual quick            # Skip Stages 1-2 (sources already prepared)
make -C data/src/manual regen-qa         # Regenerate from Stage 4
make -C data/src/manual refilter         # Rerun from Stage 5
make -C data/src/manual clean            # Clean work/ artifacts
```

### Training (requires Modal + HuggingFace setup)
```bash
make train                  # Full training on Modal (A100-80GB)
make train-dev              # Dev run (100 samples)
make train-resume           # Resume from checkpoint (detached)
make train-logs             # Check training logs from Modal volume
```

### Evaluation
```bash
make eval-sample            # Create stratified eval sample from val set
make eval-modal-baseline    # Baseline eval on Modal (no local GPU needed)
make eval-modal-finetuned   # Fine-tuned eval on Modal (requires ADAPTER_REPO)
make eval-modal-quick       # Quick test (10 samples)
make eval-compare           # Generate comparison report
make eval-mock              # Test eval infra without GPU
```

### Scraper
```bash
python scraper/01_discover_forums.py    # Discover forum structure
python scraper/02_scrape_threads.py     # Scrape thread listings
python scraper/03_scrape_posts.py       # Download post content
python scraper/04_download_images.py    # Download images
```

### Testing
```bash
pytest data/src/manual/tests/           # Pipeline tests
pytest scraper/tests/                   # Scraper tests
pytest eval/test_vlm.py                 # Evaluation tests
```

### Environment
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key      # Required for pipeline Stages 3-4
```

## Architecture

### Data Source Convention

Every data source is self-contained under `data/src/<name>/`:

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

Sources know nothing about each other. A merge layer (`data/training/`) reads from each source's `prepared/` output. See `specs/data_architecture_spec.md` for full details.

### Manual Pipeline Flow
```
data/src/manual/raw/ (JPG/PDF/HTML)
    ↓
pipeline/01_inventory.py        → work/inventory.csv
pipeline/02_prepare_sources.py  → work/inventory_prepared.csv (+ PDF→JPG)
pipeline/03_classify_pages.py   → work/classified/pages.csv + work/indices/*.json  [Claude API]
pipeline/04a_generate_qa_images.py  ┐
pipeline/04b_generate_qa_html.py    ┘→ work/qa_raw/*.json  [Claude API]
pipeline/05_filter_qa.py        → work/qa_filtered/*.json
pipeline/06_deduplicate_qa.py   → work/qa_unique/*.json
pipeline/07_emit_vlm_dataset.py → prepared/manual_train.jsonl + manual_val.jsonl + images/
pipeline/08_validate_vlm.py     → work/logs/vlm_qa_report.md
pipeline/09_upload_vlm.py       → HuggingFace Hub
```

### Training Pipeline
- `training/modal_train.py` — Qwen2-VL-7B LoRA fine-tuning on Modal A100-80GB
- Loads dataset from HuggingFace (`drumwell/vlm3`), not local files
- Config: `training/configs/lora_qwen2vl.yaml` (rank 64, alpha 128, 4-bit quantization)

### Evaluation Pipeline
- `eval/sample_eval_set.py` — Stratified sampling from validation set
- `eval/run_eval.py` — Run inference with base or fine-tuned model
- `eval/modal_eval.py` — Same but on Modal cloud GPU
- `eval/compare_results.py` — Generate comparison reports
- `eval/metrics.py` — Evaluation metrics
- `eval/benchmarks/manual_probes.json` — Hand-crafted test cases

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

- `data/src/manual/config.yaml` — Pipeline settings (API model, rate limits, classification patterns, Q&A generation, filters, train/val split)
- `training/configs/lora_qwen2vl.yaml` — LoRA training hyperparameters
- `data/training/config.yaml` — Merge layer source weights (placeholder)

## Directory Layout

```
vlm3/
├── data/
│   ├── src/
│   │   ├── manual/                 # Service manual data source (complete)
│   │   │   ├── Makefile
│   │   │   ├── config.yaml
│   │   │   ├── raw/                # ~45 section folders of scanned pages
│   │   │   ├── pipeline/           # Scripts 01-09
│   │   │   ├── work/               # Intermediate artifacts (not committed)
│   │   │   ├── prepared/           # 11,154 train + 1,256 val examples + 1,408 images
│   │   │   └── tests/              # 11 test files + fixtures
│   │   └── forum/                  # Forum data source (planned, raw/ stub only)
│   ├── training/                   # Merge layer (placeholder merge.py + config)
│   └── Makefile                    # Data orchestrator
├── training/                       # Modal training infrastructure
│   ├── modal_train.py              # LoRA fine-tuning on A100-80GB
│   └── configs/lora_qwen2vl.yaml
├── eval/                           # Evaluation framework
│   ├── run_eval.py                 # Local GPU evaluation
│   ├── modal_eval.py               # Modal cloud evaluation
│   ├── sample_eval_set.py          # Stratified sampling
│   ├── compare_results.py          # Comparison reports
│   ├── metrics.py                  # Evaluation metrics
│   └── benchmarks/manual_probes.json
├── scraper/                        # Forum scraper (01-04 scripts)
├── specs/                          # Architecture specs and pipeline stage specs
└── Makefile                        # Root: delegates to data/, training/, eval/
```

## Script Conventions

Pipeline scripts follow consistent patterns:
- CLI with argparse and `--help`
- All paths passed via CLI arguments (no hardcoded defaults)
- Idempotent (safe to rerun)
- Config loaded from source-local `config.yaml`
- Logging to stdout and `work/logs/`
- Zero inter-script Python imports

## Workflow

### Branching
- **Never commit directly to main.** Always create a feature branch first.
- Branch naming: `feat/<topic>`, `fix/<topic>`, or `data/<topic>` (e.g., `feat/forum-pipeline`, `fix/filter-edge-case`).
- Keep branches focused — one logical change per branch.

### Pull Requests
- All changes reach main via pull request. Never push directly to main.
- PR title should be concise (<70 chars). Use the body for details.
- Ensure CI passes (tests, lint) before requesting merge.

### Merging
- **Prefer rebase** over squash merges or merge commits. Use `git rebase main` to update feature branches before merging.
- Keep commit history linear and clean on main.

### Commits & Testing
- Run relevant tests before committing: `pytest data/src/manual/tests/` for pipeline changes, `pytest scraper/tests/` for scraper changes, etc.
- New pipeline scripts should have corresponding tests.
- Don't commit secrets, large binaries, or `work/` artifacts (`.gitignore` covers most of this, but be mindful).
