# vlm3 - BMW E30 M3 Vision Language Model Project

Build a Vision-Language Model that understands BMW E30 M3 service documentation. This project provides the complete stack: scraping community knowledge, processing service manuals into training data, fine-tuning Qwen2-VL-7B with LoRA on Modal, and evaluating results.

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Data Pipeline** | ✅ Complete | 12,410 Q&A pairs from service manuals |
| **Forum Pipeline** | ✅ Complete | 1,454 train + 165 val pairs from community forums |
| **Merge Layer** | ✅ Complete | Multi-source merge (manual 80%, forum 20%) |
| **Scraper** | ✅ Complete | 51 JSONL files scraped from 10 forums |
| **Training** | ✅ 4 runs | Qwen2-VL-7B LoRA on Modal A100-80GB |
| **Evaluation** | ✅ Active | Multi-run comparison, manual probes, ROUGE-L tracking |

## Quick Start

### Environment Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key  # Required for pipeline Stages 3-4
```

### Run Data Pipeline

```bash
make data             # Run all data source pipelines
make data-status      # Check progress
make data-manual      # Run manual pipeline only
make data-merge       # Run merge layer
```

**Output**: `data/training/merged_train.jsonl` + `merged_val.jsonl` (manual + forum combined)

### Train on Modal

```bash
make train            # Full training on Modal A100-80GB (detached)
make train-dev        # Dev run (100 samples, ~10 min)
make train-logs       # Check training logs
make train-runs       # List completed runs
```

### Evaluate

```bash
make eval-modal-baseline     # Baseline eval on Modal (no local GPU needed)
make eval-modal-finetuned    # Fine-tuned eval (requires ADAPTER_REPO)
make eval-compare            # Generate comparison report
make eval-compare-runs       # Multi-run progression report
```

## Project Structure

```
vlm3/
├── data/                         # All data concerns
│   ├── src/
│   │   ├── manual/               # Service manual data source (complete)
│   │   │   ├── Makefile           # Source-level targets
│   │   │   ├── config.yaml        # Pipeline configuration
│   │   │   ├── raw/               # ~45 section folders of scanned pages
│   │   │   ├── pipeline/          # Scripts 01-09
│   │   │   ├── work/              # Pipeline intermediates
│   │   │   ├── prepared/          # 11,154 train + 1,256 val + 1,408 images
│   │   │   └── tests/             # pytest suite
│   │   └── forum/                 # Forum data source (complete)
│   │       ├── Makefile
│   │       ├── config.yaml
│   │       ├── raw/               # 51 JSONL files (posts + threads)
│   │       ├── pipeline/          # Stages 01-07
│   │       ├── prepared/          # 1,454 train + 165 val
│   │       └── tests/
│   ├── training/                  # Merge layer
│   │   ├── merge.py               # Combines all sources with weights
│   │   ├── config.yaml            # Source weights (manual 80%, forum 20%)
│   │   ├── merged_train.jsonl     # Unified training set
│   │   ├── merged_val.jsonl       # Unified validation set
│   │   └── images/                # Symlinks to all source images
│   └── Makefile                   # Data orchestrator
│
├── scraper/                       # Web scraper for community knowledge
│   ├── 01_discover_forums.py
│   ├── 02_scrape_threads.py
│   ├── 03_scrape_posts.py
│   ├── 04_download_images.py
│   ├── core.py                    # Session management, rate limiting, checkpointing
│   ├── parser.py                  # vBulletin HTML parsing
│   ├── scraper_config.yaml
│   └── tests/
│
├── training/                      # VLM fine-tuning on Modal
│   ├── modal_train.py             # LoRA fine-tuning on A100-80GB
│   ├── configs/
│   │   └── lora_qwen2vl.yaml      # LoRA training config
│   └── README.md
│
├── eval/                          # Evaluation framework
│   ├── run_eval.py                # Local GPU evaluation
│   ├── modal_eval.py              # Modal cloud evaluation
│   ├── run_eval_anthropic.py      # Claude model evaluation
│   ├── sample_eval_set.py         # Stratified sampling
│   ├── compare_results.py         # Baseline vs fine-tuned comparison
│   ├── compare_runs.py            # Multi-run progression analysis
│   ├── metrics.py                 # Evaluation metrics
│   ├── model_wrapper.py           # Model loading & inference
│   ├── run_meta.py                # Run metadata management
│   ├── benchmarks/
│   │   └── manual_probes.json     # 40-56 hand-crafted test cases
│   └── reports/
│       └── archive/               # Archived eval runs (v1-v4)
│
├── specs/                         # Project specifications
├── Makefile                       # Root: delegates to data/, training/, eval/
└── README.md
```

---

## Data Pipeline

Converts scanned service manual pages into VLM training data using Claude's vision capabilities — no OCR needed. Each data source is self-contained under `data/src/<name>/`.

### Manual Pipeline Flow

```
data/src/manual/raw/ (JPG/PDF/HTML)
    ↓
Stage 1: Inventory    → work/inventory.csv
Stage 2: Prepare      → work/inventory_prepared.csv (PDF→JPG)
Stage 3: Classify     → work/classified/pages.csv [Claude API]
Stage 4: Generate Q&A → work/qa_raw/*.json [Claude API]
Stage 5: Filter       → work/qa_filtered/*.json → work/qa_unique/*.json
Stage 6: Emit         → prepared/manual_train.jsonl + manual_val.jsonl
```

### Source Materials

| Source | Format | Content |
|--------|--------|---------|
| Service Manual (00-97) | JPG scans | Procedures, specs, diagrams |
| Electrical Manual | JPG scans | Wiring, pinouts, flowcharts |
| Bosch Motronic ML 3-1 | JPG scans | ECU signals, parameters |
| Getrag 265/5 Rebuild | PDF | Transmission procedures |
| Tech specs (HTML) | HTML | Vehicle specifications |
| Community forums | JSONL | Troubleshooting, DIY, maintenance |

### Output Format

```json
{
  "image": "images/21-03.jpg",
  "conversations": [
    {"role": "user", "content": "What should I inspect the clutch for?"},
    {"role": "assistant", "content": "Inspect for cracks, wear, and burnt spots..."}
  ],
  "metadata": {
    "page_id": "21-03",
    "section_name": "Clutch",
    "content_type": "procedure",
    "question_type": "inspection"
  }
}
```

### Data Make Targets

| Target | Description |
|--------|-------------|
| `make data` | Run all data source pipelines |
| `make data-manual` | Run manual pipeline only |
| `make data-status` | Show progress |
| `make data-merge` | Run merge layer |
| `make data-clean` | Clean intermediates |
| `make upload` | Upload merged dataset to HuggingFace |
| `make -C data/src/manual quick` | Skip Stages 1-2 |
| `make -C data/src/manual regen-qa` | Regenerate from Stage 4 |

---

## Scraper

Collects E30 M3 community knowledge from vBulletin forums for additional training data.

### Features

- **Rate limiting**: Polite scraping with randomized 1.5-2.5s delays
- **Checkpoint/resume**: Stop and restart without losing progress
- **Structured storage**: Raw HTML + parsed JSON
- **Image downloading**: Downloads embedded images with references
- **Proxy support**: Residential proxies (e.g., Oxylabs)

### Usage

```bash
python scraper/01_discover_forums.py              # Discover forum structure
python scraper/02_scrape_threads.py --forum-id 42  # Scrape specific forum
python scraper/03_scrape_posts.py --forum-id 42
python scraper/04_download_images.py --forum-id 42

# Or scrape everything
python scraper/02_scrape_threads.py --all
```

See `scraper/README.md` for detailed usage and configuration.

---

## Training

Fine-tune Qwen2-VL-7B-Instruct using LoRA on Modal GPU cloud. Dataset is loaded from HuggingFace (`drumwell/vlm3`), not local files.

### Configuration (`training/configs/lora_qwen2vl.yaml`)

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2-VL-7B-Instruct |
| Method | LoRA (rank 64, alpha 128) |
| LoRA Targets | q/k/v/o/gate/up/down projections |
| Quantization | 4-bit (nf4, bfloat16) |
| Training | 3 epochs, batch 16 (4x4 accumulation) |
| Learning Rate | 2e-4, cosine decay |
| Max Seq Length | 2048 |
| GPU | A100-80GB |

### Training Make Targets

| Target | Description |
|--------|-------------|
| `make train` | Full training on Modal (detached) |
| `make train-dev` | Dev run (100 samples) |
| `make train-resume` | Resume from checkpoint |
| `make train-logs` | Check training logs |
| `make train-runs` | List training runs |
| `make train-archive` | Archive current run |
| `make train-clean` | Delete current run from Modal |

### Completed Runs

| Run | Date | Samples | Description |
|-----|------|---------|-------------|
| v1-manual-only | Feb 9 | 334 | Initial manual-only training |
| v2-manual-retrain | Feb 13 | 328 | Manual retrain |
| v3-manual-retrain | Feb 14 | 328 | Manual retrain |
| v4-combined | Feb 15 | 328 | Manual (80%) + forum (20%) data |

---

## Evaluation

Custom metrics framework with multi-run tracking and manual probe benchmarks.

### Metrics

| Metric | Purpose |
|--------|---------|
| `rouge_l` | Answer similarity |
| `keyword_presence` | Technical term detection |
| `numeric_accuracy` | Torque specs, measurements |
| `unit_correctness` | Unit validation (Nm, bar, etc.) |

### Manual Probes

40-56 hand-crafted test cases covering critical E30 M3 scenarios. 100% critical pass rate achieved on v3/v4.

### Evaluation Make Targets

| Target | Description |
|--------|-------------|
| `make eval-sample` | Create stratified eval sample |
| `make eval-modal-baseline` | Baseline eval on Modal |
| `make eval-modal-finetuned` | Fine-tuned eval (requires `ADAPTER_REPO`) |
| `make eval-modal-quick` | Quick test (10 samples) |
| `make eval-modal-probes` | Manual probes on Modal |
| `make eval-compare` | Baseline vs fine-tuned comparison |
| `make eval-compare-runs` | Multi-run progression report |
| `make eval-archive` | Archive current reports (`LABEL=` optional) |
| `make eval-runs` | List archived eval runs |
| `make eval-mock` | Test infra without GPU |

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `anthropic` — Claude API for classification/Q&A generation
- `pillow`, `opencv-python` — Image processing
- `pdf2image` — PDF conversion
- `sentence-transformers` — Semantic deduplication
- `requests`, `beautifulsoup4` — Web scraping / HTML parsing
- `datasets`, `huggingface_hub` — Dataset management

Training dependencies (installed on Modal):
- `torch`, `transformers`, `accelerate`, `peft` — Model training
- `bitsandbytes` — 4-bit quantization
- `qwen-vl-utils` — Qwen2-VL utilities

---

## Testing

```bash
pytest data/src/manual/tests/      # Pipeline tests
pytest scraper/tests/              # Scraper tests
pytest eval/test_vlm.py            # Evaluation tests
pytest -v                          # Verbose
pytest -k "classify"               # Pattern match
```

---

## License

Research/educational purposes. Check original BMW service manual licensing.

## Acknowledgments

- BMW E30 M3 service manuals
- Anthropic Claude for vision-based Q&A generation
- E30 M3 enthusiast community
