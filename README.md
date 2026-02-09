# vlm3 - BMW E30 M3 Vision Language Model Project

Build a Vision-Language Model that understands BMW E30 M3 service documentation. This project provides the complete stack: scraping community knowledge, processing service manuals into training data, and fine-tuning VLMs.

## Project Status

| Component | Status | Description |
|-----------|--------|-------------|
| **Data Pipeline** | ✅ Complete | 12,410 Q&A pairs from service manuals |
| **Scraper** | ✅ Implemented | Web scraper for community knowledge |
| **Training** | ⚙️ Config Only | Qwen2-VL-7B LoRA fine-tuning ready |
| **Evaluation** | 📋 Planned | DeepEval framework with Claude-as-judge |

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
```

**Output**: `data/src/manual/prepared/manual_train.jsonl` (11,154 examples) + `manual_val.jsonl` (1,256 examples)

### Run Scraper

```bash
# See scraper/README.md for full usage
python scraper/01_discover_forums.py    # Discover site structure
python scraper/02_scrape_threads.py     # Scrape thread listings
python scraper/03_scrape_posts.py       # Download post content
python scraper/04_download_images.py    # Download images
```

## Project Structure

```
vlm3/
├── data/                         # All data concerns
│   ├── src/
│   │   ├── manual/               # Service manual data source
│   │   │   ├── Makefile           # Source-level targets
│   │   │   ├── config.yaml        # Pipeline configuration
│   │   │   ├── raw/               # Scanned manual pages (read-only)
│   │   │   ├── pipeline/          # Scripts 01-09
│   │   │   ├── work/              # Pipeline intermediates
│   │   │   ├── prepared/          # Final outputs (manual_train.jsonl, etc.)
│   │   │   └── tests/             # pytest suite
│   │   └── forum/                 # Forum data source (future)
│   ├── training/                  # Merge layer (config + merge.py)
│   └── Makefile                   # Data orchestrator
│
├── scraper/                       # Web scraper for community knowledge
│   ├── 01_discover_forums.py
│   ├── 02_scrape_threads.py
│   ├── 03_scrape_posts.py
│   ├── 04_download_images.py
│   ├── core.py
│   ├── parser.py
│   └── tests/
│
├── training/                      # VLM fine-tuning
│   └── configs/
│       └── lora_qwen2vl.yaml
│
├── eval/                          # Model evaluation
│   └── benchmarks/
│       └── manual_probes.json
│
├── specs/                         # Project specifications
├── Makefile                       # Root: delegates to data/, training/, eval/
└── README.md
```

---

## Data Pipeline

Converts scanned service manual pages into VLM training data using Claude's vision capabilities—no OCR needed. Each data source is self-contained under `data/src/<name>/`.

### Pipeline Flow

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

### Make Targets

| Target | Description |
|--------|-------------|
| `make data` | Run all data source pipelines |
| `make data-manual` | Run manual pipeline only |
| `make data-status` | Show progress |
| `make data-clean` | Clean intermediates |
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

### Usage

```bash
# Discover forum structure
python scraper/01_discover_forums.py

# Scrape specific forum
python scraper/02_scrape_threads.py --forum-id 42
python scraper/03_scrape_posts.py --forum-id 42
python scraper/04_download_images.py --forum-id 42

# Or scrape everything
python scraper/02_scrape_threads.py --all
python scraper/03_scrape_posts.py --all
python scraper/04_download_images.py --all
```

See `scraper/README.md` for detailed usage and configuration.

---

## Training Infrastructure

Fine-tune Qwen2-VL-7B-Instruct using LoRA on Modal GPU cloud.

### Configuration (`training/configs/lora_qwen2vl.yaml`)

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen2-VL-7B-Instruct |
| Method | LoRA (rank 64, alpha 128) |
| Quantization | 4-bit (nf4, bfloat16) |
| Training | 3 epochs, batch 16 (4×4 accumulation) |
| Learning Rate | 2e-4, cosine decay |
| GPU | A100-80GB (~$8-16 estimated cost) |

### Planned Scripts

- `prepare_dataset.py` - Convert JSONL → HuggingFace Dataset
- `modal_train.py` - LoRA training on Modal
- `modal_serve.py` - Inference endpoint

---

## Evaluation (Planned)

DeepEval-based framework using Claude-as-judge.

### Planned Metrics

| Metric | Purpose | Threshold |
|--------|---------|-----------|
| AnswerRelevancy | Does answer address question? | >0.7 |
| Faithfulness | Is answer grounded in image? | >0.7 |
| NumericExactMatch | Torque specs, measurements | >0.85 |
| KeywordPresence | Required technical terms | >0.80 |

### Approach

1. Baseline evaluation on unmodified Qwen2-VL-7B
2. Post-training evaluation
3. Manual probe benchmarks (20-30 critical questions)

---

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- `anthropic` - Claude API for classification/Q&A
- `pillow`, `opencv-python` - Image processing
- `pdf2image` - PDF conversion
- `sentence-transformers` - Semantic deduplication
- `requests`, `beautifulsoup4` - Web scraping
- `datasets`, `huggingface_hub` - Dataset management

---

## Testing

```bash
pytest data/src/manual/tests/      # Pipeline tests
pytest scraper/tests/              # Scraper tests
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
