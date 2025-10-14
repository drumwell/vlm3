# BMW E30 M3 Service Manual - Instruction Tuning Dataset

> Convert scanned service manual pages into high-quality instruction-tuning data for finetuning LLMs on automotive technical knowledge.

## Quick Start

### 1. Prepare Dataset

```bash
# Run full pipeline (from scratch)
make all

# This runs:
make inventory       # Catalog all images
make preprocess      # Clean and deskew
make ocr             # Extract text and tables
make blocks          # Parse into structured blocks
make emit            # Generate consolidated JSONL
make validate        # QA report
make extract_html    # Add tech specs from HTML
make autotrain_prep  # Convert to AutoTrain flat text format
make synthetic_val   # Generate synthetic validation
```

**Output**:
- `data/hf_train_autotrain.jsonl` (2,510 training examples)
- `data/hf_val_synthetic.jsonl` (248 synthetic validation examples)

### 2. Upload to HuggingFace

```bash
# First time setup
pip install datasets huggingface_hub
huggingface-cli login

# Upload dataset
python scripts/10_upload_to_hf.py --repo drumwell/llm3
```

### 3. Train on AutoTrain

1. Go to https://huggingface.co/autotrain
2. Create new project → "LLM Fine-tuning"
3. Select dataset: `drumwell/llm3`
4. Choose base model: `meta-llama/Llama-3.1-8B-Instruct`
5. Column mapping: `text_column = 'text'`
6. Train (~30-60 min on A100, ~$5-10)

**See `AUTOTRAIN_READY.md` for complete training guide**

## Dataset Overview

**AutoTrain-compatible format with synthetic validation!**

| Split | Examples | Source |
|-------|----------|--------|
| **Training** | 2,510 | All service manual + HTML specs (consolidated) |
| **Validation** | 248 | Synthetic (question paraphrasing) |
| **Total** | 2,758 | - |

**Data Sources**:
- OCR (scanned pages): ~1,800 examples (improved block extraction)
- HTML (tech specs): ~700 examples

**Task Types**:
- `[SPEC]` - Extract technical values (torque, clearances, part numbers)
- `[PROCEDURE]` - Step-by-step repair instructions
- `[EXPLANATION]` - Component descriptions and operation
- `[WIRING]` - Wiring diagram annotations
- `[TROUBLESHOOTING]` - Diagnostic checklists

## What's New (Latest)

✅ **AutoTrain-compatible flat text format**
- No more Parquet serialization errors
- Format: `{"text": "User: [TASK] Q\nAssistant: A"}`
- Successfully tested with Llama-3.1-8B-Instruct

✅ **All 2,510 examples used for training**
- No validation split waste
- Maximizes service manual coverage
- Synthetic validation for generalization testing

✅ **Improved block extraction**
- Added fallback logic for non-numbered procedures
- Increased blocks by 27.6% (794 → 1,013)
- Procedures increased 6x (43 → 262 examples)

✅ **Synthetic validation strategy**
- 248 paraphrased questions
- Tests model generalization without data overlap
- Generated via `scripts/09_generate_synthetic_validation.py`

✅ **Complete training documentation**
- `AUTOTRAIN_READY.md` - Step-by-step training guide
- `LEARNING_EXPERIMENTS.md` - QLoRA learning experiments

## Project Structure

```
llm3/
├── data_src/           # Source images and HTML files
│   ├── 11 - Engine/    # Service manual sections
│   ├── M3-techspec.html        # ← NEW: Tech specifications
│   └── 320is-techspec.html     # ← NEW: 320is variant specs
├── data/               # Generated datasets
│   ├── dataset.jsonl              # 2,510 consolidated examples (all data)
│   ├── hf_train_autotrain.jsonl   # 2,510 training examples (flat text)
│   └── hf_val_synthetic.jsonl     # 248 validation examples (synthetic)
├── work/               # Intermediate artifacts
│   ├── images_clean/   # Preprocessed images
│   ├── ocr_raw/        # OCR JSON outputs
│   ├── blocks/         # Parsed content blocks
│   └── logs/           # QA reports
├── scripts/            # Pipeline stages
│   ├── 01_inventory.py                    # Catalog images
│   ├── 02_preprocess.py                   # Clean/deskew
│   ├── 03_ocr.py                          # PaddleOCR extraction
│   ├── 03b_ocr_tables.py                  # Table extraction
│   ├── 04_parse_blocks.py                 # Structure content
│   ├── 05_emit_jsonl.py                   # Generate consolidated JSONL
│   ├── 06_validate.py                     # QA checks
│   ├── 07_extract_html_specs.py           # HTML parsing
│   ├── 08_convert_to_autotrain.py         # AutoTrain flat text format
│   ├── 09_generate_synthetic_validation.py # Synthetic validation
│   └── 10_upload_to_hf.py                 # HF upload
├── config.yaml         # Pipeline configuration
├── Makefile            # Orchestration
└── *.md                # Documentation
```

## Model Training

**Recommended Configuration** (see `MODEL_CONFIG.md` and `AUTOTRAIN_READY.md`):

- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct` (proven to work)
- **Method**: QLoRA (4-bit quantization)
- **LoRA rank**: 16
- **Learning rate**: 2e-4
- **Epochs**: 3
- **Batch size**: 4-8
- **Platform**: HuggingFace AutoTrain (recommended)

**Dataset Configuration**:
- Training: 2,510 examples (all service manual data)
- Validation: 248 synthetic examples
- Format: Flat text `{"text": "User: [TASK] Q\nAssistant: A"}`

## Expected Results

With the enhanced dataset, your model should correctly answer:

```python
# Previously failed (hallucinated "4.0 V"):
"[SPEC] What is the engine displacement?" → "2302 CC" or "2.3 L" ✅

# Now also works:
"What is the bore and stroke?" → "93.4 × 84.0 mm" ✅
"What is the compression ratio?" → "10.5:1" ✅
"What is the power output?" → "197 BHP / 147 kW @ 6750 rpm" ✅
```

## Iterating on Model Quality

See `LEARNING_EXPERIMENTS.md` for systematic experimentation guide:

1. **Experiment 1**: Baseline with enhanced dataset
2. **Experiment 2**: Try different base models (Mistral-7B)
3. **Experiment 3**: Hyperparameter tuning (LoRA rank, learning rate)
4. **Experiment 4**: Data quality improvements (OCR cleanup)

## Requirements

```bash
# Python dependencies
pip install -r requirements.txt

# Key packages:
# - paddlepaddle, paddleocr (OCR)
# - opencv-python, pillow (image processing)
# - pyyaml (config)
# - datasets, huggingface_hub (for upload)
```

## Troubleshooting

### "Model hallucinated engine displacement"
✅ **Fixed!** Run `make extract_html` to add HTML tech specs to dataset.

### "Dataset has gaps in general specs"
✅ **Fixed!** HTML extraction now includes:
- Engine: displacement, bore/stroke, compression, power/torque
- Transmission: ratios, clutch specs
- Chassis: suspension, brakes, wheels
- Fluids: oil, coolant, brake fluid capacities

### "Training failed with bitsandbytes error"
✅ **Solved!** Use HuggingFace AutoTrain instead of Colab:
- No environment setup needed
- Reliable training infrastructure
- Auto-configured QLoRA parameters

## Documentation

| File | Purpose |
|------|---------|
| **README.md** | This file - project overview and quick start |
| **AUTOTRAIN_READY.md** | **→ START HERE for training guide** |
| **LEARNING_EXPERIMENTS.md** | QLoRA learning experiments and iteration |
| **CLAUDE.md** | Project brief for Claude Code |

## License

This dataset is for research/educational purposes only. Check original BMW service manual licensing.

## Citation

```bibtex
@dataset{bmw_service_manual_2025,
  title={BMW E30 M3 Service Manual Instruction-Tuning Dataset},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/llm3}},
  note={Extracted from scanned BMW service manual pages and HTML tech specs}
}
```

## Acknowledgments

- BMW E30 M3 service manual (original source)
- PaddleOCR for text extraction
- HuggingFace for AutoTrain platform
- BMW enthusiast community

---

**Ready to start?** Run `make all` and then upload to HuggingFace AutoTrain!
