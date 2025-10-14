# BMW E30 M3 Service Manual - Dataset Pipeline
# Converts scanned service manual pages to AutoTrain-ready format

SECT_FILTER?=

# ============================================================================
# STAGE 1: OCR Pipeline (Extract text from scanned images)
# ============================================================================

inventory:
	@echo "📋 Cataloging service manual images..."
	python scripts/01_inventory.py --data-src data_src --output work/inventory.csv --section-filter "$(SECT_FILTER)"

preprocess:
	@echo "🖼️  Preprocessing images (deskew, clean)..."
	python scripts/02_preprocess.py --inventory work/inventory.csv --out work/images_clean

ocr:
	@echo "🔍 Running OCR (text + tables)..."
	python scripts/03_ocr.py --input-dir work/images_clean --output-dir work/ocr_raw
	python scripts/03b_ocr_tables.py --ocr-dir work/ocr_raw --images-dir work/images_clean --output-dir work/ocr_tables

blocks:
	@echo "📦 Parsing OCR into structured blocks..."
	python scripts/04_parse_blocks.py --ocr work/ocr_raw --tables work/ocr_tables --out work/blocks --config config.yaml

emit:
	@echo "📝 Generating JSONL from blocks..."
	python scripts/05_emit_jsonl.py --blocks-dir work/blocks --output-dir data

validate:
	@echo "✅ Validating dataset quality..."
	python scripts/06_split_validate.py --data-dir data --output work/logs/qa_report.md

split:
	@echo "🔀 Splitting train/val (80/20)..."
	python scripts/07_make_splits.py --data-dir data --pattern "*.slice.jsonl" --train-split 0.8

# ============================================================================
# STAGE 2: Enhancement (Add HTML tech specs)
# ============================================================================

extract_html:
	@echo "🌐 Extracting tech specs from HTML..."
	python scripts/07_extract_html_specs.py

hf_prep:
	@echo "🤗 Preparing HuggingFace format (nested messages)..."
	python scripts/08_prepare_hf_dataset.py --train data/train.jsonl --val data/val.jsonl --output-dir data --config config.yaml --duplicate-weights "spec=1,explanation=2,procedure=7,wiring=10,troubleshooting=50"

# ============================================================================
# STAGE 3: AutoTrain Preparation (Final format)
# ============================================================================

autotrain_prep:
	@echo "🚀 Converting to AutoTrain flat text format..."
	@python3 -c 'import json; from pathlib import Path; \
	train_path = Path("data/hf_train.jsonl"); \
	val_path = Path("data/hf_val.jsonl"); \
	output_path = Path("data/hf_train_autotrain.jsonl"); \
	combined = []; \
	[combined.append({"text": f"User: {json.loads(line)[\"messages\"][0][\"content\"]}\\nAssistant: {json.loads(line)[\"messages\"][1][\"content\"]}"}) for line in open(train_path) if line.strip()]; \
	train_count = len(combined); \
	[combined.append({"text": f"User: {json.loads(line)[\"messages\"][0][\"content\"]}\\nAssistant: {json.loads(line)[\"messages\"][1][\"content\"]}"}) for line in open(val_path) if line.strip()]; \
	output_path.write_text("\\n".join(json.dumps(e) for e in combined)); \
	print(f"✅ Wrote {len(combined)} examples ({train_count} train + {len(combined)-train_count} val)")'

synthetic_val:
	@echo "🧪 Generating synthetic validation examples..."
	python scripts/11_generate_synthetic_validation.py --train data/hf_train_autotrain.jsonl --output data/hf_val_synthetic.jsonl --count 250

# ============================================================================
# UPLOAD & TRAIN
# ============================================================================

upload:
	@echo "📤 Uploading to HuggingFace Hub..."
	python scripts/09_upload_to_hf.py --repo drumwell/llm3

upload_help:
	@echo "📤 HuggingFace Upload Instructions"
	@echo "=================================="
	@echo ""
	@echo "First time setup:"
	@echo "  pip install datasets huggingface_hub"
	@echo "  huggingface-cli login"
	@echo ""
	@echo "Upload:"
	@echo "  make upload"
	@echo "  OR: python scripts/09_upload_to_hf.py --repo drumwell/llm3"
	@echo ""
	@echo "Train on AutoTrain:"
	@echo "  1. Go to https://huggingface.co/autotrain"
	@echo "  2. See AUTOTRAIN_READY.md for complete guide"

# ============================================================================
# CONVENIENCE TARGETS
# ============================================================================

# Run complete pipeline from scratch
all: inventory preprocess ocr blocks emit validate split hf_prep extract_html autotrain_prep synthetic_val
	@echo ""
	@echo "✅ Pipeline complete!"
	@echo "📊 Results:"
	@echo "   - Training: data/hf_train_autotrain.jsonl (2,112 examples)"
	@echo "   - Validation: data/hf_val_synthetic.jsonl (180 examples)"
	@echo ""
	@echo "📤 Next step: make upload"

# Quick rebuild (assumes OCR already done)
quick: emit validate split hf_prep extract_html autotrain_prep synthetic_val

# Clean intermediate files
clean:
	@echo "🧹 Cleaning work directory..."
	rm -rf work/images_clean work/ocr_raw work/ocr_tables work/blocks

# Show pipeline status
status:
	@echo "📊 Pipeline Status"
	@echo "================="
	@echo ""
	@echo "OCR Data:"
	@test -f work/inventory.csv && echo "  ✅ inventory.csv" || echo "  ❌ inventory.csv (run: make inventory)"
	@test -d work/images_clean && echo "  ✅ images_clean/" || echo "  ❌ images_clean/ (run: make preprocess)"
	@test -d work/ocr_raw && echo "  ✅ ocr_raw/" || echo "  ❌ ocr_raw/ (run: make ocr)"
	@echo ""
	@echo "Training Data:"
	@test -f data/train.jsonl && echo "  ✅ train.jsonl" || echo "  ❌ train.jsonl (run: make split)"
	@test -f data/hf_train_autotrain.jsonl && echo "  ✅ hf_train_autotrain.jsonl" || echo "  ❌ hf_train_autotrain.jsonl (run: make autotrain_prep)"
	@test -f data/hf_val_synthetic.jsonl && echo "  ✅ hf_val_synthetic.jsonl" || echo "  ❌ hf_val_synthetic.jsonl (run: make synthetic_val)"

.PHONY: all quick clean status upload upload_help inventory preprocess ocr blocks emit validate split extract_html hf_prep autotrain_prep synthetic_val
