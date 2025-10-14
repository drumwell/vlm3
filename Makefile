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
	python scripts/06_validate.py --data-dir data --file dataset.jsonl --output work/logs/qa_report.md

# ============================================================================
# STAGE 2: Enhancement (Add HTML tech specs)
# ============================================================================

extract_html:
	@echo "🌐 Extracting tech specs from HTML..."
	python scripts/07_extract_html_specs.py

# ============================================================================
# STAGE 3: AutoTrain Preparation (Final format)
# ============================================================================

autotrain_prep:
	@echo "🚀 Converting to AutoTrain flat text format..."
	python scripts/08_convert_to_autotrain.py

synthetic_val:
	@echo "🧪 Generating synthetic validation examples..."
	python scripts/09_generate_synthetic_validation.py --train data/hf_train_autotrain.jsonl --output data/hf_val_synthetic.jsonl --count 250

# ============================================================================
# UPLOAD & TRAIN
# ============================================================================

upload:
	@echo "📤 Uploading to HuggingFace Hub..."
	python scripts/10_upload_to_hf.py --repo drumwell/llm3

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
all: inventory preprocess ocr blocks emit validate extract_html autotrain_prep synthetic_val
	@echo ""
	@echo "✅ Pipeline complete!"
	@echo "📊 Results:"
	@echo "   - Training: data/hf_train_autotrain.jsonl"
	@echo "   - Validation: data/hf_val_synthetic.jsonl"
	@echo ""
	@echo "📤 Next step: make upload"

# Quick rebuild (assumes OCR already done)
quick: emit validate extract_html autotrain_prep synthetic_val

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
	@test -f data/dataset.jsonl && echo "  ✅ dataset.jsonl" || echo "  ❌ dataset.jsonl (run: make emit extract_html)"
	@test -f data/hf_train_autotrain.jsonl && echo "  ✅ hf_train_autotrain.jsonl" || echo "  ❌ hf_train_autotrain.jsonl (run: make autotrain_prep)"
	@test -f data/hf_val_synthetic.jsonl && echo "  ✅ hf_val_synthetic.jsonl" || echo "  ❌ hf_val_synthetic.jsonl (run: make synthetic_val)"

.PHONY: all quick clean status upload upload_help inventory preprocess ocr blocks emit validate extract_html autotrain_prep synthetic_val
