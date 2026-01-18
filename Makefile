# BMW E30 M3 Service Manual - VLM Dataset Pipeline
# Converts scanned service manual pages to Vision-Language Model training format

SECT_FILTER?=

# ============================================================================
# STAGE 1: INVENTORY
# ============================================================================

inventory:
	@echo "📋 Stage 1: Cataloging source files..."
	python scripts/01_inventory.py \
		--data-src data_src \
		--output work/inventory.csv \
		--section-filter "$(SECT_FILTER)"

# ============================================================================
# STAGE 2: SOURCE PREPARATION
# ============================================================================

prepare:
	@echo "🔄 Stage 2: Converting PDFs and validating images..."
	python scripts/02_prepare_sources.py \
		--inventory work/inventory.csv \
		--data-src data_src \
		--output work/inventory_prepared.csv \
		--log work/logs/source_preparation.csv

# ============================================================================
# STAGE 3: CLASSIFICATION & INDEX PARSING
# ============================================================================

classify:
	@echo "🏷️  Stage 3: Classifying pages and parsing indices..."
	python scripts/03_classify_pages.py \
		--inventory work/inventory_prepared.csv \
		--output-csv work/classified/pages.csv \
		--output-indices work/indices \
		--config config.yaml

# Optional: Validate classification results
classify-validate:
	@echo "✅ Validating classification results..."
	python scripts/03b_validate_classification.py \
		--classified work/classified/pages.csv \
		--indices work/indices \
		--output work/logs/classification_report.md

# ============================================================================
# STAGE 4: Q&A GENERATION
# ============================================================================

generate-qa-images:
	@echo "🤖 Stage 4a: Generating Q&A from images via Claude API..."
	python scripts/04a_generate_qa_images.py \
		--classified work/classified/pages.csv \
		--indices work/indices \
		--data-src data_src \
		--output work/qa_raw \
		--config config.yaml

generate-qa-html:
	@echo "📄 Stage 4b: Generating Q&A from HTML specs..."
	python scripts/04b_generate_qa_html.py \
		--data-src data_src \
		--output work/qa_raw \
		--config config.yaml

generate-qa: generate-qa-images generate-qa-html

# ============================================================================
# STAGE 5: Q&A QUALITY CONTROL
# ============================================================================

filter-qa:
	@echo "🔍 Stage 5a: Filtering Q&A for quality..."
	python scripts/05_filter_qa.py \
		--input work/qa_raw \
		--output work/qa_filtered \
		--log work/logs/qa_filtered_out.csv \
		--report work/logs/qa_filter_report.md \
		--config config.yaml

deduplicate-qa:
	@echo "🧹 Stage 5b: Deduplicating Q&A pairs..."
	python scripts/06_deduplicate_qa.py \
		--input work/qa_filtered \
		--output work/qa_unique \
		--log work/logs/qa_duplicates.csv \
		--report work/logs/qa_dedup_report.md \
		--config config.yaml

quality-control: filter-qa deduplicate-qa

# ============================================================================
# STAGE 6: EMIT & VALIDATE
# ============================================================================

emit:
	@echo "📝 Stage 6a: Emitting VLM training dataset..."
	python scripts/07_emit_vlm_dataset.py \
		--qa work/qa_unique \
		--data-src data_src \
		--output data \
		--report work/logs/emit_report.md \
		--config config.yaml

validate:
	@echo "✅ Stage 6b: Validating VLM dataset..."
	python scripts/08_validate_vlm.py \
		--train data/vlm_train.jsonl \
		--val data/vlm_val.jsonl \
		--images data \
		--output work/logs/vlm_qa_report.md \
		--config config.yaml

upload:
	@echo "📤 Stage 6c: Uploading to HuggingFace Hub..."
	python scripts/09_upload_vlm.py \
		--train data/vlm_train.jsonl \
		--val data/vlm_val.jsonl \
		--images data/images \
		--config config.yaml

# ============================================================================
# CONVENIENCE TARGETS
# ============================================================================

# Run complete pipeline from scratch
all: inventory prepare classify generate-qa quality-control emit validate
	@echo ""
	@echo "✅ VLM Pipeline complete!"
	@echo "📊 Results:"
	@echo "   - Training: data/vlm_train.jsonl"
	@echo "   - Validation: data/vlm_val.jsonl"
	@echo "   - Images: data/images/"
	@echo ""
	@echo "📤 Next step: make upload"

# Skip source preparation (already done)
quick: classify generate-qa quality-control emit validate

# Regenerate Q&A only (classification unchanged)
regen-qa: generate-qa quality-control emit validate

# Reprocess quality control only (Q&A already generated)
refilter: quality-control emit validate

# Just emit and validate (Q&A already filtered)
finalize: emit validate

# ============================================================================
# CLEAN TARGETS
# ============================================================================

# Clean Q&A artifacts only (keeps inventory and classification)
clean-qa:
	@echo "🧹 Cleaning Q&A artifacts..."
	rm -rf work/qa_raw work/qa_filtered work/qa_unique
	rm -f work/logs/qa_*.csv work/logs/vlm_qa_report.md

# Clean classification (keeps inventory)
clean-classify:
	@echo "🧹 Cleaning classification artifacts..."
	rm -rf work/classified work/indices

# Clean all intermediate files
clean:
	@echo "🧹 Cleaning all work artifacts..."
	rm -rf work/qa_raw work/qa_filtered work/qa_unique
	rm -rf work/classified work/indices
	rm -f work/logs/*.csv work/logs/*.md

# Clean everything including outputs
clean-all:
	@echo "🧹 Cleaning everything..."
	rm -rf work/ data/

# ============================================================================
# STATUS & HELP
# ============================================================================

status:
	@echo "📊 VLM Pipeline Status"
	@echo "======================"
	@echo ""
	@echo "Stage 1 - Inventory:"
	@test -f work/inventory.csv && echo "  ✅ work/inventory.csv" || echo "  ❌ work/inventory.csv (run: make inventory)"
	@echo ""
	@echo "Stage 2 - Source Preparation:"
	@test -f work/inventory_prepared.csv && echo "  ✅ work/inventory_prepared.csv" || echo "  ❌ work/inventory_prepared.csv (run: make prepare)"
	@echo ""
	@echo "Stage 3 - Classification:"
	@test -f work/classified/pages.csv && echo "  ✅ work/classified/pages.csv" || echo "  ❌ work/classified/pages.csv (run: make classify)"
	@test -d work/indices && echo "  ✅ work/indices/" || echo "  ❌ work/indices/ (run: make classify)"
	@echo ""
	@echo "Stage 4 - Q&A Generation:"
	@test -d work/qa_raw && echo "  ✅ work/qa_raw/" || echo "  ❌ work/qa_raw/ (run: make generate-qa)"
	@echo ""
	@echo "Stage 5 - Quality Control:"
	@test -d work/qa_filtered && echo "  ✅ work/qa_filtered/" || echo "  ❌ work/qa_filtered/ (run: make filter-qa)"
	@test -d work/qa_unique && echo "  ✅ work/qa_unique/" || echo "  ❌ work/qa_unique/ (run: make deduplicate-qa)"
	@echo ""
	@echo "Stage 6 - Output:"
	@test -f data/vlm_train.jsonl && echo "  ✅ data/vlm_train.jsonl" || echo "  ❌ data/vlm_train.jsonl (run: make emit)"
	@test -f data/vlm_val.jsonl && echo "  ✅ data/vlm_val.jsonl" || echo "  ❌ data/vlm_val.jsonl (run: make emit)"
	@test -d data/images && echo "  ✅ data/images/" || echo "  ❌ data/images/ (run: make emit)"
	@test -f work/logs/vlm_qa_report.md && echo "  ✅ work/logs/vlm_qa_report.md" || echo "  ❌ work/logs/vlm_qa_report.md (run: make validate)"

help:
	@echo "BMW E30 M3 Service Manual - VLM Pipeline"
	@echo "========================================="
	@echo ""
	@echo "Full Pipeline:"
	@echo "  make all              Run complete pipeline (Stages 1-6)"
	@echo ""
	@echo "Individual Stages:"
	@echo "  make inventory        Stage 1: Catalog source files"
	@echo "  make prepare          Stage 2: Convert PDFs, validate images"
	@echo "  make classify         Stage 3: Classify pages, parse indices"
	@echo "  make generate-qa      Stage 4: Generate Q&A (images + HTML)"
	@echo "  make quality-control  Stage 5: Filter and deduplicate Q&A"
	@echo "  make emit             Stage 6a: Emit VLM JSONL dataset"
	@echo "  make validate         Stage 6b: Validate dataset"
	@echo "  make upload           Stage 6c: Upload to HuggingFace"
	@echo ""
	@echo "Partial Runs:"
	@echo "  make quick            Skip Stage 1-2 (sources already prepared)"
	@echo "  make regen-qa         Regenerate from Stage 4"
	@echo "  make refilter         Rerun from Stage 5"
	@echo "  make finalize         Just emit and validate"
	@echo ""
	@echo "Utilities:"
	@echo "  make status           Show pipeline status"
	@echo "  make clean            Clean intermediate files"
	@echo "  make clean-all        Clean everything"
	@echo "  make help             Show this help"

.PHONY: all quick regen-qa refilter finalize \
        inventory prepare classify classify-validate \
        generate-qa generate-qa-images generate-qa-html \
        filter-qa deduplicate-qa quality-control \
        emit validate upload \
        clean clean-qa clean-classify clean-all \
        status help
