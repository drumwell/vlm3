# BMW E30 M3 Service Manual - Data Source Pipeline
# Self-contained pipeline: raw/ → pipeline/ → work/ → prepared/

DATA_SRC ?= raw
SECT_FILTER?=

# ============================================================================
# STAGE 1: INVENTORY
# ============================================================================

inventory:
	@echo "Stage 1: Cataloging source files..."
	python pipeline/01_inventory.py \
		--data-src $(DATA_SRC) \
		--output work/inventory.csv \
		--section-filter "$(SECT_FILTER)"

# ============================================================================
# STAGE 2: SOURCE PREPARATION
# ============================================================================

prepare:
	@echo "Stage 2: Converting PDFs and validating images..."
	python pipeline/02_prepare_sources.py \
		--inventory work/inventory.csv \
		--data-src $(DATA_SRC) \
		--output work/inventory_prepared.csv \
		--log work/logs/source_preparation.csv

# ============================================================================
# STAGE 3: CLASSIFICATION & INDEX PARSING
# ============================================================================

classify:
	@echo "Stage 3: Classifying pages and parsing indices..."
	python pipeline/03_classify_pages.py \
		--inventory work/inventory_prepared.csv \
		--output-csv work/classified/pages.csv \
		--output-indices work/indices \
		--config config.yaml

# Optional: Validate classification results
classify-validate:
	@echo "Validating classification results..."
	python pipeline/03b_validate_classification.py \
		--classified work/classified/pages.csv \
		--indices work/indices \
		--output work/logs/classification_report.md

# ============================================================================
# STAGE 4: Q&A GENERATION
# ============================================================================

generate-qa-images:
	@echo "Stage 4a: Generating Q&A from images via Claude API..."
	python pipeline/04a_generate_qa_images.py \
		--classified work/classified/pages.csv \
		--indices work/indices \
		--data-src $(DATA_SRC) \
		--output work/qa_raw \
		--config config.yaml

generate-qa-html:
	@echo "Stage 4b: Generating Q&A from HTML specs..."
	python pipeline/04b_generate_qa_html.py \
		--data-src $(DATA_SRC) \
		--output work/qa_raw \
		--config config.yaml

generate-qa: generate-qa-images generate-qa-html

# ============================================================================
# STAGE 5: Q&A QUALITY CONTROL
# ============================================================================

filter-qa:
	@echo "Stage 5a: Filtering Q&A for quality..."
	python pipeline/05_filter_qa.py \
		--input work/qa_raw \
		--output work/qa_filtered \
		--log work/logs/qa_filtered_out.csv \
		--report work/logs/qa_filter_report.md \
		--config config.yaml

deduplicate-qa:
	@echo "Stage 5b: Deduplicating Q&A pairs..."
	python pipeline/06_deduplicate_qa.py \
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
	@echo "Stage 6a: Emitting VLM training dataset..."
	python pipeline/07_emit_vlm_dataset.py \
		--qa work/qa_unique \
		--data-src $(DATA_SRC) \
		--output prepared \
		--prefix manual \
		--report work/logs/emit_report.md \
		--config config.yaml

validate:
	@echo "Stage 6b: Validating VLM dataset..."
	python pipeline/08_validate_vlm.py \
		--train prepared/manual_train.jsonl \
		--val prepared/manual_val.jsonl \
		--images prepared \
		--output work/logs/vlm_qa_report.md \
		--config config.yaml

upload:
	@echo "Stage 6c: Uploading to HuggingFace Hub..."
	python pipeline/09_upload_vlm.py \
		--train prepared/manual_train.jsonl \
		--val prepared/manual_val.jsonl \
		--images prepared/images \
		--repo drumwell/vlm3 \
		--report work/logs/upload_report.md \
		--config config.yaml

# ============================================================================
# CONVENIENCE TARGETS
# ============================================================================

# Run complete pipeline from scratch
all: inventory prepare classify generate-qa quality-control emit validate
	@echo ""
	@echo "Manual pipeline complete!"
	@echo "Output in prepared/"

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
	@echo "Cleaning Q&A artifacts..."
	rm -rf work/qa_raw work/qa_filtered work/qa_unique
	rm -f work/logs/qa_*.csv work/logs/vlm_qa_report.md

# Clean classification (keeps inventory)
clean-classify:
	@echo "Cleaning classification artifacts..."
	rm -rf work/classified work/indices

# Clean all intermediate files
clean:
	@echo "Cleaning all work artifacts..."
	rm -rf work/

# ============================================================================
# STATUS
# ============================================================================

status:
	@echo "Manual Pipeline Status"
	@echo "======================"
	@echo ""
	@echo "Stage 1 - Inventory:"
	@test -f work/inventory.csv && echo "  done" || echo "  pending (run: make inventory)"
	@echo ""
	@echo "Stage 2 - Source Preparation:"
	@test -f work/inventory_prepared.csv && echo "  done" || echo "  pending (run: make prepare)"
	@echo ""
	@echo "Stage 3 - Classification:"
	@test -f work/classified/pages.csv && echo "  done" || echo "  pending (run: make classify)"
	@test -d work/indices && echo "  indices: done" || echo "  indices: pending"
	@echo ""
	@echo "Stage 4 - Q&A Generation:"
	@test -d work/qa_raw && echo "  done" || echo "  pending (run: make generate-qa)"
	@echo ""
	@echo "Stage 5 - Quality Control:"
	@test -d work/qa_filtered && echo "  filtered: done" || echo "  filtered: pending"
	@test -d work/qa_unique && echo "  deduplicated: done" || echo "  deduplicated: pending"
	@echo ""
	@echo "Prepared Output:"
	@test -f prepared/manual_train.jsonl && echo "  train: $$(wc -l < prepared/manual_train.jsonl) examples" || echo "  train: pending (run: make emit)"
	@test -f prepared/manual_val.jsonl && echo "  val:   $$(wc -l < prepared/manual_val.jsonl) examples" || echo "  val:   pending (run: make emit)"
	@test -d prepared/images && echo "  images: $$(ls prepared/images/ | wc -l) files" || echo "  images: pending"

.PHONY: all quick regen-qa refilter finalize \
        inventory prepare classify classify-validate \
        generate-qa generate-qa-images generate-qa-html \
        filter-qa deduplicate-qa quality-control \
        emit validate upload \
        clean clean-qa clean-classify \
        status
