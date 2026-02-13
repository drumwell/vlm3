# s14.net Forum Data - Curation Pipeline
# Self-contained pipeline: raw/ → pipeline/ → work/ → prepared/

# ============================================================================
# STAGE 1: RECONSTRUCT THREADS
# ============================================================================

reconstruct:
	@echo "Stage 1: Reconstructing threads from raw posts..."
	python pipeline/01_reconstruct.py \
		--raw raw \
		--output work/threads \
		--config config.yaml

# ============================================================================
# STAGE 2: MECHANICAL FILTERING
# ============================================================================

filter:
	@echo "Stage 2: Filtering threads (mechanical rules)..."
	python pipeline/02_filter.py \
		--input work/threads \
		--output work/threads_clean \
		--report work/logs/filter_report.md \
		--config config.yaml

# ============================================================================
# STAGE 3: THREAD CLASSIFICATION
# ============================================================================

classify:
	@echo "Stage 3: Classifying threads via Claude API..."
	python pipeline/03_classify.py \
		--input work/threads_clean \
		--output work/threads_classified \
		--config config.yaml

# ============================================================================
# STAGE 4: QUALITY SCORING
# ============================================================================

score:
	@echo "Stage 4: Scoring threads via Claude API..."
	python pipeline/04_score.py \
		--input work/threads_classified \
		--output work/threads_scored \
		--config config.yaml

# ============================================================================
# STAGE 5: Q&A EXTRACTION
# ============================================================================

extract-qa:
	@echo "Stage 5: Extracting Q&A pairs via Claude API..."
	python pipeline/05_extract_qa.py \
		--input work/threads_scored \
		--output work/qa_raw \
		--config config.yaml

# ============================================================================
# STAGE 6: CROSS-REFERENCE & FILTER
# ============================================================================

validate:
	@echo "Stage 6: Cross-referencing and filtering Q&A..."
	python pipeline/06_validate.py \
		--input work/qa_raw \
		--output work/qa_validated \
		--report work/logs/crossref_report.md \
		--config config.yaml

# ============================================================================
# STAGE 7: EMIT TRAINING DATA
# ============================================================================

emit:
	@echo "Stage 7: Emitting training dataset..."
	python pipeline/07_emit.py \
		--input work/qa_validated \
		--output prepared \
		--prefix forum \
		--report work/logs/emit_report.md \
		--config config.yaml

# ============================================================================
# CONVENIENCE TARGETS
# ============================================================================

# Run complete pipeline
all: reconstruct filter classify score extract-qa validate emit
	@echo ""
	@echo "Forum pipeline complete!"
	@echo "Output in prepared/"

# Skip stages 1-2 (threads already reconstructed and filtered)
from-classify: classify score extract-qa validate emit

# Re-extract Q&A only (classification and scoring unchanged)
regen-qa: extract-qa validate emit

# Rerun from validation only
refilter: validate emit

# ============================================================================
# CLEAN TARGETS
# ============================================================================

clean-qa:
	@echo "Cleaning Q&A artifacts..."
	rm -rf work/qa_raw work/qa_validated
	rm -f work/logs/crossref_report.md work/logs/emit_report.md

clean-classify:
	@echo "Cleaning classification and scoring artifacts..."
	rm -rf work/threads_classified work/threads_scored

clean:
	@echo "Cleaning all work artifacts..."
	rm -rf work/

# ============================================================================
# STATUS
# ============================================================================

status:
	@echo "Forum Pipeline Status"
	@echo "====================="
	@echo ""
	@echo "Stage 1 - Reconstruct Threads:"
	@test -d work/threads && echo "  done ($$(ls work/threads/*.json 2>/dev/null | wc -l | tr -d ' ') threads)" || echo "  pending (run: make reconstruct)"
	@echo ""
	@echo "Stage 2 - Mechanical Filtering:"
	@test -d work/threads_clean && echo "  done ($$(ls work/threads_clean/*.json 2>/dev/null | wc -l | tr -d ' ') threads)" || echo "  pending (run: make filter)"
	@echo ""
	@echo "Stage 3 - Classification:"
	@test -d work/threads_classified && echo "  done ($$(ls work/threads_classified/*.json 2>/dev/null | wc -l | tr -d ' ') threads)" || echo "  pending (run: make classify)"
	@echo ""
	@echo "Stage 4 - Quality Scoring:"
	@test -d work/threads_scored && echo "  done ($$(ls work/threads_scored/*.json 2>/dev/null | wc -l | tr -d ' ') threads)" || echo "  pending (run: make score)"
	@echo ""
	@echo "Stage 5 - Q&A Extraction:"
	@test -d work/qa_raw && echo "  done ($$(ls work/qa_raw/*.json 2>/dev/null | wc -l | tr -d ' ') threads)" || echo "  pending (run: make extract-qa)"
	@echo ""
	@echo "Stage 6 - Cross-Reference & Filter:"
	@test -d work/qa_validated && echo "  done ($$(ls work/qa_validated/*.json 2>/dev/null | wc -l | tr -d ' ') threads)" || echo "  pending (run: make validate)"
	@echo ""
	@echo "Prepared Output:"
	@test -f prepared/forum_train.jsonl && echo "  train: $$(wc -l < prepared/forum_train.jsonl) examples" || echo "  train: pending (run: make emit)"
	@test -f prepared/forum_val.jsonl && echo "  val:   $$(wc -l < prepared/forum_val.jsonl) examples" || echo "  val:   pending (run: make emit)"

.PHONY: all from-classify regen-qa refilter \
        reconstruct filter classify score extract-qa validate emit \
        clean clean-qa clean-classify \
        status
