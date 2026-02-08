# BMW E30 M3 Service Manual - VLM Dataset Pipeline
# Converts scanned service manual pages to Vision-Language Model training format

DATA_SRC ?= data_src
SECT_FILTER?=

# ============================================================================
# STAGE 1: INVENTORY
# ============================================================================

inventory:
	@echo "📋 Stage 1: Cataloging source files..."
	python pipeline/scripts/01_inventory.py \
		--data-src $(DATA_SRC) \
		--output work/inventory.csv \
		--section-filter "$(SECT_FILTER)"

# ============================================================================
# STAGE 2: SOURCE PREPARATION
# ============================================================================

prepare:
	@echo "🔄 Stage 2: Converting PDFs and validating images..."
	python pipeline/scripts/02_prepare_sources.py \
		--inventory work/inventory.csv \
		--data-src $(DATA_SRC) \
		--output work/inventory_prepared.csv \
		--log work/logs/source_preparation.csv

# ============================================================================
# STAGE 3: CLASSIFICATION & INDEX PARSING
# ============================================================================

classify:
	@echo "🏷️  Stage 3: Classifying pages and parsing indices..."
	python pipeline/scripts/03_classify_pages.py \
		--inventory work/inventory_prepared.csv \
		--output-csv work/classified/pages.csv \
		--output-indices work/indices \
		--config pipeline/config.yaml

# Optional: Validate classification results
classify-validate:
	@echo "✅ Validating classification results..."
	python pipeline/scripts/03b_validate_classification.py \
		--classified work/classified/pages.csv \
		--indices work/indices \
		--output work/logs/classification_report.md

# ============================================================================
# STAGE 4: Q&A GENERATION
# ============================================================================

generate-qa-images:
	@echo "🤖 Stage 4a: Generating Q&A from images via Claude API..."
	python pipeline/scripts/04a_generate_qa_images.py \
		--classified work/classified/pages.csv \
		--indices work/indices \
		--data-src $(DATA_SRC) \
		--output work/qa_raw \
		--config pipeline/config.yaml

generate-qa-html:
	@echo "📄 Stage 4b: Generating Q&A from HTML specs..."
	python pipeline/scripts/04b_generate_qa_html.py \
		--data-src $(DATA_SRC) \
		--output work/qa_raw \
		--config pipeline/config.yaml

generate-qa: generate-qa-images generate-qa-html

# ============================================================================
# STAGE 5: Q&A QUALITY CONTROL
# ============================================================================

filter-qa:
	@echo "🔍 Stage 5a: Filtering Q&A for quality..."
	python pipeline/scripts/05_filter_qa.py \
		--input work/qa_raw \
		--output work/qa_filtered \
		--log work/logs/qa_filtered_out.csv \
		--report work/logs/qa_filter_report.md \
		--config pipeline/config.yaml

deduplicate-qa:
	@echo "🧹 Stage 5b: Deduplicating Q&A pairs..."
	python pipeline/scripts/06_deduplicate_qa.py \
		--input work/qa_filtered \
		--output work/qa_unique \
		--log work/logs/qa_duplicates.csv \
		--report work/logs/qa_dedup_report.md \
		--config pipeline/config.yaml

quality-control: filter-qa deduplicate-qa

# ============================================================================
# STAGE 6: EMIT & VALIDATE
# ============================================================================

emit:
	@echo "📝 Stage 6a: Emitting VLM training dataset..."
	python pipeline/scripts/07_emit_vlm_dataset.py \
		--qa work/qa_unique \
		--data-src $(DATA_SRC) \
		--output training_data \
		--report work/logs/emit_report.md \
		--config pipeline/config.yaml

validate:
	@echo "✅ Stage 6b: Validating VLM dataset..."
	python pipeline/scripts/08_validate_vlm.py \
		--train training_data/vlm_train.jsonl \
		--val training_data/vlm_val.jsonl \
		--images training_data \
		--output work/logs/vlm_qa_report.md \
		--config pipeline/config.yaml

upload:
	@echo "📤 Stage 6c: Uploading to HuggingFace Hub..."
	python pipeline/scripts/09_upload_vlm.py \
		--train training_data/vlm_train.jsonl \
		--val training_data/vlm_val.jsonl \
		--images training_data/images \
		--repo drumwell/vlm3 \
		--report work/logs/upload_report.md \
		--config pipeline/config.yaml

# ============================================================================
# CONVENIENCE TARGETS
# ============================================================================

# Run complete pipeline from scratch
all: inventory prepare classify generate-qa quality-control emit validate
	@echo ""
	@echo "✅ VLM Pipeline complete!"
	@echo "📊 Results:"
	@echo "   - Training: training_data/vlm_train.jsonl"
	@echo "   - Validation: training_data/vlm_val.jsonl"
	@echo "   - Images: training_data/images/"
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
	rm -rf work/ training_data/

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
	@test -f training_data/vlm_train.jsonl && echo "  ✅ training_data/vlm_train.jsonl" || echo "  ❌ training_data/vlm_train.jsonl (run: make emit)"
	@test -f training_data/vlm_val.jsonl && echo "  ✅ training_data/vlm_val.jsonl" || echo "  ❌ training_data/vlm_val.jsonl (run: make emit)"
	@test -d training_data/images && echo "  ✅ training_data/images/" || echo "  ❌ training_data/images/ (run: make emit)"
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
	@echo "Training:"
	@echo "  make train            Full training on Modal"
	@echo "  make train-dev        Dev training (100 samples)"
	@echo "  make train-resume     Resume from checkpoint"
	@echo "  make train-logs       Check training logs (after crash)"
	@echo ""
	@echo "Evaluation (Local GPU):"
	@echo "  make eval-sample      Create stratified eval sample (~300 examples)"
	@echo "  make eval-baseline    Evaluate base model (requires local GPU)"
	@echo "  make eval-finetuned   Evaluate fine-tuned (requires ADAPTER_PATH + GPU)"
	@echo "  make eval-compare     Generate comparison report"
	@echo "  make eval-probes      Run manual probe tests"
	@echo "  make eval-mock        Test eval infrastructure (no GPU)"
	@echo "  make eval-test        Run pytest evaluation tests"
	@echo ""
	@echo "Evaluation (Modal Cloud GPU - Recommended):"
	@echo "  make eval-modal-baseline    Baseline eval on Modal"
	@echo "  make eval-modal-finetuned   Fine-tuned eval (requires ADAPTER_REPO)"
	@echo "  make eval-modal-checkpoint  Fine-tuned eval (from Modal checkpoint)"
	@echo "  make eval-modal-probes      Manual probes on Modal"
	@echo "  make eval-modal-quick       Quick test (10 samples)"
	@echo "  make eval-modal-all         Full pipeline on Modal"
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
	@echo "  make clean-eval       Clean evaluation artifacts"
	@echo "  make clean-all        Clean everything"
	@echo "  make help             Show this help"

# ============================================================================
# STAGE 7: TRAINING (requires Modal setup - see training/README.md)
# ============================================================================

# Variables for training (override via command line)
HF_DATASET_REPO ?= drumwell/vlm3
HF_MODEL_REPO ?= drumwell/vlm3-lora

train:
	@echo "🚀 Starting full training on Modal..."
	@echo "   Dataset: $(HF_DATASET_REPO)"
	@echo "   Output:  $(HF_MODEL_REPO)"
	modal run training/modal_train.py::main \
		--dataset-repo $(HF_DATASET_REPO) \
		--output-repo $(HF_MODEL_REPO)

train-dev:
	@echo "🧪 Starting dev training run (100 samples)..."
	modal run training/modal_train.py::main \
		--dataset-repo $(HF_DATASET_REPO) \
		--max-samples 100

train-resume:
	@echo "🔄 Resuming training from checkpoint (detached)..."
	modal run --detach training/modal_train.py::main \
		--dataset-repo $(HF_DATASET_REPO) \
		--output-repo $(HF_MODEL_REPO) \
		--resume

train-logs:
	@echo "📋 Checking training logs from Modal volume..."
	modal run training/modal_train.py::check_logs_cli

# ============================================================================
# STAGE 8: EVALUATION (compare baseline vs fine-tuned models)
# ============================================================================

# Variables for evaluation
ADAPTER_PATH ?=
EVAL_SAMPLES ?= 300

# Create stratified evaluation sample from validation set
eval-sample:
	@echo "📊 Creating stratified evaluation sample..."
	python eval/sample_eval_set.py \
		--input training_data/vlm_val.jsonl \
		--output eval/eval_sample.jsonl \
		--n-samples $(EVAL_SAMPLES) \
		--stats-output eval/reports/sample_stats.json

# Run baseline model evaluation (Qwen2-VL-7B-Instruct without fine-tuning)
eval-baseline:
	@echo "🔬 Running baseline evaluation..."
	python eval/run_eval.py \
		--model Qwen/Qwen2-VL-7B-Instruct \
		--eval-set eval/eval_sample.jsonl \
		--image-base training_data \
		--output eval/reports/baseline.json

# Run fine-tuned model evaluation (with LoRA adapter)
eval-finetuned:
	@test -n "$(ADAPTER_PATH)" || (echo "Error: ADAPTER_PATH not set. Usage: make eval-finetuned ADAPTER_PATH=/path/to/adapter" && exit 1)
	@echo "🔬 Running fine-tuned evaluation..."
	@echo "   Adapter: $(ADAPTER_PATH)"
	python eval/run_eval.py \
		--model Qwen/Qwen2-VL-7B-Instruct \
		--adapter $(ADAPTER_PATH) \
		--eval-set eval/eval_sample.jsonl \
		--image-base training_data \
		--output eval/reports/finetuned.json

# Generate comparison report between baseline and fine-tuned
eval-compare:
	@echo "📈 Generating comparison report..."
	python eval/compare_results.py \
		--baseline eval/reports/baseline.json \
		--finetuned eval/reports/finetuned.json \
		--output eval/reports/comparison.md
	@echo ""
	@echo "📊 Report saved to: eval/reports/comparison.md"

# Run evaluation on manual probes (hand-crafted test cases)
eval-probes:
	@echo "🎯 Running manual probe evaluation..."
	python eval/run_eval.py \
		--model Qwen/Qwen2-VL-7B-Instruct \
		$(if $(ADAPTER_PATH),--adapter $(ADAPTER_PATH),) \
		--eval-set eval/benchmarks/manual_probes.json \
		--image-base training_data \
		--output eval/reports/probes.json

# ----------------------------------------------------------------------------
# Modal-based evaluation (runs on cloud GPU - no local GPU required)
# ----------------------------------------------------------------------------

# Variables for Modal evaluation
ADAPTER_REPO ?=

# Run baseline evaluation on Modal (recommended if no local GPU)
eval-modal-baseline:
	@echo "🔬 Running baseline evaluation on Modal..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--output eval/reports/baseline.json

# Run fine-tuned evaluation on Modal (adapter from HuggingFace)
eval-modal-finetuned:
	@test -n "$(ADAPTER_REPO)" || (echo "Error: ADAPTER_REPO not set. Usage: make eval-modal-finetuned ADAPTER_REPO=username/vlm3-lora" && exit 1)
	@echo "🔬 Running fine-tuned evaluation on Modal..."
	@echo "   Adapter: $(ADAPTER_REPO)"
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--adapter-repo $(ADAPTER_REPO) \
		--output eval/reports/finetuned.json

# Run fine-tuned evaluation on Modal (adapter from Modal volume checkpoint)
eval-modal-checkpoint:
	@echo "🔬 Running fine-tuned evaluation on Modal (from checkpoint)..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--adapter-path /checkpoints/vlm3-lora/final \
		--output eval/reports/finetuned.json

# Quick Modal evaluation test (10 samples)
eval-modal-quick:
	@echo "🧪 Running quick Modal evaluation (10 samples)..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--max-samples 10

# Run only manual probes on Modal
eval-modal-probes:
	@echo "🎯 Running manual probes on Modal..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		$(if $(ADAPTER_REPO),--adapter-repo $(ADAPTER_REPO),) \
		--probes-only

# Full Modal evaluation pipeline: baseline -> train -> finetuned -> compare
eval-modal-all: eval-sample eval-modal-baseline train eval-modal-checkpoint eval-compare
	@echo ""
	@echo "✅ Full Modal evaluation pipeline complete!"
	@echo "📊 Results:"
	@echo "   - Baseline: eval/reports/baseline.json"
	@echo "   - Fine-tuned: eval/reports/finetuned.json"
	@echo "   - Comparison: eval/reports/comparison.md"

# Run mock evaluation (for testing infrastructure without GPU)
eval-mock:
	@echo "🧪 Running mock evaluation (no GPU required)..."
	python eval/run_eval.py \
		--mock \
		--eval-set eval/eval_sample.jsonl \
		--image-base training_data \
		--max-samples 10 \
		--output eval/reports/mock.json

# Run pytest evaluation tests
eval-test:
	@echo "🧪 Running evaluation unit tests..."
	pytest eval/test_vlm.py -v

# Run full evaluation pipeline: sample -> baseline -> finetuned -> compare
eval-all: eval-sample eval-baseline eval-finetuned eval-compare eval-probes
	@echo ""
	@echo "✅ Full evaluation complete!"
	@echo "📊 Results:"
	@echo "   - Baseline: eval/reports/baseline.json"
	@echo "   - Fine-tuned: eval/reports/finetuned.json"
	@echo "   - Comparison: eval/reports/comparison.md"
	@echo "   - Probes: eval/reports/probes.json"

# Clean evaluation artifacts
clean-eval:
	@echo "🧹 Cleaning evaluation artifacts..."
	rm -f eval/eval_sample.jsonl
	rm -rf eval/reports/*.json eval/reports/*.md

.PHONY: all quick regen-qa refilter finalize \
        inventory prepare classify classify-validate \
        generate-qa generate-qa-images generate-qa-html \
        filter-qa deduplicate-qa quality-control \
        emit validate upload \
        train train-dev train-resume train-logs \
        eval-sample eval-baseline eval-finetuned eval-compare eval-probes eval-mock eval-test eval-all clean-eval \
        eval-modal-baseline eval-modal-finetuned eval-modal-checkpoint eval-modal-quick eval-modal-probes eval-modal-all \
        clean clean-qa clean-classify clean-all \
        status help
