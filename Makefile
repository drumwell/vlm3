# BMW E30 M3 VLM Project
# Root Makefile — delegates to data/, training/, eval/

# ============================================================================
# DATA PIPELINE (delegates to data/)
# ============================================================================

data:
	$(MAKE) -C data all

data-manual:
	$(MAKE) -C data/src/manual all

data-status:
	$(MAKE) -C data status

data-merge:
	$(MAKE) -C data merge

data-clean:
	$(MAKE) -C data clean

# ============================================================================
# TRAINING (requires Modal setup - see training/README.md)
# ============================================================================

# Variables for training (override via command line)
HF_DATASET_REPO ?= drumwell/vlm3
HF_MODEL_REPO ?= drumwell/vlm3-lora

upload:
	python data/src/manual/pipeline/09_upload_vlm.py \
		--train data/training/merged_train.jsonl \
		--val data/training/merged_val.jsonl \
		--images data/training/images \
		--repo $(HF_DATASET_REPO) \
		--report data/training/upload_report.md \
		--config data/src/manual/config.yaml

train:
	@echo "Starting full training on Modal (detached)..."
	@echo "   Dataset: $(HF_DATASET_REPO)"
	@echo "   Output:  $(HF_MODEL_REPO)"
	modal run --detach training/modal_train.py::main \
		--dataset-repo $(HF_DATASET_REPO) \
		--output-repo $(HF_MODEL_REPO)

train-dev:
	@echo "Starting dev training run (100 samples)..."
	modal run training/modal_train.py::main \
		--dataset-repo $(HF_DATASET_REPO) \
		--max-samples 100

train-resume:
	@echo "Resuming training from checkpoint (detached)..."
	modal run --detach training/modal_train.py::main \
		--dataset-repo $(HF_DATASET_REPO) \
		--output-repo $(HF_MODEL_REPO) \
		--resume

train-logs:
	@echo "Checking training logs from Modal volume..."
	modal run training/modal_train.py::check_logs_cli

train-archive:
	@echo "Archiving current training run on Modal..."
	modal run training/modal_train.py::archive_run_cli

train-runs:
	@echo "Listing training runs on Modal..."
	modal run training/modal_train.py::list_runs_cli

train-clean:
	@echo "Cleaning current training run from Modal volume..."
	modal run training/modal_train.py::clean_runs_cli

# ============================================================================
# EVALUATION (compare baseline vs fine-tuned models)
# ============================================================================

# Variables for evaluation
ADAPTER_PATH ?=
EVAL_SAMPLES ?= 300
ADAPTER_REPO ?=

# Create stratified evaluation sample from validation set
eval-sample:
	@echo "Creating stratified evaluation sample..."
	python eval/sample_eval_set.py \
		--input data/src/manual/prepared/manual_val.jsonl \
		--output eval/eval_sample.jsonl \
		--n-samples $(EVAL_SAMPLES) \
		--stats-output eval/reports/sample_stats.json

# Run baseline model evaluation (Qwen2-VL-7B-Instruct without fine-tuning)
eval-baseline:
	@echo "Running baseline evaluation..."
	python eval/run_eval.py \
		--model Qwen/Qwen2-VL-7B-Instruct \
		--eval-set eval/eval_sample.jsonl \
		--image-base data/src/manual/prepared \
		--output eval/reports/baseline.json

# Run fine-tuned model evaluation (with LoRA adapter)
eval-finetuned:
	@test -n "$(ADAPTER_PATH)" || (echo "Error: ADAPTER_PATH not set. Usage: make eval-finetuned ADAPTER_PATH=/path/to/adapter" && exit 1)
	@echo "Running fine-tuned evaluation..."
	@echo "   Adapter: $(ADAPTER_PATH)"
	python eval/run_eval.py \
		--model Qwen/Qwen2-VL-7B-Instruct \
		--adapter $(ADAPTER_PATH) \
		--eval-set eval/eval_sample.jsonl \
		--image-base data/src/manual/prepared \
		--output eval/reports/finetuned.json

# Generate comparison report between baseline and fine-tuned
eval-compare:
	@echo "Generating comparison report..."
	python eval/compare_results.py \
		--baseline eval/reports/baseline.json \
		--finetuned eval/reports/finetuned.json \
		--output eval/reports/comparison.md
	@echo ""
	@echo "Report saved to: eval/reports/comparison.md"

# Run evaluation on manual probes (hand-crafted test cases)
eval-probes:
	@echo "Running manual probe evaluation..."
	python eval/run_eval.py \
		--model Qwen/Qwen2-VL-7B-Instruct \
		$(if $(ADAPTER_PATH),--adapter $(ADAPTER_PATH),) \
		--eval-set eval/benchmarks/manual_probes.json \
		--image-base data/src/manual/prepared \
		--output eval/reports/probes.json

# ----------------------------------------------------------------------------
# Modal-based evaluation (runs on cloud GPU - no local GPU required)
# ----------------------------------------------------------------------------

# Run baseline evaluation on Modal (recommended if no local GPU)
eval-modal-baseline:
	@echo "Running baseline evaluation on Modal..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--output eval/reports/baseline.json

# Run fine-tuned evaluation on Modal (adapter from HuggingFace)
eval-modal-finetuned:
	@test -n "$(ADAPTER_REPO)" || (echo "Error: ADAPTER_REPO not set. Usage: make eval-modal-finetuned ADAPTER_REPO=username/vlm3-lora" && exit 1)
	@echo "Running fine-tuned evaluation on Modal..."
	@echo "   Adapter: $(ADAPTER_REPO)"
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--adapter-repo $(ADAPTER_REPO) \
		--output eval/reports/finetuned.json

# Run fine-tuned evaluation on Modal (adapter from Modal volume checkpoint)
eval-modal-checkpoint:
	@echo "Running fine-tuned evaluation on Modal (from checkpoint)..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--adapter-path /checkpoints/vlm3-lora/final \
		--output eval/reports/finetuned.json

# Quick Modal evaluation test (10 samples)
eval-modal-quick:
	@echo "Running quick Modal evaluation (10 samples)..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		--max-samples 10

# Run only manual probes on Modal
eval-modal-probes:
	@echo "Running manual probes on Modal..."
	modal run eval/modal_eval.py \
		--dataset-repo $(HF_DATASET_REPO) \
		$(if $(ADAPTER_REPO),--adapter-repo $(ADAPTER_REPO),) \
		--probes-only

# Full Modal evaluation pipeline: baseline -> train -> finetuned -> compare
eval-modal-all: eval-sample eval-modal-baseline train eval-modal-checkpoint eval-compare
	@echo ""
	@echo "Full Modal evaluation pipeline complete!"
	@echo "Results:"
	@echo "   - Baseline: eval/reports/baseline.json"
	@echo "   - Fine-tuned: eval/reports/finetuned.json"
	@echo "   - Comparison: eval/reports/comparison.md"

# Run mock evaluation (for testing infrastructure without GPU)
eval-mock:
	@echo "Running mock evaluation (no GPU required)..."
	python eval/run_eval.py \
		--mock \
		--eval-set eval/eval_sample.jsonl \
		--image-base data/src/manual/prepared \
		--max-samples 10 \
		--output eval/reports/mock.json

# Run pytest evaluation tests
eval-test:
	@echo "Running evaluation unit tests..."
	pytest eval/test_vlm.py -v

# Run full evaluation pipeline: sample -> baseline -> finetuned -> compare
eval-all: eval-sample eval-baseline eval-finetuned eval-compare eval-probes
	@echo ""
	@echo "Full evaluation complete!"
	@echo "Results:"
	@echo "   - Baseline: eval/reports/baseline.json"
	@echo "   - Fine-tuned: eval/reports/finetuned.json"
	@echo "   - Comparison: eval/reports/comparison.md"
	@echo "   - Probes: eval/reports/probes.json"

# Clean evaluation artifacts
clean-eval:
	@echo "Cleaning evaluation artifacts..."
	rm -f eval/eval_sample.jsonl
	rm -rf eval/reports/*.json eval/reports/*.md

# Archive / list eval report sets
# Usage: make eval-archive [LABEL=v1-manual-only]
LABEL ?=
eval-archive:
	@if ls eval/reports/*.json eval/reports/*.md 1>/dev/null 2>&1; then \
		tag=$$(date +%Y%m%d_%H%M%S); \
		if [ -n "$(LABEL)" ]; then tag="$${tag}_$(LABEL)"; fi; \
		mkdir -p eval/reports/archive/run_$$tag; \
		for f in eval/reports/*.json eval/reports/*.md; do \
			case "$$(basename $$f)" in multi_run_comparison.md) continue;; esac; \
			mv "$$f" eval/reports/archive/run_$$tag/; \
		done; \
		python eval/run_meta.py eval/reports/archive/run_$$tag --auto \
			$$([ -n "$(LABEL)" ] && echo "--label $(LABEL)"); \
		echo "Archived to eval/reports/archive/run_$$tag/"; \
	else echo "No eval reports to archive"; fi

# Retroactively label an archived run
# Usage: make eval-label RUN=run_20260207_134206 LABEL=v1-manual-only
RUN ?=
eval-label:
	@test -n "$(RUN)" || (echo "Error: RUN not set. Usage: make eval-label RUN=run_... LABEL=..." && exit 1)
	@test -d "eval/reports/archive/$(RUN)" || (echo "Error: eval/reports/archive/$(RUN) not found" && exit 1)
	python eval/run_meta.py eval/reports/archive/$(RUN) --auto \
		$$([ -n "$(LABEL)" ] && echo "--label $(LABEL)")

# Multi-run progression comparison
eval-compare-runs:
	@echo "Generating multi-run comparison report..."
	python eval/compare_runs.py --output eval/reports/multi_run_comparison.md
	@echo ""
	@echo "Report saved to: eval/reports/multi_run_comparison.md"

eval-runs:
	@echo "Archived eval runs:"
	@if [ -d eval/reports/archive ]; then \
		for d in eval/reports/archive/run_*; do \
			name=$$(basename $$d); \
			files=$$(ls $$d/*.json 2>/dev/null | grep -v run_meta | wc -l | tr -d ' '); \
			if [ -f "$$d/run_meta.json" ]; then \
				label=$$(python3 -c "import json; print(json.load(open('$$d/run_meta.json')).get('label',''))" 2>/dev/null); \
				if [ -n "$$label" ]; then \
					echo "  $$name  $$files files  [$$label]"; \
				else \
					echo "  $$name  $$files files"; \
				fi; \
			else \
				echo "  $$name  $$files files"; \
			fi; \
		done; \
	else echo "  (none)"; fi

# ============================================================================
# HELP
# ============================================================================

help:
	@echo "BMW E30 M3 VLM Project"
	@echo "======================"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make data              Run all data source pipelines"
	@echo "  make data-manual       Run manual pipeline only"
	@echo "  make data-status       Show pipeline status"
	@echo "  make data-merge        Run merge layer only"
	@echo "  make data-clean        Clean data pipeline artifacts"
	@echo "  make upload            Upload merged dataset to HuggingFace"
	@echo ""
	@echo "Manual Pipeline (run from data/src/manual/):"
	@echo "  make -C data/src/manual all        Full pipeline"
	@echo "  make -C data/src/manual status     Show progress"
	@echo "  make -C data/src/manual quick      Skip Stages 1-2"
	@echo "  make -C data/src/manual regen-qa   Regenerate from Stage 4"
	@echo "  make -C data/src/manual refilter   Rerun from Stage 5"
	@echo ""
	@echo "Training:"
	@echo "  make train             Full training on Modal (detached)"
	@echo "  make train-dev         Dev training (100 samples)"
	@echo "  make train-resume      Resume from checkpoint"
	@echo "  make train-logs        Check training logs"
	@echo "  make train-archive     Archive current run on Modal"
	@echo "  make train-runs        List training runs on Modal"
	@echo "  make train-clean       Delete current run from Modal volume"
	@echo ""
	@echo "Evaluation (Local GPU):"
	@echo "  make eval-sample       Create stratified eval sample"
	@echo "  make eval-baseline     Evaluate base model"
	@echo "  make eval-finetuned    Evaluate fine-tuned (requires ADAPTER_PATH)"
	@echo "  make eval-compare      Generate comparison report"
	@echo "  make eval-probes       Run manual probe tests"
	@echo "  make eval-mock         Test eval infrastructure (no GPU)"
	@echo "  make eval-test         Run pytest evaluation tests"
	@echo ""
	@echo "Evaluation (Modal Cloud GPU):"
	@echo "  make eval-modal-baseline    Baseline eval on Modal"
	@echo "  make eval-modal-finetuned   Fine-tuned eval (requires ADAPTER_REPO)"
	@echo "  make eval-modal-checkpoint  Fine-tuned eval (from checkpoint)"
	@echo "  make eval-modal-quick       Quick test (10 samples)"
	@echo "  make eval-modal-probes      Manual probes on Modal"
	@echo "  make eval-modal-all         Full pipeline on Modal"
	@echo ""
	@echo "Utilities:"
	@echo "  make help              Show this help"
	@echo "  make clean-eval        Clean evaluation artifacts"
	@echo "  make eval-archive      Archive current eval reports (LABEL= optional)"
	@echo "  make eval-label        Label an archived run (RUN= LABEL=)"
	@echo "  make eval-compare-runs Multi-run progression report"
	@echo "  make eval-runs         List archived eval runs"

.PHONY: data data-manual data-merge data-status data-clean upload \
        train train-dev train-resume train-logs train-archive train-runs train-clean \
        eval-sample eval-baseline eval-finetuned eval-compare eval-probes eval-mock eval-test eval-all clean-eval \
        eval-modal-baseline eval-modal-finetuned eval-modal-checkpoint eval-modal-quick eval-modal-probes eval-modal-all \
        eval-archive eval-label eval-compare-runs eval-runs \
        help
