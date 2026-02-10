# VLM Fine-tuning & Evaluation Plan

> Implementation plan for training and evaluation infrastructure.

## Overview

**Goal:** Fine-tune Qwen2-VL-7B on the BMW E30 M3 service manual Q&A dataset and evaluate its ability to answer technical questions from manual images.

**Stack:**
- Model: Qwen2-VL-7B-Instruct
- Training: Modal (GPU cloud) + HuggingFace Transformers + LoRA
- Eval: Custom `VLMEvaluator` (ROUGE-L, NumericExactMatch, UnitConsistency, KeywordPresence) + optional DeepEval LLM-as-judge

**Current state:**
- Dataset: 11,154 train / 1,256 val examples, 1,408 images (HuggingFace: `drumwell/vlm3`)
- HuggingFace: Personal account
- Training: Implemented (`training/modal_train.py`), has been run on Modal A100-80GB
- Evaluation: Framework implemented, not yet run against a fine-tuned model

---

## Directory Structure

```
vlm3-training/
├── data/
│   ├── src/
│   │   ├── manual/                 # Service manual data source (complete)
│   │   │   ├── Makefile
│   │   │   ├── config.yaml         # Pipeline config (API model, rate limits, filters, split)
│   │   │   ├── raw/                # ~45 section folders of scanned pages
│   │   │   ├── pipeline/           # Scripts 01-09
│   │   │   ├── work/               # Intermediate artifacts (not committed)
│   │   │   ├── prepared/           # 11,154 train + 1,256 val examples + 1,408 images
│   │   │   │   ├── manual_train.jsonl
│   │   │   │   ├── manual_val.jsonl
│   │   │   │   └── images/
│   │   │   └── tests/              # 11 test files + fixtures
│   │   └── forum/                  # Forum data source (planned, raw/ stub only)
│   ├── training/                   # Merge layer (placeholder merge.py + config)
│   └── Makefile                    # Data orchestrator
├── training/                       # Modal training infrastructure
│   ├── modal_train.py              # LoRA fine-tuning on A100-80GB
│   └── configs/lora_qwen2vl.yaml   # LoRA training config
├── eval/                           # Evaluation framework
│   ├── run_eval.py                 # Local GPU evaluation runner
│   ├── modal_eval.py               # Modal cloud GPU evaluation
│   ├── sample_eval_set.py          # Stratified sampling from val set
│   ├── compare_results.py          # Generate comparison reports
│   ├── metrics.py                  # VLMEvaluator + custom metrics
│   ├── model_wrapper.py            # Model loading/inference abstraction
│   ├── test_vlm.py                 # Evaluation tests
│   ├── benchmarks/
│   │   └── manual_probes.json      # Hand-crafted critical questions
│   └── reports/                    # Evaluation output (not committed)
├── scraper/                        # Forum scraper (01-04 scripts)
├── specs/                          # Architecture specs
│   └── training_eval_plan.md       # This file
└── Makefile                        # Root: delegates to data/, training/, eval/
```

---

## Phase 2: Training Infrastructure

### 2.1 Dataset

The manual pipeline outputs `data/src/manual/prepared/manual_train.jsonl` in this format:
```json
{
  "image": "images/21-03.jpg",
  "conversations": [
    {"role": "user", "content": "What is the torque specification for..."},
    {"role": "assistant", "content": "The torque specification is..."}
  ],
  "metadata": {...}
}
```

Dataset is uploaded to HuggingFace (`drumwell/vlm3`) via `data/src/manual/pipeline/09_upload_vlm.py`. Training loads from HuggingFace, not local files.

### 2.2 Modal Training App

`training/modal_train.py` implements LoRA fine-tuning on Modal:

- Loads Qwen2-VL-7B-Instruct with 4-bit quantization
- Applies LoRA adapters (rank 64, alpha 128)
- Loads dataset from HuggingFace Hub
- Trains with HF Trainer
- Saves adapter weights to Modal volume + optionally pushes to HuggingFace Hub

### 2.3 Training Configuration

**LoRA Config:** `training/configs/lora_qwen2vl.yaml`
```yaml
base_model: Qwen/Qwen2-VL-7B-Instruct
method: lora

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  lr_scheduler: cosine
  warmup_ratio: 0.03
  max_grad_norm: 1.0

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: bfloat16
  bnb_4bit_quant_type: nf4

eval:
  eval_steps: 100
  save_steps: 100
  logging_steps: 10
```

### 2.4 Estimated Training Costs

| Config | GPU | Time (est.) | Cost (Modal) |
|--------|-----|-------------|--------------|
| LoRA 7B, 3 epochs, ~11k examples | A100-80GB | 2-4 hours | $8-16 |
| LoRA 7B, 3 epochs, ~11k examples | H100 | 1-2 hours | $10-20 |
| Full fine-tune (not recommended) | 4xA100 | 8-12 hours | $100+ |

**Recommendation:** Start with LoRA on A100-80GB. Only consider full fine-tune if LoRA results are insufficient.

---

## Phase 3: Evaluation Framework

### 3.0 Baseline Eval (Run First!)

**Before fine-tuning**, establish baseline performance on Qwen2-VL-7B-Instruct:

```bash
# Create stratified eval sample from validation set
make eval-sample

# Run baseline eval on Modal (no local GPU needed)
make eval-modal-baseline

# Or run locally if you have a GPU
make eval-baseline
```

This provides a reference point to measure fine-tuning improvement.

### 3.1 Metrics Framework

The evaluation uses custom metrics implemented in `eval/metrics.py`. No external eval framework is required for the core automated metrics.

**Core automated metrics:**

| Metric | What it measures | Threshold | Class |
|--------|------------------|-----------|-------|
| `ROUGE-L` | General text quality (F1) | > 0.3 | `RougeL` |
| `NumericExactMatch` | Torque specs, measurements match within tolerance | > 0.9 | `NumericExactMatch` |
| `UnitConsistency` | Uses canonical units (Nm, mm, bar, L, C) | = 1.0 | `UnitConsistency` |
| `KeywordPresence` | Required technical terms present | > 0.7 | `KeywordPresence` |

**Domain-specific metrics:**

| Metric | What it measures | Threshold | Class |
|--------|------------------|-----------|-------|
| `SafetyCriticalAccuracy` | Higher bar for safety-related answers | > 0.9 | `SafetyCriticalAccuracy` |
| `VisualGrounding` | Answer uses image-specific details | > 0.3 | `VisualGrounding` |
| `ProcedureStepOrdering` | Procedural steps are numbered and sequential | > 0.7 | `ProcedureStepOrdering` |

**Optional LLM-as-judge metrics (via DeepEval):**

The `VLMEvaluator` class can optionally load DeepEval metrics (AnswerRelevancy, Faithfulness, GEval) using Claude as the judge model. These require `pip install deepeval` and are not used by the Modal eval pipeline.

### 3.2 Metric Implementation

```python
# eval/metrics.py — key classes

class VLMEvaluator:
    """Aggregated evaluator combining all metrics."""

    def __init__(self, use_llm_judge: bool = True):
        self.rouge_l = RougeL()
        self.numeric_match = NumericExactMatch()
        self.unit_consistency = UnitConsistency()
        self.keyword_presence = KeywordPresence()
        self.safety_accuracy = SafetyCriticalAccuracy()
        self.visual_grounding = VisualGrounding()
        self.procedure_ordering = ProcedureStepOrdering()

    def evaluate(self, prediction, reference, question, ...) -> dict[str, MetricResult]:
        """Run all applicable metrics on a single example."""
        ...

    def compute_aggregate_scores(self, results) -> dict:
        """Compute mean scores and pass rates across examples."""
        ...
```

`NumericExactMatch` extracts numbers with units and compares within tolerance (e.g., ±1 Nm for torque, ±0.05 mm for clearances). `UnitConsistency` flags non-canonical units (ft-lbs, psi, inches, gallons, Fahrenheit).

### 3.3 Eval Pipeline Flow

```
data/src/manual/prepared/manual_val.jsonl
       │
       ▼
┌──────────────────┐
│ sample_eval_set.py│  Stratified sampling by question_type
│ (make eval-sample)│  and content_type
└────────┬─────────┘
         │
         ▼
    eval/eval_sample.jsonl
         │
         ▼
┌──────────────────┐
│  modal_eval.py   │  (or run_eval.py for local GPU)
│  - Load model    │
│  - Load images   │
│    from HF Hub   │
│  - Generate      │
│    predictions   │
│  - Compute       │
│    metrics       │
└────────┬─────────┘
         │
         ▼
  eval/reports/baseline.json   (or finetuned.json)
         │
         ▼
┌──────────────────┐
│ compare_results.py│
│  - Score deltas  │
│  - By q-type     │
│  - By content    │
│  - Probe results │
└────────┬─────────┘
         │
         ▼
  eval/reports/comparison.md
```

### 3.4 Running Evals

```bash
# Create eval sample (stratified from val set)
make eval-sample

# Modal-based evaluation (recommended — no local GPU needed)
make eval-modal-baseline                        # Baseline
make eval-modal-finetuned ADAPTER_REPO=user/vlm3-lora  # Fine-tuned
make eval-modal-quick                           # Quick test (10 samples)

# Local evaluation (requires GPU)
make eval-baseline                              # Baseline
make eval-finetuned ADAPTER_PATH=/path/to/adapter  # Fine-tuned

# Compare results
make eval-compare

# Run manual probes
make eval-probes
make eval-modal-probes

# Test infrastructure without GPU
make eval-mock
```

### 3.5 Manual Benchmark Probes

Hand-crafted critical test cases in `eval/benchmarks/manual_probes.json`:

```json
[
    {
        "id": "torque_001",
        "category": "specifications",
        "image": "images/27-05.jpg",
        "question": "What is the torque specification for the cylinder head bolts?",
        "expected": "Stage 1: 40 Nm, Stage 2: 90 degrees, Stage 3: 90 degrees",
        "is_critical": true
    },
    {
        "id": "procedure_001",
        "category": "procedures",
        "image": "images/21-12.jpg",
        "question": "What are the steps to remove the clutch assembly?",
        "expected_keywords": ["transmission", "pressure plate", "alignment tool"]
    }
]
```

Probes include hallucination/out-of-distribution checks where the expected behavior is for the model to decline answering questions not shown in the image.

### 3.6 Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| ROUGE-L | > 0.3 | General text quality baseline |
| NumericExactMatch | > 0.9 | Critical for specifications |
| UnitConsistency | = 1.0 | Must use canonical metric units |
| KeywordPresence | > 0.7 | Required technical terms |
| Manual probe pass rate | > 85% | Hand-crafted critical questions |
| Critical probe pass rate | > 90% | Safety-critical subset |

### 3.7 Iteration Based on Results

**If thresholds not met:**
1. Check aggregate metric breakdown — which metric fails most?
2. Check `by_question_type` breakdown — which question types underperform?
3. Review training data for failing categories
4. Regenerate QA for weak sections → retrain → re-eval

**Adding complexity later:**
- Enable LLM-as-judge metrics via DeepEval for richer evaluation
- Hallucination detection for safety-critical info
- A/B comparison between model versions (supported by `compare_results.py`)

---

## Phase 4: Inference & Deployment

### 4.1 Modal Serving

```python
# training/modal_serve.py (pseudocode)

@app.cls(
    image=inference_image,
    gpu="A10G",  # Cheaper GPU for inference
    container_idle_timeout=300,
)
class VLM3Model:
    @modal.enter()
    def load_model(self):
        # Load base model + LoRA adapter
        pass

    @modal.method()
    def predict(self, image_path: str, question: str) -> str:
        # Run inference
        pass

    @modal.web_endpoint(method="POST")
    def api(self, request):
        # HTTP endpoint for external access
        pass
```

### 4.2 Local Testing

```python
# Simple local test script
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

base_model = Qwen2VLForConditionalGeneration.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, "path/to/adapter")

# Test inference
```

---

## Phase 5: Iteration Loop

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   Pipeline Output ──► Train ──► Eval ──► Analyze        │
│        ▲                                    │           │
│        │                                    │           │
│        └────── Regenerate weak sections ◄───┘           │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**Iteration triggers:**
- Eval scores below threshold
- Specific content types underperforming
- New source materials added to `data/src/`

---

## Implementation Checklist

### Data Pipeline ✅ COMPLETE
- [x] Service manual pipeline (Stages 01-09)
- [x] 11,154 train + 1,256 val examples + 1,408 images
- [x] Dataset uploaded to HuggingFace (`drumwell/vlm3`)
- [x] Pipeline tests (`data/src/manual/tests/`)
- [x] Config: `data/src/manual/config.yaml`

### Training ✅ INFRASTRUCTURE COMPLETE
- [x] Write `training/modal_train.py`
- [x] Training config: `training/configs/lora_qwen2vl.yaml`
- [x] Set up Modal secrets (HF token)
- [x] Test training on small subset (~100 examples)
- [x] Run full training
- [ ] Push final adapter to HuggingFace
- [ ] Verify adapter quality with eval

### Evaluation ✅ FRAMEWORK COMPLETE
- [x] Write `eval/metrics.py` (RougeL, NumericExactMatch, UnitConsistency, KeywordPresence, SafetyCriticalAccuracy, VisualGrounding, ProcedureStepOrdering)
- [x] Write `eval/run_eval.py` (local GPU evaluation runner)
- [x] Write `eval/modal_eval.py` (Modal cloud evaluation)
- [x] Write `eval/sample_eval_set.py` (stratified sampling)
- [x] Write `eval/compare_results.py` (comparison reports)
- [x] Write `eval/model_wrapper.py` (model loading abstraction)
- [x] Create `eval/benchmarks/manual_probes.json` (hand-crafted probes)
- [x] Write `eval/test_vlm.py` (evaluation tests)
- [ ] Run baseline eval: `make eval-modal-baseline`
- [ ] Run fine-tuned eval: `make eval-modal-finetuned ADAPTER_REPO=...`
- [ ] Compare baseline vs fine-tuned: `make eval-compare`
- [ ] Analyze failures by content type

### Deployment
- [ ] Write `training/modal_serve.py`
- [ ] Test inference endpoint
- [ ] Document API usage

### Iteration
- [ ] Review eval failures
- [ ] Identify weak areas
- [ ] Regenerate/improve data
- [ ] Retrain and re-eval

---

## Makefile Targets

From the root `Makefile`:

```makefile
# Training
make train                  # Full training on Modal (A100-80GB)
make train-dev              # Dev run (100 samples)
make train-resume           # Resume from checkpoint (detached)
make train-logs             # Check training logs from Modal volume

# Evaluation — create sample first
make eval-sample            # Stratified sample from val set (300 samples default)

# Evaluation — local GPU
make eval-baseline          # Base model
make eval-finetuned ADAPTER_PATH=/path  # Fine-tuned model
make eval-probes            # Manual probes
make eval-mock              # Test infra (no GPU)
make eval-test              # Pytest evaluation tests

# Evaluation — Modal cloud GPU (recommended)
make eval-modal-baseline    # Base model on Modal
make eval-modal-finetuned ADAPTER_REPO=user/repo  # Fine-tuned on Modal
make eval-modal-checkpoint  # Fine-tuned from Modal volume checkpoint
make eval-modal-quick       # Quick test (10 samples)
make eval-modal-probes      # Manual probes on Modal
make eval-modal-all         # Full pipeline: sample → baseline → train → eval → compare

# Comparison
make eval-compare           # Generate comparison report (baseline vs finetuned)
make eval-all               # Full local pipeline: sample → baseline → finetuned → compare → probes

# Cleanup
make clean-eval             # Remove eval artifacts
```

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| Dataset size | 11,154 train / 1,256 val (from `data/src/manual/prepared/`) |
| Validation split | 90/10 (configured in `data/src/manual/config.yaml`) |
| HuggingFace org | Personal account (`drumwell/vlm3`) |
| Experiment tracking | Skip W&B for simplicity (can add later) |
| Multi-image support | No — each QA pair references single image |
| Eval framework | Custom `VLMEvaluator` in `eval/metrics.py` (optional DeepEval LLM-as-judge) |

---

## Notes

- Training infrastructure is implemented and has been run on Modal
- Evaluation framework is implemented but not yet run against a fine-tuned model
- Modal account required (free tier has GPU credits)
- HuggingFace account required for dataset/model hosting
- **Baseline eval on Qwen2-VL-7B-Instruct should run BEFORE fine-tuning**
- Optional: `pip install deepeval` for LLM-as-judge metrics using Claude as evaluator
