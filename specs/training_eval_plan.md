# Training & Evaluation Reference

> Reference for training infrastructure, evaluation pipeline, and results.

## Overview

**Goal:** Fine-tune Qwen2-VL-7B on the BMW E30 M3 service manual Q&A dataset and evaluate its ability to answer technical questions from manual images.

**Stack:**
- Model: Qwen2-VL-7B-Instruct
- Training: Modal (GPU cloud) + HuggingFace Transformers + LoRA
- Eval: Custom `VLMEvaluator` (ROUGE-L, NumericExactMatch, UnitConsistency, KeywordPresence) + optional DeepEval LLM-as-judge

**Current state:**
- Dataset: 11,154 train / 1,256 val examples, 1,408 images (HuggingFace: `drumwell/vlm3`)
- Adapter: `drumwell/vlm3-lora` (on HuggingFace)
- Training: Complete (LoRA on A100-80GB, 3 epochs)
- Eval: Baseline + fine-tuned complete (334 samples) — ROUGE-L 0.507 → 0.759 (+49.5%)
- Manual probes: 22.5% pass rate (target 85%) — probes need regeneration (see `specs/manual_probes_fix_spec.md`)

**Infrastructure status:**
- [x] Data pipeline (Stages 01-09), dataset on HuggingFace
- [x] Training script (`training/modal_train.py`), config, Modal secrets
- [x] Full training run, adapter pushed to HuggingFace
- [x] Eval framework (`eval/metrics.py`, `eval/modal_eval.py`, `eval/compare_results.py`)
- [x] Baseline eval, fine-tuned eval, comparison report
- [x] Manual probes (`eval/benchmarks/manual_probes.json`) — 40 probes, but need regeneration
- [ ] Probe regeneration (image paths incorrect, questions not grounded in actual images)
- [ ] Serving endpoint (`modal_serve.py` not yet built)
- [ ] Forum data source (planned, `data/src/forum/` stub only)

---

## Training

### Dataset

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

### Modal Training App

`training/modal_train.py` implements LoRA fine-tuning on Modal:

- Loads Qwen2-VL-7B-Instruct with 4-bit quantization
- Applies LoRA adapters (rank 64, alpha 128)
- Loads dataset from HuggingFace Hub
- Trains with HF Trainer
- Saves adapter weights to Modal volume + optionally pushes to HuggingFace Hub

### Training Configuration

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

### Estimated Training Costs

| Config | GPU | Time (est.) | Cost (Modal) |
|--------|-----|-------------|--------------|
| LoRA 7B, 3 epochs, ~11k examples | A100-80GB | 2-4 hours | $8-16 |
| LoRA 7B, 3 epochs, ~11k examples | H100 | 1-2 hours | $10-20 |
| Full fine-tune (not recommended) | 4xA100 | 8-12 hours | $100+ |

**Recommendation:** Start with LoRA on A100-80GB. Only consider full fine-tune if LoRA results are insufficient.

---

## Evaluation

### Metrics Framework

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

`NumericExactMatch` extracts numbers with units and compares within tolerance (e.g., ±1 Nm for torque, ±0.05 mm for clearances). `UnitConsistency` flags non-canonical units (ft-lbs, psi, inches, gallons, Fahrenheit).

**Optional LLM-as-judge metrics (via DeepEval):**

The `VLMEvaluator` class can optionally load DeepEval metrics (AnswerRelevancy, Faithfulness, GEval) using Claude as the judge model. These require `pip install deepeval` and are not used by the Modal eval pipeline.

### Eval Pipeline Flow

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

### Running Evals

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

### Manual Benchmark Probes

Hand-crafted critical test cases in `eval/benchmarks/manual_probes.json` (40 probes):

```json
[
    {
        "id": "torque_001",
        "category": "specification",
        "question_type": "factual",
        "image": "images/data_src_00_-_Torque_Specs_BMW_Torque_Specs_046.jpg",
        "question": "What is the torque specification for the control pipe to bypass valve/turbocharger?",
        "expected": "30 Nm",
        "keywords": ["30", "Nm"],
        "is_critical": false,
        "notes": "Basic torque spec extraction from table"
    }
]
```

Probes include hallucination/out-of-distribution checks where the expected behavior is for the model to decline answering questions not shown in the image.

**Known issue:** Current probes were written without viewing actual images, resulting in mismatched questions and low pass rates. See `specs/manual_probes_fix_spec.md` for the regeneration plan using Claude API to view actual images and produce grounded probes.

### Success Criteria

| Metric | Threshold | Notes |
|--------|-----------|-------|
| ROUGE-L | > 0.3 | General text quality baseline |
| NumericExactMatch | > 0.9 | Critical for specifications |
| UnitConsistency | = 1.0 | Must use canonical metric units |
| KeywordPresence | > 0.7 | Required technical terms |
| Manual probe pass rate | > 85% | Hand-crafted critical questions |
| Critical probe pass rate | > 90% | Safety-critical subset |

---

## Results

Evaluation run on 334 stratified samples from the validation set. Adapter: `drumwell/vlm3-lora`.

### Overall Scores

| Metric | Baseline | Fine-tuned | Delta | Change |
|--------|----------|------------|-------|--------|
| rouge_l | 0.507 | 0.759 | +0.251 | **+49.5%** |
| numeric | 0.832 | 0.904 | +0.072 | +8.6% |
| unit | 0.995 | 0.997 | +0.002 | +0.2% |
| keyword | 1.000 | 1.000 | +0.000 | +0.0% |

All core thresholds met: ROUGE-L 0.759 > 0.3, NumericExactMatch 0.904 > 0.9, UnitConsistency 0.997 ≈ 1.0, KeywordPresence 1.0 > 0.7.

### By Question Type (ROUGE-L)

Sorted by improvement delta:

| Type | Baseline | Fine-tuned | Delta | n |
|------|----------|------------|-------|---|
| parameter | 0.365 | 0.792 | +0.427 | 4 |
| diagnostic | 0.346 | 0.678 | +0.332 | 16 |
| troubleshooting | 0.468 | 0.774 | +0.306 | 10 |
| procedural | 0.531 | 0.828 | +0.297 | 87 |
| factual | 0.480 | 0.738 | +0.258 | 75 |
| component | 0.539 | 0.779 | +0.241 | 21 |
| inspection | 0.496 | 0.737 | +0.241 | 10 |
| navigation | 0.535 | 0.768 | +0.233 | 21 |
| safety | 0.613 | 0.831 | +0.218 | 16 |
| tool | 0.596 | 0.809 | +0.213 | 10 |
| visual | 0.482 | 0.694 | +0.212 | 27 |
| wiring | 0.486 | 0.673 | +0.187 | 16 |
| connector | 0.489 | 0.593 | +0.104 | 11 |
| operation | 0.605 | 0.692 | +0.087 | 10 |

### Safety-Critical Analysis

| Metric | Baseline | Fine-tuned | Delta |
|--------|----------|------------|-------|
| rouge_l | 0.613 | 0.831 | +0.218 |
| numeric_match | 0.000 | 0.000 | +0.000 |
| safety_accuracy | 0.000 | 0.000 | +0.000 |

Safety samples: 16. ROUGE-L improved substantially, but `safety_accuracy` and `numeric_match` score 0.000 for both models — likely a metric implementation issue (safety questions may not contain extractable numeric values for `NumericExactMatch`, and `SafetyCriticalAccuracy` scoring may need review).

### Manual Probes

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| Pass Rate | 15.0% | 22.5% | +7.5% |
| Critical Pass Rate | 40.0% | 40.0% | +0.0% |

Total probes: 40 | Passed: 6 (baseline) → 9 (fine-tuned)

Probe pass rate is well below the 85% target. This is primarily due to probes being written without viewing actual images (mismatched questions, incorrect image paths). Critical pass rate did not improve. See `specs/manual_probes_fix_spec.md` for the fix plan.

### Known Issues

1. **Manual probe pass rate 22.5% vs 85% target** — Probes need regeneration with actual image viewing
2. **safety_accuracy 0.000 for both models** — Metric may not be triggering correctly on safety-typed questions; needs investigation
3. **connector (0.593) and operation (0.692)** — Weakest post-fine-tuning categories; may need more training examples in these areas
4. **Critical probe rate flat at 40%** — No improvement on safety-critical probes; blocked by probe quality issue

---

## Next Steps

### High Priority

1. **Regenerate manual probes** — Use Claude API to view actual images and produce grounded probes with correct image paths. Script and image list spec'd in `specs/manual_probes_fix_spec.md`.
2. **Investigate safety_accuracy metric** — Determine why it scores 0.000 for both models. Check whether safety-typed questions are being routed to the metric correctly and whether the scoring logic is appropriate.
3. **Improve weak categories** — connector (+0.104) and operation (+0.087) showed the smallest gains. Review training data for these question types and consider adding more examples.

### Medium Priority

4. **Re-evaluate after probe fix** — Once probes are regenerated, rerun `make eval-modal-finetuned ADAPTER_REPO=drumwell/vlm3-lora` and `make eval-compare` to get accurate probe pass rates.
5. **Add ETM training examples** — Wiring questions (+0.187) could benefit from more electrical manual content in the training set.
6. **Try DeepEval LLM-as-judge** — Enable the optional DeepEval metrics (AnswerRelevancy, Faithfulness) for richer evaluation on a subset.

### Lower Priority

7. **Forum data pipeline** — Build `data/src/forum/` pipeline to add community knowledge to the training set.
8. **Hyperparameter experiments** — Try different LoRA rank, learning rate, or epoch count if weak categories don't improve with data changes.
9. **Serving endpoint** — Build `training/modal_serve.py` for inference API when model quality is sufficient.

### Iteration Process

When improving a weak question type:
1. Check `by_question_type` breakdown to identify the weakest category
2. Review training data for that category (`grep` for `question_type` in `manual_train.jsonl`)
3. Identify whether the issue is data volume, data quality, or question difficulty
4. Regenerate or add Q&A pairs for weak sections → re-upload to HuggingFace
5. Retrain: `make train`
6. Re-eval: `make eval-modal-finetuned ADAPTER_REPO=drumwell/vlm3-lora && make eval-compare`
7. Compare `by_question_type` deltas to confirm improvement
