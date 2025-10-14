# QLoRA Finetuning Learning Experiments

## Experiment Log Template

For each model you train, record:
1. **Hypothesis**: What are you testing?
2. **Config**: Base model, LoRA rank, LR, epochs, batch size
3. **Training observations**: Loss curves, time taken
4. **Eval results**: Validation loss, sample outputs
5. **Learnings**: What worked? What didn't?

---

## Experiment 1: Baseline with Enhanced Dataset

### Hypothesis
Establish baseline performance with improved dataset (2,510 examples including HTML tech specs and enhanced block extraction).

### Config
- Base model: `meta-llama/Llama-3.1-8B-Instruct` (recommended)
- LoRA rank: 16
- Learning rate: 2e-4
- Epochs: 3
- Batch size: 4-8
- Dataset: 2,510 train, 248 synthetic val

### Results
⏳ **Ready for training** - Dataset prepared with:
- All 2,510 service manual examples
- HTML tech specs included
- Improved procedure extraction (6x increase)
- Synthetic validation set

### Key Improvements Since Initial Attempt
- ✅ All general specs now included (HTML extraction)
- ✅ Improved block extraction (+27.6% more data)
- ✅ No data split waste (all 2,510 examples for training)
- ✅ Synthetic validation (248 examples)

### Next Steps
- [ ] Train baseline model on AutoTrain
- [ ] Test with service manual queries
- [ ] Document performance metrics

---

## Experiment 2: Improve Procedure Extraction (✅ Complete)

### Hypothesis
Adding fallback logic for non-numbered procedures will significantly increase training data coverage.

### Implementation
Updated `scripts/04_parse_blocks.py` with fallback extraction logic:
- Detects action verbs (remove, install, check, adjust, etc.)
- Captures prose procedures without explicit numbering
- Splits by sentence/line breaks for implicit steps

### Config
- Relaxed extraction limits in `config.yaml`:
  - max_steps: 12 → 25
  - max_checks: 10 → 15
  - max_sentences: 4 → 6

### Results (✅ Complete)
- **Blocks**: 794 → 1,013 (+219, +27.6%)
- **Procedures**: 43 → 262 (+219, +6x increase)
- **Coverage**: Significantly improved procedure extraction

### Learnings
- ✅ Fallback logic dramatically improved data coverage
- ✅ Action verb detection works well for automotive procedures
- ✅ Relaxing limits allowed more complete procedures
- 📊 **Quality of extraction logic matters more than source quantity**

---

## Experiment 3: Compare Model Sizes (Planned)

### Hypothesis
Different model sizes may have different trade-offs for technical documentation tasks.

### Models to Compare
| Model | Params | VRAM | Best For |
|-------|--------|------|----------|
| Llama-3.2-3B-Instruct | 3B | ~8GB | Fast iteration, small GPU |
| Llama-3.1-8B-Instruct | 8B | ~16GB | Balanced quality/speed ⭐ |
| Mistral-7B-Instruct | 7B | ~16GB | Technical terminology |
| Qwen2.5-7B-Instruct | 7B | ~16GB | Multilingual support |

### Config (All Models)
- LoRA rank: 16
- Learning rate: 2e-4
- Epochs: 3
- Dataset: 2,510 train, 248 val

### Metrics to Compare
- [ ] Eval loss
- [ ] Spec extraction accuracy
- [ ] Procedure generation quality
- [ ] Training time
- [ ] Inference speed

### Learnings to Capture
- Is larger model worth the resource cost?
- Do technical terms require larger models?
- What's the sweet spot for this use case?

---

## Experiment 4: Hyperparameter Tuning (Planned)

### Hypothesis
Increasing LoRA rank to 32 will improve model's ability to learn technical patterns.

### Config
- Base model: Winner from Exp 2 vs 3
- **LoRA rank: 32** (was 16)
- Learning rate: 2e-4
- Epochs: 3
- Dataset: Same as Experiment 2

### What to Compare
| Config | Trainable Params | Eval Loss | Training Time | Quality |
|--------|------------------|-----------|---------------|---------|
| r=8 | ~4M | ? | Fastest | ? |
| r=16 (baseline) | ~8M | 1.4571 | 3 min | Known |
| r=32 | ~16M | ? | ~4 min | ? |
| r=64 | ~32M | ? | ~6 min | ? |

### Learnings to Capture
- Is there a point of diminishing returns?
- Does higher rank help with rare specs (e.g., compression ratio)?
- Does training become unstable at high ranks?

---

## Experiment 5: Data Quality vs Quantity (Planned)

### Hypothesis
Cleaning OCR errors will improve model quality more than adding more pages.

### Approach
1. Manually review 50 training examples for OCR errors
2. Fix common mistakes:
   - "0" vs "O" (zero vs letter O)
   - "1" vs "l" vs "I" (one vs L vs i)
   - Missing decimal points (45Nm → 45 Nm)
   - Garbled special tool numbers

### Config
- Same as best model so far
- **Dataset**: Cleaned version of 2,510 examples

### Learnings to Capture
- How many errors are in the current data?
- Does cleaning help more than adding 200 more examples?
- What types of errors hurt model most?

---

## Experiment 6: Task-Specific Models (Advanced)

### Hypothesis
Training separate models for SPEC vs PROCEDURE vs EXPLANATION might improve quality.

### Approach
- Train 3 separate models:
  - Model A: SPEC only
  - Model B: PROCEDURE only
  - Model C: EXPLANATION only

### Expected Trade-offs
- ✅ Each model deeply specialized
- ❌ Need to route queries to correct model
- ❌ 3x training cost

### Learnings to Capture
- Is multi-task learning helping or hurting?
- Do SPEC tasks benefit from seeing PROCEDURE examples?

---

## Metrics to Track Across All Experiments

### Quantitative
- [ ] Final training loss
- [ ] Validation loss
- [ ] Training time (wall-clock)
- [ ] Token accuracy (if available)
- [ ] Inference time per query

### Qualitative
- [ ] Sample outputs for 10 held-out queries
- [ ] Hallucination rate (answers not in manual)
- [ ] Format compliance (e.g., torque values with units)
- [ ] Procedure numbering (1., 2., 3. vs unformatted)

### User Experience
- [ ] Response quality for BMW community users
- [ ] Confidence in answers (does it sound certain when wrong?)
- [ ] Usefulness for real mechanic tasks

---

## AutoTrain Workflow

### 1. Prepare Data
```bash
# Your data is already in HF format! Just need to upload.
# Each line: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### 2. Upload to HuggingFace Hub
```python
from datasets import load_dataset
from huggingface_hub import login

login()  # Paste your HF token

dataset = load_dataset('json', data_files={
    'train': 'data/hf_train.jsonl',
    'validation': 'data/hf_val.jsonl'
})

dataset.push_to_hub("your-username/bmw-e30-service-manual")
```

### 3. Launch AutoTrain
- Go to https://huggingface.co/autotrain
- Click "New Project" → "LLM Fine-tuning"
- Select your dataset: `your-username/bmw-e30-service-manual`
- Choose base model: `meta-llama/Llama-3.2-3B-Instruct`
- AutoTrain will suggest LoRA config
- Click "Train" → Costs ~$5-10

### 4. Monitor Training
- Watch loss curves in real-time
- Training completes in ~10-30 minutes
- Model auto-uploaded to your HF account

### 5. Test Inference
```python
from transformers import pipeline

pipe = pipeline("text-generation", model="your-username/bmw-e30-service-manual-v1")
result = pipe("[SPEC] What is the engine displacement?")
print(result[0]['generated_text'])
```

### 6. Compare Models
- Keep a spreadsheet of all experiments
- Track which config produced best results
- Share best model with BMW community

---

## Quick Reference: What Each Parameter Does

| Parameter | What It Controls | Typical Range | When to Increase | When to Decrease |
|-----------|------------------|---------------|------------------|------------------|
| **LoRA Rank (r)** | # trainable params | 8-64 | Model underfitting | Overfitting, slow training |
| **Learning Rate** | Step size in training | 1e-5 to 5e-4 | Training too slow | Loss is jumpy/unstable |
| **Epochs** | Passes through data | 1-10 | Eval loss still decreasing | Eval loss increasing (overfit) |
| **Batch Size** | Examples per update | 4-32 | Stable training, faster | More generalization needed |
| **Alpha** | LoRA scaling factor | r to 2×r | Increase with rank | Rarely changed |

---

## Resources for Learning

### Understanding QLoRA
- Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- Key idea: Train 4-bit quantized model to save memory, keeps quality

### HuggingFace Docs
- AutoTrain: https://huggingface.co/docs/autotrain
- PEFT (LoRA library): https://huggingface.co/docs/peft
- TRL (training library): https://huggingface.co/docs/trl

### Loss Curves Interpretation
- Decreasing train + val loss: ✅ Learning
- Decreasing train, flat val: 🟡 Plateau (might need more data)
- Decreasing train, increasing val: ❌ Overfitting (reduce epochs)

---

## Next Steps

**Today (30 minutes):**
1. Upload your dataset to HuggingFace Hub
2. Start Experiment 2 (Baseline on AutoTrain)
3. While training, prepare index page specs

**This Week:**
1. Run Experiments 2-4 (different models, add data)
2. Build evaluation script for consistent testing
3. Document learnings in this file

**Next Week:**
1. Share best model with BMW community
2. Gather feedback on real-world queries
3. Iterate based on user needs

Would you like me to help you start with uploading your data to HuggingFace Hub? That's the first step to getting hands-on with AutoTrain.
