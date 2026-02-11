# Manual Probes Regeneration Specification

## Problem Statement

The `eval/benchmarks/manual_probes.json` contains 40 hand-crafted probes with incorrect image paths. These were written without viewing actual images, resulting in mismatched questions and 404 errors during evaluation.

## Benchmark Context

From the baseline vs fine-tuned comparison (334 eval samples):

| Metric | Baseline | Fine-tuned | Change |
|--------|----------|------------|--------|
| ROUGE-L | 0.507 | 0.759 | **+49.5%** |
| Numeric Match | 0.832 | 0.904 | +8.6% |
| Manual Probe Pass Rate | 15% | 22.5% | +7.5% |

**Top improving question types (ROUGE-L):**
| Type | Baseline | Fine-tuned | Delta |
|------|----------|------------|-------|
| parameter | 0.365 | 0.792 | +0.427 |
| diagnostic | 0.346 | 0.678 | +0.332 |
| troubleshooting | 0.468 | 0.774 | +0.306 |
| procedural | 0.531 | 0.828 | +0.297 |
| safety | 0.613 | 0.831 | +0.218 |

**Weaker areas:**
| Type | Baseline | Fine-tuned | Delta |
|------|----------|------------|-------|
| connector | 0.489 | 0.593 | +0.104 |
| operation | 0.605 | 0.692 | +0.087 |

## Approach: Claude API Regeneration

Use Claude API to view actual images and generate validated probe questions.

### Image Selection Strategy

Select ~15-20 representative images covering:

1. **High-performing categories** (validate gains):
   - Procedural pages (clutch, brakes, transmission)
   - Diagnostic/troubleshooting content
   - Safety warnings and cautions

2. **Weaker categories** (probe for gaps):
   - Connector/wiring diagrams
   - Operational procedures

3. **Special test categories**:
   - Torque spec tables (numeric extraction)
   - Wiring diagrams (ETM)
   - Component photos (Getrag rebuild)

### Probe Categories

Generate 3-4 probes per image across these categories:

| Category | Purpose | Target Count |
|----------|---------|--------------|
| `specification` | Extract numeric specs, torques, measurements | 8 |
| `procedural` | Step sequences, prerequisites, tool requirements | 8 |
| `diagnostic` | Symptom-cause reasoning, troubleshooting | 6 |
| `safety` | Identify warnings, critical precautions | 6 |
| `component` | Part identification, materials, types | 4 |
| `wiring` | Read electrical diagrams, trace circuits | 4 |
| `visual_grounding` | Questions only answerable from image | 6 |
| `hallucination_detection` | Questions about content NOT in image | 6 |
| `out_of_distribution` | Wrong vehicle/scope questions | 4 |

**Total: ~52 probes**

### Generation Script

Create `eval/scripts/generate_probes.py`:

```python
"""
Generate manual probes using Claude API.

For each selected image:
1. Send image to Claude with generation prompt
2. Request probes in specific categories
3. Validate image paths exist
4. Output structured JSON
"""

GENERATION_PROMPT = """
You are creating evaluation probes for a BMW E30 M3 service manual VLM.

Look at this image and generate test questions in these categories:

1. FACTUAL: Questions with specific answers visible in the image
   - Include expected answer and keywords

2. PROCEDURAL: Questions about steps, sequences, prerequisites
   - What should be done before/after X?

3. SAFETY (if applicable): Questions about warnings, cautions
   - Mark as is_critical: true

4. VISUAL_GROUNDING: Questions that can ONLY be answered by looking at this specific image
   - E.g., "What tool is shown in the left photo?"

5. HALLUCINATION_DETECTION: Questions about content NOT in this image
   - The model should decline or caveat these
   - E.g., asking about oil specs on a brake diagram

Output JSON array with structure:
{
  "id": "category_NNN",
  "category": "specification|procedural|safety|...",
  "question": "...",
  "expected": "exact answer if known",
  "expected_keywords": ["key", "words"],
  "is_critical": true/false,
  "notes": "why this probe matters"
}
"""
```

### Implementation Steps

1. **Select images** - Curate list of ~15-20 images from:
   ```
   data/src/manual/prepared/images/
   ```

2. **Run generation script**:
   ```bash
   python eval/scripts/generate_probes.py \
     --images eval/probe_images.txt \
     --output eval/benchmarks/manual_probes.json
   ```

3. **Validate output**:
   - All image paths exist
   - Each probe has required fields
   - Mix of categories achieved

4. **Run evaluation**:
   ```bash
   make eval-modal-finetuned ADAPTER_REPO=drumwell/vlm3-lora
   ```

### Image Selection List

Create `eval/probe_images.txt` with paths to:

```
# Torque specs (numeric extraction)
images/raw_00_-_Torque_Specs_BMW_Torque_Specs_044.jpg
images/raw_00_-_Torque_Specs_BMW_Torque_Specs_046.jpg

# Clutch procedures
images/raw_21_-_Clutch_21-01.jpg
images/raw_21_-_Clutch_21-03.jpg

# Brake procedures
images/raw_34_-_Brakes_34-01.jpg
images/raw_34_-_Brakes_34-05.jpg

# Transmission rebuild (visual grounding)
images/raw_Getrag265_5_Rebuild_ShiftBMW_003.jpg
images/raw_Getrag265_5_Rebuild_ShiftBMW_018.jpg
images/raw_Getrag265_5_Rebuild_ShiftBMW_023.jpg

# Wiring diagrams
images/raw_1990_BMW_M3_Electrical_Troubleshooting_Manual_0670-01.jpg
images/raw_1990_BMW_M3_Electrical_Troubleshooting_Manual_0670-04.jpg

# Maintenance/safety
images/raw_00_-_Maintenance_00-00-index-a.jpg
images/raw_00_-_Maintenance_00-01.jpg

# Engine electrical
images/raw_12_-_Engine_Electrical_Equipment_12-01.jpg
```

### Probe Quality Criteria

Each probe must:
- Reference an existing image path
- Have a question answerable (or deliberately unanswerable) from that image
- Include `expected` or `expected_keywords` for scoring
- Be tagged with appropriate `is_critical` flag
- Have `notes` explaining the test purpose

### Success Metrics

After regeneration:
- 0 image 404 errors during evaluation
- Probe pass rate baseline established for future comparison
- Coverage across all question types from benchmark
- At least 6 hallucination/OOD probes to test model boundaries

## Files to Create/Modify

| File | Action |
|------|--------|
| `eval/scripts/generate_probes.py` | Create - Claude API probe generator |
| `eval/probe_images.txt` | Create - Curated image list |
| `eval/benchmarks/manual_probes.json` | Regenerate - Valid probes |

## Verification

```bash
# 1. Generate probes
python eval/scripts/generate_probes.py

# 2. Validate all images exist
python -c "
import json
from pathlib import Path
with open('eval/benchmarks/manual_probes.json') as f:
    probes = json.load(f)
missing = [p['image'] for p in probes
           if not (Path('data/src/manual/prepared') / p['image']).exists()]
print(f'Missing: {len(missing)}')
for m in missing: print(f'  {m}')
"

# 3. Run evaluation
make eval-modal-finetuned ADAPTER_REPO=drumwell/vlm3-lora

# 4. Check probe results
python -c "
import json
with open('eval/reports/finetuned.json') as f:
    r = json.load(f)
ps = r.get('probe_summary', {})
print(f'Pass rate: {ps.get(\"passed\")}/{ps.get(\"total\")} ({ps.get(\"pass_rate\"):.1%})')
print(f'Critical: {ps.get(\"critical_passed\")}/{ps.get(\"critical_total\")}')
"
```
