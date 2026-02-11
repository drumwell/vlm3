# Forum Data Curation Pipeline — Spec

> Lives at `data/src/forum/`. Follows conventions defined in `specs/data_architecture_spec.md`.

## Overview

**Purpose**: Curate scraped forum data from s14.net into high-quality text-based Q&A pairs for VLM fine-tuning. Forum data provides real-world troubleshooting knowledge, community-discovered techniques, and aftermarket parts experience that complement the authoritative service manual dataset.

**Data source**: `raw/` — 580,839 posts across 40 forums, scraped from s14net.vbulletin.net (BMW E30 M3 community).

**Key constraint**: Forum data is text-only. Unlike the manual source (image → Q&A), forum Q&A pairs have no associated images. The training script must handle records with and without the `image` field.

---

## Scope: Phased Approach

### Phase 1 — Focused Technical Forums (this spec)

Target the high-signal forums where posts are topically constrained and signal density is highest:

| Forum | Posts | Threads | Rationale |
|-------|-------|---------|-----------|
| `d-i-y` | 33 | 3 | Step-by-step procedures with images |
| `no-start` | 511 | 53 | Diagnostic troubleshooting conversations |
| `ecu-chips` | 777 | 109 | ECU tuning, aftermarket knowledge |
| `forced-induction` | 2,114 | 190 | Turbo/supercharger builds and tuning |
| `oils-fluids` | 612 | 68 | Maintenance knowledge |
| `spark-plugs` | ~200 | ~30 | Maintenance knowledge |
| `water-leaks` | ~300 | ~40 | Diagnostic troubleshooting |
| `batteries` | ~200 | ~30 | Electrical troubleshooting |
| `alpha-n-carbon-fiber` | 3,913 | ~400 | S14 Alpha-N conversion (deep technical) |
| `engine-swap-cars` | 9,711 | 904 | Engine swap procedures and knowledge |

**Estimated input**: ~18,000 posts across ~1,800 threads.

### Phase 2 — General Discussion Mining (future)

Apply LLM-as-judge classification to the 238K-post `general-e30-m3-discussion` forum. Only proceed if Phase 1 demonstrates the curation pipeline produces quality training data that improves eval scores.

---

## Pipeline Stages

All paths below are relative to `data/src/forum/`.

```
raw/posts_*.jsonl
    ↓
01  Reconstruct Threads     → work/threads/*.json
    ↓
02  Mechanical Filtering    → work/threads_clean/*.json
    ↓
03  Thread Classification   → work/threads_classified/*.json     [Claude API]
    ↓
04  Quality Scoring         → work/threads_scored/*.json         [Claude API]
    ↓
05  Q&A Extraction          → work/qa_raw/*.json                 [Claude API]
    ↓
06  Cross-Reference         → work/qa_validated/*.json
    ↓
07  Emit                    → prepared/forum_train.jsonl + forum_val.jsonl
```

---

## Stage 01: Reconstruct Threads

**Script**: `pipeline/01_reconstruct.py`

**Purpose**: Group individual posts into complete thread conversations, ordered chronologically.

**Input**: `raw/posts_*.jsonl` (one file per forum)

**Output**: `work/threads/*.json` (one file per thread)

**Logic**:
1. Read all posts from target forum JSONL files
2. Group by `thread_id`
3. Sort posts within each thread by `post_number`
4. Reconstruct quoted reply chains using `quotes` field
5. Extract thread metadata from the first post (`is_first_post: true`)

**Output schema**:
```json
{
  "thread_id": "1258841",
  "thread_title": "Valve Adjustment",
  "forum": "d-i-y",
  "author": "Split_S",
  "post_count": 12,
  "first_post_date": "2017-05-29",
  "last_post_date": "2017-06-15",
  "posts": [
    {
      "post_id": "1258841",
      "author": "Split_S",
      "post_number": 1,
      "is_first_post": true,
      "content_text": "...",
      "content_images": ["https://..."],
      "quotes": []
    }
  ]
}
```

**Image handling**: Strip forum signature images (`member_sponsors/` URLs). Retain content-embedded images but flag them as `content_images` — these are URLs only, not downloaded files. Image availability is not guaranteed (many are dead photobucket/postimg links).

---

## Stage 02: Mechanical Filtering

**Script**: `pipeline/02_filter.py`

**Purpose**: Remove noise at the post and thread level using deterministic rules. No API calls.

**Input**: `work/threads/*.json`

**Output**: `work/threads_clean/*.json` + `work/logs/filter_report.md`

**Post-level filters** (remove individual posts from threads):
- `content_text` length < 30 characters after stripping
- Pure social replies: exact or fuzzy match against patterns:
  - `["thanks", "bump", "+1", "subscribed", "following", "nice", "great post", "well done", "awesome", "lol", "haha"]`
- Posts that are entirely quoted content with no original text
- Posts containing only URLs with no explanatory text
- Posts containing only images with no text

**Thread-level filters** (remove entire threads):
- Threads with < 2 remaining posts after post filtering
- Threads with first post length < 50 characters (not a real question)
- Thread titles matching noise patterns:
  - `["WTB", "WTS", "FS:", "for sale", "wanted", "price check", "how much", "shipping"]`

**Content cleaning** (applied to surviving posts):
- Strip leading `#N` post number markers (e.g., "#1", "#16")
- Strip timestamp lines (e.g., "06-20-2006, 12:17 AM")
- Strip forum signature content (text after signature dividers)
- Normalize whitespace (collapse multiple newlines, strip trailing spaces)
- Strip HTML artifacts that leaked into `content_text`

**Report**: Log counts at each filter stage — posts removed per filter, threads removed, survival rates by forum.

---

## Stage 03: Thread Classification

**Script**: `pipeline/03_classify.py`

**Purpose**: Classify each thread by type and technical relevance using Claude API.

**Input**: `work/threads_clean/*.json`

**Output**: `work/threads_classified/*.json`

**Classification prompt**: Send the thread title + first post + first 2-3 replies (truncated to ~2000 chars total) to Claude Haiku for fast, cheap classification.

**Classification schema**:
```json
{
  "thread_type": "troubleshooting | how_to | modification | specification | parts_discussion | experience_report | general_question | off_topic",
  "technical_depth": "high | medium | low",
  "s14_specific": true,
  "has_resolution": true,
  "confidence": 0.85
}
```

**Thread type definitions**:
- `troubleshooting` — Problem described, diagnostic steps, resolution attempted
- `how_to` — Step-by-step procedure or guide (like the valve adjustment DIY)
- `modification` — Aftermarket parts, performance mods, custom work
- `specification` — Discussion of specific measurements, torque values, clearances
- `parts_discussion` — Which parts to use, part number references, supplier recommendations
- `experience_report` — First-hand experience with a repair or modification
- `general_question` — Technical question without diagnostic framing
- `off_topic` — Not relevant to E30 M3 maintenance or modification

**Filter gate**: Only threads with `technical_depth` >= `medium` AND `s14_specific` == `true` proceed to Stage 04.

**Cost control**: Use Claude Haiku. Estimated ~1,800 threads × ~1K tokens/thread = ~1.8M input tokens. Negligible cost.

---

## Stage 04: Quality Scoring

**Script**: `pipeline/04_score.py`

**Purpose**: Score qualifying threads on dimensions that predict training value.

**Input**: `work/threads_classified/*.json` (classified threads that passed the gate)

**Output**: `work/threads_scored/*.json`

**Scoring prompt**: Send full thread content (truncated to ~4000 chars) to Claude Sonnet with structured scoring rubric.

**Scoring dimensions** (each 1-5):

| Dimension | Description | Weight |
|-----------|-------------|--------|
| `factual_specificity` | Contains specific numbers, part numbers, measurements, tool names | 0.30 |
| `answer_completeness` | Thread reaches a clear resolution or provides a complete answer | 0.25 |
| `procedural_clarity` | Steps are described clearly enough to follow | 0.20 |
| `s14_relevance` | Information is specific to E30 M3 / S14 engine, not generic BMW | 0.15 |
| `unique_knowledge` | Contains knowledge unlikely to be in the factory service manual | 0.10 |

**Composite score**: Weighted average, 1.0-5.0 scale.

**Filter gate**: Threads scoring >= 3.0 proceed to Q&A extraction.

**Cost control**: Use Claude Sonnet. Estimated ~500-800 qualifying threads × ~2K tokens = ~1.5M input tokens. Low cost.

---

## Stage 05: Q&A Extraction

**Script**: `pipeline/05_extract_qa.py`

**Purpose**: Extract structured Q&A pairs from qualifying threads using Claude API.

**Input**: `work/threads_scored/*.json` (threads that scored >= 3.0)

**Output**: `work/qa_raw/*.json` (one file per thread)

**Extraction prompt**: Send the full thread to Claude Sonnet with instructions to:

1. Identify the core question(s) being asked in the thread
2. Synthesize the best answer from all replies (not just copy one reply)
3. Resolve contradictions by favoring:
   - Posts with specific measurements/part numbers over vague claims
   - Posts corroborated by multiple authors over single opinions
   - Posts from authors who describe first-hand experience over speculation
4. Generate 1-5 Q&A pairs per thread depending on content richness
5. Flag any factual claims that should be cross-referenced against the service manual

**Output schema**:
```json
{
  "thread_id": "1258841",
  "forum": "d-i-y",
  "thread_title": "Valve Adjustment",
  "source_type": "forum",
  "content_type": "how_to",
  "quality_score": 4.2,

  "qa_pairs": [
    {
      "id": "forum-1258841-q01",
      "question": "What is the valve shim clearance specification for the BMW S14 engine?",
      "answer": "The BMW S14 valve shim clearance specification is 0.28mm to 0.33mm, measured cold. Shims come in gradations of 0.05mm. To calculate the required shim thickness: add the measured valve gap to the measured shim thickness, then subtract the desired gap.",
      "question_type": "specification",
      "cross_reference_needed": true,
      "cross_reference_note": "Verify 0.28-0.33mm spec against service manual Section 11"
    }
  ],

  "extraction": {
    "model": "claude-sonnet-4-20250514",
    "timestamp": "2025-02-10T10:00:00Z",
    "thread_posts_used": 12,
    "thread_posts_total": 15
  }
}
```

**Key difference from manual pipeline**: Q&A answers are *synthesized* from multiple community replies, not extracted from a single authoritative source. The `source_type: "forum"` field distinguishes these from manual-derived pairs.

---

## Stage 06: Cross-Reference & Filter

**Script**: `pipeline/06_validate.py`

**Purpose**: Validate forum Q&A pairs against the manual dataset and apply quality filters.

**Input**:
- `work/qa_raw/*.json` — Forum Q&A pairs
- `../manual/prepared/manual_train.jsonl` — Manual-derived Q&A pairs (ground truth reference)

**Output**:
- `work/qa_validated/*.json` — Validated Q&A pairs
- `work/logs/crossref_report.md` — Cross-reference findings

**Cross-reference checks**:
1. **Numeric validation**: Extract numbers with units (Nm, mm, bar, °C, etc.) from forum answers. Compare against manual Q&A pairs that reference the same section/topic. Flag contradictions.
2. **Semantic overlap detection**: Use sentence-transformers (same as manual dedup pipeline) to find forum Q&A pairs that are semantically similar to existing manual pairs. If similarity > 0.85, the forum pair is redundant — drop it unless the forum answer adds information the manual answer doesn't.
3. **Factual confidence tagging**: Tag each Q&A pair with a confidence level:
   - `verified` — Matches or is consistent with manual data
   - `plausible` — No contradiction found, but not directly verifiable
   - `unverified` — Contains claims not present in manual data (e.g., aftermarket parts)
   - `contradicted` — Contradicts manual data — **exclude from training set**

**Quality filters**:
- Answer length < 50 characters → reject
- Generic answers ("it depends", "check the manual") → reject
- Questions that are too vague ("what do you think?") → reject
- Duplicate Q&A pairs within forum dataset → deduplicate

**Note**: This is the one stage where the forum pipeline reads from another source's `prepared/` output. This is a deliberate cross-reference, not a coupling — the forum pipeline could run without it (skipping numeric validation), but the quality is better with it.

---

## Stage 07: Emit

**Script**: `pipeline/07_emit.py`

**Purpose**: Emit forum Q&A pairs in the prepared output format defined by the data architecture contract.

**Input**: `work/qa_validated/*.json`

**Output**:
- `prepared/forum_train.jsonl`
- `prepared/forum_val.jsonl`

**Output schema** (per data architecture contract):
```json
{
  "conversations": [
    {"role": "user", "content": "What is the valve shim clearance specification for the BMW S14 engine?"},
    {"role": "assistant", "content": "The BMW S14 valve shim clearance specification is 0.28mm to 0.33mm..."}
  ],
  "metadata": {
    "source": "forum",
    "source_id": "forum-1258841-q01",
    "thread_id": "1258841",
    "forum": "d-i-y",
    "content_type": "how_to",
    "question_type": "specification",
    "quality_score": 4.2,
    "factual_confidence": "verified"
  }
}
```

No `"image"` field — text-only source.

**Split**: 90/10 train/val ratio, same as manual source.

---

## Estimated Yield

| Stage | Input | Output | Survival Rate |
|-------|-------|--------|---------------|
| Reconstruct | ~18,000 posts | ~1,800 threads | — |
| Mechanical filter | ~1,800 threads | ~1,200 threads | 67% |
| Classification | ~1,200 threads | ~600 threads | 50% |
| Quality scoring | ~600 threads | ~300 threads | 50% |
| Q&A extraction | ~300 threads | ~600-1,000 Q&A pairs | 2-3 per thread |
| Cross-ref & filter | ~600-1,000 pairs | ~400-700 pairs | 70% |
| Final output | | **~400-700 forum Q&A pairs** | |

**Expected contribution**: 400-700 text-only Q&A pairs added to the existing ~12,400 manual pairs. Roughly a 5% increase in dataset size, but targeting a different knowledge domain (real-world troubleshooting, aftermarket, community techniques).

---

## Cost Estimate

| Stage | API | Model | Est. Tokens | Est. Cost |
|-------|-----|-------|-------------|-----------|
| 03 Classification | Claude | Haiku | ~1.8M input | < $1 |
| 04 Scoring | Claude | Sonnet | ~1.5M input | ~$5 |
| 05 Extraction | Claude | Sonnet | ~3M input | ~$10 |
| **Total** | | | | **~$16** |

Stages 01, 02, 06, 07 are local processing — no API cost.

---

## Dependencies on Training Script

The training script (`training/modal_train.py`) currently assumes every training example has an `"image"` field. To support mixed datasets:

1. `VLMDataset.__getitem__` must handle records without `"image"` — format as text-only conversation
2. `collate_fn` must handle batches where some items have `pixel_values`/`image_grid_thw` and some don't
3. The text-only examples train the language model weights without exercising the vision encoder — this is expected and beneficial for knowledge acquisition

This is a separate change from the label masking fix, but both should be implemented before the first merged training run with forum data.

---

## Success Criteria

1. Forum Q&A pairs pass the same quality filters as manual Q&A pairs (answer length, specificity, deduplication)
2. No forum Q&A pair contradicts verified service manual data
3. Model trained on merged dataset scores >= model trained on manual-only dataset on all eval metrics
4. Model trained on merged dataset shows improvement specifically on troubleshooting-type eval questions
5. Forum data does not degrade performance on specification-type questions (torque values, clearances)
