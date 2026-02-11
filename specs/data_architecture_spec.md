# Data Architecture Spec

## Problem

The project currently scatters data concerns across the repo root: `data_src/`, `pipeline/`, `work/`, `training_data/`. This was fine when there was one data source (the service manual), but it doesn't scale. Adding the forum pipeline exposed the issue — naming and placement got awkward because the structure assumed a single pipeline.

## Principle

Every data source is a self-contained unit: raw input, processing pipeline, and prepared output all live together. Sources know nothing about each other. A separate merge layer reads from each source's prepared output and produces the final training set.

Adding a new data source is: create a directory, follow the convention, and register it in the merge config.

---

## Target Structure

```
vlm3/
├── data/
│   ├── src/
│   │   ├── manual/                     # Service manual data source
│   │   │   ├── Makefile                # Source-level targets
│   │   │   ├── config.yaml             # Pipeline configuration
│   │   │   ├── raw/                    # Scanned manual pages (the section folders)
│   │   │   ├── pipeline/               # 01_inventory.py … 09_upload.py
│   │   │   ├── work/                   # Intermediate outputs (classified/, qa_raw/, etc.)
│   │   │   ├── prepared/               # vlm_train.jsonl, vlm_val.jsonl, images/
│   │   │   └── tests/
│   │   │
│   │   ├── forum/                      # Forum data source
│   │   │   ├── Makefile
│   │   │   ├── config.yaml
│   │   │   ├── raw/                    # Scraped JSONL files (posts_*.jsonl, threads_*.jsonl)
│   │   │   ├── pipeline/               # 01_reconstruct.py … 07_emit.py
│   │   │   ├── work/                   # Intermediate outputs
│   │   │   ├── prepared/               # forum_train.jsonl, forum_val.jsonl
│   │   │   └── tests/
│   │   │
│   │   └── ...                         # Future sources follow same convention
│   │
│   ├── training/                       # Merged training set (consumed by training scripts)
│   │   ├── merge.py                    # Reads from all src/*/prepared/, emits merged set
│   │   ├── config.yaml                 # Source weights, shuffle seed, upsampling
│   │   ├── merged_train.jsonl          # Output
│   │   └── merged_val.jsonl            # Output
│   │
│   └── Makefile                        # Orchestrates: all sources + merge
│
├── training/                           # Model training infrastructure (unchanged)
│   ├── modal_train.py
│   └── ...
│
├── eval/                               # Evaluation framework (unchanged)
│   ├── modal_eval.py
│   └── ...
│
├── scraper/                            # Data acquisition tools (unchanged)
│
├── specs/                              # Project-level specifications
│   ├── data_architecture_spec.md       # This document
│   ├── forum_pipeline_spec.md          # Forum pipeline details
│   ├── training_eval_plan.md
│   └── 01-06_*.md                      # Historical manual pipeline specs
│
├── Makefile                            # Root: delegates to data/, training/, eval/
├── CLAUDE.md
└── README.md
```

---

## Convention: Data Source Directory

Every directory under `data/src/` follows this layout:

```
data/src/<name>/
├── Makefile            # All targets for this source: all, status, clean
├── config.yaml         # Source-specific configuration
├── raw/                # Immutable input data. Never modified by the pipeline.
├── pipeline/           # Numbered scripts: 01_*.py, 02_*.py, ...
├── work/               # Intermediate outputs. Safe to delete and regenerate.
├── prepared/           # Final output. JSONL + optional images.
│   ├── <name>_train.jsonl
│   ├── <name>_val.jsonl
│   └── images/         # (optional, only if source has visual data)
└── tests/              # Pipeline tests
```

**Rules**:
- Scripts are numbered `01` through `NN`, independent per source.
- `raw/` is read-only from the pipeline's perspective. It's populated by the scraper or by manual placement.
- `prepared/` is the contract surface between a source and the merge layer. The merge layer reads only from `prepared/` directories.
- `work/` contains everything between `raw/` and `prepared/`. It can be blown away at any time.
- Each source has its own `Makefile` with at minimum: `all`, `status`, `clean` targets.
- The `config.yaml` is source-specific. No global pipeline config.
- **Cross-source reads**: A pipeline may optionally read another source's `prepared/` output for validation (e.g., the forum pipeline cross-references extracted Q&A against the manual's prepared output to flag contradictions). This is a read-only dependency — the pipeline should still be able to run without it (degraded quality, not failure).

**Prepared output format** (contract):
```json
{
  "conversations": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "image": "images/section_11_page_042.png",
  "metadata": {
    "source": "manual",
    "source_id": "section_11_page_042_q01",
    ...
  }
}
```

The `image` field is optional. Text-only sources omit it. Two metadata fields are required: `source` (identifies the data source by name) and `source_id` (unique identifier for the Q&A pair within that source). Everything else in metadata is source-specific — sources should include whatever provenance and quality signals are useful for downstream analysis (e.g., `quality_score`, `factual_confidence`, `content_type`).

---

## Merge Layer: `data/training/`

The merge layer is not a data source — it's a thin aggregation step that reads from all `data/src/*/prepared/` directories and produces the unified training set.

**`data/training/config.yaml`**:
```yaml
sources:
  manual:
    path: ../src/manual/prepared
    weight: 0.8
  forum:
    path: ../src/forum/prepared
    weight: 0.2

upsample_to_balance: true
shuffle_seed: 42
```

**`data/training/merge.py`**:
- Reads `config.yaml` to discover sources and weights
- Loads `<source>_train.jsonl` and `<source>_val.jsonl` from each source's prepared directory
- Applies weighting (upsample smaller sources or downsample larger ones)
- Shuffles deterministically
- Emits `merged_train.jsonl` and `merged_val.jsonl`
- Logs composition: count by source, content type, confidence level

**Adding a new source**: Add an entry to `config.yaml`, point it at the new source's prepared directory, assign a weight. No code changes to `merge.py`.

---

## Makefile Hierarchy

Three levels of Makefiles, each delegating downward:

### Root `Makefile` (project root)

Orchestrates everything. Delegates to `data/`, `training/`, and `eval/`.

```makefile
# Top-level targets
data:
	$(MAKE) -C data all

data-status:
	$(MAKE) -C data status

train:
	modal run training/modal_train.py::main ...

eval:
	modal run eval/modal_eval.py ...
```

### `data/Makefile` (data orchestrator)

Fans out to all sources, then merges.

```makefile
SOURCES := manual forum

# Run all source pipelines (in parallel) then merge
all:
	@for src in $(SOURCES); do $(MAKE) -C src/$$src all & done; wait
	$(MAKE) merge

# Run a specific source
manual:
	$(MAKE) -C src/manual all

forum:
	$(MAKE) -C src/forum all

# Merge all prepared outputs
merge:
	python training/merge.py --config training/config.yaml

# Status across all sources
status:
	@for src in $(SOURCES); do echo "=== $$src ===" && $(MAKE) -C src/$$src status && echo; done
	@test -f training/merged_train.jsonl && echo "=== merged ===" && echo "$$(wc -l < training/merged_train.jsonl) train examples" || echo "=== merged === not built"

# Clean everything
clean:
	@for src in $(SOURCES); do $(MAKE) -C src/$$src clean; done
	rm -f training/merged_train.jsonl training/merged_val.jsonl
```

### `data/src/<name>/Makefile` (per-source)

Each source pipeline is fully self-contained.

**Manual example** (`data/src/manual/Makefile`):
```makefile
all: inventory prepare classify generate-qa filter deduplicate emit validate
	@echo "Manual pipeline complete. Output in prepared/"

inventory:
	python pipeline/01_inventory.py --data-src raw/ --output work/inventory.csv

# ... remaining targets mirror existing Makefile ...

status:
	@echo "Manual Pipeline Status"
	@test -f work/inventory.csv && echo "  01 inventory:  done" || echo "  01 inventory:  pending"
	# ...
	@test -f prepared/manual_train.jsonl && echo "  PREPARED:      $$(wc -l < prepared/manual_train.jsonl) examples" || echo "  PREPARED:      pending"

clean:
	rm -rf work/
```

**Forum example** (`data/src/forum/Makefile`):
```makefile
FORUMS := d-i-y no-start ecu-chips forced-induction oils-fluids \
          spark-plugs water-leaks batteries alpha-n-carbon-fiber engine-swap-cars

all: reconstruct filter classify score extract validate emit
	@echo "Forum pipeline complete. Output in prepared/"

reconstruct:
	python pipeline/01_reconstruct.py --forums $(FORUMS) --raw raw/ --output work/threads/

filter:
	python pipeline/02_filter.py --input work/threads/ --output work/threads_clean/

# ... etc ...

status:
	@echo "Forum Pipeline Status"
	# ...

clean:
	rm -rf work/
```

---

## Migration Plan

Map from current location to new location:

| Current | New | Notes |
|---------|-----|-------|
| `data_src/e30-m3-320is/` | `data/src/manual/raw/` | Move the section folders |
| `pipeline/scripts/01-09` | `data/src/manual/pipeline/` | Move scripts |
| `pipeline/config.yaml` | `data/src/manual/config.yaml` | Move config |
| `pipeline/tests/` | `data/src/manual/tests/` | Move tests |
| `work/` (manual intermediates) | `data/src/manual/work/` | Move or regenerate |
| `training_data/vlm_train.jsonl` | `data/src/manual/prepared/manual_train.jsonl` | Move + rename |
| `training_data/vlm_val.jsonl` | `data/src/manual/prepared/manual_val.jsonl` | Move + rename |
| `training_data/images/` | `data/src/manual/prepared/images/` | Move |
| (vlm-scraper forum data) | `data/src/forum/raw/` | Copy or symlink |
| `specs/01-06_*.md` | `specs/` | Keep in place (historical) |

**Migration steps**:
1. Create the `data/` directory tree
2. `git mv` manual source data, pipeline scripts, config, tests, and prepared output into `data/src/manual/`
3. Update the Makefile paths (this is the main change — scripts take all paths via CLI args, so the scripts themselves need minimal edits)
4. Fix the three minor hardcoded path references:
   - `03b_validate_classification.py`: test image path referencing `data_src/e30-m3-320is/`
   - `04a_generate_qa_images.py`: auto-discovery of `work/indices*` directories
   - `pipeline/scripts/__init__.py`: module aliasing (may need path update)
5. Create `data/src/manual/Makefile` with all manual pipeline targets
6. Create `data/Makefile` (orchestrator) and update root `Makefile` to delegate
7. Copy/symlink forum raw data into `data/src/forum/raw/`
8. Stub out `data/training/` with `merge.py` and `config.yaml`
9. Update `training/modal_train.py` to read from new paths
10. Update `CLAUDE.md` and `README.md` to reflect new structure

**Risk assessment**: Low. An audit of all 10 pipeline scripts confirmed that all critical paths are passed as CLI arguments — there are no hardcoded references to `data_src/`, `work/`, or `training_data/` as defaults. The `config.yaml` contains only API settings and tuning knobs, no file paths. Scripts have zero inter-script Python imports. The migration is primarily a Makefile rewrite with `git mv` for the files.

**Recommendation**: Do the migration in a single atomic commit. Don't try to do it incrementally — you'll end up with a broken intermediate state. Use `git mv` where possible to preserve history.

---

## Verification Strategy

The migration must not break the manual pipeline. Verification happens in two phases:

### Phase 1: Structural verification (immediate, after migration commit)

Run each script with `--help` to confirm imports resolve and argument parsing works in the new location:

```bash
cd data/src/manual
for script in pipeline/*.py; do python "$script" --help > /dev/null && echo "OK: $script" || echo "FAIL: $script"; done
```

Run `make -C data/src/manual status` to confirm the Makefile targets resolve and the prepared output files are found at their new paths.

### Phase 2: Functional verification (full pipeline re-run)

Re-run the complete manual pipeline end-to-end. This serves double duty: it reconstructs the intermediate files (lost during a previous project reorganization) and proves the reorganized paths work.

```bash
make -C data/src/manual all
```

**Comparison against known-good output**: Before the migration, capture baselines from the current prepared output:

```bash
# Pre-migration: capture baselines
wc -l training_data/vlm_train.jsonl training_data/vlm_val.jsonl
python -c "import json; data=open('training_data/vlm_train.jsonl').readlines(); print(f'{len(data)} examples'); print(json.loads(data[0]).keys())"
ls training_data/images/ | wc -l
```

After the re-run, compare:
- Example count should match (12,410 total, ~11,170 train / ~1,240 val)
- Schema should be identical (same keys per record)
- Image count should match
- Q&A content will differ slightly (Claude API is non-deterministic) but coverage and distribution should be consistent

If both phases pass, the migration is verified and safe to build on.

---

## Training Script Integration

`training/modal_train.py` currently reads from `training_data/` at the repo root. After migration it reads from `data/training/`:

- `data/training/merged_train.jsonl` — the default training set (all sources merged)
- Or `data/src/manual/prepared/manual_train.jsonl` — for single-source training runs

The `--dataset-repo` flag for HuggingFace uploads should still work — it just points at a different local path. The Modal volume mount paths don't change.

The training script also needs the `text-only examples` support (records without `"image"` field) before it can train on forum data. This is independent of the reorganization but should happen before the first merged training run.

---

## What This Enables

1. **Independent execution**: `make -C data/src/forum all` runs the forum pipeline without touching the manual pipeline. `make -C data all` fans out to all sources.

2. **Clean extensibility**: A third data source (e.g., parts catalog, YouTube transcripts, another forum) drops in as `data/src/<name>/` following the same convention. Register it in `data/training/config.yaml` and it's included in the next merge.

3. **Source isolation**: Each source's `work/` directory can be blown away independently. A bug in the forum pipeline doesn't touch manual intermediates.

4. **Selective training**: Train on manual-only (`data/src/manual/prepared/`), forum-only (`data/src/forum/prepared/`), or merged (`data/training/`). The training script just takes a path.

5. **Clear ownership**: Anyone looking at `data/src/forum/` sees everything about the forum data source in one place — raw input, processing logic, configuration, intermediates, and final output.
