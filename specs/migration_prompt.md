# Claude Code Prompt: Data Architecture Migration

## Goal

Reorganize the project directory structure so that each data source is self-contained under `data/src/<name>/`. See `specs/data_architecture_spec.md` for the full target structure. The service manual pipeline must still work after this migration.

## Context

- All 10 pipeline scripts (01-09, plus 03b) take paths via CLI arguments — no hardcoded defaults for `data_src/`, `work/`, or `training_data/`.
- `pipeline/config.yaml` contains only API settings and tuning knobs, no file paths.
- Scripts have zero inter-script Python imports.
- Tests import via `from scripts import inventory` etc., using `__init__.py` module aliases.
- The `conftest.py` computes project root as `Path(__file__).parent.parent`.

## Step-by-step instructions

### 1. Create the directory tree

```bash
mkdir -p data/src/manual/{pipeline,raw,work,prepared,tests}
mkdir -p data/src/forum/raw
mkdir -p data/training
```

### 2. Move manual source data (use `git mv` to preserve history)

```bash
# Raw source data — move the section folders into raw/
git mv data_src/e30-m3-320is/* data/src/manual/raw/
# Note: data_src/e30-m3-320is/ contains ~45 section folders like "00 - Maintenance", "11 - Engine", etc.
# Also includes 320is-techspec.html

# Pipeline scripts — move from pipeline/scripts/ into pipeline/
git mv pipeline/scripts/01_inventory.py data/src/manual/pipeline/
git mv pipeline/scripts/02_prepare_sources.py data/src/manual/pipeline/
git mv pipeline/scripts/03_classify_pages.py data/src/manual/pipeline/
git mv pipeline/scripts/03b_validate_classification.py data/src/manual/pipeline/
git mv pipeline/scripts/04a_generate_qa_images.py data/src/manual/pipeline/
git mv pipeline/scripts/04b_generate_qa_html.py data/src/manual/pipeline/
git mv pipeline/scripts/05_filter_qa.py data/src/manual/pipeline/
git mv pipeline/scripts/06_deduplicate_qa.py data/src/manual/pipeline/
git mv pipeline/scripts/07_emit_vlm_dataset.py data/src/manual/pipeline/
git mv pipeline/scripts/08_validate_vlm.py data/src/manual/pipeline/
git mv pipeline/scripts/09_upload_vlm.py data/src/manual/pipeline/
git mv pipeline/scripts/__init__.py data/src/manual/pipeline/__init__.py

# Config
git mv pipeline/config.yaml data/src/manual/config.yaml

# Tests
git mv pipeline/tests/conftest.py data/src/manual/tests/
git mv pipeline/tests/test_*.py data/src/manual/tests/
git mv pipeline/tests/__init__.py data/src/manual/tests/
git mv pipeline/tests/fixtures data/src/manual/tests/fixtures

# Prepared output
git mv training_data/vlm_train.jsonl data/src/manual/prepared/manual_train.jsonl
git mv training_data/vlm_val.jsonl data/src/manual/prepared/manual_val.jsonl
git mv training_data/images data/src/manual/prepared/images
```

After these moves, remove the now-empty directories:
```bash
git rm -r pipeline/
git rm -r data_src/
git rm -r training_data/
```

### 3. Fix test imports

All test files currently do `from scripts import ...`. Since the scripts moved from `pipeline/scripts/` to `data/src/manual/pipeline/`, and the tests use `Path(__file__).parent.parent` as the root, the import path changes:

In every `test_*.py` file under `data/src/manual/tests/`:
- Replace `from scripts import` with `from pipeline import`
- Replace `from scripts.` with `from pipeline.`
- Replace `import scripts` with `import pipeline`

The `conftest.py` `PROJECT_ROOT = Path(__file__).parent.parent` computation still works: from `data/src/manual/tests/conftest.py`, `parent.parent` resolves to `data/src/manual/`, which is the correct root for finding `pipeline/`.

### 4. Fix the three minor hardcoded paths in scripts

**`data/src/manual/pipeline/03b_validate_classification.py`**:
- Find any hardcoded reference to `data_src/e30-m3-320is/` and update to `raw/`

**`data/src/manual/pipeline/04a_generate_qa_images.py`**:
- Find the auto-discovery logic that looks for `work/indices`, `work/indices_100`, `work/indices_50`
- This is fine as-is because these paths are relative and the Makefile will be run from `data/src/manual/`

**`data/src/manual/pipeline/__init__.py`**:
- No changes needed — it uses relative imports (`import_module(".01_inventory", __name__)`)

### 5. Create `data/src/manual/Makefile`

Create a new Makefile at `data/src/manual/Makefile` that contains all the manual pipeline targets currently in the root `Makefile`. Key changes from the existing root Makefile:

- `pipeline/scripts/01_inventory.py` → `pipeline/01_inventory.py`
- `--data-src $(DATA_SRC)` where `DATA_SRC ?= raw`
- All `work/` paths stay as-is (they're relative, and make runs from this directory)
- Output paths: `training_data/` → `prepared/`
- `vlm_train.jsonl` → `manual_train.jsonl`, `vlm_val.jsonl` → `manual_val.jsonl`
- `training_data/images` → `prepared/images`
- Include: `all`, `status`, `clean` targets at minimum
- Remove training and eval targets (those stay at the root level)

Use the existing Makefile as the template. Copy the manual pipeline targets (inventory through upload) and the convenience targets (quick, regen-qa, refilter, finalize), updating all paths.

### 6. Create `data/Makefile` (orchestrator)

This Makefile delegates to source Makefiles and runs the merge:

```makefile
SOURCES := manual

all:
	@for src in $(SOURCES); do $(MAKE) -C src/$$src all; done

manual:
	$(MAKE) -C src/manual all

merge:
	python training/merge.py --config training/config.yaml

status:
	@for src in $(SOURCES); do echo "=== $$src ===" && $(MAKE) -C src/$$src status && echo; done

clean:
	@for src in $(SOURCES); do $(MAKE) -C src/$$src clean; done
```

### 7. Update root `Makefile`

Replace all the manual pipeline targets (inventory, prepare, classify, generate-qa, filter-qa, deduplicate-qa, emit, validate, upload, and convenience targets) with delegation:

```makefile
# Data pipeline
data:
	$(MAKE) -C data all

data-manual:
	$(MAKE) -C data/src/manual all

data-status:
	$(MAKE) -C data status
```

Keep the training targets (train, train-dev, train-resume, train-logs) and eval targets, but update any path references from `training_data/` to `data/src/manual/prepared/` or `data/training/`.

Keep the `help` target and update it to reflect the new structure.

### 8. Stub out `data/src/forum/raw/` and `data/training/`

For `data/src/forum/raw/`: just create the directory. The forum data will be copied/symlinked in later.

For `data/training/`: create `config.yaml` and a placeholder `merge.py`:

**`data/training/config.yaml`**:
```yaml
sources:
  manual:
    path: ../src/manual/prepared
    weight: 1.0

upsample_to_balance: true
shuffle_seed: 42
```

**`data/training/merge.py`**: a placeholder that prints "not yet implemented" — the real implementation comes with the forum pipeline.

### 9. Update `.gitignore`

Current entries that need updating:
- `work/` → `data/src/*/work/`
- `data_src/forum/` → remove (no longer at this path)
- Add: `data/training/merged_*.jsonl` (generated output, don't commit)

### 10. Update `CLAUDE.md` and `README.md`

Update the project structure descriptions, pipeline command references, and file paths to reflect the new `data/` organization. The key message: all data concerns live under `data/`, each source is self-contained under `data/src/<name>/`.

### 11. Structural verification

After all changes, run this verification:

```bash
# Verify all scripts load correctly
cd data/src/manual
for script in pipeline/[0-9]*.py; do
  python "$script" --help > /dev/null 2>&1 && echo "OK: $script" || echo "FAIL: $script"
done

# Verify Makefile targets resolve
make status

# Verify tests can import
cd tests
python -c "import sys; sys.path.insert(0, '..'); from pipeline import inventory; print('imports OK')"
cd ../../../..

# Verify prepared output exists at new location
ls -la data/src/manual/prepared/manual_train.jsonl
ls -la data/src/manual/prepared/manual_val.jsonl
ls data/src/manual/prepared/images/ | head -5
```

If all checks pass, commit.

### 12. Commit

Stage all changes and commit with a message like:
```
Reorganize project: self-contained data sources under data/src/

Move manual pipeline (scripts, config, tests, raw data, prepared output)
into data/src/manual/ following the data architecture convention defined
in specs/data_architecture_spec.md. Each data source is now independent
with its own Makefile, raw input, pipeline scripts, work directory, and
prepared output.

No logic changes to pipeline scripts — only path references in Makefiles
and test imports were updated.
```

## Important notes

- Use `git mv` for every file move to preserve history.
- Do NOT modify the logic of any pipeline script. The only changes to .py files should be: test import paths (`scripts` → `pipeline`), and the hardcoded path in `03b_validate_classification.py`.
- The root Makefile should still have `train` and `eval` targets — only pipeline targets move.
- If you encounter any `__pycache__` directories, just delete them (`find . -name __pycache__ -exec rm -rf {} +`).
- The existing `data_src/.gitignore` can be removed if it only contained forum-related ignores.
