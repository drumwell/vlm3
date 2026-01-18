# Stage 1: Inventory Implementation Spec

## Objective

Implement `scripts/01_inventory.py` to catalog all source files in `data_src/` and produce `work/inventory.csv` using Test-Driven Development (TDD).

## Implementation Approach

**CRITICAL: Test-Driven Development Required**

1. **Write tests FIRST** - Create comprehensive tests that define expected behavior
2. **Run tests** - Confirm they fail (red phase)
3. **Implement** - Write minimal code to make tests pass (green phase)
4. **Verify** - Confirm all tests pass

## Input Contract

**Source directory:** `data_src/`

Expected to contain:
- Nested directories (e.g., `21 - Clutch/`, `1990 BMW M3 Electrical Troubleshooting Manual/`)
- Image files: `.jpg`, `.jpeg`, `.png`
- PDF files: `.pdf`
- HTML files: `.html`
- Mix of directly contained files and files in subdirectories

**No modifications to `data_src/` allowed** - Read-only operation.

## Output Contract

**Output file:** `work/inventory.csv`

### Schema

```csv
file_path,file_type,section_dir,filename,needs_conversion
```

### Field Definitions

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `file_path` | string | Relative path from project root | `data_src/21 - Clutch/21-01.jpg` |
| `file_type` | string | Normalized file type | `jpg`, `pdf`, `html` |
| `section_dir` | string | Immediate parent directory name | `21 - Clutch` |
| `filename` | string | Base filename with extension | `21-01.jpg` |
| `needs_conversion` | boolean | Whether file requires conversion | `true` for PDFs, `false` otherwise |

### Type Normalization Rules

- `.jpg`, `.jpeg`, `.JPG`, `.JPEG` → `jpg`
- `.png`, `.PNG` → `png`
- `.pdf`, `.PDF` → `pdf`
- `.html`, `.HTML`, `.htm`, `.HTM` → `html`

### Sorting

- Sort by `file_path` (alphabetical)
- Ensures reproducible output

## Test Requirements

### Test File Location

`tests/test_01_inventory.py`

### Required Test Cases

#### 1. Test: Empty directory
- **Setup**: Empty temp directory
- **Expected**: Empty CSV with header only

#### 2. Test: Single JPG file
- **Setup**: One `.jpg` file in root
- **Expected**: One row, `file_type=jpg`, `needs_conversion=false`

#### 3. Test: Mixed file types
- **Setup**: `.jpg`, `.pdf`, `.html` files in root
- **Expected**: Three rows, correct type mapping, PDF flagged for conversion

#### 4. Test: Nested directories
- **Setup**: Files in subdirectories
- **Expected**: `section_dir` populated with immediate parent name

#### 5. Test: Case insensitivity
- **Setup**: `.JPG`, `.jpg`, `.PNG` files
- **Expected**: All normalized to lowercase types

#### 6. Test: Unsupported file types
- **Setup**: `.txt`, `.zip`, `.md` files mixed with supported types
- **Expected**: Unsupported files ignored (not in output)

#### 7. Test: Files without parent directory
- **Setup**: HTML file directly in `data_src/`
- **Expected**: `section_dir` is empty string

#### 8. Test: Alphabetical sorting
- **Setup**: Multiple files in random order
- **Expected**: Output sorted by `file_path`

#### 9. Test: Idempotency
- **Setup**: Run script twice on same input
- **Expected**: Identical output both times

#### 10. Test: PDF conversion flag
- **Setup**: Mix of PDFs and images
- **Expected**: Only PDFs have `needs_conversion=true`

### Test Framework

Use `pytest` with temporary directories:

```python
import pytest
import tempfile
import csv
from pathlib import Path

@pytest.fixture
def temp_data_src(tmp_path):
    """Create temporary data_src directory"""
    data_src = tmp_path / "data_src"
    data_src.mkdir()
    return data_src

@pytest.fixture
def temp_work(tmp_path):
    """Create temporary work directory"""
    work = tmp_path / "work"
    work.mkdir()
    return work
```

## Implementation Requirements

### Script Interface

**Location:** `scripts/01_inventory.py`

**CLI Arguments:**

```bash
python scripts/01_inventory.py \
    --data-src data_src \
    --output work/inventory.csv
```

**Help output:**

```bash
python scripts/01_inventory.py --help
```

Should display:
- Purpose
- Required arguments
- Optional arguments
- Example usage

### Implementation Constraints

1. **Read-only**: Never modify `data_src/` contents
2. **Create output directory**: If `work/` doesn't exist, create it
3. **Overwrite output**: If `work/inventory.csv` exists, overwrite it
4. **Error handling**:
   - Handle permission errors gracefully
   - Handle missing `data_src/` directory
   - Log errors to stderr
5. **Logging**:
   - Summary statistics to stdout (files found, by type)
   - Use Python `logging` module

### Code Structure

```python
#!/usr/bin/env python3
"""
Stage 1: Inventory
Catalogs all source files in data_src/ and produces work/inventory.csv
"""

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Dict

def scan_directory(data_src: Path) -> List[Dict[str, str]]:
    """
    Recursively scan data_src for supported files.

    Returns:
        List of file records with keys: file_path, file_type, section_dir, filename, needs_conversion
    """
    pass

def normalize_file_type(extension: str) -> str:
    """
    Normalize file extension to standard type.

    Args:
        extension: File extension including dot (e.g., '.jpg', '.PDF')

    Returns:
        Normalized type ('jpg', 'pdf', 'html', 'png') or None if unsupported
    """
    pass

def write_inventory_csv(records: List[Dict[str, str]], output_path: Path):
    """
    Write inventory records to CSV file.

    Args:
        records: List of file records
        output_path: Path to output CSV file
    """
    pass

def main():
    """Main entry point"""
    pass

if __name__ == "__main__":
    main()
```

### Dependencies

Add to `requirements.txt` (if not already present):

```
pytest>=7.0.0
```

No external dependencies needed for the script itself (use stdlib only).

## Acceptance Criteria

### Tests

- [ ] All 10 test cases written
- [ ] Tests run with `pytest tests/test_01_inventory.py`
- [ ] Tests initially fail (before implementation)
- [ ] Tests pass after implementation
- [ ] Test coverage >95% on `01_inventory.py`

### Implementation

- [ ] Script runs without errors on actual `data_src/`
- [ ] Produces `work/inventory.csv` with correct schema
- [ ] Handles all file types correctly
- [ ] Sorts output alphabetically
- [ ] Idempotent (same output on repeated runs)
- [ ] Has `--help` flag with clear documentation
- [ ] Logs summary statistics

### Output Validation

Run on actual `data_src/` and verify:

- [ ] All `.jpg`/`.png` files cataloged
- [ ] All `.pdf` files cataloged with `needs_conversion=true`
- [ ] All `.html` files cataloged
- [ ] No unsupported file types in output
- [ ] Section directories correctly extracted
- [ ] Total file count matches manual inspection

## Example Expected Output

Given this structure:

```
data_src/
├── 21 - Clutch/
│   ├── 21-00-index-a.jpg
│   └── 21-01.jpg
├── Getrag265/
│   └── Getrag265_Rebuild.pdf
└── M3-techspec.html
```

Expected `work/inventory.csv`:

```csv
file_path,file_type,section_dir,filename,needs_conversion
data_src/21 - Clutch/21-00-index-a.jpg,jpg,21 - Clutch,21-00-index-a.jpg,false
data_src/21 - Clutch/21-01.jpg,jpg,21 - Clutch,21-01.jpg,false
data_src/Getrag265/Getrag265_Rebuild.pdf,pdf,Getrag265,Getrag265_Rebuild.pdf,true
data_src/M3-techspec.html,html,,M3-techspec.html,false
```

## Implementation Steps

1. **Create test file** `tests/test_01_inventory.py` with all 10 test cases
2. **Run tests**: `pytest tests/test_01_inventory.py -v`
3. **Confirm RED**: Tests should fail (script doesn't exist yet)
4. **Create script skeleton** `scripts/01_inventory.py` with imports and structure
5. **Implement `normalize_file_type()`** - Make type normalization tests pass
6. **Implement `scan_directory()`** - Make directory scanning tests pass
7. **Implement `write_inventory_csv()`** - Make output writing tests pass
8. **Implement `main()` and argparse** - Make CLI tests pass
9. **Run all tests**: `pytest tests/test_01_inventory.py -v`
10. **Confirm GREEN**: All tests pass
11. **Run on actual data**: `python scripts/01_inventory.py --data-src data_src --output work/inventory.csv`
12. **Validate output**: Inspect `work/inventory.csv` for correctness
13. **Commit**: `git add -A && git commit -m "feat: implement Stage 1 inventory script with tests"`

## Success Metrics

- **Zero test failures**
- **Script runs in <5 seconds** on expected data_src/ size (~500 files)
- **Output is valid CSV** (parseable, correct columns)
- **Reproducible**: Same input → same output (byte-for-byte identical)

## Additional Notes

- Use `pathlib.Path` for all path operations (cross-platform)
- Use Python 3.8+ features (assume modern Python)
- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Add type hints where helpful

## What NOT to Do

- ❌ Don't modify any files in `data_src/`
- ❌ Don't perform image validation (just catalog)
- ❌ Don't extract metadata from files (just file-level info)
- ❌ Don't convert PDFs (that's Stage 2)
- ❌ Don't classify content (that's Stage 3)
- ❌ Don't use external libraries (stdlib only for the script)

## Next Stage Preview

After Stage 1 completes successfully, `work/inventory.csv` will be the input for Stage 2 (Source Preparation), which will:
- Convert PDFs to JPGs
- Validate images are readable
- Produce `work/inventory_prepared.csv`

But for now, focus only on Stage 1: cataloging what exists.
