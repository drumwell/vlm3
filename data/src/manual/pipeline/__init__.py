"""
BMW E30 M3 Service Manual - VLM Pipeline Scripts Package

This package provides clean module aliases for the numbered pipeline scripts.
Use these aliases in tests and other code that needs to import script functions.

Usage in tests:
    from scripts import inventory, prepare_sources, classify_pages

    def test_scan():
        result = inventory.scan_directory(Path("/tmp/test"))
        assert result is not None

The numbered filenames are preserved for CLI ordering visibility,
while these aliases provide valid Python import names.
"""

from importlib import import_module

# Stage 1: Inventory
inventory = import_module(".01_inventory", __name__)

# Stage 2: Source Preparation
prepare_sources = import_module(".02_prepare_sources", __name__)

# Stage 3: Classification & Index Parsing
classify_pages = import_module(".03_classify_pages", __name__)

# Stage 3b: Classification Validation (optional)
try:
    validate_classification = import_module(".03b_validate_classification", __name__)
except ImportError:
    validate_classification = None

# Stage 4a: Q&A Generation from Images
generate_qa_images = import_module(".04a_generate_qa_images", __name__)

# Stage 4b: Q&A Generation from HTML
generate_qa_html = import_module(".04b_generate_qa_html", __name__)

# Stage 5: Q&A Filtering
filter_qa = import_module(".05_filter_qa", __name__)

# Stage 6: Q&A Deduplication
deduplicate_qa = import_module(".06_deduplicate_qa", __name__)

# Stage 7: Emit VLM Dataset
emit_vlm_dataset = import_module(".07_emit_vlm_dataset", __name__)

# Stage 8: Validate VLM Dataset
validate_vlm = import_module(".08_validate_vlm", __name__)

# Stage 9: Upload to HuggingFace
upload_vlm = import_module(".09_upload_vlm", __name__)

# Export all module aliases
__all__ = [
    "inventory",
    "prepare_sources",
    "classify_pages",
    "validate_classification",
    "generate_qa_images",
    "generate_qa_html",
    "filter_qa",
    "deduplicate_qa",
    "emit_vlm_dataset",
    "validate_vlm",
    "upload_vlm",
]
