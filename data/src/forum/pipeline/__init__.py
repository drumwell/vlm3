"""
s14.net Forum Data - Curation Pipeline Scripts Package

Module aliases for numbered pipeline scripts, following manual pipeline convention.

Usage in tests:
    from pipeline import reconstruct, filter_threads, classify
"""

from importlib import import_module

# Stage 1: Reconstruct Threads
reconstruct = import_module(".01_reconstruct", __name__)

# Stage 2: Mechanical Filtering
filter_threads = import_module(".02_filter", __name__)

# Stage 3: Thread Classification
classify = import_module(".03_classify", __name__)

# Stage 4: Quality Scoring
score = import_module(".04_score", __name__)

# Stage 5: Q&A Extraction
extract_qa = import_module(".05_extract_qa", __name__)

# Stage 6: Cross-Reference & Filter
validate = import_module(".06_validate", __name__)

# Stage 7: Emit Training Data
emit = import_module(".07_emit", __name__)

__all__ = [
    "reconstruct",
    "filter_threads",
    "classify",
    "score",
    "extract_qa",
    "validate",
    "emit",
]
