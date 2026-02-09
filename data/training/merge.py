#!/usr/bin/env python3
"""
Merge prepared outputs from all data sources into unified training set.

Reads from each source's prepared/ directory, applies weighting,
shuffles, and emits merged_train.jsonl and merged_val.jsonl.

Usage:
    python merge.py --config config.yaml
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge data source outputs")
    parser.add_argument("--config", required=True, help="Path to merge config")
    args = parser.parse_args()

    print(f"merge.py: not yet implemented (config: {args.config})")
    print("This will be implemented with the forum pipeline.")
    sys.exit(0)


if __name__ == "__main__":
    main()
