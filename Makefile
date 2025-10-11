SECT_FILTER?=

inventory:
	python scripts/01_inventory.py --root data_src --out work/inventory.csv --sect_filter "$(SECT_FILTER)"

preprocess:
	python scripts/02_preprocess.py --inventory work/inventory.csv --out work/images_clean

ocr:
	python scripts/03_ocr.py --images work/images_clean --out work/ocr_raw
	python scripts/03b_ocr_tables.py --images work/images_clean --out work/ocr_tables

blocks:
	python scripts/04_parse_blocks.py --ocr work/ocr_raw --tables work/ocr_tables --out work/blocks --config config.yaml

emit:
	python scripts/05_emit_jsonl.py --blocks work/blocks --out data --mixed_inputs true

validate:
	python scripts/06_split_validate.py --data-dir data --output work/logs/qa_report.md

all: inventory preprocess ocr blocks emit validate
