SECT_FILTER?=

inventory:
\tpython scripts/01_inventory.py --root data_src --out work/inventory.csv --sect_filter "$(SECT_FILTER)"

preprocess:
\tpython scripts/02_preprocess.py --inventory work/inventory.csv --out work/images_clean

ocr:
\tpython scripts/03_ocr.py --images work/images_clean --out work/ocr_raw
\tpython scripts/03b_ocr_tables.py --images work/images_clean --out work/ocr_tables

blocks:
\tpython scripts/04_parse_blocks.py --ocr work/ocr_raw --tables work/ocr_tables --out work/blocks --config config.yaml

emit:
\tpython scripts/05_emit_jsonl.py --blocks work/blocks --out data --mixed_inputs true

validate:
\tpython scripts/06_split_validate.py --data_dir data --report work/logs/qa_report.md

all: inventory preprocess ocr blocks emit validate
