# Data Orchestrator - Delegates to source pipelines and runs merge
SOURCES := manual forum

# Run all source pipelines then merge
all:
	@for src in $(SOURCES); do $(MAKE) -C src/$$src all; done
	$(MAKE) merge

# Run a specific source
manual:
	$(MAKE) -C src/manual all

forum:
	$(MAKE) -C src/forum all

# Merge all prepared outputs into unified training set
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
	rm -rf training/images/

.PHONY: all manual forum merge status clean
