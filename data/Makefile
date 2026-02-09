# Data Orchestrator - Delegates to source pipelines and runs merge
SOURCES := manual

# Run all source pipelines then merge
all:
	@for src in $(SOURCES); do $(MAKE) -C src/$$src all; done

# Run a specific source
manual:
	$(MAKE) -C src/manual all

# Merge all prepared outputs into unified training set
merge:
	python training/merge.py --config training/config.yaml

# Status across all sources
status:
	@for src in $(SOURCES); do echo "=== $$src ===" && $(MAKE) -C src/$$src status && echo; done

# Clean everything
clean:
	@for src in $(SOURCES); do $(MAKE) -C src/$$src clean; done

.PHONY: all manual merge status clean
