# Makefile for Minions
#
# Configuration and evaluation targets

CONFIG_DIR := config
KCONFIG := $(CONFIG_DIR)/Kconfig
CONFIG := $(CONFIG_DIR)/.config
DEFCONFIG := $(CONFIG_DIR)/presets/defconfig

.PHONY: menuconfig guiconfig defconfig loadconfig savedefconfig run correctness clean cleanall help

help:
	@echo "Minions - Available targets:"
	@echo ""
	@echo "  Configuration:"
	@echo "    menuconfig     - Interactive configuration menu (ncurses)"
	@echo "    guiconfig      - Graphical configuration menu (requires Tk)"
	@echo "    defconfig      - Load default configuration"
	@echo "    loadconfig     - Load a specific config file"
	@echo "                     Usage: make loadconfig FILE=path/to/config"
	@echo "    savedefconfig  - Save current config as defconfig"
	@echo ""
	@echo "  Evaluation:"
	@echo "    run            - Run evaluation with current config"
	@echo "    correctness    - Run correctness evaluation on latest results"
	@echo ""
	@echo "  Cleanup:"
	@echo "    clean          - Remove configuration files"
	@echo "    cleanall       - Remove config and all results"
	@echo ""
	@echo "Quick start:"
	@echo "  1. make defconfig        # Load default configuration"
	@echo "  2. make menuconfig       # Customize settings (optional)"
	@echo "  3. make run              # Run evaluation"
	@echo "  4. make correctness      # Evaluate correctness (optional)"
	@echo ""

menuconfig:
	@python3 -c "from kconfiglib import Kconfig; import menuconfig; \
		import os; os.environ['KCONFIG_CONFIG'] = '$(CONFIG)'; \
		menuconfig.menuconfig(Kconfig('$(KCONFIG)'))"

guiconfig:
	@python3 $(CONFIG_DIR)/kguiconfig.py

defconfig:
	@if [ -f $(DEFCONFIG) ]; then \
		cp $(DEFCONFIG) $(CONFIG); \
		echo "Loaded default configuration from $(DEFCONFIG)"; \
		echo "Configuration saved to $(CONFIG)"; \
	else \
		echo "Error: $(DEFCONFIG) not found"; \
		exit 1; \
	fi

loadconfig:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make loadconfig FILE=path/to/config"; \
		exit 1; \
	fi
	@if [ -f "$(FILE)" ]; then \
		cp "$(FILE)" $(CONFIG); \
		echo "Loaded configuration from $(FILE) to $(CONFIG)"; \
	else \
		echo "Error: $(FILE) not found"; \
		exit 1; \
	fi

savedefconfig:
	@if [ -f $(CONFIG) ]; then \
		cp $(CONFIG) $(DEFCONFIG); \
		echo "Saved current configuration to $(DEFCONFIG)"; \
	else \
		echo "Error: $(CONFIG) not found. Run 'make defconfig' or 'make menuconfig' first."; \
		exit 1; \
	fi

run:
	@if [ ! -f $(CONFIG) ]; then \
		echo "No $(CONFIG) found. Run 'make defconfig' or 'make menuconfig' first."; \
		exit 1; \
	fi
	python3 evaluate/financebench_evaluator.py $(CONFIG)

correctness:
	@LATEST=$$(ls -td evaluate/results/*/ 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "Error: No results found in evaluate/results/"; \
		echo "Run 'make run' first to generate results."; \
		exit 1; \
	fi; \
	echo "Running correctness evaluation on: $$LATEST"; \
	python3 evaluate/correctness.py "$$LATEST" --verbose --update-summary

clean:
	rm -f $(CONFIG) $(CONFIG).old
	@echo "Cleaned configuration files"

cleanall: clean
	rm -rf evaluate/results/*
	@echo "Cleaned results directory"
