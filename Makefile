# Top-level Makefile for Minions
#
# This delegates evaluation-related targets to evaluate/Makefile

.PHONY: menuconfig guiconfig defconfig loadconfig savedefconfig run correctness clean cleanall help

help:
	@echo "Minions - Available targets:"
	@echo ""
	@echo "  Evaluation (delegated to evaluate/):"
	@echo "    menuconfig     - Interactive configuration menu (ncurses)"
	@echo "    guiconfig      - Graphical configuration menu (requires Tk)"
	@echo "    defconfig      - Load default configuration"
	@echo "    loadconfig     - Load a specific config file (FILE=path)"
	@echo "    savedefconfig  - Save current config as defconfig"
	@echo "    run            - Run evaluation with current .config"
	@echo "    correctness    - Run correctness evaluation on latest results"
	@echo "    clean          - Remove configuration files"
	@echo "    cleanall       - Remove config and all results"
	@echo ""

menuconfig:
	@$(MAKE) -C evaluate menuconfig

guiconfig:
	@$(MAKE) -C evaluate guiconfig

defconfig:
	@$(MAKE) -C evaluate defconfig

loadconfig:
	@$(MAKE) -C evaluate loadconfig FILE="$(FILE)"

savedefconfig:
	@$(MAKE) -C evaluate savedefconfig

run:
	@$(MAKE) -C evaluate run

correctness:
	@$(MAKE) -C evaluate correctness

clean:
	@$(MAKE) -C evaluate clean

cleanall:
	@$(MAKE) -C evaluate cleanall
