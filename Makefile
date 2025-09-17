###################################
# Makefile for shared clusters simulation
###################################

# Configuration variables
NAME ?= shared_clusters_test
BASE_DIR = ./simulations
TRIAL_DIR = $(BASE_DIR)/$(NAME)
SCRIPT_DIR = $(shell pwd)
VENV_PATH       = $(HOME)/.venv/ml

# Output directories
OUTPUT_WITHOUT = $(TRIAL_DIR)/without_shared_clusters
OUTPUT_WITH = $(TRIAL_DIR)/with_shared_clusters

# Simulation script
SIM_SCRIPT = run_simulations.sh
CHECK_SCRIPT = run_check.sh
CHECK_MACRO = CheckTracksCA.C
COMPARISON_SCRIPT = tests/compare_efficiency_fake.py
PREPROCESS_SH   = run_preprocess.sh
PREPROCESS_PY  = $(SCRIPT_DIR)/macros/preprocess.py
ANALYSIS_SH     = run_analysis.sh
ANALYSIS_PY     = $(SCRIPT_DIR)/macros/draw_shared.py $(SCRIPT_DIR)/macros/study_doubly_reco.py 

# Configuration variables for simulation
NWORKERS ?= 30
NSIGEVENTS ?= 20000
NTIMEFRAMES ?= 1
SPLITID ?= 42
SIMENGINE ?= TGeant4

# Output files (tracking completion)
SIM_WITHOUT_OUTPUT = $(OUTPUT_WITHOUT)/simulation.done
SIM_WITH_OUTPUT = $(OUTPUT_WITH)/simulation.done
COPY_OUTPUT = $(OUTPUT_WITH)/copy.done
CHECK_OUTPUT = $(TRIAL_DIR)/check.done
PREPROC_OUTPUT = $(TRIAL_DIR)/preprocess.done
ANALYSIS_OUTPUT    = $(TRIAL_DIR)/analysis.done

# Define the targets
.PHONY: all clean-output help setup simulate-without copy-output simulate-with validate-environment force-simulate-with preprocess analysis

# Default target
all: validate-environment setup simulate-without copy-output simulate-with check preprocess analysis

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Run complete pipeline (setup -> simulate-without -> copy-output -> simulate-with)"
	@echo "  setup            - Create directory structure"
	@echo "  simulate-without - Run simulation without shared clusters"
	@echo "  copy-output      - Copy simulation output to shared clusters folder"
	@echo "  simulate-with    - Run simulation with shared clusters"
	@echo "  clean-output     - Remove output files but keep directory structure"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  NAME=$(NAME)           - Trial name (override with NAME=your_trial)"
	@echo "  NWORKERS=$(NWORKERS)         - Number of workers for simulation"
	@echo "  NSIGEVENTS=$(NSIGEVENTS)       - Number of signal events"
	@echo "  NTIMEFRAMES=$(NTIMEFRAMES)        - Number of time frames"
	@echo "  SPLITID=$(SPLITID)            - Seed for simulation"
	@echo "  SIMENGINE=$(SIMENGINE)      - Simulation engine"
	@echo ""
	@echo "Examples:"
	@echo "  make NAME=test_run NSIGEVENTS=10000    - Run with custom parameters"
	@echo "  make simulate-without                  - Run only simulation without shared clusters"
	@echo ""
	@echo "Notes:"
	@echo "  - Simulation script must be present in current directory"
	@echo "  - O2DPG_ROOT and O2_ROOT must be set in environment"
	@echo "  - Shared clusters simulation extracts command from tf1/itsreco_1.log_done"

# Setup directories
setup: $(OUTPUT_WITHOUT) $(OUTPUT_WITH)

# Create directories
$(OUTPUT_WITHOUT):
	@mkdir -p $@

$(OUTPUT_WITH):
	@mkdir -p $@

# Run simulation without shared clusters
$(SIM_WITHOUT_OUTPUT): $(OUTPUT_WITHOUT) $(SIM_SCRIPT)
	@echo "Running simulation without shared clusters..."
	@if [ ! -f "$(SIM_SCRIPT)" ]; then \
		echo "Error: $(SIM_SCRIPT) not found in current directory"; \
		exit 1; \
	fi
	@if [ -z "$(O2DPG_ROOT)" ]; then \
		echo "Error: O2DPG_ROOT not set"; \
		exit 1; \
	fi
	@if [ -z "$(O2_ROOT)" ]; then \
		echo "Error: O2_ROOT not set"; \
		exit 1; \
	fi
	@echo "Starting simulation in directory: $(OUTPUT_WITHOUT)"
	@cd $(OUTPUT_WITHOUT) && \
		export OUTPUT_DIR="." && \
		export NWORKERS=$(NWORKERS) && \
		export NSIGEVENTS=$(NSIGEVENTS) && \
		export NTIMEFRAMES=$(NTIMEFRAMES) && \
		export SPLITID=$(SPLITID) && \
		export SIMENGINE=$(SIMENGINE) && \
		export SHARED_CLUSTERS=false && \
		bash $(abspath $(SIM_SCRIPT)) > simulation.log 2>&1
	@if [ -d "$(OUTPUT_WITHOUT)/tf1" ]; then \
		echo "✓ Simulation completed successfully"; \
	else \
		echo "✗ Simulation failed - tf1 directory not found"; \
		exit 1; \
	fi
	@touch $@

# Copy output to shared clusters folder
$(COPY_OUTPUT): $(SIM_WITHOUT_OUTPUT)
	@echo "Copying simulation output to shared clusters folder..."
	@if [ ! -d "$(OUTPUT_WITHOUT)/tf1" ]; then \
		echo "Error: Source tf1 directory $(OUTPUT_WITHOUT)/tf1 does not exist"; \
		echo "Make sure the simulation without shared clusters completed successfully"; \
		exit 1; \
	fi
	@echo "Copying from $(OUTPUT_WITHOUT) to $(OUTPUT_WITH)..."
	@rsync -av --exclude="*.done" $(OUTPUT_WITHOUT)/ $(OUTPUT_WITH)/
	@if [ -d "$(OUTPUT_WITH)/tf1" ]; then \
		echo "✓ Copy completed successfully"; \
	else \
		echo "✗ Copy failed - tf1 directory not found in destination"; \
		exit 1; \
	fi
	@touch $@

# Run simulation with shared clusters
$(SIM_WITH_OUTPUT): $(COPY_OUTPUT)
	@echo "Running simulation with shared clusters..."
	@if [ ! -f "$(SIM_SCRIPT)" ]; then \
		echo "Error: $(SIM_SCRIPT) not found in current directory"; \
		exit 1; \
	fi
	@if [ -z "$(O2DPG_ROOT)" ]; then \
		echo "Error: O2DPG_ROOT not set"; \
		exit 1; \
	fi
	@if [ -z "$(O2_ROOT)" ]; then \
		echo "Error: O2_ROOT not set"; \
		exit 1; \
	fi
	@echo "Starting simulation in directory: $(OUTPUT_WITH)"
	@cd $(OUTPUT_WITH) && \
		export OUTPUT_DIR="." && \
		export NWORKERS=$(NWORKERS) && \
		export NSIGEVENTS=$(NSIGEVENTS) && \
		export NTIMEFRAMES=$(NTIMEFRAMES) && \
		export SPLITID=$(SPLITID) && \
		export SIMENGINE=$(SIMENGINE) && \
		export SHARED_CLUSTERS=true && \
		bash $(abspath $(SIM_SCRIPT)) > simulation.log 2>&1
	@if [ -d "$(OUTPUT_WITH)/tf1" ]; then \
		echo "✓ Simulation completed successfully"; \
	else \
		echo "✗ Simulation failed - tf1 directory not found"; \
		exit 1; \
	fi
	@touch $@
# 	@if [ ! -f "$(OUTPUT_WITH)/tf1/itsreco_1.log_done" ]; then \
# 		echo "Error: $(OUTPUT_WITH)/tf1/itsreco_1.log_done not found"; \
# 		echo "Make sure the simulation without shared clusters completed successfully"; \
# 		exit 1; \
# 	fi



# 	@echo "Backing up log file..."; \
# 	cp "$(OUTPUT_WITH)/tf1/itsreco_1.log" "$(OUTPUT_WITH)/tf1/itsreco_1.log.bak"

# 	@echo "Extracting command from itsreco_1.log_done..."
# 	@FULL_LINE=$$(grep 'Command ' $(OUTPUT_WITH)/tf1/itsreco_1.log_done | head -1); \
# 	echo "Full line: $$FULL_LINE"; \
# 	COMMAND_PART=$$(echo "$$FULL_LINE" | sed 's/^Command "//;s/" successfully finished.*//'); \
# 	echo "Extracted command: $$COMMAND_PART"; \
# 	if [ -z "$$COMMAND_PART" ]; then \
# 		echo "Error: Could not extract command from log file"; \
# 		mv "$(OUTPUT_WITH)/tf1/itsreco_1.log.bak" "$(OUTPUT_WITH)/tf1/itsreco_1.log"; \
# 		exit 1; \
# 	fi; \
# 	MODIFIED_COMMAND=$$(echo "$$COMMAND_PART" | sed -E 's/(--configKeyValues ")([^"]*)/\1\2;ITSCATrackerParam.allowSharingFirstCluster=true/'); \
# 	echo "Modified command: $$MODIFIED_COMMAND"; \
# 	cd $(OUTPUT_WITH)/tf1 && \
# 	if eval "$$MODIFIED_COMMAND" > shared_clusters_log.txt 2>&1; then \
# 		rm -f itsreco_1.log.bak itsreco_1.log; \
# 		mv shared_clusters_log.txt itsreco_1.log; \
# 	else \
# 		echo "Simulation failed — restoring original log file."; \
# 		mv shared_clusters_log.txt itsreco_1.log.err; \
# 		mv itsreco_1.log.bak itsreco_1.log; \
# 		exit 1; \
# 	fi
# 	@touch $@

# Run check script
$(CHECK_OUTPUT): $(SIM_WITH_OUTPUT) $(SIM_WITHOUT_OUTPUT) $(CHECK_SCRIPT) $(COMPARISON_SCRIPT) $(CHECK_MACRO)
	@echo "Running check script..."
	@if [ ! -f "$(CHECK_SCRIPT)" ]; then \
		echo "Error: $(CHECK_SCRIPT) not found in current directory"; \
		exit 1; \
	fi
	@echo "Executing check script..."
	@export OUTPUT_DIR="$(TRIAL_DIR)" && bash $(abspath $(CHECK_SCRIPT))
	@if [ $$? -eq 0 ]; then \
		echo "✓ Check completed successfully"; \
	else \
		echo "✗ Check failed"; \
		exit 1; \
	fi
	@touch $@

# Preprocessing (ROOT → Parquet)
$(PREPROC_OUTPUT): $(CHECK_OUTPUT) $(PREPROCESS_PY) $(PREPROCESS_SH)
	@echo "Pre-processing ROOT → Parquet (without shared clusters)..."
	@export OUTPUT_DIR="$(TRIAL_DIR)" && bash $(abspath $(PREPROCESS_SH))
	@if [ $$? -eq 0 ]; then \
		echo "✓ Preprocess completed successfully"; \
	else \
		echo "✗ Preprocess failed"; \
		exit 1; \
	fi
	@touch $@

$(ANALYSIS_OUTPUT): $(PREPROC_OUTPUT) $(ANALYSIS_PY) $(ANALYSIS_SH)
	@echo "Running analysis script..."
	@export OUTPUT_DIR="$(TRIAL_DIR)/outputs" && bash $(abspath $(ANALYSIS_SH))
	@if [ $$? -eq 0 ]; then \
		echo "✓ Analysis completed successfully"; \
	else \
		echo "✗ Analysis failed"; \
		exit 1; \
	fi
	@touch $@

# Convenient aliases
simulate-without: $(SIM_WITHOUT_OUTPUT)
copy-output: $(COPY_OUTPUT)
simulate-with: $(SIM_WITH_OUTPUT)
check: $(CHECK_OUTPUT)
preprocess: $(PREPROC_OUTPUT)
analysis: $(ANALYSIS_OUTPUT)

# Clean output files but preserve directory structure
clean-output:
	@echo "Cleaning output files..."
	@find $(TRIAL_DIR) -name "*.done" -delete 2>/dev/null || true
	@find $(TRIAL_DIR) -name "*.root" -delete 2>/dev/null || true
	@find $(TRIAL_DIR) -name "*.log" -delete 2>/dev/null || true
	@find $(TRIAL_DIR) -name "*.txt" -delete 2>/dev/null || true
	@find $(TRIAL_DIR) -name "workflow.json" -delete 2>/dev/null || true
	@find $(TRIAL_DIR) -type d -name "tf*" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned output files from $(TRIAL_DIR)"

# Complete clean (remove everything including directories)
clean:
	@echo "Removing entire trial directory..."
	@rm -rf $(TRIAL_DIR)
	@echo "Removed $(TRIAL_DIR)"

# Debug target to show variables and check environment
debug:
	@echo "=== Configuration Variables ==="
	@echo "NAME: $(NAME)"
	@echo "TRIAL_DIR: $(TRIAL_DIR)"
	@echo "OUTPUT_WITHOUT: $(OUTPUT_WITHOUT)"
	@echo "OUTPUT_WITH: $(OUTPUT_WITH)"
	@echo "SCRIPT_DIR: $(SCRIPT_DIR)"
	@echo ""
	@echo "=== Simulation Parameters ==="
	@echo "NWORKERS: $(NWORKERS)"
	@echo "NSIGEVENTS: $(NSIGEVENTS)"
	@echo "NTIMEFRAMES: $(NTIMEFRAMES)"
	@echo "SPLITID: $(SPLITID)"
	@echo "SIMENGINE: $(SIMENGINE)"
	@echo ""
	@echo "=== Environment Check ==="
	@if [ -n "$(O2DPG_ROOT)" ]; then echo "O2DPG_ROOT: $(O2DPG_ROOT)"; else echo "O2DPG_ROOT: NOT SET"; fi
	@if [ -n "$(O2_ROOT)" ]; then echo "O2_ROOT: $(O2_ROOT)"; else echo "O2_ROOT: NOT SET"; fi
	@echo ""
	@echo "=== File Status ==="
	@if [ -f "$(SIM_SCRIPT)" ]; then echo "Simulation script: EXISTS"; else echo "Simulation script: MISSING"; fi
	@if [ -d "$(OUTPUT_WITHOUT)" ]; then echo "Without shared clusters dir: EXISTS"; else echo "Without shared clusters dir: MISSING"; fi
	@if [ -d "$(OUTPUT_WITH)" ]; then echo "With shared clusters dir: EXISTS"; else echo "With shared clusters dir: MISSING"; fi
	@if [ -f "$(SIM_WITHOUT_OUTPUT)" ]; then echo "Simulation without: COMPLETED"; else echo "Simulation without: PENDING"; fi
	@if [ -f "$(COPY_OUTPUT)" ]; then echo "Copy output: COMPLETED"; else echo "Copy output: PENDING"; fi
	@if [ -f "$(SIM_WITH_OUTPUT)" ]; then echo "Simulation with: COMPLETED"; else echo "Simulation with: PENDING"; fi
	@echo ""
	@echo "=== Directory Structure ==="
	@if [ -d "$(TRIAL_DIR)" ]; then \
		echo "Trial directory contents:"; \
		find $(TRIAL_DIR) -type f -name "*.done" 2>/dev/null || echo "No .done files found"; \
	else \
		echo "Trial directory does not exist yet"; \
	fi

# Status check
status:
	@echo "=== Simulation Status ==="
	@if [ -f "$(SIM_WITHOUT_OUTPUT)" ]; then \
		echo "✓ Simulation without shared clusters: COMPLETED"; \
	else \
		echo "✗ Simulation without shared clusters: PENDING"; \
	fi
	@if [ -f "$(COPY_OUTPUT)" ]; then \
		echo "✓ Copy output: COMPLETED"; \
	else \
		echo "✗ Copy output: PENDING"; \
	fi
	@if [ -f "$(SIM_WITH_OUTPUT)" ]; then \
		echo "✓ Simulation with shared clusters: COMPLETED"; \
	else \
		echo "✗ Simulation with shared clusters: PENDING"; \
	fi
	@echo ""
	@if [ -f "$(SIM_WITHOUT_OUTPUT)" ] && [ -f "$(COPY_OUTPUT)" ] && [ -f "$(SIM_WITH_OUTPUT)" ]; then \
		echo "🎉 All simulations completed successfully!"; \
	else \
		echo "⏳ Simulations in progress or pending..."; \
	fi

# Validate environment before running
validate-environment:
	@echo "=== Environment Validation ==="
	@if [ -z "$(O2DPG_ROOT)" ]; then \
		echo "❌ Error: O2DPG_ROOT not set"; \
		exit 1; \
	fi
	@if [ -z "$(O2_ROOT)" ]; then \
		echo "❌ Error: O2_ROOT not set"; \
		exit 1; \
	fi
	@if [ ! -f "$(SIM_SCRIPT)" ]; then \
		echo "❌ Error: $(SIM_SCRIPT) not found"; \
		exit 1; \
	fi
	@echo "O2DPG_ROOT = $(O2DPG_ROOT)"
	@ls -l "$(O2DPG_ROOT)/MC/bin/o2dpg_sim_workflow.py"
	@if [ ! -f "$(O2DPG_ROOT)/MC/bin/o2dpg_sim_workflow.py" ]; then \
		echo "❌ Error: o2dpg_sim_workflow.py not found or not executable"; \
		exit 1; \
	fi
	@if [ ! -x "$(O2_ROOT)/bin/o2-its-reco-workflow" ]; then \
		echo "❌ Error: o2-its-reco-workflow not found or not executable"; \
		exit 1; \
	fi
	@echo "✅ Environment validation passed"

# Force re-run of shared clusters simulation (useful for debugging)
force-simulate-with:
	@rm -f $(SIM_WITH_OUTPUT)
	@$(MAKE) simulate-with

# Check if outputs exist and are valid
validate-outputs:
	@echo "=== Output Validation ==="
	@if [ -f "$(SIM_WITHOUT_OUTPUT)" ] && [ -d "$(OUTPUT_WITHOUT)/tf1" ]; then \
		echo "✅ Simulation without shared clusters: Valid"; \
	else \
		echo "❌ Simulation without shared clusters: Invalid or missing"; \
	fi
	@if [ -f "$(COPY_OUTPUT)" ] && [ -d "$(OUTPUT_WITH)/tf1" ]; then \
		echo "✅ Copy output: Valid"; \
	else \
		echo "❌ Copy output: Invalid or missing"; \
	fi
	@if [ -f "$(SIM_WITH_OUTPUT)" ] && [ -f "$(OUTPUT_WITH)/tf1/shared_clusters_log.txt" ]; then \
		echo "✅ Simulation with shared clusters: Valid"; \
	else \
		echo "❌ Simulation with shared clusters: Invalid or missing"; \
	fi