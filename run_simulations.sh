#!/bin/bash

#
# A example workflow MC->RECO->AOD for a simple pp min bias
# production, targetting test beam conditions.

# make sure O2DPG + O2 is loaded
[ ! "${O2DPG_ROOT}" ] && echo "Error: This needs O2DPG loaded" && exit 1
[ ! "${O2_ROOT}" ] && echo "Error: This needs O2 loaded" && exit 1

# ----------- LOAD UTILITY FUNCTIONS --------------------------
. ${O2_ROOT}/share/scripts/jobutils.sh

# ----------- START ACTUAL JOB  -----------------------------

# Use current directory if OUTPUT_DIR is not set or is "."
OUTPUT_DIR=${OUTPUT_DIR:-"default"}
NWORKERS=${NWORKERS:-20}
MODULES="--skipModules ZDC"
SIMENGINE=${SIMENGINE:-TGeant4}
NSIGEVENTS=${NSIGEVENTS:-20000}
NTIMEFRAMES=${NTIMEFRAMES:-1}
ISPP=${ISPP:-true}
[[ ${SPLITID} != "" ]] && SEED_VALUE="${SPLITID}" || SEED_VALUE="42"
SHARED_CLUSTERS=${SHARED_CLUSTERS:-false}
LOW_FIELD=${LOW_FIELD:-false}

echo "=== Simulation Parameters ==="
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NWORKERS: ${NWORKERS}"
echo "NSIGEVENTS: ${NSIGEVENTS}"
echo "NTIMEFRAMES: ${NTIMEFRAMES}"
echo "SIMENGINE: ${SIMENGINE}"
echo "SHARED_CLUSTERS: ${SHARED_CLUSTERS}"
echo "LOW_FIELD: ${LOW_FIELD}"
echo "SEED: ${SEED_VALUE}"
echo "=========================="

# Store original directory
ORIGINAL_DIR=$(pwd)

# If OUTPUT_DIR is "." or current directory, work in place
if [ "${OUTPUT_DIR}" = "." ]; then
    echo "Working in current directory: $(pwd)"
    WORK_DIR="."
else
    echo "Creating and changing to directory: ${OUTPUT_DIR}"
    mkdir -p ${OUTPUT_DIR}
    cd ${OUTPUT_DIR}
    WORK_DIR=$(pwd)
fi

echo "Working directory: ${WORK_DIR}"

# Generate workflow
echo "Generating simulation workflow..."

# Initialize parameter array
params=()

# Set collision-specific parameters
if [ "${ISPP}" = "true" ]; then
    params+=(-eCM 13600 -col pp -gen pythia8pp)
else
    params+=(-eCM 5360 -col PbPb -gen pythia8)
fi

# Add common parameters
params+=(-j "${NWORKERS}")
params+=(-ns "${NSIGEVENTS}")
params+=(-tf "${NTIMEFRAMES}")
params+=(-interactionRate 500000)
params+=(-confKey "Diamond.width[2]=6.")
params+=(-e "${SIMENGINE}")
params+=(-seed "${SEED_VALUE}")
params+=(-mod "--skipModules ZDC")

# Add conditional parameters
if [ "${SHARED_CLUSTERS}" = "true" ]; then
    params+=(--its-first-cluster-sharing)
fi

if [ "${LOW_FIELD}" = "true" ]; then
    params+=(-field 2)
fi

# Execute with all parameters
"${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py" "${params[@]}"

# Check if workflow was generated successfully
if [ ! -f "workflow.json" ]; then
    echo "Error: workflow.json not generated"
    exit 1
fi

# Run workflow
echo "Running simulation workflow..."
if [ "${SHARED_CLUSTERS}" = "false" ]; then
    ${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt aod --cpu-limit 32
else
    ${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt aod --cpu-limit 32 --rerun-from itsreco_*
fi

if [ $? -ne 0 ]; then
    echo "Error: Workflow execution failed"
    exit 1
fi

# Return to original directory
cd ${ORIGINAL_DIR}

echo "Simulation script completed successfully in: ${WORK_DIR}"