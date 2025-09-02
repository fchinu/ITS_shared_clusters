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
[[ ${SPLITID} != "" ]] && SEED="-seed ${SPLITID}" || SEED="-seed 42"
SHARED_CLUSTERS=${SHARED_CLUSTERS:-false}

echo "=== Simulation Parameters ==="
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "NWORKERS: ${NWORKERS}"
echo "NSIGEVENTS: ${NSIGEVENTS}"
echo "NTIMEFRAMES: ${NTIMEFRAMES}"
echo "SIMENGINE: ${SIMENGINE}"
echo "SHARED_CLUSTERS: ${SHARED_CLUSTERS}"
echo "SEED: ${SEED}"
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
if [ "${ISPP}" = "true" ]; then
    ${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py \
        -eCM 13600 \
        -col pp \
        -gen pythia8pp \
        -j ${NWORKERS} \
        -ns ${NSIGEVENTS} \
        -tf ${NTIMEFRAMES} \
        -interactionRate 500000 \
        -confKey "Diamond.width[2]=6." \
        -e ${SIMENGINE} \
        ${SEED} \
        -mod "--skipModules ZDC"
else
    ${O2DPG_ROOT}/MC/bin/o2dpg_sim_workflow.py \
        -eCM 5360 \
        -col PbPb \
        -gen pythia8 \
        -j ${NWORKERS} \
        -ns ${NSIGEVENTS} \
        -tf ${NTIMEFRAMES} \
        -interactionRate 500000 \
        -confKey "Diamond.width[2]=6." \
        -e ${SIMENGINE} \
        ${SEED} \
        -mod "--skipModules ZDC"
fi

# Check if workflow was generated successfully
if [ ! -f "workflow.json" ]; then
    echo "Error: workflow.json not generated"
    exit 1
fi

# Run workflow
echo "Running simulation workflow..."
${O2DPG_ROOT}/MC/bin/o2_dpg_workflow_runner.py -f workflow.json -tt aod --cpu-limit 32

# Check if simulation completed successfully
if [ ! -d "tf1" ]; then
    echo "Error: tf1 directory not created - simulation may have failed"
    exit 1
fi

# Check shared clusters
if [ "${SHARED_CLUSTERS}" = "true" ]; then
    echo "Running with shared clusters enabled..."
    cd tf1
    ${O2_ROOT}/bin/o2-its-reco-workflow \
        -b --run \
        --condition-not-after 3385078236000 \
        --trackerCA \
        --tracking-mode async \
        --configKeyValues "HBFUtils.orbitFirstSampled=256;HBFUtils.nHBFPerTF=32;HBFUtils.orbitFirst=256;HBFUtils.runNumber=300000;HBFUtils.startTime=1546300800000;ITSVertexerParam.phiCut=0.5;ITSVertexerParam.clusterContributorsCut=3;ITSVertexerParam.tanLambdaCut=0.2;NameConf.mDirMatLUT=..;ITSCATrackerParam.allowSharingFirstCluster=true" \
        > shared_clusters_log.txt 2>&1
    
    if [ $? -eq 0 ]; then
        echo "Shared clusters simulation completed successfully"
    else
        echo "Error: Shared clusters simulation failed"
        exit 1
    fi
    cd ..
fi

# Return to original directory
cd ${ORIGINAL_DIR}

echo "Simulation script completed successfully in: ${WORK_DIR}"