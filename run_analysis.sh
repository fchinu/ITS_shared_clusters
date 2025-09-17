#!/bin/bash
set -euo pipefail

# ======= Configuration =======
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
VENV_PATH="${HOME}/.venv/ml"
DRAW_SHARED_SCRIPT="${HOME}/ITS_sharedclusters/macros/draw_shared.py"
STUDY_DOUBLY_RECO_SCRIPT="${HOME}/ITS_sharedclusters/macros/study_doubly_reco.py"

# ======= Environment Validation =======
[ ! "${O2DPG_ROOT:-}" ] && echo "❌ Error: O2DPG_ROOT not set. Load the O2DPG environment." && exit 1
[ ! "${O2_ROOT:-}" ] && echo "❌ Error: O2_ROOT not set. Load the O2 environment." && exit 1

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ORIGINAL_DIR="$(pwd)"

echo "=== Pre-processing ROOT → Parquet ==="
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "VENV_PATH : ${VENV_PATH}"
echo "Script dir: ${SCRIPT_DIR}"
echo "======================================="

# ======= Function to draw_shared with venv =======
draw_shared() {
    local venv_path=$1
    local script_path=$2
    local input_file=$3
    local output_file=$4
    local all=$5

    # Save O2 Python env
    OLD_PYTHONHOME="${PYTHONHOME:-}"
    OLD_PYTHONPATH="${PYTHONPATH:-}"

    # Unset them for venv
    unset PYTHONHOME
    unset PYTHONPATH

    echo "➡️  Running ${script_path}"

    if [ ! -f "${input_file}" ]; then
        echo "❌ Error: Input file ${input_file} does not exist"
        exit 1
    fi

    if [ ! -d "${venv_path}" ]; then
        echo "❌ Error: Virtual environment ${venv_path} does not exist"
        exit 1
    fi

    if [ "$all" = "true" ]; then
        "${venv_path}/bin/python3" "${script_path}" "${input_file}" "${output_file}" --all
    else
        "${venv_path}/bin/python3" "${script_path}" "${input_file}" "${output_file}"
    fi

    # Restore O2 Python env
    if [ -n "${OLD_PYTHONHOME}" ]; then
        export PYTHONHOME="${OLD_PYTHONHOME}"
    else
        unset PYTHONHOME
    fi

    if [ -n "${OLD_PYTHONPATH}" ]; then
        export PYTHONPATH="${OLD_PYTHONPATH}"
    else
        unset PYTHONPATH
    fi

    echo "✅ Done: ${output_file}"
}

study_doubly_reco() {
    local venv_path=$1
    local script_path=$2
    local input_file=$3
    local output_file=$4

    # Save O2 Python env
    OLD_PYTHONHOME="${PYTHONHOME:-}"
    OLD_PYTHONPATH="${PYTHONPATH:-}"

    # Unset them for venv
    unset PYTHONHOME
    unset PYTHONPATH

    echo "➡️  Running ${script_path}"

    if [ ! -f "${input_file}" ]; then
        echo "❌ Error: Input file ${input_file} does not exist"
        exit 1
    fi

    if [ ! -d "${venv_path}" ]; then
        echo "❌ Error: Virtual environment ${venv_path} does not exist"
        exit 1
    fi

    "${venv_path}/bin/python3" "${script_path}" "${input_file}" "${output_file}"

    # Restore O2 Python env
    if [ -n "${OLD_PYTHONHOME}" ]; then
        export PYTHONHOME="${OLD_PYTHONHOME}"
    else
        unset PYTHONHOME
    fi

    if [ -n "${OLD_PYTHONPATH}" ]; then
        export PYTHONPATH="${OLD_PYTHONPATH}"
    else
        unset PYTHONPATH
    fi

    echo "✅ Done: ${output_file}"
}

# ======= Run scripts =======
draw_shared "${VENV_PATH}" "${DRAW_SHARED_SCRIPT}" \
    "${OUTPUT_DIR}/without_shared_clusters/CheckTracksCAwithout_shared_clusters.parquet" \
    "${OUTPUT_DIR}/without_shared_clusters/CheckTracksCAwithout_shared_clusters_all.pdf" \
    true

draw_shared "${VENV_PATH}" "${DRAW_SHARED_SCRIPT}" \
    "${OUTPUT_DIR}/with_shared_clusters/CheckTracksCAwith_shared_clusters.parquet" \
    "${OUTPUT_DIR}/with_shared_clusters/CheckTracksCAwith_shared_clusters.pdf" \
    false

draw_shared "${VENV_PATH}" "${DRAW_SHARED_SCRIPT}" \
    "${OUTPUT_DIR}/with_shared_clusters/CheckTracksCAwith_shared_clusters.parquet" \
    "${OUTPUT_DIR}/with_shared_clusters/CheckTracksCAwith_shared_clusters_all.pdf" \
    true

study_doubly_reco "${VENV_PATH}" "${STUDY_DOUBLY_RECO_SCRIPT}" \
    "${OUTPUT_DIR}/with_shared_clusters/CheckTracksCAwith_shared_clusters.parquet" \
    "${OUTPUT_DIR}/with_shared_clusters/CheckTracksCAwith_shared_clusters_doubly_reco.pdf"

study_doubly_reco "${VENV_PATH}" "${STUDY_DOUBLY_RECO_SCRIPT}" \
    "${OUTPUT_DIR}/without_shared_clusters/CheckTracksCAwithout_shared_clusters.parquet" \
    "${OUTPUT_DIR}/without_shared_clusters/CheckTracksCAwithout_shared_clusters_doubly_reco.pdf"

# ======= Done =======
echo "🎉 Pre-processing finished"
cd "${ORIGINAL_DIR}"
