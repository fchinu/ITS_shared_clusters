#!/bin/bash
set -euo pipefail

# ======= Configuration =======
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
MACRO_NAME="CheckTracksCA.C"
OUTPUT_NAME_PDF="fakeClusters.pdf"
OUTPUT_NAME_ROOT="CheckTracksCA.root"

# ======= Environment Validation =======
[ ! "${O2DPG_ROOT:-}" ] && echo "❌ Error: O2DPG_ROOT not set. Load the O2DPG environment." && exit 1
[ ! "${O2_ROOT:-}" ] && echo "❌ Error: O2_ROOT not set. Load the O2 environment." && exit 1

# ======= Setup =======
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ORIGINAL_DIR="$(pwd)"

echo "=== Shared Clusters Plot Generation ==="
echo "OUTPUT_DIR: ${OUTPUT_DIR}"
echo "Script directory: ${SCRIPT_DIR}"
echo "======================================="

# ======= Function to process a folder =======
run_macro_in_dir() {
    local variant=$1  # "with_shared_clusters" or "without_shared_clusters"
    local dest_dir="${OUTPUT_DIR}/${variant}/tf1"
    local output_name_pdf="fakeClusters_${variant}.pdf"
    local output_name_root="CheckTracksCA${variant}.root"

    echo "Processing: ${variant}"
    echo "Target directory: ${dest_dir}"

    if [ ! -d "${dest_dir}" ]; then
        echo "❌ Error: ${dest_dir} does not exist"
        exit 1
    fi

    cp "${SCRIPT_DIR}/${MACRO_NAME}" "${dest_dir}/"

    pushd "${dest_dir}" > /dev/null
    root -l -b -q "${MACRO_NAME}"
    popd > /dev/null

    if [ ! -f "${dest_dir}/${OUTPUT_NAME_PDF}" ]; then
        echo "❌ Error: ${OUTPUT_NAME_PDF} not generated in ${dest_dir}"
        exit 1
    fi

    if [ ! -f "${dest_dir}/${OUTPUT_NAME_ROOT}" ]; then
        echo "❌ Error: ${OUTPUT_NAME_ROOT} not generated in ${dest_dir}"
        exit 1
    fi

    mv "${dest_dir}/${OUTPUT_NAME_PDF}" "${OUTPUT_DIR}/${output_name_pdf}"
    mv "${dest_dir}/${OUTPUT_NAME_ROOT}" "${OUTPUT_DIR}/${output_name_root}"
    rm -f "${dest_dir}/${MACRO_NAME}"
    echo "✅ Output moved to ${OUTPUT_DIR}/${output_name_pdf}"
}

# ======= Run both variants =======
run_macro_in_dir "with_shared_clusters"
run_macro_in_dir "without_shared_clusters"
python3 compare_efficiency_fake.py --file_with "${OUTPUT_DIR}/CheckTracksCAwith_shared_clusters.root" --file_without "${OUTPUT_DIR}/CheckTracksCAwithout_shared_clusters.root" --output "${OUTPUT_DIR}/comparison.pdf"

# ======= Done =======
echo "🎉 All plots generated and copied to ${OUTPUT_DIR}"
cd "${ORIGINAL_DIR}"

# without_shared_cluster
# ** Some statistics:
#         - Total number of tracks: 24163
#         - Total number of tracks not corresponding to particles: 1902 (7.87154%)
#         - Total number of fakes: 863 (3.57158%)
#         - Total number of good: 21398 (88.5569%)
# with_shared_clusters
# ** Some statistics:
#         - Total number of tracks: 24174
#         - Total number of tracks not corresponding to particles: 1913 (7.91346%)
#         - Total number of fakes: 863 (3.56995%)
#         - Total number of good: 21398 (88.5166%)