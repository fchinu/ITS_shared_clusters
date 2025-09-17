#!/bin/bash
set -euo pipefail

# ======= Configuration =======
OUTPUT_DIR=${OUTPUT_DIR:-"output"}
OUTPUTS_DIR="${OUTPUT_DIR}/outputs"
PARTIAL_OUTPUTS_DIR="${OUTPUTS_DIR}/partial"
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
echo "OUTPUTS_DIR: ${OUTPUTS_DIR}"
echo "PARTIAL_OUTPUTS_DIR: ${PARTIAL_OUTPUTS_DIR}"
echo "Script directory: ${SCRIPT_DIR}"
echo "======================================="

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUTS_DIR}"
mkdir -p "${PARTIAL_OUTPUTS_DIR}"

# ======= Function to process a folder =======
run_macro_in_dir() {
    local variant=$1  # "with_shared_clusters" or "without_shared_clusters"
    local tf_folder=$2  # e.g., "tf1", "tf2", etc.
    local dest_dir="${OUTPUT_DIR}/${variant}/${tf_folder}"
    local output_name_pdf="fakeClusters_${variant}_${tf_folder}.pdf"
    local output_name_root="CheckTracksCA${variant}_${tf_folder}.root"

    mkdir -p "${PARTIAL_OUTPUTS_DIR}/${variant}"

    echo "Processing: ${variant} in ${tf_folder}"
    echo "Target directory: ${dest_dir}"

    if [ ! -d "${dest_dir}" ]; then
        echo "⚠️  Warning: ${dest_dir} does not exist, skipping..."
        return 1
    fi

    cp "${SCRIPT_DIR}/${MACRO_NAME}" "${dest_dir}/"

    pushd "${dest_dir}" > /dev/null
    root -l -b -q "${MACRO_NAME}"
    popd > /dev/null

    if [ ! -f "${dest_dir}/${OUTPUT_NAME_PDF}" ]; then
        echo "⚠️  Warning: ${OUTPUT_NAME_PDF} not generated in ${dest_dir}"
        return 1
    fi

    if [ ! -f "${dest_dir}/${OUTPUT_NAME_ROOT}" ]; then
        echo "⚠️  Warning: ${OUTPUT_NAME_ROOT} not generated in ${dest_dir}"
        return 1
    fi

    # Move outputs to the unmerged outputs directory
    mv "${dest_dir}/${OUTPUT_NAME_PDF}" "${PARTIAL_OUTPUTS_DIR}/${variant}/${output_name_pdf}"
    mv "${dest_dir}/${OUTPUT_NAME_ROOT}" "${PARTIAL_OUTPUTS_DIR}/${variant}/${output_name_root}"
    rm -f "${dest_dir}/${MACRO_NAME}"
    echo "✅ Output moved to ${PARTIAL_OUTPUTS_DIR}/${variant}/${output_name_pdf}"
    
    return 0
}

# ======= Find all tf* folders and process them =======
process_variant() {
    local variant=$1
    local variant_dir="${OUTPUT_DIR}/${variant}"
    local root_files=()
    local processed_count=0
    
    echo "--- Processing variant: ${variant} ---"
    
    if [ ! -d "${variant_dir}" ]; then
        echo "⚠️  Warning: ${variant_dir} does not exist, skipping variant..."
        return
    fi
    
    # Find all tf* directories in the variant directory
    for tf_dir in "${variant_dir}"/tf*; do
        if [ -d "$tf_dir" ]; then
            tf_folder=$(basename "$tf_dir")
            echo "Found tf folder: ${tf_folder}"

            if run_macro_in_dir "${variant}" "${tf_folder}"; then
                root_files+=("${PARTIAL_OUTPUTS_DIR}/CheckTracksCA${variant}_${tf_folder}.root")
                processed_count=$((processed_count + 1))
            fi
        fi
    done
    
    if [ ${processed_count} -eq 0 ]; then
        echo "⚠️  Warning: No tf* folders were successfully processed for ${variant}"
        return
    fi
    
    echo "Successfully processed ${processed_count} tf* folders for ${variant}"
}

# ======= Run both variants =======
# Uncomment the line below if you want to process with_shared_clusters
process_variant "with_shared_clusters"
process_variant "without_shared_clusters"
python3 tests/compare_efficiency_fake.py --file_with "${OUTPUTS_DIR}/CheckTracksCAwith_shared_clusters.root" --file_without "${OUTPUTS_DIR}/CheckTracksCAwithout_shared_clusters.root" --output "${OUTPUTS_DIR}/comparison.pdf"

cd "${ORIGINAL_DIR}"