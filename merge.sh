#!/bin/bash

# --- Configuration ---
# Directory where the user has downloaded all files (split parts and original small files)
# Defaulting to current directory. Users might need to change this.
DOWNLOAD_DIR="."
# Directory where restored original files will be placed
RESTORED_DIR="${DOWNLOAD_DIR}/RESTORED_ORIGINAL_FILES"
# Suffix base used during splitting (e.g., .sfpart_ which creates .sfpart_aa, .sfpart_ab)
SPLIT_PART_SUFFIX_BASE=".sfpart_"

# --- Script Start ---
echo "Starting file merging process..."
echo "Download Directory (containing parts and small files): ${DOWNLOAD_DIR}"
echo "Restored Files will be placed in: ${RESTORED_DIR}"
echo ""

if [ ! -d "${DOWNLOAD_DIR}" ]; then
    echo "Error: Download directory '${DOWNLOAD_DIR}' does not exist."
    exit 1
fi

mkdir -p "${RESTORED_DIR}"
if [ $? -ne 0 ]; then
    echo "Error: Could not create restoration directory '${RESTORED_DIR}'."
    exit 1
fi

# Flag to track if any files were processed
processed_any_files=false

# Process files that were split
# Find the first part of any split sequence (e.g., filename.sfpart_aa)
find "${DOWNLOAD_DIR}" -maxdepth 1 -type f -name "*${SPLIT_PART_SUFFIX_BASE}aa" | while IFS= read -r first_part_file; do
    processed_any_files=true
    # Extract the original filename before the ".sfpart_aa"
    # Example: /path/to/data.tar.gz.sfpart_aa -> data.tar.gz
    original_filename=$(basename "${first_part_file}" "${SPLIT_PART_SUFFIX_BASE}aa")
    restored_filepath="${RESTORED_DIR}/${original_filename}"

    echo "Found split file sequence for: ${original_filename}"
    echo "  Merging parts to: ${restored_filepath}"

    # Concatenate all parts for this original file.
    # The shell's glob expansion `*` will sort `_aa, _ab, ...` correctly.
    cat "${DOWNLOAD_DIR}/${original_filename}${SPLIT_PART_SUFFIX_BASE}"* > "${restored_filepath}"

    if [ $? -eq 0 ]; then
        echo "  Successfully merged: ${original_filename} (Size: $(du -h "${restored_filepath}" | cut -f1))"
        # Optional: Offer to delete parts after successful merge
        # read -p "Delete original parts for '${original_filename}'? (yes/No): " delete_confirm
        # if [[ "${delete_confirm,,}" == "yes" ]]; then
        #     rm "${DOWNLOAD_DIR}/${original_filename}${SPLIT_PART_SUFFIX_BASE}"*
        #     echo "  Deleted parts for '${original_filename}'."
        # fi
    else
        echo "  Error merging parts for '${original_filename}'. Restored file might be incomplete or corrupted."
        rm -f "${restored_filepath}" # Remove potentially corrupt file
    fi
    echo "----------------------------------------"
done

# Copy over files that were not split (i.e., do not have the SPLIT_PART_SUFFIX_BASE)
echo ""
echo "Copying original small files (if any) that were not split..."
find "${DOWNLOAD_DIR}" -maxdepth 1 -type f ! -name "*${SPLIT_PART_SUFFIX_BASE}*" | while IFS= read -r original_small_file_path; do
    processed_any_files=true
    original_small_filename=$(basename "${original_small_file_path}")
    # Avoid copying if it's already in RESTORED_DIR (e.g. from a previous run or if it's the script itself)
    if [ "${original_small_file_path}" = "${RESTORED_DIR}/${original_small_filename}" ]; then
        continue
    fi
    # Also avoid copying the script itself if it's in DOWNLOAD_DIR
    if [ "$(basename "$0")" = "${original_small_filename}" ] && [ "$(realpath "$0")" = "$(realpath "${original_small_file_path}")" ]; then
        echo "  Skipping copying the merge script itself: ${original_small_filename}"
        continue
    fi


    echo "  Copying small file: ${original_small_filename} to ${RESTORED_DIR}/"
    cp "${original_small_file_path}" "${RESTORED_DIR}/"
done


if ! ${processed_any_files}; then
    echo "No files found to process in '${DOWNLOAD_DIR}' (no split parts or other files detected)."
fi

echo ""
echo "File merging and copying complete."
echo "All restored and original small files should now be in:"
echo "${RESTORED_DIR}"