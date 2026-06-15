#!/bin/bash
# --predictions_dir for stats.py.
INPUT_FOLDERS=(
    "predictions_cnn2plus1d"
    "predictions_cnn2plus1dfilters"
    "predictions_cnn2plus1dlayers"
    "predictions_cnn2plus1dlight"
    "predictions_faceshape"
    "predictions_lipshape"
)

INPUT_BASE="/Data/data"                       # parent dir holding the input folders
OUTPUT_BASE="/Data/data/stats"                # results land in OUTPUT_BASE/<folder>/
GROUNDTRUTH_CSV="/Data/data/csv/val_orig.csv" # master ground-truth CSV
STATS="stats.py"
PYTHON="${PYTHON:-python3}"

# ── run ─────────────────────────────────────────────────────────────────────

cd "$(dirname "$0")"

for folder in "${INPUT_FOLDERS[@]}"; do
    input_dir="${INPUT_BASE}/${folder}"
    result_dir="${OUTPUT_BASE}/${folder}"

    if [[ ! -d "$input_dir" ]]; then
        echo "[WARN] skipping '$folder': input dir not found ($input_dir)" >&2
        continue
    fi

    mkdir -p "$result_dir"
    echo "[INFO] processing folder '$folder' -> $result_dir"

    "$PYTHON" "$STATS" \
        --predictions_dir "$input_dir" \
        --groundtruth_csv "$GROUNDTRUTH_CSV" \
        --result_dir "$result_dir" \
        --workers 8
done

echo "[INFO] done."
