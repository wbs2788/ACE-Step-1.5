#!/bin/bash
set -euo pipefail
# run_latency_benchmarks.sh
# -----------------------------------------------------------------------------
# Latency benchmark runner for real-time music generation claims.
# -----------------------------------------------------------------------------

export PYTHONPATH="${PYTHONPATH:-}:."

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CSV_PATH="${CSV_PATH:-../filtered_songs.csv}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-../output/songdescriber_filtered/latency}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-acestep-v15-xl-turbo}"
DEVICE="${DEVICE:-cuda}"
LIMIT="${LIMIT:-10}"
REPEATS="${REPEATS:-3}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"

STANDARD_STEP_LIST="${STANDARD_STEP_LIST:-1 2 4 8}"
STREAMING_CHUNK_LIST="${STREAMING_CHUNK_LIST:-1 3 5}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
PREDICTION_SECONDS="${PREDICTION_SECONDS:-30}"
MAX_DISTILL_SECONDS="${MAX_DISTILL_SECONDS:-150}"

mkdir -p "${BASE_OUTPUT_DIR}"

MODEL_NAMES=(
    "mse"
    "fft"
    "v1"
    "v1_chunk1"
)

MODEL_PATHS=(
    "./output/xl_consistency_mse_gpu1/final"
    "./output/xl_consistency_fft_gpu0/final"
    "./output/xl_consistency_v1/final"
    "./output/xl_consistency_v1_chunk_1/final"
)

for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    lora_path="${MODEL_PATHS[$idx]}"

    for step in ${STANDARD_STEP_LIST}; do
        python benchmark_latency.py \
            --mode standard \
            --csv-path "${CSV_PATH}" \
            --output-csv "${BASE_OUTPUT_DIR}/${model_name}_standard_step_${step}.csv" \
            --lora-path "${lora_path}" \
            --experiment-name "${model_name}_standard_step_${step}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --config-path "${CONFIG_PATH}" \
            --device "${DEVICE}" \
            --limit "${LIMIT}" \
            --repeats "${REPEATS}" \
            --warmup-runs "${WARMUP_RUNS}" \
            --inference-steps "${step}"
    done

    for chunk in ${STREAMING_CHUNK_LIST}; do
        python benchmark_latency.py \
            --mode streaming \
            --csv-path "${CSV_PATH}" \
            --output-csv "${BASE_OUTPUT_DIR}/${model_name}_streaming_chunk_${chunk}.csv" \
            --lora-path "${lora_path}" \
            --experiment-name "${model_name}_streaming_chunk_${chunk}" \
            --checkpoint-dir "${CHECKPOINT_DIR}" \
            --config-path "${CONFIG_PATH}" \
            --device "${DEVICE}" \
            --limit "${LIMIT}" \
            --repeats "${REPEATS}" \
            --warmup-runs "${WARMUP_RUNS}" \
            --warmup-seconds "${WARMUP_SECONDS}" \
            --prediction-seconds "${PREDICTION_SECONDS}" \
            --max-distill-seconds "${MAX_DISTILL_SECONDS}" \
            --max-distill-chunks "${chunk}" \
            --use-tiled-decode
    done
done

echo ">>> Latency benchmarks completed."
