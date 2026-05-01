#!/bin/bash
set -euo pipefail
# run_streaming_inference_ablations.sh
# -----------------------------------------------------------------------------
# Batch launcher for streaming inference ablations across trained LoRA runs.
# -----------------------------------------------------------------------------

export PYTHONPATH="${PYTHONPATH:-}:."

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CSV_PATH="${CSV_PATH:-../filtered_songs.csv}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-../output/songdescriber_filtered}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-acestep-v15-xl-turbo}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
PREDICTION_SECONDS="${PREDICTION_SECONDS:-30}"
MAX_DISTILL_SECONDS="${MAX_DISTILL_SECONDS:-150}"
AUDIO_FORMAT="${AUDIO_FORMAT:-wav}"
USE_TILED_DECODE="${USE_TILED_DECODE:-1}"

LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
    LIMIT_ARGS+=(--limit "${LIMIT}")
fi

TILED_ARGS=()
if [[ "${USE_TILED_DECODE}" == "1" ]]; then
    TILED_ARGS+=(--use-tiled-decode)
fi

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

MODEL_CHUNKS=(
    "5"
    "5"
    "5"
    "1"
)

for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    lora_path="${MODEL_PATHS[$idx]}"
    max_distill_chunks="${MODEL_CHUNKS[$idx]}"
    output_dir="${BASE_OUTPUT_DIR}/streaming_${model_name}_step_1"

    echo "============================================================"
    echo "Running ablation: ${model_name}"
    echo "LoRA: ${lora_path}"
    echo "Output: ${output_dir}"
    echo "Chunks: ${max_distill_chunks}"
    echo "============================================================"

    python streaming_infer_lora_csv.py \
        --csv-path "${CSV_PATH}" \
        --output-dir "${output_dir}" \
        --id-column "caption_id" \
        --caption-column "caption" \
        --lyrics "[Instrumental]" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --config-path "${CONFIG_PATH}" \
        --lora-path "${lora_path}" \
        --lora-scale 1.0 \
        --device "${DEVICE}" \
        --seed "${SEED}" \
        --warmup-seconds "${WARMUP_SECONDS}" \
        --prediction-seconds "${PREDICTION_SECONDS}" \
        --max-distill-seconds "${MAX_DISTILL_SECONDS}" \
        --max-distill-chunks "${max_distill_chunks}" \
        --audio-format "${AUDIO_FORMAT}" \
        "${TILED_ARGS[@]}" \
        "${LIMIT_ARGS[@]}"
done

echo ">>> All streaming ablation inference runs completed."
