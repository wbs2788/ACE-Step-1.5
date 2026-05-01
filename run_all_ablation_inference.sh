#!/bin/bash
set -euo pipefail
# run_all_ablation_inference.sh
# -----------------------------------------------------------------------------
# Run both:
# 1. standard CSV inference step sweeps via infer_lora_csv.py
# 2. streaming CSV chunk sweeps via streaming_infer_lora_csv.py
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
AUDIO_FORMAT="${AUDIO_FORMAT:-wav}"

# Standard inference sweep settings
DURATION="${DURATION:-30}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STEP_LIST="${STEP_LIST:-1 2 4 8}"

# Streaming inference sweep settings
WARMUP_SECONDS="${WARMUP_SECONDS:-30}"
PREDICTION_SECONDS="${PREDICTION_SECONDS:-30}"
MAX_DISTILL_SECONDS="${MAX_DISTILL_SECONDS:-150}"
CHUNK_LIST="${CHUNK_LIST:-1 3 5}"
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
    "diff"
)

MODEL_PATHS=(
    "./output/xl_consistency_diff_gpu1/final"
    # "./output/xl_consistency_fft_gpu0/final"
    # "./output/xl_consistency_v1/final"
    # "./output/xl_consistency_v1_chunk_1/final"
)

echo "============================================================"
echo "Stage 1/2: standard inference step sweep"
echo "============================================================"
for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    lora_path="${MODEL_PATHS[$idx]}"

    for step in ${STEP_LIST}; do
        output_dir="${BASE_OUTPUT_DIR}/no_streaming_${model_name}_chunk_1_step_${step}"
        echo "[standard] model=${model_name} step=${step} -> ${output_dir}"

        python infer_lora_csv.py \
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
            --duration "${DURATION}" \
            --batch-size "${BATCH_SIZE}" \
            --seed "${SEED}" \
            --inference-steps "${step}" \
            --audio-format "${AUDIO_FORMAT}" \
            "${LIMIT_ARGS[@]}"
    done
done

echo "============================================================"
echo "Stage 2/2: streaming inference chunk sweep"
echo "============================================================"
for idx in "${!MODEL_NAMES[@]}"; do
    model_name="${MODEL_NAMES[$idx]}"
    lora_path="${MODEL_PATHS[$idx]}"

    for chunk in ${CHUNK_LIST}; do
        output_dir="${BASE_OUTPUT_DIR}/${model_name}_chunk_${chunk}_step_1"
        echo "[streaming] model=${model_name} chunk=${chunk} -> ${output_dir}"

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
            --max-distill-chunks "${chunk}" \
            --audio-format "${AUDIO_FORMAT}" \
            "${TILED_ARGS[@]}" \
            "${LIMIT_ARGS[@]}"
    done
done

echo ">>> All ablation inference runs completed."
