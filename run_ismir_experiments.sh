#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# ISMIR Paper Latency Benchmarks
# -----------------------------------------------------------------------------
export PYTHONPATH="${PYTHONPATH:-}:."

CSV_PATH="${CSV_PATH:-../filtered_songs.csv}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-./output/ismir_benchmarks}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
CONFIG_PATH="${CONFIG_PATH:-acestep-v15-xl-turbo}"
DEVICE="${DEVICE:-cuda}"
LORA_PATH="${LORA_PATH:-./output/xl_consistency_v1/final}"
LIMIT="${LIMIT:-10}"
REPEATS="${REPEATS:-3}"

mkdir -p "${BASE_OUTPUT_DIR}"

echo "============================================================"
echo "Experiment 1: Generation Paradigm Comparison"
echo "============================================================"

# Baseline 1: Original DiT (No LoRA, 8 Steps, Batch)
echo "Running Baseline 1 (No LoRA, 8 Steps)..."
python benchmark_latency.py \
    --mode standard \
    --csv-path "${CSV_PATH}" \
    --output-csv "${BASE_OUTPUT_DIR}/baseline1_no_lora_8steps.csv" \
    --experiment-name "Baseline1_NoLoRA_8Steps" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --config-path "${CONFIG_PATH}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --repeats "${REPEATS}" \
    --inference-steps 8

# Baseline 2: Consistency Model (With LoRA, 8 Steps, Batch)
echo "Running Baseline 2 (With LoRA, 8 Steps)..."
python benchmark_latency.py \
    --mode standard \
    --csv-path "${CSV_PATH}" \
    --output-csv "${BASE_OUTPUT_DIR}/baseline2_with_lora_8steps.csv" \
    --lora-path "${LORA_PATH}" \
    --experiment-name "Baseline2_WithLoRA_8Steps" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --config-path "${CONFIG_PATH}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --repeats "${REPEATS}" \
    --inference-steps 8

# Baseline 3: Consistency Model (With LoRA, 1 Step, Batch)
echo "Running Baseline 3 (With LoRA, 1 Step)..."
python benchmark_latency.py \
    --mode standard \
    --csv-path "${CSV_PATH}" \
    --output-csv "${BASE_OUTPUT_DIR}/baseline3_with_lora_1step.csv" \
    --lora-path "${LORA_PATH}" \
    --experiment-name "Baseline3_WithLoRA_1Step" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --config-path "${CONFIG_PATH}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --repeats "${REPEATS}" \
    --inference-steps 1

# Ours: Consistency Model (With LoRA, 1 Step, Streaming)
# Note: Using default 1.0s prediction chunks for Paradigm Comparison
echo "Running Ours (With LoRA, 1 Step, Streaming)..."
python benchmark_latency.py \
    --mode streaming \
    --csv-path "${CSV_PATH}" \
    --output-csv "${BASE_OUTPUT_DIR}/ours_with_lora_streaming_1s.csv" \
    --lora-path "${LORA_PATH}" \
    --experiment-name "Ours_WithLoRA_Streaming" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --config-path "${CONFIG_PATH}" \
    --device "${DEVICE}" \
    --limit "${LIMIT}" \
    --repeats "${REPEATS}" \
    --prediction-seconds 1.0 \
    --use-tiled-decode


echo "============================================================"
echo "Experiment 2: Chunk Size Ablation"
echo "============================================================"

CHUNK_SIZES=(0.5 1.0 1.5 2.0)

for chunk in "${CHUNK_SIZES[@]}"; do
    echo "Running Streaming Chunk Size: ${chunk}s..."
    python benchmark_latency.py \
        --mode streaming \
        --csv-path "${CSV_PATH}" \
        --output-csv "${BASE_OUTPUT_DIR}/ablation_streaming_chunk_${chunk}s.csv" \
        --lora-path "${LORA_PATH}" \
        --experiment-name "Ablation_Chunk_${chunk}s" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --config-path "${CONFIG_PATH}" \
        --device "${DEVICE}" \
        --limit "${LIMIT}" \
        --repeats "${REPEATS}" \
        --prediction-seconds "${chunk}" \
        --use-tiled-decode
done

echo ">>> All ISMIR experiments completed. Results saved to ${BASE_OUTPUT_DIR}."
