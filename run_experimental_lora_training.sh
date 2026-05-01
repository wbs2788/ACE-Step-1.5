#!/bin/bash
set -e
# run_experimental_lora_training.sh
# -----------------------------------------------------------------------------
# LoRA training for the experimental glitch / uncanny electronic dataset.
# -----------------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CHECKPOINT_ROOT="./checkpoints"
MODEL_VARIANT="xl_turbo"
DATASET_JSON="./experimental_train_data/experimental_glitch_dataset.json"
TENSOR_DIR="./preprocessed_tensors/experimental_glitch"
OUTPUT_DIR="./output/experimental_glitch_lora"

mkdir -p "$TENSOR_DIR"
mkdir -p "$OUTPUT_DIR"

python -m acestep.training_v2.cli.train_fixed --preprocess \
    --yes \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --model-variant "$MODEL_VARIANT" \
    --dataset-json "$DATASET_JSON" \
    --tensor-output "$TENSOR_DIR" \
    --device "cuda" \
    --precision "bf16" \
    --max-duration 240

python -m acestep.training_v2.cli.train_fixed \
    --yes \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --model-variant "$MODEL_VARIANT" \
    --dataset-dir "$TENSOR_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "cuda" \
    --precision "bf16" \
    --batch-size 1 \
    --num-workers 0 \
    --epochs 300 \
    --learning-rate 1e-4 \
    --rank 64 \
    --alpha 128 \
    --save-every 25 \
    --save-every-steps 100

echo ">>> Experimental LoRA training completed."
