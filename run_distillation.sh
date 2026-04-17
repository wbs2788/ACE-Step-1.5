#!/bin/bash
# run_distillation.sh
# -----------------------------------------------------------------------------
# Production launch script for ACE-Step 1.5-XL-Turbo Distillery
# -----------------------------------------------------------------------------

# 1. Environment Setup
export PYTHONPATH=$PYTHONPATH:.
CONDA_PYTHON="/Users/wangbaisen/miniforge3/envs/ace/bin/python"

# 2. Path Configuration (Update these to your real paths!)
CHECKPOINT_ROOT="./checkpoints"
DATASET_DIR="./datasets/preprocessed_xl_audio"
OUTPUT_DIR="./output/xl_consistency_v1"
WANDB_PROJECT="acestep-distillation"

# 3. Launch Distillation
# Note: Using --gradient-checkpointing is highly recommended for XL models (4B).
$CONDA_PYTHON -m acestep.training_v2.cli.train_consistency \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --model-variant "xl_turbo" \
    --teacher-variant "xl_turbo" \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device "cuda" \
    --precision "bf16" \
    --batch-size 1 \
    --gradient-accumulation 8 \
    --epochs 100 \
    --learning-rate 5e-5 \
    --adapter-type "lora" \
    --rank 64 \
    --alpha 128 \
    --fft-weight 1.0 \
    --diff-weight 1.0 \
    --condition-seconds 10.0 \
    --prediction-seconds 30.0 \
    --gradient-checkpointing \
    --use-wandb \
    --yes

echo ">>> Distillation process initiated. Monitor progress via W&B or TUI."
