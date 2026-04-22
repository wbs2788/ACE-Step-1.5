#!/bin/bash
set -e
# run_distillation.sh
# -----------------------------------------------------------------------------
# Production launch script for ACE-Step 1.5-XL-Turbo Distillery
# -----------------------------------------------------------------------------

# 1. Environment Setup
export PYTHONPATH=$PYTHONPATH:.
CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# 2. Path Configuration (Update these to your real paths!)
CHECKPOINT_ROOT="./checkpoints"
OUTPUT_DIR="./output/xl_consistency_v1_chunk_1"
export WANDB_PROJECT="acestep-distillation"
export WANDB_NAME="xl-turbo-consistency-3k_chunk_1"

# 3. Launch Distillation
# Note: Using --gradient-checkpointing is highly recommended for XL models (4B).
python -m acestep.training_v2.cli.train_consistency --yes consistency \
    --checkpoint-dir "$CHECKPOINT_ROOT" \
    --model-variant "xl_turbo" \
    --teacher-variant "xl_turbo" \
    --data-free \
    --prompt-file prompt \
    --output-dir "$OUTPUT_DIR" \
    --device "cuda" \
    --precision "bf16" \
    --batch-size 32 \
    --gradient-accumulation 1 \
    --epochs 100000 \
    --max-iterations 2000 \
    --learning-rate 1e-4 \
    --adapter-type "lora" \
    --rank 64 \
    --alpha 128 \
    --fft-weight 1.0 \
    --diff-weight 1.0 \
    --condition-seconds 10.0 \
    --warmup-seconds 30.0 \
    --prediction-seconds 30.0 \
    --max-distill-seconds 150.0 \
    --max-distill-chunks 1 \
    --use-wandb

echo ">>> Distillation process initiated. Monitor progress via W&B or TUI."
