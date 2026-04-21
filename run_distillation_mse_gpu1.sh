#!/bin/bash
set -e
# run_distillation_mse_gpu1.sh
# -----------------------------------------------------------------------------
# Ablation launch script: consistency MSE loss only on physical GPU 1.
# -----------------------------------------------------------------------------

# 1. Environment Setup
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# 2. Path Configuration
CHECKPOINT_ROOT="./checkpoints"
OUTPUT_DIR="${OUTPUT_DIR:-./output/xl_consistency_mse_gpu1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-2000}"
export WANDB_PROJECT="${WANDB_PROJECT:-acestep-distillation}"
export WANDB_NAME="${WANDB_NAME:-xl-turbo-consistency-mse-${MAX_ITERATIONS}it-gpu1}"

# 3. Launch Distillation
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
    --max-iterations "$MAX_ITERATIONS" \
    --save-every-steps 250 \
    --learning-rate 1e-4 \
    --adapter-type "lora" \
    --rank 64 \
    --alpha 128 \
    --fft-weight 0.0 \
    --diff-weight 0.0 \
    --condition-seconds 10.0 \
    --warmup-seconds 30.0 \
    --prediction-seconds 30.0 \
    --max-distill-seconds 150.0 \
    --max-distill-chunks 5 \
    --use-wandb

echo ">>> MSE ablation initiated on GPU 1. Monitor progress via W&B or TUI."
