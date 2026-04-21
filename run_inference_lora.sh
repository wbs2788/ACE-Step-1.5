#!/bin/bash
set -e
# run_inference_lora.sh
# -----------------------------------------------------------------------------
# Minimal inference wrapper that loads the trained consistency LoRA adapter.
# -----------------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python infer_lora.py \
    --config-path "acestep-v15-xl-turbo" \
    --lora-path "./output/xl_consistency_v1/final" \
    --lora-scale 1.0 \
    --caption "emotional cinematic piano with swelling strings and spacious reverb" \
    --lyrics "[Instrumental]" \
    --duration 30 \
    --save-dir "./output/inference_lora" \
    --device "cuda" \
    --batch-size 1 \
    --seed 42 \
    --inference-steps 8
