#!/bin/bash
set -e
# run_streaming_inference_lora.sh
# -----------------------------------------------------------------------------
# Student-only streaming inference aligned with consistency training.
# -----------------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python streaming_infer_lora.py \
    --checkpoint-dir "./checkpoints" \
    --config-path "acestep-v15-xl-turbo" \
    --lora-path "./output/xl_consistency_v1/final" \
    --lora-scale 1.0 \
    --caption "emotional cinematic piano with swelling strings and spacious reverb" \
    --lyrics "[Instrumental]" \
    --save-path "./output/streaming_inference_lora.wav" \
    --device "cuda" \
    --seed 42 \
    --warmup-seconds 30 \
    --prediction-seconds 30 \
    --max-distill-seconds 150 \
    --max-distill-chunks 5 \
    --use-tiled-decode
