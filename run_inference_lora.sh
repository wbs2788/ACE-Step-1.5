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
    --checkpoint-dir "./checkpoints" \
    --config-path "acestep-v15-xl-turbo" \
    --lora-path "./output/xl_consistency_v1/final" \
    --lora-scale 1.0 \
    --caption "A cutting-edge Glitch Experimental Electronic track. Features intricate granular synthesis, micro-sampling of digital errors, and sharp rhythmic clicks and pops. Stuttering bit-crushed textures layered over a deep, pulsating sine-wave sub-bass. Non-linear structure with sudden shifts in spatial panning. Cold, futuristic, and clinical aesthetic. High-frequency digital debris, erratic percussion patterns, and distorted data-stream soundscapes. 110 BPM, avant-garde and sterile." \
    --lyrics "[Instrumental]" \
    --duration 30 \
    --save-dir ".." \
    --device "cuda" \
    --batch-size 1 \
    --seed 42 \
    --inference-steps 1
