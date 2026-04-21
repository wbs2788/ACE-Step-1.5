#!/bin/bash
set -e
# run_inference_lora_csv.sh
# -----------------------------------------------------------------------------
# Batch CSV inference wrapper for a trained ACE-Step LoRA adapter.
# -----------------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python infer_lora_csv.py \
    --csv-path "./prompts.csv" \
    --output-dir "./output/inference_lora_csv" \
    --id-column "caption_id" \
    --caption-column "caption" \
    --lyrics "[Instrumental]" \
    --lora-path "./output/xl_consistency_v1/final" \
    --lora-scale 1.0 \
    --config-path "acestep-v15-xl-turbo" \
    --device "cuda" \
    --duration 30 \
    --batch-size 1 \
    --seed 42 \
    --inference-steps 8 \
    --audio-format "wav"
