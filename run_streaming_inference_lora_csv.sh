#!/bin/bash
set -e
# run_streaming_inference_lora_csv.sh
# -----------------------------------------------------------------------------
# Batch CSV streaming inference aligned with consistency training.
# -----------------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

python streaming_infer_lora_csv.py \
    --csv-path "../filtered_songs.csv" \
    --output-dir "../output/songdescriber_filtered/streaming_full_loss_step_1" \
    --id-column "caption_id" \
    --caption-column "caption" \
    --lyrics "[Instrumental]" \
    --checkpoint-dir "./checkpoints" \
    --config-path "acestep-v15-xl-turbo" \
    --lora-path "./output/xl_consistency_v1/final" \
    --lora-scale 1.0 \
    --device "cuda" \
    --seed 42 \
    --warmup-seconds 30 \
    --prediction-seconds 30 \
    --max-distill-seconds 150 \
    --max-distill-chunks 5 \
    --audio-format "wav" \
    --use-tiled-decode
