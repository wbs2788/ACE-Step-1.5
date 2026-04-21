#!/bin/bash
set -e
# run_lr_sweep_gpu1.sh
# -----------------------------------------------------------------------------
# Overnight LR sweep for data-free XL consistency distillation.
# Defaults to an approximately 10-hour sweep on two L20X GPUs.
# -----------------------------------------------------------------------------

export PYTHONPATH=$PYTHONPATH:.

CONDA_ENV="ace"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

CHECKPOINT_ROOT="./checkpoints"
PROMPT_FILE="prompt"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/lr_sweep}"

export WANDB_PROJECT="${WANDB_PROJECT:-acestep-distillation}"

BATCH_SIZE="${BATCH_SIZE:-32}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-1}"
MAX_ITERATIONS="${MAX_ITERATIONS:-80}"
MAX_DISTILL_CHUNKS="${MAX_DISTILL_CHUNKS:-5}"
GPU_LIST="${GPU_LIST:-0,1}"

# Override from the shell when you want a wider or narrower sweep:
#   LR_LIST_STR="1e-5 3e-5 5e-5 1e-4 2e-4" bash run_lr_sweep_gpu1.sh
LR_LIST_STR="${LR_LIST_STR:-1e-6 2e-6 5e-6 1e-5 2e-5 5e-5 1e-4 2e-4}"

IFS="," read -r -a GPUS <<< "$GPU_LIST"
read -r -a LR_LIST <<< "$LR_LIST_STR"
MAX_PARALLEL="${MAX_PARALLEL:-${#GPUS[@]}}"

if [ "${#GPUS[@]}" -eq 0 ]; then
    echo "GPU_LIST is empty. Example: GPU_LIST=0,1 bash run_lr_sweep_gpu1.sh" >&2
    exit 1
fi

run_one_lr() {
    local lr="$1"
    local gpu="$2"
    local lr_tag run_name

    lr_tag="${lr//./p}"
    lr_tag="${lr_tag//-e-/em}"
    lr_tag="${lr_tag//e-/em}"
    run_name="xl-consistency-lr-${lr_tag}-bs${BATCH_SIZE}-ga${GRADIENT_ACCUMULATION}-${MAX_ITERATIONS}it-gpu${gpu}"

    echo ">>> Starting LR sweep run: lr=${lr}, gpu=${gpu}, batch=${BATCH_SIZE}, accum=${GRADIENT_ACCUMULATION}, iterations=${MAX_ITERATIONS}"

    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    WANDB_NAME="$run_name" \
    python -m acestep.training_v2.cli.train_consistency --yes consistency \
        --checkpoint-dir "$CHECKPOINT_ROOT" \
        --model-variant "xl_turbo" \
        --teacher-variant "xl_turbo" \
        --data-free \
        --prompt-file "$PROMPT_FILE" \
        --output-dir "$OUTPUT_ROOT/$run_name" \
        --device "cuda" \
        --precision "bf16" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation "$GRADIENT_ACCUMULATION" \
        --epochs 100000 \
        --max-iterations "$MAX_ITERATIONS" \
        --learning-rate "$lr" \
        --adapter-type "lora" \
        --rank 64 \
        --alpha 128 \
        --fft-weight 1.0 \
        --diff-weight 1.0 \
        --condition-seconds 10.0 \
        --warmup-seconds 30.0 \
        --prediction-seconds 30.0 \
        --max-distill-seconds 150.0 \
        --max-distill-chunks "$MAX_DISTILL_CHUNKS" \
        --use-wandb
    local status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        echo ">>> Finished LR sweep run: lr=${lr}, gpu=${gpu}"
    else
        echo ">>> LR sweep run failed: lr=${lr}, gpu=${gpu}, exit=${status}" >&2
    fi
}

running=0
idx=0
for lr in "${LR_LIST[@]}"; do
    gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
    run_one_lr "$lr" "$gpu" &
    running=$((running + 1))
    idx=$((idx + 1))

    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait -n
        running=$((running - 1))
    fi
done

wait
echo ">>> LR sweep complete. Pick candidates by early loss slope, spike/NaN behavior, and short-run final loss in W&B."
