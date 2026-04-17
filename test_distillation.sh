#!/bin/bash
# test_distillation.sh
# -----------------------------------------------------------------------------
# This script runs a dry-run distillation training with random models.
# -----------------------------------------------------------------------------

# 1. Environment and Path setup
export PYTHONPATH=$PYTHONPATH:.
CONDA_PYTHON="/Users/wangbaisen/miniforge3/envs/ace/bin/python"

# 2. Generate dummy dataset
echo ">>> [1/2] Generating dummy dataset..."
$CONDA_PYTHON acestep/training_v2/generate_dummy_preprocessed.py

# 3. Launch training test
echo ">>> [2/2] Launching dry-run training loop..."
$CONDA_PYTHON acestep/training_v2/launch_test_training.py

echo ">>> Done. If the training ran at least one step, the logic is verified."
