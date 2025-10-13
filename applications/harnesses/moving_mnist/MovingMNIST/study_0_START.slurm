#!/bin/bash

# NOTE: Number of CPUs per GPU must be an even number since there are
# 2 threads per core. If an odd number is requested the next higher
# even number gets used. Also, there are 120Gb of memory on the
# node. However, if 30Gb per-job are requested for 4 jobs there are
# not enough resources. You may request up to 29G per-job.

#SBATCH --job-name=mnist_study_0
#SBATCH --partition=shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=8
#SBATCH --gpu-bind=closest
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-bind=local
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=study_0_%j.out

set -a
source "${SLURM_SUBMIT_DIR}/.env"
set +a

# Define environment variables directly in the script
YOKE_TORCH="yoke"  # Your environment name

# Set up the environment
source $YOKE_CONDA/bin/activate $YOKE_TORCH

# Print environment info for debugging
#echo "Python version:"
#python --version
#echo "PyTorch version:"
#python -c "import torch; print(torch.__version__)" 2>&1 || echo "Error importing torch"
#echo "CUDA available:"
#python -c "import torch; print(torch.cuda.is_available())"
#echo "CUDA version:"
#python -c "import torch; print(torch.version.cuda)"

# Run the training script
python train_moving_mnist.py @study_0_START.input
