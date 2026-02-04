#!/bin/bash
#SBATCH --job-name=vla_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=ruche-gpu02,ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge
source ~/.bashrc
conda activate research

PROJECT_ROOT=/gpfs/workdir/fernandeda/vla_world_model
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Point HuggingFace to your local cache and Force OFFLINE mode
export HF_HOME="/gpfs/workdir/fernandeda/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_DISTRIBUTED_DEBUG=INFO

if [ -n "$CONDA_PREFIX" ]; then
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

mkdir -p logs results/checkpoints
cd "$PROJECT_ROOT"

echo "STARTING VLA (LLAMA 3) TRAINING"
# Use accelerate for better LLM training handling if installed, otherwise python
python -u -m src.training.train_vla_policy --config configs/default.yaml
