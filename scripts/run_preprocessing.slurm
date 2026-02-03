#!/bin/bash
#SBATCH --job-name=vla_preprocess
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ruche-gpu02,ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge
source ~/.bashrc
conda activate research

PROJECT_ROOT=/gpfs/workdir/fernandeda/vla_world_model
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

if [ -n "$CONDA_PREFIX" ]; then
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

mkdir -p logs
cd "$PROJECT_ROOT"

echo "STARTING PREPROCESSING"
python -u -m src.training.precompute_embeddings