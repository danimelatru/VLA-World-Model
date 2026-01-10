#!/bin/bash
#SBATCH --job-name=vla_wm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --exclude=ruche-gpu02,ruche-gpu11,ruche-gpu16,ruche-gpu17,ruche-gpu19

module purge
source ~/.bashrc
conda activate vla-wm

# --- PROJECT CONFIGURATION ---
PROJECT_ROOT=/gpfs/workdir/fernandeda/vla_world_model
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# --- CRITICAL LIBRARY FIX (GLIBCXX) ---
# Necessary for h5py and torch to work together on Ruche
if [ -n "$CONDA_PREFIX" ]; then
  export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
fi

# Create necessary directories
mkdir -p logs
mkdir -p results/checkpoints
cd "$PROJECT_ROOT"

# --- DEBUG BLOCK ---
echo "=========================================="
echo "[DEBUG SLURM] Hostname: $(hostname)"
echo "[DEBUG SLURM] Working directory: $(pwd)"
echo "[DEBUG SLURM] Python: $(which python)"
echo "[DEBUG SLURM] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

nvidia-smi || echo "[DEBUG SLURM] nvidia-smi failed"

# Quick PyTorch test before running heavy script
python -u - << 'EOF'
import torch, os
print("[DEBUG PY] torch:", torch.__file__)
print("[DEBUG PY] version:", torch.__version__)
print("[DEBUG PY] cuda available:", torch.cuda.is_available())
print("[DEBUG PY] cuda count:", torch.cuda.device_count())
print("[DEBUG PY] visible:", os.environ.get("CUDA_VISIBLE_DEVICES"))
EOF

echo "=========================================="
echo "STARTING TRAINING SCRIPT"
echo "=========================================="

# --- MAIN EXECUTION ---
python -u scripts/train_world_model.py