import argparse
import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Add project root
sys.path.append(os.getcwd())

from src.datasets.vla_dataset import VLAEmbeddingDataset
from src.models.world_model import WorldModel
from src.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate World Model Dreaming")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a single trajectory to verify pipeline")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.load(args.config)
    
    device = cfg.training.device if cfg.training.device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting Evaluation on {device}")
    
    # 1. Load Data
    if not os.path.exists(cfg.data.dataset_path):
        if args.dry_run:
             print("⚠️ Dataset not found, skipping evaluation in dry-run.")
             return
        raise FileNotFoundError(f"Dataset not found at {cfg.data.dataset_path}")
        
    dataset = VLAEmbeddingDataset(cfg.data.dataset_path)
    
    # Get dimensions dynamically
    sample = dataset[0]
    state_dim = sample["state"].shape[0]
    action_dim = sample["action"].shape[0]
    text_dim = sample["text"].shape[0]
    
    # 2. Load Models
    print("Loading models...")
    wm = WorldModel(
        state_dim=state_dim, 
        action_dim=action_dim, 
        text_dim=text_dim, # Use actual dim from data
        hidden_dim=cfg.model.hidden_dim
    ).to(device)
    
    # Load Checkpoint
    checkpoint_path = f"{cfg.training.save_dir}/wm_epoch_{cfg.training.epochs}.pth"
    if os.path.exists(checkpoint_path):
        try:
            wm.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"✅ Loaded checkpoint: {checkpoint_path}")
        except RuntimeError as e:
            print(f"⚠️ Warning: Could not load checkpoint due to shape mismatch (Expected if you changed model size): {e}")
            print("➡️ Proceeding with random weights.")
    else:
        print(f"⚠️ Warning: Checkpoint not found at {checkpoint_path}. Using random weights.")
    
    wm.eval()
    
    # 3. Evaluation Loop (One-Step Lookahead)
    start_indices = [50] if args.dry_run else [50, 150, 250, 350, 450]
    trajectory_len = 40 
    
    print(f"Visualizing {len(start_indices)} trajectories...")
    
    output_dir = "./results/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, start_idx in enumerate(start_indices):
        if start_idx + trajectory_len >= len(dataset):
            continue
            
        # --- A. COLLECT GROUND TRUTH ---
        gt_states = []
        gt_actions = []
        gt_texts = []
        
        for t in range(trajectory_len):
            if start_idx + t >= len(dataset): break
            
            sample = dataset[start_idx + t]
            gt_states.append(sample["state"].numpy())
            gt_actions.append(sample["action"].numpy())
            gt_texts.append(sample["text"].numpy())
        
        gt_states = np.array(gt_states) 
        gt_actions = np.array(gt_actions)
        gt_texts = np.array(gt_texts)
        
        if len(gt_states) < 2: continue

        # --- B. ONE-STEP PREDICTION (PHYSICS CHECK) ---
        one_step_preds = []
        
        with torch.no_grad():
            for t in range(len(gt_states) - 1):
                # Prepare inputs
                curr_state = torch.from_numpy(gt_states[t]).float().to(device).unsqueeze(0)
                curr_action = torch.from_numpy(gt_actions[t]).float().to(device).unsqueeze(0)
                
                # CRITICAL FIX: Use REAL text embedding from dataset, not zeros
                curr_text = torch.from_numpy(gt_texts[t]).float().to(device).unsqueeze(0)
                
                # Predict
                next_state_pred = wm(curr_state, curr_action, curr_text)
                one_step_preds.append(next_state_pred.cpu().numpy()[0])
        
        one_step_plot = np.array(one_step_preds)
        gt_plot = gt_states[1:]
        
        # --- C. PLOTTING ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        dims_labels = ["X Pos", "Y Pos", "Z Pos"]
        
        for d in range(3):
            ax = axes[d]
            ax.plot(gt_plot[:, d], label="Real Next Step (Ground Truth)", color="black", linestyle="--", linewidth=1.5)
            if d < one_step_plot.shape[1]:
                ax.plot(one_step_plot[:, d], label="World Model Prediction", color="green", alpha=0.8, linewidth=1.5)
            
            ax.set_ylabel(dims_labels[d])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(f"World Model Physics Validation (One-Step)\nTrajectory {i+1} (Index {start_idx})", fontsize=14)
        plt.xlabel("Time Steps")
        plt.tight_layout()
        
        save_path = f"{output_dir}/physics_check_{i+1}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved plot to {save_path}")

if __name__ == "__main__":
    main()