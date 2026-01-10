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
from src.models.policy import PolicyNetwork

# --- CONFIG ---
DATASET_PATH = "./data/lift_ph_embeddings.hdf5"
WM_CHECKPOINT = "./results/checkpoints/wm_epoch_50.pth"
POLICY_CHECKPOINT = "./results/checkpoints/policy_epoch_50.pth"
OUTPUT_DIR = "./results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define device globally
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"Starting Evaluation on {device}")
    
    # 1. Load Data
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
        
    dataset = VLAEmbeddingDataset(DATASET_PATH)
    
    # Get dimensions dynamically from the first sample
    sample = dataset[0]
    state_dim = sample["state"].shape[0]
    action_dim = sample["action"].shape[0]
    text_dim = sample["text"].shape[0]
    
    # 2. Load Models
    print("Loading models...")
    wm = WorldModel(state_dim, action_dim, text_dim).to(device)
    if os.path.exists(WM_CHECKPOINT):
        wm.load_state_dict(torch.load(WM_CHECKPOINT, map_location=device))
    else:
        print(f"⚠️ Warning: WM Checkpoint not found at {WM_CHECKPOINT}")
    wm.eval()
    
    # 3. Evaluation Loop (One-Step Lookahead)
    # We select specific indices to check different parts of the data
    start_indices = [50, 150, 250] 
    trajectory_len = 40 
    
    print(f"Visualizing {len(start_indices)} trajectories...")
    
    for i, start_idx in enumerate(start_indices):
        if start_idx + trajectory_len >= len(dataset):
            continue
            
        # --- A. COLLECT GROUND TRUTH ---
        gt_states = []
        gt_actions = []
        for t in range(trajectory_len):
            # Be careful with dataset boundaries
            if start_idx + t >= len(dataset): break
            
            sample = dataset[start_idx + t]
            gt_states.append(sample["state"].numpy())
            gt_actions.append(sample["action"].numpy())
        
        gt_states = np.array(gt_states) # Shape: (T, State_Dim)
        gt_actions = np.array(gt_actions) # Shape: (T, Action_Dim)
        
        if len(gt_states) < 2: continue

        # --- B. ONE-STEP PREDICTION (PHYSICS CHECK) ---
        # "Given current REAL state + REAL action -> Predict Next State"
        one_step_preds = []
        
        with torch.no_grad():
            for t in range(len(gt_states) - 1):
                # Prepare inputs
                curr_state = torch.from_numpy(gt_states[t]).float().to(device).unsqueeze(0)
                curr_action = torch.from_numpy(gt_actions[t]).float().to(device).unsqueeze(0)
                text_emb = torch.zeros((1, text_dim)).to(device) # Placeholder
                
                # Predict
                next_state_pred = wm(curr_state, curr_action, text_emb)
                one_step_preds.append(next_state_pred.cpu().numpy()[0])
        
        one_step_plot = np.array(one_step_preds)
        
        # Align GT for plotting (GT starts at t=1 for comparison)
        gt_plot = gt_states[1:]
        
        # --- C. PLOTTING ---
        # Plot X, Y, Z (first 3 dims of state)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        dims_labels = ["X Pos", "Y Pos", "Z Pos"]
        
        for d in range(3):
            ax = axes[d]
            ax.plot(gt_plot[:, d], label="Real Next Step (Ground Truth)", color="black", linestyle="--", linewidth=1.5)
            # Ensure we don't plot out of bounds if prediction failed
            if d < one_step_plot.shape[1]:
                ax.plot(one_step_plot[:, d], label="World Model Prediction", color="green", alpha=0.8, linewidth=1.5)
            
            ax.set_ylabel(dims_labels[d])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.suptitle(f"World Model Physics Validation (One-Step)\nTrajectory {i+1} (Index {start_idx})", fontsize=14)
        plt.xlabel("Time Steps")
        plt.tight_layout()
        
        save_path = f"{OUTPUT_DIR}/physics_check_{i+1}.png"
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved plot to {save_path}")

if __name__ == "__main__":
    main()