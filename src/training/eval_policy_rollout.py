import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from src.datasets.vla_dataset import VLAEmbeddingDataset
from src.models.world_model import WorldModel
from src.models.policy import PolicyNetwork
from src.config import Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running Policy Rollout in World Model Dream on {device}")

    # 1. Load Data (Just to get dimensions and a sample start state)
    dataset = VLAEmbeddingDataset(cfg.data.dataset_path)
    
    # 2. Load Models
    # World Model
    wm = WorldModel(state_dim=1024, action_dim=7, text_dim=cfg.model.text_dim, hidden_dim=cfg.model.hidden_dim).to(device)
    # Note: State/Action dims might need adjustment based on dataset inspection logic, 
    # but for now we'll infer from dataset like training scripts do.
    
    # Infer Dims from dataset
    sample = dataset[0] # Grab first item
    state_dim = sample["state"].shape[0]
    action_dim = sample["action"].shape[0]
    
    # Re-init with correct dims
    wm = WorldModel(state_dim, action_dim, cfg.model.text_dim, cfg.model.hidden_dim).to(device)
    wm.load_state_dict(torch.load(f"{cfg.training.save_dir}/wm_best.pth", map_location=device, weights_only=True))
    wm.eval()
    
    # Policy
    policy = PolicyNetwork(state_dim, action_dim, cfg.model.text_dim, cfg.model.hidden_dim).to(device)
    policy.load_state_dict(torch.load(f"{cfg.training.save_dir}/policy_best.pth", map_location=device, weights_only=True))
    policy.eval()
    
    print("✅ Models Loaded")

    # 3. Dream Rollout Loop
    # We will pick 3 random demos (one from each task if possible) and simulate
    num_rollouts = 3
    indices = np.random.choice(len(dataset), num_rollouts, replace=False)
    
    save_dir = "results/dream_rollouts"
    os.makedirs(save_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        print(f"Simulating rollout {i+1}/{num_rollouts} (Index {idx})...")
        
        # Get Expert Trajectory for ground truth comparison
        # Note: VLAEmbeddingDataset returns transitions (t, t+1), not full trajectories directly.
        # But we can simulate forward from t=0.
        # For simplicity, we just take a single transition and rollout "blindly" from there?
        # Ideally we want a full trajectory start. 
        # Hack: The dataset has `demos` list. We can access raw HDF5 logic if needed, 
        # but `dataset[idx]` gives us a valid (state, action, text).
        
        data = dataset[idx]
        curr_state = data["state"].unsqueeze(0).to(device) # (1, StateDim)
        text_emb = data["text"].unsqueeze(0).to(device)    # (1, TextDim)
        
        # Store trajectory
        dream_traj = [curr_state.cpu().detach().numpy().squeeze()]
        
        # Simulate 50 steps into the future (or until "done")
        horizon = 40 
        
        with torch.no_grad():
            for t in range(horizon):
                # 1. Policy decides action based on CURRENT imagined state
                # (Policy thinks it's real life, but inputs come from World Model)
                action = policy(curr_state, text_emb)
                
                # 2. World Model predicts NEXT state based on action
                # (Simulates physics)
                next_state_pred = wm(curr_state, action, text_emb)
                
                # Update state
                curr_state = next_state_pred
                dream_traj.append(curr_state.cpu().detach().numpy().squeeze())

        dream_traj = np.array(dream_traj) # (T, StateDim)
        
        # Plotting (Dimensions 0,1,2 usually XYZ of robot EEF)
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        dims = ["X", "Y", "Z"]
        
        for d in range(3):
            ax = axes[d]
            ax.plot(dream_traj[:, d], label="Dreamed Policy Trajectory", color="purple", linewidth=2)
            # Ideally we would plot Ground Truth too, but mapping transition idx back to full GT trajectory is tricky in this simplified script.
            # But seeing Smooth curves implies physical plausibility.
            ax.set_ylabel(f"Robot {dims[d]}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.suptitle(f"Dream Rollout #{i+1} (Fully Hallucinated by WM + Policy)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/rollout_{i}.png")
        plt.close()
        print(f"Saved plot to {save_dir}/rollout_{i}.png")

if __name__ == "__main__":
    main()
