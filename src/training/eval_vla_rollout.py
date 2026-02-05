import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from src.datasets.vla_dataset import VLAEmbeddingDataset
from src.models.world_model import WorldModel
from src.models.vla_llm_policy import VLAPolicy
from src.config import Config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.load(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running VLA (Llama 3) Dream Rollout on {device}")

    # 1. Load Data
    dataset = VLAEmbeddingDataset(cfg.data.dataset_path)
    
    # 2. Load Models
    
    # World Model (The Physics Engine)
    # We infer dims from dataset
    sample = dataset[0]
    state_dim = sample["state"].shape[0]
    action_dim = sample["action"].shape[0]
    
    wm = WorldModel(state_dim, action_dim, cfg.model.text_dim, cfg.model.hidden_dim).to(device)
    wm.load_state_dict(torch.load(f"{cfg.training.save_dir}/wm_best.pth", map_location=device, weights_only=True))
    wm.eval()
    
    # VLA Policy (The Brain)
    # Note: VLA config might dictate 4bit, but for eval we can probably load in 16bit if VRAM allows, 
    # or stick to config. We'll stick to config to match training environment.
    vla = VLAPolicy(
        model_name=cfg.vla.model_name,
        state_dim=state_dim,
        action_dim=action_dim,
        load_in_4bit=cfg.vla.load_in_4bit
    ).to(device)
    
    # Load the trained LoRA weights
    # We saved the state_dict of the whole VLA wrapper or just PEFT?
    # In train_vla_policy.py we did: torch.save(vla.state_dict(), ...)
    # So we load it back.
    checkpoint_path = f"{cfg.training.save_dir}/vla_epoch_{cfg.training.epochs}.pth"
    print(f"Loading VLA from: {checkpoint_path}")
    
    # Since VLAPolicy wraps the LLM + Adapters, load_state_dict should work 
    # IF the keys match (which they should).
    # Note: PEFT models sometimes have tricky state_dict keys (base_model.model...)
    # But since we saved the whole vla.state_dict(), it should be consistent.
    
    # We need to ensure we don't run OOM loading 2 models.
    # WM is small. VLA is big.
    
    try:
        vla.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    except Exception as e:
        print(f"Standard load failed, trying loose loading (strict=False) for LoRA compatibility: {e}")
        vla.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True), strict=False)
        
    vla.eval()
    
    print("✅ Models Loaded")

    # 3. Dream Rollout Loop
    num_rollouts = 3
    indices = np.random.choice(len(dataset), num_rollouts, replace=False)
    
    save_dir = "results/vla_dream_rollouts"
    os.makedirs(save_dir, exist_ok=True)

    for i, idx in enumerate(indices):
        print(f"Simulating VLA rollout {i+1}/{num_rollouts} (Index {idx})...")
        
        data = dataset[idx]
        curr_state = data["state"].unsqueeze(0).to(device) # (1, StateDim)
        
        # Determine Text Prompt
        # We can use the one from the dataset or override it to test understanding
        # Let's use the one from the dataset to verify it learned the task.
        raw_text = data["raw_text"]
        print(f"Instruction: '{raw_text}'")
        
        # For the World Model, we still need the CLIP embedding
        text_emb = data["text"].unsqueeze(0).to(device)
        
        # Casting state for VLA
        # (Handled inside VLA forward, but good to remember)
        
        dream_traj = [curr_state.cpu().detach().numpy().squeeze()]
        horizon = 40 
        
        with torch.no_grad():
            for t in range(horizon):
                # 1. VLA decides action (State + Text String)
                # VLA expects text_list
                action = vla(curr_state, [raw_text])
                
                # 2. World Model predicts NEXT state (State + Action + Query Embedding)
                next_state_pred = wm(curr_state, action.float(), text_emb)
                
                curr_state = next_state_pred
                dream_traj.append(curr_state.cpu().detach().numpy().squeeze())

        dream_traj = np.array(dream_traj) # (T, StateDim)
        
        # Plotting
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        dims = ["X", "Y", "Z"]
        
        for d in range(3):
            ax = axes[d]
            ax.plot(dream_traj[:, d], label="VLA (Llama 3) Trajectory", color="orange", linewidth=2)
            ax.set_ylabel(f"Robot {dims[d]}")
            ax.set_title(f"Dimension {dims[d]}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
        plt.suptitle(f"VLA Dream: '{raw_text}'\n(Llama 3 Policy + World Model Physics)")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/vla_rollout_{i}.png")
        plt.close()
        print(f"Saved plot to {save_dir}/vla_rollout_{i}.png")

if __name__ == "__main__":
    main()
