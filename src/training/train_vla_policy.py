import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from src.datasets.vla_dataset import VLAEmbeddingDataset
from src.models.vla_llm_policy import VLAPolicy
from src.config import Config

# Re-define prompts here to map them to tasks (or import them)
# Ideally this should be centralized, but for now we duplicate for simplicity
TASK_PROMPTS = {
    "lift": ["Lift the red cube", "Pick up the red cube", "Grasp the red object"],
    "can": ["Pick up the coke can", "Grasp the soda can", "Lift the metal can"],
    "square": ["Push the square nut", "Insert the square nut onto the peg", "Assemble the nut"]
}

def get_random_prompt(task_name):
    # Fallback to lift if unknown
    key = task_name if task_name in TASK_PROMPTS else "lift"
    return np.random.choice(TASK_PROMPTS[key])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    cfg = Config.load(args.config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Starting VLA Policy Training (LLM-based) on {device}")
    
    # 1. Dataset
    full_dataset = VLAEmbeddingDataset(cfg.data.dataset_path)
    
    # We need to access the "Task Name" to give the right text to the LLM
    # VLAEmbeddingDataset currently doesn't easily expose "Task Name" per item in __getitem__
    # It just returns tensors. 
    # HACK: We will subclass or modify logic to infer task from internal data, 
    # or just assume specific task distribution?
    # BETTER: The HDF5 `text_mode="multiple"` logic serves CLIP. 
    # Let's inspect dataset structure. data.keys() are like "lift_demo_0". 
    # We can parse the key to get the task!
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4)
    
    # 2. Model
    # We infer dims from one sample
    sample = next(iter(train_loader))
    state_dim = sample["state"].shape[1]
    action_dim = sample["action"].shape[1]
    
    vla = VLAPolicy(
        model_name=cfg.vla.model_name,
        state_dim=state_dim,
        action_dim=action_dim,
        load_in_4bit=cfg.vla.load_in_4bit
    ).to(device)
    
    # 3. Optimizer
    optimizer = optim.AdamW(vla.parameters(), lr=1e-4) # Lower LR for LLMs
    criterion = nn.MSELoss()
    
    epochs = cfg.training.epochs
    save_dir = cfg.training.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    # 4. Training Loop
    for epoch in range(epochs):
        vla.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for batch in pbar:
            state = batch["state"].to(device)
            target_action = batch["action"].to(device)
            
            # Generate Text Prompts for this batch on the fly
            # Since our Dataset doesn't return the raw key, checking task is hard without modifying Dataset.
            # CRITICAL TODO: Modify Dataset to return 'task_name' string?
            # Workaround: Since we merged 3 tasks, we can probabilistically guess or just use "Generic" instructions?
            # NO, that ruins the point.
            # Let's Modify VLAEmbeddingDataset to return 'task' meta-info.
            
            # (Assuming we updated Dataset - see next step tool call)
            # tasks = batch["task_name"] 
            # text_batch = [get_random_prompt(t) for t in tasks]
            
            # TEMPORARY BACKUP until dataset update: Randomly pick valid actions?
            # No, that's bad.
            # Let's assume the user will update dataset code next.
            # For this file writing, I will assume batch["task"] exists.
            
            text_batch = batch["raw_text"] # We will add this to dataset
            
            pred_action = vla(state, text_batch)
            loss = criterion(pred_action, target_action)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")
        
        # Save
        if (epoch+1) % 5 == 0:
            torch.save(vla.state_dict(), f"{save_dir}/vla_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
