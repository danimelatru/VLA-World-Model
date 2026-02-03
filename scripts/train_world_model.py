import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Add project root to path so we can import src
sys.path.append(os.getcwd())

from src.datasets.vla_dataset import VLAEmbeddingDataset
from src.models.world_model import WorldModel

import argparse
import sys
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

# Add project root to path so we can import src
sys.path.append(os.getcwd())

from src.datasets.vla_dataset import VLAEmbeddingDataset
from src.models.world_model import WorldModel
from src.config import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Train World Model")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Run a single batch to verify pipeline")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.load(args.config)
    
    device = cfg.training.device if cfg.training.device and torch.cuda.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Starting World Model Training on {device}")
    
    # 1. Load Dataset
    if not os.path.exists(cfg.data.dataset_path):
        if args.dry_run:
            print(f"⚠️ Dataset {cfg.data.dataset_path} not found, but dry-run is seemingly fine with mocks (not implemented yet). ERROR for now.")
            # For now, just error out if file missing, unless we want to mock it. 
            # But the user likely has data or will download it.
        raise FileNotFoundError(f"Dataset not found at {cfg.data.dataset_path}")

    train_dataset = VLAEmbeddingDataset(cfg.data.dataset_path)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg.training.batch_size, 
        shuffle=True, 
        num_workers=cfg.data.num_workers
    )
    
    # 2. Inspect one batch to determine dimensions dynamically
    sample = next(iter(train_loader))
    state_dim = sample["state"].shape[1]
    action_dim = sample["action"].shape[1]
    text_dim = sample["text"].shape[1]
    
    print(f"ℹ️ Dims -> State: {state_dim}, Action: {action_dim}, Text: {text_dim}")
    
    # 3. Initialize Model
    # Use config overrides if specified, else use inferred
    # For now we use inferred dims for state/action as they depend on data
    model = WorldModel(
        state_dim=state_dim, 
        action_dim=action_dim, 
        text_dim=cfg.model.text_dim, 
        hidden_dim=cfg.model.hidden_dim
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)
    
    # LR Scheduler
    scheduler = None
    if cfg.training.use_scheduler:
        # Simple Cosine Annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
        print("✅ LR Scheduler: CosineAnnealingLR enabled")

    criterion = nn.MSELoss()
    
    os.makedirs(cfg.training.save_dir, exist_ok=True)
    
    # 4. Training Loop
    model.train()
    epochs = 1 if args.dry_run else cfg.training.epochs
    best_loss = float("inf")
    
    for epoch in range(epochs):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            # Move to GPU
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            text = batch["text"].to(device)
            target = batch["next_state"].to(device)
            
            # Forward
            pred = model(state, action, text)
            
            # Loss
            loss = criterion(pred, target)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
            if args.dry_run and i >= 1:
                print("Dry run: breaking after 2 batches.")
                break
        
        avg_loss = total_loss / (len(train_loader) if not args.dry_run else 2)
        print(f"📉 Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.6f}")
        
        # Step Scheduler
        if scheduler:
            scheduler.step()
            
        # Save Checkpoint & Best Model
        if not args.dry_run:
            # Periodic save
            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), f"{cfg.training.save_dir}/wm_epoch_{epoch+1}.pth")
            
            # Best save
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"{cfg.training.save_dir}/wm_best.pth")
                print(f"🌟 New Best Model Saved (Loss: {best_loss:.6f})")

if __name__ == "__main__":
    main()