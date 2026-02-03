import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

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
    print(f"Starting training on {device}")
    
    # 1. Load Dataset
    if not os.path.exists(cfg.data.dataset_path):
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
    
    # 3. Initialize Model
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

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
            state = batch["state"].to(device)
            action = batch["action"].to(device)
            text = batch["text"].to(device)
            target = batch["next_state"].to(device)
            
            pred = model(state, action, text)
            loss = criterion(pred, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
            if args.dry_run and i >= 1:
                print("Dry run: breaking after 2 batches.")
                break
        
        avg_loss = total_loss / (len(train_loader) if not args.dry_run else 2)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")
        
        if scheduler:
            scheduler.step()
            
        if not args.dry_run:
            if (epoch+1) % 10 == 0:
                torch.save(model.state_dict(), f"{cfg.training.save_dir}/wm_epoch_{epoch+1}.pth")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"{cfg.training.save_dir}/wm_best.pth")

if __name__ == "__main__":
    main()