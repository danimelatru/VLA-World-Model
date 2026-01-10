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

# --- CONFIG ---
DATASET_PATH = "./data/lift_ph_embeddings.hdf5"
BATCH_SIZE = 256
EPOCHS = 50
LR = 1e-3
SAVE_DIR = "./results/checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    print(f"🚀 Starting World Model Training on {device}")
    
    # 1. Load Dataset
    train_dataset = VLAEmbeddingDataset(DATASET_PATH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # 2. Inspect one batch to determine dimensions dynamically
    sample = next(iter(train_loader))
    state_dim = sample["state"].shape[1]
    action_dim = sample["action"].shape[1]
    text_dim = sample["text"].shape[1]
    
    print(f"ℹ️ Dims -> State: {state_dim}, Action: {action_dim}, Text: {text_dim}")
    
    # 3. Initialize Model
    model = WorldModel(state_dim, action_dim, text_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss() # Simple Mean Squared Error for regression
    
    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in pbar:
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
        
        avg_loss = total_loss / len(train_loader)
        print(f"📉 Epoch {epoch+1} Summary: Avg Loss = {avg_loss:.6f}")
        
        # Save Checkpoint
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), f"{SAVE_DIR}/wm_epoch_{epoch+1}.pth")
            print(f"💾 Checkpoint saved.")

if __name__ == "__main__":
    main()