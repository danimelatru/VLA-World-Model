import pytest
import torch
from src.models.world_model import WorldModel

def test_world_model_shapes():
    """Verify that WorldModel outputs the correct shape."""
    B, S, A, T = 4, 10, 5, 20  # Batch, State, Action, Text dims
    model = WorldModel(state_dim=S, action_dim=A, text_dim=T)
    
    state = torch.randn(B, S)
    action = torch.randn(B, A)
    text = torch.randn(B, T)
    
    output = model(state, action, text)
    
    assert output.shape == (B, S), f"Expected output shape {(B, S)}, got {output.shape}"

def test_world_model_overfit_small_batch():
    """Verify that WorldModel can overfit a tiny batch (sanity check)."""
    B, S, A, T = 2, 4, 2, 8
    model = WorldModel(state_dim=S, action_dim=A, text_dim=T, hidden_dim=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.MSELoss()
    
    # Random data
    state = torch.randn(B, S)
    action = torch.randn(B, A)
    text = torch.randn(B, T)
    target = state + 0.1 # Dummy target
    
    # Overfit loop
    for _ in range(100):
        pred = model(state, action, text)
        loss = criterion(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    assert loss.item() < 0.01, f"Model failed to overfit small batch, final loss: {loss.item()}"
