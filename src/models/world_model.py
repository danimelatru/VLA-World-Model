import torch
import torch.nn as nn

class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, text_dim=512, hidden_dim=256):
        super().__init__()
        
        # 1. Input Fusion
        # We concatenate State + Action + Text
        input_dim = state_dim + action_dim + text_dim
        
        # 2. Main Network (Residual MLP)
        # Residual connections help gradients flow, making training very stable
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            
            nn.Linear(hidden_dim, state_dim) # Output: Predicted Delta
        )
        
    def forward(self, state, action, text):
        """
        Predicts next state given current state, action and text command.
        """
        # Concatenate inputs
        x = torch.cat([state, action, text], dim=-1)
        
        # Predict change (delta)
        delta_state = self.net(x)
        
        # Next state = Current state + Delta
        next_state_pred = state + delta_state
        
        return next_state_pred