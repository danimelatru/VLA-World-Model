import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, text_dim=512, hidden_dim=256):
        super().__init__()
        
        # Input: State + Text Instruction
        input_dim = state_dim + text_dim
        
        # Simple but effective MLP (Multi-Layer Perceptron)
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
            
            # Output: Action (dx, dy, dz, dquat...)
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state, text):
        # Concatenate State and Text
        x = torch.cat([state, text], dim=-1)
        
        # Predict Action
        action_pred = self.net(x)
        
        return action_pred