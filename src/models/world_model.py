import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """
    A simple residual MLP World Model that predicts the next state dynamics.
    
    Args:
        state_dim (int): Dimensionality of the state vector.
        action_dim (int): Dimensionality of the action vector.
        text_dim (int): Dimensionality of the text embedding.
        hidden_dim (int): Hidden layer size.
    """
    def __init__(self, state_dim: int, action_dim: int, text_dim: int = 512, hidden_dim: int = 256):
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
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Predicts next state given current state, action and text command.
        
        Args:
            state (torch.Tensor): Current state [B, state_dim]
            action (torch.Tensor): Action taken [B, action_dim]
            text (torch.Tensor): Text embedding [B, text_dim]
            
        Returns:
            torch.Tensor: Predicted next state [B, state_dim]
        """
        # Concatenate inputs
        x = torch.cat([state, action, text], dim=-1)
        
        # Predict change (delta)
        delta_state = self.net(x)
        
        # Next state = Current state + Delta
        next_state_pred = state + delta_state
        
        return next_state_pred