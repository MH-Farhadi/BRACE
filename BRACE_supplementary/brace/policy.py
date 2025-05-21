import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class DualHeadActorCritic(nn.Module):
    """
    Minimal dual‑head network used in BRACE:
      • shared MLP backbone
      • γ‑head   → arbitration weight   γ∈(0, 1)
      • value‑head → V(s)
    Extend as needed if you also want a Gaussian actor for full RL.
    """
    def __init__(self, obs_dim: int, hidden: int = 256, device: str = "cpu"):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.gamma_head = nn.Linear(hidden, 1)   # σ → (0, 1) via sigmoid
        self.value_head = nn.Linear(hidden, 1)
        self.to(device)

    def forward(self, obs: torch.Tensor):
        """
        obs : (B, obs_dim)
        returns γ ∈ (0, 1) and V(s) — both (B, 1)
        """
        z      = self.backbone(obs)
        gamma  = torch.sigmoid(self.gamma_head(z))
        value  = self.value_head(z)
        return gamma, value
