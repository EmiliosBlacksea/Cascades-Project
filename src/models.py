from __future__ import annotations
import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.linear = nn.Linear(d_in, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class MLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
