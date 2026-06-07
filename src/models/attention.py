from __future__ import annotations

import torch
from torch import nn


class AttentionPooling(nn.Module):
    """Learned attention pooling over the temporal dimension."""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh(),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: batch, sequence, channels
        scores = self.score(x).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return pooled, weights
