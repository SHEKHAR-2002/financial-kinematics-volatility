from __future__ import annotations

import torch
from torch import nn

from src.models.attention import AttentionPooling
from src.models.tcn import TemporalConvNet


class MultiTaskTCN(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
        attention: bool = False,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_features=num_features,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.attention = AttentionPooling(hidden_channels) if attention else None
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.tcn(x)
        if self.attention is not None:
            pooled, weights = self.attention(features.transpose(1, 2))
        else:
            pooled = features[:, :, -1]
            weights = None
        return {
            "volatility": self.regression_head(pooled).squeeze(-1),
            "regime_logit": self.classification_head(pooled).squeeze(-1),
            "attention_weights": weights,
        }
