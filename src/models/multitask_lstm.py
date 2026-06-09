from __future__ import annotations

import torch
from torch import nn


class LSTMMultiTask(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.regression_head = nn.Linear(hidden_size, 1)
        self.classification_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output, _ = self.lstm(x)
        last = output[:, -1, :]
        return {
            "volatility": self.regression_head(last).squeeze(-1),
            "regime_logit": self.classification_head(last).squeeze(-1),
            "attention_weights": None,
        }
