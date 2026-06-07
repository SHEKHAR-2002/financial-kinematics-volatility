from __future__ import annotations

import numpy as np
import torch
from torch import nn


def positive_class_weight(labels: np.ndarray) -> torch.Tensor | None:
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return None
    return torch.tensor(negatives / positives, dtype=torch.float32)


class MultiTaskLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        pos_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.volatility_loss = nn.HuberLoss()
        self.regime_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        y_vol: torch.Tensor,
        y_regime: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        vol_loss = self.volatility_loss(outputs["volatility"], y_vol)
        regime_loss = self.regime_loss(outputs["regime_logit"], y_regime)
        total = self.alpha * vol_loss + self.beta * regime_loss
        return total, {
            "loss": float(total.detach().cpu()),
            "volatility_loss": float(vol_loss.detach().cpu()),
            "regime_loss": float(regime_loss.detach().cpu()),
        }
