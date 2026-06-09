from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.net(x) + self.downsample(x))


class TemporalConvNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        in_channels = num_features
        for dilation in dilations:
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=int(dilation),
                    dropout=dropout,
                )
            )
            in_channels = hidden_channels
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: batch, sequence, features. Conv1d expects batch, features, sequence.
        return self.network(x.transpose(1, 2))


class TCNVolatilityRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
        pooling: str = "last",
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_features=num_features,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.pooling = pooling
        self.head = nn.Linear(hidden_channels, 1)

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return features.mean(dim=-1)
        return features[:, :, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        pooled = self.pool(features)
        return self.head(pooled).squeeze(-1)


class TCNVolatilityRegressorMLP(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int = 32,
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.2,
        pooling: str = "last",
    ):
        super().__init__()
        self.tcn = TemporalConvNet(
            num_features=num_features,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dilations=dilations,
            dropout=dropout,
        )
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def pool(self, features: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return features.mean(dim=-1)
        return features[:, :, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.tcn(x)
        pooled = self.pool(features)
        return self.head(pooled).squeeze(-1)
