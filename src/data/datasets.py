from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.features import apply_risk_threshold, fit_risk_threshold

try:
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    class TorchDataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class SplitConfig:
    train_start: str = "2008-01-01"
    train_end: str = "2018-12-31"
    val_start: str = "2019-01-01"
    val_end: str = "2021-12-31"
    test_start: str = "2022-01-01"
    test_end: str | None = None


@dataclass
class SequenceArrays:
    X: np.ndarray
    y_vol: np.ndarray
    y_regime: np.ndarray
    dates: np.ndarray
    assets: np.ndarray

    @property
    def empty(self) -> bool:
        return self.X.size == 0


@dataclass
class PreparedSplits:
    train: SequenceArrays
    val: SequenceArrays
    test: SequenceArrays
    frames: dict[str, pd.DataFrame]
    feature_scaler: StandardScaler
    risk_threshold: float
    feature_columns: list[str]


def split_config_from_mapping(mapping: Mapping[str, object] | None) -> SplitConfig:
    if mapping is None:
        return SplitConfig()
    allowed = SplitConfig.__dataclass_fields__.keys()
    values = {key: mapping[key] for key in allowed if key in mapping}
    return SplitConfig(**values)


def chronological_split(
    frame: pd.DataFrame,
    config: SplitConfig | Mapping[str, object] | None = None,
    date_col: str = "Date",
    label_end_col: str | None = None,
) -> dict[str, pd.DataFrame]:
    """Split without shuffling or cross-split sequence/label construction."""
    cfg = config if isinstance(config, SplitConfig) else split_config_from_mapping(config)
    df = frame.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    if label_end_col and label_end_col in df.columns:
        df[label_end_col] = pd.to_datetime(df[label_end_col])
    df = df.sort_values(["asset", date_col] if "asset" in df.columns else [date_col])

    test_end = pd.Timestamp(cfg.test_end) if cfg.test_end else df[date_col].max()
    masks = {
        "train": (df[date_col] >= pd.Timestamp(cfg.train_start))
        & (df[date_col] <= pd.Timestamp(cfg.train_end)),
        "val": (df[date_col] >= pd.Timestamp(cfg.val_start))
        & (df[date_col] <= pd.Timestamp(cfg.val_end)),
        "test": (df[date_col] >= pd.Timestamp(cfg.test_start))
        & (df[date_col] <= test_end),
    }
    if label_end_col and label_end_col in df.columns:
        split_ends = {
            "train": pd.Timestamp(cfg.train_end),
            "val": pd.Timestamp(cfg.val_end),
            "test": test_end,
        }
        for name, end_date in split_ends.items():
            masks[name] = masks[name] & df[label_end_col].notna() & (df[label_end_col] <= end_date)
    return {name: df.loc[mask].copy().reset_index(drop=True) for name, mask in masks.items()}


def infer_horizon_from_target(target_col: str) -> int | None:
    match = re.search(r"_(\d+)$", target_col)
    return int(match.group(1)) if match else None


def ensure_label_end_date(
    frame: pd.DataFrame,
    target_col: str,
    date_col: str = "Date",
    group_col: str = "asset",
) -> tuple[pd.DataFrame, str | None]:
    """Return a frame with a target label-end date column when it can be inferred."""
    end_col = f"{target_col}_end_date"
    out = frame.copy()
    if end_col in out.columns:
        out[end_col] = pd.to_datetime(out[end_col])
        return out, end_col

    horizon = infer_horizon_from_target(target_col)
    if horizon is None:
        return out, None

    out[date_col] = pd.to_datetime(out[date_col])
    if group_col in out.columns:
        out[end_col] = (
            out.sort_values([group_col, date_col])
            .groupby(group_col, sort=False)[date_col]
            .shift(-horizon)
        )
    else:
        out = out.sort_values(date_col)
        out[end_col] = out[date_col].shift(-horizon)
    return out, end_col


def fit_feature_scaler(train_frame: pd.DataFrame, feature_columns: list[str]) -> StandardScaler:
    complete = train_frame.dropna(subset=feature_columns)
    if complete.empty:
        raise ValueError("Training split has no complete feature rows for scaler fitting.")
    scaler = StandardScaler()
    scaler.fit(complete[feature_columns].astype(float))
    return scaler


def transform_features(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    out = frame.copy()
    mask = out[feature_columns].notna().all(axis=1)
    out.loc[mask, feature_columns] = scaler.transform(out.loc[mask, feature_columns].astype(float))
    return out


def make_sequences(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    target_col: str = "future_vol_30",
    regime_col: str = "high_risk",
    date_col: str = "Date",
    group_col: str = "asset",
) -> SequenceArrays:
    """Create sliding windows inside a single split and group only."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    Xs: list[np.ndarray] = []
    y_vol: list[float] = []
    y_regime: list[float] = []
    dates: list[pd.Timestamp] = []
    assets: list[str] = []
    groups = frame.groupby(group_col, sort=False) if group_col in frame.columns else [(None, frame)]
    required = [*feature_columns, target_col, regime_col]

    for asset, group in groups:
        g = group.sort_values(date_col).reset_index(drop=True)
        usable = g.dropna(subset=required).reset_index(drop=True)
        if len(usable) < sequence_length:
            continue

        features = usable[feature_columns].astype(float).to_numpy()
        vol_values = usable[target_col].astype(float).to_numpy()
        regime_values = usable[regime_col].astype(float).to_numpy()
        date_values = usable[date_col].to_numpy()

        for end_idx in range(sequence_length - 1, len(usable)):
            start_idx = end_idx - sequence_length + 1
            Xs.append(features[start_idx : end_idx + 1])
            y_vol.append(vol_values[end_idx])
            y_regime.append(regime_values[end_idx])
            dates.append(date_values[end_idx])
            assets.append(str(asset) if asset is not None else "UNKNOWN")

    if not Xs:
        return SequenceArrays(
            X=np.empty((0, sequence_length, len(feature_columns)), dtype=np.float32),
            y_vol=np.empty((0,), dtype=np.float32),
            y_regime=np.empty((0,), dtype=np.float32),
            dates=np.empty((0,), dtype="datetime64[ns]"),
            assets=np.empty((0,), dtype=object),
        )

    return SequenceArrays(
        X=np.asarray(Xs, dtype=np.float32),
        y_vol=np.asarray(y_vol, dtype=np.float32),
        y_regime=np.asarray(y_regime, dtype=np.float32),
        dates=np.asarray(dates),
        assets=np.asarray(assets, dtype=object),
    )


def sequence_target_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    sequence_length: int,
    target_col: str = "future_vol_30",
    regime_col: str = "high_risk",
    date_col: str = "Date",
    group_col: str = "asset",
) -> pd.DataFrame:
    """Return the target rows used by make_sequences for tabular baselines."""
    rows = []
    groups = frame.groupby(group_col, sort=False) if group_col in frame.columns else [(None, frame)]
    required = [*feature_columns, target_col, regime_col]

    for _, group in groups:
        usable = group.sort_values(date_col).dropna(subset=required).reset_index(drop=True)
        if len(usable) >= sequence_length:
            rows.append(usable.iloc[sequence_length - 1 :].copy())

    if not rows:
        return frame.iloc[0:0].copy()
    return pd.concat(rows, ignore_index=True)


def prepare_splits(
    frame: pd.DataFrame,
    feature_columns: list[str],
    split_config: SplitConfig | Mapping[str, object] | None = None,
    sequence_length: int = 60,
    target_col: str = "future_vol_30",
    regime_quantile: float = 0.75,
    enforce_label_boundaries: bool = True,
) -> PreparedSplits:
    """Prepare leakage-free scaled sequence splits."""
    prepared_frame, label_end_col = ensure_label_end_date(frame, target_col)
    split_frames = chronological_split(
        prepared_frame,
        split_config,
        label_end_col=label_end_col if enforce_label_boundaries else None,
    )
    threshold = fit_risk_threshold(split_frames["train"], target_col, regime_quantile)
    labeled_frames = {
        name: apply_risk_threshold(split, threshold, target_col=target_col)
        for name, split in split_frames.items()
    }

    scaler = fit_feature_scaler(labeled_frames["train"], feature_columns)
    scaled_frames = {
        name: transform_features(split, scaler, feature_columns)
        for name, split in labeled_frames.items()
    }
    sequences = {
        name: make_sequences(
            split,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            target_col=target_col,
        )
        for name, split in scaled_frames.items()
    }

    return PreparedSplits(
        train=sequences["train"],
        val=sequences["val"],
        test=sequences["test"],
        frames=scaled_frames,
        feature_scaler=scaler,
        risk_threshold=threshold,
        feature_columns=feature_columns,
    )


class SequenceDataset(TorchDataset):
    """Torch Dataset wrapper imported lazily so preprocessing tests do not need torch."""

    def __init__(self, arrays: SequenceArrays):
        import torch

        self.X = torch.as_tensor(arrays.X, dtype=torch.float32)
        self.y_vol = torch.as_tensor(arrays.y_vol, dtype=torch.float32)
        self.y_regime = torch.as_tensor(arrays.y_regime, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y_vol)

    def __getitem__(self, idx: int):
        return {
            "X": self.X[idx],
            "y_vol": self.y_vol[idx],
            "y_regime": self.y_regime[idx],
        }


def latest_step_tabular(arrays: SequenceArrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use the latest timestep in each sequence for classical ML baselines."""
    if arrays.empty:
        return np.empty((0, 0)), arrays.y_vol, arrays.y_regime
    return arrays.X[:, -1, :], arrays.y_vol, arrays.y_regime


def flattened_window_tabular(arrays: SequenceArrays) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten complete windows for classical ML ablations."""
    if arrays.empty:
        return np.empty((0, 0)), arrays.y_vol, arrays.y_regime
    return arrays.X.reshape(arrays.X.shape[0], -1), arrays.y_vol, arrays.y_regime
