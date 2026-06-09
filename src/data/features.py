from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


KINEMATIC_FEATURES = ["log_price", "velocity", "acceleration"]
LOG_PRICE_VELOCITY_FEATURES = ["log_price", "velocity"]
FULL_DOMAIN_FEATURES = [
    "log_price",
    "velocity",
    "acceleration",
    "past_vol_7",
    "past_vol_14",
    "past_vol_30",
    "momentum_7",
    "momentum_30",
    "momentum_90",
    "drawdown_90",
    "tr_mean_7",
    "tr_mean_30",
    "log_volume",
    "volume_change",
    "volume_z_30",
]


@dataclass(frozen=True)
class FeatureConfig:
    horizon: int = 30
    rolling_vol_windows: tuple[int, ...] = (7, 14, 30)
    momentum_windows: tuple[int, ...] = (7, 30, 90)
    drawdown_window: int = 90
    true_range_windows: tuple[int, ...] = (7, 30)
    volume_z_window: int = 30
    include_jerk: bool = False


def resolve_price_column(df: pd.DataFrame, preferred: str = "Adj Close") -> str:
    """Use adjusted close when available, otherwise fall back to close."""
    if preferred in df.columns and df[preferred].notna().any():
        return preferred
    if "Close" in df.columns:
        return "Close"
    raise ValueError("Expected an 'Adj Close' or 'Close' column for price calculations.")


def normalize_ohlcv_frame(raw: pd.DataFrame, asset: str | None = None) -> pd.DataFrame:
    """Return a date-sorted OHLCV frame with a Date column and optional asset id."""
    df = raw.copy()
    if "Date" not in df.columns:
        if df.index.name:
            df = df.reset_index()
        else:
            raise ValueError("Raw data must contain a Date column or a named Date index.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if asset is not None:
        df["asset"] = asset
    elif "asset" not in df.columns:
        df["asset"] = "UNKNOWN"
    return df


def compute_future_realized_volatility(returns: pd.Series, horizon: int) -> pd.Series:
    """
    Align std(v[t+1], ..., v[t+horizon]) at index t.

    The shift happens after a forward-looking label window is formed, keeping the
    result suitable only as a target, never as an input feature.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future_returns = returns.shift(-1)
    return (
        future_returns.rolling(window=horizon, min_periods=horizon)
        .std(ddof=0)
        .shift(-(horizon - 1))
    )


def compute_future_window_end_date(dates: pd.Series, horizon: int) -> pd.Series:
    """Align the final date consumed by a t+1...t+horizon target window."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    return pd.to_datetime(dates).shift(-horizon)


def compute_future_drawdown(price: pd.Series, horizon: int) -> pd.Series:
    """Align min(P[t+1], ..., P[t+horizon]) / P[t] - 1 at index t."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    future_min = (
        price.shift(-1)
        .rolling(window=horizon, min_periods=horizon)
        .min()
        .shift(-(horizon - 1))
    )
    return future_min / price - 1.0


def add_financial_kinematics_features(
    raw: pd.DataFrame,
    config: FeatureConfig | None = None,
    price_column: str = "Adj Close",
) -> pd.DataFrame:
    """
    Add leakage-safe features and future risk targets.

    Feature rows at time t only use values available at or before t. Future
    realized volatility and drawdown are target columns and must not be included
    in feature lists.
    """
    cfg = config or FeatureConfig()
    df = normalize_ohlcv_frame(raw)
    price_col = resolve_price_column(df, preferred=price_column)

    price = df[price_col].astype(float).where(lambda s: s > 0)
    close = df["Close"].astype(float).where(lambda s: s > 0) if "Close" in df else price
    high = df["High"].astype(float) if "High" in df else price
    low = df["Low"].astype(float) if "Low" in df else price

    df["log_price"] = np.log(price)
    df["velocity"] = df["log_price"].diff()
    df["acceleration"] = df["velocity"].diff()

    if cfg.include_jerk:
        df["jerk"] = df["acceleration"].diff()

    for window in cfg.rolling_vol_windows:
        df[f"past_vol_{window}"] = (
            df["velocity"].rolling(window=window, min_periods=window).std(ddof=0)
        )

    for window in cfg.momentum_windows:
        df[f"momentum_{window}"] = df["log_price"] - df["log_price"].shift(window)

    rolling_max = price.rolling(window=cfg.drawdown_window, min_periods=cfg.drawdown_window).max()
    df[f"drawdown_{cfg.drawdown_window}"] = (price - rolling_max) / rolling_max

    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = true_range
    for window in cfg.true_range_windows:
        df[f"tr_mean_{window}"] = true_range.rolling(
            window=window, min_periods=window
        ).mean()

    if "Volume" in df.columns:
        log_volume = np.log(df["Volume"].astype(float).clip(lower=0) + 1.0)
        df["log_volume"] = log_volume
        df["volume_change"] = log_volume.diff()
        volume_mean = log_volume.rolling(
            window=cfg.volume_z_window, min_periods=cfg.volume_z_window
        ).mean()
        volume_std = log_volume.rolling(
            window=cfg.volume_z_window, min_periods=cfg.volume_z_window
        ).std(ddof=0)
        df[f"volume_z_{cfg.volume_z_window}"] = (log_volume - volume_mean) / volume_std
    else:
        df["log_volume"] = np.nan
        df["volume_change"] = np.nan
        df[f"volume_z_{cfg.volume_z_window}"] = np.nan

    df[f"future_vol_{cfg.horizon}"] = compute_future_realized_volatility(
        df["velocity"], cfg.horizon
    )
    df[f"future_vol_{cfg.horizon}_end_date"] = compute_future_window_end_date(
        df["Date"], cfg.horizon
    )
    df[f"future_drawdown_{cfg.horizon}"] = compute_future_drawdown(price, cfg.horizon)
    df[f"future_drawdown_{cfg.horizon}_end_date"] = compute_future_window_end_date(
        df["Date"], cfg.horizon
    )
    return df


def fit_risk_threshold(
    train_frame: pd.DataFrame,
    target_col: str = "future_vol_30",
    quantile: float = 0.75,
) -> float:
    """Fit the high-risk threshold using the training target distribution only."""
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    values = train_frame[target_col].dropna()
    if values.empty:
        raise ValueError(f"No non-null training values found for {target_col}.")
    return float(values.quantile(quantile))


def apply_risk_threshold(
    frame: pd.DataFrame,
    threshold: float,
    target_col: str = "future_vol_30",
    output_col: str = "high_risk",
) -> pd.DataFrame:
    """Apply a train-fitted threshold to any split without refitting it."""
    out = frame.copy()
    out[output_col] = (out[target_col] >= threshold).astype(int)
    out.loc[out[target_col].isna(), output_col] = np.nan
    return out


def get_feature_columns(feature_set: str | Sequence[str]) -> list[str]:
    """Resolve named ablation feature sets to concrete column names."""
    if not isinstance(feature_set, str):
        return list(feature_set)

    normalized = feature_set.strip().lower()
    if normalized in {"log_price_only", "price", "a1"}:
        return ["log_price"]
    if normalized in {"log_price_velocity", "price_velocity", "a2"}:
        return LOG_PRICE_VELOCITY_FEATURES.copy()
    if normalized in {"kinematics", "log_price_velocity_acceleration", "a3"}:
        return KINEMATIC_FEATURES.copy()
    if normalized in {"full", "full_domain", "domain", "a4", "a5", "a6"}:
        return FULL_DOMAIN_FEATURES.copy()
    raise ValueError(f"Unknown feature set: {feature_set}")


def required_columns(feature_columns: Iterable[str], target_columns: Iterable[str]) -> list[str]:
    return list(dict.fromkeys([*feature_columns, *target_columns]))
