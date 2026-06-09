from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.models.baselines import PredictionBundle


def _safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson": _safe_corr(y_true, y_pred),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    if y_pred is None:
        if y_score is None:
            raise ValueError("classification_metrics needs y_score or y_pred.")
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics.update(
        {
            "tn": float(cm[0, 0]),
            "fp": float(cm[0, 1]),
            "fn": float(cm[1, 0]),
            "tp": float(cm[1, 1]),
        }
    )
    return metrics


def evaluate_bundle(bundle: PredictionBundle) -> dict[str, float | str]:
    metrics: dict[str, float | str] = {"model": bundle.model}
    metrics.update(regression_metrics(bundle.y_vol_true, bundle.y_vol_pred))
    if bundle.regime_threshold is not None:
        metrics["regime_threshold"] = float(bundle.regime_threshold)
    if bundle.y_regime_true is not None and (
        bundle.y_regime_score is not None or bundle.y_regime_pred is not None
    ):
        class_metrics = classification_metrics(
            bundle.y_regime_true,
            y_score=bundle.y_regime_score,
            y_pred=bundle.y_regime_pred,
        )
        metrics.update({f"regime_{key}": value for key, value in class_metrics.items()})
    return metrics


def evaluate_bundles(bundles: list[PredictionBundle]) -> pd.DataFrame:
    return pd.DataFrame([evaluate_bundle(bundle) for bundle in bundles])


def save_metrics(metrics: pd.DataFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output, index=False)
