from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight


@dataclass
class PredictionBundle:
    model: str
    y_vol_true: np.ndarray
    y_vol_pred: np.ndarray
    y_regime_true: np.ndarray | None = None
    y_regime_score: np.ndarray | None = None
    y_regime_pred: np.ndarray | None = None
    regime_threshold: float | None = None


def naive_volatility_predictions(
    target_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    target_col: str = "future_vol_30",
    past_vol_col: str = "past_vol_30",
    regime_col: str = "high_risk",
    quantile: float = 0.75,
) -> PredictionBundle:
    """Mandatory baseline: current past volatility predicts future volatility."""
    usable = target_frame.dropna(subset=[target_col, past_vol_col, regime_col])
    threshold = float(train_frame[past_vol_col].dropna().quantile(quantile))
    y_score = usable[past_vol_col].astype(float).to_numpy()
    return PredictionBundle(
        model="naive_past_volatility",
        y_vol_true=usable[target_col].astype(float).to_numpy(),
        y_vol_pred=y_score,
        y_regime_true=usable[regime_col].astype(int).to_numpy(),
        y_regime_score=y_score,
        y_regime_pred=(y_score >= threshold).astype(int),
        regime_threshold=threshold,
    )


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray, default: float = 0.5) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2 or y_score.size == 0:
        return default

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return default
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    if not np.isfinite(f1s).any():
        return default
    return float(thresholds[int(np.nanargmax(f1s))])


def inverse_scaled_feature(
    arrays,
    feature_columns: list[str],
    scaler,
    feature_name: str,
) -> np.ndarray:
    if feature_name not in feature_columns:
        raise ValueError(f"{feature_name} is required for the naive volatility baseline.")
    feature_idx = feature_columns.index(feature_name)
    latest_scaled = arrays.X[:, -1, feature_idx].astype(float)
    return latest_scaled * float(scaler.scale_[feature_idx]) + float(scaler.mean_[feature_idx])


def naive_volatility_predictions_from_arrays(
    eval_arrays,
    feature_columns: list[str],
    scaler,
    val_arrays=None,
    past_vol_col: str = "past_vol_30",
) -> PredictionBundle:
    """Mandatory baseline evaluated on the exact prepared sequence rows."""
    y_score = inverse_scaled_feature(eval_arrays, feature_columns, scaler, past_vol_col)
    threshold = float(np.nanmedian(y_score)) if y_score.size else 0.0
    if val_arrays is not None and not val_arrays.empty:
        val_score = inverse_scaled_feature(val_arrays, feature_columns, scaler, past_vol_col)
        threshold = find_optimal_threshold(val_arrays.y_regime, val_score, default=threshold)
    return PredictionBundle(
        model="naive_past_volatility",
        y_vol_true=eval_arrays.y_vol,
        y_vol_pred=y_score,
        y_regime_true=eval_arrays.y_regime.astype(int),
        y_regime_score=y_score,
        y_regime_pred=(y_score >= threshold).astype(int),
        regime_threshold=threshold,
    )


def fit_classical_regressors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> dict[str, object]:
    return {
        "ridge": Ridge(alpha=1.0).fit(X_train, y_train),
        "lasso": Lasso(alpha=0.0001, max_iter=10000).fit(X_train, y_train),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            n_jobs=-1,
        ).fit(X_train, y_train),
        "gradient_boosting_regressor": GradientBoostingRegressor(
            random_state=random_state
        ).fit(X_train, y_train),
    }


def fit_classical_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
) -> dict[str, object]:
    unique = np.unique(y_train)
    if len(unique) < 2:
        return {}
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ).fit(X_train, y_train),
        "random_forest_classifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        ).fit(X_train, y_train),
        "gradient_boosting_classifier": GradientBoostingClassifier(
            random_state=random_state
        ).fit(X_train, y_train, sample_weight=sample_weight),
    }


def classifier_scores(classifier, X: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(X)[:, 1]
    return classifier.decision_function(X)


def classical_prediction_bundles(
    X_train: np.ndarray,
    y_vol_train: np.ndarray,
    y_regime_train: np.ndarray,
    X_val: np.ndarray,
    y_regime_val: np.ndarray,
    X_eval: np.ndarray,
    y_vol_eval: np.ndarray,
    y_regime_eval: np.ndarray,
    random_state: int = 42,
) -> list[PredictionBundle]:
    regressors = fit_classical_regressors(X_train, y_vol_train, random_state)
    classifiers = fit_classical_classifiers(X_train, y_regime_train.astype(int), random_state)

    bundles = []
    for model_name, regressor in regressors.items():
        classifier_name = model_name.replace("_regressor", "_classifier")
        classifier = classifiers.get(classifier_name)
        if model_name in {"ridge", "lasso"}:
            classifier = classifiers.get("logistic_regression")
        score = pred = None
        threshold = None
        if classifier is not None:
            score = classifier_scores(classifier, X_eval)
            val_score = classifier_scores(classifier, X_val)
            threshold = find_optimal_threshold(y_regime_val, val_score)
            pred = (score >= threshold).astype(int)
        bundles.append(
            PredictionBundle(
                model=model_name,
                y_vol_true=y_vol_eval,
                y_vol_pred=regressor.predict(X_eval),
                y_regime_true=y_regime_eval.astype(int),
                y_regime_score=score,
                y_regime_pred=pred,
                regime_threshold=threshold,
            )
        )
    return bundles
