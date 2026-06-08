from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


@dataclass
class PredictionBundle:
    model: str
    y_vol_true: np.ndarray
    y_vol_pred: np.ndarray
    y_regime_true: np.ndarray | None = None
    y_regime_score: np.ndarray | None = None
    y_regime_pred: np.ndarray | None = None


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
        ).fit(X_train, y_train),
    }


def classical_prediction_bundles(
    X_train: np.ndarray,
    y_vol_train: np.ndarray,
    y_regime_train: np.ndarray,
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
        if classifier is not None:
            if hasattr(classifier, "predict_proba"):
                score = classifier.predict_proba(X_eval)[:, 1]
            else:
                score = classifier.decision_function(X_eval)
            pred = classifier.predict(X_eval)
        bundles.append(
            PredictionBundle(
                model=model_name,
                y_vol_true=y_vol_eval,
                y_vol_pred=regressor.predict(X_eval),
                y_regime_true=y_regime_eval.astype(int),
                y_regime_score=score,
                y_regime_pred=pred,
            )
        )
    return bundles
