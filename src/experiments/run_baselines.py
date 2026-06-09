from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.datasets import (
    flattened_window_tabular,
    latest_step_tabular,
    prepare_splits,
)
from src.data.features import get_feature_columns
from src.models.baselines import (
    classical_prediction_bundles,
    naive_volatility_predictions_from_arrays,
)
from src.training.evaluate import evaluate_bundles, save_metrics
from src.utils.config import ensure_dir, load_config


def load_processed_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def run_baselines(config: dict, processed_path: str | Path) -> pd.DataFrame:
    features_cfg = config.get("features", {})
    target_cfg = config.get("target", {})
    horizon = int(target_cfg.get("horizon", 30))
    target_col = f"future_vol_{horizon}"
    past_vol_col = f"past_vol_{horizon}"
    feature_columns = get_feature_columns(features_cfg.get("feature_set", "full_domain"))
    sequence_length = int(features_cfg.get("sequence_length", 60))
    quantile = float(target_cfg.get("regime_quantile", 0.75))

    frame = load_processed_frame(processed_path)
    splits = prepare_splits(
        frame=frame,
        feature_columns=feature_columns,
        split_config=config.get("splits", {}),
        sequence_length=sequence_length,
        target_col=target_col,
        regime_quantile=quantile,
    )
    bundles = [
        naive_volatility_predictions_from_arrays(
            eval_arrays=splits.test,
            feature_columns=feature_columns,
            scaler=splits.feature_scaler,
            val_arrays=splits.val,
            past_vol_col=past_vol_col,
        )
    ]
    if config.get("experiments", {}).get("tabular_window", "latest") == "flattened":
        x_train, y_vol_train, y_regime_train = flattened_window_tabular(splits.train)
        x_val, _, y_regime_val = flattened_window_tabular(splits.val)
        x_test, y_vol_test, y_regime_test = flattened_window_tabular(splits.test)
    else:
        x_train, y_vol_train, y_regime_train = latest_step_tabular(splits.train)
        x_val, _, y_regime_val = latest_step_tabular(splits.val)
        x_test, y_vol_test, y_regime_test = latest_step_tabular(splits.test)

    bundles.extend(
        classical_prediction_bundles(
            X_train=x_train,
            y_vol_train=y_vol_train,
            y_regime_train=y_regime_train,
            X_val=x_val,
            y_regime_val=y_regime_val,
            X_eval=x_test,
            y_vol_eval=y_vol_test,
            y_regime_eval=y_regime_test,
            random_state=int(config.get("project", {}).get("seed", 42)),
        )
    )
    metrics = evaluate_bundles(bundles)
    tables_dir = ensure_dir(config.get("paths", {}).get("tables_dir", "results/tables"))
    save_metrics(metrics, tables_dir / "baseline_metrics.csv")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run naive and classical baselines.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed", default="data/processed/features_all.csv")
    args = parser.parse_args()

    metrics = run_baselines(load_config(args.config), args.processed)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
