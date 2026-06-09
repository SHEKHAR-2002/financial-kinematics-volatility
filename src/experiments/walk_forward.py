from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.data.datasets import SplitConfig, prepare_splits
from src.data.features import get_feature_columns
from src.training.evaluate import evaluate_bundles, save_metrics
from src.training.train import (
    DEFAULT_REGIME_THRESHOLD,
    build_model,
    load_processed_frame,
    model_artifact_name,
    predict_bundle,
    train_model,
    tune_regime_threshold,
)
from src.utils.config import ensure_dir, load_config
from src.utils.seed import set_seed


def expanding_year_folds(
    train_start: str = "2008-01-01",
    first_test_year: int = 2016,
    last_test_year: int = 2024,
) -> list[SplitConfig]:
    folds = []
    for test_year in range(first_test_year, last_test_year + 1):
        folds.append(
            SplitConfig(
                train_start=train_start,
                train_end=f"{test_year - 2}-12-31",
                val_start=f"{test_year - 1}-01-01",
                val_end=f"{test_year - 1}-12-31",
                test_start=f"{test_year}-01-01",
                test_end=f"{test_year}-12-31",
            )
        )
    return folds


def run_walk_forward(config: dict, processed_path: str | Path) -> pd.DataFrame:
    frame = load_processed_frame(processed_path)
    features_cfg = config.get("features", {})
    target_cfg = config.get("target", {})
    feature_columns = get_feature_columns(features_cfg.get("feature_set", "full_domain"))
    sequence_length = int(features_cfg.get("sequence_length", 60))
    target_col = f"future_vol_{int(target_cfg.get('horizon', 30))}"
    model_name = model_artifact_name(config.get("model", {}))
    max_date = pd.to_datetime(frame["Date"]).max()
    last_test_year = min(max_date.year, int(config.get("walk_forward", {}).get("last_test_year", max_date.year)))
    first_test_year = int(config.get("walk_forward", {}).get("first_test_year", 2016))
    rows = []

    for fold_idx, split_cfg in enumerate(
        expanding_year_folds(
            train_start=config.get("splits", {}).get("train_start", "2008-01-01"),
            first_test_year=first_test_year,
            last_test_year=last_test_year,
        ),
        start=1,
    ):
        cfg = deepcopy(config)
        cfg["splits"] = split_cfg.__dict__
        set_seed(int(cfg.get("project", {}).get("seed", 42)))
        splits = prepare_splits(
            frame=frame,
            feature_columns=feature_columns,
            split_config=split_cfg,
            sequence_length=sequence_length,
            target_col=target_col,
            regime_quantile=float(target_cfg.get("regime_quantile", 0.75)),
        )
        if splits.train.empty or splits.val.empty or splits.test.empty:
            rows.append(
                {
                    "fold": fold_idx,
                    "test_start": split_cfg.test_start,
                    "test_end": split_cfg.test_end,
                    "status": "empty_split",
                }
            )
            continue

        model = build_model(cfg.get("model", {}), num_features=len(feature_columns))
        model, _ = train_model(model, splits, cfg.get("training", {}))
        model_cfg = cfg.get("model", {})
        regime_threshold = DEFAULT_REGIME_THRESHOLD
        if bool(
            model_cfg.get(
                "multitask",
                model_cfg.get("type") in {"multitask_lstm", "multitask_tcn"},
            )
        ):
            regime_threshold = tune_regime_threshold(
                model,
                splits.val,
                device=cfg.get("training", {}).get("device", "auto"),
            )
        bundle = predict_bundle(
            model,
            splits.test,
            model_name=model_name,
            device=cfg.get("training", {}).get("device", "auto"),
            regime_threshold=regime_threshold,
        )
        metrics = evaluate_bundles([bundle]).iloc[0].to_dict()
        if bundle.y_regime_score is not None:
            metrics["regime_threshold"] = regime_threshold
        metrics.update(
            {
                "fold": fold_idx,
                "test_start": split_cfg.test_start,
                "test_end": split_cfg.test_end,
                "status": "ok",
            }
        )
        rows.append(metrics)

    results = pd.DataFrame(rows)
    tables_dir = ensure_dir(config.get("paths", {}).get("tables_dir", "results/tables"))
    save_metrics(results, tables_dir / f"{model_name}_walk_forward_metrics.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run expanding walk-forward validation.")
    parser.add_argument("--config", default="configs/lstm.yaml")
    parser.add_argument("--processed", default="data/processed/features_all.csv")
    args = parser.parse_args()

    results = run_walk_forward(load_config(args.config), args.processed)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
