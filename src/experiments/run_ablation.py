from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.data.datasets import prepare_splits
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


ABLATIONS = {
    "A1": {"feature_set": "log_price_only", "model": {"type": "lstm", "multitask": False}},
    "A2": {"feature_set": "log_price_velocity", "model": {"type": "lstm", "multitask": False}},
    "A3": {"feature_set": "kinematics", "model": {"type": "lstm", "multitask": False}},
    "A4": {"feature_set": "full_domain", "model": {"type": "lstm", "multitask": False}},
}


def is_multitask_config(model_cfg: dict) -> bool:
    model_type = model_cfg.get("type")
    return bool(model_cfg.get("multitask", model_type in {"multitask_lstm", "multitask_tcn"}))


def run_ablation(config: dict, processed_path: str | Path) -> pd.DataFrame:
    frame = load_processed_frame(processed_path)
    rows = []
    target_cfg = config.get("target", {})
    features_cfg = config.get("features", {})
    target_col = f"future_vol_{int(target_cfg.get('horizon', 30))}"
    sequence_length = int(features_cfg.get("sequence_length", 60))

    for ablation_id, ablation in ABLATIONS.items():
        cfg = deepcopy(config)
        cfg.setdefault("features", {})["feature_set"] = ablation["feature_set"]
        model_cfg = deepcopy(config.get("model", {}))
        model_cfg.update(ablation["model"])
        model_cfg["name"] = f"{ablation_id}_{model_cfg['type']}"
        cfg["model"] = model_cfg

        set_seed(int(cfg.get("project", {}).get("seed", 42)))
        feature_columns = get_feature_columns(ablation["feature_set"])
        splits = prepare_splits(
            frame=frame,
            feature_columns=feature_columns,
            split_config=cfg.get("splits", {}),
            sequence_length=sequence_length,
            target_col=target_col,
            regime_quantile=float(target_cfg.get("regime_quantile", 0.75)),
        )
        if splits.train.empty or splits.val.empty or splits.test.empty:
            rows.append(
                {
                    "ablation": ablation_id,
                    "model": model_artifact_name(model_cfg),
                    "feature_set": ablation["feature_set"],
                    "status": "empty_split",
                }
            )
            continue

        model = build_model(model_cfg, num_features=len(feature_columns))
        model, _ = train_model(model, splits, cfg.get("training", {}))
        regime_threshold = DEFAULT_REGIME_THRESHOLD
        if is_multitask_config(model_cfg):
            regime_threshold = tune_regime_threshold(
                model,
                splits.val,
                device=cfg.get("training", {}).get("device", "auto"),
            )
        bundle = predict_bundle(
            model,
            splits.test,
            model_name=model_artifact_name(model_cfg),
            device=cfg.get("training", {}).get("device", "auto"),
            regime_threshold=regime_threshold,
        )
        metrics = evaluate_bundles([bundle]).iloc[0].to_dict()
        if bundle.y_regime_score is not None:
            metrics["regime_threshold"] = regime_threshold
        metrics.update(
            {
                "ablation": ablation_id,
                "feature_set": ablation["feature_set"],
                "status": "ok",
            }
        )
        rows.append(metrics)

    results = pd.DataFrame(rows)
    tables_dir = ensure_dir(config.get("paths", {}).get("tables_dir", "results/tables"))
    save_metrics(results, tables_dir / "ablation_metrics.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run A1-A4 LSTM feature ablations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed", default="data/processed/features_all.csv")
    args = parser.parse_args()

    results = run_ablation(load_config(args.config), args.processed)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
