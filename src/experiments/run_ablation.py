from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from src.data.datasets import prepare_splits
from src.data.features import get_feature_columns
from src.training.evaluate import evaluate_bundles, save_metrics
from src.training.train import build_model, load_processed_frame, predict_bundle, train_model
from src.utils.config import ensure_dir, load_config
from src.utils.seed import set_seed


ABLATIONS = {
    "A1": {"feature_set": "log_price_only", "model": {"type": "tcn", "multitask": False}},
    "A2": {"feature_set": "log_price_velocity", "model": {"type": "tcn", "multitask": False}},
    "A3": {"feature_set": "kinematics", "model": {"type": "tcn", "multitask": False}},
    "A4": {"feature_set": "full_domain", "model": {"type": "tcn", "multitask": False}},
    "A5": {"feature_set": "full_domain", "model": {"type": "multitask_tcn", "attention": False}},
    "A6": {"feature_set": "full_domain", "model": {"type": "multitask_tcn", "attention": True}},
}


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
                    "model": model_cfg["type"],
                    "feature_set": ablation["feature_set"],
                    "status": "empty_split",
                }
            )
            continue

        model = build_model(model_cfg, num_features=len(feature_columns))
        model, _ = train_model(model, splits, cfg.get("training", {}))
        bundle = predict_bundle(
            model,
            splits.test,
            model_name=f"{ablation_id}_{model_cfg['type']}",
            device=cfg.get("training", {}).get("device", "auto"),
        )
        metrics = evaluate_bundles([bundle]).iloc[0].to_dict()
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
    parser = argparse.ArgumentParser(description="Run A1-A6 feature/model ablations.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--processed", default="data/processed/features_all.csv")
    args = parser.parse_args()

    results = run_ablation(load_config(args.config), args.processed)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
