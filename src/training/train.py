from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.datasets import PreparedSplits, SequenceDataset, prepare_splits
from src.data.features import get_feature_columns
from src.models.baselines import PredictionBundle
from src.models.lstm import LSTMMultiTask, LSTMVolatilityRegressor
from src.models.multitask_tcn import MultiTaskTCN
from src.models.tcn import TCNVolatilityRegressor
from src.training.evaluate import evaluate_bundles, save_metrics
from src.training.losses import MultiTaskLoss, positive_class_weight
from src.utils.config import ensure_dir, load_config
from src.utils.seed import set_seed


def _require_torch():
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("Install PyTorch first: pip install '.[deep]'") from exc
    return torch, DataLoader


def resolve_device(requested: str = "auto") -> str:
    torch, _ = _require_torch()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build_model(model_config: dict, num_features: int):
    model_type = model_config.get("type", "multitask_tcn")
    multitask = bool(model_config.get("multitask", model_type == "multitask_tcn"))

    if model_type == "lstm":
        if multitask:
            return LSTMMultiTask(
                num_features=num_features,
                hidden_size=int(model_config.get("hidden_size", 64)),
                num_layers=int(model_config.get("num_layers", 1)),
                dropout=float(model_config.get("dropout", 0.0)),
            )
        return LSTMVolatilityRegressor(
            num_features=num_features,
            hidden_size=int(model_config.get("hidden_size", 64)),
            num_layers=int(model_config.get("num_layers", 1)),
            dropout=float(model_config.get("dropout", 0.0)),
        )

    if model_type == "tcn":
        return TCNVolatilityRegressor(
            num_features=num_features,
            hidden_channels=int(model_config.get("hidden_channels", 32)),
            kernel_size=int(model_config.get("kernel_size", 3)),
            dilations=model_config.get("dilations", [1, 2, 4, 8]),
            dropout=float(model_config.get("dropout", 0.2)),
        )

    if model_type == "multitask_tcn":
        return MultiTaskTCN(
            num_features=num_features,
            hidden_channels=int(model_config.get("hidden_channels", 32)),
            kernel_size=int(model_config.get("kernel_size", 3)),
            dilations=model_config.get("dilations", [1, 2, 4, 8]),
            dropout=float(model_config.get("dropout", 0.2)),
            attention=bool(model_config.get("attention", False)),
        )

    raise ValueError(f"Unknown model type: {model_type}")


def _is_multitask_model(model) -> bool:
    return isinstance(model, (MultiTaskTCN, LSTMMultiTask))


def train_regression_model(
    model,
    splits: PreparedSplits,
    training_config: dict,
) -> tuple[object, list[dict[str, float]]]:
    torch, DataLoader = _require_torch()
    device = resolve_device(training_config.get("device", "auto"))
    model = model.to(device)
    train_loader = DataLoader(
        SequenceDataset(splits.train),
        batch_size=int(training_config.get("batch_size", 64)),
        shuffle=False,
    )
    val_loader = DataLoader(
        SequenceDataset(splits.val),
        batch_size=int(training_config.get("batch_size", 64)),
        shuffle=False,
    )
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    patience = int(training_config.get("patience", 5))
    patience_left = patience
    history: list[dict[str, float]] = []

    for epoch in range(1, int(training_config.get("epochs", 30)) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["X"].to(device)
            y = batch["y_vol"].to(device)
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = _regression_loss(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, history


def _regression_loss(model, loader, criterion, device: str) -> float:
    torch, _ = _require_torch()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y = batch["y_vol"].to(device)
            losses.append(float(criterion(model(X), y).detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_multitask_model(
    model,
    splits: PreparedSplits,
    training_config: dict,
) -> tuple[object, list[dict[str, float]]]:
    torch, DataLoader = _require_torch()
    device = resolve_device(training_config.get("device", "auto"))
    model = model.to(device)
    train_loader = DataLoader(
        SequenceDataset(splits.train),
        batch_size=int(training_config.get("batch_size", 64)),
        shuffle=False,
    )
    val_loader = DataLoader(
        SequenceDataset(splits.val),
        batch_size=int(training_config.get("batch_size", 64)),
        shuffle=False,
    )
    pos_weight = positive_class_weight(splits.train.y_regime)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = MultiTaskLoss(
        alpha=float(training_config.get("alpha", 1.0)),
        beta=float(training_config.get("beta", 1.0)),
        pos_weight=pos_weight,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
    )
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    patience = int(training_config.get("patience", 5))
    patience_left = patience
    history: list[dict[str, float]] = []

    for epoch in range(1, int(training_config.get("epochs", 30)) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["X"].to(device)
            y_vol = batch["y_vol"].to(device)
            y_regime = batch["y_regime"].to(device)
            loss, parts = criterion(model(X), y_vol, y_regime)
            loss.backward()
            optimizer.step()
            train_losses.append(parts["loss"])

        val_loss = _multitask_loss(model, val_loader, criterion, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
            }
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return model, history


def _multitask_loss(model, loader, criterion: MultiTaskLoss, device: str) -> float:
    torch, _ = _require_torch()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y_vol = batch["y_vol"].to(device)
            y_regime = batch["y_regime"].to(device)
            loss, _ = criterion(model(X), y_vol, y_regime)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("inf")


def train_model(model, splits: PreparedSplits, training_config: dict):
    if _is_multitask_model(model):
        return train_multitask_model(model, splits, training_config)
    return train_regression_model(model, splits, training_config)


def predict_bundle(model, arrays, model_name: str, device: str = "auto") -> PredictionBundle:
    torch, DataLoader = _require_torch()
    resolved_device = resolve_device(device)
    model = model.to(resolved_device)
    model.eval()
    loader = DataLoader(SequenceDataset(arrays), batch_size=256, shuffle=False)
    vol_preds = []
    regime_scores = []
    regime_preds = []
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(resolved_device)
            outputs = model(X)
            if isinstance(outputs, dict):
                vol = outputs["volatility"]
                score = torch.sigmoid(outputs["regime_logit"])
                vol_preds.append(vol.detach().cpu().numpy())
                regime_scores.append(score.detach().cpu().numpy())
                regime_preds.append((score >= 0.5).long().detach().cpu().numpy())
            else:
                vol_preds.append(outputs.detach().cpu().numpy())

    y_score = np.concatenate(regime_scores) if regime_scores else None
    y_pred = np.concatenate(regime_preds) if regime_preds else None
    return PredictionBundle(
        model=model_name,
        y_vol_true=arrays.y_vol,
        y_vol_pred=np.concatenate(vol_preds),
        y_regime_true=arrays.y_regime.astype(int) if y_score is not None else None,
        y_regime_score=y_score,
        y_regime_pred=y_pred,
    )


def load_processed_frame(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["Date"] = pd.to_datetime(frame["Date"])
    return frame


def run_training(config: dict, processed_path: str | Path):
    set_seed(int(config.get("project", {}).get("seed", 42)))
    features_cfg = config.get("features", {})
    target_cfg = config.get("target", {})
    horizon = int(target_cfg.get("horizon", 30))
    feature_columns = get_feature_columns(features_cfg.get("feature_set", "full_domain"))
    frame = load_processed_frame(processed_path)
    splits = prepare_splits(
        frame=frame,
        feature_columns=feature_columns,
        split_config=config.get("splits", {}),
        sequence_length=int(features_cfg.get("sequence_length", 60)),
        target_col=f"future_vol_{horizon}",
        regime_quantile=float(target_cfg.get("regime_quantile", 0.75)),
    )
    if splits.train.empty or splits.val.empty or splits.test.empty:
        raise ValueError("One or more sequence splits are empty. Check dates and sequence length.")

    model = build_model(config.get("model", {}), num_features=len(feature_columns))
    model, history = train_model(model, splits, config.get("training", {}))
    model_name = config.get("model", {}).get("type", "model")
    bundle = predict_bundle(
        model,
        splits.test,
        model_name=model_name,
        device=config.get("training", {}).get("device", "auto"),
    )
    metrics = evaluate_bundles([bundle])

    paths = config.get("paths", {})
    tables_dir = ensure_dir(paths.get("tables_dir", "results/tables"))
    checkpoints_dir = ensure_dir(paths.get("checkpoints_dir", "results/checkpoints"))
    save_metrics(metrics, tables_dir / f"{model_name}_test_metrics.csv")
    pd.DataFrame(history).to_csv(tables_dir / f"{model_name}_history.csv", index=False)

    torch, _ = _require_torch()
    torch.save(model.state_dict(), checkpoints_dir / f"{model_name}.pt")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a leakage-free sequence model.")
    parser.add_argument("--config", default="configs/multitask_tcn.yaml")
    parser.add_argument("--processed", default="data/processed/features_all.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = run_training(config, args.processed)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
