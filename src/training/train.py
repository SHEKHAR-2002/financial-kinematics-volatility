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


DEFAULT_REGIME_THRESHOLD = 0.5


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


def _progress_enabled(training_config: dict) -> bool:
    return bool(training_config.get("print_progress", True))


def _current_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _gradient_clip_norm(training_config: dict) -> float | None:
    value = training_config.get("gradient_clip_norm", 1.0)
    if value is None:
        return None
    value = float(value)
    return value if value > 0 else None


def _build_scheduler(torch, optimizer, training_config: dict):
    scheduler_config = training_config.get("scheduler", {"type": "reduce_on_plateau"})
    if scheduler_config in (None, False):
        return None
    if scheduler_config is True:
        scheduler_config = {"type": "reduce_on_plateau"}
    if isinstance(scheduler_config, str):
        scheduler_config = {"type": scheduler_config}

    scheduler_type = str(scheduler_config.get("type", "reduce_on_plateau")).lower()
    if scheduler_type in {"none", "off", "disabled"}:
        return None
    if scheduler_type not in {"reduce_on_plateau", "reduce_lr_on_plateau"}:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(scheduler_config.get("factor", 0.5)),
        patience=int(scheduler_config.get("patience", 5)),
        min_lr=float(scheduler_config.get("min_lr", 1e-6)),
    )


def _print_training_start(
    model,
    split_name: str,
    splits: PreparedSplits,
    device: str,
    epochs: int,
    training_config: dict,
) -> None:
    if not _progress_enabled(training_config):
        return
    print(
        f"Training {type(model).__name__} ({split_name}) on {device}: "
        f"epochs={epochs}, train={len(splits.train.y_vol)}, "
        f"val={len(splits.val.y_vol)}",
        flush=True,
    )


def _print_epoch_progress(
    model,
    epoch: int,
    epochs: int,
    row: dict[str, float],
    improved: bool,
    training_config: dict,
) -> None:
    if not _progress_enabled(training_config):
        return
    status = "improved" if improved else f"patience_left={int(row['patience_left'])}"
    parts = [
        f"{type(model).__name__} epoch {epoch:03d}/{epochs}",
        f"train_loss={row['train_loss']:.6g}",
        f"val_loss={row['val_loss']:.6g}",
        f"lr={row['lr']:.3g}",
        status,
    ]
    if "train_volatility_loss" in row and "train_regime_loss" in row:
        parts.extend(
            [
                f"train_vol={row['train_volatility_loss']:.6g}",
                f"train_regime={row['train_regime_loss']:.6g}",
                f"val_vol={row['val_volatility_loss']:.6g}",
                f"val_regime={row['val_regime_loss']:.6g}",
            ]
        )
    print(" | ".join(parts), flush=True)


def _append_loss_parts(target: dict[str, list[float]], parts: dict[str, float]) -> None:
    for key, value in parts.items():
        target.setdefault(key, []).append(float(value))


def _average_loss_parts(parts: dict[str, list[float]]) -> dict[str, float]:
    return {key: float(np.mean(values)) for key, values in parts.items()}


def find_optimal_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Select a regime threshold on validation data by maximizing F1."""
    from sklearn.metrics import precision_recall_curve

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2 or y_score.size == 0:
        return DEFAULT_REGIME_THRESHOLD

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return DEFAULT_REGIME_THRESHOLD

    precisions = precisions[:-1]
    recalls = recalls[:-1]
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    if not np.isfinite(f1s).any():
        return DEFAULT_REGIME_THRESHOLD
    return float(thresholds[int(np.nanargmax(f1s))])


def tune_regime_threshold(model, arrays, device: str = "auto") -> float:
    """Tune a classification threshold without touching the test split."""
    validation_bundle = predict_bundle(
        model,
        arrays,
        model_name="validation_threshold",
        device=device,
        regime_threshold=DEFAULT_REGIME_THRESHOLD,
    )
    if validation_bundle.y_regime_score is None or validation_bundle.y_regime_true is None:
        return DEFAULT_REGIME_THRESHOLD
    return find_optimal_threshold(
        validation_bundle.y_regime_true,
        validation_bundle.y_regime_score,
    )


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
        shuffle=True,
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
    scheduler = _build_scheduler(torch, optimizer, training_config)
    grad_clip_norm = _gradient_clip_norm(training_config)
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    patience = int(training_config.get("patience", 5))
    patience_left = patience
    history: list[dict[str, float]] = []
    epochs = int(training_config.get("epochs", 30))

    _print_training_start(model, "regression", splits, device, epochs, training_config)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["X"].to(device)
            y = batch["y_vol"].to(device)
            loss = criterion(model(X), y)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = _regression_loss(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)

        improved = val_loss < best_val
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "lr": _current_lr(optimizer),
        }
        if improved:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
        row["best_val_loss"] = best_val
        row["patience_left"] = float(patience_left)
        history.append(row)
        _print_epoch_progress(model, epoch, epochs, row, improved, training_config)
        if patience_left <= 0:
            if _progress_enabled(training_config):
                print(f"Early stopping {type(model).__name__} at epoch {epoch}.", flush=True)
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
        shuffle=True,
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
    scheduler = _build_scheduler(torch, optimizer, training_config)
    grad_clip_norm = _gradient_clip_norm(training_config)
    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    patience = int(training_config.get("patience", 5))
    patience_left = patience
    history: list[dict[str, float]] = []
    epochs = int(training_config.get("epochs", 30))

    _print_training_start(model, "multi-task", splits, device, epochs, training_config)

    for epoch in range(1, epochs + 1):
        model.train()
        train_parts: dict[str, list[float]] = {}
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch["X"].to(device)
            y_vol = batch["y_vol"].to(device)
            y_regime = batch["y_regime"].to(device)
            loss, parts = criterion(model(X), y_vol, y_regime)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            _append_loss_parts(train_parts, parts)

        train_summary = _average_loss_parts(train_parts)
        val_summary = _multitask_loss_parts(model, val_loader, criterion, device)
        val_loss = val_summary["loss"]
        if scheduler is not None:
            scheduler.step(val_loss)

        improved = val_loss < best_val
        row = {
            "epoch": float(epoch),
            "train_loss": train_summary["loss"],
            "val_loss": val_loss,
            "lr": _current_lr(optimizer),
        }
        for key, value in train_summary.items():
            if key != "loss":
                row[f"train_{key}"] = value
        for key, value in val_summary.items():
            if key != "loss":
                row[f"val_{key}"] = value

        if improved:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience_left = patience
        else:
            patience_left -= 1
        row["best_val_loss"] = best_val
        row["patience_left"] = float(patience_left)
        history.append(row)
        _print_epoch_progress(model, epoch, epochs, row, improved, training_config)
        if patience_left <= 0:
            if _progress_enabled(training_config):
                print(f"Early stopping {type(model).__name__} at epoch {epoch}.", flush=True)
            break

    model.load_state_dict(best_state)
    return model, history


def _multitask_loss_parts(
    model,
    loader,
    criterion: MultiTaskLoss,
    device: str,
) -> dict[str, float]:
    torch, _ = _require_torch()
    model.eval()
    loss_parts: dict[str, list[float]] = {}
    with torch.no_grad():
        for batch in loader:
            X = batch["X"].to(device)
            y_vol = batch["y_vol"].to(device)
            y_regime = batch["y_regime"].to(device)
            loss, parts = criterion(model(X), y_vol, y_regime)
            parts["loss"] = float(loss.detach().cpu())
            _append_loss_parts(loss_parts, parts)
    return _average_loss_parts(loss_parts) if loss_parts else {"loss": float("inf")}


def train_model(model, splits: PreparedSplits, training_config: dict):
    if _is_multitask_model(model):
        return train_multitask_model(model, splits, training_config)
    return train_regression_model(model, splits, training_config)


def predict_bundle(
    model,
    arrays,
    model_name: str,
    device: str = "auto",
    regime_threshold: float = DEFAULT_REGIME_THRESHOLD,
) -> PredictionBundle:
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
                regime_preds.append((score >= regime_threshold).long().detach().cpu().numpy())
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
    training_cfg = config.get("training", {})
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
    if _progress_enabled(training_cfg):
        print(
            f"Prepared splits: features={len(feature_columns)}, "
            f"risk_threshold={splits.risk_threshold:.6g}, "
            f"train={len(splits.train.y_vol)}, val={len(splits.val.y_vol)}, "
            f"test={len(splits.test.y_vol)}",
            flush=True,
        )

    model, history = train_model(model, splits, training_cfg)
    model_name = config.get("model", {}).get("type", "model")
    regime_threshold = DEFAULT_REGIME_THRESHOLD
    if _is_multitask_model(model):
        regime_threshold = tune_regime_threshold(
            model,
            splits.val,
            device=training_cfg.get("device", "auto"),
        )
        if _progress_enabled(training_cfg):
            print(
                f"Selected validation F1 regime threshold for {model_name}: "
                f"{regime_threshold:.6g}",
                flush=True,
            )
    bundle = predict_bundle(
        model,
        splits.test,
        model_name=model_name,
        device=training_cfg.get("device", "auto"),
        regime_threshold=regime_threshold,
    )
    metrics = evaluate_bundles([bundle])
    if bundle.y_regime_score is not None:
        metrics["regime_threshold"] = regime_threshold

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
