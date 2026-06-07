from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _prepare_matplotlib():
    import matplotlib.pyplot as plt

    return plt


def plot_price_return_overview(frame: pd.DataFrame, output_path: str | Path) -> None:
    plt = _prepare_matplotlib()
    df = frame.sort_values("Date")
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    price_col = "Adj Close" if "Adj Close" in df.columns else "Close"
    axes[0].plot(df["Date"], df[price_col], linewidth=1.1)
    axes[0].set_ylabel("Price")
    axes[1].plot(df["Date"], df["velocity"], linewidth=0.8)
    axes[1].set_ylabel("Log return")
    axes[1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_kinematics(frame: pd.DataFrame, output_path: str | Path) -> None:
    plt = _prepare_matplotlib()
    df = frame.sort_values("Date")
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(df["Date"], df["log_price"], linewidth=1.0)
    axes[0].set_ylabel("Log price")
    axes[1].plot(df["Date"], df["velocity"], linewidth=0.8)
    axes[1].set_ylabel("Velocity")
    axes[2].plot(df["Date"], df["acceleration"], linewidth=0.8)
    axes[2].set_ylabel("Acceleration")
    axes[2].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_volatility_actual_vs_predicted(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> None:
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, y_true, label="Actual future volatility", linewidth=1.1)
    ax.plot(dates, y_pred, label="Predicted", linewidth=1.1)
    ax.set_ylabel("30-day realized volatility")
    ax.set_xlabel("Date")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_confusion_matrix(matrix: np.ndarray, output_path: str | Path) -> None:
    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(4, 4))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Pred normal", "Pred high"])
    ax.set_yticks([0, 1], labels=["True normal", "True high"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, metric: str, output_path: str | Path) -> None:
    plt = _prepare_matplotlib()
    ordered = metrics.sort_values(metric)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.barh(ordered["model"], ordered[metric])
    ax.set_xlabel(metric)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
