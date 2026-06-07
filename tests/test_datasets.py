import numpy as np
import pandas as pd

from src.data.datasets import SplitConfig, chronological_split, make_sequences, prepare_splits


def test_sequence_windows_do_not_cross_split_boundaries():
    frame = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "asset": ["A"] * 6,
            "f": np.arange(6, dtype=float),
            "target": np.arange(6, dtype=float),
            "high_risk": [0, 0, 1, 0, 1, 1],
        }
    )
    splits = chronological_split(
        frame,
        SplitConfig(
            train_start="2020-01-01",
            train_end="2020-01-03",
            val_start="2020-01-04",
            val_end="2020-01-06",
            test_start="2020-01-04",
            test_end="2020-01-06",
        ),
    )

    train_seq = make_sequences(
        splits["train"],
        feature_columns=["f"],
        sequence_length=3,
        target_col="target",
    )
    val_seq = make_sequences(
        splits["val"],
        feature_columns=["f"],
        sequence_length=3,
        target_col="target",
    )

    assert train_seq.X[:, :, 0].tolist() == [[0.0, 1.0, 2.0]]
    assert val_seq.X[:, :, 0].tolist() == [[3.0, 4.0, 5.0]]


def test_feature_scaler_is_fit_on_train_split_only():
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]
            ),
            "asset": ["A", "A", "A", "A"],
            "f": [0.0, 2.0, 100.0, 200.0],
            "future_vol_30": [0.1, 0.2, 0.3, 0.4],
        }
    )

    prepared = prepare_splits(
        frame,
        feature_columns=["f"],
        split_config=SplitConfig(
            train_start="2020-01-01",
            train_end="2020-01-02",
            val_start="2020-01-03",
            val_end="2020-01-03",
            test_start="2020-01-04",
            test_end="2020-01-04",
        ),
        sequence_length=1,
    )

    assert prepared.frames["train"]["f"].tolist() == [-1.0, 1.0]
    assert prepared.frames["val"].loc[0, "f"] == 99.0
    assert prepared.frames["test"].loc[0, "f"] == 199.0
