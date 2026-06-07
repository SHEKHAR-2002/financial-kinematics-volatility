import numpy as np
import pandas as pd
import pytest

from src.data.features import (
    FeatureConfig,
    add_financial_kinematics_features,
    apply_risk_threshold,
    compute_future_drawdown,
    compute_future_realized_volatility,
    fit_risk_threshold,
)


def test_future_realized_volatility_is_aligned_to_t_plus_one_window():
    returns = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])

    result = compute_future_realized_volatility(returns, horizon=2)

    assert result.iloc[0] == 0.5
    assert result.iloc[1] == 0.5
    assert result.iloc[2] == 0.5
    assert np.isnan(result.iloc[3])
    assert np.isnan(result.iloc[4])


def test_future_drawdown_uses_only_next_window_as_label():
    price = pd.Series([100.0, 90.0, 80.0, 120.0])

    result = compute_future_drawdown(price, horizon=2)

    assert result.iloc[0] == pytest.approx(-0.2)
    assert result.iloc[1] == pytest.approx(-1.0 / 9.0)
    assert np.isnan(result.iloc[2])
    assert np.isnan(result.iloc[3])


def test_risk_threshold_is_fit_on_train_only():
    train = pd.DataFrame({"future_vol_30": [1.0, 2.0, 3.0, 4.0]})
    validation = pd.DataFrame({"future_vol_30": [10.0]})

    threshold = fit_risk_threshold(train, quantile=0.75)
    labeled = apply_risk_threshold(validation, threshold)

    assert threshold == 3.25
    assert labeled.loc[0, "high_risk"] == 1


def test_feature_engineering_does_not_add_future_vol_to_feature_columns():
    dates = pd.date_range("2020-01-01", periods=12, freq="D")
    price = np.exp(np.linspace(10.0, 10.5, len(dates)))
    raw = pd.DataFrame(
        {
            "Date": dates,
            "Open": price,
            "High": price + 1.0,
            "Low": price - 1.0,
            "Close": price,
            "Adj Close": price,
            "Volume": np.arange(1, len(dates) + 1) * 100,
            "asset": "TEST",
        }
    )

    engineered = add_financial_kinematics_features(
        raw,
        FeatureConfig(
            horizon=3,
            rolling_vol_windows=(2,),
            momentum_windows=(2,),
            drawdown_window=2,
            true_range_windows=(2,),
            volume_z_window=2,
        ),
    )

    assert "future_vol_3" in engineered.columns
    assert "log_price" in engineered.columns
    assert engineered["log_price"].iloc[0] == np.log(price[0])
