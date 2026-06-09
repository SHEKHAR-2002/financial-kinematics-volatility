# Known Limitations and Next Work

## Current Empirical Summary

The strongest fixed-split result is the compact kinematics-only LSTM ablation:

```text
A3_lstm = log_price + velocity + acceleration
```

This model outperforms the broader full-domain LSTM and the saved classical
baselines by MAE and RMSE on the main chronological test split.

The full-domain LSTM is still useful as the default sequence model and as the
walk-forward model, but the ablation suggests that adding many engineered
features can hurt generalization.

## Limitations

- The asset universe is limited to seven Indian equities.
- Results are empirical and should not be interpreted as trading profitability.
- Walk-forward performance is mixed; the LSTM often learns a positive ranking
  signal, but absolute volatility calibration is unstable in some years.
- Regime classification remains an auxiliary diagnostic, not a primary success
  claim.
- The current walk-forward table uses the full-domain LSTM; a future run should
  test the winning A3 kinematics-only feature set across walk-forward folds.
- No statistical confidence intervals or significance tests are included yet.

## Recommended Next Work

1. Add a dedicated `configs/lstm_kinematics.yaml`.
2. Run walk-forward validation for the A3 kinematics-only LSTM.
3. Add simple plots for the README/report:
   - ablation MAE/RMSE bar chart,
   - walk-forward yearly R2/Pearson chart,
   - baseline comparison table.
4. Expand the asset universe or replicate on another market.
5. Add bootstrap confidence intervals or paired fold-level comparisons.

## Publishable Claim

The most defensible claim is:

> A compact derivative-inspired representation of log price, return velocity,
> and return acceleration can outperform broader engineered feature sets for
> future realized-volatility forecasting on the main chronological test split,
> while walk-forward validation shows that financial forecasting remains
> temporally unstable and should be reported conservatively.

