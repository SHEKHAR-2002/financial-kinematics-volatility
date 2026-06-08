# Financial Kinematics Project Report

## Abstract

This project reframes stock-market prediction as financial risk forecasting. Rather than predicting absolute future prices, it models financial motion through log price, log return, and return acceleration, then evaluates whether derivative-enriched temporal features improve future realized-volatility and high-risk-regime forecasting.

## Research Question

Do derivative-enriched temporal features such as log price, return velocity, and return acceleration improve future volatility and risk-regime forecasting compared with raw price-only or simple risk baselines?

## Methodology

- Assets: Indian equities such as RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, ICICIBANK.NS, SBIN.NS, and LT.NS.
- Target: future 30-day realized volatility.
- Regime label: high risk when future volatility exceeds the train-split 75th percentile.
- Sequence length: 60 trading days.
- Splits: chronological train, validation, and test periods.
- Validation: main chronological split plus expanding walk-forward folds.

## Model Ladder

1. Naive historical-volatility baseline.
2. Ridge and Lasso regression.
3. Random Forest and Gradient Boosting baselines.
4. LSTM baseline.
5. Standard TCN baseline.
6. Derivative-enriched multi-task TCN.
7. Optional attention-pooled multi-task TCN.

## Evaluation

Regression metrics:

- MAE
- RMSE
- R2
- Pearson correlation

Classification metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Primary metrics:

- Volatility: MAE
- High-risk regime: F1-score

## Ablations

| ID | Setup | Purpose |
|---|---|---|
| A1 | Log price only + TCN | Weak raw-position baseline |
| A2 | Log price + velocity + TCN | Test return feature |
| A3 | Log price + velocity + acceleration + TCN | Test financial kinematics |
| A4 | Full domain features + TCN | Test engineered features |
| A5 | Full domain features + multi-task TCN | Test multi-task learning |
| A6 | Full domain features + multi-task TCN + attention | Test attention |

## Reporting Standard

The final report should not claim trading profitability or guaranteed market prediction. The strongest careful claim is:

> This project reframes stock-market prediction as risk forecasting and shows how derivative-enriched temporal representations can be evaluated under leakage-free conditions for future volatility and high-risk regime prediction.
