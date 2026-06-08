# Financial Kinematics Project Instructions

## Project Title

Financial Kinematics: Derivative-Enriched Multi-Task TCN for Volatility and Risk-Regime Forecasting

## 1. Project Purpose

This project redesigns a basic stock-price prediction prototype into a meaningful M.Tech resume project.

The project must not predict absolute future stock prices as the main target. Instead, it should forecast financial risk signals using domain-informed temporal features.

The central idea is to treat a financial asset as a moving object:

- Log price is position.
- Log return is velocity.
- Change in log return is acceleration.
- Future realized volatility is market risk intensity.

The project acts as a predecessor to a physics-informed neural network thesis. It should show derivative-aware feature design, temporal deep learning, multi-task prediction, leakage-free validation, and ablation-based experimentation.

## 2. Core Research Question

Do derivative-enriched temporal features such as log-price, return velocity, and return acceleration improve future volatility and risk-regime forecasting compared with raw price-only or simple risk baselines?

Secondary questions:

- Does a TCN outperform LSTM and classical ML baselines for risk forecasting?
- Does multi-task learning improve risk forecasting compared with single-task learning?
- Do derivative features help more during high-volatility periods?
- Can the method generalize across multiple Indian equities?

## 3. Hard Rules

These rules are mandatory.

1. Do not use absolute future stock price prediction as the main objective.
2. Do not use random train-test split.
3. Do not fit scalers on the full dataset.
4. Do not use future data in input features.
5. Do not compute test labels or thresholds in a way that leaks future/test distribution information into training.
6. Always use chronological splits and walk-forward validation.
7. Always include a naive volatility baseline.
8. Do not claim trading profitability.
9. Report negative or mixed findings honestly.

## 4. Final Model Goal

Build a leakage-free multi-task temporal model with two main outputs:

1. Regression output:
   - Predict future 30-day realized volatility.

2. Classification output:
   - Predict future risk regime.
   - Start with binary classification: normal-risk vs high-risk.
   - Add three-class classification later only if the binary version is stable.

Optional third output:

3. Drawdown warning:
   - Predict whether the asset drops more than a selected threshold in the next prediction window.
   - This should be added only after the core two-output model works.

## 5. Recommended Dataset

Use daily Indian equity market data.

Suggested assets:

- RELIANCE.NS
- TCS.NS
- INFY.NS
- HDFCBANK.NS
- ICICIBANK.NS
- SBIN.NS
- LT.NS
- ^NSEI, if available

Suggested time range:

- Start: 2008-01-01
- End: latest available date, or fixed end date for reproducibility

Required raw columns:

- Date
- Open
- High
- Low
- Close
- Adjusted Close, if available
- Volume

Use adjusted close for return calculations when available.

## 6. Feature Engineering

All features must be computed so that a sample at time t uses only information available at or before time t.

If the model is predicting future risk from the end of day t, then features may use data up to t and targets must use t+1 onward.

### 6.1 Position

Use log price:

```text
x_t = log(P_t)
```

where `P_t` is adjusted close.

### 6.2 Velocity

Use log return:

```text
v_t = x_t - x_{t-1}
```

Equivalent:

```text
v_t = log(P_t / P_{t-1})
```

### 6.3 Acceleration

Use change in log return:

```text
a_t = v_t - v_{t-1}
```

### 6.4 Optional Jerk

Use only as an ablation feature:

```text
j_t = a_t - a_{t-1}
```

Jerk can be noisy, so it should not be in the default feature set.

### 6.5 Rolling Volatility

Past rolling volatility:

```text
past_vol_w_t = std(v_{t-w+1}, ..., v_t)
```

Recommended windows:

- 7 days
- 14 days
- 30 days

### 6.6 Momentum

Momentum:

```text
momentum_w_t = x_t - x_{t-w}
```

Recommended windows:

- 7 days
- 30 days
- 90 days

### 6.7 Drawdown

Rolling drawdown:

```text
drawdown_w_t = (P_t - rolling_max(P, w)_t) / rolling_max(P, w)_t
```

Recommended window:

- 90 days

### 6.8 True Range

True range can be used as a non-price risk feature:

```text
true_range_t = max(
    high_t - low_t,
    abs(high_t - close_{t-1}),
    abs(low_t - close_{t-1})
)
```

Use rolling averages of true range rather than raw price levels.

Recommended windows:

- 7 days
- 30 days

### 6.9 Volume Features

Use relative or normalized volume features:

```text
log_volume_t = log(volume_t + 1)
volume_change_t = log_volume_t - log_volume_{t-1}
volume_z_t = (log_volume_t - rolling_mean(log_volume, w)_t) / rolling_std(log_volume, w)_t
```

Recommended window:

- 30 days

## 7. Target Engineering

### 7.1 Future Realized Volatility

Main regression target:

```text
future_vol_30_t = std(v_{t+1}, v_{t+2}, ..., v_{t+30})
```

This uses future data only as a label. It must never be included as an input feature.

Why 30 days?

- Less noisy than 7-day volatility.
- Easier to interpret as a one-month risk forecast.
- More stable for resume-quality results.

Optional experiment:

- Compare 7-day, 14-day, and 30-day future volatility.

### 7.2 Risk-Regime Classification

Recommended first version:

```text
high_risk = 1 if future_vol_30_t >= train_75th_percentile
high_risk = 0 otherwise
```

Important:

- Compute the percentile threshold on the training split only.
- Apply the same threshold to validation and test.

Optional three-class version:

```text
low risk    = below train 33rd percentile
medium risk = between train 33rd and 66th percentiles
high risk   = above train 66th percentile
```

Start with binary high-risk classification. It is simpler, more stable, and easier to evaluate.

### 7.3 Optional Drawdown Warning

Future drawdown over the next 30 days:

```text
future_drawdown_30_t = min(P_{t+1}, ..., P_{t+30}) / P_t - 1
```

Binary label:

```text
drawdown_warning = 1 if future_drawdown_30_t <= -0.05 else 0
```

Use this only after the main project works.

## 8. Sequence Construction

Use sliding windows.

Input sequence:

```text
X_t = features from t-L+1 to t
```

Targets:

```text
y_vol_t = future_vol_30_t
y_regime_t = high_risk_t
```

Recommended sequence length:

```text
L = 60 trading days
```

This gives roughly three months of history.

Optional ablation:

- L = 30
- L = 60
- L = 90

## 9. Data Splitting

Use strict chronological split.

Recommended main split:

```text
Train:      2008-01-01 to 2018-12-31
Validation: 2019-01-01 to 2021-12-31
Test:       2022-01-01 to latest available date
```

Rules:

- Fit scalers only on train.
- Transform validation and test using train-fitted scalers.
- Do not shuffle time.
- Do not let sequence windows cross split boundaries unless explicitly designed and documented.

## 10. Walk-Forward Validation

Add walk-forward testing after the main split works.

Example:

```text
Fold 1: train 2008-2014, validate 2015, test 2016
Fold 2: train 2008-2015, validate 2016, test 2017
Fold 3: train 2008-2016, validate 2017, test 2018
...
```

Purpose:

- Simulate real deployment.
- Reduce dependence on one lucky split.
- Make the evaluation more credible.

## 11. Model Ladder

The project must use a model ladder. Do not jump directly to the proposed model.

### 11.1 Naive Baseline

Predict future volatility using current past volatility:

```text
pred_future_vol_30_t = past_vol_30_t
```

For classification:

```text
pred_high_risk_t = 1 if past_vol_30_t >= train_75th_percentile else 0
```

This baseline is mandatory.

### 11.2 Classical ML Baselines

Use latest-step tabular features or flattened windows.

Recommended models:

- Ridge regression for volatility
- Lasso regression for volatility
- Random Forest regressor/classifier
- Gradient Boosting or XGBoost if available
- Logistic Regression for high-risk classification

### 11.3 LSTM Baseline

Use a simple LSTM sequence model.

Purpose:

- Connects to the original prototype.
- Provides recurrent deep learning baseline.

### 11.4 Standard TCN Baseline

Use a TCN without attention and without multi-task learning.

Purpose:

- Tests whether TCN is better than LSTM for this task.

### 11.5 Proposed Model

Derivative-Enriched Multi-Task TCN.

Architecture:

```text
Input sequence
    -> causal TCN blocks
    -> temporal pooling
    -> volatility regression head
    -> risk-regime classification head
```

Optional enhanced model:

```text
Input sequence
    -> causal TCN blocks
    -> attention pooling
    -> volatility regression head
    -> risk-regime classification head
```

Attention should be optional, not required for the first working version.

## 12. TCN Architecture Instructions

Use PyTorch.

Recommended TCN details:

- 1D causal convolutions.
- Residual connections.
- Dropout.
- Weight normalization optional.
- Dilations: [1, 2, 4, 8].
- Kernel size: 3.
- Hidden channels: start with 32 or 64.

Input shape:

```text
batch_size, sequence_length, num_features
```

For Conv1D, transpose to:

```text
batch_size, num_features, sequence_length
```

## 13. Multi-Task Loss

Use combined loss:

```text
total_loss = alpha * volatility_loss + beta * regime_loss
```

Recommended:

```text
volatility_loss = HuberLoss or MSELoss
regime_loss = CrossEntropyLoss or BCEWithLogitsLoss
alpha = 1.0
beta = 1.0
```

If binary risk classification is used, prefer:

```text
BCEWithLogitsLoss
```

If classes are imbalanced, use:

- positive class weighting, or
- class weights, or
- threshold tuning on validation data.

## 14. Evaluation Metrics

### 14.1 Volatility Regression

Report:

- MAE
- RMSE
- R2
- Pearson correlation
- Spearman correlation, optional

Primary metric:

- MAE

### 14.2 Risk-Regime Classification

Report:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

Primary metric:

- F1-score for high-risk class

Reason:

- Accuracy can be misleading if high-risk periods are rare.

### 14.3 Drawdown Warning, Optional

Report:

- Recall
- Precision
- F1-score
- False positive rate

Primary metric:

- Recall

Reason:

- Missing a severe drawdown is worse than producing some false alarms.

## 15. Required Experiments

### 15.1 Main Model Comparison

Compare:

1. Naive historical volatility baseline
2. Ridge or Lasso
3. Random Forest or Gradient Boosting
4. LSTM
5. Standard TCN
6. Derivative-Enriched Multi-Task TCN
7. Derivative-Enriched Multi-Task TCN with attention, optional

### 15.2 Ablation Study

Run:

| ID | Setup | Purpose |
|---|---|---|
| A1 | Log price only + TCN | Weak raw-position baseline |
| A2 | Log price + velocity + TCN | Test return feature |
| A3 | Log price + velocity + acceleration + TCN | Test financial kinematics |
| A4 | Full domain features + TCN | Test engineered features |
| A5 | Full domain features + multi-task TCN | Test multi-task learning |
| A6 | Full domain features + multi-task TCN + attention | Test attention |

Do not force the expected result. Report what the results actually show.

### 15.3 Single-Task vs Multi-Task

Compare:

- Train only on future volatility.
- Train only on high-risk regime.
- Train jointly on both.

Purpose:

- Test whether multi-task learning helps.

### 15.4 Regime-Specific Evaluation

Evaluate performance separately for:

- Low-volatility periods
- High-volatility periods
- COVID crash period
- Recent test period

### 15.5 Cross-Asset Generalization, Optional

Example:

```text
Train: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS
Test:  ICICIBANK.NS or SBIN.NS
```

Question:

Does the model learn general risk dynamics or only asset-specific patterns?

## 16. Required Visualizations

Generate:

1. Price and return overview.
2. Log price, velocity, and acceleration plot.
3. Past volatility vs future volatility.
4. Predicted vs actual future volatility.
5. High-risk classification confusion matrix.
6. High-risk timeline with predicted warnings.
7. Model comparison table.
8. Ablation bar chart.
9. Walk-forward fold performance.
10. Optional attention heatmap.

## 17. Recommended Repository Structure

```text
financial-kinematics/
  README.md
  environment.yml
  configs/
    default.yaml
    lstm.yaml
    tcn.yaml
    multitask_tcn.yaml
  data/
    raw/
    processed/
  notebooks/
    01_data_exploration.ipynb
    02_feature_validation.ipynb
    03_results_analysis.ipynb
  src/
    data/
      download.py
      preprocess.py
      features.py
      datasets.py
    models/
      baselines.py
      lstm.py
      tcn.py
      multitask_tcn.py
      attention.py
    training/
      train.py
      losses.py
      evaluate.py
    experiments/
      run_baselines.py
      run_ablation.py
      walk_forward.py
    utils/
      config.py
      seed.py
      plotting.py
  results/
    tables/
    figures/
    checkpoints/
  reports/
    project_report.md
```

## 18. Implementation Phases

### Phase 1: Minimal Proof of Concept

Use:

- One asset: RELIANCE.NS
- One regression target: future 30-day volatility
- One naive baseline: past 30-day volatility
- One model: TCN
- One ablation: log price only vs log price + velocity + acceleration

Success criteria:

- Features and labels align correctly.
- No leakage.
- Model trains and evaluates on chronological split.

### Phase 2: Multi-Asset Dataset

Add:

- 5 to 10 assets.
- Asset identifier, optional.
- Consistent feature pipeline across assets.

Success criteria:

- Processed dataset is reproducible.
- Per-asset and pooled results can be computed.

### Phase 3: Baseline Ladder

Add:

- Naive baseline.
- Classical ML baselines.
- LSTM.
- Standard TCN.

Success criteria:

- All baselines evaluated on same splits.
- Results saved to tables.

### Phase 4: Multi-Task Model

Add:

- Future volatility regression head.
- High-risk classification head.
- Combined loss.
- Class imbalance handling.

Success criteria:

- Both heads train properly.
- Metrics are reported for both tasks.

### Phase 5: Ablation and Walk-Forward Validation

Add:

- A1 to A6 ablations.
- Walk-forward folds.
- Regime-specific evaluation.

Success criteria:

- Clear evidence for which features/components matter.
- No cherry-picked results.

### Phase 6: Packaging

Add:

- README.
- Project report.
- Result plots.
- Resume bullets.

Success criteria:

- A reviewer can understand the project quickly.
- A technical interviewer can inspect the methodology and trust it.

## 19. Expected Findings

Possible positive finding:

```text
Derivative-enriched temporal features improved future volatility MAE and high-risk F1-score over raw-price TCN and classical baselines.
```

Possible mixed finding:

```text
Acceleration features improved high-risk classification but did not significantly improve volatility regression.
```

Possible negative finding:

```text
Historical volatility remained a strong baseline, showing that leakage-free financial risk forecasting is difficult and that model complexity must be justified.
```

All three are acceptable if the experiments are clean.

## 20. Thesis Connection

This project is not the thesis. It is the foundation.

Connection:

| Financial Kinematics | Physics-Informed NN Thesis |
|---|---|
| Log price as position | Joint angle or biomechanical position |
| Return as velocity | Physical velocity |
| Return change as acceleration | Physical acceleration |
| Volatility as risk intensity | Physical load, residual, or dynamic response |
| TCN temporal modeling | TCN or Transformer movement modeling |
| Domain-informed features | Physics-informed structure |
| Multi-task outputs | Prediction plus constraint/residual outputs |
| Leakage-free temporal split | Subject/movement split discipline |

Interview narrative:

```text
I first built a basic stock forecasting prototype, but direct price prediction was noisy and not very meaningful. I redesigned the project around financial risk forecasting by treating log price, returns, and return acceleration as derivative-like motion features. This helped me build a foundation in temporal modeling, derivative-aware feature design, and leakage-free evaluation, which later informed my physics-informed neural network thesis on biomechanical time series.
```

## 21. Final Project Claim

Use a careful claim:

```text
This project reframes stock-market prediction as risk forecasting and shows how derivative-enriched temporal representations can be evaluated under leakage-free conditions for future volatility and high-risk regime prediction.
```

Avoid claims like:

```text
The model predicts the stock market.
The model guarantees profitable trading.
The model discovers true financial physics.
```

## 22. Resume Bullets

Short version:

```text
Built a derivative-enriched multi-task TCN for Indian equity risk forecasting, predicting 30-day realized volatility and high-risk regimes using leakage-free walk-forward validation.
```

Technical version:

```text
Designed a financial kinematics pipeline treating log price, returns, and return acceleration as temporal motion features; trained LSTM, TCN, and multi-task TCN models to forecast realized volatility and risk regimes across Indian equities.
```

Experiment version:

```text
Validated volatility and risk-regime forecasts using chronological splits, walk-forward testing, naive/classical/deep baselines, and ablation studies measuring the impact of derivative-enriched features.
```

Thesis-bridge version:

```text
Developed a derivative-aware temporal modeling project as a precursor to physics-informed neural network research, connecting financial motion features with TCN-based sequence modeling and multi-output prediction.
```

## 23. Suggested README Abstract

```text
This project reframes stock-market prediction as a risk forecasting problem. Instead of predicting exact future prices, it models financial motion through log price, return, and return acceleration, then uses a multi-task Temporal Convolutional Network to forecast future realized volatility and high-risk market regimes. The system is evaluated using leakage-free chronological splits, walk-forward validation, classical and deep learning baselines, and ablation studies that measure the contribution of derivative-enriched features.
```

