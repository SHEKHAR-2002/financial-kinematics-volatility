# Financial Kinematics

Derivative-enriched sequence modeling for Indian equity volatility forecasting.

This project reframes stock-market prediction as risk forecasting. Instead of predicting absolute future prices, it treats log price as position, log return as velocity, and return change as acceleration, then forecasts future realized volatility under leakage-free chronological validation. High-risk regime labels are retained as an auxiliary diagnostic.

## What This Implements

- Daily OHLCV ingestion for Indian equities.
- Leakage-safe feature engineering:
  - log price
  - return velocity
  - return acceleration
  - rolling historical volatility
  - momentum
  - drawdown
  - true range averages
  - normalized volume features
- Future 30-day realized volatility target.
- Binary high-risk regime labels using the train-split 75th percentile only.
- Sliding-window sequence construction with no cross-split windows.
- Mandatory naive historical-volatility baseline.
- Classical ML baselines: Ridge, Lasso, Random Forest, Gradient Boosting, Logistic Regression.
- Main LSTM sequence model for final feature ablation and walk-forward validation.
- TCN model for architecture comparison against LSTM.
- Experimental multi-task LSTM/TCN and attention modules kept outside the main workflow.
- Main training, baseline, ablation, and walk-forward experiment entrypoints.

## Repository Structure

```text
configs/              Experiment configs
data/raw/             Raw OHLCV CSVs
data/processed/       Feature-engineered CSVs
src/data/             Downloading, preprocessing, features, datasets
src/models/           Baselines, LSTM, TCN, experimental multi-task models
src/training/         Losses, training loop, evaluation metrics
src/experiments/      Baseline, ablation, and walk-forward scripts
src/utils/            Config, seed, plotting helpers
results/              Tables, figures, checkpoints
reports/              Project report skeleton
tests/                Leakage and alignment tests
```

## Setup

```bash
conda env create -f environment.yml
conda activate financial-kinematics
```

Or with pip:

```bash
pip install -e ".[deep,download,plots,test]"
```

## Data

Download Yahoo Finance data for the configured Indian equities:

```bash
python -m src.data.download --config configs/default.yaml
```

If you already have CSV files, place them in `data/raw/`. Each CSV should contain:

```text
Date, Open, High, Low, Close, Adj Close, Volume
```

Then create processed feature files:

```bash
python -m src.data.preprocess --config configs/default.yaml
```

This writes one feature file per asset plus `data/processed/features_all.csv`.

## Run Experiments

Run the mandatory naive and classical baselines:

```bash
python -m src.experiments.run_baselines --config configs/default.yaml --processed data/processed/features_all.csv
```

Train the main LSTM model:

```bash
python -m src.training.train --config configs/lstm.yaml --processed data/processed/features_all.csv
```

Train the vanilla TCN comparison model:

```bash
python -m src.training.train --config configs/tcn.yaml --processed data/processed/features_all.csv
```

Run LSTM feature ablations A1-A4:

```bash
python -m src.experiments.run_ablation --config configs/default.yaml --processed data/processed/features_all.csv
```

Run expanding walk-forward validation for LSTM:

```bash
python -m src.experiments.walk_forward --config configs/lstm.yaml --processed data/processed/features_all.csv
```

Experimental configs such as `configs/multitask_lstm.yaml`, `configs/multitask_tcn.yaml`,
and `configs/tcn_mlp.yaml` are retained for exploration, but they are not part of
the main publishable workflow.

## Methodological Guardrails

- No absolute future price prediction objective.
- No random train-test split.
- No full-dataset scaler fitting.
- No future data in input features.
- Risk thresholds are fit on the training split only.
- Validation and test reuse train-fitted scalers and thresholds.
- Sequence windows are built inside each split and asset group.
- Results should be reported honestly, including negative or mixed findings.

## Resume Claim

Built a leakage-aware financial kinematics pipeline for Indian equity volatility
forecasting, evaluating derivative-enriched temporal features with classical
baselines, LSTM sequence modeling, TCN comparison, ablation, and walk-forward
validation.
