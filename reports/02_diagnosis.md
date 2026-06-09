# Diagnosis: Major Issues and Fix Plan

## Current Status Note

This diagnosis captures the broader pre-cleanup audit, including TCN-MLP,
multi-task, and attention variants. The current publishable workflow is narrower:
single-task LSTM is the main deep model for feature ablation and walk-forward
validation, vanilla TCN is retained only as an architecture comparison, and
multi-task/attention variants are treated as experimental code outside the main
report path.

## Purpose

This document consolidates the current code-audit findings into an implementation
plan. It focuses on issues that can materially affect the validity of the
results, the fairness of model comparisons, or the clarity of the final report.

The current saved results should be treated as provisional. They are useful for
debugging, but the experiments should be rerun after the fixes below.

## Current Empirical State

The present results show that simple baselines are highly competitive:

- Lasso is currently the best volatility model by MAE, RMSE, R2, and Pearson
  among the saved runs.
- Single-task LSTM is the best saved deep model by MAE and Pearson.
- Single-task TCN underperforms LSTM and simple baselines.
- Multi-task TCN improves over single-task TCN, but it does not clearly beat
  LSTM or Lasso.
- Multi-task LSTM currently performs poorly on volatility.
- Regime classification is weak across nearly all models. Accuracy is high
  mostly because high-risk positives are rare.

These results should not be used as final claims until the methodology and
comparison issues below are fixed.

## Priority 1: Methodology-Critical Issues

### 1. Target-Window Leakage Across Split Boundaries

Files:

- `src/data/features.py`
- `src/data/datasets.py`
- `src/data/preprocess.py`

Problem:

`future_vol_30` is computed on the full asset time series before chronological
splitting. The split then uses only the row date. Therefore, late training rows
can have labels computed using validation-period returns, and late validation
rows can have labels computed using test-period returns.

Example:

For a row dated `2018-12-31`, the target `future_vol_30` uses returns from
`2019-01-01` onward. That row is assigned to training, but its label is computed
from validation-period data.

Impact:

- This breaks the project's "leakage-free chronological split" claim.
- It affects all models similarly, so it does not directly explain TCN vs LSTM
  differences.
- It does make the reported validation and test protocol methodologically
  weaker than intended.

Fix:

- Track the final date used by each future label, for example
  `future_vol_30_end_date`.
- During splitting, keep a row in a split only if its label end date stays inside
  that split's boundary.
- Alternatively, drop the final `horizon` trading rows per asset from train and
  validation before sequence creation.

Preferred implementation:

```text
train row allowed if row_date <= train_end and label_end_date <= train_end
val row allowed if row_date <= val_end and label_end_date <= val_end
test row allowed if row_date <= test_end and label_end_date <= test_end
```

After fixing:

- Rerun preprocessing if label-end dates are added to processed data.
- Rerun baselines and all neural models.
- Update any report language claiming leakage-free validation.

### 2. Multi-Task Early Stopping Optimizes the Wrong Signal

Files:

- `src/training/train.py`
- `src/training/losses.py`
- `configs/default.yaml`

Problem:

Multi-task models select checkpoints using the combined validation loss:

```text
alpha * volatility_loss + beta * regime_loss
```

With the current scale, regime loss dominates the combined loss even after
`alpha=1000.0`. In the saved histories, both multi-task TCN and multi-task LSTM
select epoch 1 as the best checkpoint because regime loss worsens, even though
volatility loss remains the primary regression metric.

Impact:

- Multi-task volatility results are not directly comparable to single-task
  volatility results.
- Multi-task models may be undertrained for the volatility objective.
- The current multi-task TCN improvement over single-task TCN may be unstable or
  not representative.

Fix:

- Add a config option for early-stopping target:

```yaml
training:
  early_stopping_metric: val_volatility_loss
```

- For multi-task models, support at least:
  - `val_loss`: current combined objective
  - `val_volatility_loss`: primary volatility objective
  - optionally `val_regime_loss`
- Save the selected early-stopping metric in the history or metrics output.

Recommended default:

- Use `val_volatility_loss` when ranking models by volatility MAE/RMSE/R2.
- Use combined loss only when explicitly evaluating the joint objective.

After fixing:

- Rerun multi-task TCN and multi-task LSTM.
- Compare both "best combined" and "best volatility" checkpoints if useful.

### 3. TCN Multi-Task Ablation Is Confounded by Head Architecture

Files:

- `src/models/tcn.py`
- `src/models/multitask_tcn.py`
- `src/experiments/run_ablation.py`

Problem:

Single-task TCN uses a simple linear regression head:

```text
Linear(hidden_channels, 1)
```

Multi-task TCN uses deeper MLP heads:

```text
Linear(hidden_channels, hidden_channels)
ReLU
Dropout
Linear(hidden_channels, 1)
```

Therefore, the single-task TCN to multi-task TCN comparison changes both the
training objective and the regression-head capacity.

Impact:

- The A4 to A5 ablation does not isolate multi-task learning.
- Any multi-task TCN improvement may come from the deeper head, the classifier
  head, the loss, or their interaction.

Fix options:

- Add `TCNVolatilityRegressorMLP` with the same regression head as multi-task
  TCN but no classification head.
- Or simplify `MultiTaskTCN` to use linear heads.
- Keep both variants if the report wants to separate "head capacity" from
  "multi-task learning."

Recommended ablation:

```text
TCN linear head
TCN MLP head
Multi-task TCN with matched MLP regression head
Multi-task TCN with attention, optional
```

After fixing:

- Rerun the TCN ablation.
- Avoid claiming that multi-task learning helped unless the matched-head
  comparison supports it.

## Priority 2: Fairness and Evaluation Issues

### 4. LSTM and TCN Training Configs Are Not Matched

Files:

- `configs/lstm.yaml`
- `configs/multitask_lstm.yaml`
- `configs/tcn.yaml`
- `configs/multitask_tcn.yaml`
- `configs/default.yaml`

Problem:

The LSTM family currently uses different training settings from the TCN family:

```text
LSTM family: patience 5, weight_decay 0.001, scheduler none
TCN family:  patience 15, weight_decay 0.0001, ReduceLROnPlateau
```

Impact:

- LSTM vs TCN comparisons are not clean architecture comparisons.
- Multi-task LSTM may be especially disadvantaged because it early-stops quickly.

Fix:

- Create matched experiment configs for architecture comparison.
- Use the same training budget, scheduler, learning rate, patience, batch size,
  gradient clipping, and weight decay unless there is a documented tuning reason.

Recommended config policy:

```text
configs/lstm.yaml
configs/tcn.yaml
configs/multitask_lstm.yaml
configs/multitask_tcn.yaml
```

should differ mainly in model architecture, not optimizer policy.

After fixing:

- Rerun single-task LSTM, single-task TCN, multi-task LSTM, and multi-task TCN.
- Keep old runs clearly labeled as pre-fix.

### 5. Classification Thresholds Are Inconsistent

Files:

- `src/models/baselines.py`
- `src/training/train.py`
- `src/training/evaluate.py`

Problem:

Different model families use different thresholding strategies:

- Naive baseline thresholds raw `past_vol_30` by a train quantile.
- Classical classifiers use sklearn's default `predict()` threshold.
- Neural multi-task models tune a threshold on validation F1.

Impact:

- Precision, recall, and F1 are not directly comparable.
- ROC-AUC remains more comparable because it is ranking-based.

Fix:

- Compute validation scores for all classifiers.
- Select thresholds on validation data using the same rule for all models.
- Apply the selected threshold once to the test set.
- Save the selected threshold in the metrics table.

Recommended reporting:

- Primary threshold-free metric: ROC-AUC, and preferably PR-AUC.
- Primary thresholded metric: F1 using validation-selected threshold.
- Do not use accuracy as the main classification metric.

### 6. Classical Models and Deep Models Use Different Temporal Inputs

Files:

- `src/experiments/run_baselines.py`
- `src/data/datasets.py`
- `configs/default.yaml`

Problem:

Classical baselines default to:

```text
latest_step_tabular: X[:, -1, :]
```

Deep models receive:

```text
full sequence: X[:, 60, features]
```

Impact:

- "Lasso beats LSTM" currently means a latest-day tabular model beats a sequence
  model, not that both models received identical information.
- This is a meaningful practical comparison, but it is not a pure architecture
  comparison.

Fix:

- Run two baseline modes:
  - `latest`: practical snapshot baseline.
  - `flattened`: full 60-step window flattened for classical models.

Recommended reporting:

```text
Table 1: Latest-step classical baselines vs sequence models
Table 2: Flattened-window classical baselines vs sequence models
```

### 7. Naive Baseline Should Use the Same Prepared Test Arrays

Files:

- `src/experiments/run_baselines.py`
- `src/models/baselines.py`
- `src/data/datasets.py`

Problem:

The naive baseline currently builds target rows from raw labeled test frames,
while other baselines and neural models use `prepare_splits()`. The present
sample counts match, so this is not currently producing different evaluation
sample sizes. However, the code does not enforce that guarantee.

Impact:

- Fragile comparison if feature sets or preprocessing behavior change.
- Makes it harder to prove all models were evaluated on identical rows.

Fix:

- Make the naive baseline operate on `SequenceArrays`.
- Use the latest timestep's `past_vol_30` feature for prediction.
- Or carry row identifiers through `SequenceArrays` and join to the exact same
  target rows.

After fixing:

- Assert that all model bundles have identical test sample count, dates, and
  assets.

## Priority 3: Model and Config Robustness

### 8. `build_model()` Uses Inconsistent Multi-Task Semantics

File:

- `src/training/train.py`

Problem:

For LSTM:

```text
type: lstm
multitask: true
```

builds a multi-task LSTM. For TCN:

```text
type: tcn
multitask: true
```

is ignored, because multi-task TCN is selected by `type: multitask_tcn`.

Impact:

- Config semantics are inconsistent.
- Ablations are easier to misread or misconfigure.

Fix:

- Choose one convention and use it consistently.

Recommended convention:

```yaml
model:
  type: tcn
  multitask: false
```

and

```yaml
model:
  type: tcn
  multitask: true
```

or keep explicit model types but remove redundant/confusing flags:

```yaml
type: tcn
type: multitask_tcn
type: lstm
type: multitask_lstm
```

The second convention is cleaner for artifact naming and reports.

### 9. Default Model Type Can Hide Config Mistakes

File:

- `src/training/train.py`

Problem:

`build_model()` defaults to `multitask_tcn` if `model.type` is missing.

Impact:

- A broken or incomplete config can silently train the wrong model.

Fix:

- Require `model.type`.
- Raise a clear error if it is missing.

Suggested behavior:

```text
ValueError: config.model.type is required
```

### 10. Artifact Naming Should Remain Explicit

Files:

- `src/training/train.py`
- `src/experiments/walk_forward.py`
- `configs/multitask_lstm.yaml`

Current state:

Artifact naming has been improved with `model.name`, and
`configs/multitask_lstm.yaml` now saves as `multitask_lstm_*`.

Remaining fix:

- Add explicit `model.name` to every config used in final experiments.
- Ensure walk-forward, ablation, checkpoints, histories, and metrics all include
  enough identity to avoid overwriting results.

Recommended names:

```text
lstm
multitask_lstm
tcn
tcn_mlp_head
multitask_tcn
multitask_tcn_attention
```

## Priority 4: Regime Task Quality

### 11. High-Risk Classification Is Currently Weak

Files:

- `src/training/losses.py`
- `src/training/train.py`
- `src/models/baselines.py`
- `configs/default.yaml`

Problem:

Saved regime results have very low F1, and several ROC-AUC values are near or
below random. The high accuracy values are mostly due to class imbalance.

Impact:

- The regime head should not be used as a central success claim yet.
- Multi-task learning may be hurting or destabilizing shared representations.

Fixes to try after methodology fixes:

- Use validation-selected thresholds for every classifier.
- Report PR-AUC in addition to ROC-AUC.
- Add quantile sensitivity, for example 0.70, 0.75, and 0.80.
- Consider focal loss or calibrated class weights only after thresholding and
  leakage fixes are complete.
- Consider making regime classification an auxiliary diagnostic instead of a
  primary claim if it remains weak.

## Priority 5: Missing Final Experiments

### 12. Ablation and Walk-Forward Results Need to Be Rerun After Fixes

Files:

- `src/experiments/run_ablation.py`
- `src/experiments/walk_forward.py`
- `results/tables/`

Problem:

The final experiment set is incomplete or stale relative to the current code and
the audit findings.

Fix:

- Rerun baselines after leakage-safe splitting.
- Rerun single-task and multi-task deep models with matched configs.
- Rerun ablations with matched TCN heads.
- Rerun walk-forward validation and save model-specific outputs.

Expected result files:

```text
results/tables/baseline_metrics.csv
results/tables/lstm_test_metrics.csv
results/tables/tcn_test_metrics.csv
results/tables/tcn_mlp_test_metrics.csv
results/tables/multitask_lstm_test_metrics.csv
results/tables/multitask_tcn_test_metrics.csv
results/tables/ablation_metrics.csv
results/tables/*_walk_forward_metrics.csv
```

## Priority 6: Small Gaps Worth Adding

### 13. Gradient Boosting Classification Is Unweighted

Files:

- `src/models/baselines.py`

Problem:

`LogisticRegression` and `RandomForestClassifier` use `class_weight="balanced"`,
but `GradientBoostingClassifier` does not use class weighting or sample weights.

Impact:

- This is a likely direct explanation for gradient boosting predicting zero
  positives in the saved results.
- It mainly affects GBM's regime precision, recall, and F1. It does not affect
  GBM's volatility regression result.

Fix:

- Pass `sample_weight` when fitting `GradientBoostingClassifier`.
- Compute class-balanced sample weights from `y_regime_train`.
- Keep this separate from threshold tuning, because both can affect F1.

### 14. `past_vol_30` Is a Built-In Diagnostic Baseline Signal

Files:

- `src/data/features.py`
- `src/models/baselines.py`

Problem:

`past_vol_30` is included in `FULL_DOMAIN_FEATURES`, and the naive baseline uses
`past_vol_30` directly as its volatility prediction.

Impact:

- Neural models using the full feature set already receive the naive baseline's
  key signal in their input window.
- If a neural model performs worse than the naive baseline, that is useful
  diagnostic evidence that the model is not exploiting a simple available
  signal.
- This does not prove that the regime head is damaging the representation. The
  cause could also be optimization, scaling, head architecture, checkpoint
  selection, regularization, or insufficient tuning.

Fix:

- Add a diagnostic metric comparing each neural model against the naive baseline.
- Add an ablation that includes only `past_vol_30` or only volatility-history
  features.
- For full-domain neural runs, optionally log the index of `past_vol_30` and
  compare predictions against the latest input value.

### 15. Flattened-Window Baseline Command Is Missing

Files:

- `configs/default.yaml`
- `src/experiments/run_baselines.py`

Problem:

The diagnosis recommends running both latest-step and flattened-window classical
baselines, but the rerun command list only runs the default latest-step mode.

Impact:

- The rerun checklist does not fully execute the comparison recommended in
  Issue 6.

Fix:

- Add a dedicated config such as `configs/baselines_flattened.yaml`:

```yaml
base: default.yaml
experiments:
  tabular_window: flattened
```

- Then run:

```powershell
python -m src.experiments.run_baselines --config configs/baselines_flattened.yaml --processed data/processed/features_all.csv
```

### 16. Add Tests for Label-Window Boundary Enforcement

Files:

- `tests/`
- `src/data/features.py`
- `src/data/datasets.py`

Problem:

Stage A says to add assertions for leakage-safe splitting, but the current test
suite does not include a concrete test for label-window boundary enforcement.

Impact:

- The leakage fix could regress silently.
- The project's most important validation guarantee would remain untested.

Fix:

- Add a unit test with a small synthetic asset timeline and a short horizon.
- Verify that rows whose future-label window crosses `train_end` or `val_end`
  are excluded from that split.
- Verify that generated sequences do not contain input rows outside the split.
- Verify that the retained label end dates are within the assigned split.

### 17. Walk-Forward Output Naming Is Already Fixed, But Should Stay Verified

Files:

- `src/experiments/walk_forward.py`
- `src/training/train.py`

Current state:

`walk_forward.py` now uses `model_artifact_name()` and saves model-specific
walk-forward files:

```text
results/tables/{model_name}_walk_forward_metrics.csv
```

Impact:

- Running walk-forward for LSTM, TCN, multi-task LSTM, and multi-task TCN should
  no longer overwrite the same `walk_forward_metrics.csv` file.

Remaining fix:

- Add `model.name` to every final config.
- Keep a smoke test or manual check that each walk-forward command writes a
  distinct output file.

## Recommended Fix Order

### Stage A: Validity Fixes

1. Add label end dates or split-boundary target filtering.
2. Require labels to stay inside each split.
3. Add assertions that train, validation, and test sequences do not cross split
   boundaries through either inputs or labels.
4. Add a unit test for label-window boundary enforcement.
5. Rerun preprocessing and tests.

### Stage B: Fair Model Comparisons

1. Equalize LSTM and TCN training configs.
2. Add matched TCN head variants.
3. Make early stopping configurable and use volatility loss for volatility
   comparisons.
4. Require explicit model type and model name in final configs.

### Stage C: Classification Evaluation

1. Add validation threshold tuning for classical classifiers.
2. Save thresholds for every classifier.
3. Add PR-AUC.
4. Add class-balanced sample weights for GBM classification.
5. Report accuracy only as a secondary metric.

### Stage D: Rerun Experiments

1. Rerun latest-step baselines.
2. Rerun flattened-window baselines.
3. Rerun LSTM, TCN, TCN MLP, multi-task LSTM, and multi-task TCN.
4. Rerun ablation.
5. Rerun walk-forward.

### Stage E: Report Update

1. Update the project report with corrected tables.
2. State clearly whether simple baselines beat deep models.
3. Separate volatility conclusions from regime classification conclusions.
4. Avoid claiming TCN superiority unless corrected results support it.
5. Frame negative results honestly.

## Suggested Rerun Commands

After code fixes and preprocessing:

```powershell
python -m src.experiments.run_baselines --config configs/default.yaml --processed data/processed/features_all.csv
python -m src.experiments.run_baselines --config configs/baselines_flattened.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/lstm.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/tcn.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/tcn_mlp.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/multitask_lstm.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/multitask_tcn.yaml --processed data/processed/features_all.csv
python -m src.experiments.run_ablation --config configs/default.yaml --processed data/processed/features_all.csv
python -m src.experiments.walk_forward --config configs/lstm.yaml --processed data/processed/features_all.csv
python -m src.experiments.walk_forward --config configs/tcn.yaml --processed data/processed/features_all.csv
python -m src.experiments.walk_forward --config configs/tcn_mlp.yaml --processed data/processed/features_all.csv
python -m src.experiments.walk_forward --config configs/multitask_lstm.yaml --processed data/processed/features_all.csv
python -m src.experiments.walk_forward --config configs/multitask_tcn.yaml --processed data/processed/features_all.csv
```

## Definition of Done

The next stage is complete when:

- Split-boundary label leakage is fixed or explicitly documented.
- All model configs are matched or intentionally tuned with documentation.
- Multi-task checkpoint selection aligns with the metric being reported.
- Classification thresholds are selected consistently.
- Baseline and deep models are evaluated on auditable, aligned samples.
- Ablation and walk-forward CSVs exist.
- The report no longer claims leakage-free TCN superiority unless corrected
  results support it.
