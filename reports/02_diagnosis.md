# Diagnosis for Next-Stage Bug Fixes and Improvements

## Purpose

This document consolidates the useful findings from `academic_review.md` and
`bugfixes_and_improvements.md` into a cleaner implementation guide.

It separates verified issues from hypotheses. The current results are poor, but
not every proposed explanation is proven yet. The next stage should fix clear
bugs, improve training observability, rerun experiments, and then decide whether
larger model changes are justified.

## Current State

The saved test results show that the project infrastructure is stronger than the
current empirical performance.

Regression:

- All saved models have negative test R2.
- Lasso is currently the best volatility model by MAE and R2.
- The proposed multi-task TCN underperforms simple baselines.
- The single-task TCN result is especially poor and should be treated as a
  training or configuration failure until rerun.

Classification:

- High-risk regime classification is effectively non-functional in the saved
  results.
- Accuracy is high mainly because positives are rare.
- F1 scores are near zero, and the multi-task TCN ROC-AUC is below random.

Artifacts:

- `results/tables/ablation_metrics.csv` is missing.
- `results/tables/walk_forward_metrics.csv` is missing.
- `results/figures/` contains no generated figures.
- `reports/project_report.md` is still a skeleton.
- `notebooks/` contains no exploratory or results-analysis notebooks.

## Verified Issues to Fix First

### 1. Training DataLoader Shuffling

File: `src/training/train.py`

The training loaders use `shuffle=False` in both regression and multi-task
training. Sequence order has already been encoded inside each sample, so training
batches can be shuffled.

Action:

- Set `shuffle=True` for training loaders only.
- Keep validation, test, and prediction loaders unshuffled.

Expected benefit:

- More stable SGD updates and less dependence on temporally clustered batches.

### 2. Multi-Task Loss Scale Mismatch

Files: `src/training/losses.py`, `src/training/train.py`, `configs/default.yaml`

The volatility Huber loss is much smaller than the BCE regime loss. With
`alpha=1.0` and `beta=1.0`, the regime objective can dominate shared
representation learning.

Action:

- Log separate train and validation loss components for volatility and regime.
- Start with a conservative manual loss reweighting option in config.
- Consider uncertainty weighting only after the simpler path is measurable.

Expected benefit:

- The volatility head should no longer be starved by the regime objective.

### 3. No Gradient Clipping

File: `src/training/train.py`

Neither training loop clips gradients. This is especially risky for LSTM training
and for multi-task objectives.

Action:

- Add configurable gradient clipping, defaulting to `1.0`.
- Record the configured value in training output or history metadata.

Expected benefit:

- Better training stability and reduced chance of extreme failed runs.

### 4. No Learning Rate Scheduler

File: `src/training/train.py`

The training loops use a fixed learning rate. The saved histories suggest some
models either stop early or continue improving at the epoch limit.

Action:

- Add `ReduceLROnPlateau` on validation loss.
- Save the current learning rate in each history row.

Expected benefit:

- Better convergence without manually guessing a single fixed learning rate.

### 5. Training Budget Is Too Small

File: `configs/default.yaml`

The current default of `epochs=30` and `patience=5` is thin for neural sequence
models, especially once LR scheduling is added.

Action:

- Increase defaults for deep models, for example `epochs=100` and `patience=15`.
- Keep quick smoke-test configs separate from serious experiment configs.

Expected benefit:

- Fewer undertrained model comparisons.

### 6. Hardcoded Regime Threshold at Prediction Time

File: `src/training/train.py`

`predict_bundle()` hardcodes a sigmoid threshold of `0.5`. With imbalanced
classification, this is rarely the best F1 threshold.

Action:

- Add validation-set threshold tuning for F1.
- Save the selected threshold with metrics or history.
- Use the tuned threshold for test predictions.

Expected benefit:

- More meaningful precision, recall, and F1 evaluation.

### 7. Lasso Baseline Missing Classification Metrics

File: `src/models/baselines.py`

Lasso has blank classification fields because no classifier is paired with it.
Ridge is manually mapped to logistic regression, but Lasso is not.

Action:

- Map Lasso to the same logistic-regression classifier used by Ridge.

Expected benefit:

- Complete baseline comparison table.

### 8. Config Inheritance Path Is Fragile

File: `src/utils/config.py`

`base:` config paths are resolved relative to the current working directory, not
relative to the child config file. This can break runs launched outside the repo
root.

Action:

- Resolve `base:` relative to `config_path.parent`.

Expected benefit:

- More robust CLI behavior.

### 9. SequenceDataset Should Inherit Torch Dataset

File: `src/data/datasets.py`

`SequenceDataset` works through duck typing, but it should inherit
`torch.utils.data.Dataset` when torch is available.

Action:

- Import torch lazily and subclass the torch dataset class.
- Preserve the current lazy-import behavior so preprocessing tests do not require
  torch unnecessarily.

Expected benefit:

- Better compatibility with DataLoader features and samplers.

## Important Corrections to Earlier Reviews

### TCN Receptive Field Is Not a Clear Bug

The earlier review says the TCN receptive field is only 30 timesteps. That
calculation ignores that each `TemporalBlock` has two dilated convolutions.

With `kernel_size=3` and dilations `[1, 2, 4, 8]`, the receptive field is roughly:

```text
1 + 2 * (kernel_size - 1) * sum(dilations)
= 1 + 2 * 2 * 15
= 61
```

That covers the 60-step input. Increasing hidden channels or dilations may still
be useful, but it is a capacity experiment, not a verified bug.

### Results Are Ignored by Git Already

The earlier review says there is no `.gitignore` for results. The current
`.gitignore` includes `results/`.

Action:

- No bug fix needed unless the project wants a different policy, such as tracking
  selected final CSV summaries while ignoring checkpoints.

### Do Not Automatically Lower the Regime Quantile

Lowering `regime_quantile` from `0.75` to `0.50` would make the classification
task easier and more balanced, but it also changes the meaning of "high risk."

Action:

- First keep the 75th percentile definition and tune the decision threshold.
- If the 75th percentile task remains unusable, add a sensitivity experiment for
  multiple quantiles and report the tradeoff honestly.

## Next Implementation Stage

### Stage 1: Reliability and Training Fixes

Implement:

- Training-loader shuffle.
- Gradient clipping.
- LR scheduler.
- Loss-component logging.
- Configurable multi-task loss weights.
- Longer serious-training defaults.
- Config inheritance fix.
- Lasso classification metric fix.

Then run:

```bash
pytest
python -m src.experiments.run_baselines --config configs/default.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/lstm.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/tcn.yaml --processed data/processed/features_all.csv
python -m src.training.train --config configs/multitask_tcn.yaml --processed data/processed/features_all.csv
```

Success criteria:

- Tests pass.
- All model metric CSVs are regenerated.
- Histories include learning rate and, for multi-task models, separate loss
  components.
- Lasso classification columns are populated.

### Stage 2: Classification Evaluation Fixes

Implement:

- Validation-set F1 threshold tuning.
- Saved classification threshold per model.
- Optional precision-recall curve data or figure.

Then compare:

- Default 0.5 threshold.
- Tuned validation threshold.
- Current 75th percentile regime definition.

Success criteria:

- Regime F1 is evaluated using a threshold selected without test leakage.
- Accuracy is no longer treated as the main classification metric.
- ROC-AUC and PR behavior are reported alongside F1.

### Stage 3: Missing Experiments

Run or repair:

- Ablation study.
- Walk-forward validation.
- Single-task versus multi-task comparison.

Success criteria:

- `results/tables/ablation_metrics.csv` exists.
- `results/tables/walk_forward_metrics.csv` exists.
- The report can answer whether derivative features helped.
- The report can answer whether multi-task learning helped.

### Stage 4: Figures and Report

Generate:

- Model comparison table.
- Predicted versus actual volatility plot.
- Confusion matrices.
- High-risk timeline.
- Ablation comparison plot.
- Walk-forward fold plot.
- At least one feature or target distribution figure.

Then update `reports/project_report.md` with:

- Actual results.
- Honest discussion of negative R2 values.
- Explanation of class imbalance.
- Comparison against naive, linear, and tree baselines.
- Limitations and next steps.

Success criteria:

- The report no longer reads like a template.
- Claims are supported by saved tables and figures.
- Negative findings are framed as a methodological result, not hidden.

## Later Improvements

These are useful but should come after the verified fixes and reruns:

- GARCH or HAR volatility baseline.
- Transformer baseline.
- Feature importance or permutation importance.
- Bootstrap confidence intervals or paired tests.
- Quantile sensitivity study for regime labels.
- Target transformation, such as log volatility.
- Attention-pooling experiment.
- Notebooks for EDA and results review.

## Working Principles

- Do not claim that training fixes caused improvement until reruns prove it.
- Do not change the research question just to make metrics look better.
- Keep test-set decisions leakage-free.
- Prefer small, measurable changes over large architecture rewrites.
- Treat simple baselines as serious competitors.
- Keep every regenerated result tied to a config and code version.

## Definition of Done for the Next Stage

The next stage is complete when:

- Core bugs are fixed.
- Tests pass.
- Main baselines and neural models are rerun.
- Classification thresholds are selected on validation data.
- Ablation and walk-forward CSVs exist.
- Figures exist for the primary result claims.
- `project_report.md` contains real results and an honest conclusion.

