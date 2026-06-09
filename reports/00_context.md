# Project Context

This project started as a broader exploration of derivative-aware financial time
series modeling. The cleaned public version focuses on a narrower and more
defensible question:

> Do compact financial-kinematics features improve future realized-volatility
> forecasting under leakage-aware chronological validation?

The main workflow now uses:

- classical volatility baselines,
- a single-task LSTM sequence model,
- vanilla TCN as an architecture comparison,
- LSTM feature ablation,
- expanding walk-forward validation.

Experimental multi-task, attention, and TCN-MLP variants are retained in the
codebase for future exploration, but they are not part of the main published
claim.

