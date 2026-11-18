# `predictor.py`

`DemandLUT` calibrates and queries CDP's frozen quartile tables. The `paper-v1`
path uses `calibration.fit_tables()` on the independent calibration scenes.

`query_scores()` L1-normalizes each checkpoint's query scores.
`state_features()` computes adjacent-checkpoint L1 variation and the normalized
candidate-score gap. `remaining_correction(..., 'cumulative_l1')` sums all
subsequent adjacent-checkpoint L1 changes. The same definition supplies the
response of an already completed consumer.

Completed responses match checkpoint, original shared context, and winning
category. Up to four applicable records reduce by maximum. Missing feedback
uses the local table. `decide()` compares the larger of the selected table
prediction and applicable response against the configured threshold. Output
quality is evaluated separately from the causal decision.
