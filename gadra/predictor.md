# `predictor.py`

## External Interface

`DemandLUT` calibrates causal exit actions. Selection, response, witness,
storage, feature, and correction helpers implement A's policy and accounting.

`query_scores()` L2-normalizes each checkpoint's query-score vector.
`state_features()` computes the windowed L1 variation between adjacent
normalized vectors, while `remaining_correction()` measures the decision-aware
disadvantage, at the final checkpoint, of the current checkpoint's winner.

## Internal Helpers

Exact error-budget selection uses only completed observations and fixed quality
limits.
