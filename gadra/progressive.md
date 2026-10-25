# `progressive.py`

## External Interface

`progressive_projector()` and `ProgressiveHead` provide the causal channel-level
numerical path whose operations can be captured by `gadra.capture`.

## Internal Helpers

Calibration, context-bank, checkpoint, and observation helpers consume real
model states and completed consumers; this module is not a synthetic fallback.
