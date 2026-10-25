# `graph.py`

## External Interface

`Graph` validates and loads a capture, and `DemandPlan` stores conservative
per-row demand closure. `rows()`, `key()`, and `coverage_layout()` are shared
shape and identity helpers.

## Internal Helpers

Versioned storage edges prevent replay from consuming overwritten or future
values.
