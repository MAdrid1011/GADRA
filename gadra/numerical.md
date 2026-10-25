# `numerical.py`

## External Interface

`Replay` reconstructs numerical execution from initial tensors. `metrics()` and
`bounded_metrics()` compute the formal semantic quality fields.

## Internal Helpers

Storage-view and recorder helpers preserve aliasing and operator semantics; the
replay never treats recorded feature outputs as inputs.
