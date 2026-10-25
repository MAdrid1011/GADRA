# `coverage.py`

## External Interface

`geometry()` loads real Gaussian support and `build_order()` emits the B
coverage order plus its metadata archive.

## Internal Helpers

KD-tree candidates are checked using the captured ellipsoid geometry before an
ordering edge is accepted.
