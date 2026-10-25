# `writeback.py`

## External Interface

`candidates()` finds legal sole-consumer cast folds and `join()` performs their
finite producer writeback.

## Internal Helpers

Mutable, observed, reduction, and mismatched-view cases conservatively fall
back to native storage.
