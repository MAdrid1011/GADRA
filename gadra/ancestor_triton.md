# `ancestor_triton.py`

## External Interface

`merge_gelu()` launches the fused Chorus ancestor-projection kernel.

## Internal Helpers

`_merge` is the Triton implementation for mapped gathers, normalization
correction, optional fine projection, and GELU.
