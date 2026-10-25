# `concatenation.py`

## External Interface

`lower()` identifies legal sole-consumer concatenations and `join()` executes
their physical placement during replay.

## Internal Helpers

Planning preserves tensor versions and logical dependencies while avoiding
redundant intermediate copies.
