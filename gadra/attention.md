# `attention.py`

## External Interface

Capacity helpers describe attention staging, while `pipeline()` and
`normalize()` execute bounded QK, softmax, and AV work on simulator resources.

## Internal Helpers

Staging keys, scratch transfers, and register bookkeeping enforce finite
ownership and reuse.
