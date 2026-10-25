# `coverage_references.py`

## External Interface

`RowInterval`, `prepare()`, `storage_bytes()`, and `intersection()` encode and
query finite native-row references for coverage-qualified inputs.

## Internal Helpers

Intervals compress consecutive identities only and never imply feature
residency.
