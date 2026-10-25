# `activity.py`

## External Interface

`Activity` records aggregate execution intervals and `sequential_activity()`
combines non-overlapping interval summaries.

## Internal Helpers

Interval storage is host-side reporting state and never supplies readiness or
accelerator capacity.
