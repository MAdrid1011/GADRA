# `control_timing.py`

## External Interface

`capture()` annotates integer-control operators and `execute()` lowers them to
finite vector, SRAM, and queue service.

## Internal Helpers

Compaction, scans, and indexed updates use captured topology only; they never
carry semantic feature values.
