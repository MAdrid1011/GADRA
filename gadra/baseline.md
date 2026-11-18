# `baseline.py`

## External Interface

`project()` produces the stage-wise AGX Orin reference baseline used by formal runs.
Counter parsing and arithmetic classification functions support that report.

## Internal Helpers

Reference scenarios scale RTX CUDA-event timing using each stage's arithmetic,
DRAM traffic, and configured RTX/Orin compute and bandwidth parameters.
`reference_metadata()` accepts legacy reference labels and preserves the recorded
timings, counter sources, and measurement metadata.
