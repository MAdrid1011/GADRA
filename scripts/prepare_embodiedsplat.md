# `prepare_embodiedsplat.py`

## External Interface

Runs EmbodiedSplat on one or all three real datasets. `--dataset`, `--output`,
and `--repeats` control selection, destination, and GPU timing repetitions.

## Internal Helpers

Each case captures the native-cache execution, semantic observations, geometry,
coverage order, and GPU reference.
