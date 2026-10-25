# `prepare_scenesplat.py`

## External Interface

Runs SceneSplat on all three real datasets and writes capture bundles to
`local/captures/scenesplat`. `--output` and `--repeats` are supported.

## Internal Helpers

Each official capture is followed by real-geometry coverage ordering and a GPU
baseline measurement.
