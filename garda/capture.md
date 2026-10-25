# `capture.py`

## External Interface

`capture()` runs one real model/dataset case and writes its operator graph,
sparse maps, observations, and geometry. `trace_type()` provides the model-aware
PyTorch dispatch recorder.

## Internal Helpers

Trace dumping, semantic readout collection, and native geometry export retain
the concrete model execution while excluding host preprocessing from timing.
