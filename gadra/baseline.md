# `baseline.py`

`from_orin_measurement()` validates actual AGX Orin CUDA-event samples and
returns their median in milliseconds. `project()` is the compatibility-named
loader used by graph replay. It matches the trace's scene, fragment, and row
identity to the measured baseline. `reference_metadata()` accepts board-measured
records with the `orin_cuda_events` method.

The module also parses Nsight counter exports and classifies native arithmetic
by precision. Power records retain their source rail and encoding interval.
