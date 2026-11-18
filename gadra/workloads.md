# `workloads.py`

The adapters select real model inputs and preserve native preprocessing.
`select_fragment()` selects an explicit fragment from the complete scene
partition. `load_case()` resolves independent calibration, tuning, and test
source roles. The legacy `chunk-first` scope supports a small local example.

`gpu_baseline()` runs on AGX Orin, warms up native encoding, and records raw
CUDA-event samples and input identity. Formal evaluation uses the complete
scene manifest in `study.py`.
