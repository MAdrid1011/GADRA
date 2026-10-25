# `workloads.py`

## External Interface

`load_case()` loads one official model/dataset input, `gpu_baseline()` measures
it, and `select_fragment()` applies the deterministic real-fragment policy.

`GARDA_WORKLOADS_ROOT` overrides the default `local/workloads` asset root.

## Internal Helpers

Adapters import the prepared external runtime modules without copying
third-party code into this repository.
