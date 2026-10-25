# `engine.py`

## External Interface

`Simulator` executes a captured graph. `Memory` and `ReferenceMemory` provide
the finite and accounting memory models used by each variant.

## Internal Helpers

Issue, dependency, allocation, retirement, coverage, diagnostic, and transfer
logic share one cycle timeline.
