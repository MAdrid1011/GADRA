# `memory_core.py`

## External Interface

Directory, transfer, access, eviction, residency, and matrix-tile functions
implement the simulator's compiled SRAM/CAM/DMA state transitions.

## Internal Helpers

The same source executes with or without Numba; compilation changes host speed,
not modeled behavior.
