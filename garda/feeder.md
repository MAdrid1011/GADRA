# `feeder.py`

## External Interface

`execute()` supplies matrix operands through bounded buffers; capacity, row,
column, and transfer helpers are also used by the engine's specialized paths.

## Internal Helpers

Compiled tile loops retain every bank access, copy, padded array tile, output
drain, and ownership interval.
