# `copyroute.py`

## External Interface

`mapping()` and transfer helpers lower strided SRAM-to-SRAM copies into bounded
sector routes and control records.

## Internal Helpers

Byte-control construction and scratch accounting preserve bank, port, and
latency constraints.
