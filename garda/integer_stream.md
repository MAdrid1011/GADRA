# `integer_stream.py`

## External Interface

`execute()` dispatches finite integer operations; `needed()` reports resource
requirements and `Stream` owns packetized state.

## Internal Helpers

Radix, unique, scatter, scan, and compact programs use vector lanes, output
owners, SRAM, and DRAM explicitly.
