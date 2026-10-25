# `ready_queue.py`

## External Interface

`ReadyQueue` provides finite non-preemptive reservation and retirement for the
shared vector issue queue.

## Internal Helpers

Sorted host records model fixed-size control entries rather than uncharged
accelerator memory.
