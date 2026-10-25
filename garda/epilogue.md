# `epilogue.py`

## External Interface

`plans()` finds legal sole-consumer dense GELU epilogues.

## Internal Helpers

The planner only accepts cases supported by the existing dense row-window
execution path.
