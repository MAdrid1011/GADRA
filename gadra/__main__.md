# `__main__.py`

## External Interface

`main()` implements `python -m gadra` and the installed `gadra-sim` command.
It exposes low-level `baseline`, `capture`, and `simulate-case` operations.

## Internal Helpers

Subcommand imports are delayed so CPU replay does not require capture-only GPU
dependencies.
