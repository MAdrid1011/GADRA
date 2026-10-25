# `reproduce_chorus_scannet.py`

## External Interface

Replays Chorus original base and `none/A/B/AB` for one or all real datasets.
The CLI accepts capture, configuration, output, GPU-reference, worker, and reuse
options.

## Internal Helpers

Summary generation computes speedups, resource floors, composition targets,
and fixed semantic quality checks.
