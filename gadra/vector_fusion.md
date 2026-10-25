# `vector_fusion.py`

## External Interface

Candidate functions identify legal integer-mask and BatchNorm fusion patterns;
join functions execute their fused finite writeback programs.

## Internal Helpers

Fusion removes intermediate materialization while retaining every vector
operation and dependency.
