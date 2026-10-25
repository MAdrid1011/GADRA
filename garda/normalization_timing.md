# `normalization_timing.py`

## External Interface

BatchNorm preparation, generation, and release functions schedule finite global
statistics followed by row-local affine work.

## Internal Helpers

Chunk reduction, reference, control-beat, and allocation helpers share the
engine's vector, SRAM, DMA, and output resources.
