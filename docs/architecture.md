# Architecture

GADRA separates numerical execution from architectural replay. Numerical
capture runs an unmodified real model input and records operator shapes,
dependencies, sparse mappings, semantic observations, geometry, and a GPU
reference. Replay consumes those records without invoking a synthetic model or
substituting precomputed stop masks.

## Data Flow

1. `scripts/prepare_*.py` selects the first real spatial fragment for each
   dataset and invokes the official model runtime through `gadra.workloads`.
2. `gadra.capture` records tensor and sparse-graph boundaries;
   `gadra.coverage` derives the real ellipsoid coverage order.
3. `gadra.graph` validates and loads the capture into a dependency graph.
4. `gadra.engine` schedules that graph through finite arrays, vector lanes,
   SRAM banks, queues, DMA, and DRAM.
5. `gadra.experiment` runs the independent `none`, `A`, `B`, and `AB` variants,
   checks semantic quality, and emits full accounting under `results/`.

## Timing Boundary

The reported latency covers complete 3D semantic encoding for the selected
fragment. Disk access and host-side dataset preprocessing are excluded. Matrix,
vector, control, transfer, dependency, and capacity stalls inside encoding are
included. The optimistic resource floor is the maximum of finite-array,
vector-unit, and measured-traffic DRAM service time under perfect global
overlap; it is a lower bound rather than another measured design.

## Hardware Invariants

Every configuration has 4,096 FP32 accumulation PEs, a 1.25 GHz clock, a
two-cycle FP32 MAC, 2,048 vector lanes, 16 MiB feature SRAM, 8 MiB weight SRAM,
4 MiB output SRAM, 0.5 MiB control SRAM, and 204.8 GB/s DRAM. The general
feature directory has at most 8,192 entries; its allocation, transfer, and
eviction slot is 2 KiB. The 64-entry coverage candidate window and 64-row
coverage group are scheduler limits, not the capacity or line size of a
separate resident-feature structure. Subarray shape may change, but total PE
count and capacity do not. Original-base and optimized runs use separate
tracked configuration files.

## Projection Boundary

GPU baselines use CUDA-event timing and real operator/traffic counters. AGX
Orin values scale those observations with tracked compute and bandwidth
assumptions. GADRA labels them `projection_only` and `Orin_measured: false`;
they must not be described as measurements on an Orin device.
