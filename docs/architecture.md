# Architecture

`gadra.engine` replays native operator/dependency graphs through a MAC array,
vector execution, banked feature storage, finite task queues, and a shared DRAM
port. `gadra.profile.validate()` checks the manuscript hardware profile before
formal replay, and each generated run records its configuration fingerprint.

## Physical resources

| Resource | Default |
|---|---:|
| Clock | 1 GHz |
| FP32 MAC array | One 64 × 64 array, one MAC per PE per cycle |
| FP32 peak | 8.192 TFLOP/s |
| Shared-feature directory and payload | 64 entries, 128 B per slot, 8 KiB total |
| Weight SRAM | 32 KiB |
| Accumulator registers | 16 KiB |
| Output SRAM | 16 KiB |
| Query-history SRAM | 64 KiB |
| Tables, active responses, queues, and control metadata | 16 KiB |
| Total storage | 152 KiB |
| DRAM transaction | 64 B |
| DRAM bandwidth and request latency | 204.8 GB/s, 180 ns |

The scalar configuration is one FP MAC, one iterative FP unit, and one
sequencer. `vector_lanes = 1` specifies serialized scalar microprogram work.
These resources are independent of the 64 × 64 MAC array. Matrix work and
scalar work therefore do not borrow capacity from each other. All scalar
reservations, including compound metadata programs and their memory waits,
occupy the same bounded ready queue. Arithmetic work cycles and reserved
intervals are recorded separately: an interval may also include waiting time.

`protocol.py` defines the executed 128 B allocation and 64 B transfer granules.
Feature capacity is `min(cam_entries, feature_sram_bytes / 128)`, giving exactly
64 resident slots at the default setting. Hit masks, dirty masks, fills,
evictions, and writebacks use these same granules. Residency sensitivity varies
the number of CAM entries and payload slots together.

The feature directory matches original feature identity and channel fragment.
A scheduler candidate window does not increase directory or payload capacity.
The history partition holds active query state. The active response store has
64 entries. Inactive state is backed in DRAM, and state eviction, reload, and
writeback use the shared transaction model. Frozen tables, active responses,
owner-selection state, and ready-queue metadata share the 16 KiB metadata
partition. Operand and accumulator windows are charged to their declared
physical storage.

## Timing and baseline

Encoding replay includes diagnostic projections, grouping, task issue,
state access, data transfers, and spill/writeback traffic. Offline fitting,
model loading, and common input preparation precede this interval.

The GPU baseline reads CUDA-event samples collected on AGX Orin for the same
scene fragment. `baseline.py` validates device identity, input identity, and
positive raw samples. It stores their median as `orin_measured_ms` and records
`Orin_measured: true`. The formal comparison does not transform another GPU's
timing into an Orin measurement.

Generated records include source revision, configuration fingerprint, scene and
fragment identity, actual Orin milliseconds, simulated cycles, frequency, and
DRAM bytes. PPA comes from the physical implementation flow described in the
manuscript. The Python replay engine supplies architectural timing and traffic
accounting.

Scalar microprogram steps are an architectural work model. A physical timing
comparison additionally needs the implementation's operator mapping and a
paired execution record. `implementation.py` checks source hashes, resource
counts, scalar reservations, task admissions, decisions, cancellations, and
total cycles before paper aggregation. The mapping must include the iterative
operations used by the native graph; the model does not establish their
physical latency by declaring a lane count.
