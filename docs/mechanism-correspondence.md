# Mechanism correspondence

## CDP

At checkpoint d, CDP forms the fixed-query score vector q and normalizes it by
its absolute sum, giving z = q / max(sum(abs(q)), smallest positive FP32 value).
The two-checkpoint trajectory statistic is sum(abs(z[d] - z[d-1])). The score
gap is the largest normalized score minus the second largest.

The calibration target at checkpoint d is the sum of adjacent-checkpoint L1
changes from d+1 through the final checkpoint. This includes later reversals,
even when the final winning category equals the current winner. The same
target defines the response of an already completed consumer.

SceneSplat uses four encoder and three recovery checkpoints with 768-dimensional
semantic readouts. Chorus uses five encoder checkpoints and 1152-dimensional
readouts of the available multiscale features. EmbodiedSplat uses four encoder
and four decoder checkpoints, mapping to its frozen 64-dimensional semantic
head input and then to the original 768-dimensional semantic output. The final
native output is also recorded as the reference endpoint. Original position
maps align each readout to its Gaussian consumers.

`prepare.fit_readouts()` fits affine least-squares diagnostic maps on the fixed
calibration scenes while preserving backbone and native-head weights.
`calibration.fit_tables()` pools up to 4096 positions per scene across its
fragments, builds quartile bins, and stores the 95th percentile of remaining
cumulative L1 change. Cells with fewer than 64 observations continue execution.
Fitted artifacts remain local and are identified by hashes during capture and
replay.

Completed response records match checkpoint, shared input, and current winning
category. Up to four completed consumers per context provide records, with
matching responses reduced by maximum. A matched conditional table combines
variation, score gap, and response. A missing response uses the local table.
The larger of the selected table output and applicable response is compared
with the default termination threshold of 0.02. Only completed consumers supply
future-response evidence. Their computation and response publication precede
any decision that uses it.

Output termination cancels later output work. Original dependency closure
retains producers referenced by other active consumers. `predictor.py` models
the decision, `diagnostic_pipeline.py` lowers its paid observation work, and
`engine.py` issues the retained dependency tasks.

## CGR

For Gaussian centers mu and three-axis scales s, the priority is
`sum((mu_i-mu_j)^2) / (sum(s_i^2) + sum(s_j^2))`. Original convolution edges,
attention membership, pooling maps, and recovery indices determine actual
shared-input consumers. Scale-normalized distance determines their ordering.

The scheduler advances input-ready groups that reuse resident fragments and
need fewer additional inputs. Exact feature tags select the 64 physical
128 B slots. Remaining consumers keep a fragment eligible for retention.
Finished consumption releases its slot, and evicted live data retains its
backing reference for subsequent access. CDP updates active consumers, while
CGR manages their shared feature lifetimes.

## Variants

`none` disables CDP and CGR. `A` enables CDP, `B` enables CGR, and `AB` enables
both on the same hardware profile. The default and base configuration files
share the manuscript clock, array shape, and storage capacities. Generated
records identify the exact source and configuration used for each run.
