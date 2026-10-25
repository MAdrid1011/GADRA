# Mechanism Correspondence

| Mechanism | Implementation | Charged evidence |
|---|---|---|
| A: context-demand termination | `predictor.py`, `diagnostic_pipeline.py`, `checkpoint.py`, `readout.py` | Predictor tables, witnesses, response/correction records, diagnostic vector/matrix work, control-SRAM peak, removed matrix FLOPs, and quality gates |
| B: workload-aware supply | `coverage.py`, `coverage_rows.py`, `feeder.py`, `wavefront.py`, `memory_core.py` | Real ellipsoid order, resident features, bounded feeder transfers, bank/queue service, finite array occupancy, DMA, and DRAM bytes |
| Shared A/B path | `engine.py`, `experiment.py`, `vector_fusion.py` | Dependency-ready issue, shared scheduling, fused vector retirement, total cycles, same-resource controls, and resource-clipped composition target |

## Variant Definitions

`none` uses the optimized hardware resources with A and B disabled. `A` and
`B` enable one mechanism at a time on that same resource baseline. `AB` enables
both. `original base` uses the tracked single-`64x64` pre-adaptation
configuration.

The common scheduling path is a prerequisite of both mechanisms. AB instantiates
that path once, then applies A's termination decisions and B's supply policy on
the same finite array, vector, SRAM, and DRAM resources.

## Predictor Definition

At checkpoint $d$, A forms the query-score vector $q_{i,d}$ and stores
$z_{i,d}=q_{i,d}/\max(\lVert q_{i,d}\rVert_2,\epsilon)$. Its trajectory signal
is the windowed mean of $\lVert z_{i,t}-z_{i,t-1}\rVert_1$. Let
$c_{i,d}=\arg\max_k q_{i,d,k}$ and let $D$ be the final checkpoint.
Calibration and replay use
$\max_k q_{i,D,k}-q_{i,D,c_{i,d}}$, the final-checkpoint disadvantage of the
current winner, as the decision-aware remaining correction. Thus L2
normalization, L1 trajectory variation, and final-score correction are
distinct parts of one definition rather than competing normalization
conventions.

## Runtime Validation

Each run compares early outputs against full-depth inference and enforces the
quality thresholds stored in the selected configuration. A failed gate aborts
summary generation instead of silently accepting the variant.
