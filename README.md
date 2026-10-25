# GADRA

GADRA is a cycle-level simulator for generalized 3D Gaussian semantic
encoding. It captures real SceneSplat, Chorus, and EmbodiedSplat executions on
ScanNet, ScanNet++, and Matterport3D, then replays their operator and dependency
graphs on a fixed 4,096-PE FP32 architecture.

The repository contains only the simulator, real-workload capture/replay
entrypoints, configuration, manifests, and mechanism documentation. Model
repositories, datasets, checkpoints, multi-gigabyte captures, and regenerated
results stay under the ignored `local/` and `results/` directories.

## Mechanisms

- **A: context-demand termination.** A calibrated diagnostic readout ends
  per-Gaussian semantic processing once the result meets the fixed quality
  gates. The simulator charges predictor, witness, correction, and control-SRAM
  costs.
- **B: workload-aware supply.** Coverage ordering, shared-feature residency,
  bounded operand feeding, and row-chain scheduling improve utilization while
  retaining finite SRAM, DRAM, queue, and port constraints.
- **AB:** both mechanisms share the same bounded scheduling and supply path.
  The joint variant accounts for that shared path once and applies the same
  finite resource constraints as the independent variants.

The target runs at 1.25 GHz with 4,096 FP32 accumulation PEs, a two-cycle FP32
MAC, 2,048 vector lanes, 16/8/4/0.5 MiB feature/weight/output/control SRAM, and
204.8 GB/s DRAM. SceneSplat uses one `64x64` array. Chorus and EmbodiedSplat use
four `32x32` subarrays with the same total PE count; every original-base run
uses one `64x64` array.

## Install

The replay simulator requires Python 3.10 or newer:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

Real capture additionally requires CUDA, PyTorch, SciPy, the upstream model
environments, datasets, and checkpoints described in
[Workload Inputs](docs/workloads.md).

## Local Assets

Place or link the external workload tree at `local/workloads`, or point GADRA
to an existing tree:

```bash
export GADRA_WORKLOADS_ROOT=/path/to/real/workloads
```

The tree must contain the upstream `SceneSplat`, `Chorus`, and `EmbodiedSplat`
repositories, the three datasets, checkpoints, runtime adapters, and asset
metadata. The tracked manifests in `workloads/` record the required identity
and commands. Captures are written to `local/captures/<model>/`.

## Capture

Run the official model paths to rebuild all nine inputs and GPU references:

```bash
CAPTURE_PYTHON="${GADRA_WORKLOADS_ROOT:-$PWD/local/workloads}/.venv-cu121/bin/python"
OMP_NUM_THREADS=4 "$CAPTURE_PYTHON" scripts/prepare_scenesplat.py --repeats 7
OMP_NUM_THREADS=4 "$CAPTURE_PYTHON" scripts/prepare_chorus_scannet.py --repeats 7
OMP_NUM_THREADS=4 "$CAPTURE_PYTHON" scripts/prepare_embodiedsplat.py --repeats 7
```

These commands execute real inputs. There is no synthetic or test-only
fallback.

## Reproduce

Replay original base and `none/A/B/AB` for all three datasets:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/reproduce_scenesplat.py --dataset all --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/reproduce_chorus_scannet.py --dataset all --workers 4

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/reproduce_embodiedsplat.py --dataset all --workers 4
```

Use `--dataset scannet`, `scannetpp`, or `matterport3d` for one case. Generated
records go to `results/` and are intentionally untracked. Every replay checks
the quality gates declared by the selected configuration before producing a
summary.

See [Architecture](docs/architecture.md) for the simulator boundary and
[Mechanism Correspondence](docs/mechanism-correspondence.md) for the mapping
from A/B to implementation and resource accounting.
