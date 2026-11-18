# GADRA

GADRA provides executable models of context-demand prediction (CDP),
coverage-guided residency (CGR), and their shared task and memory dataflow for
3D Gaussian scene understanding. It captures native SceneSplat, Chorus, and
EmbodiedSplat graphs and replays their dependencies on a finite-resource
architecture.

The default `paper-v1` profile uses one 64 × 64 FP32 MAC array at 1 GHz,
64 resident-feature slots of 128 B, and 64 B DRAM transactions. Its 152 KiB
storage budget includes 8 KiB shared features, 32 KiB weights, 16 KiB
accumulator registers, 16 KiB output SRAM, and 80 KiB control/history SRAM.
The 64-entry feature directory directly owns the 64 payload slots. Scheduler
candidate windows and consumer queues have separate limits.

CDP uses L1-normalized query scores, adjacent-checkpoint L1 variation, candidate
score gaps, and completed shared-consumer responses to predict remaining
cumulative semantic change. CGR orders consumers using scale-normalized Gaussian
center distances and the original network dependencies, then tracks exact
feature identity and remaining uses.

## Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m gadra config
python -m unittest discover -s tests -v
```

Native capture additionally uses the upstream CUDA/PyTorch environments and
runtime adapters described in [Workload Inputs](docs/workloads.md).

## Run

Create the private scene manifest at `local/study.json` following the
[experiment protocol](docs/evaluation.md). It fixes independent calibration,
tuning, and test scenes and enumerates every native inference fragment.

```bash
# Extract calibration features and fit frozen diagnostic maps.
python -m gadra prepare --model SceneSplat --manifest local/study.json --phase features
python -m gadra prepare --model SceneSplat --manifest local/study.json --phase fit

# Capture each split with the same frozen maps.
python -m gadra prepare --model SceneSplat --manifest local/study.json --phase capture --role calibration
python -m gadra prepare --model SceneSplat --manifest local/study.json --phase capture --role tuning
python -m gadra prepare --model SceneSplat --manifest local/study.json --phase capture --role test

# Run this command on the AGX Orin board for actual CUDA-event samples.
python -m gadra prepare --model SceneSplat --manifest local/study.json --phase orin --role test

# Replay all test fragments and aggregate predictions at original scene positions.
python -m gadra evaluate --model SceneSplat --manifest local/study.json \
  --config configs/scenesplat_scannet.json --output results/paper/scenesplat
```

Use `Chorus` with `configs/chorus_scannet.json`, or `EmbodiedSplat` with
`configs/embodiedsplat_scannet.json`, for the other models. The model-specific
scripts in `scripts/` expose the same manifest-driven interface.

The repository contains source, configuration, tests, and documentation.
Datasets, model weights, fitted readouts, timing records, predictions, and
generated reports remain in ignored local directories.

## Documentation

- [Architecture and physical storage](docs/architecture.md)
- [CDP and CGR definitions](docs/mechanism-correspondence.md)
- [Fixed-split, all-fragment evaluation](docs/evaluation.md)
- [External workload adapters](docs/workloads.md)
