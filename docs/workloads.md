# Workload Inputs

GARDA publishes small manifests but not third-party repositories, licensed
datasets, checkpoints, or multi-gigabyte captures. Set `GARDA_WORKLOADS_ROOT`
to a prepared external tree, or place that tree at `local/workloads/`.

## External Tree

```text
local/workloads/
  repos/{SceneSplat,Chorus,EmbodiedSplat}/
  data/{scannet,scannetpp,matterport3d}/
  checkpoints/
  results/{assets.json,chorus_assets.json,scenesplat_fp16.json,...}
  scripts/{scenesplat_runtime.py,chorus_runtime.py,...}
  build/minkowski_graph_backend/
```

The runtime adapters resolve official model checkpoints and concrete dataset
scenes from the asset metadata. SceneSplat and Chorus use the first deterministic
spatial fragment selected from a real scene. EmbodiedSplat uses its first real
native-cache scene. Calibration and evaluation roles are kept separate by the
asset metadata.

## Capture Contract

For each model/dataset pair, `local/captures/<model>/` contains:

- `<Model>_<dataset>_s0.json`: operator and dependency trace;
- `_maps.pkl`: captured sparse maps used by semantic replay;
- `_observations.npz`: real per-stage semantic observations;
- `_geometry.npz`: real Gaussian geometry;
- `_coverage.npz` and `_coverage.json`: coverage order and metadata;
- `_gpu.json`: measured local-GPU reference and Orin projection inputs.

Replay rejects missing files and mismatched model or dataset identities. It
does not generate random replacements or silently fall back to toy inputs.

## Commands

The manifests in `workloads/` name the capture script, replay script,
configuration, external model repository, and dataset roots for each model.
The root README gives the complete commands. Capture needs the CUDA environment
that can run the upstream model; replay needs the package dependencies in
`pyproject.toml` and the Numba-compiled simulator path.
