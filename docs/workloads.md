# Workload inputs

The native adapters use official SceneSplat, Chorus, and EmbodiedSplat model
repositories, checkpoints, datasets, and preprocessing. Set
`GADRA_WORKLOADS_ROOT` to the prepared external tree:

```text
local/workloads/
  repos/{SceneSplat,Chorus,EmbodiedSplat}/
  data/{scannet,scannetpp,matterport3d}/
  checkpoints/
  scripts/{scenesplat_runtime.py,chorus_runtime.py,...}
  results/{assets.json,chorus_assets.json,scenesplat_fp16.json,...}
  build/minkowski_graph_backend/
```

SceneSplat and Chorus use 1 cm presampling and 2 cm inference grids.
EmbodiedSplat uses native 2 cm voxel/cache inputs. Native adapters expose scene
identity and split, deterministic fragment enumeration, original position
indices, ground-truth labels, model invocation, and pretrained semantic-head
weights. Calibration, tuning, and evaluation roles resolve to independent
source scene lists.

`prepare.py` visits every entry of the private study manifest. SceneSplat and
Chorus select the specified fragment from the complete native scene partition.
EmbodiedSplat visits the selected complete native-cache scene. The scene
manifest covers every inference unit used by that model's original protocol.
The low-level `chunk-first` adapter remains available for a quick local example.
Formal evaluation validates complete scene coverage before replay.

Each native capture writes operator/dependency metadata, integer maps,
per-checkpoint observations, and Gaussian geometry under its unique fragment
tag. Calibration feature capture additionally exports native stage features
and corresponding full-depth fitting targets. Frozen readout identity is
recorded in every diagnostic capture. Existing captures from another diagnostic
definition require capture with the selected frozen maps before paper replay.

The [experiment protocol](evaluation.md) defines the private manifest, original
position/label arrays, Orin timing records, and aggregation commands. Model
assets and generated numerical records stay in ignored local directories.
