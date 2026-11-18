# Experiment protocol

## Private scene manifest

Store experiment identities in `local/study.json`, using schema
`gadra-study/v1`. The manifest has these fields:

- `splits`: one object for each of `scannet`, `scannetpp`, and `matterport3d`.
  Each contains 10 distinct `calibration` scene IDs, five `tuning` scene IDs,
  and 10 independent `test` scene IDs. Calibration and tuning scenes come from
  the official training split, and test scenes from official validation.
- `classes`: the original evaluated class IDs, in query-column order, for each
  dataset. Ground-truth values outside this list are ignored.
- `models`: one object per model. It contains a private `readout_dir` and the
  complete list of native `fragments` across all three roles and datasets.

Each fragment entry supplies `dataset`, `role`, `scene_id`, `source_split`
(`training` or `validation`), `source_index` in the native adapter's scene list,
`fragment_index`, `fragment_count`, `tag`, and `raw` capture directory.
Its tag is `<Model>_<dataset>_s<source_index>_f<fragment_index>`.
The entry also supplies paths to `readout_manifest`, `orin_timing`,
`scene_indices`, and `scene_labels`. Paper aggregation also supplies
`implementation_contract` and `rtl_trace`, described in
[Implementation comparison](implementation.md). Paths resolve relative to the
manifest. A `.npy` `scene_indices` maps output rows to original scene positions.
For voxel-to-original backprojection, a `.npz` instead stores `vertex_ids` and
`prediction_rows`, mapping each original position to its native output row.
`scene_labels` contains original ground-truth labels. These inputs come from
the native preprocessing protocol and remain local.

`study.records()` checks split independence, scene counts, unique fragments,
and complete fragment-index coverage for each scene. Every evaluated scene
must have all indices from zero through `fragment_count - 1`.

## Calibration and threshold selection

Run `prepare --phase features` on calibration scenes, followed by
`prepare --phase fit`. Fitting samples at most 4096 positions per scene and
checkpoint across all fragments. Each model uses its 30 calibration scenes
across the three datasets. The fitted readout manifest records these identities
and hashes its weight files.

Run `prepare --phase capture` for calibration, tuning, and test roles with the
same readout manifest. Replay pools the fixed calibration observations to build
the quartile prediction tables. Test trajectories do not modify these tables.
Only completed consumers within the current inference update runtime responses.

Select the termination threshold on the tuning role. The default manuscript
configuration sets it to 0.02. For a threshold or residency-capacity sweep, use
a separate local configuration, set `sensitivity_study` to true, and pass
`--sensitivity`. A residency sweep changes `cam_entries` and
`feature_sram_bytes = cam_entries * 128` together. Source revision, configuration
hash, and frozen readout identity accompany each generated record.

## Board timing and power

Run `prepare --phase orin` on AGX Orin. The collector performs warmup and records
raw CUDA-event milliseconds for each complete encoding call. The baseline
loader requires an Orin source device and matching scene/fragment identities.

Power records name the measured `rail`, `interval` with `start_ns` and `end_ns`,
and `mean_watts`, and state
whether the measurement `excludes_offchip_dram`. Attach them to the private
timing record as `power`. Compute an on-chip energy comparison only from
measurements with the same DRAM boundary and the corresponding encoding
interval. `study.energy_efficiency()` checks these fields before calculation.

## Scene-level quality and performance

`evaluate` traverses all test fragments and runs `none`, `A`, `B`, and `AB`.
Each CDP run commits decisions checkpoint by checkpoint, records native task
admissions, and retains all issued work. Its decision file is hashed, and
scene aggregation reads that same file rather than recomputing exit depths.
It merges raw semantic query scores through original position indices, sums
scores from overlapping fragments, and makes one prediction per original
scene position. It checks that every labeled position has at least one
prediction. Ground truth comes from `scene_labels`, while full-depth outputs
provide the native-model comparison.

The evaluator records per-scene confusion matrices and accumulates them using
the original class set to compute dataset mIoU and mAcc. It sums actual Orin
fragment milliseconds and simulated cycles over the same scenes. GADRA time is
`cycles / frequency_hz * 1000`, and speedup is actual Orin milliseconds divided
by that time. Source revision and configuration hashes accompany the absolute
cycle, timing, and traffic records.

The default aggregation validates a supplied paired implementation record for
each fragment. `--model-only` performs architectural exploration and marks its
records `architectural_model_only`; it does not report an implementation
match. This flag does not change the online policy or resource capacities.

All generated measurements, calibration artifacts, predictions, and summaries
remain under `local/` or `results/`, which Git ignores. The repository publishes
code, configuration, tests, and instructions.
