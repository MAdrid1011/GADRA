# `readout.py`

Affine recipes define the captured SceneSplat and Chorus diagnostic source,
input order, bias augmentation, and semantic projection. EmbodiedSplat uses an
affine map to its original 64-dimensional head input and the frozen native
semantic head.

`prepare.fit_readouts()` fits maps from calibration-only feature exports.
`capture.py` records the frozen weight identity, stage maps, and corresponding
semantic observations. `diagnostic_pipeline.py` lowers the same readout
structure to the modeled matrix and vector units.
