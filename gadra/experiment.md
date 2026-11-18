# `experiment.py`

`run_case()` replays one explicitly identified manifest fragment with the
manuscript profile, fixed independent calibration records, and matching
board-measured Orin timing. It selects the model's declared stage checkpoints
and requires the L1 cumulative diagnostic definition.

The function executes `none`, `A`, `B`, and `AB` on the same physical profile.
Output termination retains the original dependency closure of active
consumers. Predictor tables, bounded active response state, diagnostic work,
metadata transfers, and DRAM spills enter the replay accounting.

`study.py` invokes this fragment replay over the complete fixed scene manifest
and aggregates quality at original scene positions. Generated output stays in
ignored local directories.
