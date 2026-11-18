# Scripts

Model-specific preparation and replay commands use a private, fixed scene
manifest. Preparation extracts calibration features, fits frozen readouts,
captures all scene fragments, or collects native Orin timing. Replay evaluates
`none/A/B/AB` and aggregates predictions using original scene positions.

See [Experiment protocol](../docs/evaluation.md) for input fields and commands.
Generated artifacts stay in ignored local directories.
