# Configuration

All six tracked configurations use the manuscript `paper-v1` physical profile:
1 GHz, one 64 × 64 single-cycle FP32 MAC array, 64 directory entries with
128 B payload slots, and 152 KiB total storage. `gadra.profile.validate()` checks
the profile and reports its fingerprint.

Model-specific options select legal operator scheduling and index handling.
The base presets use the same clock, array shape, and storage capacities.
`none`, `A`, `B`, and `AB` switch CDP and CGR on that common resource budget.

Threshold and paired directory/payload sensitivity settings belong in local
configuration files. Set `sensitivity_study` to true, update the desired
threshold or paired capacity, and use `evaluate --sensitivity`. Each directory
entry owns one 128 B slot. The scheduler candidate window is a separate limit.
