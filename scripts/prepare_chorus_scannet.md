# `prepare_chorus_scannet.py`

## External Interface

Runs Chorus on ScanNet, ScanNet++, and Matterport3D, writing real captures and
GPU references to `local/captures/chorus` by default. `--output` and `--repeats`
override destination and timing repetitions.

## Internal Helpers

The script derives coverage ordering after each official model capture.
