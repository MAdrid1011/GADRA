#!/usr/bin/env python3
"""Capture SceneSplat inputs and GPU references from three real datasets."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from garda.capture import capture
from garda.coverage import build_order
from garda.workloads import gpu_baseline


DATASETS = ("scannet", "scannetpp", "matterport3d")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "local/captures/scenesplat"
    )
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/scenesplat_scannet.json").read_text())

    for dataset in DATASETS:
        capture(
            "SceneSplat",
            dataset,
            0,
            args.output,
            query_fp16=True,
            query_transposed=True,
            output_fp16=True,
        )
        tag = f"SceneSplat_{dataset}_s0"
        _, metadata = build_order(
            "SceneSplat",
            dataset,
            0,
            config,
            geometry_path=args.output / f"{tag}_geometry.npz",
            archive_path=args.output / f"{tag}_coverage.npz",
        )
        (args.output / f"{tag}_coverage.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        reference = gpu_baseline("SceneSplat", dataset, 0, args.repeats)
        (args.output / f"{tag}_gpu.json").write_text(
            json.dumps(reference, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
