#!/usr/bin/env python3
"""Capture EmbodiedSplat inputs and GPU references from three real datasets."""

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
        "--output", type=Path, default=ROOT / "local/captures/embodiedsplat"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/embodiedsplat_scannet.json").read_text())

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        capture(
            "EmbodiedSplat",
            dataset,
            0,
            args.output,
            query_fp16=True,
            query_transposed=True,
            query_factorized=True,
            embodied_native_previous=True,
            output_fp16=True,
            terminal_latent_query=True,
        )
        tag = f"EmbodiedSplat_{dataset}_s0"
        _, metadata = build_order(
            "EmbodiedSplat",
            dataset,
            0,
            config,
            geometry_path=args.output / f"{tag}_geometry.npz",
            archive_path=args.output / f"{tag}_coverage.npz",
        )
        (args.output / f"{tag}_coverage.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        reference = gpu_baseline("EmbodiedSplat", dataset, 0, args.repeats)
        (args.output / f"{tag}_gpu.json").write_text(
            json.dumps(reference, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
