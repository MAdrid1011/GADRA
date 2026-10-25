"""Command-line access to real model capture and complete four-way replay."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    baseline = commands.add_parser("baseline")
    baseline.add_argument("--model", choices=("SceneSplat", "Chorus", "EmbodiedSplat"), required=True)
    baseline.add_argument("--dataset", required=True)
    baseline.add_argument("--index", type=int, default=0)
    baseline.add_argument("--repeats", type=int, default=7)
    baseline.add_argument("--output", type=Path, required=True)

    capture = commands.add_parser("capture")
    capture.add_argument("--model", choices=("SceneSplat", "Chorus", "EmbodiedSplat"), required=True)
    capture.add_argument("--dataset", required=True)
    capture.add_argument("--index", type=int, default=0)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--query-fp16", action="store_true")
    capture.add_argument("--query-transposed", action="store_true")
    capture.add_argument("--query-partitioned", action="store_true")
    capture.add_argument("--output-fp16", action="store_true")
    capture.add_argument("--output-partitioned", action="store_true")

    simulate = commands.add_parser("simulate-case")
    simulate.add_argument("--model", choices=("SceneSplat", "Chorus", "EmbodiedSplat"), required=True)
    simulate.add_argument("--dataset", required=True)
    simulate.add_argument("--raw", type=Path, required=True)
    simulate.add_argument("--training-raw", type=Path)
    simulate.add_argument("--reference", type=Path)
    simulate.add_argument("--config", type=Path, required=True)
    simulate.add_argument("--output", type=Path, required=True)
    simulate.add_argument("--workers", type=int, default=4)

    args = parser.parse_args()
    if args.command == "baseline":
        from .workloads import gpu_baseline

        result = gpu_baseline(args.model, args.dataset, args.index, args.repeats)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    elif args.command == "capture":
        from .capture import capture as capture_workload

        capture_workload(
            args.model,
            args.dataset,
            args.index,
            args.output,
            query_fp16=args.query_fp16,
            query_transposed=args.query_transposed,
            output_fp16=args.output_fp16,
            query_partitioned=args.query_partitioned,
            output_partitioned=args.output_partitioned,
        )
    else:
        if args.workers < 1:
            raise ValueError("--workers must be positive")
        args.output.mkdir(parents=True, exist_ok=True)
        if args.workers == 1:
            from .experiment import run_case

            result = run_case(
                args.model,
                args.dataset,
                args.config,
                args.raw,
                args.output,
                args.reference,
                args.training_raw,
            )
        else:
            from .experiment import run_case_parallel

            result = run_case_parallel(
                args.model,
                args.dataset,
                args.config,
                args.raw,
                args.output,
                args.reference,
                args.training_raw,
                args.workers,
            )
        print(result)


if __name__ == "__main__":
    main()
