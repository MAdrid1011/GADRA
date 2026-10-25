#!/usr/bin/env python3
"""Reproduce SceneSplat on three real datasets with one 64x64 array."""

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gadra.experiment import run_case, run_case_parallel
from gadra.memory_core import COMPILED


DATASETS = ("scannet", "scannetpp", "matterport3d")
DISPLAY_NAMES = {
    "scannet": "ScanNet",
    "scannetpp": "ScanNet++",
    "matterport3d": "Matterport3D",
}
VARIANTS = ("none", "A", "B", "AB")


def copy_reference(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def summarize(dataset, output, record, base_record):
    variants = record["variants"]
    base_result = base_record["variants"]["none"]
    base_ms = base_result["ms"]
    same_resource_ms = variants["none"]["ms"]
    clock_hz = record["config"]["clock_hz"]

    def resource_floor_ms(result, config):
        statistics = result["statistics"]
        memory = result["memory_statistics"]
        components = {
            "unpadded_arithmetic": result["optimistic_arithmetic_floor_ms"],
            "finite_array": (
                statistics["array_occupied_cycles"]
                / (config["arrays"] * config["clock_hz"])
                * 1000
            ),
            "vector": statistics["vector_cycles"] / config["clock_hz"] * 1000,
            "dram": (
                (memory["dram_read_bytes"] + memory["dram_write_bytes"])
                / config["dram_bandwidth_bytes_per_second"]
                * 1000
            ),
        }
        return max(components.values()), components

    summarized = {}
    for name in VARIANTS:
        result = variants[name]
        floor_ms = result["optimistic_arithmetic_floor_ms"]
        resource_floor, resource_components = resource_floor_ms(
            result, record["config"]
        )
        summarized[name] = {
            "ms": result["ms"],
            "speedup_vs_original_base": base_ms / result["ms"],
            "speedup_vs_same_resource_none": same_resource_ms / result["ms"],
            "speedup_vs_orin_projection": (
                record["baseline"]["orin_reference_ms"] / result["ms"]
            ),
            "speedup_vs_ideal_orin_roof": (
                record["baseline"]["ideal_orin_roof_ms"] / result["ms"]
            ),
            "optimistic_arithmetic_floor_ms": floor_ms,
            "theoretical_speedup_ceiling_vs_original_base": base_ms / floor_ms,
            "fraction_of_theoretical_ceiling": floor_ms / result["ms"],
            "optimistic_resource_floor_ms": resource_floor,
            "resource_speedup_ceiling_vs_original_base": base_ms / resource_floor,
            "fraction_of_resource_ceiling": resource_floor / result["ms"],
            "resource_floor_components_ms": resource_components,
            "finite_array_occupied_ms": (
                result["statistics"]["array_occupied_cycles"]
                / (record["config"]["arrays"] * clock_hz)
                * 1000
            ),
            "finite_array_occupancy": result["resource_activity"][
                "array_issue_fraction"
            ],
        }

    base_resource_floor, base_resource_components = resource_floor_ms(
        base_result, base_record["config"]
    )
    summary = {
        "model": "SceneSplat",
        "dataset": dataset,
        "scope": record["baseline"]["scope"],
        "hardware": {
            key: record["config"][key]
            for key in (
                "clock_hz",
                "array_rows",
                "array_cols",
                "arrays",
                "fp32_mac_cycles",
                "feature_sram_bytes",
                "weight_sram_bytes",
                "output_sram_bytes",
                "control_sram_bytes",
                "dram_bandwidth_bytes_per_second",
                "bank_bytes_per_cycle",
                "cam_lookups_per_cycle",
                "queue_entries",
            )
        },
        "original_base_ms": base_ms,
        "original_base_resource_floor_ms": base_resource_floor,
        "original_base_fraction_of_resource_ceiling": base_resource_floor / base_ms,
        "original_base_resource_floor_components_ms": base_resource_components,
        "same_resource_none_ms": same_resource_ms,
        "variants": summarized,
        "quality": record["quality"],
        "pruning": record["pruning"],
        "orin_projection": record["baseline"],
        "config": record["config"],
    }
    product = (
        summarized["A"]["speedup_vs_original_base"]
        * summarized["B"]["speedup_vs_original_base"]
    )
    resource_ceiling = summarized["AB"][
        "resource_speedup_ceiling_vs_original_base"
    ]
    attainable = min(product, resource_ceiling)
    ab_speedup = summarized["AB"]["speedup_vs_original_base"]
    incremental_a = summarized["A"]["speedup_vs_same_resource_none"]
    incremental_b = summarized["B"]["speedup_vs_same_resource_none"]
    incremental_ab = summarized["AB"]["speedup_vs_same_resource_none"]
    incremental_product = incremental_a * incremental_b
    incremental_resource_ceiling = (
        same_resource_ms / summarized["AB"]["optimistic_resource_floor_ms"]
    )
    incremental_attainable = min(incremental_product, incremental_resource_ceiling)
    summary["composition"] = {
        "raw_A_times_B_speedup": product,
        "raw_product_target_ms": base_ms / product,
        "AB_speedup": ab_speedup,
        "AB_fraction_of_raw_product": ab_speedup / product,
        "AB_resource_speedup_ceiling": resource_ceiling,
        "attainable_target_speedup": attainable,
        "AB_fraction_of_attainable_target": ab_speedup / attainable,
        "raw_product_exceeds_resource_ceiling": product > resource_ceiling,
        "same_resource_A_speedup": incremental_a,
        "same_resource_B_speedup": incremental_b,
        "same_resource_AB_speedup": incremental_ab,
        "same_resource_A_times_B_speedup": incremental_product,
        "same_resource_product_target_ms": same_resource_ms / incremental_product,
        "AB_fraction_of_same_resource_product": incremental_ab / incremental_product,
        "same_resource_AB_resource_speedup_ceiling": incremental_resource_ceiling,
        "same_resource_attainable_target_speedup": incremental_attainable,
        "AB_fraction_of_same_resource_attainable_target": (
            incremental_ab / incremental_attainable
        ),
        "scope": (
            "A-times-B is a diagnostic target, not a physical ceiling: A and B "
            "share the finite scheduling path. The attainable target clips it "
            "to AB's optimistic finite-array/vector/DRAM service ceiling."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_readme(output, summary, base_result)
    return summary


def write_readme(output, summary, base_result):
    dataset = summary["dataset"]
    variants = summary["variants"]
    base_floor = base_result["optimistic_arithmetic_floor_ms"]
    base_occupancy = base_result["resource_activity"]["array_issue_fraction"]
    rows = [
        (
            "original base",
            summary["original_base_ms"],
            1.0,
            None,
            base_floor,
            base_floor / summary["original_base_ms"],
            summary["original_base_resource_floor_ms"],
            summary["original_base_fraction_of_resource_ceiling"],
            base_occupancy,
        )
    ]
    for name in VARIANTS:
        result = variants[name]
        rows.append(
            (
                name,
                result["ms"],
                result["speedup_vs_original_base"],
                result["speedup_vs_same_resource_none"],
                result["optimistic_arithmetic_floor_ms"],
                result["fraction_of_theoretical_ceiling"],
                result["optimistic_resource_floor_ms"],
                result["fraction_of_resource_ceiling"],
                result["finite_array_occupancy"],
            )
        )

    table = [
        "| Variant | Latency | vs original base | vs same-resource none | "
        "Arithmetic floor | Floor/latency | Resource floor | Floor/latency | Array occupancy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (name, latency, base_speedup, none_speedup, floor, efficiency,
         resource_floor, resource_efficiency, occupancy) in rows:
        none_cell = "-" if none_speedup is None else f"{none_speedup:.6f}x"
        table.append(
            f"| {name} | {latency:.6f} ms | {base_speedup:.6f}x | "
            f"{none_cell} | {floor:.6f} ms | {efficiency:.2%} | "
            f"{resource_floor:.6f} ms | {resource_efficiency:.2%} | {occupancy:.2%} |"
        )

    quality = summary["quality"]
    pruning = summary["pruning"]
    projection = summary["orin_projection"]
    composition = summary["composition"]
    config = summary["config"]
    pe_count = config["arrays"] * config["array_rows"] * config["array_cols"]
    fp32_peak = (
        2 * pe_count * config["clock_hz"] / config["fp32_mac_cycles"] / 1e12
    )
    orin_peak = config["baseline_orin_fp32_flops"] / 1e12
    if composition["raw_product_exceeds_resource_ceiling"]:
        composition_note = (
            f"Because the product exceeds the {composition['AB_resource_speedup_ceiling']:.6f}x "
            "optimistic resource ceiling, the physically attainable target is clipped to that "
            f"ceiling; AB reaches {composition['AB_fraction_of_attainable_target']:.2%} of it."
        )
    else:
        composition_note = (
            f"The product is below the {composition['AB_resource_speedup_ceiling']:.6f}x "
            "optimistic resource ceiling, so it remains the attainable target; AB reaches "
            f"{composition['AB_fraction_of_attainable_target']:.2%} of it."
        )
    lines = [
        f"# SceneSplat / {DISPLAY_NAMES[dataset]}",
        "",
        "This directory contains the first real spatial fragment captured from the "
        f"official SceneSplat model on {DISPLAY_NAMES[dataset]}. The scope is complete "
        "3D semantic encoding and excludes disk and host input preprocessing.",
        "",
        f"The fixed target is one FP32-accumulating `64x64` array at "
        f"{config['clock_hz'] / 1e9:g} GHz with a {config['fp32_mac_cycles']}-cycle "
        f"FP32 MAC ({fp32_peak:.3f} TFLOP/s, below the {orin_peak:.3f} TFLOP/s "
        "AGX Orin dense-FP32 reference) and "
        "16/8/4/0.5 MiB feature/weight/output/control SRAM and 204.8 GB/s DRAM. "
        "All four final variants use identical capacities, ports, and peak throughput. "
        "The original base retains the pre-adaptation bank, CAM, queue, and copy-register "
        "configuration; `none` is the fair same-resource control with A and B disabled.",
        "",
        *table,
        "",
        f"A exits {quality['exited_fraction']:.2%} of Gaussians at depth 4, removes "
        f"{pruning['removed_fraction']:.2%} of reference matrix FLOPs, reaches "
        f"{quality['class_agreement']:.4%} class agreement, and keeps "
        f"{quality['within_tolerance']:.4%} of final-score regrets at or below 0.01.",
        "",
        "The arithmetic floor assumes perfect unpadded PE use and excludes data supply, "
        "dependency, and control delay. The resource floor additionally includes finite-array "
        "padding and takes the maximum of measured array, vector, and DRAM service demand under "
        "perfect global overlap. Both are optimistic lower bounds.",
        "",
        f"The raw independent product target is {composition['raw_A_times_B_speedup']:.6f}x, "
        f"while AB reaches {composition['AB_fraction_of_raw_product']:.2%} of that product. "
        + composition_note,
        "",
        f"After removing the shared base-to-adapted execution gain, the same-resource "
        f"A-times-B target is {composition['same_resource_A_times_B_speedup']:.6f}x; "
        f"AB reaches {composition['AB_fraction_of_same_resource_product']:.2%} of it and "
        f"{composition['AB_fraction_of_same_resource_attainable_target']:.2%} of the "
        "resource-clipped same-resource target.",
        "",
        f"The RTX measurement is {projection['rtx_measured_ms']:.6f} ms. The AGX Orin "
        f"center projection is {projection['orin_reference_ms']:.6f} ms with a "
        f"{projection['orin_ms_range'][0]:.6f}--{projection['orin_ms_range'][1]:.6f} ms "
        "efficiency range; these are projections rather than Orin measurements.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        f"python scripts/reproduce_scenesplat.py --dataset {dataset} --workers 4",
        "```",
        "",
        "The command runs the original base and independent `none/A/B/AB` replays, then "
        "checks the fixed quality gates. Full cycles, traffic, control costs, and resource "
        f"activity are in `SceneSplat_{dataset}_s0.json` and "
        f"`base/SceneSplat_{dataset}_s0.json`.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def run_dataset(args, dataset):
    tag = f"SceneSplat_{dataset}_s0"
    output = args.output or ROOT / "results" / "scenesplat" / dataset
    gpu_reference = args.gpu_reference or args.raw / f"{tag}_gpu.json"
    output.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing:
        result_path = output / f"{tag}.json"
        base_path = output / "base" / f"{tag}.json"
    else:
        copy_reference(gpu_reference, output / f"{tag}_gpu.json")
        base_output = output / "base"
        copy_reference(gpu_reference, base_output / f"{tag}_gpu.json")
        base_path = base_output / f"{tag}.json"
        if not args.reuse_base:
            base_path = Path(run_case(
                "SceneSplat",
                dataset,
                args.base_config,
                args.raw,
                base_output,
                training_raw=args.raw,
                variant_names=("none",),
            ))
        result_path = Path(run_case_parallel(
            "SceneSplat",
            dataset,
            args.config,
            args.raw,
            output,
            training_raw=args.raw,
            workers=args.workers,
        ))

    missing = [str(path) for path in (result_path, base_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing completed replay files:\n" + "\n".join(missing))
    record = json.loads(Path(result_path).read_text())
    base_record = json.loads(Path(base_path).read_text())
    if record.get("model") != "SceneSplat" or record.get("dataset") != dataset:
        raise ValueError(f"Replay record does not match SceneSplat/{dataset}")
    if not record["quality"]["passes_thresholds"]:
        raise RuntimeError("The reproduced A/AB result failed the fixed quality gates")
    summary = summarize(dataset, output, record, base_record)

    print(f"\nSceneSplat / {DISPLAY_NAMES[dataset]}")
    print("variant       ms   vs same none   vs original base   ideal ceiling")
    for name in VARIANTS:
        result = summary["variants"][name]
        print(
            f"{name:>7} {result['ms']:9.6f} "
            f"{result['speedup_vs_same_resource_none']:14.6f}x "
            f"{result['speedup_vs_original_base']:17.6f}x "
            f"{result['theoretical_speedup_ceiling_vs_original_base']:14.6f}x"
        )
    quality = summary["quality"]
    print(f"class agreement: {quality['class_agreement']:.6%}")
    print(f"regret <= 0.01:  {quality['within_tolerance']:.6%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "local/captures/scenesplat"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/scenesplat_scannet.json"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/scenesplat_single64_base.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu-reference", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="regenerate summaries from completed real replay records",
    )
    parser.add_argument(
        "--reuse-base",
        action="store_true",
        help="reuse the completed original-base record in the output directory",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.dataset == "all" and (args.output or args.gpu_reference):
        raise ValueError("--output and --gpu-reference require one explicit dataset")
    if args.reuse_existing and args.reuse_base:
        raise ValueError("--reuse-existing and --reuse-base are mutually exclusive")
    if not args.reuse_existing and not COMPILED:
        raise RuntimeError(
            "SceneSplat replay requires the Numba-compiled simulator; run this script "
            "with the repository's system Python environment."
        )

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    required = [
        args.raw / f"SceneSplat_{dataset}_s0{suffix}"
        for dataset in datasets
        for suffix in (
            ".json",
            "_maps.pkl",
            "_observations.npz",
            "_coverage.npz",
            "_coverage.json",
        )
    ]
    if not args.reuse_existing:
        required.extend(
            (args.gpu_reference or args.raw / f"SceneSplat_{dataset}_s0_gpu.json")
            for dataset in datasets
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing real capture files:\n" + "\n".join(missing))

    for dataset in datasets:
        run_dataset(args, dataset)


if __name__ == "__main__":
    main()
