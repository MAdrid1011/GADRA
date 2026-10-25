#!/usr/bin/env python3
"""Reproduce Chorus on three real datasets with 4096 FP32 PEs."""

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from garda.experiment import run_case, run_case_parallel
from garda.memory_core import COMPILED


VARIANTS = ("none", "A", "B", "AB")
DATASETS = ("scannet", "scannetpp", "matterport3d")
DISPLAY_NAMES = {
    "scannet": "ScanNet",
    "scannetpp": "ScanNet++",
    "matterport3d": "Matterport3D",
}


def copy_reference(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)


def summarize(dataset, output, record, base_record):
    base = base_record["variants"]["none"]
    same_none = record["variants"]["none"]["ms"]
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

    variants = {}
    for name in VARIANTS:
        result = record["variants"][name]
        floor = result["optimistic_arithmetic_floor_ms"]
        statistics = result["statistics"]
        resource_floor, resource_components = resource_floor_ms(
            result, record["config"]
        )
        variants[name] = {
            "ms": result["ms"],
            "speedup_vs_original_base": base["ms"] / result["ms"],
            "speedup_vs_same_resource_none": same_none / result["ms"],
            "speedup_vs_orin_projection": (
                record["baseline"]["orin_reference_ms"] / result["ms"]
            ),
            "speedup_vs_ideal_orin_roof": (
                record["baseline"]["ideal_orin_roof_ms"] / result["ms"]
            ),
            "optimistic_arithmetic_floor_ms": floor,
            "theoretical_speedup_ceiling_vs_original_base": base["ms"] / floor,
            "fraction_of_theoretical_ceiling": floor / result["ms"],
            "optimistic_resource_floor_ms": resource_floor,
            "resource_speedup_ceiling_vs_original_base": base["ms"] / resource_floor,
            "fraction_of_resource_ceiling": resource_floor / result["ms"],
            "resource_floor_components_ms": resource_components,
            "finite_array_occupied_ms": (
                statistics["array_occupied_cycles"]
                / (record["config"]["arrays"] * clock_hz)
                * 1000
            ),
            "finite_array_occupancy": result["resource_activity"][
                "array_issue_fraction"
            ],
        }
    base_resource_floor, base_resource_components = resource_floor_ms(
        base, base_record["config"]
    )
    summary = {
        "model": "Chorus",
        "dataset": dataset,
        "scope": record["baseline"]["scope"],
        "original_base_ms": base["ms"],
        "original_base_resource_floor_ms": base_resource_floor,
        "original_base_fraction_of_resource_ceiling": base_resource_floor / base["ms"],
        "original_base_resource_floor_components_ms": base_resource_components,
        "same_resource_none_ms": same_none,
        "variants": variants,
        "quality": record["quality"],
        "pruning": record["pruning"],
        "orin_projection": record["baseline"],
        "config": record["config"],
    }
    product = (
        variants["A"]["speedup_vs_original_base"]
        * variants["B"]["speedup_vs_original_base"]
    )
    resource_ceiling = variants["AB"]["resource_speedup_ceiling_vs_original_base"]
    attainable = min(product, resource_ceiling)
    ab_speedup = variants["AB"]["speedup_vs_original_base"]
    incremental_a = variants["A"]["speedup_vs_same_resource_none"]
    incremental_b = variants["B"]["speedup_vs_same_resource_none"]
    incremental_ab = variants["AB"]["speedup_vs_same_resource_none"]
    incremental_product = incremental_a * incremental_b
    incremental_resource_ceiling = (
        same_none / variants["AB"]["optimistic_resource_floor_ms"]
    )
    incremental_attainable = min(incremental_product, incremental_resource_ceiling)
    summary["composition"] = {
        "raw_A_times_B_speedup": product,
        "raw_product_target_ms": base["ms"] / product,
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
        "same_resource_product_target_ms": same_none / incremental_product,
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
    write_readme(dataset, output, summary, base)
    return summary


def write_readme(dataset, output, summary, base):
    rows = [
        (
            "original base",
            summary["original_base_ms"],
            1.0,
            None,
            base["optimistic_arithmetic_floor_ms"],
            base["optimistic_arithmetic_floor_ms"] / summary["original_base_ms"],
            summary["original_base_resource_floor_ms"],
            summary["original_base_fraction_of_resource_ceiling"],
            base["resource_activity"]["array_issue_fraction"],
        )
    ]
    for name in VARIANTS:
        result = summary["variants"][name]
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
        resource_floor_cell = "-" if resource_floor is None else f"{resource_floor:.6f} ms"
        resource_efficiency_cell = "-" if resource_efficiency is None else f"{resource_efficiency:.2%}"
        table.append(
            f"| {name} | {latency:.6f} ms | {base_speedup:.6f}x | "
            f"{none_cell} | {floor:.6f} ms | {efficiency:.2%} | "
            f"{resource_floor_cell} | {resource_efficiency_cell} | {occupancy:.2%} |"
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
        f"# Chorus / {DISPLAY_NAMES[dataset]}",
        "",
        "This result uses the official Chorus checkpoint and the first real "
        f"{DISPLAY_NAMES[dataset]} fragment. The other two datasets provide independent "
        "calibration scenes.",
        "",
        f"Both configurations contain 4096 FP32-accumulating PEs at "
        f"{config['clock_hz'] / 1e9:g} GHz with a {config['fp32_mac_cycles']}-cycle "
        f"FP32 MAC ({fp32_peak:.3f} TFLOP/s, below the {orin_peak:.3f} TFLOP/s "
        "AGX Orin dense-FP32 reference) and share "
        "16/8/4/0.5 MiB feature/weight/output/control SRAM and 204.8 GB/s DRAM. The "
        "original base is one `64x64` array; the optimized path partitions it into "
        "four `32x32` subarrays for fine-grained scheduling.",
        f"The shared finite scheduler uses K stripes of {summary['config']['consumer_k_tile']} "
        f"for A-only, {summary['config']['full_depth_coverage_k_tile']} for B-only, "
        f"and {summary['config']['joint_consumer_k_tile']} for AB under the same fixed SRAM capacity.",
        "",
        *table,
        "",
        f"A exits {quality['exited_fraction']:.2%} of Gaussians and removes "
        f"{pruning['removed_fraction']:.2%} of reference matrix FLOPs. Class agreement "
        f"is {quality['class_agreement']:.4%}; {quality['within_tolerance']:.4%} of "
        "final-query regrets are at most 0.01.",
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
        f"python scripts/reproduce_chorus_scannet.py --dataset {dataset} --workers 4",
        "```",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n")


def run_dataset(args, dataset):
    tag = f"Chorus_{dataset}_s0"
    output = args.output or ROOT / "results" / "chorus" / dataset
    reference = args.gpu_reference or args.raw / f"{tag}_gpu.json"
    output.mkdir(parents=True, exist_ok=True)

    if args.reuse_existing:
        result_path = output / f"{tag}.json"
        base_path = output / "base" / f"{tag}.json"
    else:
        if not reference.exists():
            raise FileNotFoundError(f"Missing real GPU reference: {reference}")
        copy_reference(reference, output / f"{tag}_gpu.json")
        base_output = output / "base"
        copy_reference(reference, base_output / f"{tag}_gpu.json")
        base_path = base_output / f"{tag}.json"
        if not args.reuse_base:
            base_path = Path(
                run_case(
                    "Chorus",
                    dataset,
                    args.base_config,
                    args.raw,
                    base_output,
                    training_raw=args.raw,
                    variant_names=("none",),
                )
            )
        result_path = Path(
            run_case_parallel(
                "Chorus",
                dataset,
                args.config,
                args.raw,
                output,
                training_raw=args.raw,
                workers=args.workers,
            )
        )

    missing = [str(path) for path in (result_path, base_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing completed replay files:\n" + "\n".join(missing))
    record = json.loads(result_path.read_text())
    base_record = json.loads(base_path.read_text())
    if record.get("model") != "Chorus" or record.get("dataset") != dataset:
        raise ValueError(f"Replay record does not match Chorus/{dataset}")
    if not record["quality"]["passes_thresholds"]:
        raise RuntimeError("The reproduced A/AB result failed the fixed quality gates")
    summary = summarize(dataset, output, record, base_record)

    print(f"\nChorus / {DISPLAY_NAMES[dataset]}")
    print("variant       ms   vs same none   vs original base   ideal ceiling")
    for name in VARIANTS:
        result = summary["variants"][name]
        print(
            f"{name:>7} {result['ms']:9.6f} "
            f"{result['speedup_vs_same_resource_none']:14.6f}x "
            f"{result['speedup_vs_original_base']:17.6f}x "
            f"{result['theoretical_speedup_ceiling_vs_original_base']:14.6f}x"
        )
    print(f"class agreement: {summary['quality']['class_agreement']:.6%}")
    print(f"regret <= 0.01:  {summary['quality']['within_tolerance']:.6%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw", type=Path, default=ROOT / "local/captures/chorus"
    )
    parser.add_argument(
        "--output", type=Path
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/chorus_scannet.json"
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=ROOT / "configs/chorus_single64_base.json",
    )
    parser.add_argument("--gpu-reference", type=Path)
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--reuse-base", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.dataset == "all" and (args.output or args.gpu_reference):
        raise ValueError("--output and --gpu-reference require one explicit dataset")
    if args.reuse_existing and args.reuse_base:
        raise ValueError("--reuse-existing and --reuse-base are mutually exclusive")
    if not args.reuse_existing and not COMPILED:
        raise RuntimeError("Chorus replay requires the Numba-compiled simulator")

    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    required = [
        args.raw / f"Chorus_{dataset}_s0{suffix}"
        for dataset in datasets
        for suffix in (
            ".json",
            "_maps.pkl",
            "_observations.npz",
            "_coverage.npz",
            "_coverage.json",
        )
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing real capture files:\n" + "\n".join(missing))
    for dataset in datasets:
        run_dataset(args, dataset)


if __name__ == "__main__":
    main()
