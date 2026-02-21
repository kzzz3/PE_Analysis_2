from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

from shared_strength_alignment import default_color_input_root, project_root


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if token:
            items.append(int(token))
    return items


def parse_float_list(value: str) -> list[float]:
    items: list[float] = []
    for token in value.split(","):
        token = token.strip()
        if token:
            items.append(float(token))
    return items


def run_command(cmd: list[str], cwd: Path, timeout_sec: int) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True, timeout=timeout_sec)


def n_output_root(run_root: Path, n_level: int) -> Path:
    return run_root / f"n{n_level}" / "OutputImage" / "Color"


def n_metric_root(run_root: Path, n_level: int) -> Path:
    return run_root / f"n{n_level}" / "AnalysisQualityCompression"


def run_generation_for_n(args, n_level: int) -> None:
    output_root = n_output_root(args.run_root, n_level)
    output_root.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(args.pe_exe),
        "--mode",
        "both",
        "--threads",
        str(args.threads),
        "--input",
        str(args.input_root),
        "--output",
        str(output_root),
        "--qf",
        args.qfs,
        "--ac-st",
        args.ac_st,
        "--subsampling",
        args.subsampling,
        "--dc-max",
        str(args.dc_max),
        "--iwt-level",
        str(n_level),
        "--verify-reversible",
        "0",
    ]
    log_info(f"Run generation for n={n_level}")
    run_command(cmd, args.pe_workdir, args.timeout_sec)


def run_metrics_for_n(args, n_level: int) -> None:
    output_root = n_output_root(args.run_root, n_level)
    metric_root = n_metric_root(args.run_root, n_level)
    metric_root.mkdir(parents=True, exist_ok=True)

    ac_root = output_root / "AcEncryption"
    dc_root = output_root / "DcEncryption"

    cmd = [
        sys.executable,
        str(args.metric_script),
        "--datasets",
        "color",
        "--modes",
        "both",
        "--subsampling",
        args.subsampling,
        "--qfs",
        args.qfs,
        "--workers",
        str(args.workers),
        "--color-input-root",
        str(args.input_root),
        "--color-ac-root",
        str(ac_root),
        "--color-dc-root",
        str(dc_root),
        "--analysis-root",
        str(metric_root),
    ]

    if args.images.strip():
        cmd.extend(["--images", args.images])

    log_info(f"Run metric aggregation for n={n_level}")
    run_command(cmd, Path(__file__).resolve().parent, args.timeout_sec)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def pick_nearest_rho_rows(rows: list[dict[str, str]], rho_targets: list[float]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            int(row["n_level"]),
            row["dataset"],
            row["subsampling"],
            row["mode"],
            int(row["qf"]),
        )
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, Any]] = []
    for key, group in grouped.items():
        for target in rho_targets:
            nearest = min(group, key=lambda r: abs(float(r["rho"]) - target))
            selected.append(
                {
                    "n_level": key[0],
                    "dataset": key[1],
                    "subsampling": key[2],
                    "mode": key[3],
                    "qf": key[4],
                    "rho_target": f"{target:.4f}",
                    "rho_selected": nearest["rho"],
                    "mean_bpp": nearest.get("mean_bpp", ""),
                    "mean_psnr": nearest.get("mean_psnr", ""),
                    "mean_ssim": nearest.get("mean_ssim", ""),
                    "coverage_bpp": nearest.get("coverage_bpp", ""),
                    "coverage_psnr": nearest.get("coverage_psnr", ""),
                    "coverage_ssim": nearest.get("coverage_ssim", ""),
                    "n": nearest.get("n", ""),
                }
            )
    return sorted(
        selected,
        key=lambda r: (
            int(r["n_level"]),
            str(r["dataset"]),
            str(r["subsampling"]),
            str(r["mode"]),
            int(r["qf"]),
            float(r["rho_target"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def collect_and_merge(args, n_levels: list[int]) -> None:
    merged_all: list[dict[str, Any]] = []
    merged_endpoint: list[dict[str, Any]] = []

    for n_level in n_levels:
        metric_root = n_metric_root(args.run_root, n_level)
        all_path = metric_root / "all_rho_summary.csv"
        endpoint_path = metric_root / "endpoint_summary.csv"

        all_rows = read_csv_rows(all_path)
        endpoint_rows = read_csv_rows(endpoint_path)

        if not all_rows:
            log_warn(f"Missing all_rho_summary.csv for n={n_level}: {all_path}")
        if not endpoint_rows:
            log_warn(f"Missing endpoint_summary.csv for n={n_level}: {endpoint_path}")

        for row in all_rows:
            merged = dict(row)
            merged["n_level"] = str(n_level)
            merged_all.append(merged)

        for row in endpoint_rows:
            merged = dict(row)
            merged["n_level"] = str(n_level)
            merged_endpoint.append(merged)

    merged_all = sorted(
        merged_all,
        key=lambda r: (
            int(r["n_level"]),
            str(r.get("dataset", "")),
            str(r.get("subsampling", "")),
            str(r.get("mode", "")),
            int(r.get("qf", 0)),
            float(r.get("rho", 0.0)),
        ),
    )

    merged_endpoint = sorted(
        merged_endpoint,
        key=lambda r: (
            int(r["n_level"]),
            str(r.get("dataset", "")),
            str(r.get("subsampling", "")),
            str(r.get("mode", "")),
            int(r.get("qf", 0)),
            float(r.get("rho", 0.0)),
        ),
    )

    summary_root = args.run_root / "summary"
    write_csv(
        summary_root / "n_ablation_all_rho.csv",
        merged_all,
        headers=[
            "n_level",
            "variant",
            "dataset",
            "subsampling",
            "mode",
            "qf",
            "rho",
            "mean_bpp",
            "mean_psnr",
            "mean_ssim",
            "coverage_bpp",
            "coverage_psnr",
            "coverage_ssim",
            "n",
        ],
    )
    write_csv(
        summary_root / "n_ablation_endpoint.csv",
        merged_endpoint,
        headers=[
            "n_level",
            "variant",
            "dataset",
            "subsampling",
            "mode",
            "qf",
            "rho",
            "mean_bpp",
            "mean_psnr",
            "mean_ssim",
            "n",
        ],
    )

    rho_targets = parse_float_list(args.rho_targets)
    sampled_rows = pick_nearest_rho_rows(merged_all, rho_targets)
    write_csv(
        summary_root / "n_ablation_sampled_rho.csv",
        sampled_rows,
        headers=[
            "n_level",
            "dataset",
            "subsampling",
            "mode",
            "qf",
            "rho_target",
            "rho_selected",
            "mean_bpp",
            "mean_psnr",
            "mean_ssim",
            "coverage_bpp",
            "coverage_psnr",
            "coverage_ssim",
            "n",
        ],
    )

    log_info(f"Wrote merged ablation tables under: {summary_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="IWT-level ablation orchestrator (generation + metric merge).")
    parser.add_argument("--n-levels", type=str, default="2,3,4")
    parser.add_argument("--qfs", type=str, default="30,50,70,90")
    parser.add_argument("--subsampling", type=str, default="444,420")
    parser.add_argument("--ac-st", type=str, default="1,2,3,5,7,10,15,20,30,40,50,60,64")
    parser.add_argument("--dc-max", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--images", type=str, default="", help="Optional image filter passed to metric script")
    parser.add_argument("--rho-targets", type=str, default="0.25,0.50,0.75,0.95")
    parser.add_argument("--timeout-sec", type=int, default=14400)

    parser.add_argument("--run-generation", action="store_true", help="Run PE generation for each n")
    parser.add_argument("--skip-metric-run", action="store_true", help="Skip metric_quality_compression.py execution")

    parser.add_argument("--pe-workdir", type=Path, default=project_root() / "PE")
    parser.add_argument("--pe-exe", type=Path, default=project_root() / "PE" / "build" / "global-x64" / "Release" / "PE.exe")
    parser.add_argument("--metric-script", type=Path, default=Path(__file__).with_name("metric_quality_compression.py"))
    parser.add_argument("--input-root", type=Path, default=default_color_input_root())
    parser.add_argument("--run-root", type=Path, default=project_root() / "Result" / "Ablation" / "IWTLevel")
    args = parser.parse_args()

    n_levels = parse_int_list(args.n_levels)
    if not n_levels:
        raise ValueError("No n-level is provided")

    if args.run_generation:
        if not args.pe_exe.exists():
            raise FileNotFoundError(f"PE executable not found: {args.pe_exe}")
        for n_level in n_levels:
            run_generation_for_n(args, n_level)

    if not args.skip_metric_run:
        if not args.metric_script.exists():
            raise FileNotFoundError(f"Metric script not found: {args.metric_script}")
        for n_level in n_levels:
            run_metrics_for_n(args, n_level)

    collect_and_merge(args, n_levels)


if __name__ == "__main__":
    main()
