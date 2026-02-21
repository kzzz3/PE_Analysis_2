from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shared_strength_alignment import (
    MetricCurve,
    align_curves,
    build_rho_grid,
    calculate_bpp,
    calculate_psnr_ssim,
    default_color_input_root,
    project_root,
)


@dataclass
class SamplePoint:
    rho: float
    bpp: float
    psnr: float
    ssim: float


def parse_int_suffix(name: str, prefix: str) -> int | None:
    if not name.startswith(prefix):
        return None
    try:
        return int(name.split("=", 1)[1])
    except (IndexError, ValueError):
        return None


def points_to_curve(image_name: str, points: list[SamplePoint], metric: str) -> MetricCurve | None:
    if not points:
        return None
    points = sorted(points, key=lambda p: p.rho)

    rho_values: list[float] = []
    metric_values: list[float] = []
    for point in points:
        if rho_values and abs(point.rho - rho_values[-1]) < 1e-9:
            metric_values[-1] = getattr(point, metric)
        else:
            rho_values.append(point.rho)
            metric_values.append(getattr(point, metric))

    if len(rho_values) == 1:
        rho_values = [0.0, 1.0]
        metric_values = [metric_values[0], metric_values[0]]

    rho = np.array(rho_values, dtype=np.float64)
    values = np.array(metric_values, dtype=np.float64)
    strengths = np.array(rho_values, dtype=np.float64)
    return MetricCurve(image_name=image_name, strengths=strengths, rhos=rho, values=values)


def rows_from_stage_curves(curves: dict[str, dict[str, MetricCurve]], rho_step: float, min_coverage: float) -> list[dict]:
    if not curves["bpp"] or not curves["psnr"] or not curves["ssim"]:
        return []

    rho_grid = build_rho_grid(rho_step)
    bpp_stats = align_curves(curves["bpp"], rho_grid, min_coverage=min_coverage)
    psnr_stats = align_curves(curves["psnr"], rho_grid, min_coverage=min_coverage)
    ssim_stats = align_curves(curves["ssim"], rho_grid, min_coverage=min_coverage)

    bpp_map = {
        round(float(rho), 6): (float(mean), float(std), float(cov), int(n))
        for rho, mean, std, cov, n in zip(
            bpp_stats.rho,
            bpp_stats.mean,
            bpp_stats.std,
            bpp_stats.coverage,
            bpp_stats.sample_count,
        )
    }
    psnr_map = {
        round(float(rho), 6): (float(mean), float(std), float(cov), int(n))
        for rho, mean, std, cov, n in zip(
            psnr_stats.rho,
            psnr_stats.mean,
            psnr_stats.std,
            psnr_stats.coverage,
            psnr_stats.sample_count,
        )
    }
    ssim_map = {
        round(float(rho), 6): (float(mean), float(std), float(cov), int(n))
        for rho, mean, std, cov, n in zip(
            ssim_stats.rho,
            ssim_stats.mean,
            ssim_stats.std,
            ssim_stats.coverage,
            ssim_stats.sample_count,
        )
    }

    rows: list[dict] = []
    for rho in sorted(set(bpp_map.keys()) & set(psnr_map.keys()) & set(ssim_map.keys())):
        bpp_mean, bpp_std, bpp_cov, bpp_n = bpp_map[rho]
        psnr_mean, psnr_std, psnr_cov, psnr_n = psnr_map[rho]
        ssim_mean, ssim_std, ssim_cov, ssim_n = ssim_map[rho]
        rows.append(
            {
                "rho": rho,
                "mean_bpp": bpp_mean,
                "std_bpp": bpp_std,
                "mean_psnr": psnr_mean,
                "std_psnr": psnr_std,
                "mean_ssim": ssim_mean,
                "std_ssim": ssim_std,
                "coverage_bpp": bpp_cov,
                "coverage_psnr": psnr_cov,
                "coverage_ssim": ssim_cov,
                "n": int(min(bpp_n, psnr_n, ssim_n)),
            }
        )
    return rows


def write_stage_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "rho",
                "mean_bpp",
                "std_bpp",
                "mean_psnr",
                "std_psnr",
                "mean_ssim",
                "std_ssim",
                "coverage_bpp",
                "coverage_psnr",
                "coverage_ssim",
                "n",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row['rho']:.6f}",
                    f"{row['mean_bpp']:.8f}",
                    f"{row['std_bpp']:.8f}",
                    f"{row['mean_psnr']:.8f}",
                    f"{row['std_psnr']:.8f}",
                    f"{row['mean_ssim']:.8f}",
                    f"{row['std_ssim']:.8f}",
                    f"{row['coverage_bpp']:.4f}",
                    f"{row['coverage_psnr']:.4f}",
                    f"{row['coverage_ssim']:.4f}",
                    row["n"],
                ]
            )


def save_psnr_bpp_plot(path: Path, title: str, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    bpp = [row["mean_bpp"] for row in rows]
    psnr = [row["mean_psnr"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(bpp, psnr, marker="o", linewidth=1.5)
    plt.xlabel("BPP (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def write_final_csv(path: Path, points: list[SamplePoint], total_images: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bpp = np.array([p.bpp for p in points], dtype=np.float64)
    psnr = np.array([p.psnr for p in points], dtype=np.float64)
    ssim = np.array([p.ssim for p in points], dtype=np.float64)

    n = int(len(points))
    coverage = (n / float(total_images)) if total_images > 0 else 0.0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mean_bpp", "std_bpp", "mean_psnr", "std_psnr", "mean_ssim", "std_ssim", "coverage", "n"])
        writer.writerow(
            [
                f"{float(np.nanmean(bpp)):.8f}" if n > 0 else "nan",
                f"{float(np.nanstd(bpp)):.8f}" if n > 0 else "nan",
                f"{float(np.nanmean(psnr)):.8f}" if n > 0 else "nan",
                f"{float(np.nanstd(psnr)):.8f}" if n > 0 else "nan",
                f"{float(np.nanmean(ssim)):.8f}" if n > 0 else "nan",
                f"{float(np.nanstd(ssim)):.8f}" if n > 0 else "nan",
                f"{coverage:.4f}",
                n,
            ]
        )


def process_one_setting(
    qf_dir: Path,
    subsampling: int,
    qf: int,
    input_root: Path,
    tables_dir: Path,
    figures_dir: Path,
    rho_step: float,
    min_coverage: float,
) -> dict:
    ac_curves = {"bpp": {}, "psnr": {}, "ssim": {}}
    dc_curves = {"bpp": {}, "psnr": {}, "ssim": {}}
    final_points: list[SamplePoint] = []
    caps_rows: list[dict] = []

    image_dirs = [p for p in qf_dir.iterdir() if p.is_dir()]
    for image_dir in sorted(image_dirs, key=lambda p: p.name.lower()):
        source_path = input_root / image_dir.name
        meta_path = image_dir / "meta.csv"
        if not source_path.exists() or not meta_path.exists():
            continue

        ac_points: list[SamplePoint] = []
        dc_points: list[SamplePoint] = []
        image_final_point: SamplePoint | None = None
        image_caps_written = False

        with meta_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                output_path = image_dir / row["file_name"]
                if not output_path.exists():
                    continue

                bpp = calculate_bpp(output_path)
                psnr, ssim = calculate_psnr_ssim(source_path, output_path)
                point = SamplePoint(rho=float(row["rho_stage"]), bpp=bpp, psnr=psnr, ssim=ssim)

                stage = row["stage"]
                if stage == "ac":
                    ac_points.append(point)
                elif stage == "dc_progress":
                    dc_points.append(point)
                elif stage == "dc_final":
                    image_final_point = point

                if not image_caps_written:
                    caps_rows.append(
                        {
                            "image_name": image_dir.name,
                            "b_y": int(row["b_y"]),
                            "f_y": int(row["f_y"]),
                            "b_cb": int(row["b_cb"]),
                            "f_cb": int(row["f_cb"]),
                            "b_cr": int(row["b_cr"]),
                            "f_cr": int(row["f_cr"]),
                        }
                    )
                    image_caps_written = True

        ac_curve_bpp = points_to_curve(image_dir.name, ac_points, "bpp")
        ac_curve_psnr = points_to_curve(image_dir.name, ac_points, "psnr")
        ac_curve_ssim = points_to_curve(image_dir.name, ac_points, "ssim")
        if ac_curve_bpp and ac_curve_psnr and ac_curve_ssim:
            ac_curves["bpp"][image_dir.name] = ac_curve_bpp
            ac_curves["psnr"][image_dir.name] = ac_curve_psnr
            ac_curves["ssim"][image_dir.name] = ac_curve_ssim

        dc_curve_bpp = points_to_curve(image_dir.name, dc_points, "bpp")
        dc_curve_psnr = points_to_curve(image_dir.name, dc_points, "psnr")
        dc_curve_ssim = points_to_curve(image_dir.name, dc_points, "ssim")
        if dc_curve_bpp and dc_curve_psnr and dc_curve_ssim:
            dc_curves["bpp"][image_dir.name] = dc_curve_bpp
            dc_curves["psnr"][image_dir.name] = dc_curve_psnr
            dc_curves["ssim"][image_dir.name] = dc_curve_ssim

        if image_final_point is not None:
            final_points.append(image_final_point)

    ac_rows = rows_from_stage_curves(ac_curves, rho_step=rho_step, min_coverage=min_coverage)
    dc_rows = rows_from_stage_curves(dc_curves, rho_step=rho_step, min_coverage=min_coverage)

    write_stage_csv(tables_dir / f"consistency_ac_ss{subsampling}_qf{qf}.csv", ac_rows)
    write_stage_csv(tables_dir / f"consistency_dc_ss{subsampling}_qf{qf}.csv", dc_rows)
    write_final_csv(tables_dir / f"consistency_final_ss{subsampling}_qf{qf}.csv", final_points, total_images=len(image_dirs))

    save_psnr_bpp_plot(
        figures_dir / f"consistency_ac_ss{subsampling}_qf{qf}_psnr_bpp.png",
        f"Consistency AC stage PSNR-BPP (QF={qf}, subsampling={subsampling})",
        ac_rows,
    )
    save_psnr_bpp_plot(
        figures_dir / f"consistency_dc_ss{subsampling}_qf{qf}_psnr_bpp.png",
        f"Consistency DC stage PSNR-BPP (QF={qf}, subsampling={subsampling})",
        dc_rows,
    )

    caps_path = tables_dir / f"consistency_caps_ss{subsampling}_qf{qf}.csv"
    caps_path.parent.mkdir(parents=True, exist_ok=True)
    with caps_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "b_y", "f_y", "b_cb", "f_cb", "b_cr", "f_cr"])
        for row in sorted(caps_rows, key=lambda x: x["image_name"].lower()):
            writer.writerow([row["image_name"], row["b_y"], row["f_y"], row["b_cb"], row["f_cb"], row["b_cr"], row["f_cr"]])

    return {
        "subsampling": subsampling,
        "qf": qf,
        "ac_points": len(ac_rows),
        "dc_points": len(dc_rows),
        "final_n": len(final_points),
        "images": len(image_dirs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze color consistency-control outputs.")
    parser.add_argument(
        "--consistency-root",
        type=Path,
        default=project_root() / "Result" / "OutputImage" / "Color" / "Consistency",
        help="Consistency output root (contains Subsampling=*/QF=*).",
    )
    parser.add_argument("--input-root", type=Path, default=default_color_input_root())
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=project_root() / "Result" / "AnalysisColorConsistency",
    )
    parser.add_argument("--rho-step", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.7)
    args = parser.parse_args()

    if not args.consistency_root.exists():
        raise FileNotFoundError(f"Consistency root does not exist: {args.consistency_root}")
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    tables_dir = args.analysis_root / "tables"
    figures_dir = args.analysis_root / "figures"
    rows_summary: list[dict] = []

    for sub_dir in sorted(args.consistency_root.iterdir(), key=lambda p: p.name):
        if not sub_dir.is_dir():
            continue
        subsampling = parse_int_suffix(sub_dir.name, "Subsampling=")
        if subsampling is None:
            continue

        for qf_dir in sorted(sub_dir.iterdir(), key=lambda p: p.name):
            if not qf_dir.is_dir():
                continue
            qf = parse_int_suffix(qf_dir.name, "QF=")
            if qf is None:
                continue

            summary = process_one_setting(
                qf_dir=qf_dir,
                subsampling=subsampling,
                qf=qf,
                input_root=args.input_root,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                rho_step=args.rho_step,
                min_coverage=args.min_coverage,
            )
            rows_summary.append(summary)
            print(
                f"Done subsampling={subsampling} qf={qf}: "
                f"ac_points={summary['ac_points']} dc_points={summary['dc_points']} "
                f"final_n={summary['final_n']}/{summary['images']}"
            )

    summary_path = args.analysis_root / "consistency_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["subsampling", "qf", "ac_points", "dc_points", "final_n", "images"])
        for row in sorted(rows_summary, key=lambda x: (x["subsampling"], x["qf"])):
            writer.writerow([row["subsampling"], row["qf"], row["ac_points"], row["dc_points"], row["final_n"], row["images"]])
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
