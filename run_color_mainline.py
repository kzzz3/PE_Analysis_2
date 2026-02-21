from __future__ import annotations

import argparse
import csv
from pathlib import Path

from shared_strength_alignment import (
    align_curves,
    build_curves_from_metric,
    build_rho_grid,
    build_source_lookup,
    calculate_bpp,
    calculate_psnr_ssim,
    collect_ac_image_strength_files,
    collect_dc_image_strength_files,
    default_color_input_root,
    default_color_output_root,
    project_root,
)


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def stats_to_map(stats) -> dict[float, tuple[float, float, float, int]]:
    return {
        round(float(rho), 6): (float(mean), float(std), float(cov), int(n))
        for rho, mean, std, cov, n in zip(
            stats.rho,
            stats.mean,
            stats.std,
            stats.coverage,
            stats.sample_count,
        )
    }


def write_aligned_table_csv(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
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


def save_psnr_bpp_plot(plot_path: Path, mode: str, subsampling: int, qf: int, rows: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("Skip plot generation: matplotlib is not installed.")
        return

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    bpp = [row["mean_bpp"] for row in rows]
    psnr = [row["mean_psnr"] for row in rows]

    plt.figure(figsize=(10, 6))
    plt.plot(bpp, psnr, marker="o", linewidth=1.5)
    plt.xlabel("BPP (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"{mode.upper()} color PSNR-BPP (QF={qf}, subsampling={subsampling})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()


def build_rows_for_qf(
    qf: int,
    bpp_stats,
    psnr_stats,
    ssim_stats,
) -> list[dict]:
    bpp_map = stats_to_map(bpp_stats)
    psnr_map = stats_to_map(psnr_stats)
    ssim_map = stats_to_map(ssim_stats)

    common_rho = sorted(set(bpp_map.keys()) & set(psnr_map.keys()) & set(ssim_map.keys()))
    rows: list[dict] = []
    for rho in common_rho:
        bpp_mean, bpp_std, bpp_cov, bpp_n = bpp_map[rho]
        psnr_mean, psnr_std, psnr_cov, psnr_n = psnr_map[rho]
        ssim_mean, ssim_std, ssim_cov, ssim_n = ssim_map[rho]
        rows.append(
            {
                "qf": qf,
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


def run_mode(
    mode: str,
    subsampling: int,
    input_root: Path,
    output_root: Path,
    rho_step: float,
    min_coverage: float,
    tables_dir: Path,
    figures_dir: Path,
    endpoint_rows: list[dict],
) -> None:
    mode_root = output_root / ("AcEncryption" if mode == "ac" else "DcEncryption") / f"Subsampling={subsampling}"
    if not mode_root.exists():
        print(f"Skip {mode} subsampling={subsampling}: missing {mode_root}")
        return

    if mode == "ac":
        files_by_qf = collect_ac_image_strength_files(mode_root, input_root=input_root)
    else:
        files_by_qf = collect_dc_image_strength_files(mode_root)

    if not files_by_qf:
        print(f"Skip {mode} subsampling={subsampling}: no parsed files")
        return

    source_lookup = build_source_lookup(input_root)

    def psnr_metric(path: Path, image_name: str) -> float:
        source = source_lookup.get(image_name.lower())
        if source is None:
            raise FileNotFoundError(f"Missing source image for {image_name}")
        psnr, _ = calculate_psnr_ssim(source, path)
        return psnr

    def ssim_metric(path: Path, image_name: str) -> float:
        source = source_lookup.get(image_name.lower())
        if source is None:
            raise FileNotFoundError(f"Missing source image for {image_name}")
        _, ssim = calculate_psnr_ssim(source, path)
        return ssim

    bpp_curves = build_curves_from_metric(files_by_qf, lambda path, _: calculate_bpp(path))
    psnr_curves = build_curves_from_metric(files_by_qf, psnr_metric)
    ssim_curves = build_curves_from_metric(files_by_qf, ssim_metric)
    rho_grid = build_rho_grid(rho_step)

    for qf in sorted(files_by_qf.keys()):
        bpp_stats = align_curves(bpp_curves[qf], rho_grid, min_coverage=min_coverage)
        psnr_stats = align_curves(psnr_curves[qf], rho_grid, min_coverage=min_coverage)
        ssim_stats = align_curves(ssim_curves[qf], rho_grid, min_coverage=min_coverage)

        rows = build_rows_for_qf(qf, bpp_stats, psnr_stats, ssim_stats)
        if not rows:
            print(f"Skip {mode} subsampling={subsampling} qf={qf}: no common aligned rho")
            continue

        csv_path = tables_dir / f"{mode}_ss{subsampling}_qf{qf}_aligned.csv"
        write_aligned_table_csv(csv_path, rows)

        plot_path = figures_dir / f"{mode}_ss{subsampling}_qf{qf}_psnr_bpp.png"
        save_psnr_bpp_plot(plot_path, mode, subsampling, qf, rows)

        endpoint = rows[-1]
        endpoint_rows.append(
            {
                "mode": mode,
                "subsampling": subsampling,
                "qf": qf,
                "rho": endpoint["rho"],
                "mean_bpp": endpoint["mean_bpp"],
                "mean_psnr": endpoint["mean_psnr"],
                "mean_ssim": endpoint["mean_ssim"],
                "n": endpoint["n"],
            }
        )
        print(f"Done {mode} subsampling={subsampling} qf={qf}: {csv_path.name}, {plot_path.name}")


def write_endpoint_summary(csv_path: Path, endpoint_rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "subsampling", "qf", "rho", "mean_bpp", "mean_psnr", "mean_ssim", "n"])
        for row in sorted(endpoint_rows, key=lambda x: (x["mode"], x["subsampling"], x["qf"])):
            writer.writerow(
                [
                    row["mode"],
                    row["subsampling"],
                    row["qf"],
                    f"{row['rho']:.6f}",
                    f"{row['mean_bpp']:.8f}",
                    f"{row['mean_psnr']:.8f}",
                    f"{row['mean_ssim']:.8f}",
                    row["n"],
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run color-only mainline analysis (AC/DC + subsampling).")
    parser.add_argument("--input-root", type=Path, default=default_color_input_root(), help="Color input image directory")
    parser.add_argument("--output-root", type=Path, default=default_color_output_root(), help="Color encryption output root")
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=project_root() / "Result" / "AnalysisColorMainline",
        help="Directory for generated tables/figures",
    )
    parser.add_argument("--subsampling", type=str, default="444,420", help="Comma-separated subsampling list")
    parser.add_argument("--modes", type=str, default="ac,dc", help="Comma-separated modes: ac,dc")
    parser.add_argument("--rho-step", type=float, default=0.05)
    parser.add_argument("--min-coverage", type=float, default=0.7)
    args = parser.parse_args()

    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if not args.output_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {args.output_root}")

    subsamplings = parse_int_list(args.subsampling)
    modes = [token.strip().lower() for token in args.modes.split(",") if token.strip()]
    for mode in modes:
        if mode not in {"ac", "dc"}:
            raise ValueError(f"Unsupported mode: {mode}")

    tables_dir = args.analysis_root / "tables"
    figures_dir = args.analysis_root / "figures"
    endpoint_rows: list[dict] = []

    for subsampling in subsamplings:
        for mode in modes:
            run_mode(
                mode=mode,
                subsampling=subsampling,
                input_root=args.input_root,
                output_root=args.output_root,
                rho_step=args.rho_step,
                min_coverage=args.min_coverage,
                tables_dir=tables_dir,
                figures_dir=figures_dir,
                endpoint_rows=endpoint_rows,
            )

    summary_path = args.analysis_root / "endpoint_rho_max_summary.csv"
    write_endpoint_summary(summary_path, endpoint_rows)
    print(f"Wrote endpoint summary: {summary_path}")


if __name__ == "__main__":
    main()
