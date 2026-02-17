from shared_strength_alignment import (
    align_curves,
    build_curves_from_metric,
    build_rho_grid,
    calculate_bpp,
    collect_dc_image_strength_files,
    default_dc_root,
)


RHO_STEP = 0.05
MIN_COVERAGE = 0.7


def main() -> None:
    dc_root = default_dc_root()
    files_by_qf = collect_dc_image_strength_files(dc_root)
    curves_by_qf = build_curves_from_metric(files_by_qf, lambda path, _: calculate_bpp(path))
    rho_grid = build_rho_grid(RHO_STEP)

    for qf in sorted(curves_by_qf.keys()):
        stats = align_curves(curves_by_qf[qf], rho_grid, min_coverage=MIN_COVERAGE)

        print(f"QF={qf}")
        print("rho mean_bpp std coverage n")
        for rho, mean, std, coverage, n_sample in zip(
            stats.rho,
            stats.mean,
            stats.std,
            stats.coverage,
            stats.sample_count,
        ):
            print(f"{rho:.2f} {mean:.6f} {std:.6f} {coverage:.3f} {int(n_sample)}")


if __name__ == "__main__":
    main()



