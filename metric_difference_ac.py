from shared_strength_alignment import (
    align_curves,
    build_curves_from_metric,
    build_rho_grid,
    calculate_difference_value,
    collect_ac_image_strength_files,
    default_ac_root,
    default_input_root,
)


RHO_STEP = 0.05
MIN_COVERAGE = 0.7


def main() -> None:
    input_root = default_input_root()
    ac_root = default_ac_root()

    files_by_qf = collect_ac_image_strength_files(ac_root, input_root=input_root)
    curves_by_qf = build_curves_from_metric(files_by_qf, lambda path, _: calculate_difference_value(path))
    rho_grid = build_rho_grid(RHO_STEP)

    for qf in sorted(curves_by_qf.keys()):
        stats = align_curves(curves_by_qf[qf], rho_grid, min_coverage=MIN_COVERAGE)

        print(f"QF={qf}")
        print("rho mean_dv_div1e5 std_dv_div1e5 coverage n")
        for rho, mean, std, coverage, n_sample in zip(
            stats.rho,
            stats.mean,
            stats.std,
            stats.coverage,
            stats.sample_count,
        ):
            print(f"{rho:.2f} {mean/1e5:.6f} {std/1e5:.6f} {coverage:.3f} {int(n_sample)}")


if __name__ == "__main__":
    main()
