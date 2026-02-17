import matplotlib.pyplot as plt

from shared_strength_alignment import (
    align_curves,
    build_curves_from_metric,
    build_rho_grid,
    build_source_lookup,
    calculate_bpp,
    calculate_psnr_ssim,
    collect_dc_image_strength_files,
    default_dc_root,
    default_input_root,
)


TARGET_QF = 90
RHO_STEP = 0.05
MIN_COVERAGE = 0.7


def main() -> None:
    input_root = default_input_root()
    dc_root = default_dc_root()

    source_lookup = build_source_lookup(input_root)
    files_by_qf = collect_dc_image_strength_files(dc_root)
    if TARGET_QF not in files_by_qf:
        raise ValueError(f"QF={TARGET_QF} not found in {dc_root}")

    qf_files = {TARGET_QF: files_by_qf[TARGET_QF]}

    def psnr_metric(path, image_name):
        source = source_lookup.get(image_name.lower())
        if source is None:
            raise FileNotFoundError(f"Missing source image for {image_name}")
        psnr, _ = calculate_psnr_ssim(source, path)
        return psnr

    bpp_curves = build_curves_from_metric(qf_files, lambda path, _: calculate_bpp(path))
    psnr_curves = build_curves_from_metric(qf_files, psnr_metric)

    rho_grid = build_rho_grid(RHO_STEP)
    bpp_stats = align_curves(bpp_curves[TARGET_QF], rho_grid, min_coverage=MIN_COVERAGE)
    psnr_stats = align_curves(psnr_curves[TARGET_QF], rho_grid, min_coverage=MIN_COVERAGE)

    bpp_by_rho = {round(float(rho), 6): (mean, cov, n) for rho, mean, cov, n in zip(
        bpp_stats.rho,
        bpp_stats.mean,
        bpp_stats.coverage,
        bpp_stats.sample_count,
    )}
    psnr_by_rho = {round(float(rho), 6): (mean, cov, n) for rho, mean, cov, n in zip(
        psnr_stats.rho,
        psnr_stats.mean,
        psnr_stats.coverage,
        psnr_stats.sample_count,
    )}

    common_rho = sorted(set(bpp_by_rho.keys()) & set(psnr_by_rho.keys()))
    if not common_rho:
        raise RuntimeError("No aligned rho points remain after coverage filtering.")

    bpp_values = [bpp_by_rho[rho][0] for rho in common_rho]
    psnr_values = [psnr_by_rho[rho][0] for rho in common_rho]

    plt.figure(figsize=(10, 6))
    plt.plot(bpp_values, psnr_values, marker="o", color="b", label=f"QF={TARGET_QF}")
    plt.xlabel("BPP (bits per pixel)")
    plt.ylabel("PSNR (dB)")
    plt.title(f"DC encryption: aligned mean PSNR vs BPP (QF={TARGET_QF})")
    plt.grid(True)
    plt.legend()
    output_name = f"dc_psnr_vs_bpp_qf{TARGET_QF}_aligned.png"
    plt.savefig(output_name, dpi=300, bbox_inches="tight")

    print(f"Saved plot: {output_name}")
    print("rho mean_bpp mean_psnr bpp_coverage psnr_coverage n")
    for rho in common_rho:
        bpp_mean, bpp_cov, bpp_n = bpp_by_rho[rho]
        psnr_mean, psnr_cov, psnr_n = psnr_by_rho[rho]
        print(
            f"{rho:.2f} {bpp_mean:.6f} {psnr_mean:.6f} "
            f"{bpp_cov:.3f} {psnr_cov:.3f} {int(min(bpp_n, psnr_n))}"
        )

    plt.show()


if __name__ == "__main__":
    main()
