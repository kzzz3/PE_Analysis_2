from shared_strength_alignment import (
    align_curves,
    build_curves_from_metric,
    build_rho_grid,
    build_source_lookup,
    calculate_psnr_ssim,
    collect_dc_image_strength_files,
    default_dc_root,
    default_input_root,
)


RHO_STEP = 0.05
MIN_COVERAGE = 0.7


def main() -> None:
    input_root = default_input_root()
    dc_root = default_dc_root()

    source_lookup = build_source_lookup(input_root)
    files_by_qf = collect_dc_image_strength_files(dc_root)

    def psnr_metric(path, image_name):
        source = source_lookup.get(image_name.lower())
        if source is None:
            raise FileNotFoundError(f"Missing source image for {image_name}")
        psnr, _ = calculate_psnr_ssim(source, path)
        return psnr

    def ssim_metric(path, image_name):
        source = source_lookup.get(image_name.lower())
        if source is None:
            raise FileNotFoundError(f"Missing source image for {image_name}")
        _, ssim = calculate_psnr_ssim(source, path)
        return ssim

    psnr_curves = build_curves_from_metric(files_by_qf, psnr_metric)
    ssim_curves = build_curves_from_metric(files_by_qf, ssim_metric)
    rho_grid = build_rho_grid(RHO_STEP)

    for qf in sorted(files_by_qf.keys()):
        psnr_stats = align_curves(psnr_curves[qf], rho_grid, min_coverage=MIN_COVERAGE)
        ssim_stats = align_curves(ssim_curves[qf], rho_grid, min_coverage=MIN_COVERAGE)

        print(f"QF={qf}")
        print("PSNR:")
        print("rho mean_psnr std_psnr coverage n")
        for rho, mean, std, coverage, n_sample in zip(
            psnr_stats.rho,
            psnr_stats.mean,
            psnr_stats.std,
            psnr_stats.coverage,
            psnr_stats.sample_count,
        ):
            print(f"{rho:.2f} {mean:.6f} {std:.6f} {coverage:.3f} {int(n_sample)}")

        print("SSIM:")
        print("rho mean_ssim std_ssim coverage n")
        for rho, mean, std, coverage, n_sample in zip(
            ssim_stats.rho,
            ssim_stats.mean,
            ssim_stats.std,
            ssim_stats.coverage,
            ssim_stats.sample_count,
        ):
            print(f"{rho:.2f} {mean:.6f} {std:.6f} {coverage:.3f} {int(n_sample)}")


if __name__ == "__main__":
    main()
