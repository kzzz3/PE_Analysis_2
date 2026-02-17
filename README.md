# Analysis Scripts

This folder is organized by purpose.

## Naming convention

- `metric_*`: metric aggregation scripts (table-ready outputs)
- `plot_*`: plotting scripts
- `attack_*`: attack/reconstruction scripts
- `tool_*`: utility scripts
- `shared_*`: shared helper module
- descriptive standalone names: dedicated analysis/inspection scripts

## Core metrics

- `metric_compression_ac.py`: AC compression metric with normalized strength (`rho`) + interpolation/coverage filtering
- `metric_compression_dc.py`: DC compression metric with normalized strength (`rho`) + interpolation/coverage filtering
- `metric_difference_ac.py`: AC difference-value metric on normalized strength
- `metric_difference_dc.py`: DC difference-value metric on normalized strength
- `metric_quality_ac.py`: AC PSNR/SSIM metric on normalized strength
- `metric_quality_dc.py`: DC PSNR/SSIM metric on normalized strength

## Plotting

- `plot_psnr_bpp_ac.py`: aligned mean PSNR-BPP curve for AC
- `plot_psnr_bpp_dc.py`: aligned mean PSNR-BPP curve for DC
- `plot_psnr_bpp_sampled.py`: sampled PSNR-BPP plotting helper for custom folders

## Attack scripts

- `attack_optimization_ac.py`: optimization-based attack for AC encryption outputs
- `attack_optimization_dc.py`: optimization-based attack for DC encryption outputs
- `attack_sketch.py`: DCT-block sketch analysis (NCC/EAC/PLZ feature map generation)

## Dedicated analysis scripts

- `histogram_analysis.py`: plain/cipher histogram comparison and vector-figure export
- `edge_detection.py`: Canny/Sobel edge visualization on target images
- `probe_single_image_quality_dc.py`: single-image DC quality scan across strengths
- `probe_monotonic_quality.py`: monotonicity check over encryption strength
- `probe_security_metrics.py`: NPCR/UACI/EDR statistics for paired folders

## Utility

- `tool_merge_dc_ac_dct.py`: merge DC and AC DCT components from two image sets

## Shared module

- `shared_strength_alignment.py`: folder parsing, normalized-strength alignment, interpolation, coverage filtering, and common metric helpers
