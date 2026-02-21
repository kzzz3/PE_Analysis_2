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

- `metric_quality_compression.py`: unified AC/DC quality+compression analysis for gray and color (supports color subsampling)
- `metric_difference.py`: unified AC/DC difference-value analysis for gray and color (supports color subsampling)
- `metric_runtime_benchmark.py`: command-template based runtime benchmark (proposed + related methods)
- `metric_iwt_level_ablation.py`: IWT-level (`n`) ablation orchestrator (generation + metrics merge)
- `shared_unified_metrics.py`: shared collectors and metric helpers used by unified scripts
- `all_rho_summary_to_latex.py`: convert `all_rho_summary.csv` into pgfplots LaTeX snippets

Dependencies:
- Required: `scikit-image`, `tqdm`, `numpy`, `Pillow`

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
- `run_color_mainline.py`: color-only AC/DC mainline pipeline (aligned CSV + endpoint summary + PSNR-BPP figures)
- `run_color_consistency.py`: consistency-control analysis pipeline (AC-stage, DC-stage, DC-final tables and summary)

## Utility

- `tool_merge_dc_ac_dct.py`: merge DC and AC DCT components from two image sets

## Shared module

- `shared_strength_alignment.py`: folder parsing, normalized-strength alignment, interpolation, coverage filtering, and common metric helpers

## Unified metrics quick run

```bash
conda run -n expr python Analysis/metric_quality_compression.py --datasets both --modes both --subsampling 444,420
conda run -n expr python Analysis/metric_difference.py --datasets both --modes both --subsampling 444,420
conda run -n expr python Analysis/all_rho_summary_to_latex.py --input-csv Result/AnalysisUnified/QualityCompression/all_rho_summary.csv
```

Use multi-process workers:

```bash
conda run -n expr python Analysis/metric_quality_compression.py --datasets both --modes both --subsampling 444,420 --workers 8
conda run -n expr python Analysis/metric_difference.py --datasets both --modes both --subsampling 444,420 --workers 8
```

Notes on `--workers`:
- Scripts use multiprocessing (`ProcessPoolExecutor`).
- On Windows, worker count is auto-capped to 8 to avoid `BrokenProcessPool` and SciPy DLL/pagefile failures when spawning too many processes.

Quick single-image mode:

```bash
conda run -n expr python Analysis/metric_quality_compression.py --datasets both --modes both --subsampling 444,420 --qfs 90 --single-image-gray barbara_gray.bmp --single-image-color kodim01.png
conda run -n expr python Analysis/metric_difference.py --datasets both --modes both --subsampling 444,420 --qfs 90 --single-image-gray barbara_gray.bmp --single-image-color kodim01.png
```

Notes:
- If `--datasets both` and quick mode are used, both `--single-image-gray` and `--single-image-color` are required.
- `--single-image` is only for non-`both` dataset runs.
- `metric_quality_compression.py` saves preview plots automatically in `<analysis-root>/preview/`.
- AC `ST` is interpreted as encrypted AC-count strength (larger `ST` means stronger encryption).
- Files under folders named `decrypt/` are excluded from metric aggregation.
