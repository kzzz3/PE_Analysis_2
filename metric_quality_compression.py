from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from shared_strength_alignment import (
	MetricCurve,
	align_curves,
	build_rho_grid,
	default_color_input_root,
	default_color_output_root,
	default_gray_input_root,
	default_gray_output_root,
	project_root,
)
from shared_unified_metrics import (
	SampleRecord,
	Variant,
	build_curves_from_samples,
	build_source_lookup,
	calculate_bpp,
	calculate_psnr_ssim_with_source_array,
	collect_samples,
	metric_backend_name,
	open_image_array,
	parse_int_list,
	parse_token_list,
)


def log_info(message: str) -> None:
	print(f"[INFO] {message}")


def log_warn(message: str) -> None:
	print(f"[WARN] {message}")


def normalize_datasets_arg(datasets_arg: str) -> list[str]:
	datasets = parse_token_list(datasets_arg)
	if "both" in datasets:
		return ["gray", "color"]
	return datasets


def build_selected_images(images_arg: str, single_image_arg: str) -> set[str] | None:
	tokens: list[str] = []
	if single_image_arg.strip():
		tokens.append(single_image_arg.strip())
	else:
		tokens.extend(token.strip() for token in images_arg.split(",") if token.strip())

	selected: set[str] = set()
	for token in tokens:
		lower = token.lower()
		selected.add(lower)
		selected.add(Path(lower).name.lower())
		selected.add(Path(lower).stem.lower())

	return selected or None


def build_dataset_selected_images(args, datasets: list[str]) -> dict[str, set[str] | None]:
	base_selected = build_selected_images(args.images, "")
	quick_any = bool(args.single_image.strip() or args.single_image_gray.strip() or args.single_image_color.strip())

	if "gray" in datasets and "color" in datasets:
		if args.single_image.strip():
			raise ValueError("When --datasets both is used, do not use --single-image; use --single-image-gray and --single-image-color.")
		if quick_any and (not args.single_image_gray.strip() or not args.single_image_color.strip()):
			raise ValueError("When --datasets both and quick mode are used, both --single-image-gray and --single-image-color are required.")

	selected_by_dataset: dict[str, set[str] | None] = {
		"gray": base_selected,
		"color": base_selected,
	}

	if args.single_image.strip():
		selected_single = build_selected_images("", args.single_image)
		for ds in datasets:
			selected_by_dataset[ds] = selected_single

	if args.single_image_gray.strip():
		selected_by_dataset["gray"] = build_selected_images("", args.single_image_gray)

	if args.single_image_color.strip():
		selected_by_dataset["color"] = build_selected_images("", args.single_image_color)

	return selected_by_dataset


def variant_matches_selected_images(variant: Variant, selected: set[str] | None) -> bool:
	if not selected:
		return True
	if not variant.input_root.exists():
		return False
	for path in variant.input_root.iterdir():
		if not path.is_file():
			continue
		name = path.name.lower()
		stem = path.stem.lower()
		if name in selected or stem in selected:
			return True
	return False


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


def write_rows(path: Path, rows: list[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	rows = sorted(rows, key=lambda r: float(r["rho"]))
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


def build_rows(bpp_stats, psnr_stats, ssim_stats) -> list[dict]:
	bpp_map = stats_to_map(bpp_stats)
	psnr_map = stats_to_map(psnr_stats)
	ssim_map = stats_to_map(ssim_stats)

	rows: list[dict] = []
	common = sorted(set(bpp_map.keys()) & set(psnr_map.keys()) & set(ssim_map.keys()))
	for rho in common:
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


def _series_label(row: dict) -> str:
	if row["dataset"] == "gray":
		return f"gray-{row['mode']}-QF{row['qf']}"
	return f"color{row['subsampling']}-{row['mode']}-QF{row['qf']}"


def write_preview_plot(all_rows: list[dict], output_path: Path, metric_key: str, y_label: str) -> None:
	if not all_rows:
		return

	# keep preview concise: only highest QF curves for each (variant, mode)
	max_qf = max(int(row["qf"]) for row in all_rows)
	rows = [row for row in all_rows if int(row["qf"]) == max_qf]
	if not rows:
		return

	series: dict[str, list[tuple[float, float]]] = {}
	for row in rows:
		label = _series_label(row)
		series.setdefault(label, []).append((float(row["rho"]), float(row[metric_key])))

	for label in series:
		series[label].sort(key=lambda x: x[0])

	all_values = [v for points in series.values() for _, v in points]
	if not all_values:
		return

	y_min = min(all_values)
	y_max = max(all_values)
	if abs(y_max - y_min) < 1e-12:
		y_max = y_min + 1.0
	margin = 0.05 * (y_max - y_min)
	y_min -= margin
	y_max += margin

	width, height = 1400, 880
	left, right, top, bottom = 120, 40, 60, 150
	plot_w = width - left - right
	plot_h = height - top - bottom

	img = Image.new("RGB", (width, height), "white")
	draw = ImageDraw.Draw(img)

	# axes
	draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)
	draw.line((left, top, left, top + plot_h), fill="black", width=2)

	# grid and ticks
	for i in range(11):
		x = left + int(plot_w * i / 10)
		draw.line((x, top, x, top + plot_h), fill=(235, 235, 235), width=1)
		label = f"{i/10:.1f}" if i not in {0, 10} else str(i // 10)
		draw.text((x - 12, top + plot_h + 10), label, fill="black")

	for i in range(9):
		r = i / 8
		y = top + int(plot_h * (1 - r))
		draw.line((left, y, left + plot_w, y), fill=(235, 235, 235), width=1)
		v = y_min + (y_max - y_min) * r
		draw.text((20, y - 8), f"{v:.2f}", fill="black")

	draw.text((left + plot_w // 2 - 45, top + plot_h + 50), "rho", fill="black")
	draw.text((left - 90, top - 30), y_label, fill="black")
	draw.text((left, 15), f"Preview ({metric_key.upper()}, QF={max_qf})", fill="black")

	palette = [
		(228, 26, 28),
		(55, 126, 184),
		(77, 175, 74),
		(152, 78, 163),
		(255, 127, 0),
		(166, 86, 40),
		(247, 129, 191),
		(153, 153, 153),
	]

	for idx, (label, points) in enumerate(sorted(series.items())):
		color = palette[idx % len(palette)]
		xy: list[tuple[int, int]] = []
		for rho, value in points:
			x = left + int(plot_w * rho)
			y = top + int(plot_h * (1.0 - (value - y_min) / (y_max - y_min)))
			xy.append((x, y))
		if len(xy) >= 2:
			draw.line(xy, fill=color, width=3)
		for x, y in xy:
			draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)

	# legend
	legend_x = left
	legend_y = top + plot_h + 80
	for idx, label in enumerate(sorted(series.keys())):
		color = palette[idx % len(palette)]
		x = legend_x + (idx % 3) * 430
		y = legend_y + (idx // 3) * 22
		draw.line((x, y + 8, x + 24, y + 8), fill=color, width=3)
		draw.text((x + 30, y), label, fill="black")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	img.save(output_path)


def metric_bpp(path: Path, _image_name: str) -> float:
	return calculate_bpp(path)


def dedupe_sorted_samples(samples: list[SampleRecord]) -> list[SampleRecord]:
	samples_sorted = sorted(samples, key=lambda s: (s.rho, s.strength))
	merged: list[SampleRecord] = []
	for sample in samples_sorted:
		if merged and abs(sample.rho - merged[-1].rho) < 1e-9:
			merged[-1] = sample
		else:
			merged.append(sample)
	return merged


_SOURCE_ARRAY_CACHE: dict[str, np.ndarray] = {}


def quality_metric_task(task: tuple[str, str]) -> tuple[float, float]:
	source_path_str, cipher_path_str = task
	source = _SOURCE_ARRAY_CACHE.get(source_path_str)
	if source is None:
		source = open_image_array(Path(source_path_str))
		_SOURCE_ARRAY_CACHE[source_path_str] = source
	return calculate_psnr_ssim_with_source_array(source, Path(cipher_path_str))


def build_quality_curves_from_samples(
	samples_by_qf: dict[int, dict[str, list[SampleRecord]]],
	source_lookup: dict[str, Path],
	workers: int,
	progress_desc: str,
):
	workers = max(1, int(workers))
	meta_list: list[tuple[int, str, int]] = []
	task_inputs: list[tuple[str, str]] = []
	templates: dict[int, dict[str, dict[str, np.ndarray]]] = {}

	for qf, image_map in samples_by_qf.items():
		templates[qf] = {}
		for image_name, samples in image_map.items():
			merged = dedupe_sorted_samples(samples)
			if not merged:
				continue
			source_path = source_lookup.get(image_name.lower())
			if source_path is None:
				raise FileNotFoundError(f"Missing source image for {image_name}")

			templates[qf][image_name] = {
				"rhos": np.array([s.rho for s in merged], dtype=np.float64),
				"strengths": np.array([s.strength for s in merged], dtype=np.float64),
				"psnr": np.zeros(len(merged), dtype=np.float64),
				"ssim": np.zeros(len(merged), dtype=np.float64),
			}

			for idx, sample in enumerate(merged):
				task_inputs.append((str(source_path), str(sample.path)))
				meta_list.append((qf, image_name, idx))

	if not task_inputs:
		return {}, {}

	progress = tqdm(total=len(task_inputs), desc=progress_desc, unit="sample", dynamic_ncols=True, leave=False)
	try:
		results: list[tuple[float, float]] = []
		if workers == 1:
			for task in task_inputs:
				results.append(quality_metric_task(task))
				progress.update(1)
		else:
			chunksize = max(1, len(task_inputs) // (workers * 8))
			with ProcessPoolExecutor(max_workers=workers) as executor:
				for value in executor.map(quality_metric_task, task_inputs, chunksize=chunksize):
					results.append(value)
					progress.update(1)
	finally:
		progress.close()

	for (qf, image_name, idx), (psnr, ssim) in zip(meta_list, results):
		templates[qf][image_name]["psnr"][idx] = psnr
		templates[qf][image_name]["ssim"][idx] = ssim

	psnr_curves: dict[int, dict[str, MetricCurve]] = {}
	ssim_curves: dict[int, dict[str, MetricCurve]] = {}
	for qf, image_map in templates.items():
		psnr_curves[qf] = {}
		ssim_curves[qf] = {}
		for image_name, bundle in image_map.items():
			psnr_curves[qf][image_name] = MetricCurve(
				image_name=image_name,
				strengths=bundle["strengths"],
				rhos=bundle["rhos"],
				values=bundle["psnr"],
			)
			ssim_curves[qf][image_name] = MetricCurve(
				image_name=image_name,
				strengths=bundle["strengths"],
				rhos=bundle["rhos"],
				values=bundle["ssim"],
			)

	return psnr_curves, ssim_curves


def resolve_variants(args) -> list[Variant]:
	datasets = normalize_datasets_arg(args.datasets)

	variants: list[Variant] = []
	if "gray" in datasets:
		variants.append(
			Variant(
				name="gray",
				input_root=args.gray_input_root,
				ac_root=args.gray_ac_root,
				dc_root=args.gray_dc_root,
				dataset="gray",
				subsampling=None,
			)
		)

	if "color" in datasets:
		subsamplings = parse_int_list(args.subsampling)
		for ss in subsamplings:
			variants.append(
				Variant(
					name=f"color_ss{ss}",
					input_root=args.color_input_root,
					ac_root=args.color_ac_root / f"Subsampling={ss}",
					dc_root=args.color_dc_root / f"Subsampling={ss}",
					dataset="color",
					subsampling=ss,
				)
			)

	return variants


def resolve_workers(requested: int) -> int:
	workers = max(1, int(requested))
	cpu = os.cpu_count() or workers
	workers = min(workers, cpu)
	if sys.platform.startswith("win") and workers > 8:
		log_warn(f"workers={workers} is high for Windows multiprocessing spawn; auto-cap to 8.")
		workers = 8
	return workers


def main() -> None:
	parser = argparse.ArgumentParser(description="Unified quality/compression analysis for gray and color datasets.")
	parser.add_argument("--datasets", type=str, default="both", help="gray,color,both")
	parser.add_argument("--modes", type=str, default="both", help="ac,dc,both")
	parser.add_argument("--subsampling", type=str, default="444,420", help="Color subsampling list")
	parser.add_argument("--qfs", type=str, default="30,50,70,90", help="QF list")
	parser.add_argument("--images", type=str, default="", help="Optional image filter list (name or stem)")
	parser.add_argument("--single-image", type=str, default="", help="Quick mode: run only one image name/path")
	parser.add_argument("--single-image-gray", type=str, default="", help="Quick mode image for gray dataset")
	parser.add_argument("--single-image-color", type=str, default="", help="Quick mode image for color dataset")
	parser.add_argument("--rho-step", type=float, default=0.05)
	parser.add_argument("--min-coverage", type=float, default=0.7)
	parser.add_argument("--include-dc-final", action="store_true", help="Include dc_final_extra rows for color meta")
	parser.add_argument("--workers", type=int, default=1, help="Worker processes for metric evaluation")
	parser.add_argument("--disable-preview", action="store_true", help="Disable preview image export")

	parser.add_argument("--gray-input-root", type=Path, default=default_gray_input_root())
	parser.add_argument("--gray-ac-root", type=Path, default=default_gray_output_root() / "AcEncryption")
	parser.add_argument("--gray-dc-root", type=Path, default=default_gray_output_root() / "DcEncryption")

	parser.add_argument("--color-input-root", type=Path, default=default_color_input_root())
	parser.add_argument("--color-ac-root", type=Path, default=default_color_output_root() / "AcEncryption")
	parser.add_argument("--color-dc-root", type=Path, default=default_color_output_root() / "DcEncryption")

	parser.add_argument(
		"--analysis-root",
		type=Path,
		default=project_root() / "Result" / "AnalysisUnified" / "QualityCompression",
		help="Output directory for CSV tables",
	)
	args = parser.parse_args()

	modes = parse_token_list(args.modes)
	if "both" in modes:
		modes = ["ac", "dc"]
	for mode in modes:
		if mode not in {"ac", "dc"}:
			raise ValueError(f"Unsupported mode: {mode}")

	qf_set = set(parse_int_list(args.qfs))
	datasets = normalize_datasets_arg(args.datasets)
	selected_by_dataset = build_dataset_selected_images(args, datasets)
	if args.single_image.strip():
		log_info(f"Quick mode enabled: single image '{args.single_image.strip()}'")
	if args.single_image_gray.strip():
		log_info(f"Quick mode gray image: '{args.single_image_gray.strip()}'")
	if args.single_image_color.strip():
		log_info(f"Quick mode color image: '{args.single_image_color.strip()}'")

	variants = resolve_variants(args)
	if variants:
		variants = [
			v for v in variants if variant_matches_selected_images(v, selected_by_dataset.get(v.dataset))
		]
		if not variants:
			raise ValueError("No dataset variant contains the selected image filter.")
	rho_grid = build_rho_grid(args.rho_step)
	endpoint_rows: list[dict] = []
	all_rows: list[dict] = []
	workers = resolve_workers(args.workers)
	log_info(f"Metric backend: {metric_backend_name()}")
	log_info(f"Workers: {workers}")

	for variant in variants:
		if not variant.input_root.exists():
			log_warn(f"Skip {variant.name}: missing input root {variant.input_root}")
			continue

		source_lookup, _ = build_source_lookup(variant.input_root)
		if not source_lookup:
			log_warn(f"Skip {variant.name}: no source images found in {variant.input_root}")
			continue

		for mode in modes:
			selected_images = selected_by_dataset.get(variant.dataset)
			samples_by_qf = collect_samples(
				variant=variant,
				mode=mode,
				selected_images=selected_images,
				include_dc_final=args.include_dc_final,
			)
			if not samples_by_qf:
				log_warn(f"Skip {variant.name}/{mode}: no samples")
				continue

			progress_prefix = f"{variant.name}/{mode}"
			bpp_curves = build_curves_from_samples(
				samples_by_qf,
				metric_bpp,
				progress_desc=f"{progress_prefix} BPP",
				workers=1,
			)
			psnr_curves, ssim_curves = build_quality_curves_from_samples(
				samples_by_qf,
				source_lookup=source_lookup,
				workers=workers,
				progress_desc=f"{progress_prefix} QUALITY(PSNR+SSIM)",
			)

			for qf in sorted(samples_by_qf.keys()):
				if qf_set and qf not in qf_set:
					continue
				if qf not in bpp_curves or qf not in psnr_curves or qf not in ssim_curves:
					continue

				bpp_stats = align_curves(bpp_curves[qf], rho_grid, min_coverage=args.min_coverage)
				psnr_stats = align_curves(psnr_curves[qf], rho_grid, min_coverage=args.min_coverage)
				ssim_stats = align_curves(ssim_curves[qf], rho_grid, min_coverage=args.min_coverage)

				rows = build_rows(bpp_stats, psnr_stats, ssim_stats)
				if not rows:
					log_warn(f"Skip {variant.name}/{mode}/QF={qf}: no aligned points")
					continue

				out_path = args.analysis_root / "tables" / variant.name / mode / f"qf{qf}_aligned.csv"
				write_rows(out_path, rows)

				for row in rows:
					all_rows.append(
						{
							"variant": variant.name,
							"dataset": variant.dataset,
							"subsampling": variant.subsampling if variant.subsampling is not None else "-",
							"mode": mode,
							"qf": qf,
							"rho": row["rho"],
							"mean_bpp": row["mean_bpp"],
							"mean_psnr": row["mean_psnr"],
							"mean_ssim": row["mean_ssim"],
							"coverage_bpp": row["coverage_bpp"],
							"coverage_psnr": row["coverage_psnr"],
							"coverage_ssim": row["coverage_ssim"],
							"n": row["n"],
						}
					)

				endpoint = rows[-1]
				endpoint_rows.append(
					{
						"variant": variant.name,
						"dataset": variant.dataset,
						"subsampling": variant.subsampling if variant.subsampling is not None else "-",
						"mode": mode,
						"qf": qf,
						"rho": endpoint["rho"],
						"mean_bpp": endpoint["mean_bpp"],
						"mean_psnr": endpoint["mean_psnr"],
						"mean_ssim": endpoint["mean_ssim"],
						"n": endpoint["n"],
					}
				)
				log_info(f"Done {variant.name}/{mode}/QF={qf}: {out_path}")

	if endpoint_rows:
		endpoint_rows = sorted(
			endpoint_rows,
			key=lambda r: (r["dataset"], str(r["subsampling"]), r["mode"], int(r["qf"]), float(r["rho"])),
		)
		summary_path = args.analysis_root / "endpoint_summary.csv"
		summary_path.parent.mkdir(parents=True, exist_ok=True)
		with summary_path.open("w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(["variant", "dataset", "subsampling", "mode", "qf", "rho", "mean_bpp", "mean_psnr", "mean_ssim", "n"])
			for row in endpoint_rows:
				writer.writerow(
					[
						row["variant"],
						row["dataset"],
						row["subsampling"],
						row["mode"],
						row["qf"],
						f"{row['rho']:.6f}",
						f"{row['mean_bpp']:.8f}",
						f"{row['mean_psnr']:.8f}",
						f"{row['mean_ssim']:.8f}",
						row["n"],
					]
				)
		log_info(f"Wrote endpoint summary: {summary_path}")

	if all_rows:
		all_rows = sorted(
			all_rows,
			key=lambda r: (r["dataset"], str(r["subsampling"]), r["mode"], int(r["qf"]), float(r["rho"])),
		)
		all_path = args.analysis_root / "all_rho_summary.csv"
		all_path.parent.mkdir(parents=True, exist_ok=True)
		with all_path.open("w", newline="", encoding="utf-8") as f:
			writer = csv.writer(f)
			writer.writerow(
				[
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
				]
			)
			for row in all_rows:
				writer.writerow(
					[
						row["variant"],
						row["dataset"],
						row["subsampling"],
						row["mode"],
						row["qf"],
						f"{row['rho']:.6f}",
						f"{row['mean_bpp']:.8f}",
						f"{row['mean_psnr']:.8f}",
						f"{row['mean_ssim']:.8f}",
						f"{row['coverage_bpp']:.4f}",
						f"{row['coverage_psnr']:.4f}",
						f"{row['coverage_ssim']:.4f}",
						row["n"],
					]
				)
		log_info(f"Wrote all-rho summary: {all_path}")

		if not args.disable_preview:
			preview_dir = args.analysis_root / "preview"
			psnr_preview = preview_dir / "quality_preview_psnr.png"
			bpp_preview = preview_dir / "quality_preview_bpp.png"
			write_preview_plot(all_rows, psnr_preview, metric_key="mean_psnr", y_label="PSNR")
			write_preview_plot(all_rows, bpp_preview, metric_key="mean_bpp", y_label="BPP")
			log_info(f"Saved preview image: {psnr_preview}")
			log_info(f"Saved preview image: {bpp_preview}")


if __name__ == "__main__":
	main()
