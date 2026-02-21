from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

from shared_strength_alignment import (
	align_curves,
	build_rho_grid,
	default_color_input_root,
	default_color_output_root,
	default_gray_input_root,
	default_gray_output_root,
	project_root,
)
from shared_unified_metrics import (
	Variant,
	build_curves_from_samples,
	calculate_difference_value,
	collect_samples,
	metric_backend_name,
	parse_int_list,
	parse_token_list,
)


def log_info(message: str) -> None:
	print(f"[INFO] {message}")


def log_warn(message: str) -> None:
	print(f"[WARN] {message}")


def difference_metric(path: Path, _image_name: str) -> float:
	return calculate_difference_value(path)


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


def stats_to_rows(stats) -> list[dict]:
	rows: list[dict] = []
	for rho, mean, std, cov, n in zip(stats.rho, stats.mean, stats.std, stats.coverage, stats.sample_count):
		rows.append(
			{
				"rho": float(rho),
				"mean_diff": float(mean),
				"std_diff": float(std),
				"coverage": float(cov),
				"n": int(n),
			}
		)
	return rows


def write_rows(path: Path, rows: list[dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	rows = sorted(rows, key=lambda r: float(r["rho"]))
	with path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["rho", "mean_diff", "std_diff", "mean_diff_div1e5", "std_diff_div1e5", "coverage", "n"])
		for row in rows:
			writer.writerow(
				[
					f"{row['rho']:.6f}",
					f"{row['mean_diff']:.8f}",
					f"{row['std_diff']:.8f}",
					f"{row['mean_diff'] / 1e5:.8f}",
					f"{row['std_diff'] / 1e5:.8f}",
					f"{row['coverage']:.4f}",
					row["n"],
				]
			)


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
	parser = argparse.ArgumentParser(description="Unified difference metric analysis for gray and color datasets.")
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

	parser.add_argument("--gray-input-root", type=Path, default=default_gray_input_root())
	parser.add_argument("--gray-ac-root", type=Path, default=default_gray_output_root() / "AcEncryption")
	parser.add_argument("--gray-dc-root", type=Path, default=default_gray_output_root() / "DcEncryption")

	parser.add_argument("--color-input-root", type=Path, default=default_color_input_root())
	parser.add_argument("--color-ac-root", type=Path, default=default_color_output_root() / "AcEncryption")
	parser.add_argument("--color-dc-root", type=Path, default=default_color_output_root() / "DcEncryption")

	parser.add_argument(
		"--analysis-root",
		type=Path,
		default=project_root() / "Result" / "AnalysisUnified" / "Difference",
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
	workers = resolve_workers(args.workers)
	rho_grid = build_rho_grid(args.rho_step)
	log_info(f"Metric backend: {metric_backend_name()}")
	log_info(f"Workers: {workers}")

	for variant in variants:
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

			curves = build_curves_from_samples(
				samples_by_qf,
				difference_metric,
				progress_desc=f"{variant.name}/{mode} DIFF",
				workers=workers,
			)
			for qf in sorted(samples_by_qf.keys()):
				if qf_set and qf not in qf_set:
					continue
				if qf not in curves:
					continue

				stats = align_curves(curves[qf], rho_grid, min_coverage=args.min_coverage)
				rows = stats_to_rows(stats)
				if not rows:
					log_warn(f"Skip {variant.name}/{mode}/QF={qf}: no aligned points")
					continue

				out_path = args.analysis_root / "tables" / variant.name / mode / f"qf{qf}_aligned.csv"
				write_rows(out_path, rows)
				log_info(f"Done {variant.name}/{mode}/QF={qf}: {out_path}")


if __name__ == "__main__":
	main()
