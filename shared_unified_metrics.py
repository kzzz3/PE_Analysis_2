from __future__ import annotations

import csv
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from shared_strength_alignment import MetricCurve


QF_PATTERN = re.compile(r"^QF=(\d+)$", re.IGNORECASE)
ST_PATTERN = re.compile(r"^ST=(\d+)$", re.IGNORECASE)
NUMERIC_JPG_PATTERN = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)


@dataclass
class Variant:
	name: str
	input_root: Path
	ac_root: Path
	dc_root: Path
	dataset: str
	subsampling: int | None = None


@dataclass
class SampleRecord:
	rho: float
	strength: int
	path: Path


def is_in_decrypt_folder(path: Path) -> bool:
	for part in path.parts:
		if part.lower() == "decrypt":
			return True
	return False


def parse_int_list(value: str) -> list[int]:
	items: list[int] = []
	for token in value.split(","):
		token = token.strip()
		if token:
			items.append(int(token))
	return items


def parse_token_list(value: str) -> list[str]:
	items: list[str] = []
	for token in value.split(","):
		token = token.strip().lower()
		if token:
			items.append(token)
	return items


def image_selected(image_name: str, selected: set[str] | None) -> bool:
	if not selected:
		return True
	lower = image_name.lower()
	stem = Path(image_name).stem.lower()
	return lower in selected or stem in selected


def build_source_lookup(input_root: Path) -> tuple[dict[str, Path], dict[str, str]]:
	lookup: dict[str, Path] = {}
	stem_to_name: dict[str, str] = {}
	if not input_root.exists():
		return lookup, stem_to_name

	for path in input_root.iterdir():
		if not path.is_file():
			continue
		lookup[path.name.lower()] = path
		stem_to_name[path.stem.lower()] = path.name
	return lookup, stem_to_name


def parse_qf(name: str) -> int | None:
	match = QF_PATTERN.match(name)
	if not match:
		return None
	return int(match.group(1))


def parse_st(name: str) -> int | None:
	match = ST_PATTERN.match(name)
	if not match:
		return None
	return int(match.group(1))


def parse_numeric_jpg(name: str) -> int | None:
	match = NUMERIC_JPG_PATTERN.match(name)
	if not match:
		return None
	return int(match.group(1))


def add_sample(
	target: dict[int, dict[str, list[SampleRecord]]],
	qf: int,
	image_name: str,
	rho: float,
	strength: int,
	path: Path,
	selected_images: set[str] | None,
) -> None:
	if not path.exists():
		return
	if is_in_decrypt_folder(path):
		return
	if not image_selected(image_name, selected_images):
		return
	if not np.isfinite(rho):
		return
	clamped_rho = min(max(float(rho), 0.0), 1.0)
	target.setdefault(qf, {}).setdefault(image_name, []).append(
		SampleRecord(rho=clamped_rho, strength=int(strength), path=path)
	)


def collect_gray_ac_samples(
	ac_root: Path,
	input_root: Path,
	selected_images: set[str] | None,
) -> dict[int, dict[str, list[SampleRecord]]]:
	result: dict[int, dict[str, list[SampleRecord]]] = {}
	_, stem_to_name = build_source_lookup(input_root)

	if not ac_root.exists():
		return result

	for qf_dir in sorted(ac_root.iterdir(), key=lambda p: p.name):
		if not qf_dir.is_dir():
			continue
		qf = parse_qf(qf_dir.name)
		if qf is None:
			continue

		for st_dir in sorted(qf_dir.iterdir(), key=lambda p: p.name):
			if not st_dir.is_dir():
				continue
			raw_st = parse_st(st_dir.name)
			if raw_st is None:
				continue

			# Current semantics: ST equals encrypted AC coefficient count
			# (stronger when ST is larger). Effective AC max is 63.
			ac_strength = min(63, max(0, raw_st))
			rho = ac_strength / 63.0 if ac_strength > 0 else 0.0

			for file_path in sorted(st_dir.iterdir(), key=lambda p: p.name.lower()):
				if not file_path.is_file() or file_path.suffix.lower() != ".jpg":
					continue
				stem_lower = file_path.stem.lower()
				image_name = stem_to_name.get(stem_lower)
				if image_name is None:
					# Skip auxiliary jpg files that are not source-image outputs.
					continue
				add_sample(result, qf, image_name, rho, ac_strength, file_path, selected_images)

	return result


def collect_gray_dc_samples(
	dc_root: Path,
	selected_images: set[str] | None,
) -> dict[int, dict[str, list[SampleRecord]]]:
	result: dict[int, dict[str, list[SampleRecord]]] = {}
	if not dc_root.exists():
		return result

	for qf_dir in sorted(dc_root.iterdir(), key=lambda p: p.name):
		if not qf_dir.is_dir():
			continue
		qf = parse_qf(qf_dir.name)
		if qf is None:
			continue

		for image_dir in sorted(qf_dir.iterdir(), key=lambda p: p.name.lower()):
			if not image_dir.is_dir():
				continue
			image_name = image_dir.name
			if not image_selected(image_name, selected_images):
				continue

			pairs: list[tuple[int, Path]] = []
			for file_path in image_dir.iterdir():
				if not file_path.is_file() or file_path.suffix.lower() != ".jpg":
					continue
				strength = parse_numeric_jpg(file_path.name)
				if strength is None:
					continue
				pairs.append((strength, file_path))

			if not pairs:
				continue

			pairs.sort(key=lambda x: x[0])
			max_strength = max(s for s, _ in pairs)
			if max_strength <= 0:
				continue

			for strength, path in pairs:
				rho = strength / float(max_strength)
				add_sample(result, qf, image_name, rho, strength, path, selected_images)

	return result


def _collect_color_meta_stage_samples(
	root: Path,
	target_stage: set[str],
	strength_key: str,
	selected_images: set[str] | None,
) -> dict[int, dict[str, list[SampleRecord]]]:
	result: dict[int, dict[str, list[SampleRecord]]] = {}
	if not root.exists():
		return result

	for qf_dir in sorted(root.iterdir(), key=lambda p: p.name):
		if not qf_dir.is_dir():
			continue
		qf = parse_qf(qf_dir.name)
		if qf is None:
			continue

		for image_dir in sorted(qf_dir.iterdir(), key=lambda p: p.name.lower()):
			if not image_dir.is_dir():
				continue
			image_name = image_dir.name
			if not image_selected(image_name, selected_images):
				continue

			meta_path = image_dir / "meta.csv"
			if not meta_path.exists():
				continue

			with meta_path.open("r", encoding="utf-8", newline="") as f:
				reader = csv.DictReader(f)
				for row in reader:
					stage = row.get("stage", "").strip().lower()
					if stage not in target_stage:
						continue

					file_name = row.get("file_name", "").strip()
					if not file_name:
						continue
					path = image_dir / file_name
					if not path.exists():
						continue

					try:
						rho = float(row.get("rho_stage", "nan"))
					except ValueError:
						continue

					try:
						strength = int(row.get(strength_key, row.get("k", "0")))
					except ValueError:
						try:
							strength = int(row.get("k", "0"))
						except ValueError:
							strength = 0

					add_sample(result, qf, image_name, rho, strength, path, selected_images)

	return result


def collect_color_ac_samples(
	ac_subsampling_root: Path,
	input_root: Path,
	selected_images: set[str] | None,
) -> dict[int, dict[str, list[SampleRecord]]]:
	meta_samples = _collect_color_meta_stage_samples(
		ac_subsampling_root,
		target_stage={"ac"},
		strength_key="ac_st",
		selected_images=selected_images,
	)
	if meta_samples:
		return meta_samples

	# Use ST-folder layout when AC meta rows are unavailable.
	return collect_gray_ac_samples(ac_subsampling_root, input_root, selected_images)


def collect_color_dc_samples(
	dc_subsampling_root: Path,
	selected_images: set[str] | None,
	include_dc_final: bool,
) -> dict[int, dict[str, list[SampleRecord]]]:
	stages = {"dc_progress", "dc_final_extra"} if include_dc_final else {"dc_progress"}
	meta_samples = _collect_color_meta_stage_samples(
		dc_subsampling_root,
		target_stage=stages,
		strength_key="k",
		selected_images=selected_images,
	)
	if meta_samples:
		return meta_samples

	# Use numeric-file layout when DC meta rows are unavailable.
	return collect_gray_dc_samples(dc_subsampling_root, selected_images)


def collect_samples(
	variant: Variant,
	mode: str,
	selected_images: set[str] | None = None,
	include_dc_final: bool = False,
) -> dict[int, dict[str, list[SampleRecord]]]:
	mode = mode.lower()
	if variant.dataset == "gray":
		if mode == "ac":
			return collect_gray_ac_samples(variant.ac_root, variant.input_root, selected_images)
		if mode == "dc":
			return collect_gray_dc_samples(variant.dc_root, selected_images)
		return {}

	if variant.dataset == "color":
		if mode == "ac":
			return collect_color_ac_samples(variant.ac_root, variant.input_root, selected_images)
		if mode == "dc":
			return collect_color_dc_samples(variant.dc_root, selected_images, include_dc_final)
		return {}

	return {}


def metric_backend_name() -> str:
	return "skimage"


@lru_cache(maxsize=1)
def _get_skimage_metrics():
	try:
		from skimage.metrics import peak_signal_noise_ratio as sk_psnr
		from skimage.metrics import structural_similarity as sk_ssim
		return sk_psnr, sk_ssim
	except Exception as exc:
		raise RuntimeError(
			"scikit-image is required for PSNR/SSIM computation. "
			"Please install it in the active environment."
		) from exc


def _dedupe_sorted_samples(samples: list[SampleRecord]) -> list[SampleRecord]:
	# sort by rho then strength, keep strongest sample when rho duplicates
	samples_sorted = sorted(samples, key=lambda s: (s.rho, s.strength))
	merged: list[SampleRecord] = []
	for sample in samples_sorted:
		if merged and abs(sample.rho - merged[-1].rho) < 1e-9:
			merged[-1] = sample
		else:
			merged.append(sample)
	return merged


def build_curves_from_samples(
	samples_by_qf: dict[int, dict[str, list[SampleRecord]]],
	metric_fn: Callable[[Path, str], float],
	progress_desc: str | None = None,
	workers: int = 1,
) -> dict[int, dict[str, MetricCurve]]:
	curves: dict[int, dict[str, MetricCurve]] = {}
	workers = max(1, int(workers))
	total = 0
	if progress_desc:
		for image_map in samples_by_qf.values():
			for samples in image_map.values():
				total += len(_dedupe_sorted_samples(samples))
	progress = (
		tqdm(total=total, desc=progress_desc, unit="sample", dynamic_ncols=True, leave=False)
		if progress_desc
		else None
	)

	task_paths: list[Path] = []
	task_names: list[str] = []
	task_meta: list[tuple[int, str, int]] = []
	for qf, image_map in samples_by_qf.items():
		curves[qf] = {}
		for image_name, samples in image_map.items():
			merged = _dedupe_sorted_samples(samples)
			if not merged:
				continue

			rhos = np.array([s.rho for s in merged], dtype=np.float64)
			strengths = np.array([s.strength for s in merged], dtype=np.float64)
			values = np.zeros(len(merged), dtype=np.float64)
			curves[qf][image_name] = MetricCurve(
				image_name=image_name,
				strengths=strengths,
				rhos=rhos,
				values=values,
			)

			for idx, sample in enumerate(merged):
				task_paths.append(sample.path)
				task_names.append(image_name)
				task_meta.append((qf, image_name, idx))

	try:
		if workers == 1:
			for path, image_name, (qf, name, idx) in zip(task_paths, task_names, task_meta):
				curves[qf][name].values[idx] = metric_fn(path, image_name)
				if progress is not None:
					progress.update(1)
		else:
			chunksize = max(1, len(task_paths) // (workers * 8))
			with ProcessPoolExecutor(max_workers=workers) as executor:
				for value, (qf, name, idx) in zip(
					executor.map(metric_fn, task_paths, task_names, chunksize=chunksize),
					task_meta,
				):
					curves[qf][name].values[idx] = value
					if progress is not None:
						progress.update(1)
	finally:
		if progress is not None:
			progress.close()

	return curves


def open_image_array(path: Path) -> np.ndarray:
	with Image.open(path) as img:
		if img.mode in {"1", "L", "I", "F", "I;16"}:
			arr = np.asarray(img.convert("L"), dtype=np.float64)
		else:
			arr = np.asarray(img.convert("RGB"), dtype=np.float64)
	return arr


def _match_image_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
	if a.shape == b.shape:
		return a, b

	if a.ndim == 2 and b.ndim == 3 and b.shape[2] == 3:
		a = np.repeat(a[:, :, None], 3, axis=2)
	elif b.ndim == 2 and a.ndim == 3 and a.shape[2] == 3:
		b = np.repeat(b[:, :, None], 3, axis=2)

	if a.shape != b.shape:
		raise ValueError(f"Image shape mismatch: {a.shape} vs {b.shape}")
	return a, b


def calculate_bpp(image_path: Path) -> float:
	with Image.open(image_path) as image:
		width, height = image.size
	bits = image_path.stat().st_size * 8
	return bits / float(width * height)


def calculate_difference_value(image_path: Path) -> float:
	image = open_image_array(image_path).astype(np.int64)
	horizontal = np.abs(image[:, :-1] - image[:, 1:]).sum()
	vertical = np.abs(image[:-1, :] - image[1:, :]).sum()
	return float(horizontal + vertical)


def calculate_psnr_ssim(source_path: Path, cipher_path: Path) -> tuple[float, float]:
	source = open_image_array(source_path)
	return calculate_psnr_ssim_with_source_array(source, cipher_path)


def calculate_psnr_ssim_with_source_array(source: np.ndarray, cipher_path: Path) -> tuple[float, float]:
	cipher = open_image_array(cipher_path)
	source, cipher = _match_image_shape(source, cipher)
	sk_psnr, sk_ssim = _get_skimage_metrics()
	psnr = float(sk_psnr(cipher, source, data_range=255))
	if source.ndim == 2:
		ssim = float(sk_ssim(cipher, source, data_range=255))
	else:
		ssim = float(sk_ssim(cipher, source, data_range=255, channel_axis=-1))
	return psnr, ssim
