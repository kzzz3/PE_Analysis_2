from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import numpy as np
from PIL import Image


QF_PATTERN = re.compile(r"^QF=(\d+)$", re.IGNORECASE)
STRENGTH_IMAGE_PATTERN = re.compile(r"^(\d+)\.jpg$", re.IGNORECASE)


@dataclass
class MetricCurve:
    image_name: str
    strengths: np.ndarray
    rhos: np.ndarray
    values: np.ndarray


@dataclass
class AlignedStats:
    rho: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    coverage: np.ndarray
    sample_count: np.ndarray


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_gray_input_root() -> Path:
    gray_root = project_root() / "Result" / "InputImage" / "Gray"
    if gray_root.exists():
        return gray_root
    return project_root() / "Result" / "InputImage"


def default_color_input_root() -> Path:
    return project_root() / "Result" / "InputImage" / "Color"


def default_input_root() -> Path:
    return default_gray_input_root()


def default_gray_output_root() -> Path:
    gray_root = project_root() / "Result" / "OutputImage" / "Gray"
    if gray_root.exists():
        return gray_root
    return project_root() / "Result" / "OutputImage"


def default_dc_root() -> Path:
    candidates = [
        default_gray_output_root() / "DcEncryption",
        project_root() / "Result" / "OutputImage" / "DcEncryption",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_ac_root() -> Path:
    candidates = [
        default_gray_output_root() / "AcEncryption",
        project_root() / "Result" / "OutputImage" / "AcEncryption",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def default_color_output_root() -> Path:
    candidates = [
        project_root() / "Result" / "OutputImage" / "Color",
        project_root() / "Result" / "OutputImageColor",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def build_rho_grid(step: float = 0.05) -> np.ndarray:
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    count = int(round(1.0 / step))
    return np.linspace(0.0, 1.0, count + 1)


def collect_dc_image_strength_files(dc_root: Path) -> Dict[int, Dict[str, Dict[int, Path]]]:
    if not dc_root.exists():
        raise FileNotFoundError(f"DC result folder does not exist: {dc_root}")

    output: Dict[int, Dict[str, Dict[int, Path]]] = {}
    for qf_dir in sorted(dc_root.iterdir(), key=lambda p: p.name):
        if not qf_dir.is_dir():
            continue
        qf_match = QF_PATTERN.match(qf_dir.name)
        if not qf_match:
            continue

        qf = int(qf_match.group(1))
        image_map: Dict[str, Dict[int, Path]] = {}

        for image_dir in sorted(qf_dir.iterdir(), key=lambda p: p.name.lower()):
            if not image_dir.is_dir():
                continue

            strength_map: Dict[int, Path] = {}
            for candidate in image_dir.iterdir():
                if not candidate.is_file():
                    continue
                strength_match = STRENGTH_IMAGE_PATTERN.match(candidate.name)
                if not strength_match:
                    continue
                strength = int(strength_match.group(1))
                strength_map[strength] = candidate

            if strength_map:
                image_map[image_dir.name] = strength_map

        if image_map:
            output[qf] = image_map

    return output


def collect_ac_image_strength_files(
    ac_root: Path,
    input_root: Path | None = None,
) -> Dict[int, Dict[str, Dict[int, Path]]]:
    if not ac_root.exists():
        raise FileNotFoundError(f"AC result folder does not exist: {ac_root}")

    stem_to_source: Dict[str, str] = {}
    if input_root is not None and input_root.exists():
        for source_path in input_root.iterdir():
            if source_path.is_file():
                stem_to_source[source_path.stem.lower()] = source_path.name

    output: Dict[int, Dict[str, Dict[int, Path]]] = {}
    for qf_dir in sorted(ac_root.iterdir(), key=lambda p: p.name):
        if not qf_dir.is_dir():
            continue
        qf_match = QF_PATTERN.match(qf_dir.name)
        if not qf_match:
            continue

        qf = int(qf_match.group(1))
        image_map: Dict[str, Dict[int, Path]] = {}

        for st_dir in sorted(qf_dir.iterdir(), key=lambda p: p.name):
            if not st_dir.is_dir() or not st_dir.name.startswith("ST="):
                continue

            try:
                strength = int(st_dir.name.split("=")[1])
            except (IndexError, ValueError):
                continue

            for image_file in st_dir.iterdir():
                if not image_file.is_file() or image_file.suffix.lower() != ".jpg":
                    continue

                source_name = stem_to_source.get(image_file.stem.lower(), image_file.name)
                image_map.setdefault(source_name, {})[strength] = image_file

        if image_map:
            output[qf] = image_map

    return output


def build_curves_from_metric(
    files_by_qf: Dict[int, Dict[str, Dict[int, Path]]],
    metric_fn: Callable[[Path, str], float],
) -> Dict[int, Dict[str, MetricCurve]]:
    curves: Dict[int, Dict[str, MetricCurve]] = {}

    for qf, image_map in files_by_qf.items():
        curves[qf] = {}
        for image_name, strength_map in image_map.items():
            strengths = np.array(sorted(strength_map.keys()), dtype=np.float64)
            if strengths.size == 0:
                continue

            max_strength = strengths[-1]
            if max_strength <= 0:
                continue

            values = np.array(
                [metric_fn(strength_map[int(strength)], image_name) for strength in strengths],
                dtype=np.float64,
            )
            rhos = strengths / max_strength

            curves[qf][image_name] = MetricCurve(
                image_name=image_name,
                strengths=strengths,
                rhos=rhos,
                values=values,
            )

    return curves


def align_curves(
    curves: Dict[str, MetricCurve],
    rho_grid: np.ndarray,
    min_coverage: float = 0.7,
) -> AlignedStats:
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")

    curve_list = list(curves.values())
    n_curve = len(curve_list)
    if n_curve == 0:
        empty = np.array([], dtype=np.float64)
        return AlignedStats(empty, empty, empty, empty, empty.astype(np.int64))

    interpolated = np.full((n_curve, rho_grid.size), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n_curve, rho_grid.size), dtype=bool)

    for idx, curve in enumerate(curve_list):
        rho_min = curve.rhos[0]
        rho_max = curve.rhos[-1]
        valid = (rho_grid >= rho_min) & (rho_grid <= rho_max)
        if not np.any(valid):
            continue

        interpolated[idx, valid] = np.interp(rho_grid[valid], curve.rhos, curve.values)
        valid_mask[idx, valid] = True

    sample_count = valid_mask.sum(axis=0)
    coverage = sample_count / float(n_curve)

    mean = np.full(rho_grid.shape, np.nan, dtype=np.float64)
    std = np.full(rho_grid.shape, np.nan, dtype=np.float64)
    valid_cols = sample_count > 0
    if np.any(valid_cols):
        with np.errstate(invalid="ignore"):
            mean[valid_cols] = np.nanmean(interpolated[:, valid_cols], axis=0)
            std[valid_cols] = np.nanstd(interpolated[:, valid_cols], axis=0)

    keep = coverage >= min_coverage
    return AlignedStats(
        rho=rho_grid[keep],
        mean=mean[keep],
        std=std[keep],
        coverage=coverage[keep],
        sample_count=sample_count[keep],
    )


def build_source_lookup(input_root: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in input_root.iterdir():
        if path.is_file():
            lookup[path.name.lower()] = path
    return lookup


def calculate_bpp(image_path: Path) -> float:
    with Image.open(image_path) as image:
        width, height = image.size
    bits = image_path.stat().st_size * 8
    return bits / float(width * height)


def calculate_difference_value(image_path: Path) -> float:
    image = np.array(Image.open(image_path), dtype=np.int64)
    horizontal = np.abs(image[:, :-1] - image[:, 1:]).sum()
    vertical = np.abs(image[:-1, :] - image[1:, :]).sum()
    return float(horizontal + vertical)


def calculate_psnr_ssim(source_path: Path, cipher_path: Path) -> tuple[float, float]:
    from skimage import io
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    source = io.imread(source_path)
    cipher = io.imread(cipher_path)

    psnr = peak_signal_noise_ratio(cipher, source, data_range=255)
    if source.ndim == 2:
        ssim = structural_similarity(cipher, source, data_range=255)
    else:
        ssim = structural_similarity(cipher, source, data_range=255, channel_axis=-1)
    return float(psnr), float(ssim)
