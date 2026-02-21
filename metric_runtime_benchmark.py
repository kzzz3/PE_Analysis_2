from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from shared_strength_alignment import (
    default_color_input_root,
    default_gray_input_root,
    project_root,
)


@dataclass
class MethodConfig:
    name: str
    workdir: Path
    encrypt_cmd: str
    decrypt_cmd: str
    enabled: bool = True


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warn(message: str) -> None:
    print(f"[WARN] {message}")


def parse_token_list(value: str) -> list[str]:
    items: list[str] = []
    for token in value.split(","):
        token = token.strip()
        if token:
            items.append(token)
    return items


def parse_resolution_list(value: str) -> list[tuple[int, int]]:
    results: list[tuple[int, int]] = []
    for token in parse_token_list(value):
        if "x" not in token.lower():
            raise ValueError(f"Invalid resolution token: {token}")
        w_text, h_text = token.lower().split("x", 1)
        width = int(w_text)
        height = int(h_text)
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid resolution value: {token}")
        results.append((width, height))
    if not results:
        raise ValueError("No valid resolution is provided")
    return results


def seed_images(input_root: Path, dataset: str, limit: int) -> list[Path]:
    if not input_root.exists():
        return []

    exts = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    files = [
        path
        for path in sorted(input_root.iterdir(), key=lambda p: p.name.lower())
        if path.is_file() and path.suffix.lower() in exts
    ]

    if dataset == "gray":
        # Prefer common grayscale-name hints when available.
        preferred = [p for p in files if "gray" in p.stem.lower()]
        if preferred:
            files = preferred + [p for p in files if p not in preferred]

    return files[: max(1, limit)]


def build_resized_inputs(
    source_images: list[Path],
    output_dir: Path,
    width: int,
    height: int,
    dataset: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    bicubic = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else getattr(Image, "BICUBIC", 3)
    for src in source_images:
        with Image.open(src) as img:
            if dataset == "gray":
                resized = img.convert("L").resize((width, height), bicubic)
                out_name = f"{src.stem}_{width}x{height}.bmp"
            else:
                resized = img.convert("RGB").resize((width, height), bicubic)
                out_name = f"{src.stem}_{width}x{height}.png"
            resized.save(output_dir / out_name)


def default_method_template() -> dict[str, Any]:
    return {
        "methods": [
            {
                "name": "proposed_pe",
                "enabled": True,
                "workdir": "F:/Expriment/JpegPE/PE",
                "encrypt_cmd": (
                    "build/global-x64/Release/PE.exe "
                    "--mode both --threads {threads} "
                    "--input \"{input_dir}\" --output \"{output_dir}\" "
                    "--qf {qf} --ac-st {ac_st} --subsampling {subsampling} "
                    "--dc-max {dc_max} --iwt-level {iwt_level} --verify-reversible 0"
                ),
                "decrypt_cmd": "",
                "note": "Set PE.exe path according to your local build output.",
            },
            {
                "name": "related_be",
                "enabled": False,
                "workdir": "F:/Expriment/JpegPE/RelatedWrok/BE/PE/PE",
                "encrypt_cmd": "x64/Release/PE.exe",
                "decrypt_cmd": "",
                "note": "Enable after adapting related-work executable and runtime arguments.",
            },
            {
                "name": "related_se",
                "enabled": False,
                "workdir": "F:/Expriment/JpegPE/RelatedWrok/SE/PE/PE",
                "encrypt_cmd": "x64/Release/PE.exe",
                "decrypt_cmd": "",
                "note": "Enable after adapting related-work executable and runtime arguments.",
            },
        ]
    }


def load_methods(config_path: Path) -> list[MethodConfig]:
    if not config_path.exists():
        config_path.write_text(json.dumps(default_method_template(), indent=2), encoding="utf-8")
        raise FileNotFoundError(
            f"Method config is created at: {config_path}. "
            "Please edit command paths/options, then rerun."
        )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    methods: list[MethodConfig] = []
    for item in config.get("methods", []):
        methods.append(
            MethodConfig(
                name=str(item.get("name", "unknown")),
                workdir=Path(str(item.get("workdir", "."))),
                encrypt_cmd=str(item.get("encrypt_cmd", "")).strip(),
                decrypt_cmd=str(item.get("decrypt_cmd", "")).strip(),
                enabled=bool(item.get("enabled", True)),
            )
        )
    return [m for m in methods if m.enabled and m.encrypt_cmd]


def render_command(template: str, context: dict[str, Any]) -> str:
    rendered = template
    for key, value in context.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


def run_command(command: str, cwd: Path, timeout_sec: int) -> float:
    start = time.perf_counter()
    subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=timeout_sec,
    )
    return time.perf_counter() - start


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def write_csv(path: Path, rows: list[dict[str, Any]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int, int], list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row["status"] != "ok":
            continue
        key = (
            row["method"],
            row["dataset"],
            row["subsampling"],
            int(row["width"]),
            int(row["height"]),
            int(row["qf"]),
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        enc_values = [float(r["encrypt_sec"]) for r in rows if r["encrypt_sec"] != ""]
        dec_values = [float(r["decrypt_sec"]) for r in rows if r["decrypt_sec"] != ""]
        tot_values = [float(r["total_sec"]) for r in rows if r["total_sec"] != ""]
        enc_mean, enc_std = mean_std(enc_values)
        dec_mean, dec_std = mean_std(dec_values)
        tot_mean, tot_std = mean_std(tot_values)

        summary_rows.append(
            {
                "method": key[0],
                "dataset": key[1],
                "subsampling": key[2],
                "width": key[3],
                "height": key[4],
                "qf": key[5],
                "runs": len(rows),
                "encrypt_mean_sec": f"{enc_mean:.6f}",
                "encrypt_std_sec": f"{enc_std:.6f}",
                "decrypt_mean_sec": "" if not dec_values else f"{dec_mean:.6f}",
                "decrypt_std_sec": "" if not dec_values else f"{dec_std:.6f}",
                "total_mean_sec": f"{tot_mean:.6f}",
                "total_std_sec": f"{tot_std:.6f}",
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Runtime benchmark for proposed and related JPEG-encryption methods.")
    parser.add_argument("--datasets", type=str, default="color", help="gray,color,both")
    parser.add_argument("--resolutions", type=str, default="512x512,1024x1024,1920x1080")
    parser.add_argument("--subsampling", type=str, default="444,420", help="Used for color workloads")
    parser.add_argument("--qf", type=int, default=90)
    parser.add_argument("--st", type=int, default=63)
    parser.add_argument("--iwt-level", type=int, default=4)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--ac-st", type=str, default="1,2,3,5,7,10,15,20,30,40,50,60,64")
    parser.add_argument("--dc-max", type=int, default=1000)
    parser.add_argument("--images-per-resolution", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--keep-output", action="store_true")

    parser.add_argument("--gray-input-root", type=Path, default=default_gray_input_root())
    parser.add_argument("--color-input-root", type=Path, default=default_color_input_root())
    parser.add_argument("--analysis-root", type=Path, default=project_root() / "Result" / "AnalysisRuntime")
    parser.add_argument("--method-config", type=Path, default=Path(__file__).with_name("runtime_methods.json"))
    args = parser.parse_args()

    datasets = parse_token_list(args.datasets)
    if "both" in [d.lower() for d in datasets]:
        datasets = ["gray", "color"]
    datasets = [d.lower() for d in datasets]

    resolutions = parse_resolution_list(args.resolutions)
    methods = load_methods(args.method_config)
    if not methods:
        raise ValueError("No enabled method with encrypt_cmd is found in method config.")

    raw_rows: list[dict[str, Any]] = []
    workspace = args.analysis_root / "runtime_workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        if dataset not in {"gray", "color"}:
            raise ValueError(f"Unsupported dataset token: {dataset}")

        input_root = args.gray_input_root if dataset == "gray" else args.color_input_root
        seeds = seed_images(input_root, dataset, args.images_per_resolution)
        if not seeds:
            log_warn(f"Skip dataset={dataset}: no source image found in {input_root}")
            continue

        subsampling_list = ["-"] if dataset == "gray" else parse_token_list(args.subsampling)
        for width, height in resolutions:
            input_dir = workspace / "inputs" / dataset / f"{width}x{height}"
            if input_dir.exists():
                shutil.rmtree(input_dir)
            build_resized_inputs(seeds, input_dir, width, height, dataset)

            for subsampling in subsampling_list:
                for method in methods:
                    run_count = args.warmup + args.repeat
                    for idx in range(run_count):
                        is_warmup = idx < args.warmup
                        run_id = idx - args.warmup + 1
                        output_dir = workspace / "outputs" / method.name / dataset / f"{width}x{height}" / f"ss{subsampling}" / f"run{idx + 1}"
                        if output_dir.exists() and not args.keep_output:
                            shutil.rmtree(output_dir)
                        output_dir.mkdir(parents=True, exist_ok=True)

                        context = {
                            "input_dir": input_dir,
                            "output_dir": output_dir,
                            "dataset": dataset,
                            "width": width,
                            "height": height,
                            "resolution": f"{width}x{height}",
                            "qf": args.qf,
                            "subsampling": subsampling,
                            "st": args.st,
                            "iwt_level": args.iwt_level,
                            "threads": args.threads,
                            "ac_st": args.ac_st,
                            "dc_max": args.dc_max,
                        }

                        row = {
                            "method": method.name,
                            "dataset": dataset,
                            "subsampling": subsampling,
                            "width": width,
                            "height": height,
                            "qf": args.qf,
                            "st": args.st,
                            "iwt_level": args.iwt_level,
                            "run_index": "" if is_warmup else run_id,
                            "is_warmup": int(is_warmup),
                            "encrypt_sec": "",
                            "decrypt_sec": "",
                            "total_sec": "",
                            "status": "ok",
                            "error": "",
                        }

                        try:
                            enc_cmd = render_command(method.encrypt_cmd, context)
                            enc_sec = run_command(enc_cmd, method.workdir, args.timeout_sec)
                            dec_sec = 0.0
                            if method.decrypt_cmd:
                                dec_cmd = render_command(method.decrypt_cmd, context)
                                dec_sec = run_command(dec_cmd, method.workdir, args.timeout_sec)
                            total_sec = enc_sec + dec_sec
                            row["encrypt_sec"] = f"{enc_sec:.6f}"
                            if method.decrypt_cmd:
                                row["decrypt_sec"] = f"{dec_sec:.6f}"
                            row["total_sec"] = f"{total_sec:.6f}"
                        except Exception as exc:
                            row["status"] = "failed"
                            row["error"] = str(exc)
                            log_warn(
                                f"Failed: method={method.name}, dataset={dataset}, "
                                f"res={width}x{height}, ss={subsampling}, run={idx + 1}, err={exc}"
                            )

                        raw_rows.append(row)

    raw_path = args.analysis_root / "runtime_raw_runs.csv"
    summary_path = args.analysis_root / "runtime_summary.csv"
    write_csv(
        raw_path,
        raw_rows,
        headers=[
            "method",
            "dataset",
            "subsampling",
            "width",
            "height",
            "qf",
            "st",
            "iwt_level",
            "run_index",
            "is_warmup",
            "encrypt_sec",
            "decrypt_sec",
            "total_sec",
            "status",
            "error",
        ],
    )

    summary_rows = summarize(raw_rows)
    write_csv(
        summary_path,
        summary_rows,
        headers=[
            "method",
            "dataset",
            "subsampling",
            "width",
            "height",
            "qf",
            "runs",
            "encrypt_mean_sec",
            "encrypt_std_sec",
            "decrypt_mean_sec",
            "decrypt_std_sec",
            "total_mean_sec",
            "total_std_sec",
        ],
    )

    log_info(f"Wrote raw runtime rows: {raw_path}")
    log_info(f"Wrote runtime summary: {summary_path}")


if __name__ == "__main__":
    main()
