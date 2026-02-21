from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRIC_MAP = {
	"psnr": "mean_psnr",
	"ssim": "mean_ssim",
	"bpp": "mean_bpp",
}

YLABEL_MAP = {
	"psnr": "PSNR",
	"ssim": "SSIM",
	"bpp": "mean BPP",
}

QF_STYLES = {
	30: ("color30", "*"),
	50: ("color50", "o"),
	70: ("color70", "asterisk"),
	90: ("color90", "x"),
}


def parse_list(value: str) -> list[str]:
	return [token.strip() for token in value.split(",") if token.strip()]


def log_info(message: str) -> None:
	print(f"[INFO] {message}")


def load_rows(path: Path) -> list[dict]:
	with path.open("r", encoding="utf-8", newline="") as f:
		return list(csv.DictReader(f))


def safe_float(value: str) -> float:
	return float(value)


def build_tex_for_group(rows: list[dict], metric: str, variant: str, mode: str) -> str:
	metric_key = METRIC_MAP[metric]
	ylabel = YLABEL_MAP[metric]

	rows_sorted = sorted(rows, key=lambda r: (int(r["qf"]), safe_float(r["rho"])))

	y_values = [safe_float(r[metric_key]) for r in rows_sorted]
	y_min = min(y_values)
	y_max = max(y_values)
	if y_min == y_max:
		y_max = y_min + 1.0

	# add margins for readability
	margin = 0.06 * (y_max - y_min)
	y_min -= margin
	y_max += margin

	qf_to_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
	for row in rows_sorted:
		qf = int(row["qf"])
		qf_to_points[qf].append((safe_float(row["rho"]), safe_float(row[metric_key])))

	legend_anchor = "(0.02,0.98)"

	lines: list[str] = []
	lines.append(r"\begin{tikzpicture}[scale=0.75]")
	lines.append(r"\begin{axis}[")
	lines.append(r"xlabel={$\rho$},")
	lines.append(f"ylabel={{{ylabel}}},")
	lines.append(r"xmin=0, xmax=1,")
	lines.append(r"xtick={0,0.1,...,1},")
	lines.append(f"ymin={y_min:.4f}, ymax={y_max:.4f},")
	lines.append(r"xmajorgrids, ymajorgrids,")
	lines.append(f"legend style={{at={legend_anchor}, anchor=north west}},")
	lines.append(r"axis background/.style={")
	lines.append(r"    shade,")
	lines.append(r"    top color=white,")
	lines.append(r"    bottom color=cyan!20!gray!20,")
	lines.append(r"}")
	lines.append(r"]")
	lines.append("")
	lines.append(r"% Auto-generated from all_rho_summary.csv")
	lines.append(f"% variant={variant}, mode={mode}, metric={metric}")
	lines.append(r"\definecolor{color30}{rgb}{1,0,0}")
	lines.append(r"\definecolor{color50}{rgb}{0.8,0.4,0}")
	lines.append(r"\definecolor{color70}{rgb}{0,0,1}")
	lines.append(r"\definecolor{color90}{rgb}{0.5,0,0.5}")
	lines.append("")

	for qf in sorted(qf_to_points.keys()):
		color, mark = QF_STYLES.get(qf, ("black", "*"))
		lines.append(fr"\addplot [{color}, mark={mark}] table[row sep=\\]{{")
		for rho, value in qf_to_points[qf]:
			lines.append(f"{rho:.6f} {value:.8f} \\\\")
		lines.append(r"};")
		lines.append(fr"\addlegendentry{{$QF = {qf}$}}")
		lines.append("")

	lines.append(r"\end{axis}")
	lines.append(r"\end{tikzpicture}")
	return "\n".join(lines) + "\n"


def main() -> None:
	parser = argparse.ArgumentParser(description="Convert all_rho_summary.csv to LaTeX pgfplots snippets.")
	parser.add_argument(
		"--input-csv",
		type=Path,
		default=Path("F:/Expriment/JpegPE/Result/AnalysisUnified/QualityCompression/all_rho_summary.csv"),
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("F:/Expriment/JpegPE/Result/AnalysisUnified/QualityCompression/latex"),
	)
	parser.add_argument("--metrics", type=str, default="psnr,bpp", help="psnr,ssim,bpp")
	parser.add_argument("--modes", type=str, default="ac,dc", help="ac,dc")
	parser.add_argument("--variants", type=str, default="", help="Optional variant filter, comma separated")
	args = parser.parse_args()

	metrics = [m.lower() for m in parse_list(args.metrics)]
	for m in metrics:
		if m not in METRIC_MAP:
			raise ValueError(f"Unsupported metric: {m}")

	modes = [m.lower() for m in parse_list(args.modes)]
	for m in modes:
		if m not in {"ac", "dc"}:
			raise ValueError(f"Unsupported mode: {m}")

	variant_filter = {v.lower() for v in parse_list(args.variants)}

	rows = load_rows(args.input_csv)
	if not rows:
		raise ValueError(f"No rows found in {args.input_csv}")

	grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
	for row in rows:
		variant = row["variant"].strip()
		mode = row["mode"].strip().lower()
		if mode not in modes:
			continue
		if variant_filter and variant.lower() not in variant_filter:
			continue
		grouped[(variant, mode)].append(row)

	if not grouped:
		raise ValueError("No data matched the provided filters.")

	args.output_dir.mkdir(parents=True, exist_ok=True)
	for (variant, mode), group_rows in sorted(grouped.items()):
		for metric in metrics:
			tex = build_tex_for_group(group_rows, metric, variant, mode)
			out_path = args.output_dir / f"{variant}_{mode}_{metric}.tex"
			out_path.write_text(tex, encoding="utf-8")
			log_info(f"Wrote {out_path}")


if __name__ == "__main__":
	main()
