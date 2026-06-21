"""Run encode scaling benchmarks for before/after tokenizer versions.

This wrapper executes benchmark/run_benchmark.py for several input sizes and
aggregates the generated CSV files into scaling summary artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIZES = [1_000, 2_000, 5_000, 10_000]
DEFAULT_REPEAT_THRESHOLD = 2_000


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def parse_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive comma-separated integers")
    return sizes


def size_label(size: int) -> str:
    if size % 1_000 == 0:
        return f"{size // 1_000}k"
    return str(size)


def load_language_split(corpus_dir: Path) -> tuple[int, int]:
    manifest_path = corpus_dir / "manifest.json"
    if not manifest_path.exists():
        return 1, 1

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    ko_docs = 0
    en_docs = 0
    for row in manifest.get("stats", []):
        if row.get("lang") == "ko":
            ko_docs = int(row.get("documents", 0) or 0)
        elif row.get("lang") == "en":
            en_docs = int(row.get("documents", 0) or 0)

    if ko_docs <= 0 or en_docs <= 0:
        return 1, 1
    return ko_docs, en_docs


def split_total_documents(total: int, ko_total: int, en_total: int) -> tuple[int, int]:
    ko_docs = round(total * ko_total / (ko_total + en_total))
    ko_docs = max(1, min(total - 1, ko_docs))
    return ko_docs, total - ko_docs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=parse_sizes,
        default=DEFAULT_SIZES,
        help="Comma-separated total document counts. Default: 1000,2000,5000,10000.",
    )
    parser.add_argument(
        "--repeat-threshold",
        type=positive_int,
        default=DEFAULT_REPEAT_THRESHOLD,
        help="Sizes up to this value use --small-repeats; larger sizes use --large-repeats.",
    )
    parser.add_argument("--small-repeats", type=positive_int, default=5)
    parser.add_argument("--large-repeats", type=positive_int, default=1)
    parser.add_argument("--warmup-runs", type=nonnegative_int, default=0)
    parser.add_argument("--max-doc-chars", type=nonnegative_int, default=1000)
    parser.add_argument("--progress-every-docs", type=nonnegative_int, default=500)
    parser.add_argument("--n-workers", type=positive_int, default=1)
    parser.add_argument("--corpus-dir", type=Path, default=ROOT / "data" / "public_corpus")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_encode_row(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["task"] == "encode":
                return row
    raise ValueError(f"encode row not found: {path}")


def as_float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key) or 0.0)


def aggregate_results(results_dir: Path, runs: list[dict[str, Any]]) -> None:
    detail_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    rows_by_size: dict[int, dict[str, dict[str, Any]]] = {}
    for run in runs:
        row = read_encode_row(run["csv_path"])
        detail = {
            "size_label": run["size_label"],
            "total_documents_requested": run["total_documents"],
            "ko_documents_requested": run["ko_documents"],
            "en_documents_requested": run["en_documents"],
            "version": run["version"],
            "repeats": int(float(row["repeats"])),
            "warmup_runs": int(float(row["warmup_runs"])),
            "documents": int(float(row["documents"])),
            "elapsed_sec_mean": as_float(row, "elapsed_sec_mean"),
            "elapsed_sec_std": as_float(row, "elapsed_sec_std"),
            "elapsed_sec_min": as_float(row, "elapsed_sec_min"),
            "elapsed_sec_max": as_float(row, "elapsed_sec_max"),
            "docs_per_sec": as_float(row, "docs_per_sec"),
            "tokens_per_sec": as_float(row, "tokens_per_sec"),
            "output_tokens": int(float(row["output_tokens"] or 0)),
            "peak_rss_mb_mean": as_float(row, "peak_rss_mb_mean"),
            "peak_rss_mb_max": as_float(row, "peak_rss_mb_max"),
            "cache_hits": int(float(row["cache_hits"] or 0)),
            "cache_misses": int(float(row["cache_misses"] or 0)),
            "cache_size": int(float(row["cache_size"] or 0)),
            "source_csv": run["csv_path"].relative_to(ROOT).as_posix(),
            "source_json": run["json_path"].relative_to(ROOT).as_posix(),
        }
        detail_rows.append(detail)
        rows_by_size.setdefault(run["total_documents"], {})[run["version"]] = detail

    for size in sorted(rows_by_size):
        before = rows_by_size[size].get("before")
        after = rows_by_size[size].get("after")
        if before is None or after is None:
            continue
        before_elapsed = before["elapsed_sec_mean"]
        after_elapsed = after["elapsed_sec_mean"]
        comparison_rows.append(
            {
                "size_label": before["size_label"],
                "documents": before["documents"],
                "before_elapsed_sec_mean": before_elapsed,
                "before_elapsed_sec_std": before["elapsed_sec_std"],
                "after_elapsed_sec_mean": after_elapsed,
                "after_elapsed_sec_std": after["elapsed_sec_std"],
                "speedup_x": before_elapsed / after_elapsed if after_elapsed else 0.0,
                "before_docs_per_sec": before["docs_per_sec"],
                "after_docs_per_sec": after["docs_per_sec"],
                "before_tokens_per_sec": before["tokens_per_sec"],
                "after_tokens_per_sec": after["tokens_per_sec"],
                "before_peak_rss_mb_mean": before["peak_rss_mb_mean"],
                "after_peak_rss_mb_mean": after["peak_rss_mb_mean"],
                "after_cache_hits": after["cache_hits"],
                "after_cache_misses": after["cache_misses"],
                "after_cache_size": after["cache_size"],
            }
        )

    detail_csv = results_dir / "encode_scaling_details.csv"
    comparison_csv = results_dir / "encode_scaling_summary.csv"
    summary_json = results_dir / "encode_scaling_summary.json"

    if detail_rows:
        with detail_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            writer.writeheader()
            writer.writerows(detail_rows)

    if comparison_rows:
        with comparison_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_rows[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_rows)

    payload = {
        "description": "Encode scaling benchmark across input sizes.",
        "details_csv": detail_csv.relative_to(ROOT).as_posix(),
        "summary_csv": comparison_csv.relative_to(ROOT).as_posix(),
        "sizes": comparison_rows,
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {detail_csv.relative_to(ROOT).as_posix()}", flush=True)
    print(f"Wrote {comparison_csv.relative_to(ROOT).as_posix()}", flush=True)
    print(f"Wrote {summary_json.relative_to(ROOT).as_posix()}", flush=True)


def main() -> None:
    args = parse_args()
    corpus_dir = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    results_dir = args.results_dir if args.results_dir.is_absolute() else ROOT / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    ko_total, en_total = load_language_split(corpus_dir)
    runs: list[dict[str, Any]] = []

    for total_docs in args.sizes:
        ko_docs, en_docs = split_total_documents(total_docs, ko_total, en_total)
        repeats = args.small_repeats if total_docs <= args.repeat_threshold else args.large_repeats
        label = size_label(total_docs)

        for version in ("before", "after"):
            csv_path = results_dir / f"scaling_{version}_{label}.csv"
            json_path = results_dir / f"scaling_{version}_{label}.json"
            cmd = [
                sys.executable,
                "benchmark/run_benchmark.py",
                "--version",
                version,
                "--tasks",
                "encode",
                "--ko-documents",
                str(ko_docs),
                "--en-documents",
                str(en_docs),
                "--repeats",
                str(repeats),
                "--warmup-runs",
                str(args.warmup_runs),
                "--max-doc-chars",
                str(args.max_doc_chars),
                "--progress-every-docs",
                str(args.progress_every_docs),
                "--n-workers",
                str(args.n_workers),
                "--clear-cache-before-run",
                "--output-csv",
                str(csv_path.relative_to(ROOT)),
                "--output-json",
                str(json_path.relative_to(ROOT)),
            ]
            run_command(cmd, args.dry_run)
            runs.append(
                {
                    "version": version,
                    "size_label": label,
                    "total_documents": total_docs,
                    "ko_documents": ko_docs,
                    "en_documents": en_docs,
                    "csv_path": csv_path,
                    "json_path": json_path,
                }
            )

    if not args.dry_run:
        aggregate_results(results_dir, runs)


if __name__ == "__main__":
    main()
