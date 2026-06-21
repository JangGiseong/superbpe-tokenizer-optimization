"""Generate encode-scaling figures from benchmark results.

Reads results/encode_scaling_summary.csv and writes report-ready PNG figures
into results/:

- encode_scaling_time.png      : input size vs elapsed time (log-log, before/after)
- encode_scaling_speedup.png   : speedup factor per input size
- encode_scaling_memory.png    : peak RSS per input size (before/after)
- encode_scaling_cache.png     : cache hit ratio per input size
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_CSV = ROOT / "results" / "encode_scaling_summary.csv"
HEADLINE_CSV = ROOT / "results" / "benchmark_results.csv"
RESULTS_DIR = ROOT / "results"

BEFORE_COLOR = "#c0392b"
AFTER_COLOR = "#1f77b4"


def load_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, float | str] = {"size_label": row["size_label"]}
            for key, value in row.items():
                if key == "size_label":
                    continue
                parsed[key] = float(value) if value not in (None, "") else 0.0
            rows.append(parsed)
    rows.sort(key=lambda r: r["documents"])
    return rows


def append_headline(rows: list[dict], path: Path) -> list[dict]:
    """Append the 5% (50,274-doc) encode headline point so the scaling curve
    includes the flagship result. Single run (repeats=1) -> std = 0."""
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        encode = next((r for r in csv.DictReader(f) if r["task"] == "encode"), None)
    if encode is None:
        return rows
    docs = float(encode["documents"])
    if any(r["documents"] == docs for r in rows):
        return rows
    label = f"{int(round(docs / 1000))}k"
    rows.append({
        "size_label": label,
        "documents": docs,
        "before_elapsed_sec_mean": float(encode["before_elapsed_sec"]),
        "before_elapsed_sec_std": 0.0,
        "after_elapsed_sec_mean": float(encode["after_elapsed_sec"]),
        "after_elapsed_sec_std": 0.0,
        "speedup_x": float(encode["speedup_x"]),
        "before_docs_per_sec": float(encode["before_docs_per_sec"]),
        "after_docs_per_sec": float(encode["after_docs_per_sec"]),
        "before_tokens_per_sec": float(encode["before_tokens_per_sec"]),
        "after_tokens_per_sec": float(encode["after_tokens_per_sec"]),
        "before_peak_rss_mb_mean": float(encode["before_peak_rss_mb"]),
        "after_peak_rss_mb_mean": float(encode["after_peak_rss_mb"]),
        "after_cache_hits": float(encode["after_cache_hits"]),
        "after_cache_misses": float(encode["after_cache_misses"]),
        "after_cache_size": float(encode["after_cache_size"]),
    })
    rows.sort(key=lambda r: r["documents"])
    return rows


def plot_time(rows: list[dict], out: Path) -> None:
    docs = [r["documents"] for r in rows]
    before = [r["before_elapsed_sec_mean"] for r in rows]
    after = [r["after_elapsed_sec_mean"] for r in rows]
    before_std = [r["before_elapsed_sec_std"] for r in rows]
    after_std = [r["after_elapsed_sec_std"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        docs, before, yerr=before_std, marker="o", capsize=4,
        color=BEFORE_COLOR, label="before (full pair re-scan)",
    )
    ax.errorbar(
        docs, after, yerr=after_std, marker="s", capsize=4,
        color=AFTER_COLOR, label="after (heap + linked-list + cache)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Documents (log scale)")
    ax.set_ylabel("Encode elapsed time [s] (log scale)")
    ax.set_title("Encode time vs input size")
    ax.set_xticks(docs)
    ax.set_xticklabels([r["size_label"] for r in rows])
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_speedup(rows: list[dict], out: Path) -> None:
    labels = [r["size_label"] for r in rows]
    speedup = [r["speedup_x"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, speedup, color=AFTER_COLOR, width=0.55)
    for bar, value in zip(bars, speedup):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value,
            f"{value:.1f}x", ha="center", va="bottom",
        )
    ax.set_xlabel("Input size")
    ax.set_ylabel("Speedup (before / after)")
    ax.set_title("Encode speedup by input size")
    ax.set_ylim(0, max(speedup) * 1.15)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_memory(rows: list[dict], out: Path) -> None:
    labels = [r["size_label"] for r in rows]
    before = [r["before_peak_rss_mb_mean"] for r in rows]
    after = [r["after_peak_rss_mb_mean"] for r in rows]
    x = range(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar([i - width / 2 for i in x], before, width, color=BEFORE_COLOR, label="before")
    ax.bar([i + width / 2 for i in x], after, width, color=AFTER_COLOR, label="after")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Input size")
    ax.set_ylabel("Peak RSS [MB]")
    ax.set_title("Peak memory vs input size (cache trade-off)")
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cache(rows: list[dict], out: Path) -> None:
    labels = [r["size_label"] for r in rows]
    ratios = []
    for r in rows:
        hits = r["after_cache_hits"]
        misses = r["after_cache_misses"]
        total = hits + misses
        ratios.append(100.0 * hits / total if total else 0.0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(labels, ratios, marker="o", color=AFTER_COLOR)
    for label, ratio in zip(labels, ratios):
        ax.text(label, ratio, f"{ratio:.1f}%", ha="center", va="bottom")
    ax.set_xlabel("Input size")
    ax.set_ylabel("Cache hit ratio [%]")
    ax.set_title("lru_cache hit ratio vs input size")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"{SUMMARY_CSV} not found. Run scripts/run_encode_scaling_benchmark.py first."
        )
    rows = load_rows(SUMMARY_CSV)
    rows = append_headline(rows, HEADLINE_CSV)
    plot_time(rows, RESULTS_DIR / "encode_scaling_time.png")
    plot_speedup(rows, RESULTS_DIR / "encode_scaling_speedup.png")
    plot_memory(rows, RESULTS_DIR / "encode_scaling_memory.png")
    plot_cache(rows, RESULTS_DIR / "encode_scaling_cache.png")
    for name in (
        "encode_scaling_time.png",
        "encode_scaling_speedup.png",
        "encode_scaling_memory.png",
        "encode_scaling_cache.png",
    ):
        print(f"Wrote results/{name}", flush=True)


if __name__ == "__main__":
    main()
