"""Run before/after SuperBPE benchmarks.

The benchmark records enough metadata for performance comparison:

- input corpus size and language mix,
- encode/decode throughput,
- chunk-count construction runtime and peak memory,
- Python/platform/environment details,
- repeat-level measurements for mean/std reporting.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import psutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_DIR = ROOT / "data" / "public_corpus"
DEFAULT_TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer_150k.json"
RESULTS_CSV = ROOT / "results" / "benchmark_results.csv"
RESULTS_JSON = ROOT / "results" / "benchmark_details.json"

MODULE_NAMES = [
    "tokenizer",
    "vocab",
    "pretokenizer",
    "bpe_trainer",
    "data_prep",
]


CSV_FIELDS = [
    "version",
    "task",
    "warmup_runs",
    "repeats",
    "documents",
    "ko_documents",
    "en_documents",
    "total_chars",
    "total_bytes",
    "original_total_chars",
    "original_total_bytes",
    "truncated_documents",
    "source_counted_tokens",
    "output_tokens",
    "elapsed_sec_mean",
    "elapsed_sec_std",
    "elapsed_sec_min",
    "elapsed_sec_max",
    "peak_memory_mb_mean",
    "peak_memory_mb_max",
    "peak_rss_mb_mean",
    "peak_rss_mb_max",
    "docs_per_sec",
    "tokens_per_sec",
    "unique_chunks",
    "cache_hits",
    "cache_misses",
    "cache_size",
    "final_vocab_size",
    "total_merges",
    "n_workers",
    "max_doc_chars",
    "python_version",
    "platform",
    "cpu_count",
    "tokenizer_path",
    "notes",
]


@dataclass(frozen=True)
class CorpusRecord:
    text: str
    lang: str
    source_dataset: str
    token_count: int
    original_chars: int
    original_bytes: int
    truncated: bool


@dataclass(frozen=True)
class BenchmarkConfig:
    versions: list[str]
    tasks: list[str]
    corpus_dir: Path
    tokenizer_path: Path
    max_documents_per_lang: int
    ko_documents: int | None
    en_documents: int | None
    max_doc_chars: int | None
    repeats: int
    warmup_runs: int
    n_workers: int
    chunk_count_documents_per_lang: int
    train_documents_per_lang: int
    train_vocab_size: int
    train_transition_point: int
    train_stage2_max_docs: int
    progress_every_docs: int
    clear_cache_before_run: bool
    output_csv: Path
    output_json: Path


@dataclass
class Measurement:
    elapsed_sec: float
    peak_memory_mb: float
    peak_rss_mb: float
    metrics: dict[str, Any]


class PeakRssSampler:
    """Sample current process and child-process RSS during a benchmark task."""

    def __init__(self, interval_sec: float = 0.02):
        self.interval_sec = interval_sec
        self.process = psutil.Process(os.getpid())
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _sample_once(self) -> None:
        total = 0
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except psutil.Error:
            pass
        for proc in processes:
            try:
                total += proc.memory_info().rss
            except psutil.Error:
                continue
        self.peak_bytes = max(self.peak_bytes, total)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            time.sleep(self.interval_sec)
        self._sample_once()

    def __enter__(self) -> "PeakRssSampler":
        self._sample_once()
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        self._thread.join()

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024 * 1024)


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


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def display_path(path: Path) -> str:
    """Return a GitHub-friendly project-relative path when possible."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=["before", "after", "both"],
        default="before",
        help="Which source tree to benchmark.",
    )
    parser.add_argument(
        "--tasks",
        default="encode,decode,chunk_count",
        help="Comma-separated tasks: encode,decode,chunk_count,train_small.",
    )
    parser.add_argument("--corpus-dir", type=Path, default=DEFAULT_CORPUS_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument(
        "--max-documents-per-lang",
        type=positive_int,
        default=1_000,
        help="Number of Korean and English documents loaded for encode/decode.",
    )
    parser.add_argument(
        "--ko-documents",
        type=positive_int,
        default=None,
        help="Korean document limit for encode/decode. Overrides --max-documents-per-lang.",
    )
    parser.add_argument(
        "--en-documents",
        type=positive_int,
        default=None,
        help="English document limit for encode/decode. Overrides --max-documents-per-lang.",
    )
    parser.add_argument(
        "--chunk-count-documents-per-lang",
        type=positive_int,
        default=500,
        help="Number of Korean and English documents used for chunk-count benchmark.",
    )
    parser.add_argument(
        "--train-documents-per-lang",
        type=positive_int,
        default=100,
        help="Number of Korean and English documents used for scaled training benchmark.",
    )
    parser.add_argument(
        "--train-vocab-size",
        type=positive_int,
        default=320,
        help="Small vocab target for scaled BPETrainer.train benchmark.",
    )
    parser.add_argument(
        "--train-transition-point",
        type=positive_int,
        default=288,
        help="Small Stage 1 -> Stage 2 transition point for scaled training benchmark.",
    )
    parser.add_argument(
        "--train-stage2-max-docs",
        type=positive_int,
        default=100,
        help="Stage 2 document cap for scaled training benchmark.",
    )
    parser.add_argument(
        "--max-doc-chars",
        type=nonnegative_int,
        default=2_000,
        help="Truncate documents to this many chars. Use 0 to disable truncation.",
    )
    parser.add_argument("--repeats", type=positive_int, default=3)
    parser.add_argument(
        "--warmup-runs",
        type=nonnegative_int,
        default=1,
        help="Unmeasured warm-up executions per task.",
    )
    parser.add_argument("--n-workers", type=positive_int, default=1)
    parser.add_argument(
        "--progress-every-docs",
        type=nonnegative_int,
        default=1_000,
        help="Print task progress every N documents. Use 0 to disable.",
    )
    parser.add_argument(
        "--clear-cache-before-run",
        action="store_true",
        help="Clear tokenizer cache before each warm-up and measured run when supported.",
    )
    parser.add_argument("--output-csv", type=Path, default=RESULTS_CSV)
    parser.add_argument("--output-json", type=Path, default=RESULTS_JSON)
    args = parser.parse_args()

    versions = ["before", "after"] if args.version == "both" else [args.version]
    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    valid_tasks = {"encode", "decode", "chunk_count", "train_small"}
    invalid = sorted(set(tasks) - valid_tasks)
    if invalid:
        parser.error(f"invalid tasks: {', '.join(invalid)}")

    max_doc_chars = None if args.max_doc_chars == 0 else args.max_doc_chars
    return BenchmarkConfig(
        versions=versions,
        tasks=tasks,
        corpus_dir=resolve_path(args.corpus_dir),
        tokenizer_path=resolve_path(args.tokenizer_path),
        max_documents_per_lang=args.max_documents_per_lang,
        ko_documents=args.ko_documents,
        en_documents=args.en_documents,
        max_doc_chars=max_doc_chars,
        repeats=args.repeats,
        warmup_runs=args.warmup_runs,
        n_workers=args.n_workers,
        chunk_count_documents_per_lang=args.chunk_count_documents_per_lang,
        train_documents_per_lang=args.train_documents_per_lang,
        train_vocab_size=args.train_vocab_size,
        train_transition_point=args.train_transition_point,
        train_stage2_max_docs=args.train_stage2_max_docs,
        progress_every_docs=args.progress_every_docs,
        clear_cache_before_run=args.clear_cache_before_run,
        output_csv=resolve_path(args.output_csv),
        output_json=resolve_path(args.output_json),
    )


def truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None:
        return text
    return text[:max_chars]


def read_gzip_jsonl(path: Path, lang: str, limit: int, max_chars: int | None) -> list[CorpusRecord]:
    records: list[CorpusRecord] = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            truncated = truncate_text(text, max_chars)
            records.append(
                CorpusRecord(
                    text=truncated,
                    lang=lang,
                    source_dataset=str(row.get("source_dataset", "")),
                    token_count=int(row.get("token_count", 0) or 0),
                    original_chars=len(text),
                    original_bytes=len(text.encode("utf-8")),
                    truncated=len(truncated) < len(text),
                )
            )
            if len(records) >= limit:
                break
    return records


def load_corpus(config: BenchmarkConfig) -> list[CorpusRecord]:
    ko_path = config.corpus_dir / "ko_kor_Hang.jsonl.gz"
    en_path = config.corpus_dir / "en_sample-10BT.jsonl.gz"
    missing = [str(path) for path in [ko_path, en_path] if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing corpus shard(s): "
            + ", ".join(missing)
            + ". Run scripts/prepare_public_corpus.py first."
        )

    ko_limit = config.ko_documents or config.max_documents_per_lang
    en_limit = config.en_documents or config.max_documents_per_lang
    ko_records = read_gzip_jsonl(ko_path, "ko", ko_limit, config.max_doc_chars)
    en_records = read_gzip_jsonl(en_path, "en", en_limit, config.max_doc_chars)

    mixed: list[CorpusRecord] = []
    for idx in range(max(len(ko_records), len(en_records))):
        if idx < len(ko_records):
            mixed.append(ko_records[idx])
        if idx < len(en_records):
            mixed.append(en_records[idx])
    return mixed


def corpus_summary(records: list[CorpusRecord]) -> dict[str, Any]:
    ko_docs = sum(1 for item in records if item.lang == "ko")
    en_docs = sum(1 for item in records if item.lang == "en")
    return {
        "documents": len(records),
        "ko_documents": ko_docs,
        "en_documents": en_docs,
        "total_chars": sum(len(item.text) for item in records),
        "total_bytes": sum(len(item.text.encode("utf-8")) for item in records),
        "original_total_chars": sum(item.original_chars for item in records),
        "original_total_bytes": sum(item.original_bytes for item in records),
        "truncated_documents": sum(1 for item in records if item.truncated),
        "source_counted_tokens": sum(item.token_count for item in records),
    }


def clear_target_modules() -> None:
    for name in MODULE_NAMES:
        sys.modules.pop(name, None)


def clear_source_paths() -> None:
    source_roots = {str(ROOT / "src" / "before"), str(ROOT / "src" / "after")}
    sys.path[:] = [item for item in sys.path if item not in source_roots]


def import_version_modules(version: str):
    source_dir = ROOT / "src" / version
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)

    clear_target_modules()
    clear_source_paths()
    sys.path.insert(0, str(source_dir))
    tokenizer_mod = importlib.import_module("tokenizer")
    vocab_mod = importlib.import_module("vocab")
    pretokenizer_mod = importlib.import_module("pretokenizer")
    bpe_trainer_mod = importlib.import_module("bpe_trainer")

    return tokenizer_mod, vocab_mod, pretokenizer_mod, bpe_trainer_mod


def lang_fn(text: str) -> str:
    return "ko" if any("\uac00" <= ch <= "\ud7a3" for ch in text[:500]) else "en"


def measure_once(fn: Callable[[], dict[str, Any]]) -> Measurement:
    tracemalloc.start()
    with PeakRssSampler() as rss_sampler:
        start = time.perf_counter()
        metrics = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Measurement(
        elapsed_sec=elapsed,
        peak_memory_mb=peak / (1024 * 1024),
        peak_rss_mb=rss_sampler.peak_mb,
        metrics=metrics,
    )


def summarize_measurements(
    version: str,
    task: str,
    measurements: list[Measurement],
    base: dict[str, Any],
    config: BenchmarkConfig,
    notes: str = "",
) -> dict[str, Any]:
    elapsed = [item.elapsed_sec for item in measurements]
    peaks = [item.peak_memory_mb for item in measurements]
    rss_peaks = [item.peak_rss_mb for item in measurements]
    first_metrics = measurements[0].metrics if measurements else {}
    output_tokens = int(first_metrics.get("output_tokens", 0) or 0)
    documents = int(base["documents"])

    elapsed_mean = statistics.fmean(elapsed) if elapsed else 0.0
    elapsed_std = statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0
    tokens_per_sec = output_tokens / elapsed_mean if elapsed_mean and output_tokens else 0.0
    docs_per_sec = documents / elapsed_mean if elapsed_mean and documents else 0.0

    row = {
        "version": version,
        "task": task,
        "warmup_runs": config.warmup_runs,
        "repeats": len(measurements),
        **base,
        "output_tokens": output_tokens,
        "elapsed_sec_mean": elapsed_mean,
        "elapsed_sec_std": elapsed_std,
        "elapsed_sec_min": min(elapsed) if elapsed else 0.0,
        "elapsed_sec_max": max(elapsed) if elapsed else 0.0,
        "peak_memory_mb_mean": statistics.fmean(peaks) if peaks else 0.0,
        "peak_memory_mb_max": max(peaks) if peaks else 0.0,
        "peak_rss_mb_mean": statistics.fmean(rss_peaks) if rss_peaks else 0.0,
        "peak_rss_mb_max": max(rss_peaks) if rss_peaks else 0.0,
        "docs_per_sec": docs_per_sec,
        "tokens_per_sec": tokens_per_sec,
        "unique_chunks": int(first_metrics.get("unique_chunks", 0) or 0),
        "cache_hits": int(first_metrics.get("cache_hits", 0) or 0),
        "cache_misses": int(first_metrics.get("cache_misses", 0) or 0),
        "cache_size": int(first_metrics.get("cache_size", 0) or 0),
        "final_vocab_size": int(first_metrics.get("final_vocab_size", 0) or 0),
        "total_merges": int(first_metrics.get("total_merges", 0) or 0),
        "n_workers": config.n_workers,
        "max_doc_chars": config.max_doc_chars if config.max_doc_chars is not None else 0,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count() or 0,
        "tokenizer_path": display_path(config.tokenizer_path),
        "notes": notes,
    }
    return row


def cache_info(tokenizer: Any) -> dict[str, int]:
    if hasattr(tokenizer, "cache_info"):
        info = tokenizer.cache_info()
        return {
            "hits": int(info.get("hits", 0) or 0),
            "misses": int(info.get("misses", 0) or 0),
            "size": int(info.get("size", 0) or 0),
        }
    return {"hits": 0, "misses": 0, "size": 0}


def maybe_print_progress(
    task: str,
    done: int,
    total: int,
    start: float,
    progress_every_docs: int,
) -> None:
    if not progress_every_docs:
        return
    if done != total and done % progress_every_docs != 0:
        return
    elapsed = time.perf_counter() - start
    docs_per_sec = done / elapsed if elapsed else 0.0
    print(
        f"    {task}: {done:,}/{total:,} docs "
        f"({docs_per_sec:.2f} docs/sec, elapsed={elapsed:.1f}s)",
        flush=True,
    )


def encode_task(
    tokenizer: Any,
    records: list[CorpusRecord],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    before_cache = cache_info(tokenizer)
    total_tokens = 0
    start = time.perf_counter()
    for idx, item in enumerate(records, start=1):
        ids = tokenizer.encode(item.text, lang=item.lang)
        total_tokens += len(ids)
        maybe_print_progress(
            "encode",
            idx,
            len(records),
            start,
            config.progress_every_docs,
        )
    after_cache = cache_info(tokenizer)
    return {
        "output_tokens": total_tokens,
        "cache_hits": after_cache["hits"] - before_cache["hits"],
        "cache_misses": after_cache["misses"] - before_cache["misses"],
        "cache_size": after_cache["size"],
    }


def precompute_encoded(tokenizer: Any, records: list[CorpusRecord]) -> list[tuple[list[int], str]]:
    return [(tokenizer.encode(item.text, lang=item.lang), item.text) for item in records]


def decode_task(
    tokenizer: Any,
    encoded: list[tuple[list[int], str]],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    total_tokens = 0
    total_chars = 0
    start = time.perf_counter()
    for idx, (ids, _) in enumerate(encoded, start=1):
        text = tokenizer.decode(ids, skip_special_tokens=False)
        total_tokens += len(ids)
        total_chars += len(text)
        maybe_print_progress(
            "decode",
            idx,
            len(encoded),
            start,
            config.progress_every_docs,
        )
    return {"output_tokens": total_tokens, "decoded_chars": total_chars}


def chunk_count_task(
    bpe_trainer_mod: Any,
    pretokenizer_mod: Any,
    records: list[CorpusRecord],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    pretok = pretokenizer_mod.Pretokenizer(
        stage=pretokenizer_mod.Stage.STAGE2,
        max_segment_len=150,
    )
    chunk_counts = bpe_trainer_mod.build_chunk_counts(
        (item.text for item in records),
        pretok,
        lang_fn=lang_fn,
        max_doc_len=config.max_doc_chars,
        n_workers=config.n_workers,
    )
    return {
        "unique_chunks": len(chunk_counts),
        "output_tokens": sum(chunk_counts.values()),
    }


def train_small_task(
    bpe_trainer_mod: Any,
    records: list[CorpusRecord],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    texts = [item.text for item in records]

    def texts_iter_factory():
        return iter(texts)

    trainer = bpe_trainer_mod.BPETrainer(
        vocab_size=config.train_vocab_size,
        transition_point=config.train_transition_point,
        log_interval=10_000,
    )
    trainer.train(
        texts_iter_factory=texts_iter_factory,
        lang_fn=lang_fn,
        max_doc_len=config.max_doc_chars,
        stage2_max_docs=config.train_stage2_max_docs,
        n_workers=config.n_workers,
    )
    return {
        "output_tokens": len(trainer.vocab),
        "final_vocab_size": len(trainer.vocab),
        "total_merges": len(trainer.merges),
    }


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def write_json(payload: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_top_level_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = {row["task"]: row for row in rows}
    encode = tasks.get("encode", {})
    decode = tasks.get("decode", {})
    chunk_count = tasks.get("chunk_count", {})
    train_small = tasks.get("train_small", {})

    def number(row: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            return float(row.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    return {
        "headline": {
            "version": rows[0]["version"] if rows else "",
            "documents": int(number(encode or chunk_count or train_small, "documents")),
            "ko_documents": int(number(encode or chunk_count or train_small, "ko_documents")),
            "en_documents": int(number(encode or chunk_count or train_small, "en_documents")),
            "max_doc_chars": int(number(encode or chunk_count or train_small, "max_doc_chars")),
            "total_chars": int(number(encode or chunk_count or train_small, "total_chars")),
            "total_bytes": int(number(encode or chunk_count or train_small, "total_bytes")),
            "original_total_chars": int(
                number(encode or chunk_count or train_small, "original_total_chars")
            ),
            "original_total_bytes": int(
                number(encode or chunk_count or train_small, "original_total_bytes")
            ),
            "truncated_documents": int(
                number(encode or chunk_count or train_small, "truncated_documents")
            ),
            "source_counted_tokens": int(
                number(encode or chunk_count or train_small, "source_counted_tokens")
            ),
            "repeats": int(number(encode or chunk_count or train_small, "repeats")),
            "warmup_runs": int(number(encode or chunk_count or train_small, "warmup_runs")),
        },
        "key_metrics": {
            "encode_elapsed_sec_mean": number(encode, "elapsed_sec_mean"),
            "encode_tokens_per_sec": number(encode, "tokens_per_sec"),
            "encode_cache_hits": int(number(encode, "cache_hits")),
            "encode_cache_misses": int(number(encode, "cache_misses")),
            "encode_cache_size": int(number(encode, "cache_size")),
            "decode_elapsed_sec_mean": number(decode, "elapsed_sec_mean"),
            "chunk_count_elapsed_sec_mean": number(chunk_count, "elapsed_sec_mean"),
            "chunk_count_unique_chunks": int(number(chunk_count, "unique_chunks")),
            "train_small_elapsed_sec_mean": number(train_small, "elapsed_sec_mean"),
            "train_small_final_vocab_size": int(number(train_small, "final_vocab_size")),
            "train_small_total_merges": int(number(train_small, "total_merges")),
            "max_peak_rss_mb": max(number(row, "peak_rss_mb_max") for row in rows) if rows else 0.0,
        },
        "interpretation": [
            "encode가 baseline의 주요 runtime 병목이다.",
            "decode는 encode에 비해 매우 작다.",
            "chunk_count는 tokenizer 학습/재학습 경로의 corpus scan 비용을 대표한다.",
        ],
    }


def manifest_summary(corpus_dir: Path) -> dict[str, Any] | None:
    path = corpus_dir / "manifest.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "target_tokens": data.get("target_tokens"),
        "actual_counted_tokens": data.get("actual_counted_tokens"),
        "actual_documents": data.get("actual_documents"),
        "actual_bytes_utf8": data.get("actual_bytes_utf8"),
        "sources": data.get("sources", []),
    }


def run_for_version(
    version: str,
    config: BenchmarkConfig,
    records: list[CorpusRecord],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenizer_mod, vocab_mod, pretokenizer_mod, bpe_trainer_mod = import_version_modules(version)
    vocab = vocab_mod.Vocabulary.load(str(config.tokenizer_path))
    tokenizer = tokenizer_mod.SuperBPETokenizer(vocab)

    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {"version": version, "tasks": {}}

    base_all = corpus_summary(records)
    chunk_records = records[: config.chunk_count_documents_per_lang * 2]
    train_records = records[: config.train_documents_per_lang * 2]
    base_chunk = corpus_summary(chunk_records)
    base_train = corpus_summary(train_records)

    encoded_cache: list[tuple[list[int], str]] | None = None

    for task in config.tasks:
        print(f"[{version}] task={task}", flush=True)
        task_records = (
            chunk_records
            if task == "chunk_count"
            else train_records
            if task == "train_small"
            else records
        )
        base = base_chunk if task == "chunk_count" else base_train if task == "train_small" else base_all
        measurements: list[Measurement] = []

        if task == "decode":
            encoded_cache = encoded_cache or precompute_encoded(tokenizer, records)

        def run_task_once() -> dict[str, Any]:
            if task == "encode":
                return encode_task(tokenizer, task_records, config)
            if task == "decode":
                assert encoded_cache is not None
                return decode_task(tokenizer, encoded_cache, config)
            if task == "chunk_count":
                return chunk_count_task(
                    bpe_trainer_mod,
                    pretokenizer_mod,
                    task_records,
                    config,
                )
            if task == "train_small":
                return train_small_task(
                    bpe_trainer_mod,
                    task_records,
                    config,
                )
            raise ValueError(task)

        for warmup_idx in range(config.warmup_runs):
            if config.clear_cache_before_run and hasattr(tokenizer, "clear_cache"):
                tokenizer.clear_cache()
            warmup_start = time.perf_counter()
            run_task_once()
            print(
                f"  warmup {warmup_idx + 1}/{config.warmup_runs}: "
                f"{time.perf_counter() - warmup_start:.3f}s",
                flush=True,
            )

        for repeat_idx in range(config.repeats):
            if config.clear_cache_before_run and hasattr(tokenizer, "clear_cache"):
                tokenizer.clear_cache()
            measurement = measure_once(run_task_once)

            measurements.append(measurement)
            print(
                f"  repeat {repeat_idx + 1}/{config.repeats}: "
                f"{measurement.elapsed_sec:.3f}s, "
                f"tracemalloc_peak={measurement.peak_memory_mb:.2f}MB, "
                f"rss_peak={measurement.peak_rss_mb:.2f}MB",
                flush=True,
            )

        notes = "baseline before optimization" if version == "before" else "after optimization"
        row = summarize_measurements(version, task, measurements, base, config, notes)
        rows.append(row)
        details["tasks"][task] = {
            "summary": row,
            "repeats": [asdict(item) for item in measurements],
        }

    return rows, details


def main() -> None:
    run_started_at = datetime.now(timezone.utc)
    run_start = time.perf_counter()
    config = parse_args()
    if not config.tokenizer_path.exists():
        raise FileNotFoundError(
            f"Tokenizer file not found: {config.tokenizer_path}. "
            "Expected artifacts/tokenizer_150k.json."
        )

    print("Loading benchmark corpus...", flush=True)
    records = load_corpus(config)
    print(f"Loaded {len(records):,} records: {corpus_summary(records)}", flush=True)

    all_rows: list[dict[str, Any]] = []
    all_details: dict[str, Any] = {
        "config": {
            **asdict(config),
            "corpus_dir": display_path(config.corpus_dir),
            "tokenizer_path": display_path(config.tokenizer_path),
            "output_csv": display_path(config.output_csv),
            "output_json": display_path(config.output_json),
        },
        "run": {
            "started_at_utc": run_started_at.isoformat(),
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "corpus_sample": corpus_summary(records),
        "corpus_manifest": manifest_summary(config.corpus_dir),
        "versions": {},
    }

    for version in config.versions:
        rows, details = run_for_version(version, config, records)
        all_rows.extend(rows)
        all_details["versions"][version] = details

    run_finished_at = datetime.now(timezone.utc)
    all_details["run"].update(
        {
            "finished_at_utc": run_finished_at.isoformat(),
            "elapsed_sec": time.perf_counter() - run_start,
        }
    )

    summary = build_top_level_summary(all_rows)
    ordered_details = {"summary": summary}
    ordered_details.update(all_details)

    write_csv(all_rows, config.output_csv)
    write_json(ordered_details, config.output_json)
    print(f"Wrote CSV: {config.output_csv}")
    print(f"Wrote JSON: {config.output_json}")


if __name__ == "__main__":
    main()
