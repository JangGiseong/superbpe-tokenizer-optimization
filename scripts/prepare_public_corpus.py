"""Prepare a reproducible public corpus for SuperBPE benchmarks.

The script intentionally downloads data from public Hugging Face datasets instead
of storing raw corpus files in the repository.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "public_corpus"


@dataclass(frozen=True)
class CorpusSource:
    lang: str
    dataset: str
    config: str
    split: str
    text_field: str
    target_ratio: float
    license: str
    terms: str
    url: str


@dataclass
class SourceStats:
    lang: str
    dataset: str
    config: str
    output_file: str
    documents: int = 0
    bytes_utf8: int = 0
    counted_tokens: int = 0
    rows_with_dataset_token_count: int = 0
    rows_with_estimated_token_count: int = 0
    elapsed_sec: float = 0.0


SOURCES = [
    CorpusSource(
        lang="ko",
        dataset="HuggingFaceFW/fineweb-2",
        config="kor_Hang",
        split="train",
        text_field="text",
        target_ratio=0.5,
        license="ODC-By 1.0",
        terms="Subject to Common Crawl Terms of Use",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
    ),
    CorpusSource(
        lang="en",
        dataset="HuggingFaceFW/fineweb-edu",
        config="sample-10BT",
        split="train",
        text_field="text",
        target_ratio=0.5,
        license="ODC-By 1.0",
        terms="Subject to Common Crawl Terms of Use",
        url="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu",
    ),
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def estimate_tokens(text: str) -> int:
    """Conservative fallback when a dataset row lacks token_count."""
    byte_len = len(text.encode("utf-8"))
    return max(1, math.ceil(byte_len / 4))


def row_token_count(row: dict[str, Any], text: str) -> tuple[int, bool]:
    token_count = row.get("token_count")
    if isinstance(token_count, int) and token_count > 0:
        return token_count, True
    if isinstance(token_count, float) and token_count > 0:
        return int(token_count), True
    return estimate_tokens(text), False


def iter_stream(source: CorpusSource) -> Iterable[dict[str, Any]]:
    return load_dataset(
        source.dataset,
        source.config,
        split=source.split,
        streaming=True,
    )


def write_source(
    source: CorpusSource,
    output_dir: Path,
    target_tokens: int,
    max_documents: int | None,
) -> SourceStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_config = source.config.replace("/", "_")
    output_path = output_dir / f"{source.lang}_{safe_config}.jsonl.gz"
    stats = SourceStats(
        lang=source.lang,
        dataset=source.dataset,
        config=source.config,
        output_file=str(output_path.resolve().relative_to(ROOT)),
    )

    started = time.perf_counter()
    with gzip.open(output_path, "wt", encoding="utf-8") as f:
        for row in iter_stream(source):
            text = row.get(source.text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            tokens, from_dataset = row_token_count(row, text)
            record = {
                "text": text,
                "lang": source.lang,
                "source_dataset": source.dataset,
                "source_config": source.config,
                "source_url": source.url,
                "license": source.license,
                "token_count": tokens,
                "token_count_source": "dataset" if from_dataset else "estimated_bytes_div_4",
            }
            if "id" in row:
                record["source_id"] = row["id"]
            if "url" in row:
                record["document_url"] = row["url"]

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            stats.documents += 1
            stats.bytes_utf8 += len(text.encode("utf-8"))
            stats.counted_tokens += tokens
            if from_dataset:
                stats.rows_with_dataset_token_count += 1
            else:
                stats.rows_with_estimated_token_count += 1

            if stats.documents % 10_000 == 0:
                print(
                    f"[{source.lang}] docs={stats.documents:,} "
                    f"tokens={stats.counted_tokens:,}/{target_tokens:,}",
                    flush=True,
                )

            if stats.counted_tokens >= target_tokens:
                break
            if max_documents is not None and stats.documents >= max_documents:
                break

    stats.elapsed_sec = time.perf_counter() - started
    return stats


def write_manifest(
    output_dir: Path,
    target_tokens: int,
    stats: list[SourceStats],
) -> Path:
    manifest_path = output_dir / "manifest.json"
    total_tokens = sum(item.counted_tokens for item in stats)
    total_docs = sum(item.documents for item in stats)
    total_bytes = sum(item.bytes_utf8 for item in stats)
    payload = {
        "created_at_epoch": int(time.time()),
        "target_tokens": target_tokens,
        "actual_counted_tokens": total_tokens,
        "actual_documents": total_docs,
        "actual_bytes_utf8": total_bytes,
        "sources": [asdict(source) for source in SOURCES],
        "stats": [asdict(item) for item in stats],
        "notes": [
            "Raw corpus shards are reproducible local artifacts and are not committed to Git.",
            "Dataset token_count is used when present; otherwise tokens are estimated as ceil(utf8_bytes / 4).",
        ],
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-tokens",
        type=positive_int,
        default=1_000_000_000,
        help="Total target counted tokens across all sources.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated JSONL.GZ shards and manifest.",
    )
    parser.add_argument(
        "--max-documents-per-source",
        type=positive_int,
        default=None,
        help="Optional safety cap for smoke tests.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args.output_dir)
    per_source_stats: list[SourceStats] = []

    for source in SOURCES:
        source_target = int(args.target_tokens * source.target_ratio)
        print(
            f"Downloading {source.lang} from {source.dataset}/{source.config} "
            f"until {source_target:,} counted tokens",
            flush=True,
        )
        stats = write_source(
            source=source,
            output_dir=output_dir,
            target_tokens=source_target,
            max_documents=args.max_documents_per_source,
        )
        per_source_stats.append(stats)
        print(
            f"[{source.lang}] complete docs={stats.documents:,} "
            f"tokens={stats.counted_tokens:,} bytes={stats.bytes_utf8:,} "
            f"elapsed={stats.elapsed_sec:.1f}s",
            flush=True,
        )

    manifest = write_manifest(output_dir, args.target_tokens, per_source_stats)
    total = sum(item.counted_tokens for item in per_source_stats)
    print(f"Manifest written: {manifest}")
    print(f"Total counted tokens: {total:,} / target {args.target_tokens:,}")


if __name__ == "__main__":
    main()
