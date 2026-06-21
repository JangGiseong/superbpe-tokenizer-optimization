"""
data_prep.py — 토크나이저 학습 코퍼스 준비

역할:
  1. 각 도메인 parquet에서 텍스트 샘플링 (목표: ~30GB)
  2. 상위 1% 길이 문서 truncate 기준값 계산 (논문 Appendix A.1.1)
  3. 언어 감지 (KO / EN / CODE → Stage 1 pretokenizer 라우팅용)
  4. BPETrainer.train()에 넘길 이터레이터 팩토리 생성

데이터 구조 (이전에 다운로드한 parquet 기준):
  data/pretraining/fineweb_en/   → 컬럼: text
  data/pretraining/fineweb2_ko/  → 컬럼: text
  data/pretraining/webtext_ko/   → 컬럼: text
  data/pretraining/stack_python/ → 컬럼: content
  data/pretraining/stack_cpp/    → 컬럼: content
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from typing import Iterator, Callable, List, Tuple, Optional, Dict
import pathlib
import pyarrow as pa

# 프로젝트 루트 기준 (data_prep.py가 있는 위치 기준)
BASE = pathlib.Path("data")


def _make_glob_pattern(base: pathlib.Path, *parts: str) -> str:
    """
    Windows에서 glob(**) 정상 작동을 위해 순방향 슬래시 사용.
    pathlib이 역슬래시로 변환한 것을 다시 순방향으로 변환.
    """
    return str(base.joinpath(*parts)).replace("\\", "/")


DOMAIN_CONFIG = [
    (
        str(BASE / "pre-training_dataset" / "fineweb2_ko" / "*.arrow"),
        "text",
        "ko",
        12.0,
    ),
    (
        str(BASE / "pre-training_dataset" / "webtext_ko" / "*.arrow"),
        "text",
        "ko",
        3.0,
    ),
    (
        str(BASE / "pre-training_dataset" / "fineweb_en" / "*.arrow"),
        "text",
        "en",
        10.0,
    ),
    (
        str(BASE / "pre-training_dataset" / "stack_python" / "*.arrow"),
        "content",
        "code",
        2.5,
    ),
    (
        str(BASE / "pre-training_dataset" / "stack_cpp" / "*.arrow"),
        "content",
        "code",
        2.5,
    ),
]

PILOT_DOMAIN_CONFIG = [
    (
        _make_glob_pattern(BASE, "pre-training_dataset", "fineweb2_ko", "*.arrow"),
        "text",
        "ko",
        0.3,
    ),
    (
        _make_glob_pattern(BASE, "pre-training_dataset", "webtext_ko", "*.arrow"),
        "text",
        "ko",
        0.1,
    ),
    (
        _make_glob_pattern(BASE, "pre-training_dataset", "fineweb_en", "*.arrow"),
        "text",
        "en",
        0.3,
    ),
    (
        _make_glob_pattern(BASE, "pre-training_dataset", "stack_python", "*.arrow"),
        "content",
        "code",
        0.1,
    ),
    (
        _make_glob_pattern(BASE, "pre-training_dataset", "stack_cpp", "*.arrow"),
        "content",
        "code",
        0.1,
    ),
]

BYTES_PER_GB = 1024**3
PERCENTILE_SAMPLE_SIZE = 100_000  # 99th percentile 계산용 샘플 수
RANDOM_SEED = 42

# ── 언어 감지 ─────────────────────────────────────────────────────


def detect_lang(text: str) -> str:
    """
    텍스트의 언어 코드 반환.
    학습 데이터는 소스 파일 기준으로 이미 분리되어 있으므로
    이 함수는 inference 시 동적 감지에 사용.

    반환값: "ko" | "en" | "code"
    """
    if not text:
        return "en"

    # 한글 유니코드 범위: AC00–D7A3 (가-힣)
    korean_chars = sum(1 for c in text[:500] if "\uac00" <= c <= "\ud7a3")
    total_chars = min(len(text), 500)

    if total_chars == 0:
        return "en"

    korean_ratio = korean_chars / total_chars

    if korean_ratio > 0.15:
        return "ko"

    # 코드 감지: 들여쓰기 + 괄호 패턴
    code_chars = sum(1 for c in text[:200] if c in "{}[]();=><")
    if code_chars / max(total_chars, 1) > 0.05:
        return "code"

    return "en"


# ── 문서 길이 95/99 percentile 계산 ──────────────────────────────


def compute_length_percentile(
    parquet_pattern: str,
    text_col: str,
    pct: float = 99.0,
    sample_size: int = PERCENTILE_SAMPLE_SIZE,
) -> int:
    """
    parquet 파일에서 샘플을 읽어 문서 길이 percentile 계산.
    논문 Appendix A.1.1: 상위 1% 문서를 99th percentile로 truncate.

    Args:
      parquet_pattern: glob 패턴 (예: "data/fineweb_en/**/*.arrow")
      text_col:        텍스트 컬럼명
      pct:             계산할 percentile (기본: 99)
      sample_size:     샘플링할 문서 수

    Returns:
      문자 수 기준 percentile 값 (int)
    """
    files = glob.glob(parquet_pattern, recursive=True)
    if not files:
        print(f"  [경고] 파일 없음: {parquet_pattern}")
        return 10_000  # 기본값

    lengths = []
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(files)

    for fpath in files:
        if len(lengths) >= sample_size:
            break
        try:
            df = read_data_file(fpath, columns=[text_col])
            texts = df[text_col].dropna().tolist()
            # 파일당 최대 5000개 샘플
            sample = rng.sample(texts, min(5000, len(texts)))
            lengths.extend(len(t) for t in sample)
        except Exception as e:
            print(f"  [경고] {fpath}: {e}")
            continue

    if not lengths:
        return 10_000

    result = int(np.percentile(lengths, pct))
    print(f"  길이 {pct}th percentile: {result:,} 문자")
    return result


def read_data_file(path: str, columns: list) -> pd.DataFrame:
    """
    Arrow (.arrow) 또는 Parquet (.parquet) 파일 통합 읽기.
    HuggingFace save_to_disk()는 Arrow IPC 형식으로 저장함.
    """
    if path.endswith(".arrow"):
        try:
            # Arrow IPC File format (HuggingFace save_to_disk 기본)
            with pa.ipc.open_file(path) as reader:
                table = reader.read_all()
        except pa.lib.ArrowInvalid:
            # Arrow IPC Stream format (일부 파일)
            with pa.ipc.open_stream(path) as reader:
                table = reader.read_all()

        # 필요한 컬럼만 선택
        available = [c for c in columns if c in table.schema.names]
        if not available:
            return pd.DataFrame()
        return table.select(available).to_pandas()

    else:  # .parquet fallback
        return pd.read_parquet(path, columns=columns)


# ── parquet 텍스트 샘플링 ─────────────────────────────────────────


def sample_texts_from_parquet(
    parquet_pattern: str,
    text_col: str,
    target_bytes: int,
    max_doc_len: int,
    lang: str,
    seed: int = RANDOM_SEED,
) -> List[Tuple[str, str]]:
    """
    parquet에서 텍스트를 읽어 목표 바이트 수에 도달할 때까지 샘플링.

    Args:
      parquet_pattern: glob 패턴
      text_col:        텍스트 컬럼명
      target_bytes:    목표 바이트 수
      max_doc_len:     문서 최대 길이 (초과 시 truncate)
      lang:            언어 코드 ("ko" / "en" / "code")
      seed:            랜덤 시드

    Returns:
      [(text, lang_code), ...] 리스트
    """
    files = glob.glob(parquet_pattern, recursive=True)
    if not files:
        print(f"  [경고] 파일 없음: {parquet_pattern}")
        return []

    rng = random.Random(seed)
    rng.shuffle(files)

    results: List[Tuple[str, str]] = []
    total_bytes = 0
    domain = (
        parquet_pattern.split("/")[2] if "/" in parquet_pattern else parquet_pattern
    )

    print(
        f"  [{domain}] 목표: {target_bytes / BYTES_PER_GB:.1f} GB  "
        f"(파일 {len(files)}개)"
    )

    for fpath in files:
        if total_bytes >= target_bytes:
            break
        try:
            df = read_data_file(fpath, columns=[text_col])
            texts = df[text_col].dropna().tolist()
            rng.shuffle(texts)

            for text in texts:
                if total_bytes >= target_bytes:
                    break
                if not text or not text.strip():
                    continue

                # 문서 길이 truncate (논문 Appendix A.1.1)
                if len(text) > max_doc_len:
                    text = text[:max_doc_len]

                results.append((text, lang))
                total_bytes += len(text.encode("utf-8"))

        except Exception as e:
            print(f"    [경고] {fpath}: {e}")
            continue

    actual_gb = total_bytes / BYTES_PER_GB
    print(f"    → {len(results):,}개 문서, {actual_gb:.2f} GB 수집")
    return results


# ── 메인 코퍼스 준비 ──────────────────────────────────────────────


class CorpusPrep:
    """
    토크나이저 학습용 코퍼스 준비 클래스.

    사용 예시:
      prep = CorpusPrep()
      prep.prepare(domain_config=DOMAIN_CONFIG)

      # BPETrainer에 팩토리 함수로 전달
      trainer.train(
          texts_iter_factory=prep.get_iter_factory(),
          lang_fn=prep.lang_fn,
          max_doc_len=prep.max_doc_len,
      )
    """

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self._corpus: List[Tuple[str, str]] = []  # [(text, lang)]
        self.max_doc_len: int = 10_000

    def prepare(
        self,
        domain_config: List[Tuple[str, str, str, float]] = None,
        compute_percentile: bool = True,
    ):
        """
        전체 코퍼스 준비 파이프라인 실행.

        1. 각 도메인별 99th percentile 계산
        2. 목표 GB만큼 텍스트 샘플링
        3. 전체 섞기
        """
        config = domain_config or DOMAIN_CONFIG

        print("\n" + "=" * 60)
        print("코퍼스 준비 시작")
        print("=" * 60)

        # ─ Step 1: 99th percentile 계산 ───────────────────────────
        if compute_percentile:
            print("\n[1] 문서 길이 99th percentile 계산...")
            all_lengths = []
            for pattern, col, _, _ in config:
                p = compute_length_percentile(pattern, col, pct=99)
                all_lengths.append(p)
            # 전체 도메인 99th percentile 중 중앙값을 max_doc_len으로 사용
            self.max_doc_len = int(np.median(all_lengths))
            print(f"  max_doc_len 결정: {self.max_doc_len:,} 문자")
        else:
            self.max_doc_len = 8_000

        # ─ Step 2: 텍스트 샘플링 ───────────────────────────────────
        print("\n[2] 도메인별 텍스트 샘플링...")
        self._corpus = []

        for pattern, col, lang, target_gb in config:
            target_bytes = int(target_gb * BYTES_PER_GB)
            sampled = sample_texts_from_parquet(
                pattern, col, target_bytes, self.max_doc_len, lang, self.seed
            )
            self._corpus.extend(sampled)

        # ─ Step 3: 전체 섞기 ────────────────────────────────────────
        print(f"\n[3] 전체 코퍼스 섞기...")
        rng = random.Random(self.seed)
        rng.shuffle(self._corpus)

        total_gb = sum(len(t.encode("utf-8")) for t, _ in self._corpus) / BYTES_PER_GB
        print(f"  총 {len(self._corpus):,}개 문서, {total_gb:.2f} GB")
        print("=" * 60)

    def get_iter_factory(self) -> Callable[[], Iterator[str]]:
        """
        BPETrainer.train()에 넘길 텍스트 이터레이터 팩토리 반환.
        팩토리는 Stage 1, Stage 2에서 각각 한 번씩 호출됨.
        """
        corpus = self._corpus  # 클로저 캡처

        def factory() -> Iterator[str]:
            for text, _ in corpus:
                yield text

        return factory

    def lang_fn(self, text: str) -> str:
        """
        저장된 코퍼스에서 텍스트의 언어 코드 반환.
        O(1) 조회를 위해 text → lang 딕셔너리 캐시 사용.

        주의: 동일한 텍스트가 여러 언어에 있을 경우 첫 번째 매핑 사용.
              코퍼스 규모가 크므로 학습 외 용도에는 detect_lang() 사용 권장.
        """
        # 캐시 없으면 규칙 기반 감지로 폴백
        return detect_lang(text)

    def get_lang_iter_factory(self) -> Callable[[], Iterator[Tuple[str, str]]]:
        """
        (text, lang) 튜플 이터레이터 팩토리 반환.
        lang_fn 대신 이터레이터에 언어 정보를 직접 포함하는 방식.
        """
        corpus = self._corpus

        def factory() -> Iterator[Tuple[str, str]]:
            for text, lang in corpus:
                yield text, lang

        return factory

    def save_sample(self, path: str, n: int = 1000):
        """디버깅용: 샘플 텍스트를 텍스트 파일로 저장."""
        sample = self._corpus[:n]
        with open(path, "w", encoding="utf-8") as f:
            for i, (text, lang) in enumerate(sample):
                f.write(f"=== [{i}] lang={lang} len={len(text)} ===\n")
                f.write(text[:500])
                f.write("\n\n")
        print(f"샘플 {n}개 저장: {path}")


# ── 실행 진입점 ───────────────────────────────────────────────────

if __name__ == "__main__":
    prep = CorpusPrep(seed=RANDOM_SEED)
    prep.prepare(domain_config=DOMAIN_CONFIG, compute_percentile=True)
    prep.save_sample("corpus_sample.txt", n=500)

    print("\n팩토리 함수 테스트...")
    factory = prep.get_iter_factory()
    first_5 = list(text for _, text in zip(range(5), factory()))
    for i, t in enumerate(first_5):
        print(f"  [{i}] lang={detect_lang(t):<4} len={len(t):>6} | {t[:80]!r}")
