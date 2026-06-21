# SuperBPE Tokenizer 성능 개선 실험

이 프로젝트는 LLM 연구 파이프라인에서 사용한 SuperBPE tokenizer 구현을 대상으로, Python 코드의 병목을 분석하고 자료구조, iterator/generator, decorator 기반 caching, lazy evaluation을 적용해 성능과 구조를 개선한 실험이다.

핵심 개선 대상은 학습된 tokenizer의 `encode()` 경로이다. 기존 구현은 BPE merge를 한 번 수행할 때마다 전체 인접 pair를 다시 스캔했기 때문에, 대량 문서 tokenization에서 실행 시간이 급격히 증가했다.

## 프로젝트 구조

```text
superbpe_final_report/
├─ README.md
├─ requirements.txt
├─ src/
│  ├─ before/
│  └─ after/
├─ benchmark/
│  └─ run_benchmark.py
├─ scripts/
│  ├─ prepare_public_corpus.py
│  ├─ run_encode_scaling_benchmark.py
│  └─ plot_encode_scaling.py
├─ data/
│  ├─ README.md
│  └─ public_corpus/
└─ results/
   ├─ benchmark_results.csv
   ├─ benchmark_summary.json
   ├─ encode_scaling_summary.csv
   ├─ encode_scaling_summary.json
   └─ encode_scaling_*.png
```

## 대상 코드

- `src/before/tokenizer.py` → `src/after/tokenizer.py`
- `src/before/bpe_trainer.py` → `src/after/bpe_trainer.py`

`src/before/`는 최적화 전 코드이고, `src/after/`는 동일한 tokenizer 출력을 유지하면서 성능과 처리 구조를 개선한 코드이다.

## 적용한 개선

### 1. 자료구조 기반 encode 개선

기존 `tokenizer.py`는 각 chunk를 byte token list로 만든 뒤, 매 merge마다 전체 인접 pair를 순회해 가장 낮은 rank의 pair를 찾았다.

개선 후에는 다음 자료구조를 사용한다.

- `heapq`: merge 후보 pair를 rank 기준 priority queue로 관리
- `dict`: 학습된 merge pair의 rank를 O(1)에 조회
- `prev`, `next_idx`, `active` 배열: token list를 linked-list처럼 관리해 병합 주변 pair만 갱신
- `BYTE_TOKENS`: byte → latin-1 token 변환 table을 재사용

### 2. Decorator 기반 chunk cache

반복 등장하는 chunk의 BPE 결과를 재사용하기 위해 `@lru_cache`를 적용했다.

```python
@lru_cache(maxsize=encode_cache_size)
def cached_encode_chunk(chunk: str) -> tuple[str, ...]:
    return tuple(self._bpe_encode_chunk_uncached(chunk))
```

### 3. Iterator/Generator 기반 encode API

기존 `encode()` API는 유지하되, 내부적으로 `iter_encode()`를 사용하도록 바꾸었다.

```python
def encode(...):
    return list(self.iter_encode(...))

def iter_encode(...):
    yield token_id
```

이 구조는 대량 처리 파이프라인에서 token id list 전체를 즉시 materialize하지 않고 streaming 방식으로 소비할 수 있게 한다.

### 4. 학습/재학습 경로의 streaming batch 처리

`bpe_trainer.py`의 chunk count 경로에는 `_iter_batches()` generator를 추가했다. 기존처럼 전체 입력을 먼저 `list(texts)`로 변환하지 않고 batch 단위로 처리한다. 병렬 처리에서는 pending future 수를 제한해 한 번에 너무 많은 batch가 쌓이지 않도록 했다.

학습/재학습 경로는 기존 코드도 이미 multiprocessing chunk aggregation을 사용하고 있어 개선 폭은 제한적이었다. 따라서 최종 결과의 중심은 tokenizer encode 경로이다.

## 공개 데이터

원래 연구 파이프라인에서는 약 20B-token 규모의 내부 데이터를 사용했지만, GitHub 제출과 재현성을 위해 공개 corpus를 사용한다.

- 한국어: `HuggingFaceFW/fineweb-2`, config `kor_Hang`
- 영어: `HuggingFaceFW/fineweb-edu`, config `sample-10BT`

1B 규모 공개 corpus 생성:

```powershell
python scripts/prepare_public_corpus.py --target-tokens 1000000000
```

검증된 corpus 규모:

- total counted tokens: `1,000,001,898`
- documents: `1,005,478`
- UTF-8 bytes: `4,297,956,260`
- Korean shard: `524,638` documents
- English shard: `480,840` documents
- local manifest: `data/public_corpus/manifest.json`

원본 shard는 `data/public_corpus/` 아래에 생성되지만 Git에는 올리지 않는다.

## Benchmark 실행

최종 비교는 전체 공개 corpus의 약 5%인 `50,274`문서로 수행했다.

- Korean: `26,232`
- English: `24,042`
- `max_doc_chars`: `1000`
- `repeats`: `1`
- `warmup_runs`: `0`

5% benchmark는 기존 encode 구현의 실행 시간이 매우 길어 단일 실행으로 측정했다. 반복 측정과 입력 크기별 증가율 확인은 더 작은 입력 크기에서 별도로 수행하는 것이 적절하다.

Before:

```powershell
python benchmark\run_benchmark.py --version before --tasks encode,chunk_count --ko-documents 26232 --en-documents 24042 --chunk-count-documents-per-lang 25137 --repeats 1 --warmup-runs 0 --max-doc-chars 1000 --progress-every-docs 1000 --n-workers 8 --output-csv results\baseline_before_5pct.csv --output-json results\baseline_before_5pct.json
```

After:

```powershell
python benchmark\run_benchmark.py --version after --tasks encode,chunk_count --ko-documents 26232 --en-documents 24042 --chunk-count-documents-per-lang 25137 --repeats 1 --warmup-runs 0 --max-doc-chars 1000 --progress-every-docs 1000 --n-workers 8 --output-csv results\after_5pct_iter.csv --output-json results\after_5pct_iter.json
```

반복 측정 및 입력 크기별 증가율 측정:

```powershell
python scripts\run_encode_scaling_benchmark.py
```

기본 설정은 `1k`, `2k`, `5k`, `10k` 문서에 대해 before/after encode를 순차 실행한다. `1k`, `2k`는 `repeats=5`, `5k`, `10k`는 `repeats=1`로 측정하며, 각 repeat 전에 tokenizer cache를 초기화한다. 결과는 `results/encode_scaling_summary.csv`와 `results/encode_scaling_summary.json`에 저장된다.

학습/재학습 경로 보조 확인:

```powershell
python benchmark\run_benchmark.py --version before --tasks train_small --ko-documents 26232 --en-documents 24042 --train-documents-per-lang 25137 --repeats 1 --warmup-runs 0 --max-doc-chars 1000 --train-vocab-size 1024 --train-transition-point 768 --train-stage2-max-docs 25137 --n-workers 8 --output-csv results\train_before_5pct.csv --output-json results\train_before_5pct.json

python benchmark\run_benchmark.py --version after --tasks train_small --ko-documents 26232 --en-documents 24042 --train-documents-per-lang 25137 --repeats 1 --warmup-runs 0 --max-doc-chars 1000 --train-vocab-size 1024 --train-transition-point 768 --train-stage2-max-docs 25137 --n-workers 8 --output-csv results\train_after_5pct.csv --output-json results\train_after_5pct.json
```

## 결과 요약

최종 비교 결과는 `results/benchmark_results.csv`와 `results/benchmark_summary.json`에 정리했다.

| Task        |      Before |    After | Speedup |
| ----------- | ----------: | -------: | ------: |
| encode      | 18,090.04 s | 437.71 s |  41.33x |
| chunk_count |      7.55 s |   6.33 s |   1.19x |

Encode 세부 결과:

| 항목          |     Before |      After |
| ------------- | ---------: | ---------: |
| documents     |     50,274 |     50,274 |
| output tokens | 12,730,989 | 12,730,989 |
| docs/sec      |       2.78 |     114.86 |
| tokens/sec    |     703.76 |  29,085.63 |
| cache hits    |          0 |    484,947 |
| cache misses  |          0 |    307,527 |

학습/재학습 보조 결과:

| Task        |   Before |   After | Speedup |
| ----------- | -------: | ------: | ------: |
| train_small | 106.63 s | 97.53 s |   1.09x |

학습/재학습 경로는 기존 구현도 이미 multiprocessing 기반 chunk aggregation을 사용하고 있었기 때문에 개선 폭이 작았다. 반면 encode 경로는 기존 구현의 반복적인 전체 pair scan이 지배 병목이었고, 자료구조와 cache 개선의 효과가 크게 나타났다.

## 반복 측정 및 입력 크기별 결과

`1k`, `2k` 문서는 `repeats=5`로 평균과 표준편차를 측정했고, `5k`, `10k` 문서는 실행 시간을 고려해 `repeats=1`로 측정했다. 모든 조건에서 before/after의 output token 수는 동일했다.

| Size | Before mean ± std | After mean ± std | Speedup |
| ---- | ----------------: | ---------------: | ------: |
| 1k   |  242.54 ± 41.64 s |    6.52 ± 0.48 s |  37.19x |
| 2k   |  477.34 ± 77.83 s |   12.79 ± 0.89 s |  37.33x |
| 5k   |        1,612.27 s |          36.87 s |  43.73x |
| 10k  |        3,299.44 s |          75.71 s |  43.58x |

입력 문서 수가 증가할 때 두 버전 모두 실행 시간이 증가하지만, after의 증가 기울기가 훨씬 작다. 이는 문서 수 자체에 대한 복잡도가 바뀌었다기보다, 각 chunk 내부에서 반복되던 전체 pair scan 비용을 heap 기반 후보 갱신과 cache로 줄였기 때문이다.
