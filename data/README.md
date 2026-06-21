# 공개 Corpus 데이터

이 디렉터리는 SuperBPE 성능 개선 실험에서 사용할 공개 데이터 구성 방식을 설명합니다.

원래 연구 프로젝트는 약 20B-token 규모의 내부 corpus를 사용했습니다. 그러나 해당 데이터는 GitHub 제출물에 포함하기에는 너무 크고, 재배포 권한이 명확하지 않을 수 있습니다. 따라서 공개 Hugging Face 데이터셋에서 corpus를 다시 구성합니다.

데이터 다운로드와 샘플링은 `scripts/prepare_public_corpus.py`가 담당합니다.

## 선택한 공개 데이터셋

| 구분 | Dataset | Config | 목표 비율 | 라이선스/조건 |
| --- | --- | --- | --- | --- |
| 한국어 | `HuggingFaceFW/fineweb-2` | `kor_Hang` | 50% | ODC-By 1.0, Common Crawl Terms of Use 적용 |
| 영어 | `HuggingFaceFW/fineweb-edu` | `sample-10BT` | 50% | ODC-By 1.0, Common Crawl Terms of Use 적용 |

이 보수적인 2-way 구성은 한국어·영어 SuperBPE tokenizer라는 연구 맥락과 잘 맞고, private data나 code-license 문제를 피할 수 있습니다.

## 1B Corpus 생성

```powershell
python scripts/prepare_public_corpus.py --target-tokens 1000000000
```

기본 출력 위치:

```text
data/public_corpus/
```

생성되는 raw shard와 manifest는 Git에 올리지 않습니다. GitHub에는 다운로드 방법과 생성 스크립트만 남깁니다.

## 검증된 1B Corpus 결과

실제로 위 명령을 실행해 다음 결과를 확인했습니다.

| 구분 | 문서 수 | Counted Tokens | UTF-8 Bytes | Token Count 기준 |
| --- | ---: | ---: | ---: | --- |
| Korean FineWeb2 `kor_Hang` | 524,638 | 500,001,585 | 1,999,219,741 | `ceil(utf8_bytes / 4)` 추정 |
| English FineWeb-Edu `sample-10BT` | 480,840 | 500,000,313 | 2,298,736,519 | dataset `token_count` 필드 |
| 합계 | 1,005,478 | 1,000,001,898 | 4,297,956,260 | 혼합 |

로컬에 생성되는 raw shard:

```text
data/public_corpus/ko_kor_Hang.jsonl.gz
data/public_corpus/en_sample-10BT.jsonl.gz
data/public_corpus/manifest.json
```

## 빠른 Smoke Test

```powershell
python scripts/prepare_public_corpus.py --target-tokens 1000000
```

## 주의사항

- 스크립트는 Hugging Face streaming으로 데이터를 읽고 compressed JSONL shard를 생성합니다.
- dataset row에 `token_count`가 있으면 그 값을 사용합니다.
- `token_count`가 없으면 UTF-8 byte 길이를 기준으로 `ceil(bytes / 4)` 추정값을 기록합니다.
- 생성된 raw corpus shard는 GitHub에 commit하지 않습니다.
