# 생성된 공개 Corpus 위치

이 폴더는 `scripts/prepare_public_corpus.py`의 기본 출력 위치입니다.

`*.jsonl.gz`, `manifest.json` 같은 생성 파일은 로컬 산출물이며 Git에는 올리지 않습니다.

재생성 명령:

```powershell
python scripts/prepare_public_corpus.py --target-tokens 1000000000
```
