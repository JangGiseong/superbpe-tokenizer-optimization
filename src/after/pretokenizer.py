"""pretokenizer.py — SuperBPE 사전 토크나이저."""

# (독스트링을 별도로 r-string으로 감싸거나, 위처럼 \\p 로 표기)

import re
import unicodedata
from enum import Enum
from typing import List, Optional

try:
    import regex

    HAS_REGEX = True
except ImportError:
    import re as regex

    HAS_REGEX = False
    print("[WARNING] regex 모듈 없음. pip install regex 권장.")

HAS_KIWI = False
_kiwi = None

from kiwipiepy import Kiwi

_kiwi = Kiwi()
HAS_KIWI = True


class Stage(Enum):
    STAGE1 = 1
    STAGE2 = 2


# 논문 Appendix A.1.3 기반 + 숫자 캡처 추가 (| [ ]?\p{N}+)
# \p{L}: Unicode 문자, \p{N}: Unicode 숫자
WHITESPACE_PRETOK_PATTERN = (
    r"[ ]?\p{L}+"  # 선택적 공백 + Unicode 문자
    r"|[ ]?[^\s\p{L}\p{N}]+"  # 선택적 공백 + 구두점·기호
    r"|\s+(?!\S)"  # 후행 공백
    r"|\s+"  # 나머지 공백
    r"|[ ]?\p{N}+"  # 선택적 공백 + 숫자 ← 추가
)

DIGIT_BOUNDARY_PATTERN = r"(?=(?:\d{3})+(?!\d))"


def normalize_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def split_digits_right(digit_str: str) -> List[str]:
    parts = re.split(DIGIT_BOUNDARY_PATTERN, digit_str)
    return [p for p in parts if p]


def apply_digit_split(chunks: List[str]) -> List[str]:
    result = []
    for chunk in chunks:
        if not any(c.isdigit() for c in chunk):
            result.append(chunk)
            continue
        i = 0
        while i < len(chunk):
            if chunk[i].isdigit():
                j = i
                while j < len(chunk) and chunk[j].isdigit():
                    j += 1
                result.extend(split_digits_right(chunk[i:j]))
                i = j
            else:
                j = i
                while j < len(chunk) and not chunk[j].isdigit():
                    j += 1
                result.append(chunk[i:j])
                i = j
    return [r for r in result if r]


def mecab_tokenize(text: str) -> List[str]:
    if HAS_KIWI and _kiwi is not None:
        try:
            tokens = _kiwi.tokenize(text)
            return [t.form for t in tokens if t.form]
        except Exception:
            pass
    return text.split()


class Pretokenizer:
    def __init__(
        self,
        stage: Stage = Stage.STAGE1,
        use_mecab: bool = True,
        max_segment_len: Optional[int] = 1000,
    ):
        self.stage = stage
        self.use_mecab = use_mecab and HAS_KIWI
        self.max_segment_len = max_segment_len

        if HAS_REGEX:
            self._ws_compiled = regex.compile(WHITESPACE_PRETOK_PATTERN)
        else:
            self._ws_compiled = None

    def pretokenize(self, text: str, lang: str = "en") -> List[str]:
        text = normalize_nfc(text)

        if self.stage == Stage.STAGE2:
            return self._pretokenize_stage2(text)

        if lang == "ko" and self.use_mecab:
            chunks = mecab_tokenize(text)
        elif self._ws_compiled is not None:
            chunks = self._ws_compiled.findall(text)
        else:
            chunks = text.split()

        return apply_digit_split(chunks)

    def _pretokenize_stage2(self, text: str) -> List[str]:
        if self.max_segment_len is None:
            return apply_digit_split([text])

        segments = []
        for para in text.split("\n\n"):
            if len(para) == 0:  # strip() 제거 — 완전히 빈 문자열만 제외
                continue
            while len(para) > self.max_segment_len:
                segments.append(para[: self.max_segment_len])
                para = para[self.max_segment_len :]
            if para:
                segments.append(para)

        if not segments:  # 원본 텍스트가 \n\n 없는 경우 대비
            return apply_digit_split([text]) if text else []

        return apply_digit_split(segments)
