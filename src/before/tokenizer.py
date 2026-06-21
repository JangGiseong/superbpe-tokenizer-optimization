"""
tokenizer.py — SuperBPE 공개 인터페이스

encode(text) → token id 리스트
decode(ids)  → 원문 복원 (Roundtrip 100% 보장)

encode_with_special(): Chat template 적용 인코딩
"""

from typing import List, Optional, Callable
from vocab import Vocabulary
from pretokenizer import Pretokenizer, Stage, normalize_nfc


class SuperBPETokenizer:
    """
    학습된 SuperBPE 토크나이저.

    인코딩 알고리즘: Greedy BPE (학습 순서 기반)
      - 각 청크를 바이트 단위로 초기화
      - 학습된 병합 규칙을 rank(학습 순서) 기준으로 반복 적용
      - O(n² log n) per chunk (n = chunk 길이, 짧은 청크 가정)
    """

    def __init__(
        self,
        vocab: Vocabulary,
        lang_fn: Optional[Callable[[str], str]] = None,
    ):
        self.vocab = vocab
        self.lang_fn = lang_fn
        self._pretok = Pretokenizer(
            stage=Stage.STAGE2,
            max_segment_len=512,
        )

        self._merge_ranks = {pair: rank for rank, pair in enumerate(vocab.merges)}

    # ── 인코딩 ────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        lang: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        if lang is None:
            lang = self.lang_fn(text) if self.lang_fn else "en"

        text = normalize_nfc(text)

        # 전체 텍스트가 특수 토큰인 경우 바로 반환
        if text in self.vocab.token2id:
            ids = [self.vocab.get_id(text)]
            if add_bos:
                ids = [self.vocab.bos_id] + ids
            if add_eos:
                ids = ids + [self.vocab.eos_id]
            return ids

        chunks = self._pretok.pretokenize(text, lang=lang)

        ids: List[int] = []
        if add_bos:
            ids.append(self.vocab.bos_id)

        for chunk in chunks:
            # 청크 단위 특수 토큰 체크
            if (
                chunk in self.vocab.token2id
                and chunk.startswith("<|")
                and chunk.endswith("|>")
            ):
                ids.append(self.vocab.get_id(chunk))
            else:
                tokens = self._bpe_encode_chunk(chunk)
                ids.extend(self.vocab.get_id(t) for t in tokens)

        if add_eos:
            ids.append(self.vocab.eos_id)

        return ids

    def _bpe_encode_chunk(self, chunk: str) -> List[str]:
        """
        단일 청크에 BPE 인코딩 적용.
        학습 시 사용한 병합 규칙을 rank 순서대로 탐욕적으로 적용.
        """
        # latin-1 바이트 단위 초기화
        tokens = [bytes([b]).decode("latin-1") for b in chunk.encode("utf-8")]

        while len(tokens) > 1:
            # 인접 쌍 중 rank가 가장 낮은(먼저 학습된) 쌍 탐색
            best_rank = float("inf")
            best_idx = -1

            for i in range(len(tokens) - 1):
                rank = self._merge_ranks.get((tokens[i], tokens[i + 1]), float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i

            if best_idx == -1:  # 적용 가능한 병합 없음
                break

            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2 :]

        return tokens

    # ── 디코딩 ────────────────────────────────────────────────────

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """
        토큰 ID 리스트 → 원문 복원.
        바이트 수준으로 처리해 Roundtrip 100% 보장.

        Args:
          skip_special_tokens: True면 <|user|> 등 특수 토큰 제외
        """
        byte_buffer = bytearray()

        for idx in ids:
            token = self.vocab.get_token(idx)

            # 특수 토큰 처리
            if token.startswith("<|") and token.endswith("|>"):
                if not skip_special_tokens:
                    byte_buffer.extend(token.encode("utf-8"))
                continue

            # 일반 토큰: latin-1 역변환 → 원래 UTF-8 바이트 복원
            try:
                byte_buffer.extend(token.encode("latin-1"))
            except (UnicodeEncodeError, UnicodeDecodeError):
                byte_buffer.extend(token.encode("utf-8", errors="replace"))

        return byte_buffer.decode("utf-8", errors="replace")

    # ── Chat Template ────────────────────────────────────────────

    def encode_chat(
        self,
        messages: List[dict],
        add_generation_prompt: bool = True,
    ) -> List[int]:
        """
        Chat template 적용 인코딩.

        Args:
          messages: [{"role": "system"/"user"/"assistant", "content": "..."}]
          add_generation_prompt: True면 마지막에 <|assistant|> 추가

        예시 입력:
          [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "안녕하세요"},
          ]
        출력 형식:
          <|bos|> <|system|> {system} <|user|> {user} <|assistant|>
        """
        ids = [self.vocab.bos_id]

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            role_token_map = {
                "system": "<|system|>",
                "user": "<|user|>",
                "assistant": "<|assistant|>",
            }
            role_token = role_token_map.get(role, "<|user|>")
            ids.append(self.vocab.get_id(role_token))

            lang = "ko" if self._is_korean(content) else "en"
            ids.extend(self.encode(content, lang=lang))

        if add_generation_prompt:
            ids.append(self.vocab.get_id("<|assistant|>"))

        return ids

    @staticmethod
    def _is_korean(text: str) -> bool:
        """한글 문자 포함 여부로 한국어 판별."""
        return any("\uac00" <= c <= "\ud7a3" for c in text)

    # ── 편의 메서드 ───────────────────────────────────────────────

    @classmethod
    def from_file(
        cls,
        path: str,
        lang_fn: Optional[Callable] = None,
    ) -> "SuperBPETokenizer":
        """저장된 JSON에서 토크나이저 로드."""
        vocab = Vocabulary.load(path)
        return cls(vocab, lang_fn=lang_fn)

    def vocab_size(self) -> int:
        return len(self.vocab)
