"""
tokenizer.py — SuperBPE 공개 인터페이스

encode(text) → token id 리스트
decode(ids)  → 원문 복원 (Roundtrip 100% 보장)

encode_with_special(): Chat template 적용 인코딩
"""

import heapq
from functools import lru_cache
from typing import Iterator, List, Optional, Callable
from vocab import Vocabulary
from pretokenizer import Pretokenizer, Stage, normalize_nfc


BYTE_TOKENS = tuple(bytes([b]).decode("latin-1") for b in range(256))


class SuperBPETokenizer:
    """
    학습된 SuperBPE 토크나이저.

    인코딩 알고리즘: Greedy BPE (학습 순서 기반)
      - 각 청크를 바이트 단위로 초기화
      - 학습된 병합 규칙을 rank(학습 순서) 기준으로 반복 적용
      - heap 기반 후보 관리와 linked-list 인덱스로 병합 주변만 갱신
      - 반복 chunk는 @lru_cache로 재사용
    """

    def __init__(
        self,
        vocab: Vocabulary,
        lang_fn: Optional[Callable[[str], str]] = None,
        encode_cache_size: int = 65_536,
    ):
        self.vocab = vocab
        self.lang_fn = lang_fn
        self.encode_cache_size = encode_cache_size
        self._pretok = Pretokenizer(
            stage=Stage.STAGE2,
            max_segment_len=512,
        )

        self._merge_ranks = {pair: rank for rank, pair in enumerate(vocab.merges)}
        self._bpe_encode_chunk_cached = None
        if encode_cache_size > 0:

            @lru_cache(maxsize=encode_cache_size)
            def cached_encode_chunk(chunk: str) -> tuple[str, ...]:
                return tuple(self._bpe_encode_chunk_uncached(chunk))

            self._bpe_encode_chunk_cached = cached_encode_chunk

    # ── 인코딩 ────────────────────────────────────────────────────

    def encode(
        self,
        text: str,
        lang: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        return list(
            self.iter_encode(
                text,
                lang=lang,
                add_bos=add_bos,
                add_eos=add_eos,
            )
        )

    def iter_encode(
        self,
        text: str,
        lang: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Iterator[int]:
        """
        Token id를 lazy하게 생성한다.
        대량 처리 파이프라인에서는 전체 list materialization 없이 소비할 수 있다.
        """
        if lang is None:
            lang = self.lang_fn(text) if self.lang_fn else "en"

        text = normalize_nfc(text)

        # 전체 텍스트가 특수 토큰인 경우 바로 반환
        if text in self.vocab.token2id:
            if add_bos:
                yield self.vocab.bos_id
            yield self.vocab.get_id(text)
            if add_eos:
                yield self.vocab.eos_id
            return

        chunks = self._pretok.pretokenize(text, lang=lang)

        if add_bos:
            yield self.vocab.bos_id

        for chunk in chunks:
            # 청크 단위 특수 토큰 체크
            if (
                chunk in self.vocab.token2id
                and chunk.startswith("<|")
                and chunk.endswith("|>")
            ):
                yield self.vocab.get_id(chunk)
            else:
                tokens = self._bpe_encode_chunk(chunk)
                for token in tokens:
                    yield self.vocab.get_id(token)

        if add_eos:
            yield self.vocab.eos_id

    def encode_iter(
        self,
        text: str,
        lang: Optional[str] = None,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> Iterator[int]:
        """iter_encode()의 명시적 alias."""
        return self.iter_encode(
            text,
            lang=lang,
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def _bpe_encode_chunk(self, chunk: str) -> List[str]:
        """
        단일 청크에 BPE 인코딩 적용.
        학습 시 사용한 병합 규칙을 rank 순서대로 탐욕적으로 적용.
        """
        if self._bpe_encode_chunk_cached is None:
            return self._bpe_encode_chunk_uncached(chunk)
        return list(self._bpe_encode_chunk_cached(chunk))

    def _bpe_encode_chunk_uncached(self, chunk: str) -> List[str]:
        """
        캐시 미스 시 실제 greedy BPE 인코딩을 수행한다.

        기존 구현은 merge마다 모든 인접 pair를 다시 스캔했다.
        여기서는 heap으로 가장 낮은 rank의 pair를 꺼내고, linked-list
        인덱스로 병합 주변 pair만 갱신한다.
        """
        tokens = [BYTE_TOKENS[b] for b in chunk.encode("utf-8")]
        if len(tokens) <= 1:
            return tokens

        prev = [i - 1 for i in range(len(tokens))]
        next_idx = [i + 1 for i in range(len(tokens))]
        next_idx[-1] = -1
        active = [True] * len(tokens)
        heap: list[tuple[int, int, str, str]] = []

        def push_pair(left_idx: int) -> None:
            right_idx = next_idx[left_idx]
            if right_idx == -1:
                return
            left = tokens[left_idx]
            right = tokens[right_idx]
            rank = self._merge_ranks.get((left, right))
            if rank is not None:
                heapq.heappush(heap, (rank, left_idx, left, right))

        for idx in range(len(tokens) - 1):
            push_pair(idx)

        while heap:
            _, left_idx, left, right = heapq.heappop(heap)
            right_idx = next_idx[left_idx]
            if (
                right_idx == -1
                or not active[left_idx]
                or not active[right_idx]
                or tokens[left_idx] != left
                or tokens[right_idx] != right
            ):
                continue

            tokens[left_idx] = left + right
            active[right_idx] = False
            next_after_right = next_idx[right_idx]
            next_idx[left_idx] = next_after_right
            if next_after_right != -1:
                prev[next_after_right] = left_idx

            left_before = prev[left_idx]
            if left_before != -1:
                push_pair(left_before)
            push_pair(left_idx)

        merged_tokens: list[str] = []
        idx = 0
        while idx != -1:
            if active[idx]:
                merged_tokens.append(tokens[idx])
            idx = next_idx[idx]
        return merged_tokens

    def cache_info(self) -> dict:
        """Benchmark/reporting용 encode chunk cache 상태."""
        if self._bpe_encode_chunk_cached is None:
            return {
                "max_size": 0,
                "size": 0,
                "hits": 0,
                "misses": 0,
            }
        info = self._bpe_encode_chunk_cached.cache_info()
        return {
            "max_size": info.maxsize,
            "size": info.currsize,
            "hits": info.hits,
            "misses": info.misses,
        }

    def clear_cache(self):
        """동일 조건 반복 실험을 위한 cache 초기화."""
        if self._bpe_encode_chunk_cached is not None:
            self._bpe_encode_chunk_cached.cache_clear()

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
