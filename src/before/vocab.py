"""
vocab.py — Vocabulary 관리

BPE 학습 완료 후:
  1. 특수 토큰 추가 (BPE 병합 대상 아님)
  2. 토큰 ↔ ID 양방향 매핑
  3. JSON 직렬화/역직렬화
"""

import json
from typing import Dict, List, Optional


# 논문 및 K-EXAONE 기준 특수 토큰
# Chat template 구분자는 BPE 학습 후 별도 추가
DEFAULT_SPECIAL_TOKENS: List[str] = [
    "<|pad|>",  # 패딩
    "<|bos|>",  # Begin of Sequence
    "<|eos|>",  # End of Sequence
    "<|unk|>",  # Unknown (바이트 폴백으로 실사용 드묾)
    "<|system|>",  # Chat template — 시스템 역할
    "<|user|>",  # Chat template — 사용자 역할
    "<|assistant|>",  # Chat template — 어시스턴트 역할
]


class Vocabulary:
    """
    토큰 ↔ ID 양방향 매핑 관리.

    특수 토큰은 BPE 학습 어휘 뒤에 추가된다.
    이렇게 하면 모델 임베딩 행렬에서 일반 토큰 ID와 분리된다.
    """

    def __init__(
        self,
        token2id: Optional[Dict[str, int]] = None,
        merges: Optional[List] = None,
    ):
        self.token2id: Dict[str, int] = dict(token2id) if token2id else {}
        self.id2token: Dict[int, str] = {v: k for k, v in self.token2id.items()}
        self.merges: List = list(merges) if merges else []

    # ── 생성 ─────────────────────────────────────────────────────

    @classmethod
    def from_trainer(cls, trainer) -> "Vocabulary":
        """BPETrainer 완료 후 Vocabulary 생성."""
        return cls(
            token2id=dict(trainer.vocab),
            merges=list(trainer.merges),
        )

    # ── 특수 토큰 ─────────────────────────────────────────────────

    def add_special_tokens(self, tokens: Optional[List[str]] = None):
        """
        특수 토큰을 어휘 끝에 추가.
        이미 존재하는 토큰은 건너뜀.
        Chat template 토큰은 모델이 역할 경계를 인식하는 데 필수.
        """
        tokens = tokens if tokens is not None else DEFAULT_SPECIAL_TOKENS
        added = []
        for token in tokens:
            if token not in self.token2id:
                new_id = len(self.token2id)
                self.token2id[token] = new_id
                self.id2token[new_id] = token
                added.append(token)

        if added:
            print(f"특수 토큰 {len(added)}개 추가: {added}")

    # ── 조회 ─────────────────────────────────────────────────────

    @property
    def unk_id(self) -> int:
        return self.token2id.get("<|unk|>", 0)

    @property
    def pad_id(self) -> int:
        return self.token2id.get("<|pad|>", 0)

    @property
    def bos_id(self) -> int:
        return self.token2id.get("<|bos|>", 1)

    @property
    def eos_id(self) -> int:
        return self.token2id.get("<|eos|>", 2)

    def get_id(self, token: str) -> int:
        return self.token2id.get(token, self.unk_id)

    def get_token(self, idx: int) -> str:
        return self.id2token.get(idx, "<|unk|>")

    def __len__(self) -> int:
        return len(self.token2id)

    def __contains__(self, token: str) -> bool:
        return token in self.token2id

    # ── 직렬화 ────────────────────────────────────────────────────

    def save(self, path: str):
        """JSON으로 저장."""
        data = {
            "vocab_size": len(self.token2id),
            "vocab": self.token2id,
            "merges": [list(pair) for pair in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(
            f"저장 완료: {path}  (어휘: {len(self):,}개, 병합: {len(self.merges):,}개)"
        )

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        """JSON에서 불러오기."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(
            token2id=data["vocab"],
            merges=[tuple(p) for p in data["merges"]],
        )
        print(f"불러오기 완료: {path}  (어휘: {len(vocab):,}개)")
        return vocab
