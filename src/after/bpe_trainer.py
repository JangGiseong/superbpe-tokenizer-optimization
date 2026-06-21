"""
bpe_trainer.py — SuperBPE BPE 학습기

핵심 최적화: Chunk Aggregation
  코퍼스를 한 번만 순회해 {청크 튜플: 빈도} 딕셔너리 구축.
  이후 150,000번의 병합은 딕셔너리만 대상으로 실행.

Stage 2 제약 (논문 Appendix A.1.2, A.1.4):
  - 슈퍼워드 최대 4단어
  - ": " (콜론+공백) 포함 토큰 금지

메모리/속도 특성:
  Stage 1: 고유 단어 수 ~ 수백만 → 빠름
  Stage 2: 고유 세그먼트 수 ~ 수천만 → 느림 (병렬화 필요시 multiprocessing 추가)
"""

from collections import defaultdict
from typing import Dict, Tuple, List, Iterator, Optional, Callable, Iterable
from pretokenizer import Pretokenizer, Stage
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import heapq

# ── 타입 정의 ──────────────────────────────────────────────────────

Chunk = Tuple[str, ...]
ChunkCounts = Dict[Chunk, int]
Pair = Tuple[str, str]
PairCounts = Dict[Pair, int]
MergeList = List[Pair]


# ── 보조 함수 ──────────────────────────────────────────────────────


def word_count_in_token(token: str) -> int:
    """
    토큰 내 공백으로 구분된 단어 수 반환.
    슈퍼워드 4단어 초과 제약 검사에 사용.

    예: "by the way" → 3
        "hello"      → 1
    """
    return len(token.split())


def contains_colon_space(token: str) -> bool:
    """
    ': ' (콜론+공백) 포함 여부.
    논문 Appendix A.1.4: QA 포맷 prompt-completion 경계 왜곡 방지.
    """
    return ": " in token


def text_to_byte_tuple(text: str) -> Chunk:
    """
    문자열을 UTF-8 바이트 단위 튜플로 변환. BPE 초기 상태.
    예: "hello" → ("h", "e", "l", "l", "o")
    바이트 표현을 그대로 쓰되, 표시 가능한 문자로 디코딩.
    """
    return tuple(
        bytes([b]).decode("latin-1")  # latin-1은 0~255 모든 바이트를 1:1 매핑
        for b in text.encode("utf-8")
    )


def _apply_all_merges_to_stage2(
    chunk_counts: ChunkCounts,
    merges: MergeList,
) -> ChunkCounts:
    """
    Stage 2 청크에 Stage 1 병합 전체를 한 번에 적용.

    기존 방식: apply_merge() × 119,744번 = O(merges × chunks) → 16시간
    새 방식:   각 청크를 BPE 인코딩 = O(chunks × len) → ~1분
    """
    merge_ranks = {pair: rank for rank, pair in enumerate(merges)}
    new_counts: ChunkCounts = defaultdict(int)
    total = len(chunk_counts)

    for done, (chunk, freq) in enumerate(chunk_counts.items()):
        tokens = list(chunk)

        # BPE 탐욕 인코딩: 낮은 rank(먼저 학습된 병합)부터 적용
        while len(tokens) > 1:
            best_rank = float("inf")
            best_idx = -1
            for i in range(len(tokens) - 1):
                rank = merge_ranks.get((tokens[i], tokens[i + 1]), float("inf"))
                if rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx == -1:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens = tokens[:best_idx] + [merged] + tokens[best_idx + 2 :]

        new_counts[tuple(tokens)] += freq

        if (done + 1) % 10_000 == 0:
            print(f"  {done+1:,} / {total:,} 청크 처리 완료", end="\r")

    print(f"  {total:,} / {total:,} 청크 처리 완료")
    return dict(new_counts)


# ── 병렬 처리용 워커 함수 (모듈 레벨 필수) ──────────────────────────


def _chunk_worker(args):
    """
    각 프로세스에서 독립 실행.
    kiwipiepy는 각 프로세스 안에서 생성 (pickle 불필요).
    """
    docs_batch, langs_batch, stage_val, max_doc_len, max_segment_len = args

    from pretokenizer import Pretokenizer, Stage

    stage = Stage.STAGE1 if stage_val == 1 else Stage.STAGE2
    pretok = Pretokenizer(stage=stage, use_mecab=True, max_segment_len=max_segment_len)

    local: ChunkCounts = defaultdict(int)
    for doc, lang in zip(docs_batch, langs_batch):
        if not doc or not doc.strip():
            continue
        if max_doc_len and len(doc) > max_doc_len:
            doc = doc[:max_doc_len]
        for chunk in pretok.pretokenize(doc, lang=lang):
            if chunk:
                bc = tuple(bytes([b]).decode("latin-1") for b in chunk.encode("utf-8"))
                if bc:
                    local[bc] += 1
    return dict(local)


def _count_pairs_worker(chunk_items_batch):
    """청크 배치의 인접 쌍 빈도 집계"""
    local: PairCounts = defaultdict(int)
    for chunk, freq in chunk_items_batch:
        for i in range(len(chunk) - 1):
            local[(chunk[i], chunk[i + 1])] += freq
    return dict(local)


def _apply_merge_worker(args):
    """청크 배치에 단일 병합 적용"""
    chunk_items_batch, a, b = args
    merged = a + b
    result: ChunkCounts = defaultdict(int)
    for chunk, freq in chunk_items_batch:
        if a not in chunk:
            result[chunk] += freq
            continue
        new_chunk, i = [], 0
        while i < len(chunk):
            if i < len(chunk) - 1 and chunk[i] == a and chunk[i + 1] == b:
                new_chunk.append(merged)
                i += 2
            else:
                new_chunk.append(chunk[i])
                i += 1
        result[tuple(new_chunk)] += freq
    return dict(result)


# ── Chunk Aggregation ─────────────────────────────────────────────


def _iter_batches(
    texts: Iterator[str],
    lang_fn=None,
    batch_size: int = 1024,
) -> Iterator[tuple[list[str], list[str]]]:
    docs_batch: list[str] = []
    langs_batch: list[str] = []
    for doc in texts:
        docs_batch.append(doc)
        langs_batch.append(lang_fn(doc) if lang_fn else "en")
        if len(docs_batch) >= batch_size:
            yield docs_batch, langs_batch
            docs_batch, langs_batch = [], []
    if docs_batch:
        yield docs_batch, langs_batch


def _count_batch_direct(
    docs_batch: list[str],
    langs_batch: list[str],
    pretokenizer: Pretokenizer,
    max_doc_len=None,
) -> ChunkCounts:
    local: ChunkCounts = defaultdict(int)
    for doc, lang in zip(docs_batch, langs_batch):
        if not doc or not doc.strip():
            continue
        if max_doc_len and len(doc) > max_doc_len:
            doc = doc[:max_doc_len]
        for chunk in pretokenizer.pretokenize(doc, lang=lang):
            if chunk:
                bc = tuple(bytes([b]).decode("latin-1") for b in chunk.encode("utf-8"))
                if bc:
                    local[bc] += 1
    return dict(local)


def build_chunk_counts(
    texts: Iterator[str],
    pretokenizer: Pretokenizer,
    lang_fn=None,
    max_doc_len=None,
    n_workers: int = None,
    batch_size: int = 1024,
) -> ChunkCounts:
    """
    코퍼스 → 청크 빈도 딕셔너리.
    after 버전은 전체 iterator를 list로 materialize하지 않고 batch 단위로 처리한다.
    n_workers: None이면 CPU 코어 수 자동 감지 (Windows는 물리 코어 수 권장)
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    t0 = time.time()
    merged: ChunkCounts = defaultdict(int)
    total_docs = 0

    if n_workers <= 1:
        print(f"  streaming 청크 구축: 단일 프로세스, batch_size={batch_size:,}")
        for idx, (docs_batch, langs_batch) in enumerate(
            _iter_batches(texts, lang_fn=lang_fn, batch_size=batch_size)
        ):
            total_docs += len(docs_batch)
            local = _count_batch_direct(
                docs_batch,
                langs_batch,
                pretokenizer=pretokenizer,
                max_doc_len=max_doc_len,
            )
            for k, v in local.items():
                merged[k] += v
            elapsed = time.time() - t0
            print(
                f"  배치 {idx+1} 완료 [{elapsed:.0f}s] "
                f"문서: {total_docs:,} | 고유 청크: {len(merged):,}",
                end="\r",
            )
        print()
    else:
        pending = []
        max_pending = max(1, n_workers * 2)
        batch_idx = 0

        print(f"  streaming 병렬 청크 구축: {n_workers}개 워커, batch_size={batch_size:,}")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for docs_batch, langs_batch in _iter_batches(
                texts, lang_fn=lang_fn, batch_size=batch_size
            ):
                total_docs += len(docs_batch)
                batch_idx += 1
                pending.append(
                    executor.submit(
                        _chunk_worker,
                        (
                            docs_batch,
                            langs_batch,
                            pretokenizer.stage.value,
                            max_doc_len,
                            pretokenizer.max_segment_len,
                        ),
                    )
                )

                if len(pending) >= max_pending:
                    future = pending.pop(0)
                    local = future.result()
                    for k, v in local.items():
                        merged[k] += v
                    elapsed = time.time() - t0
                    print(
                        f"  배치 {batch_idx-len(pending)}/{batch_idx} 반영 [{elapsed:.0f}s] "
                        f"문서: {total_docs:,} | 고유 청크: {len(merged):,}",
                        end="\r",
                    )

            for done_idx, future in enumerate(pending, start=1):
                local = future.result()
                for k, v in local.items():
                    merged[k] += v
                elapsed = time.time() - t0
                print(
                    f"  잔여 배치 {done_idx}/{len(pending)} 반영 [{elapsed:.0f}s] "
                    f"문서: {total_docs:,} | 고유 청크: {len(merged):,}",
                    end="\r",
                )
        print()

    # min_freq 필터: 1회 등장 청크 제거
    before = len(merged)
    merged = {k: v for k, v in merged.items() if v >= 2}
    print(
        f"  완료 [{time.time()-t0:.0f}s] | "
        f"1회 등장 청크 {before-len(merged):,}개 제거 | "
        f"문서: {total_docs:,} | 최종 고유 청크: {len(merged):,}"
    )
    return merged


# ── 쌍 빈도 집계 ──────────────────────────────────────────────────


def count_pairs(
    chunk_counts: ChunkCounts,
    n_workers: int = None,
) -> PairCounts:
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    items = list(chunk_counts.items())
    if len(items) < 50_000 or n_workers == 1:
        # 소규모면 단일 프로세스가 오버헤드 없이 더 빠름
        result: PairCounts = defaultdict(int)
        for chunk, freq in items:
            for i in range(len(chunk) - 1):
                result[(chunk[i], chunk[i + 1])] += freq
        return result

    batch_size = max(1, len(items) // n_workers)
    batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

    merged: PairCounts = defaultdict(int)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for local in executor.map(_count_pairs_worker, batches):
            for k, v in local.items():
                merged[k] += v
    return merged


def apply_merge(
    chunk_counts: ChunkCounts,
    pair: Pair,
    n_workers: int = None,
) -> ChunkCounts:
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    items = list(chunk_counts.items())
    if len(items) < 50_000 or n_workers == 1:
        # 소규모: 단일 프로세스
        a, b = pair
        merged_tok = a + b
        new: ChunkCounts = {}
        for chunk, freq in items:
            if a not in chunk:
                new[chunk] = new.get(chunk, 0) + freq
                continue
            nc, i = [], 0
            while i < len(chunk):
                if i < len(chunk) - 1 and chunk[i] == a and chunk[i + 1] == b:
                    nc.append(merged_tok)
                    i += 2
                else:
                    nc.append(chunk[i])
                    i += 1
            key = tuple(nc)
            new[key] = new.get(key, 0) + freq
        return new

    batch_size = max(1, len(items) // n_workers)
    batches = [
        (items[i : i + batch_size], pair[0], pair[1])
        for i in range(0, len(items), batch_size)
    ]

    new: ChunkCounts = defaultdict(int)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for local in executor.map(_apply_merge_worker, batches):
            for k, v in local.items():
                new[k] += v
    return new


# ── BPETrainer ─────────────────────────────────────────────────────


class BPETrainer:
    """
    SuperBPE 학습기.

    Args:
      vocab_size:          최종 어휘 크기 (기본: 150,000)
      transition_point:    Stage 1→2 전환 시점 (기본: 120,000)
      max_superword_words: 슈퍼워드 최대 단어 수 (기본: 4, 논문 A.1.2)
      log_interval:        몇 개 병합마다 로그 출력
    """

    INITIAL_VOCAB_SIZE = 256  # UTF-8 바이트 초기 어휘

    def __init__(
        self,
        vocab_size: int = 150_000,
        transition_point: int = 120_000,
        max_superword_words: int = 4,
        log_interval: int = 1_000,
    ):
        assert (
            transition_point < vocab_size
        ), f"transition_point({transition_point}) < vocab_size({vocab_size}) 필요"

        self.vocab_size = vocab_size
        self.transition_point = transition_point
        self.max_superword_words = max_superword_words
        self.log_interval = log_interval

        # 초기 어휘: latin-1 0~255 (UTF-8 바이트 1:1 매핑)
        self.vocab: Dict[str, int] = {
            bytes([i]).decode("latin-1"): i for i in range(self.INITIAL_VOCAB_SIZE)
        }
        self.merges: MergeList = []

    # ── 공개 메서드 ───────────────────────────────────────────────

    def train(
        self,
        texts_iter_factory: Callable[[], Iterator[str]],
        lang_fn=None,
        max_doc_len=None,
        stage2_max_docs: int = 100_000,  # ← 추가: Stage 2 최대 문서 수
        n_workers: int = None,
    ):
        if n_workers is None:
            n_workers = max(1, multiprocessing.cpu_count() - 1)

        # ── Stage 1 ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[Stage 1] 서브워드 학습  목표: {self.transition_point:,} 토큰")
        print(f"{'='*60}")

        pretok1 = Pretokenizer(stage=Stage.STAGE1, use_mecab=True)
        print("청크 빈도 딕셔너리 구축 중 (Stage 1)...")
        chunk_counts = build_chunk_counts(
            texts_iter_factory(),
            pretok1,
            lang_fn,
            max_doc_len,
            n_workers=n_workers,
        )
        print(f"고유 청크: {len(chunk_counts):,}개\n")
        chunk_counts = self._run_bpe_loop(chunk_counts, self.transition_point, 1)

        # ── Stage 2 ─────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"[Stage 2] 슈퍼워드 학습  목표: {self.vocab_size:,} 토큰")
        print(f"{'='*60}")

        # Stage 2: 전체 코퍼스 대신 대표 샘플만 사용
        # 슈퍼워드는 빈출 표현 학습 → 코퍼스의 3~5% 샘플로 충분
        def _stage2_sample() -> Iterator[str]:
            count = 0
            for text in texts_iter_factory():  # 이미 셔플된 코퍼스
                if count >= stage2_max_docs:
                    break
                yield text
                count += 1

        # Stage 2: 세그먼트 길이 줄여 중복 증가 → unique chunks 감소
        pretok2 = Pretokenizer(
            stage=Stage.STAGE2,
            max_segment_len=150,  # 1000 → 150 (중복 증가로 메모리 절약)
        )

        print("청크 빈도 딕셔너리 재구축 중 (Stage 2, 경계 없음)...")
        print(f"  샘플 상한: {stage2_max_docs:,}개 문서")
        chunk_counts_s2 = build_chunk_counts(
            _stage2_sample(),
            pretok2,
            lang_fn,
            max_doc_len,
            n_workers=n_workers,
        )
        print(f"고유 청크: {len(chunk_counts_s2):,}개")

        print("Stage 1 병합 결과 적용 중...")
        t0 = time.time()
        chunk_counts_s2 = _apply_all_merges_to_stage2(chunk_counts_s2, self.merges)
        print(f"  완료 [{time.time()-t0:.0f}s]")

        chunk_counts_s2 = self._run_bpe_loop(chunk_counts_s2, self.vocab_size, 2)

        print(f"\n{'='*60}")
        print(
            f"학습 완료\n  최종 어휘 크기: {len(self.vocab):,}\n  총 병합 수: {len(self.merges):,}"
        )
        print(f"{'='*60}")

    # ── 내부 메서드 ───────────────────────────────────────────────

    def _add_token(self, token: str):
        if token not in self.vocab:
            self.vocab[token] = len(self.vocab)

    def _run_bpe_loop(
        self,
        chunk_counts: ChunkCounts,
        target: int,
        stage_num: int,
    ) -> ChunkCounts:
        """
        차분 업데이트(Differential Update) BPE.

        핵심 아이디어:
        - 초기 1회: 모든 pair 빈도 집계 + 위치 인덱스 구축
        - 이후: merge 후 영향받은 pair만 업데이트 → O(pair 등장 횟수)
        - Max-heap으로 최빈 pair O(log n) 탐색

        기존 방식 대비 예상 속도: 100~1000배 향상
        """
        stage_label = "서브워드" if stage_num == 1 else "슈퍼워드"

        # ── 초기 상태 구축 ─────────────────────────────────────────────
        # 청크를 ID로 관리 (tuple 불변성 우회)
        id_to_tokens: Dict[int, list] = {}
        id_to_freq: Dict[int, int] = {}
        for cid, (chunk, freq) in enumerate(chunk_counts.items()):
            id_to_tokens[cid] = list(chunk)
            id_to_freq[cid] = freq

        # pair → 해당 pair를 포함하는 청크 ID 집합
        pair_to_ids: Dict[Pair, set] = defaultdict(set)
        pair_counts: PairCounts = defaultdict(int)

        for cid, tokens in id_to_tokens.items():
            freq = id_to_freq[cid]
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
                pair_to_ids[pair].add(cid)

        # Max-heap (음수 빈도 사용, lazy deletion 방식)
        heap = [(-freq, pair) for pair, freq in pair_counts.items() if freq > 0]
        heapq.heapify(heap)

        # ── Merge 루프 ─────────────────────────────────────────────────
        invalid_pairs: set = set()
        done_merges = 0
        total_merges = target - len(self.vocab)
        t_start = t_last = time.time()

        while len(self.vocab) < target:

            # 최빈 유효 pair 탐색 (lazy deletion)
            best_pair = None
            while heap:
                neg_freq, pair = heapq.heappop(heap)
                if pair in invalid_pairs:
                    continue
                actual = pair_counts.get(pair, 0)
                if actual <= 0:
                    continue
                if actual != -neg_freq:  # stale entry
                    heapq.heappush(heap, (-actual, pair))
                    continue

                # Stage 2 제약 검사
                if stage_num == 2:
                    cand = pair[0] + pair[1]
                    if word_count_in_token(cand) > self.max_superword_words:
                        invalid_pairs.add(pair)
                        continue
                    if contains_colon_space(cand):
                        invalid_pairs.add(pair)
                        continue
                    if cand.endswith(" ") and word_count_in_token(cand) >= 2:
                        invalid_pairs.add(pair)
                        continue

                best_pair = pair
                break

            if best_pair is None:
                print(f"  [경고] 유효한 쌍 소진. 조기 종료.")
                break

            a, b = best_pair
            merged = a + b

            # ── 영향받는 청크 차분 업데이트 ────────────────────────────
            affected = list(pair_to_ids.pop(best_pair, set()))

            for cid in affected:
                tokens = id_to_tokens[cid]
                freq = id_to_freq[cid]

                new_tokens: list = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        # 왼쪽 이웃 pair 업데이트
                        if new_tokens:
                            lp = (new_tokens[-1], a)
                            np_l = (new_tokens[-1], merged)
                            pair_counts[lp] -= freq
                            if pair_counts[lp] <= 0:
                                pair_counts.pop(lp, None)
                                pair_to_ids[lp].discard(cid)
                            else:
                                pair_to_ids[lp].discard(cid)
                                heapq.heappush(heap, (-pair_counts[lp], lp))
                            pair_counts[np_l] += freq
                            pair_to_ids[np_l].add(cid)
                            heapq.heappush(heap, (-pair_counts[np_l], np_l))

                        # 오른쪽 이웃 pair 업데이트
                        if i + 2 < len(tokens):
                            rp = (b, tokens[i + 2])
                            np_r = (merged, tokens[i + 2])
                            pair_counts[rp] -= freq
                            if pair_counts[rp] <= 0:
                                pair_counts.pop(rp, None)
                                pair_to_ids[rp].discard(cid)
                            else:
                                pair_to_ids[rp].discard(cid)
                                heapq.heappush(heap, (-pair_counts[rp], rp))
                            pair_counts[np_r] += freq
                            pair_to_ids[np_r].add(cid)
                            heapq.heappush(heap, (-pair_counts[np_r], np_r))

                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1

                id_to_tokens[cid] = new_tokens

            # best_pair 제거 (모든 청크에서 소멸됨)
            pair_counts.pop(best_pair, None)

            # vocab 업데이트
            self._add_token(merged)
            self.merges.append(best_pair)
            done_merges += 1

            # 로그
            now = time.time()
            if now - t_last >= 30 or done_merges % self.log_interval == 0:
                elapsed = now - t_start
                speed = done_merges / max(elapsed, 1e-9)
                eta = (total_merges - done_merges) / max(speed, 1e-9)
                pct = done_merges / max(total_merges, 1) * 100
                print(
                    f"  [{elapsed:6.0f}s] [{stage_label}] {pct:5.1f}% "
                    f"({done_merges:,}/{total_merges:,}) | vocab={len(self.vocab):,} | "
                    f"속도={speed:.1f} merge/s | ETA={eta/60:.0f}분 | "
                    f"최근: '{a}'+'{b}'→'{merged}'"
                )
                t_last = now

        # ── chunk_counts 형태로 재변환하여 반환 ─────────────────────────
        new_chunk_counts: ChunkCounts = defaultdict(int)
        for cid, tokens in id_to_tokens.items():
            new_chunk_counts[tuple(tokens)] += id_to_freq[cid]

        return dict(new_chunk_counts)

    def _find_best_pair(
        self,
        pair_counts: PairCounts,
        invalid_pairs: set,
        stage_num: int,
    ) -> Optional[Pair]:
        for pair in sorted(pair_counts, key=pair_counts.__getitem__, reverse=True):
            if pair in invalid_pairs:
                continue

            if stage_num == 2:
                candidate = pair[0] + pair[1]

                # 4단어 초과 금지
                if len(candidate.strip().split()) > self.max_superword_words:
                    invalid_pairs.add(pair)
                    continue

                # ': ' 포함 금지 (논문 Appendix A.1.4)
                if contains_colon_space(candidate):
                    invalid_pairs.add(pair)
                    continue

                # 후행 공백 금지 — 슈퍼워드에만 적용
                # ' the' 같은 선행 공백 서브워드는 허용
                # 'by the way ' 같은 후행 공백 슈퍼워드는 금지
                if candidate.endswith(" ") and len(candidate.strip().split()) >= 2:
                    invalid_pairs.add(pair)
                    continue

            return pair

        return None
