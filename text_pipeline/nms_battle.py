from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .chunker import Chunk


class Winner:
    def __init__(self, chunk: Chunk):
        self.chunk = chunk
        self.start = chunk.start
        self.end = chunk.end
        self.field_name = chunk.property
        self.score = chunk.score
        self.absorbed: List[dict] = []

    @property
    def text(self) -> str:
        return self.chunk.text

    def absorb(self, loser: Chunk, reason: str = "overlap") -> None:
        self.absorbed.append({
            "text": loser.text,
            "start": loser.start,
            "end": loser.end,
            "property": loser.property,
            "score": round(loser.score, 6),
            "reason": reason,
        })
        loser.status = "absorbed"
        if loser.start < self.start:
            self.start = loser.start
        if loser.end > self.end:
            self.end = loser.end
        self.chunk.absorbed.append(self.absorbed[-1])
        self.chunk.start = self.start
        self.chunk.end = self.end

    def overlaps_span(self, start: int, end: int) -> bool:
        return self.start < end and self.end > start

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "property": self.field_name,
            "score": round(self.score, 6),
            "absorbed": list(self.absorbed),
        }

    def __repr__(self) -> str:
        return (
            f"<Winner [{self.start}:{self.end}] '{self.text}' "
            f"{self.field_name} score={self.score:+.4f} "
            f"absorbed={len(self.absorbed)}>"
        )


def _sort_key(chunk: Chunk, priority: Optional[Dict[str, int]]) -> Tuple:
    prio = 0
    if priority:
        prio = priority.get(chunk.property, 0)
    return (-prio, -chunk.score, -(chunk.end - chunk.start), chunk.start)


def nms_battle_for_indexing(
    chunks: Sequence[Chunk],
    domain_priority: Optional[Dict[str, int]] = None,
    log: Optional[List[str]] = None,
) -> List[Winner]:
    candidates = [
        c for c in chunks
        if c.status == "alive" and c.property and c.property != "unclassified"
    ]
    if not candidates:
        if log is not None:
            log.append("⚪ NMS 배틀: 경쟁 후보가 없습니다.")
        return []

    candidates = sorted(candidates, key=lambda c: _sort_key(c, domain_priority))

    winners: List[Winner] = []

    for cand in candidates:
        hit: Optional[Winner] = None
        for w in winners:
            if w.overlaps_span(cand.start, cand.end):
                hit = w
                break

        if hit is None:
            winners.append(Winner(cand))
            if log is not None:
                log.append(
                    f"✨ 승자 [{cand.start}:{cand.end}] '{cand.text}' "
                    f"→ {cand.property} ({cand.score:+.4f})"
                )
            continue

        hit.absorb(cand, reason="overlap")
        if log is not None:
            log.append(
                f"🌀 흡수 [{cand.start}:{cand.end}] '{cand.text}' "
                f"({cand.property} {cand.score:+.4f}) → "
                f"'{hit.text}' [{hit.start}:{hit.end}]"
            )

    winners.sort(key=lambda w: w.start)
    return winners


def bridge_gaps(
    winners: List[Winner],
    words: Sequence[str],
    rescore_fn: Optional[Callable[[str, str], float]] = None,
    log: Optional[List[str]] = None,
) -> List[Winner]:
    if not winners or not words:
        return winners

    total = len(words)
    winners.sort(key=lambda w: w.start)

    def _span_text(start: int, end: int) -> str:
        return " ".join(words[max(0, start): min(total, end)])

    def _make_orphan(start: int, end: int) -> Chunk:
        c = Chunk(_span_text(start, end), start, end, "gap")
        return c

    if winners[0].start > 0:
        head = winners[0]
        orphan = _make_orphan(0, head.start)
        if orphan.text.strip():
            head.absorb(orphan, reason="left-edge")
            if log is not None:
                log.append(
                    f"🔗 좌측 고아 '{orphan.text}' → '{head.text}' 흡수"
                )

    if winners[-1].end < total:
        tail = winners[-1]
        orphan = _make_orphan(tail.end, total)
        if orphan.text.strip():
            tail.absorb(orphan, reason="right-edge")
            if log is not None:
                log.append(
                    f"🔗 우측 고아 '{orphan.text}' → '{tail.text}' 흡수"
                )

    idx = 0
    while idx + 1 < len(winners):
        left = winners[idx]
        right = winners[idx + 1]
        if right.start <= left.end:
            idx += 1
            continue

        orphan = _make_orphan(left.end, right.start)
        if not orphan.text.strip():
            idx += 1
            continue

        target = left
        reason = "gap-left"

        if rescore_fn is not None:
            left_text = _span_text(left.start, right.start)
            right_text = _span_text(left.end, right.end)
            try:
                left_score = float(rescore_fn(left_text, left.field_name))
            except Exception:
                left_score = left.score
            try:
                right_score = float(rescore_fn(right_text, right.field_name))
            except Exception:
                right_score = right.score
            if right_score > left_score:
                target = right
                reason = "gap-right"
        else:
            if right.score > left.score:
                target = right
                reason = "gap-right"

        target.absorb(orphan, reason=reason)
        if log is not None:
            log.append(
                f"🔗 중간 갭 '{orphan.text}' → '{target.text}' 흡수 ({reason})"
            )
        idx += 1

    winners.sort(key=lambda w: w.start)
    return winners


def winners_to_chunks(winners: Sequence[Winner]) -> List[Chunk]:
    out: List[Chunk] = []
    for w in winners:
        c = w.chunk
        c.start = w.start
        c.end = w.end
        c.status = "alive"
        out.append(c)
    return out


def coverage_ratio(winners: Sequence[Winner], total_words: int) -> float:
    if total_words <= 0:
        return 0.0
    covered = 0
    last_end = 0
    for w in sorted(winners, key=lambda x: x.start):
        s = max(w.start, last_end)
        if w.end > s:
            covered += w.end - s
            last_end = w.end
    return covered / float(total_words)