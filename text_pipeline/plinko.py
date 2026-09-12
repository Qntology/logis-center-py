from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chunker import Chunk, normalize_surface
from .field_bank import FieldBank
from .surprisal import surprisal_dual_scores


class PlinkoResult:
    def __init__(
        self,
        text: str,
        start: int,
        end: int,
        field_name: str,
        score: float,
        cliff_at: int,
        mode: str,
        trace: Optional[List[Tuple[str, float]]] = None,
    ):
        self.text = text
        self.start = int(start)
        self.end = int(end)
        self.field_name = field_name
        self.score = float(score)
        self.cliff_at = int(cliff_at)
        self.mode = mode
        self.trace = trace or []

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "property": self.field_name,
            "score": round(self.score, 6),
            "cliff_at": self.cliff_at,
            "mode": self.mode,
            "trace": [(t, round(s, 6)) for t, s in self.trace],
        }

    def __repr__(self) -> str:
        return (
            f"<Plinko [{self.start}:{self.end}] '{self.text}' → "
            f"{self.field_name} ({self.score:+.4f}) mode={self.mode}>"
        )


def _best_score(
    text: str,
    bank: FieldBank,
    embed_fn: Callable[[List[str]], np.ndarray],
    cache: Dict[str, Tuple[str, float]],
) -> Tuple[str, float]:
    if not text or not text.strip():
        return "", -np.inf
    if text in cache:
        return cache[text]

    try:
        mat = embed_fn([text])
    except Exception:
        cache[text] = ("", -np.inf)
        return cache[text]

    if mat is None:
        cache[text] = ("", -np.inf)
        return cache[text]

    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.shape[0] == 0:
        cache[text] = ("", -np.inf)
        return cache[text]

    scores = surprisal_dual_scores(mat[0], bank)
    if not scores:
        cache[text] = ("", -np.inf)
        return cache[text]

    top = scores[0]
    cache[text] = (top.field_name, top.surprisal)
    return cache[text]


def plinko_game_for_indexing(
    words: Sequence[str],
    bank: FieldBank,
    embed_fn: Callable[[List[str]], np.ndarray],
    confirmed_chunks: Optional[Sequence[Chunk]] = None,
    max_window: int = 6,
    cliff_ratio: float = 0.75,
    gate: float = 0.0,
    log: Optional[List[str]] = None,
) -> List[PlinkoResult]:
    if not words:
        return []

    total = len(words)
    confirmed_spans: List[Tuple[int, int, str, float]] = []
    for c in (confirmed_chunks or []):
        if getattr(c, "confirmed", False) and c.property and c.property != "unclassified":
            confirmed_spans.append((c.start, c.end, c.property, c.score))
    confirmed_spans.sort(key=lambda t: t[0])

    def _confirmed_at(pos: int) -> Optional[Tuple[int, int, str, float]]:
        for span in confirmed_spans:
            if span[0] <= pos < span[1]:
                return span
        return None

    cache: Dict[str, Tuple[str, float]] = {}
    results: List[PlinkoResult] = []

    cursor = 0
    while cursor < total:
        hit = _confirmed_at(cursor)
        if hit is not None:
            s, e, prop, score = hit
            text = " ".join(words[s:e])
            results.append(
                PlinkoResult(text, s, e, prop, score, cliff_at=e, mode="confirmed-skip")
            )
            if log is not None:
                log.append(
                    f"⏭ PLINKO 건너뛰기 [{s}:{e}] '{text}' → {prop} (확정)"
                )
            cursor = max(e, cursor + 1)
            continue

        limit = min(total, cursor + max_window)
        next_confirm = None
        for span in confirmed_spans:
            if span[0] > cursor:
                next_confirm = span[0]
                break
        if next_confirm is not None:
            limit = min(limit, next_confirm)
        if limit <= cursor:
            cursor += 1
            continue

        trace: List[Tuple[str, float]] = []
        best_field = ""
        best_score = -np.inf
        best_end = cursor + 1
        prev_score = -np.inf
        cliff_at = limit

        for end in range(cursor + 1, limit + 1):
            text = " ".join(words[cursor:end])
            field, score = _best_score(text, bank, embed_fn, cache)
            trace.append((text, score))

            if score > best_score:
                best_score = score
                best_field = field
                best_end = end

            if prev_score > -np.inf and score < prev_score * cliff_ratio and prev_score > gate:
                cliff_at = end - 1
                break

            prev_score = score

        if best_field and best_score > gate:
            text = " ".join(words[cursor:best_end])
            results.append(
                PlinkoResult(
                    text, cursor, best_end, best_field, best_score,
                    cliff_at=cliff_at, mode="cliff",
                )
            )
            if log is not None:
                log.append(
                    f"🎯 PLINKO 확정 [{cursor}:{best_end}] '{text}' → "
                    f"{best_field} ({best_score:+.4f}, cliff@{cliff_at})"
                )
            cursor = best_end
        else:
            text = words[cursor]
            results.append(
                PlinkoResult(
                    text, cursor, cursor + 1, "unclassified", 0.0,
                    cliff_at=cursor + 1, mode="orphan",
                )
            )
            cursor += 1

    return results


def plinko_override(
    chunks: List[Chunk],
    plinko_results: Sequence[PlinkoResult],
    log: Optional[List[str]] = None,
) -> List[Chunk]:
    by_span: Dict[Tuple[int, int], PlinkoResult] = {
        (p.start, p.end): p for p in plinko_results
    }

    for chunk in chunks:
        if chunk.confirmed:
            continue
        hit = by_span.get((chunk.start, chunk.end))
        if hit is None:
            continue
        if hit.field_name in ("", "unclassified"):
            continue
        if hit.field_name == chunk.property:
            continue
        if hit.score <= chunk.score:
            continue

        old = chunk.property or "-"
        chunk.property = hit.field_name
        chunk.score = hit.score
        if log is not None:
            log.append(
                f"🔀 PLINKO OVERRIDE [{chunk.start}:{chunk.end}] "
                f"'{chunk.text}' {old} → {hit.field_name} ({hit.score:+.4f})"
            )

    return chunks


def results_to_chunks(results: Sequence[PlinkoResult]) -> List[Chunk]:
    out: List[Chunk] = []
    for r in results:
        c = Chunk(r.text, r.start, r.end, track="plinko")
        c.property = r.field_name
        c.score = r.score
        c.confirmed = r.mode == "confirmed-skip"
        out.append(c)
    return out