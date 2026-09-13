from typing import Dict, List, Optional, Sequence, Tuple

from .chunker import Chunk


class Assignment:
    def __init__(
        self,
        field_name: str,
        chunk: Chunk,
        score: float,
        margin: float,
        rival: str = "",
        source: str = "greedy",
    ):
        self.field_name = field_name
        self.chunk = chunk
        self.score = float(score)
        self.margin = float(margin)
        self.rival = rival
        self.source = source

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def value(self) -> str:
        return self.chunk.value_part or self.chunk.text

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "text": self.text,
            "value": self.value,
            "start": self.chunk.start,
            "end": self.chunk.end,
            "score": round(self.score, 6),
            "margin": round(self.margin, 6),
            "rival": self.rival,
            "source": self.source,
            "format": self.chunk.property_format,
        }

    def __repr__(self) -> str:
        return (
            f"<Assign {self.field_name} ← '{self.text}' "
            f"score={self.score:+.4f} margin={self.margin:+.4f} ({self.source})>"
        )


def _compute_margins(
    chunks: Sequence[Chunk],
    scored: Optional[Dict[int, List]] = None,
) -> Dict[int, Tuple[float, str]]:
    out: Dict[int, Tuple[float, str]] = {}
    if not scored:
        for i, c in enumerate(chunks):
            out[i] = (c.score, "")
        return out

    for i, c in enumerate(chunks):
        ranked = scored.get(i)
        if not ranked:
            out[i] = (c.score, "")
            continue
        top = ranked[0]
        if len(ranked) > 1:
            second = ranked[1]
            out[i] = (top.surprisal - second.surprisal, second.field_name)
        else:
            out[i] = (top.surprisal, "")
    return out


def exclusive_assign_for_indexing(
    chunks: Sequence[Chunk],
    field_names: Sequence[str],
    scored: Optional[Dict[int, List]] = None,
    margin_threshold: float = 0.0,
    multi_value_fields: Optional[Sequence[str]] = None,
    nlp=None,
    log: Optional[List[str]] = None,
) -> Tuple[List[Assignment], List[Chunk]]:
    multi = set(multi_value_fields or ())
    margins = _compute_margins(chunks, scored)

    indexed = [
        (i, c) for i, c in enumerate(chunks)
        if c.status not in ("format-rejected", "absorbed")
        and c.property
        and c.property != "unclassified"
    ]

    if nlp is not None and getattr(nlp, "available", False):
        filtered = []
        dropped = 0
        for i, c in indexed:
            if c.confirmed:
                filtered.append((i, c))
                continue
            target = c.value_part or c.text
            if nlp.is_entity_candidate(target, default=True):
                filtered.append((i, c))
                continue
            c.status = "nlp-rejected"
            dropped += 1
            if log is not None:
                v = nlp.analyze(target)
                reason = v.reason if v else "기능어"
                log.append(f"🧬 NLP 게이트 탈락 '{c.text}' ({reason})")
        if dropped and log is not None:
            log.append(f"🧬 NLP 게이트: {dropped}건 제거 / {len(filtered)}건 통과")
        indexed = filtered

    assignments: List[Assignment] = []
    taken_fields = set()
    taken_spans: List[Tuple[int, int]] = []
    leftovers: List[Chunk] = []

    def _span_taken(chunk: Chunk) -> bool:
        for s, e in taken_spans:
            if chunk.start < e and chunk.end > s:
                return True
        return False

    confirmed = [(i, c) for i, c in indexed if c.confirmed]
    others = [(i, c) for i, c in indexed if not c.confirmed]

    for i, c in confirmed:
        margin, rival = margins.get(i, (c.score, ""))
        if c.property in taken_fields and c.property not in multi:
            c.status = "field-taken"
            leftovers.append(c)
            if log is not None:
                log.append(f"🔒 '{c.property}' 이미 확정됨 → '{c.text}' 차단")
            continue
        c.margin = margin
        c.status = "assigned"
        taken_fields.add(c.property)
        taken_spans.append((c.start, c.end))
        assignments.append(
            Assignment(c.property, c, c.score, margin, rival, source="confirmed")
        )
        if log is not None:
            log.append(
                f"✅ CONFIRMED 배정 '{c.property}' ← '{c.text}' "
                f"(score={c.score:+.4f}, margin={margin:+.4f})"
            )

    others.sort(key=lambda t: (-t[1].score, t[1].start))

    for i, c in others:
        margin, rival = margins.get(i, (c.score, ""))

        if c.property in taken_fields and c.property not in multi:
            c.status = "field-taken"
            leftovers.append(c)
            if log is not None:
                log.append(f"🔒 '{c.property}' 선점됨 → '{c.text}' 차단")
            continue

        if _span_taken(c):
            c.status = "span-taken"
            leftovers.append(c)
            if log is not None:
                log.append(f"🔒 스팬 [{c.start}:{c.end}] 선점됨 → '{c.text}' 차단")
            continue

        if margin < margin_threshold:
            c.status = "below-margin"
            c.margin = margin
            leftovers.append(c)
            if log is not None:
                log.append(
                    f"⚠️ '{c.property}' margin 부족 "
                    f"({margin:+.4f} < {margin_threshold:.4f}) → '{c.text}' 폐기"
                )
            continue

        c.margin = margin
        c.status = "assigned"
        taken_fields.add(c.property)
        taken_spans.append((c.start, c.end))
        assignments.append(
            Assignment(c.property, c, c.score, margin, rival, source="greedy")
        )
        if log is not None:
            log.append(
                f"✨ 배정 '{c.property}' ← '{c.text}' "
                f"(score={c.score:+.4f}, margin={margin:+.4f} vs '{rival or '-'}')"
            )

    missing = [f for f in field_names if f not in taken_fields]
    if missing and log is not None:
        log.append(f"⚪ 미배정 필드 {len(missing)}개: {', '.join(missing)}")

    return assignments, leftovers


def assignments_to_record(
    assignments: Sequence[Assignment],
    field_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {}
    for a in assignments:
        if a.field_name in record:
            prev = record[a.field_name]
            if isinstance(prev, list):
                prev.append(a.value)
            else:
                record[a.field_name] = [prev, a.value]
        else:
            record[a.field_name] = a.value

    if field_names:
        for f in field_names:
            record.setdefault(f, None)
    return record