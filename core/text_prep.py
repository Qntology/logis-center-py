import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
BOM_RE = re.compile(r"[\ufeff\u200b-\u200f\u2028\u2029\u202a-\u202e]")
MULTI_WS_RE = re.compile(r"[ \t\u00a0]{2,}")
MULTI_NL_RE = re.compile(r"\n{3,}")
REPEAT_RE = re.compile(r"(.)\1{4,}")

LABEL_SEP_RE = re.compile(r"\s*[:：]\s*")
LABEL_HEAD_RE = re.compile(
    r"^(?P<label>[^\W\d_][\w /&().\-]{1,48}?)\s*[:：]\s*(?P<value>.+)$",
    re.UNICODE,
)

UPPER_LABEL_RE = re.compile(r"^[A-Z][A-Z0-9 /&().\-]{2,48}$")

VALUE_SHAPE_RE = re.compile(
    r"\d|[A-Za-z]{2,}", re.UNICODE
)

MAX_LABEL_WORDS = 6


def sanitize_llm_input(text: str, max_chars: int = 8000) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    s = BOM_RE.sub("", s)
    s = CTRL_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = REPEAT_RE.sub(lambda m: m.group(1) * 3, s)
    s = MULTI_WS_RE.sub(" ", s)
    s = MULTI_NL_RE.sub("\n\n", s)
    lines = [ln.strip() for ln in s.split("\n")]
    s = "\n".join(ln for ln in lines if ln)
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars]
    return s.strip()


def is_printed_label(text: str, label_bank: Optional[Sequence[str]] = None) -> bool:
    t = str(text or "").strip().strip(":： ")
    if not t:
        return True
    words = t.split()
    if len(words) > MAX_LABEL_WORDS:
        return False
    if UPPER_LABEL_RE.match(t) and not any(ch.isdigit() for ch in t):
        return True
    low = _compact(t)
    for cand in (label_bank or []):
        if low and low == _compact(cand):
            return True
    return False


def _compact(text: str) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


class LabelValuePair:
    def __init__(self, label: str, value: str, source: str = "", line: int = -1):
        self.label = str(label or "").strip()
        self.value = str(value or "").strip()
        self.source = source
        self.line = int(line)

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "value": self.value,
            "source": self.source,
            "line": self.line,
        }

    def __repr__(self) -> str:
        return f"<Pair '{self.label}' = '{self.value[:28]}' ({self.source})>"


def collect_label_value_pairs(
    text: str,
    label_bank: Optional[Sequence[str]] = None,
    log: Optional[List[str]] = None,
) -> List[LabelValuePair]:
    body = sanitize_llm_input(text)
    if not body:
        return []

    lines = [ln.strip() for ln in body.split("\n") if ln.strip()]
    out: List[LabelValuePair] = []
    used = set()

    for idx, line in enumerate(lines):
        m = LABEL_HEAD_RE.match(line)
        if not m:
            continue
        label = m.group("label").strip()
        value = m.group("value").strip()
        if not VALUE_SHAPE_RE.search(value):
            continue
        if is_printed_label(value, label_bank):
            continue
        out.append(LabelValuePair(label, value, source="inline", line=idx))
        used.add(idx)

    for idx in range(len(lines) - 1):
        if idx in used or (idx + 1) in used:
            continue
        label = lines[idx]
        value = lines[idx + 1]
        if not is_printed_label(label, label_bank):
            continue
        if is_printed_label(value, label_bank):
            continue
        if not VALUE_SHAPE_RE.search(value):
            continue
        out.append(LabelValuePair(label, value, source="stacked", line=idx))
        used.add(idx)
        used.add(idx + 1)

    if log is not None:
        inline = sum(1 for p in out if p.source == "inline")
        stacked = len(out) - inline
        log.append(
            f"  🏷 [LABEL-VALUE] 쌍 {len(out)}건 추출 "
            f"(같은 줄 {inline} / 다음 줄 {stacked}) | 원문 {len(lines)}줄"
        )
        for p in out[:8]:
            log.append(f"      · '{p.label}' = '{p.value[:36]}'")
    return out


def pairs_to_confirmed_chunks(
    pairs: Sequence[LabelValuePair],
    words: Sequence[str],
):
    from text_pipeline.chunker import Chunk

    out = []
    cursor = 0
    lowered = [str(w).lower() for w in words]

    for p in pairs:
        tokens = [t.lower() for t in p.value.split() if t]
        if not tokens:
            continue
        span = _find_span(lowered, tokens, cursor)
        if span is None:
            span = _find_span(lowered, tokens, 0)
        if span is None:
            continue
        s, e = span
        c = Chunk(p.value, s, e, track="label-pair", confirmed=True)
        c.value_part = p.value
        c.bias_phrases = [p.label]
        out.append(c)
        cursor = e
    return out


def _find_span(
    words: Sequence[str],
    tokens: Sequence[str],
    start: int,
) -> Optional[Tuple[int, int]]:
    n = len(words)
    k = len(tokens)
    if k == 0 or k > n:
        return None
    for i in range(max(0, start), n - k + 1):
        if all(words[i + j] == tokens[j] for j in range(k)):
            return (i, i + k)
    return None


def enrich_chunks_with_metadata(
    chunks,
    label_bank: Optional[Sequence[str]] = None,
    log: Optional[List[str]] = None,
) -> int:
    hits = 0
    for c in chunks:
        text = getattr(c, "text", "") or ""
        if getattr(c, "confirmed", False):
            continue
        if is_printed_label(text, label_bank):
            c.status = "label-echo"
            c.property = "unclassified"
            hits += 1
    if log is not None and hits:
        log.append(
            f"  🧹 [LABEL ECHO GATE] 인쇄 라벨로 판정된 청크 {hits}건을 "
            f"후보에서 제외했습니다."
        )
    return hits