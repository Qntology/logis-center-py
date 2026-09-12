import re
import unicodedata
from typing import Dict, List, Optional, Sequence, Set, Tuple

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?。！？])\s+|\n+")

WORD_SPLIT_RE = re.compile(
    r"[^\w\u00C0-\u024F\u0370-\u1FFF\u2C00-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]+",
    re.UNICODE,
)

CJK_RANGES = (
    (0x4E00, 0x9FFF),
    (0x3400, 0x4DBF),
    (0x3040, 0x30FF),
    (0xAC00, 0xD7A3),
)

REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}", re.UNICODE)

PUNCT_EDGE_RE = re.compile(
    r"^[^\w\u00C0-\uFFFF]+|[^\w\u00C0-\uFFFF]+$",
    re.UNICODE,
)

DIGIT_GROUP_RE = re.compile(r"[\s,._'](?=\d{3}\b)", re.UNICODE)

MAX_WINDOWS = 48
MAX_WINDOW_WORDS = 6
MIN_WINDOW_WORDS = 1
SECONDARY_SPLIT_CHARS = 150


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    for lo, hi in CJK_RANGES:
        if lo <= cp <= hi:
            return True
    return False


def split_words(text: str) -> List[str]:
    if not text:
        return []
    raw = [w for w in WORD_SPLIT_RE.split(text) if w]
    out: List[str] = []
    for token in raw:
        if any(_is_cjk(c) for c in token) and len(token) > 4:
            buf = ""
            for ch in token:
                buf += ch
                if len(buf) >= 2:
                    out.append(buf)
                    buf = ""
            if buf:
                out.append(buf)
        else:
            out.append(token)
    return out


def strip_diacritics(text: str) -> str:
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def normalize_surface(word: str) -> str:
    if not word:
        return ""

    w = unicodedata.normalize("NFKC", word).strip()
    if not w:
        return ""

    w = PUNCT_EDGE_RE.sub("", w)
    if not w:
        return ""

    w = REPEAT_CHAR_RE.sub(r"\1\1", w)

    if any(ch.isdigit() for ch in w):
        w = DIGIT_GROUP_RE.sub("", w)

    lowered = w.lower()
    folded = strip_diacritics(lowered)

    return folded if folded else lowered


class Chunk:
    def __init__(
        self,
        text: str,
        start: int,
        end: int,
        track: str = "surface",
        sentence_index: int = 0,
        property_name: str = "",
        confirmed: bool = False,
    ):
        self.text = text
        self.start = int(start)
        self.end = int(end)
        self.track = track
        self.sentence_index = int(sentence_index)
        self.property = property_name
        self.confirmed = bool(confirmed)
        self.value_part = ""
        self.bias_phrases: List[str] = []
        self.prejudice_phrases: List[str] = []
        self.property_format = ""
        self.score = 0.0
        self.margin = 0.0
        self.absorbed: List[dict] = []
        self.status = "alive"

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Chunk") -> bool:
        return self.start < other.end and self.end > other.start

    def key(self) -> Tuple[int, int, str]:
        return (self.start, self.end, self.text)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "track": self.track,
            "sentence_index": self.sentence_index,
            "property": self.property,
            "property_format": self.property_format,
            "confirmed": self.confirmed,
            "value_part": self.value_part,
            "score": round(self.score, 6),
            "margin": round(self.margin, 6),
            "status": self.status,
            "absorbed": list(self.absorbed),
        }

    def __repr__(self) -> str:
        return (
            f"<Chunk [{self.start}:{self.end}] '{self.text}' "
            f"track={self.track} prop={self.property or '-'} "
            f"score={self.score:+.4f}>"
        )


class ChunkerConfig:
    def __init__(
        self,
        min_words: int = MIN_WINDOW_WORDS,
        max_words: int = MAX_WINDOW_WORDS,
        max_windows: int = MAX_WINDOWS,
        dual_track: bool = True,
        secondary_split_chars: int = SECONDARY_SPLIT_CHARS,
        merge_single_word: bool = True,
    ):
        self.min_words = max(1, int(min_words))
        self.max_words = max(self.min_words, int(max_words))
        self.max_windows = max(1, int(max_windows))
        self.dual_track = bool(dual_track)
        self.secondary_split_chars = int(secondary_split_chars)
        self.merge_single_word = bool(merge_single_word)


def split_sentences(text: str, max_chars: int = SECONDARY_SPLIT_CHARS) -> List[str]:
    if not text or not text.strip():
        return []
    raw = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s and s.strip()]
    out: List[str] = []
    for sent in raw:
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        words = sent.split()
        buf: List[str] = []
        cur = 0
        for w in words:
            if cur + len(w) + 1 > max_chars and buf:
                out.append(" ".join(buf))
                buf = [w]
                cur = len(w)
            else:
                buf.append(w)
                cur += len(w) + 1
        if buf:
            out.append(" ".join(buf))
    return out


def _merge_single_word_chunks(chunks: List[str]) -> List[str]:
    if len(chunks) < 2:
        return chunks
    out: List[str] = []
    for c in chunks:
        if out and len(c.split()) <= 1:
            out[-1] = out[-1] + " " + c
        else:
            out.append(c)
    return out


def generate_windows(
    words: Sequence[str],
    config: Optional[ChunkerConfig] = None,
    sentence_index: int = 0,
) -> List[Chunk]:
    cfg = config or ChunkerConfig()
    n = len(words)
    if n == 0:
        return []

    seen: Set[Tuple[int, int, str]] = set()
    out: List[Chunk] = []

    for size in range(cfg.min_words, cfg.max_words + 1):
        if size > n:
            break
        for start in range(0, n - size + 1):
            end = start + size
            slice_words = list(words[start:end])

            surface = " ".join(slice_words).strip()
            if surface:
                key = (start, end, surface)
                if key not in seen:
                    seen.add(key)
                    out.append(Chunk(surface, start, end, "surface", sentence_index))

            if cfg.dual_track:
                stems = [normalize_surface(w) for w in slice_words]
                stems = [s for s in stems if s]
                canonical = " ".join(stems).strip()
                if canonical and canonical != surface:
                    key = (start, end, canonical)
                    if key not in seen:
                        seen.add(key)
                        out.append(Chunk(canonical, start, end, "canonical", sentence_index))

            if len(out) >= cfg.max_windows * 4:
                break
        if len(out) >= cfg.max_windows * 4:
            break

    out.sort(key=lambda c: (-(c.end - c.start), c.start, c.track))
    if len(out) > cfg.max_windows:
        out = out[: cfg.max_windows]
    out.sort(key=lambda c: (c.start, -(c.end - c.start)))
    return out


def split_natural_language_to_chunks(
    text: str,
    config: Optional[ChunkerConfig] = None,
) -> Tuple[List[Chunk], List[str]]:
    cfg = config or ChunkerConfig()

    sentences = split_sentences(text, cfg.secondary_split_chars)
    if cfg.merge_single_word:
        sentences = _merge_single_word_chunks(sentences)

    if not sentences:
        return [], []

    all_chunks: List[Chunk] = []
    all_words: List[str] = []

    for sidx, sent in enumerate(sentences):
        words = split_words(sent)
        if not words:
            continue
        base = len(all_words)
        chunks = generate_windows(words, cfg, sentence_index=sidx)
        for c in chunks:
            c.start += base
            c.end += base
        all_chunks.extend(chunks)
        all_words.extend(words)

    return all_chunks, all_words


def chunks_from_words(
    words: Sequence[str],
    config: Optional[ChunkerConfig] = None,
) -> List[Chunk]:
    return generate_windows(words, config or ChunkerConfig(), sentence_index=0)