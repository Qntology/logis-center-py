import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PHRASE_SPLIT_RE = re.compile(r"[,;/|]+")

WEIGHT_MARKER_RE = re.compile(r"^\s*(?P<phrase>.+?)\s*(?:\^(?P<weight>[0-9]*\.?[0-9]+))?\s*$")

DEFAULT_SEMANTIC_WEIGHT = 1.25
DEFAULT_BIAS_WEIGHT = 1.00
DEFAULT_PREJUDICE_WEIGHT = 1.00


def split_bias_phrases_weighted(
    raw,
    default_weight: float = DEFAULT_BIAS_WEIGHT,
) -> List[Tuple[str, float]]:
    if raw is None:
        return []

    items: List[str] = []
    if isinstance(raw, str):
        items = [p for p in PHRASE_SPLIT_RE.split(raw)]
    elif isinstance(raw, (list, tuple, set)):
        for elem in raw:
            if elem is None:
                continue
            if isinstance(elem, str):
                items.extend(PHRASE_SPLIT_RE.split(elem))
            else:
                items.append(str(elem))
    else:
        items = [str(raw)]

    out: List[Tuple[str, float]] = []
    seen = set()
    for item in items:
        if item is None:
            continue
        m = WEIGHT_MARKER_RE.match(item)
        if not m:
            continue
        phrase = (m.group("phrase") or "").strip()
        if not phrase:
            continue
        low = phrase.lower()
        if low in seen:
            continue
        seen.add(low)
        wraw = m.group("weight")
        weight = float(wraw) if wraw else float(default_weight)
        out.append((phrase, weight))
    return out


class Phrase:
    def __init__(self, text: str, weight: float, kind: str, owner: str):
        self.text = text
        self.weight = float(weight)
        self.kind = kind
        self.owner = owner
        self.vector: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"<Phrase {self.kind}:{self.owner} '{self.text}' w={self.weight:.2f}>"


class FieldEntry:
    def __init__(self, name: str, definition: dict):
        self.name = name
        self.definition = definition or {}
        self.semantic: str = str(self.definition.get("semantic", "") or "")
        self.bias: List[Phrase] = []
        self.prejudice: List[Phrase] = []
        self.format: str = str(self.definition.get("format", "") or "")

    @property
    def bias_texts(self) -> List[str]:
        return [p.text for p in self.bias]

    @property
    def prejudice_texts(self) -> List[str]:
        return [p.text for p in self.prejudice]

    def bias_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        vecs = [p.vector for p in self.bias if p.vector is not None]
        wts = [p.weight for p in self.bias if p.vector is not None]
        if not vecs:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.stack(vecs).astype(np.float32), np.asarray(wts, dtype=np.float32)

    def prejudice_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        vecs = [p.vector for p in self.prejudice if p.vector is not None]
        wts = [p.weight for p in self.prejudice if p.vector is not None]
        if not vecs:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.stack(vecs).astype(np.float32), np.asarray(wts, dtype=np.float32)

    def bank_size(self) -> int:
        return len([p for p in self.bias if p.vector is not None])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "semantic": self.semantic,
            "format": self.format,
            "bias": [(p.text, p.weight) for p in self.bias],
            "prejudice": [(p.text, p.weight) for p in self.prejudice],
            "bank_size": self.bank_size(),
        }


class FieldBank:
    def __init__(self, domain: str = "", doc_type: str = ""):
        self.domain = domain
        self.doc_type = doc_type
        self.entries: Dict[str, FieldEntry] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self.dim: int = 0

    @property
    def field_names(self) -> List[str]:
        return list(self.entries.keys())

    def get(self, name: str) -> Optional[FieldEntry]:
        return self.entries.get(name)

    def add_field(self, name: str, definition: dict) -> FieldEntry:
        entry = FieldEntry(name, definition)
        self.entries[name] = entry
        return entry

    def build_phrases(self, cross_prejudice: bool = True) -> None:
        for name, entry in self.entries.items():
            phrases: List[Phrase] = []
            if entry.semantic:
                for txt, w in split_bias_phrases_weighted(
                    entry.semantic, DEFAULT_SEMANTIC_WEIGHT
                ):
                    phrases.append(Phrase(txt, w, "bias", name))
            for txt, w in split_bias_phrases_weighted(
                entry.definition.get("bias"), DEFAULT_BIAS_WEIGHT
            ):
                if any(p.text.lower() == txt.lower() for p in phrases):
                    continue
                phrases.append(Phrase(txt, w, "bias", name))
            entry.bias = phrases

        for name, entry in self.entries.items():
            prej: List[Phrase] = []
            seen = {p.text.lower() for p in entry.bias}

            for txt, w in split_bias_phrases_weighted(
                entry.definition.get("prejudice"), DEFAULT_PREJUDICE_WEIGHT
            ):
                if txt.lower() in seen:
                    continue
                seen.add(txt.lower())
                prej.append(Phrase(txt, w, "prejudice", name))

            if cross_prejudice:
                for other_name, other in self.entries.items():
                    if other_name == name:
                        continue
                    for p in other.bias:
                        low = p.text.lower()
                        if low in seen:
                            continue
                        seen.add(low)
                        prej.append(Phrase(p.text, p.weight, "prejudice", name))

            entry.prejudice = prej

    def all_phrase_texts(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for entry in self.entries.values():
            for p in list(entry.bias) + list(entry.prejudice):
                if p.text in seen:
                    continue
                seen.add(p.text)
                out.append(p.text)
        return out

    def embed_phrases(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        batch_size: int = 32,
    ) -> None:
        texts = [t for t in self.all_phrase_texts() if t not in self._embed_cache]

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            try:
                mat = embed_fn(batch)
            except Exception:
                mat = None
            if mat is None:
                continue
            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            for j, t in enumerate(batch):
                if j >= mat.shape[0]:
                    break
                v = mat[j]
                norm = float(np.linalg.norm(v))
                if norm < 1e-8:
                    continue
                self._embed_cache[t] = (v / norm).astype(np.float32)

        for entry in self.entries.values():
            for p in list(entry.bias) + list(entry.prejudice):
                p.vector = self._embed_cache.get(p.text)

        for v in self._embed_cache.values():
            self.dim = int(v.shape[-1])
            break

    def vector_of(self, phrase: str) -> Optional[np.ndarray]:
        return self._embed_cache.get(phrase)

    def stats(self) -> dict:
        return {
            "domain": self.domain,
            "doc_type": self.doc_type,
            "fields": len(self.entries),
            "dim": self.dim,
            "phrases": len(self._embed_cache),
            "per_field": {
                name: {
                    "bias": len(e.bias),
                    "prejudice": len(e.prejudice),
                    "embedded": e.bank_size(),
                }
                for name, e in self.entries.items()
            },
        }

    def report_lines(self) -> List[str]:
        lines = [
            f"  도메인: {self.domain or '-'} / 서식: {self.doc_type or '-'} "
            f"| 필드 {len(self.entries)}개 | 임베딩 차원 {self.dim}"
        ]
        for name, e in self.entries.items():
            lines.append(
                f"    · {name:<28} bias={len(e.bias):3d} "
                f"prej={len(e.prejudice):4d} embedded={e.bank_size():3d}"
            )
        return lines


def build_field_bank(
    schema: dict,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    cross_prejudice: bool = True,
    batch_size: int = 32,
) -> FieldBank:
    bank = FieldBank(
        domain=str(schema.get("domain", "") or ""),
        doc_type=str(schema.get("doc_type", "") or ""),
    )
    fields = schema.get("fields", {}) or {}
    for name, definition in fields.items():
        bank.add_field(str(name), definition if isinstance(definition, dict) else {})

    bank.build_phrases(cross_prejudice=cross_prejudice)

    if embed_fn is not None:
        bank.embed_phrases(embed_fn, batch_size=batch_size)

    return bank