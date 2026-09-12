import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PHRASE_SPLIT_RE = re.compile(r"[,;|\n]+")

WEIGHT_MARKER_RE = re.compile(r"^\s*(?P<phrase>.+?)\s*(?:\^(?P<weight>[0-9]*\.?[0-9]+))?\s*$")

ALNUM_RE = re.compile(r"[^\W_]", re.UNICODE)

ANY_LANG_KEY = "*"

DEFAULT_SEMANTIC_WEIGHT = 1.10
DEFAULT_LABEL_WEIGHT = 1.30
DEFAULT_BIAS_WEIGHT = 1.00
DEFAULT_VALUE_WEIGHT = 0.70
DEFAULT_PREJUDICE_WEIGHT = 1.00
DEFAULT_CROSS_PREJUDICE_WEIGHT = 0.85
DEFAULT_LABEL_ECHO_WEIGHT = 0.90
DEFAULT_EXTERNAL_BIAS_WEIGHT = 0.85
DEFAULT_EXTERNAL_PREJUDICE_WEIGHT = 0.80

TEXT_CHROME_NEGATIVES: Tuple[str, ...] = (
    "a printed column header or field label with no value after it",
    "a placeholder such as N A null none undefined or a blank dash",
    "a page number running header or footer decoration",
    "a navigation menu breadcrumb or button caption",
    "a legal boilerplate clause or a signature line",
    "a bare unit symbol or currency sign without any number",
)


def flatten_text_values(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, dict):
        out: List[str] = []
        for _key, val in raw.items():
            out.extend(flatten_text_values(val))
        return out
    if isinstance(raw, (list, tuple, set)):
        out = []
        for elem in raw:
            out.extend(flatten_text_values(elem))
        return out
    return [str(raw)]


def pick_lang_values(raw, lang_code: str = "") -> List[str]:
    if not isinstance(raw, dict):
        return flatten_text_values(raw)

    code = str(lang_code or "").strip().lower()
    if code and code in raw:
        out = flatten_text_values(raw[code])
        out.extend(flatten_text_values(raw.get(ANY_LANG_KEY)))
        return out

    out: List[str] = []
    for _key, val in raw.items():
        out.extend(flatten_text_values(val))
    return out


def is_junk_phrase(phrase: str) -> bool:
    core = ALNUM_RE.findall(phrase)
    if not core:
        return True
    if len(core) == 1 and phrase.strip().isascii():
        return True
    return False


def split_bias_phrases_weighted(
    raw,
    default_weight: float = DEFAULT_BIAS_WEIGHT,
    lang_code: str = "",
) -> List[Tuple[str, float]]:
    items: List[str] = []
    for text in pick_lang_values(raw, lang_code):
        items.extend(PHRASE_SPLIT_RE.split(text))

    out: List[Tuple[str, float]] = []
    seen = set()
    for item in items:
        if not item:
            continue
        m = WEIGHT_MARKER_RE.match(item)
        if not m:
            continue
        phrase = (m.group("phrase") or "").strip()
        if not phrase:
            continue
        if is_junk_phrase(phrase):
            continue
        low = phrase.lower()
        if low in seen:
            continue
        seen.add(low)
        wraw = m.group("weight")
        weight = float(wraw) if wraw else float(default_weight)
        out.append((phrase, weight))
    return out


PHRASE_KIND_SEMANTIC = "semantic"
PHRASE_KIND_BIAS = "bias"
PHRASE_KIND_VALUE = "value"
PHRASE_KIND_LABEL = "label"
PHRASE_KIND_PREJUDICE = "prejudice"

BIAS_KINDS = (PHRASE_KIND_SEMANTIC, PHRASE_KIND_BIAS, PHRASE_KIND_VALUE)
CONCEPT_KINDS = (PHRASE_KIND_SEMANTIC, PHRASE_KIND_BIAS)


class Phrase:
    def __init__(self, text: str, weight: float, kind: str, owner: str):
        self.text = text
        self.weight = float(weight)
        self.kind = kind
        self.owner = owner
        self.vector: Optional[np.ndarray] = None

    @property
    def is_bias(self) -> bool:
        return self.kind in BIAS_KINDS

    @property
    def is_concept(self) -> bool:
        return self.kind in CONCEPT_KINDS

    def __repr__(self) -> str:
        return f"<Phrase {self.kind}:{self.owner} '{self.text}' w={self.weight:.2f}>"


class FieldEntry:
    def __init__(self, name: str, definition: dict, lang_code: str = ""):
        self.name = name
        self.definition = definition or {}
        self.lang_code = str(lang_code or "")
        self.semantic_texts: List[str] = pick_lang_values(
            self.definition.get("semantic"), self.lang_code
        )
        self.label: List[Phrase] = []
        self.bias: List[Phrase] = []
        self.prejudice: List[Phrase] = []
        self.external_prejudice: List[str] = []
        self.format: str = str(self.definition.get("format", "") or "")

    @property
    def semantic(self) -> str:
        return " / ".join(t for t in self.semantic_texts if t)

    @property
    def bias_texts(self) -> List[str]:
        return [p.text for p in self.bias]

    @property
    def label_texts(self) -> List[str]:
        return [p.text for p in self.label]

    @property
    def prejudice_texts(self) -> List[str]:
        return [p.text for p in self.prejudice]

    @staticmethod
    def _stack(phrases: Sequence[Phrase]) -> Tuple[np.ndarray, np.ndarray]:
        vecs = [p.vector for p in phrases if p.vector is not None]
        wts = [p.weight for p in phrases if p.vector is not None]
        if not vecs:
            return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        return np.stack(vecs).astype(np.float32), np.asarray(wts, dtype=np.float32)

    def bias_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._stack(self.bias)

    def prejudice_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._stack(self.prejudice)

    def label_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._stack(self.label)

    def bank_size(self) -> int:
        return len([p for p in self.bias if p.vector is not None])

    def label_bank_size(self) -> int:
        return len([p for p in self.label if p.vector is not None])

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "semantic": self.semantic,
            "format": self.format,
            "lang_code": self.lang_code,
            "label": [(p.text, p.weight) for p in self.label],
            "bias": [(p.text, p.weight, p.kind) for p in self.bias],
            "prejudice": [(p.text, p.weight) for p in self.prejudice],
            "bank_size": self.bank_size(),
            "label_bank_size": self.label_bank_size(),
        }


class FieldBank:
    def __init__(self, domain: str = "", doc_type: str = "", lang_code: str = ""):
        self.domain = domain
        self.doc_type = doc_type
        self.lang_code = str(lang_code or "")
        self.entries: Dict[str, FieldEntry] = {}
        self._embed_cache: Dict[str, np.ndarray] = {}
        self.dim: int = 0

    @property
    def field_names(self) -> List[str]:
        return list(self.entries.keys())

    def get(self, name: str) -> Optional[FieldEntry]:
        return self.entries.get(name)

    def add_field(self, name: str, definition: dict) -> FieldEntry:
        entry = FieldEntry(name, definition, lang_code=self.lang_code)
        self.entries[name] = entry
        return entry

    def build_phrases(
        self,
        cross_prejudice: bool = True,
        label_echo_prejudice: bool = True,
        chrome_prejudice: bool = True,
        external_dictionary: bool = True,
    ) -> None:
        lang = self.lang_code

        bridge = None
        if external_dictionary:
            try:
                from core import bias_bridge as _bridge
                if _bridge.available():
                    bridge = _bridge
            except Exception:
                bridge = None

        for name, entry in self.entries.items():
            labels: List[Phrase] = []
            seen_label = set()
            for txt, w in split_bias_phrases_weighted(
                entry.definition.get("label"), DEFAULT_LABEL_WEIGHT, lang
            ):
                low = txt.lower()
                if low in seen_label:
                    continue
                seen_label.add(low)
                labels.append(Phrase(txt, w, PHRASE_KIND_LABEL, name))
            entry.label = labels

            ext_bias: List[str] = []
            ext_prej: List[str] = []
            ext_value: List[str] = []
            if bridge is not None:
                ext_bias, ext_prej = bridge.field_phrases(self.domain, name, lang)
                ext_value = bridge.value_anchor_phrases(self.domain, name)
            entry.external_prejudice = ext_prej

            phrases: List[Phrase] = []
            seen_bias = set()
            sources = (
                (entry.semantic_texts, DEFAULT_SEMANTIC_WEIGHT, PHRASE_KIND_SEMANTIC),
                (entry.definition.get("bias"), DEFAULT_BIAS_WEIGHT, PHRASE_KIND_BIAS),
                (entry.definition.get("value"), DEFAULT_VALUE_WEIGHT, PHRASE_KIND_VALUE),
                (ext_bias, DEFAULT_EXTERNAL_BIAS_WEIGHT, PHRASE_KIND_BIAS),
                (ext_value, DEFAULT_VALUE_WEIGHT, PHRASE_KIND_VALUE),
            )
            for raw, weight, kind in sources:
                for txt, w in split_bias_phrases_weighted(raw, weight, lang):
                    low = txt.lower()
                    if low in seen_bias:
                        continue
                    seen_bias.add(low)
                    phrases.append(Phrase(txt, w, kind, name))
            entry.bias = phrases

        label_echo: List[str] = []
        if label_echo_prejudice:
            seen_echo = set()
            for entry in self.entries.values():
                for p in entry.label:
                    low = p.text.lower()
                    if low in seen_echo:
                        continue
                    seen_echo.add(low)
                    label_echo.append(p.text)

        for name, entry in self.entries.items():
            prej: List[Phrase] = []
            seen = {p.text.lower() for p in entry.bias}

            for txt, w in split_bias_phrases_weighted(
                entry.definition.get("prejudice"), DEFAULT_PREJUDICE_WEIGHT, lang
            ):
                low = txt.lower()
                if low in seen:
                    continue
                seen.add(low)
                prej.append(Phrase(txt, w, PHRASE_KIND_PREJUDICE, name))

            for txt in split_bias_phrases_weighted(
                entry.external_prejudice, DEFAULT_EXTERNAL_PREJUDICE_WEIGHT, lang
            ):
                t, w = txt
                low = t.lower()
                if low in seen:
                    continue
                seen.add(low)
                prej.append(Phrase(t, w, PHRASE_KIND_PREJUDICE, name))

            for txt in label_echo:
                low = txt.lower()
                if low in seen:
                    continue
                seen.add(low)
                prej.append(
                    Phrase(txt, DEFAULT_LABEL_ECHO_WEIGHT, PHRASE_KIND_PREJUDICE, name)
                )

            if cross_prejudice:
                for other_name, other in self.entries.items():
                    if other_name == name:
                        continue
                    for p in other.bias:
                        if not p.is_concept:
                            continue
                        low = p.text.lower()
                        if low in seen:
                            continue
                        seen.add(low)
                        prej.append(
                            Phrase(
                                p.text,
                                min(p.weight, DEFAULT_CROSS_PREJUDICE_WEIGHT),
                                PHRASE_KIND_PREJUDICE,
                                name,
                            )
                        )

            if chrome_prejudice:
                for txt in TEXT_CHROME_NEGATIVES:
                    low = txt.lower()
                    if low in seen:
                        continue
                    seen.add(low)
                    prej.append(
                        Phrase(txt, DEFAULT_PREJUDICE_WEIGHT, PHRASE_KIND_PREJUDICE, name)
                    )

            entry.prejudice = prej

    def all_phrase_texts(self) -> List[str]:
        seen = set()
        out: List[str] = []
        for entry in self.entries.values():
            for p in list(entry.label) + list(entry.bias) + list(entry.prejudice):
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
            for p in list(entry.label) + list(entry.bias) + list(entry.prejudice):
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
            "lang_code": self.lang_code,
            "fields": len(self.entries),
            "dim": self.dim,
            "phrases": len(self._embed_cache),
            "per_field": {
                name: {
                    "label": len(e.label),
                    "bias": len(e.bias),
                    "prejudice": len(e.prejudice),
                    "embedded": e.bank_size(),
                    "label_embedded": e.label_bank_size(),
                }
                for name, e in self.entries.items()
            },
        }

    def report_lines(self) -> List[str]:
        scope = self.lang_code if self.lang_code else "전체 언어"
        lines = [
            f"  도메인: {self.domain or '-'} / 서식: {self.doc_type or '-'} "
            f"| 필드 {len(self.entries)}개 | 임베딩 차원 {self.dim} "
            f"| 사전 스코프 {scope}"
        ]
        starved = []
        for name, e in self.entries.items():
            lines.append(
                f"    · {name:<28} label={len(e.label):3d} bias={len(e.bias):3d} "
                f"prej={len(e.prejudice):4d} embedded={e.bank_size():3d}"
            )
            if e.bank_size() == 0:
                starved.append(name)
        if starved:
            lines.append(
                f"    ⚠ 값 앵커가 비어 배정이 불가능한 필드 {len(starved)}개: "
                + ", ".join(starved)
            )
        return lines


def build_field_bank(
    schema: dict,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    cross_prejudice: bool = True,
    batch_size: int = 32,
    lang_code: str = "",
    label_echo_prejudice: bool = True,
    chrome_prejudice: bool = True,
    external_dictionary: bool = True,
) -> FieldBank:
    bank = FieldBank(
        domain=str(schema.get("domain", "") or ""),
        doc_type=str(schema.get("doc_type", "") or ""),
        lang_code=lang_code,
    )
    fields = schema.get("fields", {}) or {}
    for name, definition in fields.items():
        bank.add_field(str(name), definition if isinstance(definition, dict) else {})

    bank.build_phrases(
        cross_prejudice=cross_prejudice,
        label_echo_prejudice=label_echo_prejudice,
        chrome_prejudice=chrome_prejudice,
        external_dictionary=external_dictionary,
    )

    if embed_fn is not None:
        bank.embed_phrases(embed_fn, batch_size=batch_size)

    return bank