import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .text_upscale import crop_region

LINE_READ_MIN_LINES = 2
LINE_READ_DEAD_CROPS = 2

DEDUP_IOU = 0.35
DEDUP_CENTER_RATIO = 0.55
DEDUP_TEXT_SIM = 0.80
DEDUP_SAME_TEXT_SIM = 0.95
DEDUP_SAME_TEXT_UNITS = 0.75

LEDGER_QUANT_PX = 6
LEDGER_REUSE_RATIO = 0.60
LEDGER_MAX_ROWS = 600

READ_MODE_FORM = "form"
READ_MODE_PROSE = "prose"

FORM_MIN_ROWS = 6
FORM_LABEL_DENSITY = 0.22
PROSE_TRUST_CONF = 0.985

REFINE_DEAD_CROPS = 2

SCRIPT_MIN_SAMPLE = 12
SCRIPT_DOMINANT_RATIO = 0.70

_DOC_SCRIPT: Dict[str, int] = {}


def reset_doc_script() -> None:
    _DOC_SCRIPT.clear()


def observe_script(text: str) -> None:
    body = str(text or "").strip()
    if not body:
        return
    try:
        from core.lang_codes import block_census
        census = block_census(body)
    except Exception:
        return
    for name, n in (census or {}).items():
        _DOC_SCRIPT[str(name)] = _DOC_SCRIPT.get(str(name), 0) + int(n)


def dominant_doc_script() -> Tuple[str, int]:
    if not _DOC_SCRIPT:
        return "", 0
    total = sum(int(v) for v in _DOC_SCRIPT.values())
    if total < SCRIPT_MIN_SAMPLE:
        return "", total
    name, n = max(_DOC_SCRIPT.items(), key=lambda kv: kv[1])
    if float(n) / float(max(1, total)) < SCRIPT_DOMINANT_RATIO:
        return "", total
    return str(name), total


def script_conflicts(text: str) -> Tuple[bool, str, str]:
    doc, total = dominant_doc_script()
    if not doc:
        return False, "", ""
    try:
        from core.lang_codes import block_census
        census = block_census(str(text or ""))
    except Exception:
        return False, doc, ""
    if not census:
        return False, doc, ""
    top = max(census.items(), key=lambda kv: kv[1])[0]
    return (str(top) != doc), doc, str(top)

_LINE_READ_DEAD = {"strikes": 0, "off": False, "calls": 0, "changed": 0}
_REFINE_HEALTH = {"strikes": 0, "off": False, "calls": 0, "gained": 0}
_READ_MODE = {"mode": READ_MODE_FORM, "density": 0.0, "rows": 0, "decided": False}
_ROW_LEDGER: Dict[Tuple[int, int, int, int], str] = {}


def reset_line_read_health() -> None:
    _LINE_READ_DEAD["strikes"] = 0
    _LINE_READ_DEAD["off"] = False
    _LINE_READ_DEAD["calls"] = 0
    _LINE_READ_DEAD["changed"] = 0
    _ROW_LEDGER.clear()


def reset_refine_health() -> None:
    _REFINE_HEALTH["strikes"] = 0
    _REFINE_HEALTH["off"] = False
    _REFINE_HEALTH["calls"] = 0
    _REFINE_HEALTH["gained"] = 0
    _REFINE_HEALTH["down"] = False


def mark_refiner_down(category: str, log: Optional[List[str]] = None) -> None:
    if _REFINE_HEALTH.get("down"):
        _REFINE_HEALTH["off"] = True
        return

    _REFINE_HEALTH["down"] = True
    _REFINE_HEALTH["off"] = True
    if log is not None:
        log.append(
            f"    ⛔ [REFINER DOWN] '{category}' 에서 정제 LLM 이 적재 자체에 "
            f"실패했음을 확인했습니다. 실패 횟수를 2회까지 세지 않고 이 "
            f"크롭부터 바로 다국어 사전 코사인 경로로 넘깁니다 — 세는 동안 "
            f"앞쪽 크롭 두 개가 읽어 놓은 글자를 그대로 버리고 있었습니다."
        )


def reset_read_mode() -> None:
    _READ_MODE["mode"] = READ_MODE_FORM
    _READ_MODE["density"] = 0.0
    _READ_MODE["rows"] = 0
    _READ_MODE["decided"] = False


def is_prose_mode() -> bool:
    return str(_READ_MODE["mode"]) == READ_MODE_PROSE


def decide_read_mode(
    rows: Optional[Sequence[Tuple[str, float, Tuple[int, int, int, int]]]],
    schema: Optional[dict],
    log: Optional[List[str]] = None,
) -> str:
    if _READ_MODE["decided"]:
        return str(_READ_MODE["mode"])

    texts = [
        str(t or "").strip() for t, _s, _b in (rows or [])
        if str(t or "").strip()
    ]
    if len(texts) < FORM_MIN_ROWS:
        return str(_READ_MODE["mode"])

    index, _bank = build_label_index(schema, category="")
    if not index:
        return str(_READ_MODE["mode"])

    hit = 0
    for t in texts:
        ranked = rank_label_fields(t, index, top_k=1)
        if ranked and float(ranked[0][1]) >= LABEL_COSINE_FLOOR:
            hit += 1

    density = hit / float(len(texts))
    mode = READ_MODE_FORM if density >= FORM_LABEL_DENSITY else READ_MODE_PROSE

    _READ_MODE["mode"] = mode
    _READ_MODE["density"] = float(density)
    _READ_MODE["rows"] = len(texts)
    _READ_MODE["decided"] = True

    if log is not None:
        if mode == READ_MODE_FORM:
            log.append(
                f"    🧭 [READ MODE] 서식 모드 — OCR 행 {len(texts)}개 중 "
                f"{hit}개가 다국어 사전 코사인으로 스키마 필드 라벨에 "
                f"붙었습니다 ({density:.0%} ≥ {FORM_LABEL_DENSITY:.0%}). "
                f"라벨↔값 경로가 성립하므로 행 단위 VLM 재판독을 "
                f"최소화합니다."
            )
        else:
            log.append(
                f"    🧭 [READ MODE] 산문 모드 — OCR 행 {len(texts)}개 중 "
                f"{hit}개만 사전 코사인으로 라벨에 붙었습니다 "
                f"({density:.0%} < {FORM_LABEL_DENSITY:.0%}). 인쇄 라벨이 "
                f"없는 지면이라 행 자체가 값입니다. 행 단위 판독을 유지하고 "
                f"전용 인식기 신뢰선을 {PROSE_TRUST_CONF:.3f} 로 올립니다."
            )
    return mode


def _ledger_key(box: Sequence[int]) -> Tuple[int, int, int, int]:
    q = max(1, int(LEDGER_QUANT_PX))
    return (
        int(box[0]) // q, int(box[1]) // q,
        int(box[2]) // q, int(box[3]) // q,
    )


def _ledger_get(box: Sequence[int]) -> str:
    return str(_ROW_LEDGER.get(_ledger_key(box)) or "")


def _ledger_store(
    boxes: Sequence[Sequence[int]],
    texts: Sequence[str],
) -> int:
    n = 0
    for b, t in zip(boxes, texts):
        s = str(t or "").strip()
        if not s:
            continue
        _ROW_LEDGER[_ledger_key(b)] = s
        n += 1
    if len(_ROW_LEDGER) > LEDGER_MAX_ROWS:
        drop = len(_ROW_LEDGER) - LEDGER_MAX_ROWS
        for key in list(_ROW_LEDGER.keys())[:drop]:
            _ROW_LEDGER.pop(key, None)
    return n


def _note_refine(
    gained: int,
    category: str,
    log: Optional[List[str]] = None,
) -> None:
    _REFINE_HEALTH["calls"] += 1
    _REFINE_HEALTH["gained"] += int(gained)

    if int(gained) > 0:
        _REFINE_HEALTH["strikes"] = 0
        return

    if is_prose_mode():
        return

    _REFINE_HEALTH["strikes"] += 1
    if log is not None:
        log.append(
            f"    ⚠ [REFINE IDLE] '{category}' 정제 호출이 필드를 한 건도 "
            f"만들지 못했습니다 "
            f"({_REFINE_HEALTH['strikes']}/{REFINE_DEAD_CROPS}회 연속)."
        )
    if _REFINE_HEALTH["strikes"] >= REFINE_DEAD_CROPS:
        _REFINE_HEALTH["off"] = True
        if log is not None:
            log.append(
                f"    ⛔ [REFINE OFF] 누적 정제 호출 "
                f"{_REFINE_HEALTH['calls']}회 / 신규 필드 "
                f"{_REFINE_HEALTH['gained']}건. 남은 크롭에서는 정제를 "
                f"중단하고 다국어 사전 코사인 라벨↔값 경로만 씁니다 — "
                f"같은 결과를 VLM 없이 뽑습니다."
            )

from .text_upscale import (
    decide_tile_count,
    fit_for_vlm,
    plan_overlap_tiles,
    prepare_crop,
)
from .vision_nms import CropPlan


class ExtractedField:
    def __init__(
        self,
        category: str,
        bbox: Tuple[int, int, int, int],
        score: float,
        margin: float,
        ocr_text: str = "",
        refined: Optional[dict] = None,
        upscale: float = 1.0,
        upscale_mode: str = "",
        tiles: int = 1,
        crop_image: Optional[Image.Image] = None,
    ):
        self.category = category
        self.bbox = tuple(int(v) for v in bbox)
        self.score = float(score)
        self.margin = float(margin)
        self.ocr_text = ocr_text
        self.refined = refined or {}
        self.upscale = float(upscale)
        self.upscale_mode = upscale_mode
        self.tiles = int(tiles)
        self.crop_image = crop_image
        self.ocr_raw = ocr_text
        self.lemma = ""
        self.nlp_meta: dict = {}
        self.field_values: Dict[str, str] = {}
        self.donations: Dict[str, Tuple[str, float]] = {}
        self.rows: List[dict] = []
        self.line_texts: List[str] = []
        self.line_windows: int = 0

    @property
    def value(self) -> str:
        if self.refined:
            for key in ("value", "text", self.category):
                v = self.refined.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return self.ocr_text.strip()

    def to_dict(self, include_image: bool = False) -> dict:
        out = {
            "category": self.category,
            "field_name": self.category,
            "bbox": list(self.bbox),
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "ocr_text": self.ocr_text,
            "ocr_raw": self.ocr_raw,
            "lemma": self.lemma,
            "nlp": self.nlp_meta,
            "value": self.value,
            "refined": self.refined,
            "upscale": round(self.upscale, 3),
            "upscale_mode": self.upscale_mode,
            "tiles": self.tiles,
            "lines": list(self.line_texts),
            "windows": self.line_windows,
        }
        if include_image and self.crop_image is not None:
            import base64
            import io
            buf = io.BytesIO()
            try:
                self.crop_image.save(buf, format="PNG")
                out["crop_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                out["crop_base64"] = ""
        return out

    def __repr__(self) -> str:
        preview = self.value[:40].replace("\n", " ")
        return f"<Extracted {self.category} '{preview}' score={self.score:+.4f}>"


LABEL_MATCH_MIN = 0.86
LABEL_MIN_LEN = 5
LABEL_CONTAIN_MIN = 8
LABEL_LEN_RATIO = 0.70

LABEL_EXACT_MIN_LEN = 3
LABEL_NGRAM_MIN = 2
LABEL_NGRAM_MAX = 3
LABEL_COSINE_FLOOR = 0.42
LABEL_COSINE_MARGIN = 0.05
LABEL_CANDIDATE_TOPK = 3

LABEL_SHORT_LEN = 9
LABEL_SHORT_FLOOR = 0.80

FORMAT_MISMATCH_MIN_CHARS = 3


def alnum_count(text: str) -> int:
    return sum(1 for ch in str(text or "") if ch.isalnum())


def label_cosine_floor(label: str) -> float:
    n = len(_compact_label(label))
    if n >= LABEL_SHORT_LEN:
        return LABEL_COSINE_FLOOR
    if n <= LABEL_EXACT_MIN_LEN:
        return LABEL_SHORT_FLOOR
    span = float(LABEL_SHORT_LEN - LABEL_EXACT_MIN_LEN)
    ratio = (float(n) - LABEL_EXACT_MIN_LEN) / max(1.0, span)
    return LABEL_SHORT_FLOOR - (LABEL_SHORT_FLOOR - LABEL_COSINE_FLOOR) * ratio

LABEL_SOURCE_WEIGHT: Dict[str, float] = {
    "label": 1.30,
    "name": 1.15,
    "semantic": 1.10,
    "bias": 1.00,
    "value": 0.70,
}

LABEL_DICT_SOURCES = ("label", "semantic", "bias", "value")
LABEL_BANK_SOURCES = ("label", "semantic")

PHRASE_SPLIT_RE = re.compile(r"[,;|\n]+")

FORMAT_MISMATCH_PENALTY = 0.72
NUMERIC_CHAR_RE = re.compile(r"\d")
PURE_NUMERIC_RE = re.compile(r"^[-+]?\d[\d\s,.'\u00A0\u066B\u066C]*$")

IDENTITY_SELF_REF_MIN = 0.35
IDENTITY_SELF_REF_MARGIN = 0.04

_LABEL_INDEX_CACHE: Dict[Tuple[int, str, str], Tuple["LabelIndex", List[str]]] = {}


def _norm_text(text: str) -> str:
    return "".join(
        ch for ch in str(text or "").lower()
        if ch.isalnum() or ord(ch) > 0x2FFF
    )


def _compact_label(text: str) -> str:
    return "".join(ch for ch in str(text or "").lower() if ch.isalnum())


def _dict_phrases(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        out: List[str] = []
        for v in raw.values():
            out.extend(_dict_phrases(v))
        return out
    if isinstance(raw, (list, tuple, set)):
        out = []
        for v in raw:
            out.extend(_dict_phrases(v))
        return out

    out = []
    seen = set()
    for piece in PHRASE_SPLIT_RE.split(str(raw)):
        t = piece.strip().strip("()[]{}").strip()
        if not t:
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
    return out


def _label_grams(text: str) -> Dict[str, int]:
    body = _compact_label(text)
    out: Dict[str, int] = {}
    if not body:
        return out
    n = len(body)
    if n < LABEL_NGRAM_MIN:
        out[body] = 1
        return out
    for size in range(LABEL_NGRAM_MIN, LABEL_NGRAM_MAX + 1):
        if n < size:
            continue
        for i in range(n - size + 1):
            g = body[i: i + size]
            out[g] = out.get(g, 0) + 1
    return out


class LabelIndex:
    def __init__(self, category: str = ""):
        self.category = str(category or "")
        self.exact: Dict[str, str] = {}
        self.surfaces: List[Tuple[str, str, float]] = []
        self.fields: List[str] = []
        self._idf: Dict[str, float] = {}
        self._vecs: List[Tuple[str, float]] = []
        self._post: Dict[str, List[Tuple[int, float]]] = {}
        self._cache: Dict[str, Dict[str, float]] = {}
        self._ready = False

    def add(self, field: str, surface: str, source: str) -> None:
        text = str(surface or "").strip()
        if not text:
            return
        key = _compact_label(text)
        if len(key) < LABEL_EXACT_MIN_LEN:
            return
        self.exact.setdefault(key, str(field))
        self.surfaces.append(
            (str(field), text, float(LABEL_SOURCE_WEIGHT.get(source, 1.0)))
        )
        if field not in self.fields:
            self.fields.append(str(field))
        self._ready = False

    def finalize(self) -> "LabelIndex":
        if self._ready:
            return self
        self._ready = True
        self._vecs = []
        self._post = {}
        self._idf = {}
        self._cache = {}

        total = len(self.surfaces)
        if total == 0:
            return self

        grams_list: List[Dict[str, int]] = []
        df: Dict[str, int] = {}
        for _field, text, _weight in self.surfaces:
            grams = _label_grams(text)
            grams_list.append(grams)
            for g in grams.keys():
                df[g] = df.get(g, 0) + 1

        for g, n in df.items():
            self._idf[g] = math.log((1.0 + total) / (1.0 + n)) + 1.0

        for (field, _text, weight), grams in zip(self.surfaces, grams_list):
            vec: Dict[str, float] = {}
            for g, tf in grams.items():
                vec[g] = float(tf) * self._idf.get(g, 1.0)
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm <= 1e-9:
                continue
            vi = len(self._vecs)
            self._vecs.append((field, float(weight)))
            for g, v in vec.items():
                self._post.setdefault(g, []).append((vi, v / norm))
        return self

    def _query_vec(self, label: str) -> Dict[str, float]:
        grams = _label_grams(label)
        if not grams:
            return {}
        vec = {g: float(tf) * self._idf.get(g, 1.0) for g, tf in grams.items()}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        if norm <= 1e-9:
            return {}
        return {g: v / norm for g, v in vec.items()}

    def scores(self, label: str) -> Dict[str, float]:
        self.finalize()
        key = _compact_label(label)
        if len(key) < LABEL_EXACT_MIN_LEN:
            return {}

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        hit = self.exact.get(key)
        if hit:
            out = {hit: 1.0}
            self._cache[key] = out
            return out

        q = self._query_vec(label)
        if not q:
            self._cache[key] = {}
            return {}

        acc: Dict[int, float] = {}
        for g, qv in q.items():
            posting = self._post.get(g)
            if not posting:
                continue
            for vi, v in posting:
                acc[vi] = acc.get(vi, 0.0) + qv * v

        out: Dict[str, float] = {}
        for vi, raw in acc.items():
            field, weight = self._vecs[vi]
            val = min(0.999, float(raw) * weight)
            if val > out.get(field, 0.0):
                out[field] = val

        self._cache[key] = out
        return out

    def resolve(self, label: str) -> Tuple[str, float, str, float]:
        ranked = sorted(
            self.scores(label).items(), key=lambda kv: kv[1], reverse=True
        )
        if not ranked:
            return "", 0.0, "", 0.0
        top, top_s = ranked[0]
        if len(ranked) > 1:
            return str(top), float(top_s), str(ranked[1][0]), float(ranked[1][1])
        return str(top), float(top_s), "", 0.0

    def get(self, key: str) -> Optional[str]:
        return self.exact.get(key)

    def __bool__(self) -> bool:
        return bool(self.surfaces)

    def __len__(self) -> int:
        return len(self.surfaces)


def build_label_index(
    schema: Optional[dict],
    category: str = "",
    drop_fields: Optional[Sequence[str]] = None,
) -> Tuple[LabelIndex, List[str]]:
    fields = (schema or {}).get("fields", {}) or {}
    drop = set(str(d) for d in (drop_fields or ()))
    cache_key = (
        id(schema), str(category or ""), "|".join(sorted(drop))
    )
    cached = _LABEL_INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached

    index = LabelIndex(category)
    bank: List[str] = []
    seen_bank = set()

    def _bank(text: str) -> None:
        key = _compact_label(text)
        if len(key) < LABEL_MIN_LEN or key in seen_bank:
            return
        seen_bank.add(key)
        bank.append(text)

    for name, definition in fields.items():
        d = definition if isinstance(definition, dict) else {}
        if category and str(d.get("category") or "misc") != category:
            continue

        plain = str(name).replace("_", " ")
        if str(name) not in drop:
            index.add(str(name), plain, "name")
        _bank(plain)

        for key in LABEL_DICT_SOURCES:
            raw = d.get(key)
            if raw is None:
                continue
            for piece in _dict_phrases(raw):
                if str(name) not in drop:
                    index.add(str(name), piece, key)
                if key in LABEL_BANK_SOURCES:
                    _bank(piece)

    index.finalize()
    _LABEL_INDEX_CACHE[cache_key] = (index, bank)
    return index, bank


def _field_category_map(schema: Optional[dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, definition in ((schema or {}).get("fields", {}) or {}).items():
        d = definition if isinstance(definition, dict) else {}
        out[str(name)] = str(d.get("category") or "misc")
    return out


def _array_field_names(schema: Optional[dict]) -> set:
    out = set()
    for name, definition in ((schema or {}).get("fields", {}) or {}).items():
        d = definition if isinstance(definition, dict) else {}
        if d.get("array") or d.get("table"):
            out.add(str(name))
    return out


def _field_format_map(schema: Optional[dict]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for name, definition in ((schema or {}).get("fields", {}) or {}).items():
        d = definition if isinstance(definition, dict) else {}
        fmt = str(d.get("format") or "").strip().lower()
        if fmt:
            out[str(name)] = fmt
    return out


def _format_penalty(fmt: str, value: str) -> float:
    v = str(value or "").strip()
    if not v:
        return 1.0
    f = str(fmt or "").strip().lower()
    if f == "numeric":
        return 1.0 if NUMERIC_CHAR_RE.search(v) else FORMAT_MISMATCH_PENALTY
    if PURE_NUMERIC_RE.match(v):
        return FORMAT_MISMATCH_PENALTY
    return 1.0


def _gram_cosine(a: str, b: str) -> float:
    ga = _label_grams(a)
    gb = _label_grams(b)
    if not ga or not gb:
        return 0.0
    na = math.sqrt(sum(float(v) * float(v) for v in ga.values()))
    nb = math.sqrt(sum(float(v) * float(v) for v in gb.values()))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    dot = sum(float(v) * float(gb.get(g, 0)) for g, v in ga.items())
    return float(dot / (na * nb))


def identity_drop_fields(
    schema: Optional[dict],
    doc_code: str = "",
) -> Dict[str, str]:
    fields = (schema or {}).get("fields", {}) or {}
    drop: Dict[str, str] = {}
    if not fields:
        return drop

    if "doc_type" in fields:
        drop["doc_type"] = (
            "STEP 1 비전 NMS 가 이미 확정한 축이라 라벨↔값 배정에서 제외합니다."
        )

    code = str(doc_code or (schema or {}).get("code") or "").strip().lower()
    for name in fields.keys():
        if not str(name).startswith("reference_") or name in drop:
            continue
        tail = str(name)[len("reference_"):].replace("_", "")
        if tail and code and (
            tail == code or code.endswith(tail) or tail.endswith(code)
        ):
            drop[str(name)] = (
                f"'{doc_code or code}' 문서가 자기 자신을 가리키는 축입니다."
            )

    own = fields.get("doc_number")
    own_text = ""
    if isinstance(own, dict):
        own_text = str(own.get("semantic") or "")
    if not own_text:
        return drop

    sims: List[Tuple[str, float]] = []
    for name, definition in fields.items():
        if not str(name).startswith("reference_") or name in drop:
            continue
        d = definition if isinstance(definition, dict) else {}
        sem = str(d.get("semantic") or "")
        if not sem:
            continue
        sims.append((str(name), _gram_cosine(own_text, sem)))

    if len(sims) < 2:
        return drop

    sims.sort(key=lambda kv: kv[1], reverse=True)
    top, top_s = sims[0]
    second_s = sims[1][1]
    if top_s >= IDENTITY_SELF_REF_MIN and (top_s - second_s) >= IDENTITY_SELF_REF_MARGIN:
        drop[top] = (
            f"이 서식의 doc_number 설명과 사전 유사도 {top_s:.2f} "
            f"(차점 {second_s:.2f}) — 자기 자신을 가리키는 참조 축입니다."
        )
    return drop


VALUE_MIN_LEN = 2
VALUE_MAX_LEN = 120
VALUE_NOISE_RATIO = 0.45


def _value_plausible(value: str) -> bool:
    s = str(value or "").strip()
    if not s or len(s) > VALUE_MAX_LEN:
        return False
    if len(s) < VALUE_MIN_LEN and not s.isdigit():
        return False

    body = [ch for ch in s if not ch.isspace()]
    if not body:
        return False

    alnum = sum(1 for ch in body if ch.isalnum())
    if alnum / float(len(body)) < (1.0 - VALUE_NOISE_RATIO):
        return False

    if len(body) <= 4:
        letters = [ch for ch in body if ch.isalpha()]
        if letters:
            lower = sum(1 for ch in letters if ch.islower())
            upper = len(letters) - lower
            if lower and upper and any(ch.isdigit() for ch in body):
                return False

    return True


def rank_label_fields(
    label: str,
    index: Optional[LabelIndex],
    top_k: int = LABEL_CANDIDATE_TOPK,
) -> List[Tuple[str, float]]:
    if index is None or not index:
        return []
    if len(_compact_label(label)) < LABEL_EXACT_MIN_LEN:
        return []

    floor = label_cosine_floor(label)
    ranked = sorted(
        index.scores(label).items(), key=lambda kv: kv[1], reverse=True
    )
    passed = [
        (str(f), float(s)) for f, s in ranked
        if float(s) >= floor
    ]
    return passed[: max(1, int(top_k))]


def score_label_to_field(
    label: str,
    index: Optional[LabelIndex],
    log: Optional[List[str]] = None,
) -> Tuple[str, float, str, float]:
    if index is None or not index:
        return "", 0.0, "", 0.0
    if len(_compact_label(label)) < LABEL_EXACT_MIN_LEN:
        return "", 0.0, "", 0.0

    field, score, rival, rival_score = index.resolve(label)
    if not field:
        return "", 0.0, "", 0.0

    if score >= 1.0:
        return field, score, rival, rival_score

    floor = label_cosine_floor(label)
    if score < floor:
        if log is not None and score >= LABEL_COSINE_FLOOR:
            log.append(
                f"       🚧 라벨 '{label[:16]}' 은 압축 {len(_compact_label(label))}자로 "
                f"짧아 n-gram 이 몇 개뿐입니다. 코사인 {score:.2f} 는 긴 라벨 "
                f"기준 {LABEL_COSINE_FLOOR:.2f} 는 넘지만 이 길이의 요구 "
                f"바닥 {floor:.2f} 에 못 미쳐 배정하지 않습니다 — 크롭이 "
                f"잘려 라벨 앞머리만 남은 경우입니다."
            )
        return "", score, rival, rival_score

    if rival and (score - rival_score) < LABEL_COSINE_MARGIN:
        if log is not None:
            log.append(
                f"       ⚖ 라벨 '{label[:24]}' — '{field}'({score:.3f}) vs "
                f"'{rival}'({rival_score:.3f}) 사전 코사인 격차 "
                f"{score - rival_score:.3f} < {LABEL_COSINE_MARGIN:.2f} "
                f"→ 배정 보류"
            )
        return "", score, rival, rival_score

    return field, score, rival, rival_score


def match_label_to_field(
    label: str,
    index: Optional[LabelIndex],
    log: Optional[List[str]] = None,
) -> str:
    field, _score, _rival, _rival_score = score_label_to_field(
        label, index, log=log
    )
    return field


PAIR_BELOW_MAX_GAP = 2.2
PAIR_RIGHT_MAX_GAP = 1.2
PAIR_VERT_TOLERANCE = 0.6
PAIR_COLUMN_PENALTY = 1.6
PAIR_RIGHT_VERT_WEIGHT = 0.5


def pair_rows_by_geometry(
    rows: Sequence[Tuple[str, float, Tuple[int, int, int, int]]],
    label_bank: Sequence[str],
    log: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    try:
        from core.text_prep import is_printed_label
    except Exception:
        return []

    items = [
        (t, s, b) for t, s, b in rows
        if str(t or "").strip()
    ]
    if len(items) < 2:
        return []

    heights = sorted(float(b[3] - b[1]) for _t, _s, b in items)
    unit = heights[len(heights) // 2] if heights else 20.0
    unit = max(8.0, unit)

    labels: List[int] = []
    values: List[int] = []
    for i, (t, _s, _b) in enumerate(items):
        if is_printed_label(t, label_bank):
            labels.append(i)
        else:
            values.append(i)

    used: set = set()
    out: List[Tuple[str, str]] = []
    below = 0
    right = 0

    for li in labels:
        ltext, _ls, lb = items[li]
        lx0, ly0, lx1, ly1 = lb
        lcx = (lx0 + lx1) * 0.5

        best = -1
        best_cost = 1e9
        best_mode = ""

        for vi in values:
            if vi in used:
                continue
            vtext, _vs, vb = items[vi]
            vx0, vy0, vx1, vy1 = vb
            vcx = (vx0 + vx1) * 0.5

            gap_y = (vy0 - ly1) / unit
            if -0.2 <= gap_y <= PAIR_BELOW_MAX_GAP:
                overlap = min(lx1, vx1) - max(lx0, vx0)
                span = max(1.0, min(float(lx1 - lx0), float(vx1 - vx0)))
                if overlap > 0:
                    share = min(1.0, float(overlap) / span)
                    cost = (
                        abs(gap_y)
                        + abs(vcx - lcx) / max(1.0, float(lx1 - lx0))
                        + (1.0 - share) * PAIR_COLUMN_PENALTY
                    )
                    if cost < best_cost:
                        best = vi
                        best_cost = cost
                        best_mode = "below"

            gap_x = (vx0 - lx1) / unit
            vert = abs(((vy0 + vy1) * 0.5) - ((ly0 + ly1) * 0.5)) / unit
            if 0.0 <= gap_x <= PAIR_RIGHT_MAX_GAP and vert <= PAIR_VERT_TOLERANCE:
                cost = gap_x + vert * PAIR_RIGHT_VERT_WEIGHT
                if cost < best_cost:
                    best = vi
                    best_cost = cost
                    best_mode = "right"

        if best < 0:
            continue
        used.add(best)
        out.append((ltext, items[best][0]))
        if best_mode == "below":
            below += 1
        else:
            right += 1

    if log is not None and out:
        log.append(
            f"    📐 [GEO PAIR] 좌표로 라벨↔값 {len(out)}쌍 구성 "
            f"(아래 배치 {below} / 오른쪽 배치 {right} | 행 높이 단위 "
            f"{unit:.0f}px)"
        )
    return out


def promote_by_labels(
    text: str,
    schema: Optional[dict],
    category: str,
    log: Optional[List[str]] = None,
    rows: Optional[Sequence[Tuple[str, float, Tuple[int, int, int, int]]]] = None,
    doc_code: str = "",
) -> Tuple[Dict[str, Tuple[str, float]], Dict[str, Tuple[str, float]]]:
    body = str(text or "").strip()
    if not schema:
        return {}, {}

    drop = identity_drop_fields(schema, doc_code)
    all_index, all_bank = build_label_index(
        schema, category="", drop_fields=list(drop.keys())
    )
    if not all_index:
        return {}, {}

    try:
        from core.text_prep import collect_label_value_pairs, is_printed_label
    except Exception as e:
        if log is not None:
            log.append(f"    ⏭ 라벨↔값 모듈 로드 실패 ({e})")
        return {}, {}

    geo = pair_rows_by_geometry(rows or [], all_bank, log=log)

    pairs: List[Tuple[str, str]] = list(geo)
    if not pairs and body:
        try:
            flat = collect_label_value_pairs(body, label_bank=all_bank)
        except Exception as e:
            if log is not None:
                log.append(f"    ⏭ 라벨↔값 추출 실패 ({e})")
            return {}, {}
        pairs = [(p.label, p.value) for p in flat]
        if log is not None and pairs:
            log.append(
                f"    📄 [FLAT PAIR] 좌표가 없어 줄 순서로 {len(pairs)}쌍 "
                f"구성했습니다. 2열 서식에서는 오매칭 위험이 있습니다."
            )

    if not pairs:
        return {}, {}

    owner = _field_category_map(schema)
    fmt_map = _field_format_map(schema)

    cands: List[Tuple[float, int, str, str]] = []
    trace: Dict[Tuple[int, str], Tuple[str, float, float]] = {}
    picked: Dict[str, Tuple[int, str]] = {}
    short_blocked: List[str] = []
    noisy = 0
    label_echo = 0
    unmatched = 0
    fmt_blocked = 0

    for idx, (label, raw_value) in enumerate(pairs):
        value = str(raw_value or "").strip()
        if not _value_plausible(value):
            noisy += 1
            continue
        if is_printed_label(value, all_bank):
            label_echo += 1
            continue
        ranked = rank_label_fields(label, all_index)
        if not ranked:
            unmatched += 1
            continue
        for field, score in ranked:
            fmt = str(fmt_map.get(field, "") or "")
            pen = float(_format_penalty(fmt, value))
            adj = float(score) * pen
            if adj < LABEL_COSINE_FLOOR:
                fmt_blocked += 1
                continue
            vlen = alnum_count(value)
            if pen < 1.0 and vlen < FORMAT_MISMATCH_MIN_CHARS:
                fmt_blocked += 1
                if len(short_blocked) < 8:
                    short_blocked.append(
                        f"{field} = {value[:16]} "
                        f"(형식 {fmt or '미추론'} / {vlen}자 / "
                        f"원점수 {float(score):.2f} × 감점 {pen:.2f})"
                    )
                continue
            trace[(idx, str(field))] = (fmt, float(score), pen)
            cands.append((float(adj), idx, str(field), value))

    own: Dict[str, Tuple[str, float]] = {}
    donated: Dict[str, Tuple[str, float]] = {}

    if cands:
        cands.sort(key=lambda t: (-t[0], t[1]))
        used_pairs: set = set()
        taken: set = set()
        for score, idx, field, value in cands:
            if idx in used_pairs or field in taken:
                continue
            used_pairs.add(idx)
            taken.add(field)
            picked[str(field)] = (int(idx), str(pairs[idx][0]))
            if str(owner.get(field, "")) == str(category):
                own[field] = (value, float(score))
            else:
                donated[field] = (value, float(score))

    if log is not None and drop:
        for dname, why in list(drop.items())[:4]:
            log.append(f"       🧹 [AXIS DROP] '{dname}' — {why}")

    def _pair_note(name: str) -> str:
        hit = picked.get(str(name))
        if hit is None:
            return ""
        fmt, base, pen = trace.get((hit[0], str(name)), ("", 0.0, 1.0))
        return (
            f" | 라벨 '{str(hit[1])[:24]}' | 형식 {fmt or '미추론'} "
            f"| 원점수 {base:.2f} × 감점 {pen:.2f}"
        )

    if log is not None and (own or donated or unmatched or noisy or label_echo):
        log.append(
            f"    🏷 [LABEL PAIR] '{category}' 라벨↔값 {len(pairs)}쌍 → "
            f"사전 코사인 배타 배정 {len(own) + len(donated)}건 "
            f"(이 카테고리 {len(own)}건 / 타 카테고리 기부 {len(donated)}건) "
            f"| 미매칭 {unmatched}건 | 형식 불일치 {fmt_blocked}건 차단 "
            f"| 라벨 에코 {label_echo}건 | 잡음값 {noisy}건 폐기"
        )
        for k, payload in list(own.items())[:8]:
            log.append(
                f"       · {k} = {str(payload[0])[:40]} "
                f"(사전 코사인 {float(payload[1]):.2f}){_pair_note(k)}"
            )
        for k, payload in list(donated.items())[:8]:
            log.append(
                f"       ↗ {k} = {str(payload[0])[:40]} "
                f"(사전 코사인 {float(payload[1]):.2f}){_pair_note(k)}"
            )

    if log is not None and short_blocked:
        log.append(
            f"       🚧 [SHORT MISMATCH] 형식을 어긴 데다 영숫자 "
            f"{FORMAT_MISMATCH_MIN_CHARS}자 미만인 값 {len(short_blocked)}건을 "
            f"곱셈 감점 대신 통째로 막았습니다 — 감점은 0.72 라 원점수가 "
            f"1.00 이면 바닥값 {LABEL_COSINE_FLOOR:.2f} 을 넘어 통과합니다. "
            f"긴 값은 스키마가 형식을 안 적어 억울하게 감점당한 식별자일 수 "
            f"있어 살리고, 한두 글자는 형식도 길이도 근거가 없어 버립니다."
        )
        for line in short_blocked:
            log.append(f"          · {line}")

    return own, donated


PROSE_CLAIM_SCORE = 0.45
PROSE_MAX_LINES = 12
PROSE_FIELD_MARGIN = 0.01
PROSE_MIN_CHARS = 3
PROSE_MIN_CHARS_DENSE = 1
PROSE_CAPACITY_SLACK = 1.15
DENSE_SCRIPTS = ("hangul", "han", "hiragana", "katakana", "kana", "cjk")


def prose_min_chars(text: str) -> int:
    try:
        from core.lang_codes import block_census
        census = block_census(str(text or ""))
    except Exception:
        return PROSE_MIN_CHARS
    if not census:
        return PROSE_MIN_CHARS
    top = str(max(census.items(), key=lambda kv: kv[1])[0]).lower()
    for name in DENSE_SCRIPTS:
        if name in top:
            return PROSE_MIN_CHARS_DENSE
    return PROSE_MIN_CHARS


def field_value_lengths(
    schema: Optional[dict],
    category: str,
) -> Dict[str, Tuple[float, float]]:
    fields = (schema or {}).get("fields", {}) or {}
    out: Dict[str, Tuple[float, float]] = {}
    for name, definition in fields.items():
        d = definition if isinstance(definition, dict) else {}
        if category and str(d.get("category") or "misc") != category:
            continue
        lens: List[float] = []
        for key in ("value", "examples", "bias"):
            for piece in _dict_phrases(d.get(key)):
                n = len("".join(str(piece).split()))
                if n > 0:
                    lens.append(float(n))
        if not lens:
            continue
        lens.sort()
        mid = lens[len(lens) // 2]
        top = lens[min(len(lens) - 1, int(len(lens) * 0.9))]
        out[str(name)] = (float(mid), float(top))
    return out


def pick_prose_field(
    text: str,
    schema: Optional[dict],
    category: str,
    fallback: str = "",
    log: Optional[List[str]] = None,
) -> str:
    index, _bank = build_label_index(schema, category=category)
    if not index:
        return fallback

    ranked = sorted(
        index.scores(text).items(), key=lambda kv: kv[1], reverse=True
    )
    top = str(ranked[0][0]) if ranked else ""
    top_s = float(ranked[0][1]) if ranked else 0.0
    second = str(ranked[1][0]) if len(ranked) > 1 else ""
    second_s = float(ranked[1][1]) if len(ranked) > 1 else 0.0

    flat = (not ranked) or top_s <= 0.0 or (top_s - second_s) < PROSE_FIELD_MARGIN

    if flat:
        profile = field_value_lengths(schema, category)
        want = float(len("".join(str(text or "").split())))
        if profile and want > 0.0:
            fits = [
                (name, mid, cap) for name, (mid, cap) in profile.items()
                if want <= cap * PROSE_CAPACITY_SLACK
            ]
            if fits:
                fits.sort(key=lambda t: (t[2], abs(t[1] - want)))
                pick = str(fits[0][0])
                typical = float(fits[0][1])
                cap = float(fits[0][2])
                mode = f"용량 {cap:.0f}자"
            else:
                scored = sorted(
                    profile.items(),
                    key=lambda kv: abs(
                        math.log((want + 1.0) / (float(kv[1][1]) + 1.0))
                    ),
                )
                pick = str(scored[0][0])
                typical = float(scored[0][1][0])
                cap = float(scored[0][1][1])
                mode = f"용량 {cap:.0f}자 (어느 필드에도 들어가지 않아 최근접)"
            if log is not None:
                log.append(
                    f"       📏 [PROSE LENGTH] '{category}' 은 사전 n-gram 이 "
                    f"'{top or '-'}'({top_s:.3f}) 와 '{second or '-'}'"
                    f"({second_s:.3f}) 로 동률이라 값 길이로 가릅니다 — 판독문 "
                    f"{int(want)}자가 들어가는 가장 좁은 그릇 '{pick}' "
                    f"({mode} / 전형 {typical:.0f}자) 을 고릅니다. 중앙값으로 "
                    f"비교하면 짧은 값만 담는 필드가 긴 문장을 가져갑니다."
                )
            return pick

        if log is not None:
            log.append(
                f"       ⚖ [PROSE FIELD] '{category}' 판독문을 사전에 걸어도 "
                f"'{top or '-'}'({top_s:.3f}) 와 '{second or '-'}'"
                f"({second_s:.3f}) 가 실질 동률이고 값 길이 표본도 없습니다. "
                f"히트맵이 고른 '{fallback or top}' 을 그대로 씁니다."
            )
        return fallback or top

    if log is not None:
        log.append(
            f"       🎯 [PROSE FIELD] '{category}' 판독문을 다국어 사전에 걸어 "
            f"'{top}'({top_s:.3f}) 로 결정했습니다 — 차점 '{second or '-'}'"
            f"({second_s:.3f}). 히트맵 후보는 '{fallback or '-'}' 였습니다."
        )
    return top


PROSE_BLOCK_GAP_UNITS = 0.9


def group_prose_blocks(
    lines: Sequence[str],
    rows: Optional[Sequence[Tuple[str, float, Tuple[int, int, int, int]]]] = None,
    log: Optional[List[str]] = None,
) -> List[List[str]]:
    items = [str(t or "").strip() for t in (lines or []) if str(t or "").strip()]
    if not items:
        return []
    if len(items) == 1:
        return [items]

    boxed: List[Tuple[int, int, int, int]] = []
    for ln in items:
        hit = None
        for t, _s, b in (rows or []):
            if str(t or "").strip() == ln:
                hit = b
                break
        if hit is None:
            return [items]
        boxed.append(tuple(int(v) for v in hit))

    heights = sorted(float(b[3] - b[1]) for b in boxed)
    unit = max(8.0, heights[len(heights) // 2])

    order = sorted(range(len(items)), key=lambda i: (boxed[i][1], boxed[i][0]))
    blocks: List[List[str]] = [[items[order[0]]]]
    cuts = 0

    for k in range(1, len(order)):
        prev = boxed[order[k - 1]]
        cur = boxed[order[k]]
        gap = (float(cur[1]) - float(prev[3])) / unit
        overlap = min(prev[2], cur[2]) - max(prev[0], cur[0])
        if gap > PROSE_BLOCK_GAP_UNITS or overlap <= 0:
            blocks.append([items[order[k]]])
            cuts += 1
            continue
        blocks[-1].append(items[order[k]])

    if log is not None and cuts:
        log.append(
            f"       🧱 [PROSE BLOCK] 줄 {len(items)}개를 세로 간격 "
            f"{PROSE_BLOCK_GAP_UNITS:.1f}줄 기준으로 {len(blocks)}덩이로 "
            f"나눴습니다 (행 높이 단위 {unit:.0f}px)."
        )
    return blocks


def promote_prose_lines(
    text: str,
    schema: Optional[dict],
    category: str,
    top_field: str = "",
    is_array: bool = False,
    rows: Optional[Sequence[Tuple[str, float, Tuple[int, int, int, int]]]] = None,
    log: Optional[List[str]] = None,
) -> Tuple[Dict[str, str], List[dict]]:
    body = str(text or "").strip()
    if not body or not schema:
        return {}, []

    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if not lines:
        return {}, []

    try:
        from core.text_prep import is_printed_label
    except Exception:
        return {}, []

    _all_index, bank = build_label_index(schema, category="")
    labelish = sum(1 for ln in lines if is_printed_label(ln, bank))
    if labelish > 0:
        if log is not None:
            log.append(
                f"    ⏭ [PROSE SKIP] '{category}' 크롭의 {len(lines)}줄 중 "
                f"{labelish}줄이 스키마의 인쇄 라벨과 일치합니다. 라벨이 섞인 "
                f"지면에서 줄 전체를 값으로 승격하면 라벨까지 값이 되므로 "
                f"건너뜁니다."
            )
        return {}, []

    kept: List[str] = []
    dropped_script = 0
    dropped_short = 0

    for ln in lines:
        if not _value_plausible(ln):
            continue
        conflict, doc_s, line_s = script_conflicts(ln)
        if conflict:
            dropped_script += 1
            if log is not None:
                log.append(
                    f"       🔤 [PROSE SCRIPT] '{ln[:20]}' 는 문서 지배 문자체계 "
                    f"'{doc_s}' 가 아니라 '{line_s}' 입니다. 인쇄 라벨이 없는 "
                    f"지면에서 줄 전체를 값으로 올리는 경로이므로, 다른 "
                    f"문자체계가 섞인 줄은 판독 환각으로 보고 버립니다."
                )
            continue
        if len("".join(ln.split())) < prose_min_chars(ln):
            dropped_short += 1
            continue
        kept.append(ln)

    kept = kept[:PROSE_MAX_LINES]

    if not kept:
        if log is not None and (dropped_script or dropped_short):
            log.append(
                f"    ⏭ [PROSE DROP] '{category}' 에서 승격 가능한 줄이 "
                f"없습니다 — 문자체계 불일치 {dropped_script}건 / 너무 짧음 "
                f"{dropped_short}건. 값을 만들지 않고 OCR 원문만 남깁니다."
            )
        return {}, []

    fields = (schema or {}).get("fields", {}) or {}
    owner = _field_category_map(schema)

    fallback = str(top_field or "")
    if fallback not in fields or str(owner.get(fallback, "")) != str(category):
        fallback = ""
        for name, cat in owner.items():
            if str(cat) == str(category):
                fallback = str(name)
                break
    if not fallback:
        return {}, []

    joined = " ".join(kept)
    target = pick_prose_field(joined, schema, category, fallback, log=log)
    if target not in fields or str(owner.get(target, "")) != str(category):
        target = fallback

    fmt_map = _field_format_map(schema)
    if _format_penalty(fmt_map.get(target, ""), joined) < 1.0:
        profile = field_value_lengths(schema, category)
        want = float(len("".join(str(joined or "").split())))

        pool: List[Tuple[float, float, str]] = []
        for name, cat in owner.items():
            if str(cat) != str(category) or name == target:
                continue
            if name not in fields:
                continue
            if _format_penalty(fmt_map.get(str(name), ""), joined) < 1.0:
                continue
            mid, cap = profile.get(str(name), (want, want))
            if want > float(cap) * PROSE_CAPACITY_SLACK:
                continue
            pool.append((float(cap), abs(float(mid) - want), str(name)))

        pool.sort()
        alt = pool[0][2] if pool else ""

        if log is not None:
            log.append(
                f"       🚧 [PROSE FORMAT] '{target}' 는 형식이 "
                f"'{fmt_map.get(target, '-')}' 인데 값 '{joined[:20]}' 가 맞지 "
                f"않습니다 — "
                + (
                    f"같은 카테고리에서 형식이 맞고 {int(want)}자가 들어가는 "
                    f"필드 {len(pool)}개 중 가장 좁은 '{alt}' 로 옮깁니다."
                    if alt else
                    "형식이 맞으면서 이 길이를 담을 필드가 없어 이 크롭은 "
                    "값을 만들지 않습니다 — 선언 순서로 아무 필드나 고르면 "
                    "통화 코드 칸에 상품명이 들어갑니다."
                )
            )
        if not alt:
            return {}, []
        target = alt

    if is_array:
        blocks = group_prose_blocks(kept, rows, log=log)
        out_rows = [{target: " ".join(b)} for b in blocks if b]
        if log is not None:
            log.append(
                f"    📜 [PROSE VALUE] '{category}' 는 인쇄 라벨이 한 줄도 없는 "
                f"지면입니다. 판독된 {len(kept)}줄을 줄 간격으로 묶어 "
                f"{len(out_rows)}개 덩이로 만들고 '{target}' 에 행 단위로 "
                f"넣습니다 — 말풍선 한 개 안의 여러 줄은 한 발화이므로 "
                f"줄마다 행을 만들면 같은 대사가 쪼개집니다."
            )
            for r in out_rows[:4]:
                log.append(f"       · {target} = {str(r[target])[:40]}")
        return {}, out_rows

    if log is not None:
        log.append(
            f"    📜 [PROSE VALUE] '{category}' 는 인쇄 라벨이 한 줄도 없는 "
            f"지면입니다. 판독된 {len(kept)}줄을 '{target}' 의 값으로 "
            f"승격합니다 — {joined[:40]}"
        )
    return {target: joined}, []


def _text_similarity(a: str, b: str) -> float:
    ca = _norm_text(a)
    cb = _norm_text(b)
    if not ca or not cb:
        return 0.0
    if ca == cb:
        return 1.0

    m, n = len(ca), len(cb)
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        cur = [0] * (n + 1)
        x = ca[i - 1]
        for j in range(1, n + 1):
            if x == cb[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur
    return (2.0 * prev[n]) / float(m + n)


def _box_iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    area_a = float(max(1, (a[2] - a[0]) * (a[3] - a[1])))
    area_b = float(max(1, (b[2] - b[0]) * (b[3] - b[1])))
    return inter / (area_a + area_b - inter)


def _center_inside(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> bool:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bh = max(1.0, float(b[3] - b[1]))
    if not (b[0] <= acx <= b[2]):
        return False
    return abs(acy - (b[1] + b[3]) * 0.5) <= bh * DEDUP_CENTER_RATIO


def _center_gap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    acx = (a[0] + a[2]) * 0.5
    acy = (a[1] + a[3]) * 0.5
    bcx = (b[0] + b[2]) * 0.5
    bcy = (b[1] + b[3]) * 0.5
    dx = float(acx - bcx)
    dy = float(acy - bcy)
    return math.sqrt(dx * dx + dy * dy)


def _median_box_height(
    rows: Sequence[Tuple[str, float, Tuple[int, int, int, int]]]
) -> float:
    hs = sorted(
        float(b[3] - b[1]) for _t, _s, b in rows if float(b[3] - b[1]) > 0.0
    )
    if not hs:
        return 18.0
    return max(8.0, hs[len(hs) // 2])


def dedup_rows(
    rows: Sequence[Tuple[str, float, Tuple[int, int, int, int]]],
    log: Optional[List[str]] = None,
    category: str = "",
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    items = [r for r in rows if str(r[0] or "").strip()]
    if len(items) <= 1:
        return list(items)

    unit = _median_box_height(items)
    near_gate = unit * DEDUP_SAME_TEXT_UNITS

    items.sort(key=lambda r: (-float(r[1]), r[2][1], r[2][0]))

    kept: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    dropped_geo = 0
    dropped_txt = 0
    twin_cells = 0

    for text, score, box in items:
        hit = -1
        reason = ""
        far_twin = False

        for i, (ktext, _ks, kbox) in enumerate(kept):
            if _box_iou(box, kbox) >= DEDUP_IOU:
                hit = i
                reason = "geo"
                break
            if _center_inside(box, kbox) or _center_inside(kbox, box):
                if _text_similarity(text, ktext) >= DEDUP_TEXT_SIM:
                    hit = i
                    reason = "geo"
                    break
            if _text_similarity(text, ktext) >= DEDUP_SAME_TEXT_SIM:
                if _center_gap(box, kbox) <= near_gate:
                    hit = i
                    reason = "txt"
                    break
                far_twin = True

        if hit < 0:
            if far_twin:
                twin_cells += 1
            kept.append((text, score, box))
            continue

        if reason == "geo":
            dropped_geo += 1
        else:
            dropped_txt += 1

        ktext, kscore, kbox = kept[hit]
        if float(score) > float(kscore):
            merged = (
                min(kbox[0], box[0]), min(kbox[1], box[1]),
                max(kbox[2], box[2]), max(kbox[3], box[3]),
            )
            kept[hit] = (text, score, merged)

    kept.sort(key=lambda r: (r[2][1], r[2][0]))

    if log is not None and (dropped_geo or dropped_txt or twin_cells):
        tail = (
            f" | 같은 글자·다른 좌표 {twin_cells}건은 표의 별개 셀로 보고 "
            f"보존했습니다"
            if twin_cells else ""
        )
        log.append(
            f"    🔁 [TILE DEDUP] '{category}' 타일 겹침에서 같은 줄이 "
            f"중복 인식되어 {dropped_geo + dropped_txt}건을 제거했습니다 "
            f"(좌표 겹침 {dropped_geo} / 문자 동일 {dropped_txt} "
            f"| 근접 기준 {near_gate:.0f}px). "
            f"{len(items)}줄 → {len(kept)}줄{tail}"
        )
    return kept


def _ocr_tiles(
    ocr,
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_px: float,
    max_tiles: int,
    log: Optional[List[str]] = None,
    category: str = "",
    table_rows: int = 0,
    legible: int = 0,
    patches: int = 0,
    text_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[str, float, str, int, Optional[Image.Image]]:
    tx0, ty0, tx1, ty1 = (int(v) for v in bbox)
    inside_boxes = [
        b for b in (text_boxes or [])
        if b[0] < tx1 and b[2] > tx0 and b[1] < ty1 and b[3] > ty0
    ]

    if int(patches) > 0 and not inside_boxes and (
        bool(text_boxes) or int(legible) <= 0
    ):
        if log is not None:
            log.append(
                f"    🚫 [EMPTY CROP SKIP] '{category}' 크롭을 가로지르는 "
                f"검출 박스가 0개입니다 (판독 가능 {legible}/{patches}). "
                f"검출기가 지면 전체에서는 글자를 찾았는데 이 자리만 "
                f"비었다는 뜻이므로 OCR·VLM 호출을 모두 생략합니다."
            )
        return "", 1.0, "", 0, None, []

    tiles, reason = decide_tile_count(
        image, bbox, max_tiles=max_tiles,
        table_rows=table_rows, legible=legible, patches=patches,
        text_boxes=text_boxes,
    )

    if log is not None:
        log.append(
            f"    🧱 [TILE PLAN] '{category}' → {tiles}타일 "
            f"| 사유: {reason} | 표행 {table_rows} "
            f"| 판독가능 {legible}/{patches} | 검출박스 {len(inside_boxes)}개"
        )

    usable = (
        ocr is not None
        and getattr(ocr, "available", False)
        and hasattr(ocr, "recognize_boxed")
    )

    collected_rows: List[Tuple[str, float, Tuple[int, int, int, int]]] = []

    def _read(crop_img, origin, scale):
        if not usable or crop_img is None:
            return []
        try:
            rows = ocr.recognize_boxed(crop_img)
        except Exception as e:
            if log is not None:
                log.append(f"    ⚠ OCR 실패: {e}")
            return []

        ox, oy = origin
        sx, sy = scale
        out = []
        for text, score, box in rows:
            gx0 = int(ox + box[0] / max(1e-6, sx))
            gy0 = int(oy + box[1] / max(1e-6, sy))
            gx1 = int(ox + box[2] / max(1e-6, sx))
            gy1 = int(oy + box[3] / max(1e-6, sy))
            out.append((text, score, (gx0, gy0, gx1, gy1)))
        return out

    if tiles <= 1:
        crop, factor, mode = prepare_crop(
            image, bbox, target_px=target_px, log=log, text_boxes=text_boxes
        )
        if not usable and log is not None:
            log.append(
                f"    ⏭ [{category}] PP-OCRv5 rec 미가동 — OCR 초안 없이 "
                f"VLM 판독에 맡깁니다."
            )
        rows = _read(crop, (bbox[0], bbox[1]), (factor, factor))
        rows = dedup_rows(rows, log=log, category=category)
        collected_rows.extend(rows)
        return (
            "\n".join(t for t, _s, _b in rows),
            factor, mode, 1, crop, list(collected_rows),
        )

    boxes = plan_overlap_tiles(
        bbox, tiles, overlap_ratio=0.12,
        image=image, text_boxes=text_boxes, log=log, category=category,
    )
    collected: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    factor = 1.0
    mode = ""
    first_crop: Optional[Image.Image] = None

    for i, tb in enumerate(boxes):
        crop, f, m = prepare_crop(
            image, tb, target_px=target_px, log=None, text_boxes=text_boxes
        )
        if i == 0:
            first_crop = crop
            factor, mode = f, m
        collected.extend(_read(crop, (tb[0], tb[1]), (f, f)))

    if log is not None and collected:
        log.append(
            f"    📥 [TILE READ] '{category}' 타일 {len(boxes)}장에서 "
            f"{len(collected)}줄 수집 (원본 좌표로 환산)"
        )

    rows = dedup_rows(collected, log=log, category=category)
    collected_rows.extend(rows)

    return (
        "\n".join(t for t, _s, _b in rows),
        factor, mode, len(boxes), first_crop, list(collected_rows),
    )


def read_by_lines(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    refiner,
    category: str,
    lang_code: str = "",
    script: str = "",
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    span: int = 1,
    stride: int = 1,
    overlap: int = 1,
    ocr=None,
    log: Optional[List[str]] = None,
) -> Tuple[str, List[str], int]:
    if refiner is None or not getattr(refiner, "vision", False):
        return "", [], 0

    prose = is_prose_mode()

    if _LINE_READ_DEAD["off"] and not prose:
        if log is not None:
            log.append(
                f"    ⏭ [LINE READ] '{category}' — VLM 행 판독이 연속 "
                f"{LINE_READ_DEAD_CROPS}개 크롭에서 전용 인식기 결과를 "
                f"한 글자도 바꾸지 못했습니다. 이 문서에서는 전용 인식기 "
                f"결과만 씁니다."
            )
        return "", [], 0

    try:
        from core.line_reader import (
            OCR_TRUST_CONF,
            collect_ocr_votes,
            read_lines,
            split_lines,
        )
    except Exception as e:
        if log is not None:
            log.append(f"    ⏭ 행 판독 모듈 로드 실패 ({e})")
        return "", [], 0

    cropped = crop_region(image, bbox)
    if cropped.width < 8 or cropped.height < 8:
        return "", [], 0

    bx0, by0, bx1, by1 = (int(v) for v in bbox)
    local: List[Tuple[int, int, int, int]] = []
    for b in (text_boxes or []):
        if b[0] >= bx1 or b[2] <= bx0 or b[1] >= by1 or b[3] <= by0:
            continue
        local.append((
            max(0, b[0] - bx0), max(0, b[1] - by0),
            min(cropped.width, b[2] - bx0), min(cropped.height, b[3] - by0),
        ))

    ocr_fn = None
    if ocr is not None and getattr(ocr, "available", False):
        if hasattr(ocr, "recognize_lines"):
            def ocr_fn(crops):
                return ocr.recognize_lines(crops)

    def _global(lb) -> Tuple[int, int, int, int]:
        return (
            bx0 + int(lb[0]), by0 + int(lb[1]),
            bx0 + int(lb[2]), by0 + int(lb[3]),
        )

    probe = split_lines(cropped, boxes=local, log=None)
    total = len(probe)
    hits: Dict[int, str] = {}
    for ln in probe:
        got = _ledger_get(_global(ln.bbox))
        if got:
            hits[ln.index] = got

    if total > 0 and len(hits) >= total * LEDGER_REUSE_RATIO:
        seen = len(hits)
        gap = [ln for ln in probe if ln.index not in hits]
        filled = 0
        if gap and ocr_fn is not None:
            votes = collect_ocr_votes(gap, ocr_fn, log=None)
            for ln in gap:
                got = votes.get(ln.index)
                if got and str(got[0]).strip():
                    hits[ln.index] = str(got[0]).strip()
                    filled += 1

        texts = [hits.get(ln.index, "") for ln in probe]
        kept = [t for t in texts if t]
        _ledger_store([_global(ln.bbox) for ln in probe], texts)

        if log is not None:
            tail = (
                f" | 남은 {len(gap)}행은 전용 인식기로만 채웠습니다"
                f"({filled}행 확보)"
                if gap else ""
            )
            log.append(
                f"    ♻️ [ROW LEDGER] '{category}' 행 {total}개 중 {seen}개가 "
                f"앞선 크롭에서 이미 확정되었습니다 ({seen / total:.0%} ≥ "
                f"{LEDGER_REUSE_RATIO:.0%}). VLM 호출 없이 재사용합니다 — "
                f"같은 행을 두 번 읽지 않습니다{tail}."
            )
        return "\n".join(kept), kept, 0

    if prose:
        eff_span, eff_stride, eff_overlap = 1, 1, max(1, int(overlap))
        trust = PROSE_TRUST_CONF
    else:
        eff_span = max(1, int(span))
        eff_stride = max(1, int(stride))
        eff_overlap = max(0, int(overlap))
        trust = float(OCR_TRUST_CONF)

    calls = {"n": 0}

    def _reader(win_img: Image.Image, start: int, end: int) -> str:
        calls["n"] += 1
        span_n = max(1, end - start)
        hint = (
            f"{category} — line {start + 1}"
            if span_n == 1
            else f"{category} — lines {start + 1}~{end}"
        )
        try:
            return refiner.read_raw(
                win_img, hint=hint, max_chars=160,
                lang_code=lang_code, script=script,
            )
        except Exception:
            return ""

    merged, lines, windows = read_lines(
        cropped, _reader, boxes=local,
        span=eff_span, stride=eff_stride, overlap=eff_overlap,
        ocr_fn=ocr_fn, trust_conf=trust, log=log,
    )

    changed = sum(
        1 for l in lines
        if str(getattr(l, "source", "")) == "vlm-override"
    )
    vlm_rows = sum(
        1 for l in lines
        if l.text and str(getattr(l, "source", "")) not in
        ("ocr-only", "ocr-trusted")
    )

    _LINE_READ_DEAD["calls"] += int(calls["n"])
    _LINE_READ_DEAD["changed"] += int(changed)

    stored = _ledger_store(
        [_global(l.bbox) for l in lines],
        [l.text for l in lines],
    )

    if calls["n"] > 0 and changed == 0 and not prose:
        _LINE_READ_DEAD["strikes"] += 1
        if log is not None:
            log.append(
                f"    ⚠ [LINE READ] '{category}' VLM 호출 {calls['n']}회가 "
                f"전용 인식기 결과를 한 글자도 바꾸지 못했습니다 "
                f"({_LINE_READ_DEAD['strikes']}/{LINE_READ_DEAD_CROPS}회 "
                f"연속)."
            )
        if _LINE_READ_DEAD["strikes"] >= LINE_READ_DEAD_CROPS:
            _LINE_READ_DEAD["off"] = True
            if log is not None:
                log.append(
                    f"    ⛔ [LINE READ OFF] 누적 VLM 호출 "
                    f"{_LINE_READ_DEAD['calls']}회 / 실제 교정 "
                    f"{_LINE_READ_DEAD['changed']}행. 남은 크롭에서는 행 "
                    f"판독을 중단합니다 — 품질 손실 없이 호출을 전량 "
                    f"절약합니다."
                )
    elif changed > 0:
        _LINE_READ_DEAD["strikes"] = 0

    if log is not None:
        src = "VLM + 전용 인식기" if ocr_fn is not None else "VLM 단독"
        tag = "산문" if prose else "서식"
        tail = f" | 원장 적재 {stored}행" if stored else ""
        log.append(
            f"    🔁 [LINE READ] '{category}' VLM 호출 {calls['n']}회 "
            f"({src} / {tag} 모드) — VLM 참여 {vlm_rows}행 / 실제 교정 "
            f"{changed}행{tail}"
        )

    return merged, [l.text for l in lines if l.text], len(windows)


TABLE_HEADER_MIN_COLS = 3
TABLE_COL_TOLERANCE = 0.55
TABLE_ROW_TOLERANCE = 0.70
TABLE_HEADER_MERGE_BANDS = 2
TABLE_HEADER_X_OVERLAP = 0.45
TABLE_HEADER_ALT_DEPTH = 8


def reconcile_claims(
    fields: Sequence[ExtractedField],
    proposals: Sequence[Tuple[float, int, str, str, str]],
    schema: Optional[dict],
    log: Optional[List[str]] = None,
) -> None:
    if not fields:
        return

    owner = _field_category_map(schema)
    ranked = sorted(proposals, key=lambda p: (-float(p[0]), int(p[1])))

    final: Dict[str, Tuple[str, float, str]] = {}
    value_owner: Dict[str, Tuple[str, float]] = {}
    dropped_field = 0
    dropped_value = 0

    for score, order, category, field, value in ranked:
        vkey = _norm_text(value)
        if field in final:
            dropped_field += 1
            continue
        held = value_owner.get(vkey) if vkey else None
        if held is not None and held[0] != field:
            dropped_value += 1
            continue
        final[str(field)] = (str(value), float(score), str(category))
        if vkey:
            value_owner[vkey] = (str(field), float(score))

    seen_rows: Dict[str, str] = {
        k: v[0] for k, v in value_owner.items()
    }
    row_dropped = 0
    row_notes: List[str] = []
    table_kept: List[str] = []

    for f in sorted(
        fields, key=lambda x: -float(getattr(x, "score", 0.0))
    ):
        rows = getattr(f, "rows", None) or []
        if not rows:
            continue
        if bool(getattr(f, "table_backed", False)):
            table_kept.append(f"{f.category}({len(rows)}행)")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for k, v in row.items():
                    vkey = _norm_text(v)
                    if vkey and vkey not in seen_rows:
                        seen_rows[vkey] = str(k)
            continue
        kept: List[dict] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            live: Dict[str, object] = {}
            for k, v in row.items():
                vkey = _norm_text(v)
                if not vkey:
                    continue
                held = seen_rows.get(vkey)
                if held is not None and held != str(k):
                    row_dropped += 1
                    if len(row_notes) < 8:
                        row_notes.append(
                            f"{f.category}.{k}={str(v)[:20]} → "
                            f"'{held}' 선점"
                        )
                    continue
                seen_rows[vkey] = str(k)
                live[k] = v
            if live:
                kept.append(live)
        f.rows = kept

    if row_dropped and log is not None:
        log.append(
            f"  🧮 [ROW CLAIM] 배열 셀 {row_dropped}건을 버렸습니다 — 점수가 "
            f"더 높은 크롭이 같은 글자를 이미 다른 필드로 확정했습니다. "
            f"예전에는 단일값만 값 배타를 거쳐, 같은 말풍선 한 줄이 여러 "
            f"카테고리 배열에 그대로 복제됐습니다."
        )
        for note in row_notes:
            log.append(f"     · {note}")

    if table_kept and log is not None:
        log.append(
            f"  🧾 [ROW CLAIM EXEMPT] 표 헤더로 컬럼이 확정된 배열 "
            f"{', '.join(table_kept)} 은 값 배타에서 제외했습니다 — 셀 위치가 "
            f"헤더 x 좌표로 이미 정해져 있으므로, 같은 숫자가 QTY 열의 여러 "
            f"행에 나오는 것이 정상입니다. 위치 근거 없는 산문 행에만 값 "
            f"배타를 겁니다."
        )

    by_cat: Dict[str, ExtractedField] = {}
    for f in fields:
        by_cat.setdefault(str(f.category), f)

    for f in fields:
        f.field_values = {}
        f.donations = {}

    orphan: Dict[str, Tuple[str, float]] = {}
    for field, payload in final.items():
        value, score, _cat = payload
        home = str(owner.get(field, ""))
        target = by_cat.get(home)
        if target is not None:
            target.field_values[field] = value
            continue
        orphan[field] = (value, float(score))

    if orphan:
        fields[0].donations = orphan

    if log is not None:
        log.append(
            f"  🧮 [GLOBAL CLAIM] 크롭 {len(fields)}개가 제안한 "
            f"{len(proposals)}건을 전역 배타 배정했습니다 — 확정 "
            f"{len(final)}건 | 필드 선점 탈락 {dropped_field}건 "
            f"| 값 선점 탈락 {dropped_value}건 | 소유 크롭 없음 "
            f"{len(orphan)}건"
        )
        for field, payload in sorted(
            final.items(), key=lambda kv: -kv[1][1]
        )[:16]:
            log.append(
                f"     · {field} = {str(payload[0])[:36]} "
                f"(점수 {float(payload[1]):.2f} / 출처 '{payload[2]}')"
            )


def build_table_rows(
    rows: Optional[Sequence[Tuple[str, float, Tuple[int, int, int, int]]]],
    schema: Optional[dict],
    category: str,
    log: Optional[List[str]] = None,
) -> List[dict]:
    items = [
        (t, s, b) for t, s, b in (rows or [])
        if str(t or "").strip()
    ]
    if len(items) < TABLE_HEADER_MIN_COLS * 2:
        return []

    index, bank = build_label_index(schema, category=category)
    if not index:
        return []

    try:
        from core.text_prep import is_printed_label
    except Exception:
        return []

    heights = sorted(float(b[3] - b[1]) for _t, _s, b in items)
    unit = max(8.0, heights[len(heights) // 2] if heights else 20.0)

    bands: Dict[int, List[int]] = {}
    for i, (_t, _s, b) in enumerate(items):
        band = int(round(((b[1] + b[3]) * 0.5) / (unit * TABLE_ROW_TOLERANCE)))
        bands.setdefault(band, []).append(i)

    def _merge_header_cells(group: Sequence[int]) -> List[Tuple[str, float, float]]:
        cells: List[List[object]] = []
        for band in group:
            for i in bands.get(band, []):
                text, _s, b = items[i]
                t = str(text).strip()
                if not t or not is_printed_label(t, bank):
                    continue
                cells.append([t, float(b[0]), float(b[2]), int(band)])
        cells.sort(key=lambda c: (c[3], c[1]))

        merged: List[List[object]] = []
        for cell in cells:
            hit = -1
            for j, m in enumerate(merged):
                if int(m[3]) == int(cell[3]):
                    continue
                ov = min(float(m[2]), float(cell[2])) - max(
                    float(m[1]), float(cell[1])
                )
                span = min(
                    float(m[2]) - float(m[1]),
                    float(cell[2]) - float(cell[1]),
                )
                if span <= 0.0:
                    continue
                if ov / span >= TABLE_HEADER_X_OVERLAP:
                    hit = j
                    break
            if hit < 0:
                merged.append(list(cell))
                continue
            m = merged[hit]
            m[0] = f"{m[0]} {cell[0]}".strip()
            m[1] = min(float(m[1]), float(cell[1]))
            m[2] = max(float(m[2]), float(cell[2]))
        return [(str(m[0]), float(m[1]), float(m[2])) for m in merged]

    ordered = sorted(bands.keys())
    header_band = -1
    header_cols: List[Tuple[str, float, float]] = []
    header_slots: List[Tuple[str, float, float]] = []
    header_dup = 0
    header_moved = 0
    header_span = 1

    for bi, _band in enumerate(ordered):
        for span in range(1, TABLE_HEADER_MERGE_BANDS + 1):
            group = ordered[bi: bi + span]
            if len(group) < span:
                break

            raw: List[Tuple[str, float, float, float, List[Tuple[str, float]]]] = []
            for text, x0, x1 in _merge_header_cells(group):
                floor = label_cosine_floor(text)
                ranked_all = sorted(
                    index.scores(text).items(),
                    key=lambda kv: kv[1], reverse=True,
                )
                alt = [
                    (str(k), float(v)) for k, v in ranked_all
                    if float(v) >= floor
                ][:TABLE_HEADER_ALT_DEPTH]
                head = alt[0][0] if alt else ""
                hs = alt[0][1] if alt else 0.0
                raw.append((head, hs, x0, x1, alt))

            pool: List[Tuple[float, int, str]] = []
            for i, (_f, _s, _x0, _x1, alt) in enumerate(raw):
                for name, sc in alt:
                    pool.append((float(sc), i, str(name)))
            pool.sort(key=lambda t: (-t[0], t[1]))

            taken_col: Dict[int, str] = {}
            taken_field: Dict[str, int] = {}
            for sc, i, name in pool:
                if i in taken_col or name in taken_field:
                    continue
                taken_col[i] = name
                taken_field[name] = i

            slots: List[Tuple[str, float, float]] = []
            cols: List[Tuple[str, float, float]] = []
            dup = 0
            moved = 0
            for i, (head, _hs, x0, x1, alt) in enumerate(raw):
                name = taken_col.get(i, "")
                if not name:
                    if alt:
                        dup += 1
                    slots.append(("", x0, x1))
                    continue
                if head and name != head:
                    moved += 1
                slots.append((name, x0, x1))
                cols.append((name, x0, x1))

            if len(cols) < TABLE_HEADER_MIN_COLS:
                continue
            cols.sort(key=lambda c: c[1])
            if len(cols) >= TABLE_HEADER_MIN_COLS:
                header_band = int(group[-1])
                header_cols = cols
                header_slots = sorted(slots, key=lambda c: c[1])
                header_dup = dup
                header_moved = moved
                header_span = span
                break
        if header_band >= 0:
            break

    if header_band < 0:
        return []

    if log is not None:
        blind = [s for s in header_slots if not s[0]]
        if blind or header_moved:
            log.append(
                f"    🕳 [TABLE BLIND COL] 헤더 칸 {len(header_slots)}개 중 "
                f"{len(blind)}개는 스키마 필드에 붙지 않았습니다 "
                f"(x≈{', '.join(str(int(s[1])) for s in blind[:6]) or '-'}). "
                f"중복으로 밀린 칸 {header_dup + header_moved}개 중 "
                f"{header_moved}개는 사전 차점 필드로 옮겨 살렸고 "
                f"{header_dup}개는 차점도 없어 맹점으로 내렸습니다 — OCR 이 "
                f"'UNIT OF MEASURE' 'UNIT WEIG' 'UNIT VALU' 를 모두 같은 "
                f"필드로 밀어 넣으면 한 칸만 이기고 나머지 자리의 값이 옆 "
                f"컬럼으로 밀립니다."
            )

    if log is not None:
        brief = " | ".join(f"{f}@{int(x0)}" for f, x0, _x1 in header_cols)
        tail = (
            f" | 세로 병합 {header_span}행 (줄바꿈 헤더 복원)"
            if header_span > 1 else ""
        )
        log.append(
            f"    🧾 [TABLE HEADER] '{category}' 컬럼 {len(header_cols)}개 "
            f"인식 — {brief}{tail}"
        )

    out: List[dict] = []
    blind_drop = 0
    for band in sorted(b for b in bands.keys() if b > header_band):
        row: Dict[str, str] = {}
        for i in bands[band]:
            text, _s, b = items[i]
            t = str(text).strip()
            if not t or is_printed_label(t, bank):
                continue
            cx = (b[0] + b[2]) * 0.5

            slot = ""
            slot_dist = 1e9
            for field, x0, x1 in header_slots:
                if x0 <= cx <= x1:
                    slot = str(field)
                    slot_dist = 0.0
                    break
                dist = min(abs(cx - x0), abs(cx - x1))
                if dist < slot_dist:
                    slot = str(field)
                    slot_dist = dist

            if not slot:
                blind_drop += 1
                continue

            width = max(1.0, header_slots[-1][2] - header_slots[0][1])
            if slot_dist <= width * TABLE_COL_TOLERANCE:
                row.setdefault(slot, t)

        if len(row) >= 2:
            out.append(row)

    if log is not None:
        tail = (
            f" | 미인식 컬럼 자리에 떨어져 버린 값 {blind_drop}건"
            if blind_drop else ""
        )
        log.append(
            f"    🧾 [TABLE ROWS] '{category}' 데이터 행 {len(out)}건 구성 "
            f"(헤더 아래 밴드 기준){tail}"
        )
        for r in out[:3]:
            brief = " | ".join(f"{k}={v[:14]}" for k, v in list(r.items())[:5])
            log.append(f"       · {brief}")

    return out


def nlp_clean_ocr(
    text: str,
    nlp=None,
    log: Optional[List[str]] = None,
    min_content_ratio: float = 0.20,
) -> Tuple[str, dict]:
    raw = (text or "").strip()
    meta = {"lines_in": 0, "lines_out": 0, "dropped": 0, "lemma": ""}

    if not raw:
        return raw, meta

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    meta["lines_in"] = len(lines)

    if nlp is None or not getattr(nlp, "available", False):
        meta["lines_out"] = len(lines)
        return "\n".join(lines), meta

    kept: List[str] = []
    for ln in lines:
        verdict = nlp.analyze(ln)
        if verdict is None:
            kept.append(ln)
            continue
        if verdict.is_entity or verdict.content_ratio >= min_content_ratio:
            kept.append(ln)
            continue
        meta["dropped"] += 1
        if log is not None:
            log.append(f"    🧬 NLP 노이즈 제거 '{ln[:40]}' ({verdict.reason})")

    if not kept:
        kept = lines
        meta["dropped"] = 0

    meta["lines_out"] = len(kept)
    cleaned = "\n".join(kept)

    try:
        meta["lemma"] = nlp.canonicalize(cleaned)
    except Exception:
        meta["lemma"] = ""

    return cleaned, meta


def extract_from_crops(
    image: Image.Image,
    plans: Sequence[CropPlan],
    ocr,
    refine_fn: Optional[Callable[[str, str, Image.Image], dict]] = None,
    target_px: float = 28.0,
    max_tiles: int = 3,
    keep_crop: bool = True,
    nlp=None,
    log: Optional[List[str]] = None,
    array_categories: Optional[set] = None,
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    line_read_fn: Optional[Callable] = None,
    schema: Optional[dict] = None,
) -> List[ExtractedField]:
    OCR_MEMO_MAX = 24

    out: List[ExtractedField] = []
    arrays = set(array_categories or ())
    tboxes = list(text_boxes or [])
    ocr_memo: Dict[Tuple[int, int, int, int], Tuple[str, float, str, int]] = {}
    proposals: List[Tuple[float, int, str, str, str]] = []

    def _guard_memo() -> None:
        if len(ocr_memo) <= OCR_MEMO_MAX:
            return
        for key in list(ocr_memo.keys())[: len(ocr_memo) - OCR_MEMO_MAX]:
            ocr_memo.pop(key, None)
        if log is not None:
            log.append(
                f"    🧹 [OCR MEMO] 재사용 캐시를 {OCR_MEMO_MAX}건으로 "
                f"잘라 RAM 을 되돌립니다. 정확도에는 영향이 없습니다."
            )
    doc_code = str(
        (schema or {}).get("code") or (schema or {}).get("doc_type") or ""
    )
    seq = 0
    twin_table_failed: set = set()
    reset_line_read_health()
    reset_refine_health()
    reset_read_mode()
    reset_doc_script()

    for _p in plans:
        _memo = ocr_memo.get(tuple(_p.bbox))
        if _memo is not None:
            observe_script(_memo[0])

    for plan in plans:
        if log is not None:
            log.append(f"  ✂️ '{plan.category}' 크롭 px{plan.bbox}")

        memo = ocr_memo.get(tuple(plan.bbox))
        if memo is not None:
            text, factor, mode, tiles, crop_rows = memo
            crop = crop_region(image, plan.bbox)
            if log is not None:
                log.append(
                    f"    ♻️ [OCR REUSE] 같은 좌표를 이미 읽었습니다 — "
                    f"재인식 없이 {len(text.splitlines())}줄을 재사용합니다."
                )
        else:
            text, factor, mode, tiles, crop, crop_rows = _ocr_tiles(
                ocr, image, plan.bbox, target_px, max_tiles, log=log,
                category=plan.category,
                table_rows=int(getattr(plan, "table_rows", 0)),
                legible=int(getattr(plan, "legible", 0)),
                patches=int(getattr(plan, "patches_total", 0)),
                text_boxes=tboxes,
            )
            ocr_memo[tuple(plan.bbox)] = (text, factor, mode, tiles, crop_rows)
            _guard_memo()

        decide_read_mode(crop_rows, schema, log=log)
        active_refine = None if _REFINE_HEALTH["off"] else refine_fn

        cleaned, nlp_meta = nlp_clean_ocr(text, nlp=nlp, log=log)
        if nlp_meta["dropped"] and log is not None:
            log.append(
                f"    🧬 NLP 게이트: {nlp_meta['lines_in']} → "
                f"{nlp_meta['lines_out']}줄 ({nlp_meta['dropped']}줄 제거)"
            )

        observe_script(cleaned)

        line_texts: List[str] = []
        windows_n = 0
        if line_read_fn is not None:
            try:
                merged, line_texts, windows_n = line_read_fn(
                    image, plan.bbox, plan.category, tboxes, ocr, log
                )
            except Exception as e:
                if log is not None:
                    log.append(f"    ⚠ 행 판독 실패 ({e})")
                merged = ""
            if merged and len(line_texts) >= LINE_READ_MIN_LINES:
                if log is not None:
                    log.append(
                        f"    🔀 [DRAFT SWAP] '{plan.category}' 초안을 행 "
                        f"단위 판독 결과로 교체합니다 "
                        f"({len(cleaned)}자 → {len(merged)}자 / "
                        f"{len(line_texts)}행)"
                    )
                cleaned = merged
            elif merged and not cleaned.strip():
                cleaned = merged

        if not line_texts and cleaned.strip():
            line_texts = [
                ln.strip() for ln in cleaned.splitlines() if ln.strip()
            ]

        refined: Dict[str, object] = {}
        values: Dict[str, str] = {}
        rows_out: List[dict] = []

        if active_refine is not None:
            vlm_crop = crop
            if vlm_crop is None:
                vlm_crop = crop_region(image, plan.bbox)
            try:
                vlm_crop = fit_for_vlm(vlm_crop, log=log)
            except Exception:
                pass
            try:
                refined = active_refine(
                    plan.category, cleaned, vlm_crop, plan.top_field
                ) or {}
            except TypeError:
                try:
                    refined = active_refine(plan.category, cleaned, vlm_crop) or {}
                except Exception as e:
                    if log is not None:
                        log.append(f"    ⚠ 정제 추출 실패: {e}")
            except Exception as e:
                if log is not None:
                    log.append(f"    ⚠ 정제 추출 실패: {e}")

            if isinstance(refined, dict) and refined.get("__refiner_down__"):
                mark_refiner_down(str(plan.category), log=log)
                refined = {}
                active_refine = None

            raw_alt = refined.get("__raw__") if isinstance(refined, dict) else ""
            if isinstance(raw_alt, str) and raw_alt.strip():
                if log is not None:
                    log.append(
                        f"    👁 [RAW READ] '{plan.category}' OCR 원문을 VLM "
                        f"판독으로 교체합니다: {raw_alt[:48]!r}"
                    )
                cleaned = raw_alt.strip()
            if isinstance(refined, dict):
                inner = refined.get("__fields__")
                if isinstance(inner, dict):
                    values = {
                        str(k): str(v) for k, v in inner.items()
                        if isinstance(v, str) and v.strip()
                    }
                rowset = refined.get("__rows__")
                if isinstance(rowset, list):
                    rows_out = [r for r in rowset if isinstance(r, dict) and r]

            gained = len(values) + len(rows_out)
            if isinstance(raw_alt, str) and raw_alt.strip():
                gained += 1
            _note_refine(gained, plan.category, log=log)

        donated: Dict[str, Tuple[str, float]] = {}
        if active_refine is None and not values and not rows_out:
            paired, donated = promote_by_labels(
                cleaned, schema, plan.category, log=log, rows=crop_rows,
                doc_code=doc_code,
            )
            for pname, ppayload in paired.items():
                values[pname] = str(ppayload[0])
                seq += 1
                proposals.append((
                    float(ppayload[1]), seq, str(plan.category),
                    str(pname), str(ppayload[0]),
                ))
            for dname, dpayload in donated.items():
                seq += 1
                proposals.append((
                    float(dpayload[1]), seq, str(plan.category),
                    str(dname), str(dpayload[0]),
                ))

        owned = int(getattr(plan, "owned_patches", -1))
        if active_refine is None and not values and not rows_out and owned == 0:
            if log is not None:
                log.append(
                    f"    ⛔ [PROSE TERRITORY] '{plan.category}' 크롭은 최종 "
                    f"좌표 안에 자기 영토 패치를 한 칸도 담지 않았습니다. "
                    f"인쇄 라벨이 없는 지면에서 줄 전체를 값으로 승격하면 "
                    f"남의 셀 글자가 이 축의 값이 됩니다 — OCR 원문만 "
                    f"보존합니다."
                )
        elif active_refine is None and not values and not rows_out:
            prose_vals, prose_rows = promote_prose_lines(
                cleaned, schema, plan.category,
                top_field=str(getattr(plan, "top_field", "") or ""),
                is_array=plan.category in arrays,
                rows=crop_rows,
                log=log,
            )
            for pname, pval in prose_vals.items():
                values[pname] = str(pval)
                seq += 1
                proposals.append((
                    PROSE_CLAIM_SCORE, seq, str(plan.category),
                    str(pname), str(pval),
                ))
            if prose_rows:
                rows_out.extend(prose_rows)

        is_twin = str(getattr(plan, "source", "")).startswith("twin-of:")
        twin_owner = ""
        if is_twin:
            twin_owner = str(plan.source).split(":", 1)[-1]
        owner_failed = bool(twin_owner) and twin_owner in twin_table_failed
        table_backed = False

        if is_twin and plan.category in arrays and not owner_failed:
            if log is not None:
                log.append(
                    f"    ⛔ [TWIN SKIP] '{plan.category}' 는 '{twin_owner}' 와 "
                    f"좌표가 같아 배열을 만들지 않습니다."
                )
        elif active_refine is None and plan.category in arrays and not rows_out:
            if is_twin and owner_failed and log is not None:
                log.append(
                    f"    🔄 [TWIN RETRY] 좌표 소유자 '{twin_owner}' 가 표 "
                    f"헤더를 찾지 못했으므로 '{plan.category}' 가 대신 "
                    f"시도합니다 — 같은 표라도 스키마에 그 컬럼을 가진 쪽만 "
                    f"배열을 만들 수 있습니다."
                )
            table = build_table_rows(
                crop_rows, schema, plan.category, log=log
            )
            if table:
                rows_out.extend(table)
                table_backed = True
            else:
                twin_table_failed.add(str(plan.category))
                if log is not None:
                    log.append(
                        f"    ⚪ [TABLE] '{plan.category}' 에서 표 헤더를 찾지 "
                        f"못해 배열을 만들지 않습니다. 라벨 없는 값 나열은 "
                        f"어느 컬럼인지 알 수 없습니다."
                    )

        if active_refine is None and plan.category not in arrays and not values:
            if log is not None:
                log.append(
                    f"    ⚪ [OCR ONLY] '{plan.category}' 에서 라벨↔값 쌍을 "
                    f"찾지 못해 값을 비워 둡니다. OCR 원문은 "
                    f"ocr_by_category 에 보존됩니다."
                )

        field = ExtractedField(
            category=plan.category,
            bbox=plan.bbox,
            score=plan.score,
            margin=plan.margin,
            ocr_text=cleaned,
            refined=refined,
            upscale=factor,
            upscale_mode=mode,
            tiles=tiles,
            crop_image=crop if keep_crop else None,
        )
        field.ocr_raw = text
        field.lemma = nlp_meta.get("lemma", "")
        field.nlp_meta = nlp_meta
        field.field_values = values
        field.donations = dict(donated)
        field.rows = rows_out if plan.category in arrays else []
        field.table_backed = bool(table_backed and field.rows)
        field.line_texts = list(line_texts)
        field.line_windows = int(windows_n)
        out.append(field)

        if log is not None:
            if values:
                brief = " | ".join(
                    f"{k}={str(v)[:24]}" for k, v in list(values.items())[:5]
                )
                log.append(f"    📝 [{plan.category}] {brief}")
            else:
                preview = field.value[:60].replace("\n", " / ")
                log.append(f"    📝 '{plan.category}' = {preview or '(공백)'}")

    reconcile_claims(out, proposals, schema, log=log)

    return out


def _array_categories(schema: Optional[dict]) -> set:
    out = set()
    for _name, definition in ((schema or {}).get("fields", {}) or {}).items():
        d = definition if isinstance(definition, dict) else {}
        if d.get("array") or d.get("table"):
            out.add(str(d.get("category") or "misc"))
    return out


def fields_to_record(
    fields: Sequence[ExtractedField],
    schema: Optional[dict] = None,
    log: Optional[List[str]] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {}
    raw_text: Dict[str, str] = {}
    schema_fields = set((schema or {}).get("fields", {}) or {})
    array_fields = _array_field_names(schema)
    array_cats = _array_categories(schema)

    carried: List[str] = []
    text_blocked: List[str] = []

    stacked: List[str] = []

    for f in fields:
        if getattr(f, "rows", None):
            prev = record.get(f.category)
            merged = list(prev) if isinstance(prev, list) else []
            for r in f.rows:
                if r not in merged:
                    merged.append(r)
            record[f.category] = merged

            for key, val in (f.field_values or {}).items():
                if not val:
                    continue
                if schema_fields and key not in schema_fields:
                    continue
                if key == f.category:
                    continue
                if record.get(key):
                    continue
                record[key] = val
                carried.append(f"{key}={str(val)[:24]}")
            continue

        if f.field_values:
            for key, val in f.field_values.items():
                if not val:
                    continue
                if schema_fields and key not in schema_fields:
                    continue
                if key in record and record[key]:
                    continue
                record[key] = val

    for f in fields:
        if getattr(f, "rows", None) or f.field_values:
            continue

        val = f.value
        if not val:
            continue
        if schema_fields and f.category not in schema_fields:
            prev = raw_text.get(f.category, "")
            raw_text[f.category] = (prev + "\n" + val).strip() if prev else val
            continue
        if f.category in array_cats:
            text_blocked.append(f.category)
            prev = raw_text.get(f.category, "")
            raw_text[f.category] = (prev + "\n" + val).strip() if prev else val
            continue
        if record.get(f.category):
            stacked.append(f.category)
            prev = raw_text.get(f.category, "")
            raw_text[f.category] = (prev + "\n" + val).strip() if prev else val
            continue
        record[f.category] = val

    if log is not None and stacked:
        uniq = sorted(set(stacked))
        log.append(
            f"  🧱 [VALUE STACK BLOCK] 카테고리 {', '.join(uniq)} 는 이름이 "
            f"스키마 필드와 같아 OCR 원문이 record 로 들어갑니다. 같은 "
            f"카테고리 크롭이 둘 이상일 때 문자열을 배열로 쌓으면 접지가 "
            f"리스트를 건너뛰어 폐기 판정이 무시되고 자연어 요약도 그 "
            f"카테고리를 통째로 빠뜨립니다. 확정값을 남기고 나머지 원문 "
            f"{len(stacked)}건은 ocr_by_category 로 보냈습니다."
        )

    pool: Dict[str, Tuple[str, float]] = {}
    for f in fields:
        for key, payload in (getattr(f, "donations", None) or {}).items():
            if not isinstance(payload, (tuple, list)) or len(payload) < 2:
                continue
            val = str(payload[0] or "").strip()
            if not val:
                continue
            if schema_fields and key not in schema_fields:
                continue
            if key in array_fields:
                continue
            score = float(payload[1])
            prev = pool.get(key)
            if prev is None or score > prev[1]:
                pool[key] = (val, score)

    adopted: List[str] = []
    blocked: List[str] = []
    for key, payload in sorted(pool.items(), key=lambda kv: -kv[1][1]):
        if record.get(key):
            blocked.append(key)
            continue
        record[key] = payload[0]
        adopted.append(f"{key}={payload[0][:24]}")

    if log is not None and text_blocked:
        uniq = sorted(set(text_blocked))
        log.append(
            f"  🧱 [ARRAY TEXT BLOCK] 배열 카테고리 {', '.join(uniq)} 에 "
            f"OCR 원문이 문자열로 들어가려는 것을 {len(text_blocked)}건 "
            f"막았습니다. 같은 키에 딕셔너리 행과 문자열이 섞이면 자연어 "
            f"요약이 그 카테고리를 통째로 건너뜁니다. 원문은 "
            f"ocr_by_category 에 보존됩니다."
        )

    if log is not None and carried:
        log.append(
            f"  🧷 [ROW CARRY] 배열을 만든 크롭이 함께 확정한 단일 필드 "
            f"{len(carried)}건을 배열과 나란히 보존했습니다 — 예전에는 rows 가 "
            f"있으면 그 크롭의 field_values 를 통째로 버려, 전역 배타 배정이 "
            f"확정한 값이 JSON 에서 사라졌습니다."
        )
        for line in carried[:10]:
            log.append(f"     · {line}")

    if log is not None and (adopted or blocked):
        log.append(
            f"  🎁 [DONATION ADOPT] 소유 카테고리가 비워 둔 필드 "
            f"{len(adopted)}건을 기부 풀에서 채웠습니다 "
            f"(소유 값이 이미 있어 건너뜀 {len(blocked)}건)."
        )
        for line in adopted[:10]:
            log.append(f"     · {line}")

    for name in schema_fields:
        record.setdefault(name, None)

    if raw_text:
        record["__ocr_by_category__"] = raw_text
    return record


def record_to_json(
    record: Dict[str, object],
    schema: Optional[dict] = None,
    drop_empty: bool = True,
    source_path: str = "",
    lang_code: str = "",
) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, val in record.items():
        if key.startswith("__"):
            continue
        if drop_empty and (val is None or val == "" or val == []):
            continue
        clean[key] = val

    ocr_raw = record.get("__ocr_by_category__")
    ocr_raw = ocr_raw if isinstance(ocr_raw, dict) else {}

    def _fallback(reason: str) -> Dict[str, object]:
        fields = (schema or {}).get("fields", {}) or {}
        grouped: Dict[str, Dict[str, object]] = {}
        for key, val in clean.items():
            definition = fields.get(key)
            cat = (
                str(definition.get("category") or "misc")
                if isinstance(definition, dict) else "misc"
            )
            grouped.setdefault(cat, {})[key] = val
        out: Dict[str, object] = {
            "doc_type": (schema or {}).get("code") or "",
            "domain": (schema or {}).get("domain") or "",
            "fields": clean,
            "by_category": grouped,
            "shape_error": reason,
        }
        if ocr_raw:
            out["ocr_by_category"] = ocr_raw
        return out

    try:
        from core.record_shape import build_record, relay_plan
    except Exception as e:
        return _fallback(f"{type(e).__name__}: {e}")

    try:
        payload = build_record(
            clean, schema or {},
            source_path=source_path,
            lang_code=lang_code,
            has_vision=True,
            ocr_pool=ocr_raw,
        )
    except TypeError as e:
        try:
            payload = build_record(
                clean, schema or {},
                source_path=source_path,
                lang_code=lang_code,
                has_vision=True,
            )
            payload["shape_warning"] = (
                f"ocr_pool 미지원 build_record — {type(e).__name__}: {e}"
            )
        except Exception as e2:
            return _fallback(f"{type(e2).__name__}: {e2}")
    except Exception as e:
        return _fallback(f"{type(e).__name__}: {e}")

    try:
        plan = relay_plan(payload, str(payload.get("doc_type") or ""))
    except Exception:
        plan = []
    if plan:
        payload["relay_plan"] = plan

    if ocr_raw:
        payload["ocr_by_category"] = ocr_raw

    return payload