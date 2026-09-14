import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .text_upscale import crop_region

LINE_READ_MIN_LINES = 2

DEDUP_IOU = 0.35
DEDUP_CENTER_RATIO = 0.55
DEDUP_TEXT_SIM = 0.80

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

_LABEL_INDEX_CACHE: Dict[Tuple[int, str], Tuple["LabelIndex", List[str]]] = {}


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
) -> Tuple[LabelIndex, List[str]]:
    fields = (schema or {}).get("fields", {}) or {}
    cache_key = (id(schema), str(category or ""))
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
        index.add(str(name), plain, "name")
        _bank(plain)

        for key in LABEL_DICT_SOURCES:
            raw = d.get(key)
            if raw is None:
                continue
            for piece in _dict_phrases(raw):
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

    ranked = sorted(
        index.scores(label).items(), key=lambda kv: kv[1], reverse=True
    )
    passed = [
        (str(f), float(s)) for f, s in ranked
        if float(s) >= LABEL_COSINE_FLOOR
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

    if score < LABEL_COSINE_FLOOR:
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
                if overlap > 0:
                    cost = abs(gap_y) + abs(vcx - lcx) / max(1.0, unit * 6.0)
                    if cost < best_cost:
                        best = vi
                        best_cost = cost
                        best_mode = "below"

            gap_x = (vx0 - lx1) / unit
            vert = abs(((vy0 + vy1) * 0.5) - ((ly0 + ly1) * 0.5)) / unit
            if 0.0 <= gap_x <= PAIR_RIGHT_MAX_GAP and vert <= PAIR_VERT_TOLERANCE:
                cost = gap_x + vert
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
) -> Tuple[Dict[str, str], Dict[str, Tuple[str, float]]]:
    body = str(text or "").strip()
    if not schema:
        return {}, {}

    all_index, all_bank = build_label_index(schema, category="")
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

    cands: List[Tuple[float, int, str, str]] = []
    noisy = 0
    label_echo = 0
    unmatched = 0

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
            cands.append((float(score), idx, str(field), value))

    own: Dict[str, str] = {}
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
            if str(owner.get(field, "")) == str(category):
                own[field] = value
            else:
                donated[field] = (value, float(score))

    if log is not None and (own or donated or unmatched or noisy or label_echo):
        log.append(
            f"    🏷 [LABEL PAIR] '{category}' 라벨↔값 {len(pairs)}쌍 → "
            f"사전 코사인 배타 배정 {len(own) + len(donated)}건 "
            f"(이 카테고리 {len(own)}건 / 타 카테고리 기부 {len(donated)}건) "
            f"| 미매칭 {unmatched}건 | 라벨 에코 {label_echo}건 "
            f"| 잡음값 {noisy}건 폐기"
        )
        for k, v in list(own.items())[:8]:
            log.append(f"       · {k} = {v[:40]}")
        for k, payload in list(donated.items())[:8]:
            log.append(
                f"       ↗ {k} = {str(payload[0])[:40]} "
                f"(사전 코사인 {float(payload[1]):.2f})"
            )

    return own, donated


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


def dedup_rows(
    rows: Sequence[Tuple[str, float, Tuple[int, int, int, int]]],
    log: Optional[List[str]] = None,
    category: str = "",
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    items = [r for r in rows if str(r[0] or "").strip()]
    if len(items) <= 1:
        return list(items)

    items.sort(key=lambda r: (-float(r[1]), r[2][1], r[2][0]))

    kept: List[Tuple[str, float, Tuple[int, int, int, int]]] = []
    dropped_geo = 0
    dropped_txt = 0

    for text, score, box in items:
        hit = -1
        reason = ""

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
            if _text_similarity(text, ktext) >= 0.95:
                hit = i
                reason = "txt"
                break

        if hit < 0:
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

    if log is not None and (dropped_geo or dropped_txt):
        log.append(
            f"    🔁 [TILE DEDUP] '{category}' 타일 겹침에서 같은 줄이 "
            f"중복 인식되어 {dropped_geo + dropped_txt}건을 제거했습니다 "
            f"(좌표 겹침 {dropped_geo} / 문자 동일 {dropped_txt}). "
            f"{len(items)}줄 → {len(kept)}줄"
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
    tiles, reason = decide_tile_count(
        image, bbox, max_tiles=max_tiles,
        table_rows=table_rows, legible=legible, patches=patches,
        text_boxes=text_boxes,
    )

    if log is not None:
        log.append(
            f"    🧱 [TILE PLAN] '{category}' → {tiles}타일 (겹침 12%) "
            f"| 사유: {reason} | 표행 {table_rows} "
            f"| 판독가능 {legible}/{patches}"
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

    boxes = plan_overlap_tiles(bbox, tiles, overlap_ratio=0.12)
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

    try:
        from core.line_reader import read_lines
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

    ocr_fn = None
    if ocr is not None and getattr(ocr, "available", False):
        if hasattr(ocr, "recognize_lines"):
            def ocr_fn(crops):
                return ocr.recognize_lines(crops)

    merged, lines, windows = read_lines(
        cropped, _reader, boxes=local,
        span=span, stride=stride, overlap=overlap,
        ocr_fn=ocr_fn, log=log,
    )

    if log is not None:
        src = "VLM + 전용 인식기" if ocr_fn is not None else "VLM 단독"
        log.append(
            f"    🔁 [LINE READ] '{category}' VLM 호출 {calls['n']}회 "
            f"({src}) — 한 번에 긴 문장을 읽지 않고 행 단위로 끊었습니다."
        )

    return merged, [l.text for l in lines if l.text], len(windows)


TABLE_HEADER_MIN_COLS = 3
TABLE_COL_TOLERANCE = 0.55
TABLE_ROW_TOLERANCE = 0.70
TABLE_HEADER_MERGE_BANDS = 2
TABLE_HEADER_X_OVERLAP = 0.45


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
    header_span = 1

    for bi, _band in enumerate(ordered):
        for span in range(1, TABLE_HEADER_MERGE_BANDS + 1):
            group = ordered[bi: bi + span]
            if len(group) < span:
                break
            cols: List[Tuple[str, float, float]] = []
            for text, x0, x1 in _merge_header_cells(group):
                field = match_label_to_field(text, index)
                if not field:
                    continue
                cols.append((field, x0, x1))
            if len(cols) < TABLE_HEADER_MIN_COLS:
                continue
            cols.sort(key=lambda c: c[1])
            seen: set = set()
            uniq: List[Tuple[str, float, float]] = []
            for c in cols:
                if c[0] in seen:
                    continue
                seen.add(c[0])
                uniq.append(c)
            if len(uniq) >= TABLE_HEADER_MIN_COLS:
                header_band = int(group[-1])
                header_cols = uniq
                header_span = span
                break
        if header_band >= 0:
            break

    if header_band < 0:
        return []

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
    for band in sorted(b for b in bands.keys() if b > header_band):
        row: Dict[str, str] = {}
        for i in bands[band]:
            text, _s, b = items[i]
            t = str(text).strip()
            if not t or is_printed_label(t, bank):
                continue
            cx = (b[0] + b[2]) * 0.5

            best = ""
            best_dist = 1e9
            for field, x0, x1 in header_cols:
                if x0 <= cx <= x1:
                    best = field
                    best_dist = 0.0
                    break
                dist = min(abs(cx - x0), abs(cx - x1))
                if dist < best_dist:
                    best = field
                    best_dist = dist

            width = max(1.0, header_cols[-1][2] - header_cols[0][1])
            if best and best_dist <= width * TABLE_COL_TOLERANCE:
                row.setdefault(best, t)

        if len(row) >= 2:
            out.append(row)

    if log is not None:
        log.append(
            f"    🧾 [TABLE ROWS] '{category}' 데이터 행 {len(out)}건 구성 "
            f"(헤더 아래 밴드 기준)"
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
    out: List[ExtractedField] = []
    arrays = set(array_categories or ())
    tboxes = list(text_boxes or [])
    ocr_memo: Dict[Tuple[int, int, int, int], Tuple[str, float, str, int]] = {}
    donation_pool: Dict[str, Tuple[str, float]] = {}

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

        cleaned, nlp_meta = nlp_clean_ocr(text, nlp=nlp, log=log)
        if nlp_meta["dropped"] and log is not None:
            log.append(
                f"    🧬 NLP 게이트: {nlp_meta['lines_in']} → "
                f"{nlp_meta['lines_out']}줄 ({nlp_meta['dropped']}줄 제거)"
            )

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

        if refine_fn is not None:
            vlm_crop = crop
            if vlm_crop is None:
                vlm_crop = crop_region(image, plan.bbox)
            try:
                vlm_crop = fit_for_vlm(vlm_crop, log=log)
            except Exception:
                pass
            try:
                refined = refine_fn(
                    plan.category, cleaned, vlm_crop, plan.top_field
                ) or {}
            except TypeError:
                try:
                    refined = refine_fn(plan.category, cleaned, vlm_crop) or {}
                except Exception as e:
                    if log is not None:
                        log.append(f"    ⚠ 정제 추출 실패: {e}")
            except Exception as e:
                if log is not None:
                    log.append(f"    ⚠ 정제 추출 실패: {e}")

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

        donated: Dict[str, Tuple[str, float]] = {}
        if refine_fn is None and not values and not rows_out:
            paired, donated = promote_by_labels(
                cleaned, schema, plan.category, log=log, rows=crop_rows
            )
            if paired:
                values.update(paired)
            for dname, dpayload in donated.items():
                prev = donation_pool.get(dname)
                if prev is None or float(dpayload[1]) > float(prev[1]):
                    donation_pool[dname] = (
                        str(dpayload[0]), float(dpayload[1])
                    )

        is_twin = str(getattr(plan, "source", "")).startswith("twin-of:")

        if is_twin and plan.category in arrays:
            if log is not None:
                owner = str(plan.source).split(":", 1)[-1]
                log.append(
                    f"    ⛔ [TWIN SKIP] '{plan.category}' 는 '{owner}' 와 "
                    f"좌표가 같아 배열을 만들지 않습니다."
                )
        elif refine_fn is None and plan.category in arrays and not rows_out:
            table = build_table_rows(
                crop_rows, schema, plan.category, log=log
            )
            if table:
                rows_out.extend(table)
            elif log is not None:
                log.append(
                    f"    ⚪ [TABLE] '{plan.category}' 에서 표 헤더를 찾지 "
                    f"못해 배열을 만들지 않습니다. 라벨 없는 값 나열은 "
                    f"어느 컬럼인지 알 수 없습니다."
                )

        if refine_fn is None and plan.category not in arrays and not values:
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

    if log is not None and donation_pool:
        log.append(
            f"  🎁 [FIELD DONATION] 크롭 경계를 넘어 발견된 필드 "
            f"{len(donation_pool)}건을 공용 풀에 모았습니다. 소유 카테고리 "
            f"크롭이 그 값을 못 봤을 때 레코드 조립 단계에서 채웁니다."
        )
        for dname, dpayload in sorted(
            donation_pool.items(), key=lambda kv: -kv[1][1]
        )[:10]:
            log.append(
                f"     · {dname} = {str(dpayload[0])[:36]} "
                f"(사전 코사인 {float(dpayload[1]):.2f})"
            )

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

    for f in fields:
        if getattr(f, "rows", None):
            prev = record.get(f.category)
            merged = list(prev) if isinstance(prev, list) else []
            for r in f.rows:
                if r not in merged:
                    merged.append(r)
            record[f.category] = merged
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
            continue

        val = f.value
        if not val:
            continue
        if schema_fields and f.category not in schema_fields:
            prev = raw_text.get(f.category, "")
            raw_text[f.category] = (prev + "\n" + val).strip() if prev else val
            continue
        if f.category in record:
            prev = record[f.category]
            if isinstance(prev, list):
                prev.append(val)
            else:
                record[f.category] = [prev, val]
        else:
            record[f.category] = val

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