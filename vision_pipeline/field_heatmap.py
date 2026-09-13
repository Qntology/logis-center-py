import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .doc_type_nms import CHROME_ANCHORS, _AnchorCache, _unit, gumbel_expected_z
from .patch_grid import VisionPatchGrid


class CategoryHeatmap:
    def __init__(
        self,
        category: str,
        scores: np.ndarray,
        top_field: str = "",
        top_score: float = 0.0,
        bank_size: int = 0,
        affinity: Optional[np.ndarray] = None,
    ):
        self.category = category
        self.scores = np.asarray(scores, dtype=np.float32)
        self.top_field = top_field
        self.top_score = float(top_score)
        self.bank_size = int(bank_size)
        self.affinity = (
            np.asarray(affinity, dtype=np.float32)
            if affinity is not None else self.scores.copy()
        )
        self.absent = False
        self.absent_reason = ""
        self.territory = 0
        self.mean_margin = 0.0
        self.top_rival = ""

    def active_count(self, gate: float = 0.0) -> int:
        return int(np.sum(self.scores > gate))

    def peak_index(self) -> int:
        if self.scores.size == 0:
            return -1
        return int(np.argmax(self.scores))

    def to_dict(self, rows: int = 0, cols: int = 0) -> dict:
        return {
            "category": self.category,
            "top_field": self.top_field,
            "top_score": round(self.top_score, 4),
            "active": self.active_count(),
            "total": int(self.scores.size),
            "bank_size": self.bank_size,
            "absent": self.absent,
            "absent_reason": self.absent_reason,
            "territory": self.territory,
            "mean_margin": round(self.mean_margin, 4),
            "top_rival": self.top_rival,
            "rows": rows,
            "cols": cols,
        }

    def __repr__(self) -> str:
        return (
            f"<Heatmap {self.category} top={self.top_field}"
            f"({self.top_score:+.4f}) active={self.active_count()}/{self.scores.size}>"
        )


LABEL_WEIGHT = 1.30
SEMANTIC_WEIGHT = 1.10
BIAS_WEIGHT = 1.00
VALUE_WEIGHT = 0.70
PREJUDICE_WEIGHT = 1.00
CROSS_PREJUDICE_WEIGHT = 0.85
ANY_LANG_KEY = "*"

DEFAULT_CATEGORY = "misc"


def _category_of(name: str, definition: dict) -> str:
    cat = str((definition or {}).get("category") or "").strip()
    return cat if cat else DEFAULT_CATEGORY


def _identity_drop_fields(schema: dict, doc_code: str) -> Dict[str, str]:
    fields = (schema or {}).get("fields", {}) or {}
    code = str(doc_code or "").strip().lower()
    drop: Dict[str, str] = {}

    if "doc_type" in fields:
        drop["doc_type"] = (
            "STEP 1 비전 판정이 이미 확정한 축입니다. 제목 행에 반응해 "
            "카테고리 봉우리를 제목 쪽으로 끌어당깁니다."
        )

    if not code:
        return drop

    for name, definition in fields.items():
        if not name.startswith("reference_"):
            continue
        tail = name[len("reference_"):].replace("_", "")
        if tail and (tail == code or code.endswith(tail) or tail.endswith(code)):
            drop[name] = (
                f"'{doc_code}' 문서가 자기 자신을 가리키는 축입니다. "
                f"doc_number 와 라벨이 겹쳐 정체성 축을 잠식합니다."
            )
    return drop


def _as_list(raw) -> List[str]:
    from text_pipeline.field_bank import flatten_text_values
    return flatten_text_values(raw)


def _pick_lang(raw, lang_code: str = "") -> List[str]:
    from text_pipeline.field_bank import pick_lang_values
    return pick_lang_values(raw, lang_code)


def _phrases_of(definition: dict, lang_code: str = "") -> List[Tuple[str, float]]:
    from text_pipeline.field_bank import is_junk_phrase

    out: List[Tuple[str, float]] = []
    seen = set()

    def _push(text, weight: float):
        t = str(text or "").strip()
        if not t:
            return
        if is_junk_phrase(t):
            return
        low = t.lower()
        if low in seen:
            return
        seen.add(low)
        out.append((t, float(weight)))

    for t in _pick_lang(definition.get("semantic"), lang_code):
        _push(t, SEMANTIC_WEIGHT)
    for t in _pick_lang(definition.get("label"), lang_code):
        _push(t, LABEL_WEIGHT)
    for t in _pick_lang(definition.get("bias"), lang_code):
        _push(t, BIAS_WEIGHT)
    for t in _pick_lang(definition.get("value"), lang_code):
        _push(t, VALUE_WEIGHT)
    return out


def _prejudice_of(definition: dict, lang_code: str = "") -> List[Tuple[str, float]]:
    from text_pipeline.field_bank import is_junk_phrase

    out: List[Tuple[str, float]] = []
    seen = set()
    for t in _pick_lang(definition.get("prejudice"), lang_code):
        t = str(t or "").strip()
        if not t:
            continue
        if is_junk_phrase(t):
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append((t, PREJUDICE_WEIGHT))
    return out


HEATMAP_CHROME_ANCHORS: Tuple[str, ...] = (
    "a column header caption printed alone with no value beneath it",
    "a repeated table ruling line separating rows with no content",
    "a legal boilerplate declaration sentence at the page bottom",
    "a signature line with a horizontal rule and no filled value",
)


def build_field_heatmaps(
    grid: VisionPatchGrid,
    schema: dict,
    embed_fn: Callable[[List[str]], np.ndarray],
    cross_prejudice: bool = False,
    lang_code: str = "",
    log: Optional[List[str]] = None,
    doc_code: str = "",
) -> Tuple[List[CategoryHeatmap], np.ndarray]:
    fields = (schema or {}).get("fields", {}) or {}
    if not fields or grid.embeddings.size == 0:
        if log is not None:
            log.append("⚪ 히트맵 생성 대상이 없습니다.")
        return [], np.zeros((0,), dtype=np.float32)

    drop = _identity_drop_fields(schema, doc_code)
    if log is not None:
        for name, why in drop.items():
            log.append(f"  🧹 [ANCHOR DROP] '{name}' — {why}")

    cache = _AnchorCache(embed_fn, expect_dim=int(grid.dim))
    negatives: Tuple[str, ...] = tuple(CHROME_ANCHORS) + HEATMAP_CHROME_ANCHORS

    cat_fields: Dict[str, List[str]] = {}
    field_bias: Dict[str, List[Tuple[str, float]]] = {}
    field_prej: Dict[str, List[Tuple[str, float]]] = {}
    all_phrases: List[str] = list(negatives)

    for name, definition in fields.items():
        if name in drop:
            continue
        definition = definition if isinstance(definition, dict) else {}
        b = _phrases_of(definition, lang_code)
        p = _prejudice_of(definition, lang_code)
        if not b:
            continue
        field_bias[name] = b
        field_prej[name] = p
        cat_fields.setdefault(_category_of(name, definition), []).append(name)
        all_phrases.extend(t for t, _w in b)
        all_phrases.extend(t for t, _w in p)

    if not cat_fields:
        if log is not None:
            log.append("  ❌ 카테고리로 묶을 필드가 없습니다.")
        return [], np.zeros((0,), dtype=np.float32)

    vecs = cache.get_many(sorted(set(all_phrases)))
    if not vecs:
        if log is not None:
            msg = cache.report() or "텍스트 앵커 임베딩을 얻지 못했습니다."
            log.append(f"  ❌ 필드 히트맵 중단 — {msg}")
        return [], np.zeros((0,), dtype=np.float32)

    chrome_vecs = [vecs[p] for p in negatives if p in vecs]

    if log is not None:
        scope = lang_code if lang_code else "전체 언어"
        total_phr = sum(
            len([1 for t, _w in field_bias[f] if t in vecs])
            for fs in cat_fields.values() for f in fs
        )
        log.append(
            f"  📐 [VISION COLUMN BANK] 카테고리 {len(cat_fields)}개 "
            f"| 필드 구 {total_phr}개 | 편견 구 {len(chrome_vecs)}개 "
            f"(카테고리 단위 축약) | 패치 {grid.num_patches}개 | 스코프 {scope}"
        )
        log.append(
            "  🧹 [PREJUDICE SCOPE v3] 교차 카테고리 편견 폐기"
            "(열 센터링이 이미 수행)"
        )
        for cat in sorted(cat_fields.keys()):
            log.append(
                f"    📖 [CAT '{cat}'] 필드 {len(cat_fields[cat])}개: "
                + ", ".join(sorted(cat_fields[cat])[:8])
            )

    n = grid.num_patches
    pm = np.asarray(grid.embeddings, dtype=np.float32)
    pm = pm / np.maximum(np.linalg.norm(pm, axis=1, keepdims=True), 1e-8)

    raw: Dict[str, np.ndarray] = {}
    top_field_map: Dict[str, str] = {}
    bank_sizes: Dict[str, int] = {}

    for cat, names_in in cat_fields.items():
        cols: List[np.ndarray] = []
        owner: List[str] = []
        for fname in names_in:
            for t, w in field_bias[fname]:
                v = vecs.get(t)
                if v is None:
                    continue
                cols.append(np.asarray(v, dtype=np.float32).reshape(-1) * float(w))
                owner.append(fname)
        if not cols:
            continue

        bmat = np.stack(cols).astype(np.float32)
        sims = pm @ bmat.T
        best_idx = np.argmax(sims, axis=1)
        bias_map = sims[np.arange(sims.shape[0]), best_idx]

        ppairs: List[np.ndarray] = list(chrome_vecs)
        for fname in names_in:
            for t, _w in field_prej.get(fname, []):
                v = vecs.get(t)
                if v is not None:
                    ppairs.append(np.asarray(v, dtype=np.float32).reshape(-1))

        if ppairs:
            pmat = np.stack(ppairs).astype(np.float32) * PREJUDICE_WEIGHT
            prej_map = np.maximum(np.max(pm @ pmat.T, axis=1), 0.0)
        else:
            prej_map = np.zeros((n,), dtype=np.float32)

        raw[cat] = (bias_map - prej_map).astype(np.float32)
        bank_sizes[cat] = int(bmat.shape[0])
        peak = int(np.argmax(raw[cat]))
        top_field_map[cat] = owner[int(best_idx[peak])]

    if not raw:
        if log is not None:
            log.append("  ❌ 모든 카테고리의 앵커 확보에 실패했습니다.")
        return [], np.zeros((0,), dtype=np.float32)

    cats = list(raw.keys())
    stack = np.stack([raw[k] for k in cats])

    patch_mu = stack.mean(axis=0, keepdims=True)
    patch_sd = np.maximum(stack.std(axis=0, keepdims=True), 1e-6)
    affinity = (stack - patch_mu) / patch_sd

    field_penalty = np.asarray(
        [gumbel_expected_z(max(1, len(cat_fields[k]))) for k in cats],
        dtype=np.float32,
    ).reshape(-1, 1)
    affinity = affinity - field_penalty

    if log is not None:
        brief = " | ".join(
            f"{k}({len(cat_fields[k])}필드 −{float(field_penalty[i, 0]):.3f})"
            for i, k in enumerate(cats)
        )
        log.append(f"  ⚖️ [CATEGORY-NEUTRAL] max-pool 필드 수 편향 보정: {brief}")

    cat_mu = affinity.mean(axis=1, keepdims=True)
    cat_sd = np.maximum(affinity.std(axis=1, keepdims=True), 1e-6)
    calibrated = (affinity - cat_mu) / cat_sd

    chrome_ref = np.zeros((n,), dtype=np.float32)
    if chrome_vecs:
        cmat = np.stack(chrome_vecs).astype(np.float32)
        chrome_raw = np.max(pm @ cmat.T, axis=1)
        chrome_aff = (chrome_raw - patch_mu.reshape(-1)) / patch_sd.reshape(-1)
        c_mu = float(chrome_aff.mean())
        c_sd = max(float(chrome_aff.std()), 1e-6)
        chrome_ref = ((chrome_aff - c_mu) / c_sd).astype(np.float32)

    out: List[CategoryHeatmap] = []
    for i, cat in enumerate(cats):
        scores = calibrated[i].astype(np.float32)
        peak = int(np.argmax(scores))
        out.append(
            CategoryHeatmap(
                category=cat,
                scores=scores,
                top_field=top_field_map.get(cat, cat),
                top_score=float(scores[peak]),
                bank_size=bank_sizes.get(cat, 0),
                affinity=affinity[i].astype(np.float32),
            )
        )

    out.sort(key=lambda h: h.top_score, reverse=True)

    if log is not None:
        log.append(
            f"  📏 정규화: 패치축 z → 카테고리 필드수 보정 → 카테고리축 재보정 "
            f"(크롬 기준선 평균 {float(chrome_ref.mean()):+.4f})"
        )
        for h in out:
            log.append(
                f"  🔥 [HEATMAP] {h.category:<16} 활성 패치 "
                f"{h.active_count()}/{h.scores.size} | Top: {h.top_field}"
                f"({h.top_score:+.4f}) | 구 {h.bank_size}개"
            )

    return out, chrome_ref


def suppress_title_rows(
    heatmaps: Sequence[CategoryHeatmap],
    grid: VisionPatchGrid,
    top_ratio: float = 0.12,
    penalty: float = 0.5,
    schema: Optional[dict] = None,
    log: Optional[List[str]] = None,
) -> List[CategoryHeatmap]:
    band = max(1, int(grid.rows * top_ratio))
    if band >= grid.rows:
        return list(heatmaps)

    fields = (schema or {}).get("fields", {}) or {}
    exempt = {
        name for name, d in fields.items()
        if isinstance(d, dict) and bool(d.get("top_region"))
    }

    cut = band * grid.cols
    hit = 0
    for h in heatmaps:
        if h.scores.size < cut:
            continue
        if h.category in exempt:
            continue
        head = h.scores[:cut]
        h.scores[:cut] = np.where(
            np.isfinite(head), head - abs(penalty), head
        )
        finite = np.isfinite(h.scores)
        h.top_score = float(h.scores[finite].max()) if finite.any() else 0.0
        hit += 1

    if log is not None:
        skip = f" | 면제 {len(exempt)}개({', '.join(sorted(exempt))})" if exempt else ""
        log.append(
            f"  🧹 제목 행 억제: 상단 {band}행에 −{abs(penalty):.2f} "
            f"패널티 적용 {hit}개{skip}"
        )
    return list(heatmaps)


def apply_arena(
    heatmaps: Sequence[CategoryHeatmap],
    arena,
    n: int,
    log: Optional[List[str]] = None,
) -> List[CategoryHeatmap]:
    out: List[CategoryHeatmap] = []

    for h in heatmaps:
        t = arena.territory_of(h.category)
        if t is None:
            h.absent = True
            h.absent_reason = "경쟁 참가 실패"
            continue

        h.territory = t.size
        h.mean_margin = t.mean_margin
        h.top_rival = t.top_rival
        h.absent = t.absent
        h.absent_reason = t.absent_reason

        if t.absent:
            continue

        owned = t.mask(min(n, int(h.scores.size)))
        masked = np.where(owned, h.scores[: owned.size], -np.inf)
        if masked.size < h.scores.size:
            pad = np.full((h.scores.size - masked.size,), -np.inf, dtype=np.float32)
            masked = np.concatenate([masked, pad])

        h.scores = masked.astype(np.float32)
        finite = np.isfinite(h.scores)
        h.top_score = float(h.scores[finite].max()) if finite.any() else 0.0
        out.append(h)

    out.sort(key=lambda x: x.top_score, reverse=True)

    if log is not None:
        log.append(
            f"  🧱 영토 마스킹 적용 — 잔존 히트맵 {len(out)}개 "
            f"(부재 필드의 점수는 −inf 로 차단)"
        )
    return out


def spatial_residual(
    heatmaps: Sequence[CategoryHeatmap],
    grid: VisionPatchGrid,
    log: Optional[List[str]] = None,
) -> List[CategoryHeatmap]:
    if not heatmaps or grid.rows < 2 or grid.cols < 2:
        return list(heatmaps)

    ratios = [h.active_count() / max(1, h.scores.size) for h in heatmaps]
    if len(ratios) < 3:
        return list(heatmaps)

    s = sorted(ratios)
    med = s[len(s) // 2]
    dev = sorted(abs(r - med) for r in ratios)
    mad = dev[len(dev) // 2]
    if mad <= 1e-6:
        if log is not None:
            log.append("  ⏭ 확산도가 고르므로 공간 잔차화를 건너뜁니다.")
        return list(heatmaps)

    gate = med + mad
    applied = 0
    reverted = 0

    for h in heatmaps:
        ratio = h.active_count() / max(1, h.scores.size)
        if ratio <= gate:
            continue

        before = h.scores.copy()
        blocked = ~np.isfinite(before)
        hot_before = int(np.sum(before > 0.0))

        work = np.where(blocked, 0.0, before)
        mat = work.reshape(grid.rows, grid.cols)
        mu = float(mat.mean())
        row_mean = mat.mean(axis=1, keepdims=True)
        col_mean = mat.mean(axis=0, keepdims=True)
        residual = (mat - row_mean - col_mean + mu).reshape(-1)
        residual = np.where(blocked, -np.inf, residual)

        hot_after = int(np.sum(residual > 0.0))

        if hot_after == 0 or hot_after >= hot_before:
            reverted += 1
            continue

        h.scores = residual.astype(np.float32)
        finite = np.isfinite(h.scores)
        h.top_score = float(h.scores[finite].max()) if finite.any() else 0.0
        applied += 1

    if log is not None:
        log.append(
            f"  🧭 공간 잔차화: gate={gate:.3f} 적용 {applied}개 / 되돌림 {reverted}개"
        )
    return list(heatmaps)