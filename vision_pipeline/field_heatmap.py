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
    cross_prejudice: bool = True,
    lang_code: str = "",
    log: Optional[List[str]] = None,
) -> Tuple[List[CategoryHeatmap], np.ndarray]:
    fields = (schema or {}).get("fields", {}) or {}
    if not fields or grid.embeddings.size == 0:
        if log is not None:
            log.append("⚪ 히트맵 생성 대상이 없습니다.")
        return [], np.zeros((0,), dtype=np.float32)

    cache = _AnchorCache(embed_fn, expect_dim=int(grid.dim))

    negatives: Tuple[str, ...] = tuple(CHROME_ANCHORS) + HEATMAP_CHROME_ANCHORS

    field_bias: Dict[str, List[Tuple[str, float]]] = {}
    field_prej: Dict[str, List[Tuple[str, float]]] = {}
    all_phrases: List[str] = list(negatives)

    for name, definition in fields.items():
        definition = definition if isinstance(definition, dict) else {}
        b = _phrases_of(definition, lang_code)
        p = _prejudice_of(definition, lang_code)
        field_bias[name] = b
        field_prej[name] = p
        all_phrases.extend(t for t, _w in b)
        all_phrases.extend(t for t, _w in p)

    vecs = cache.get_many(sorted(set(all_phrases)))

    if not vecs:
        if log is not None:
            msg = cache.report() or "텍스트 앵커 임베딩을 얻지 못했습니다."
            log.append(f"  ❌ 필드 히트맵 중단 — {msg}")
        return [], np.zeros((0,), dtype=np.float32)

    chrome_vecs = [vecs[p] for p in negatives if p in vecs]

    if log is not None:
        scope = lang_code if lang_code else "전체 언어"
        log.append(
            f"  📖 앵커 임베딩 {len(vecs)}구 × {grid.dim}차원 "
            f"(필드 {len(fields)}개 / 크롬 {len(chrome_vecs)}구 / 사전 스코프 {scope})"
        )

    n = grid.num_patches
    raw: Dict[str, np.ndarray] = {}
    bank_sizes: Dict[str, int] = {}

    for name in fields.keys():
        bpairs = [(vecs[t], w) for t, w in field_bias[name] if t in vecs]
        if not bpairs:
            if log is not None:
                log.append(f"  ⚪ '{name}' 앵커 확보 실패 → 히트맵 제외")
            continue

        ppairs = [(vecs[t], w) for t, w in field_prej[name] if t in vecs]
        ppairs.extend((v, PREJUDICE_WEIGHT) for v in chrome_vecs)
        if cross_prejudice:
            for other, pairs in field_bias.items():
                if other == name:
                    continue
                for t, w in pairs:
                    if t in vecs:
                        ppairs.append((vecs[t], min(w, CROSS_PREJUDICE_WEIGHT)))

        bmat = np.stack([v for v, _w in bpairs]).astype(np.float32)
        bw = np.asarray([w for _v, w in bpairs], dtype=np.float32)
        bias_map = np.max((grid.embeddings @ bmat.T) * bw[None, :], axis=1)

        if ppairs:
            pmat = np.stack([v for v, _w in ppairs]).astype(np.float32)
            pw = np.asarray([w for _v, w in ppairs], dtype=np.float32)
            prej_map = np.maximum(
                np.max((grid.embeddings @ pmat.T) * pw[None, :], axis=1), 0.0
            )
        else:
            prej_map = np.zeros((n,), dtype=np.float32)

        raw[name] = (bias_map - prej_map).astype(np.float32)
        bank_sizes[name] = len(bpairs)

    if not raw:
        if log is not None:
            log.append(
                "  ❌ 모든 필드의 앵커 확보에 실패했습니다. "
                "스키마의 semantic / label / value 가 비어 있는지 확인하세요."
            )
        return [], np.zeros((0,), dtype=np.float32)

    names = list(raw.keys())
    stack = np.stack([raw[k] for k in names])

    penalties = np.asarray(
        [gumbel_expected_z(max(1, bank_sizes[k])) for k in names], dtype=np.float32
    ).reshape(-1, 1)

    patch_mu = stack.mean(axis=0, keepdims=True)
    patch_sd = np.maximum(stack.std(axis=0, keepdims=True), 1e-6)
    affinity = ((stack - patch_mu) / patch_sd) - penalties * 0.25

    field_mu = affinity.mean(axis=1, keepdims=True)
    field_sd = np.maximum(affinity.std(axis=1, keepdims=True), 1e-6)
    calibrated = (affinity - field_mu) / field_sd

    chrome_ref = np.zeros((n,), dtype=np.float32)
    if chrome_vecs:
        cmat = np.stack(chrome_vecs).astype(np.float32)
        chrome_raw = np.max(grid.embeddings @ cmat.T, axis=1)
        chrome_affinity = (
            (chrome_raw - patch_mu.reshape(-1)) / patch_sd.reshape(-1)
        ) - gumbel_expected_z(max(1, len(chrome_vecs))) * 0.25
        c_mu = float(chrome_affinity.mean())
        c_sd = max(float(chrome_affinity.std()), 1e-6)
        chrome_ref = ((chrome_affinity - c_mu) / c_sd).astype(np.float32)

    out: List[CategoryHeatmap] = []
    for i, name in enumerate(names):
        scores = calibrated[i].astype(np.float32)
        peak = int(np.argmax(scores))
        out.append(
            CategoryHeatmap(
                category=name,
                scores=scores,
                top_field=name,
                top_score=float(scores[peak]),
                bank_size=bank_sizes[name],
                affinity=affinity[i].astype(np.float32),
            )
        )

    out.sort(key=lambda h: h.top_score, reverse=True)

    if log is not None:
        log.append(
            f"  📏 정규화: 패치축 z → 앵커수 패널티 → 필드축 재보정 "
            f"(크롬 기준선 평균 {float(chrome_ref.mean()):+.4f})"
        )
        for h in out:
            log.append(
                f"  🔥 {h.category:<26} top={h.top_score:+.4f} "
                f"active={h.active_count()}/{h.scores.size} bank={h.bank_size}"
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