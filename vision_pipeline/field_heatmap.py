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
    ):
        self.category = category
        self.scores = np.asarray(scores, dtype=np.float32)
        self.top_field = top_field
        self.top_score = float(top_score)
        self.bank_size = int(bank_size)

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
            "rows": rows,
            "cols": cols,
        }

    def __repr__(self) -> str:
        return (
            f"<Heatmap {self.category} top={self.top_field}"
            f"({self.top_score:+.4f}) active={self.active_count()}/{self.scores.size}>"
        )


def _phrases_of(definition: dict) -> List[str]:
    out: List[str] = []
    sem = str(definition.get("semantic", "") or "").strip()
    if sem:
        out.append(sem)
    bias = definition.get("bias") or []
    if isinstance(bias, str):
        bias = [bias]
    for b in bias:
        b = str(b or "").strip()
        if b and b not in out:
            out.append(b)
    return out


def _prejudice_of(definition: dict) -> List[str]:
    out: List[str] = []
    prej = definition.get("prejudice") or []
    if isinstance(prej, str):
        prej = [prej]
    for p in prej:
        p = str(p or "").strip()
        if p and p not in out:
            out.append(p)
    return out


def build_field_heatmaps(
    grid: VisionPatchGrid,
    schema: dict,
    embed_fn: Callable[[List[str]], np.ndarray],
    cross_prejudice: bool = True,
    log: Optional[List[str]] = None,
) -> List[CategoryHeatmap]:
    fields = (schema or {}).get("fields", {}) or {}
    if not fields or grid.embeddings.size == 0:
        if log is not None:
            log.append("⚪ 히트맵 생성 대상이 없습니다.")
        return []

    cache = _AnchorCache(embed_fn)

    field_bias: Dict[str, List[str]] = {}
    field_prej: Dict[str, List[str]] = {}
    all_phrases: List[str] = list(CHROME_ANCHORS)

    for name, definition in fields.items():
        definition = definition if isinstance(definition, dict) else {}
        b = _phrases_of(definition)
        p = _prejudice_of(definition)
        field_bias[name] = b
        field_prej[name] = p
        all_phrases.extend(b)
        all_phrases.extend(p)

    vecs = cache.get_many(sorted(set(all_phrases)))
    chrome_vecs = [vecs[p] for p in CHROME_ANCHORS if p in vecs]

    if log is not None:
        log.append(
            f"  📖 앵커 임베딩 {len(vecs)}구 "
            f"(필드 {len(fields)}개 / 크롬 {len(chrome_vecs)}구)"
        )

    n = grid.num_patches
    raw: Dict[str, np.ndarray] = {}
    bank_sizes: Dict[str, int] = {}

    for name in fields.keys():
        bvecs = [vecs[p] for p in field_bias[name] if p in vecs]
        if not bvecs:
            continue

        pvecs = [vecs[p] for p in field_prej[name] if p in vecs]
        pvecs.extend(chrome_vecs)
        if cross_prejudice:
            for other, phrases in field_bias.items():
                if other == name:
                    continue
                pvecs.extend(vecs[p] for p in phrases if p in vecs)

        bmat = np.stack(bvecs).astype(np.float32)
        bias_map = np.max(grid.embeddings @ bmat.T, axis=1)

        if pvecs:
            pmat = np.stack(pvecs).astype(np.float32)
            prej_map = np.maximum(np.max(grid.embeddings @ pmat.T, axis=1), 0.0)
        else:
            prej_map = np.zeros((n,), dtype=np.float32)

        raw[name] = (bias_map - prej_map).astype(np.float32)
        bank_sizes[name] = len(bvecs)

    if not raw:
        return []

    names = list(raw.keys())
    stack = np.stack([raw[k] for k in names])

    mean = stack.mean(axis=0, keepdims=True)
    std = stack.std(axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    z = (stack - mean) / std

    penalties = np.asarray(
        [gumbel_expected_z(max(1, bank_sizes[k])) for k in names], dtype=np.float32
    ).reshape(-1, 1)
    net = z - penalties

    patch_mean = net.mean(axis=0, keepdims=True)
    net = net - patch_mean

    out: List[CategoryHeatmap] = []
    for i, name in enumerate(names):
        scores = net[i]
        peak = int(np.argmax(scores))
        out.append(
            CategoryHeatmap(
                category=name,
                scores=scores,
                top_field=name,
                top_score=float(scores[peak]),
                bank_size=bank_sizes[name],
            )
        )

    out.sort(key=lambda h: h.top_score, reverse=True)

    if log is not None:
        for h in out:
            log.append(
                f"  🔥 {h.category:<26} top={h.top_score:+.4f} "
                f"active={h.active_count()}/{h.scores.size} bank={h.bank_size}"
            )

    return out


def suppress_title_rows(
    heatmaps: Sequence[CategoryHeatmap],
    grid: VisionPatchGrid,
    top_ratio: float = 0.12,
    penalty: float = 0.5,
    log: Optional[List[str]] = None,
) -> List[CategoryHeatmap]:
    band = max(1, int(grid.rows * top_ratio))
    if band >= grid.rows:
        return list(heatmaps)

    cut = band * grid.cols
    for h in heatmaps:
        if h.scores.size < cut:
            continue
        h.scores[:cut] = h.scores[:cut] - abs(penalty)
        peak = int(np.argmax(h.scores))
        h.top_score = float(h.scores[peak])

    if log is not None:
        log.append(f"  🧹 제목 행 억제: 상단 {band}행에 −{abs(penalty):.2f} 패널티")
    return list(heatmaps)


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
        hot_before = int(np.sum(before > 0.0))

        mat = before.reshape(grid.rows, grid.cols)
        mu = float(mat.mean())
        row_mean = mat.mean(axis=1, keepdims=True)
        col_mean = mat.mean(axis=0, keepdims=True)
        residual = (mat - row_mean - col_mean + mu).reshape(-1)

        hot_after = int(np.sum(residual > 0.0))

        if hot_after == 0 or hot_after >= hot_before:
            reverted += 1
            continue

        h.scores = residual.astype(np.float32)
        peak = int(np.argmax(h.scores))
        h.top_score = float(h.scores[peak])
        applied += 1

    if log is not None:
        log.append(
            f"  🧭 공간 잔차화: gate={gate:.3f} 적용 {applied}개 / 되돌림 {reverted}개"
        )
    return list(heatmaps)