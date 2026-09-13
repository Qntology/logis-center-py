from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .field_heatmap import CategoryHeatmap
from .patch_grid import VisionPatchGrid


class Component:
    def __init__(
        self,
        indices: List[int],
        r_min: int,
        r_max: int,
        c_min: int,
        c_max: int,
        peak: float,
        total: float,
    ):
        self.indices = indices
        self.r_min = int(r_min)
        self.r_max = int(r_max)
        self.c_min = int(c_min)
        self.c_max = int(c_max)
        self.peak = float(peak)
        self.total = float(total)

    @property
    def area(self) -> int:
        return (self.r_max - self.r_min + 1) * (self.c_max - self.c_min + 1)

    def grid_box(self) -> Tuple[int, int, int, int]:
        return (self.r_min, self.r_max, self.c_min, self.c_max)

    def __repr__(self) -> str:
        return (
            f"<Component r{self.r_min}~{self.r_max} c{self.c_min}~{self.c_max} "
            f"peak={self.peak:+.4f} n={len(self.indices)}>"
        )


class CropPlan:
    def __init__(
        self,
        category: str,
        bbox: Tuple[int, int, int, int],
        score: float,
        margin: float,
        patch_count: int,
        grid_box: Tuple[int, int, int, int],
        rival: str = "",
        source: str = "nms",
    ):
        self.category = category
        self.bbox = tuple(int(v) for v in bbox)
        self.score = float(score)
        self.margin = float(margin)
        self.patch_count = int(patch_count)
        self.grid_box = tuple(int(v) for v in grid_box)
        self.rival = rival
        self.source = source
        self.absorbed: List[dict] = []
        self.top_field: str = ""
        self.table_rows: int = 0
        self.legible: int = 0
        self.patches_total: int = 0
        self.coverage: float = 0.0

    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "bbox": list(self.bbox),
            "grid_box": list(self.grid_box),
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "patch_count": self.patch_count,
            "rival": self.rival,
            "source": self.source,
            "top_field": self.top_field,
            "table_rows": self.table_rows,
            "legible": self.legible,
            "patches_total": self.patches_total,
            "coverage": round(self.coverage, 4),
            "absorbed": list(self.absorbed),
        }

    def __repr__(self) -> str:
        return (
            f"<CropPlan {self.category} px{self.bbox} "
            f"score={self.score:+.4f} margin={self.margin:+.4f}>"
        )


def positive_stats(scores: np.ndarray) -> Tuple[float, float, int]:
    finite = scores[np.isfinite(scores)]
    pos = finite[finite > 0.0]
    if pos.size < 2:
        return 0.0, 0.0, int(pos.size)
    return float(pos.mean()), float(pos.std()), int(pos.size)


def core_threshold(scores: np.ndarray) -> float:
    mean, std, cnt = positive_stats(scores)
    if cnt < 4:
        return 0.0
    return mean + std


def extract_components(
    scores: np.ndarray,
    rows: int,
    cols: int,
    gate: float,
    blocked_cols: Optional[set] = None,
    blocked_rows: Optional[set] = None,
) -> List[Component]:
    n = rows * cols
    if scores.size < n or n == 0:
        return []

    bc = set(blocked_cols or ())
    br = set(blocked_rows or ())

    visited = np.zeros((n,), dtype=bool)
    out: List[Component] = []

    for start in range(n):
        if visited[start]:
            continue
        if scores[start] <= gate:
            visited[start] = True
            continue

        stack = [start]
        visited[start] = True
        indices: List[int] = []
        r_min, r_max = rows, 0
        c_min, c_max = cols, 0
        peak = -np.inf
        total = 0.0

        while stack:
            cur = stack.pop()
            r, c = divmod(cur, cols)
            indices.append(cur)
            r_min = min(r_min, r)
            r_max = max(r_max, r)
            c_min = min(c_min, c)
            c_max = max(c_max, c)
            peak = max(peak, float(scores[cur]))
            total += float(scores[cur])

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                    continue
                if dc != 0 and nc in bc:
                    continue
                if dr != 0 and nr in br:
                    continue
                idx = nr * cols + nc
                if visited[idx] or scores[idx] <= gate:
                    continue
                visited[idx] = True
                stack.append(idx)

        out.append(Component(indices, r_min, r_max, c_min, c_max, peak, total))

    out.sort(key=lambda c: c.peak, reverse=True)
    return out


def split_oversized(
    scores: np.ndarray,
    comp: Component,
    rows: int,
    cols: int,
    gate: float,
    area_cap: int,
    blocked_cols: Optional[set] = None,
    blocked_rows: Optional[set] = None,
    max_rounds: int = 4,
) -> List[Component]:
    n = rows * cols
    if not comp.indices:
        return [comp]

    local = np.full((n,), -np.inf, dtype=np.float32)
    local[comp.indices] = scores[comp.indices]
    vals = np.asarray([float(scores[i]) for i in comp.indices], dtype=np.float32)

    best: List[Component] = [comp]
    cur_gate = float(gate)

    for k in range(1, int(max_rounds) + 1):
        q = float(np.quantile(vals, min(0.95, 0.35 + 0.15 * k)))
        cur_gate = max(cur_gate, q)
        subs = extract_components(
            local, rows, cols, cur_gate, blocked_cols, blocked_rows
        )
        if not subs:
            break
        best = subs
        if len(subs) > 1 and max(s.area for s in subs) <= area_cap:
            return subs

    return best


def build_content_mask(
    heatmaps: Sequence[CategoryHeatmap],
    n: int,
) -> Tuple[np.ndarray, float]:
    mask = np.full((n,), -np.inf, dtype=np.float32)
    for h in heatmaps:
        m = min(n, h.scores.size)
        seg = h.scores[:m]
        mask[:m] = np.maximum(mask[:m], np.where(np.isfinite(seg), seg, -np.inf))
    mean, _std, cnt = positive_stats(mask)
    gate = 0.0 if cnt < 4 else mean
    return mask, gate


def expand_row_band(
    comp: Component,
    content: np.ndarray,
    gate: float,
    cols: int,
    blocked_cols: Optional[set] = None,
    max_width: int = 0,
) -> Tuple[int, int, int, int]:
    bc = set(blocked_cols or ())
    c_min, c_max = comp.c_min, comp.c_max
    cap = int(max_width) if max_width and max_width > 0 else int(cols)

    def band_has(c: int) -> bool:
        for r in range(comp.r_min, comp.r_max + 1):
            idx = r * cols + c
            if idx < content.size and content[idx] > gate:
                return True
        return False

    while (
        c_min > 0
        and (c_min - 1) not in bc
        and (c_max - c_min + 1) < cap
        and band_has(c_min - 1)
    ):
        c_min -= 1

    while (
        c_max + 1 < cols
        and (c_max + 1) not in bc
        and (c_max - c_min + 1) < cap
        and band_has(c_max + 1)
    ):
        c_max += 1

    return (comp.r_min, comp.r_max, c_min, c_max)


def compute_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    area_a = float(max(0, ax1 - ax0) * max(0, ay1 - ay0))
    area_b = float(max(0, bx1 - bx0) * max(0, by1 - by0))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def ensure_min_size(
    bbox: Tuple[int, int, int, int],
    orig_w: int,
    orig_h: int,
    min_w: int,
    min_h: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if x1 - x0 < min_w:
        need = min_w - (x1 - x0)
        half = need // 2
        x0 = max(0, x0 - half)
        x1 = min(orig_w, x1 + (need - half))
        if x1 - x0 < min_w:
            x0 = max(0, x1 - min_w)
    if y1 - y0 < min_h:
        need = min_h - (y1 - y0)
        half = need // 2
        y0 = max(0, y0 - half)
        y1 = min(orig_h, y1 + (need - half))
        if y1 - y0 < min_h:
            y0 = max(0, y1 - min_h)
    return (x0, y0, min(x1, orig_w), min(y1, orig_h))


def presence_gate(
    heatmaps: Sequence[CategoryHeatmap],
    n: int,
    log: Optional[List[str]] = None,
) -> set:
    out = set()
    for h in heatmaps:
        if getattr(h, "absent", False):
            if log is not None:
                log.append(
                    f"  ⚪ PRESENCE GATE '{h.category}' 부재 — "
                    f"{getattr(h, 'absent_reason', '') or '경쟁 영토 없음'}"
                )
            continue
        if int(getattr(h, "territory", 0)) <= 0 and not np.isfinite(h.scores).any():
            if log is not None:
                log.append(
                    f"  ⚪ PRESENCE GATE '{h.category}' 유효 패치 없음 → 제외"
                )
            continue
        out.add(h.category)
    return out


CROP_PAD_Y_RATIO = 0.055
CROP_PAD_X_RATIO = 0.14


def _pad_grid_box(
    gb: Tuple[int, int, int, int],
    rows: int,
    cols: int,
    grid: Optional[VisionPatchGrid] = None,
) -> Tuple[int, int, int, int]:
    r0, r1, c0, c1 = (int(v) for v in gb)

    pr, pc = 1, 2
    if grid is not None:
        ch = max(1.0, float(grid.cell_height()))
        cw = max(1.0, float(grid.cell_width()))
        pr = max(1, int(round(grid.orig_height * CROP_PAD_Y_RATIO / ch)))
        pc = max(2, int(round(grid.orig_width * CROP_PAD_X_RATIO / cw)))

    r0 = max(0, r0 - pr)
    r1 = min(rows - 1, r1 + pr)
    c0 = max(0, c0 - pc)
    c1 = min(cols - 1, c1 + pc)
    return (r0, r1, c0, c1)


def _box_patches(
    gb: Tuple[int, int, int, int],
    cols: int,
) -> set:
    r0, r1, c0, c1 = gb
    out = set()
    for r in range(int(r0), int(r1) + 1):
        base = r * cols
        for c in range(int(c0), int(c1) + 1):
            out.add(base + c)
    return out


def _identity_band(
    rows: int,
    content_rows: Sequence[int],
    title_rows: int,
) -> Tuple[int, int]:
    span = max(3, int(round(rows * 0.28)))
    start = -1
    for r in content_rows:
        if r >= int(title_rows):
            start = int(r)
            break
    if start < 0:
        return -1, -1
    return start, min(rows - 1, start + span - 1)


def plan_crops(
    heatmaps: Sequence[CategoryHeatmap],
    grid: VisionPatchGrid,
    iou_threshold: float = 0.80,
    margin_threshold: float = 0.28,
    col_gutters: Optional[set] = None,
    row_gutters: Optional[set] = None,
    max_crop_cols: int = 0,
    log: Optional[List[str]] = None,
    legibility=None,
    table_categories: Optional[set] = None,
    identity_category: str = "",
    title_rows: int = 2,
) -> List[CropPlan]:
    if not heatmaps or grid.num_patches == 0:
        return []

    rows, cols = grid.rows, grid.cols
    n = rows * cols

    bc = set(col_gutters or ())
    br = set(row_gutters or ())
    tables = set(table_categories or ())

    content, content_gate = build_content_mask(heatmaps, n)
    content_ok = content > content_gate

    if legibility is not None and int(getattr(legibility, "size", 0)) == n:
        inked = ~np.asarray(legibility.blank, dtype=bool)
        merged_ok = content_ok | inked
        keep = int(merged_ok.sum())
        if keep >= max(4, int(n * 0.05)):
            content_ok = merged_ok
        if log is not None:
            log.append(legibility.report())
            log.append(
                f"  🔎 [CONTENT UNION] 히트맵 활성 {int((content > content_gate).sum())} "
                f"∪ 잉크 패치 {int(inked.sum())} → 내용 {int(content_ok.sum())}/{n} "
                f"(판독불가 패치도 글자가 있으므로 유지합니다)"
            )

    if log is not None:
        log.append(
            f"  🗺️ [CONTENT MASK] 내용 패치 {int(content_ok.sum())}/{n} "
            f"| gate {content_gate:+.4f} (라벨↔값 행 밴드 확장 기준)"
        )
        log.append(
            f"  📏 거터 검출 — 세로 {len(bc)}개 {sorted(bc)} / "
            f"가로 {len(br)}개 {sorted(br)}"
        )

    content_rows = sorted({
        i // cols for i in range(n) if bool(content_ok[i])
    })

    tb_start, tb_end, tb_span = (-1, -1, 0)
    if legibility is not None:
        from .patch_grid import table_row_band
        min_span = max(3, int(round(rows * 0.15)))
        tb_start, tb_end, tb_span = table_row_band(
            legibility, min_span=min_span
        )
        if log is not None:
            if tb_span > 0:
                log.append(
                    f"  🧾 [TABLE BAND] 표행 밴드 r{tb_start}~r{tb_end} "
                    f"({tb_span}행 ≥ 최소 {min_span}행) 검출"
                )
            else:
                log.append(
                    f"  ⏭ [TABLE BAND] 연속 밀집 행이 최소 {min_span}행에 "
                    f"못 미쳐 표 통합을 건너뜁니다."
                )

    present = presence_gate(heatmaps, n, log=log)
    area_cap = max(4, n // max(1, len(present)))

    if log is not None:
        log.append(
            f"  📊 [PLAN_CROPS INPUT] 히트맵 {len(heatmaps)}개 "
            f"(존재 판정 통과 {len(present)}개) | 격자 {rows}x{cols}={n} "
            f"| area_cap={area_cap} | content 활성 {int(content_ok.sum())}/{n}"
        )

    per_cat: List[Tuple[str, List[Tuple[int, int, int, int]], List[float], List[int]]] = []
    active_count: Dict[str, int] = {}
    top_field_of: Dict[str, str] = {}
    peak_index_of: Dict[str, int] = {}

    for h in heatmaps:
        if h.category not in present:
            continue

        finite = np.isfinite(h.scores)
        active_count[h.category] = int(np.sum(h.scores[finite] > 0.0))
        top_field_of[h.category] = str(getattr(h, "top_field", "") or h.category)
        peak_index_of[h.category] = int(np.argmax(np.where(finite, h.scores, -np.inf)))

        if h.category in tables and tb_span > 0:
            gb = (tb_start, tb_end, 0, cols - 1)
            per_cat.append((
                h.category, [gb], [float(h.top_score)],
                [int(tb_span * cols)],
            ))
            if log is not None:
                log.append(
                    f"    🧾 [TABLE UNION] '{h.category}' | 표 밴드 "
                    f"r{tb_start}~{tb_end}, c0~{cols - 1} "
                    f"({tb_span * cols}패치) 로 통합"
                )
                log.append(
                    f"    🧩 [COMPONENTS] '{h.category}' | 영역 1개 "
                    f"| Top: {top_field_of[h.category]}({h.top_score:+.4f})"
                )
            continue

        gate = core_threshold(h.scores)
        comps = extract_components(h.scores, rows, cols, gate, bc, br)
        if not comps:
            comps = extract_components(h.scores, rows, cols, 0.0, bc, br)
        if not comps:
            if log is not None:
                log.append(f"  ⚪ '{h.category}' 활성 패치 없음 → 제외")
            continue

        boxes: List[Tuple[int, int, int, int]] = []
        peaks: List[float] = []
        counts: List[int] = []
        band_hits = 0

        for comp in comps:
            targets = [comp]
            if comp.area > area_cap:
                subs = split_oversized(
                    h.scores, comp, rows, cols, gate, area_cap, bc, br
                )
                if len(subs) > 1:
                    targets = subs

            for s in targets:
                gb = expand_row_band(
                    s, content, content_gate, cols, bc, max_crop_cols
                )
                if (gb[2], gb[3]) != (s.c_min, s.c_max):
                    band_hits += 1
                    if log is not None:
                        log.append(
                            f"    ↔️ [ROW BAND] '{h.category}' | "
                            f"c{s.c_min}~{s.c_max} → c{gb[2]}~{gb[3]} "
                            f"(같은 행 밴드의 값 셀 편입)"
                        )
                boxes.append(gb)
                peaks.append(s.peak)
                counts.append(len(s.indices))

        per_cat.append((h.category, boxes, peaks, counts))
        if log is not None:
            log.append(
                f"    🧩 [COMPONENTS] '{h.category}' | 영역 {len(boxes)}개 "
                f"| Gate: {gate:+.4f} | Top: {top_field_of[h.category]}"
                f"({h.top_score:+.4f}) | 행밴드 확장 {band_hits}건"
            )

    if not per_cat:
        return []

    raw: List[CropPlan] = []
    for category, boxes, peaks, counts in per_cat:
        for gb, peak, cnt in zip(boxes, peaks, counts):
            padded = _pad_grid_box(gb, rows, cols, grid)
            px = grid.region_bbox(*padded)
            plan = CropPlan(category, px, peak, 0.0, cnt, padded)
            plan.top_field = top_field_of.get(category, category)
            raw.append(plan)

    if log is not None:
        pr = max(1, int(round(grid.orig_height * CROP_PAD_Y_RATIO
                              / max(1.0, grid.cell_height()))))
        pc = max(2, int(round(grid.orig_width * CROP_PAD_X_RATIO
                              / max(1.0, grid.cell_width()))))
        log.append(
            f"    📐 [CROP PAD] 픽셀 기준 패딩 — 세로 ±{pr}행"
            f"(≈{int(grid.orig_height * CROP_PAD_Y_RATIO)}px) / "
            f"가로 ±{pc}열(≈{int(grid.orig_width * CROP_PAD_X_RATIO)}px)"
        )

    if log is not None:
        log.append(
            f"    🎯 [REGION POOL] 후보 영역 {len(raw)}개 "
            f"| 경쟁 카테고리 {len(per_cat)}개 | 면적 상한 {area_cap}패치"
        )

    terr_margin: Dict[str, float] = {}
    terr_rival: Dict[str, str] = {}
    for h in heatmaps:
        terr_margin[h.category] = float(getattr(h, "mean_margin", 0.0) or 0.0)
        terr_rival[h.category] = str(getattr(h, "top_rival", "") or "")

    by_region: Dict[Tuple[int, int, int, int], List[CropPlan]] = {}
    for p in raw:
        by_region.setdefault(p.grid_box, []).append(p)

    for _k, group in by_region.items():
        group.sort(key=lambda x: x.score, reverse=True)
        top = group[0]
        if len(group) > 1:
            top.margin = top.score - group[1].score
            top.rival = group[1].category
        else:
            top.margin = max(top.score, terr_margin.get(top.category, 0.0))
            top.rival = terr_rival.get(top.category, "")
        for loser in group[1:]:
            loser.margin = loser.score - top.score
            loser.rival = top.category

    for p in raw:
        tm = terr_margin.get(p.category, 0.0)
        if tm > p.margin:
            p.margin = tm
            if not p.rival:
                p.rival = terr_rival.get(p.category, "")

    raw.sort(key=lambda p: (p.score, p.area()), reverse=True)

    winners: List[CropPlan] = []
    taken_categories = set()

    min_w = max(64, int(grid.orig_width * 0.12))
    min_h = max(48, int(grid.orig_height * 0.06))

    def _emit(cand: CropPlan, tag: str = "CROP PLAN"):
        covered = _box_patches(cand.grid_box, cols)
        act = active_count.get(cand.category, 0)
        hit = 0
        for h in heatmaps:
            if h.category != cand.category:
                continue
            fin = np.isfinite(h.scores)
            for i in covered:
                if i < h.scores.size and fin[i] and h.scores[i] > 0.0:
                    hit += 1
            break
        cand.coverage = (hit / act) if act else 0.0
        if log is not None:
            pct = int(round(cand.coverage * 100))
            log.append(
                f"    📊 [CROP COVERAGE] '{cand.category}' 히트맵 활성 "
                f"{act}개 중 {hit}개 커버 ({pct}%)"
            )
            if act and cand.coverage < 0.5:
                log.append(
                    f"    ⚠️ [CROP COVERAGE LOSS] '{cand.category}' 활성 "
                    f"패치의 {100 - pct}%가 크롭 밖 — 잘림 발생."
                )
            log.append(
                f"    ✂️ [{tag}] '{cand.category}' ← grid{cand.grid_box} "
                f"→ px{cand.bbox} | Score: {cand.score:+.4f} "
                f"| Margin: {cand.margin:+.4f} | Field: {cand.top_field}"
            )

    table_box = (
        (tb_start, tb_end, 0, cols - 1) if tb_span > 0 else None
    )

    def _shares_table(a: CropPlan, b: CropPlan) -> bool:
        if table_box is None:
            return False
        return a.grid_box == b.grid_box and (
            a.category in tables and b.category in tables
        )

    for cand in raw:
        if cand.category in taken_categories:
            continue

        if cand.margin < margin_threshold:
            continue

        cand.bbox = ensure_min_size(
            cand.bbox, grid.orig_width, grid.orig_height, min_w, min_h
        )

        hit: Optional[CropPlan] = None
        hit_iou = 0.0
        for w in winners:
            if _shares_table(w, cand):
                continue
            iou = compute_iou(w.bbox, cand.bbox)
            if iou >= iou_threshold:
                hit = w
                hit_iou = iou
                break

        if hit is not None:
            hit.absorbed.append({
                "category": cand.category,
                "score": round(cand.score, 4),
                "bbox": list(cand.bbox),
            })
            ax0, ay0, ax1, ay1 = hit.bbox
            bx0, by0, bx1, by1 = cand.bbox
            hit.bbox = (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))
            if log is not None:
                log.append(
                    f"  🚫 IoU 억제 '{cand.category}' (IoU={hit_iou:.2f}) "
                    f"→ '{hit.category}' 에 흡수"
                )
            continue

        winners.append(cand)
        taken_categories.add(cand.category)
        _emit(cand)

    by_cat_regions: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for category, boxes, _peaks, _counts in per_cat:
        by_cat_regions[category] = list(boxes)

    for w in winners:
        if w.coverage >= 0.5:
            continue
        regions = by_cat_regions.get(w.category) or []
        if len(regions) < 2:
            continue

        r0, r1, c0, c1 = w.grid_box
        merged_any = False
        max_r = max(3, int(round(rows * 0.45)))
        max_c = max(4, int(round(cols * 0.75)))

        for gb in sorted(
            regions,
            key=lambda g: abs(((g[0] + g[1]) / 2.0)
                              - ((w.grid_box[0] + w.grid_box[1]) / 2.0)),
        ):
            gr0, gr1, gc0, gc1 = _pad_grid_box(gb, rows, cols, grid)
            nr0, nr1 = min(r0, gr0), max(r1, gr1)
            nc0, nc1 = min(c0, gc0), max(c1, gc1)
            span_r = nr1 - nr0 + 1
            span_c = nc1 - nc0 + 1
            if span_r > max_r or span_c > max_c:
                continue
            if span_r * span_c > area_cap * 3:
                continue
            r0, r1, c0, c1 = nr0, nr1, nc0, nc1
            merged_any = True

        if not merged_any:
            continue
        if (r0, r1, c0, c1) == w.grid_box:
            continue

        w.grid_box = (r0, r1, c0, c1)
        w.bbox = ensure_min_size(
            grid.region_bbox(r0, r1, c0, c1),
            grid.orig_width, grid.orig_height, min_w, min_h,
        )
        if log is not None:
            log.append(
                f"    🧲 [COVERAGE MERGE] '{w.category}' 커버리지 "
                f"{int(w.coverage * 100)}% → 같은 카테고리 영역 "
                f"{len(regions)}개를 grid({r0}, {r1}, {c0}, {c1}) 로 합칩니다."
            )
        _emit(w, tag="CROP REPLAN")

    for h in heatmaps:
        if h.category not in present or h.category in taken_categories:
            continue
        idx = peak_index_of.get(h.category, -1)
        if idx < 0:
            continue
        r, c = divmod(int(idx), cols)
        gb = _pad_grid_box((r, r, c, c), rows, cols, grid)
        plan = CropPlan(
            h.category, grid.region_bbox(*gb), float(h.top_score),
            float(h.top_score), 1, gb, source="rescue",
        )
        plan.top_field = top_field_of.get(h.category, h.category)
        plan.bbox = ensure_min_size(
            plan.bbox, grid.orig_width, grid.orig_height, min_w, min_h
        )
        winners.append(plan)
        taken_categories.add(h.category)
        if log is not None:
            log.append(
                f"    🛟 [STARVATION RESCUE] '{h.category}' 는 영역을 "
                f"선점당했지만 자기 최고 봉우리({h.top_score:+.4f})로 "
                f"독립 크롭합니다."
            )
        _emit(plan, tag="CROP PLAN")

    ib_start, ib_end = _identity_band(rows, content_rows, title_rows)
    ident = identity_category if identity_category in taken_categories else ""
    if ident and ib_start >= 0:
        band = _box_patches((ib_start, ib_end, 0, cols - 1), cols)
        band_content = {i for i in band if i < n and bool(content_ok[i])}
        covered = set()
        for w in winners:
            covered |= _box_patches(w.grid_box, cols)
        got = len(band_content & covered)
        need = len(band_content)
        if need and got * 2 < need:
            gb = _pad_grid_box((ib_start, ib_end, 0, cols - 1), rows, cols, grid)
            peak = 0.0
            for h in heatmaps:
                if h.category == ident:
                    peak = float(h.top_score)
                    break
            plan = CropPlan(
                ident, grid.region_bbox(*gb), peak, peak,
                len(band_content), gb, source="identity-band",
            )
            plan.top_field = top_field_of.get(ident, ident)
            winners.append(plan)
            if log is not None:
                log.append(
                    f"    🪪 [IDENTITY BAND GUARANTEE] '{ident}' 가 식별 밴드 "
                    f"내용을 {got}/{need} 밖에 못 담아 전용 크롭을 추가합니다. "
                    f"r{ib_start}~{ib_end} c0~{cols - 1} → px{plan.bbox} "
                    f"| Peak: {peak:+.4f}"
                )

    covered_rows = set()
    for w in winners:
        r0, r1, _c0, _c1 = w.grid_box
        covered_rows |= set(range(int(r0), int(r1) + 1))
    missing = [r for r in content_rows if r not in covered_rows]
    if log is not None:
        if missing:
            log.append(
                f"    ⚠️ [COVERAGE GUARANTEE] 내용 행 {len(missing)}개가 "
                f"어떤 크롭에도 없습니다: {missing[:12]}"
            )
        else:
            log.append(
                "    ✅ [COVERAGE GUARANTEE] 모든 내용 행이 최소 하나의 "
                "크롭에 포함되어 있습니다."
            )

    if missing and winners:
        for r in missing:
            gb = _pad_grid_box((r, r, 0, cols - 1), rows, cols, grid)
            px = grid.region_bbox(*gb)
            best = min(
                winners,
                key=lambda w: abs(((w.grid_box[0] + w.grid_box[1]) / 2.0) - r),
            )
            ax0, ay0, ax1, ay1 = best.bbox
            bx0, by0, bx1, by1 = px
            best.bbox = (
                min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
            )
            r0 = min(best.grid_box[0], gb[0])
            r1 = max(best.grid_box[1], gb[1])
            best.grid_box = (r0, r1, best.grid_box[2], best.grid_box[3])
        if log is not None:
            log.append(
                f"    🩹 [COVERAGE PATCH] 누락 행 {len(missing)}개를 "
                f"가장 가까운 크롭에 편입했습니다."
            )

    merged: List[CropPlan] = []
    for w in sorted(winners, key=lambda p: (p.category, -p.area())):
        same = None
        for m in merged:
            if m.category != w.category:
                continue
            if _shares_table(m, w):
                continue
            if compute_iou(m.bbox, w.bbox) > 0.0:
                same = m
                break
        if same is None:
            merged.append(w)
            continue
        ax0, ay0, ax1, ay1 = same.bbox
        bx0, by0, bx1, by1 = w.bbox
        new_box = (
            min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1)
        )
        if log is not None:
            log.append(
                f"    🔗 [CROP MERGE] '{w.category}' 의 겹치는 크롭 2개를 "
                f"합칩니다. px{same.bbox} + px{w.bbox} → px{new_box}"
            )
        same.bbox = new_box
        same.grid_box = (
            min(same.grid_box[0], w.grid_box[0]),
            max(same.grid_box[1], w.grid_box[1]),
            min(same.grid_box[2], w.grid_box[2]),
            max(same.grid_box[3], w.grid_box[3]),
        )
        same.score = max(same.score, w.score)

    for p in merged:
        box = _box_patches(p.grid_box, cols)
        p.patches_total = len(box)
        if legibility is not None and int(getattr(legibility, "size", 0)) == n:
            p.legible = int(sum(
                1 for i in box if i < n and bool(legibility.legible[i])
            ))
        r0, r1, _c0, _c1 = p.grid_box
        if tb_span > 0:
            p.table_rows = len(
                set(range(int(r0), int(r1) + 1))
                & set(range(int(tb_start), int(tb_end) + 1))
            )

    merged.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
    if ident:
        order_before = [p.category for p in merged]
        merged.sort(key=lambda p: (0 if p.category == ident else 1,))
        if log is not None and order_before != [p.category for p in merged]:
            log.append(
                f"    🥇 [IDENTITY FIRST ORDER] 문서 기본키 크롭을 선두로 "
                f"재정렬했습니다. {order_before} → "
                f"{[p.category for p in merged]}"
            )

    if log is not None:
        log.append(f"  🧾 [PLAN DONE] 크롭 계획 {len(merged)}건 확정.")
    return merged