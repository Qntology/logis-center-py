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


def plan_crops(
    heatmaps: Sequence[CategoryHeatmap],
    grid: VisionPatchGrid,
    iou_threshold: float = 0.80,
    margin_threshold: float = 0.28,
    col_gutters: Optional[set] = None,
    row_gutters: Optional[set] = None,
    max_crop_cols: int = 0,
    log: Optional[List[str]] = None,
) -> List[CropPlan]:
    if not heatmaps or grid.num_patches == 0:
        return []

    rows, cols = grid.rows, grid.cols
    n = rows * cols

    bc = set(col_gutters or ())
    br = set(row_gutters or ())

    content, content_gate = build_content_mask(heatmaps, n)
    if log is not None:
        active = int(np.sum(content > content_gate))
        log.append(
            f"  🗺 CONTENT MASK 활성 {active}/{n} | gate {content_gate:+.4f}"
        )
        log.append(
            f"  📏 거터 검출 — 세로 {len(bc)}개 {sorted(bc)} / "
            f"가로 {len(br)}개 {sorted(br)}"
        )

    present = presence_gate(heatmaps, n, log=log)
    area_cap = max(4, n // max(1, len(present)))

    per_cat: List[Tuple[str, List[Tuple[int, int, int, int]], List[float], List[int]]] = []

    for h in heatmaps:
        if h.category not in present:
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
        split_hits = 0

        for comp in comps:
            if comp.area > area_cap:
                subs = split_oversized(
                    h.scores, comp, rows, cols, gate, area_cap, bc, br
                )
                if len(subs) > 1:
                    split_hits += 1
                    for s in subs:
                        boxes.append(
                            expand_row_band(
                                s, content, content_gate, cols, bc, max_crop_cols
                            )
                        )
                        peaks.append(s.peak)
                        counts.append(len(s.indices))
                    continue
            boxes.append(
                expand_row_band(
                    comp, content, content_gate, cols, bc, max_crop_cols
                )
            )
            peaks.append(comp.peak)
            counts.append(len(comp.indices))

        per_cat.append((h.category, boxes, peaks, counts))
        if log is not None:
            extra = f" | 과대 분할 {split_hits}건" if split_hits else ""
            log.append(
                f"  🧩 '{h.category}' 영역 {len(boxes)}개 | gate {gate:+.4f} "
                f"| 면적상한 {area_cap}{extra}"
            )

    if not per_cat:
        return []

    raw: List[CropPlan] = []
    for category, boxes, peaks, counts in per_cat:
        for gb, peak, cnt in zip(boxes, peaks, counts):
            px = grid.region_bbox(*gb)
            raw.append(CropPlan(category, px, peak, 0.0, cnt, gb))

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

    for cand in raw:
        hit: Optional[CropPlan] = None
        hit_iou = 0.0
        for w in winners:
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

        if cand.category in taken_categories:
            if log is not None:
                log.append(f"  🔒 '{cand.category}' 이미 선점됨 → 차단")
            continue

        if cand.margin < margin_threshold:
            if log is not None:
                log.append(
                    f"  ⚠️ '{cand.category}' margin 부족 "
                    f"({cand.margin:+.4f} < {margin_threshold:.2f}) → 폐기"
                )
            continue

        cand.bbox = ensure_min_size(
            cand.bbox, grid.orig_width, grid.orig_height, min_w, min_h
        )
        winners.append(cand)
        taken_categories.add(cand.category)
        if log is not None:
            log.append(
                f"  ✂️ CROP '{cand.category}' grid{cand.grid_box} → px{cand.bbox} "
                f"| score={cand.score:+.4f} margin={cand.margin:+.4f}"
            )

    winners.sort(key=lambda p: (p.bbox[1], p.bbox[0]))
    return winners