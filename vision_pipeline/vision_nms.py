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


def band_gutters_from_boxes(
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]],
    grid: VisionPatchGrid,
    r0: int,
    r1: int,
) -> set:
    if not text_boxes:
        return set()

    cols = max(1, int(grid.cols))
    cw = max(1.0, float(grid.cell_width()))
    ch = max(1.0, float(grid.cell_height()))
    y0 = float(int(r0)) * ch
    y1 = float(int(r1) + 1) * ch

    hit = [False] * cols
    for b in text_boxes:
        if float(b[3]) <= y0 or float(b[1]) >= y1:
            continue
        c0 = int(max(0, min(cols - 1, int(float(b[0]) // cw))))
        c1 = int(max(0, min(cols - 1, int((float(b[2]) - 1.0) // cw))))
        for c in range(c0, c1 + 1):
            hit[c] = True

    return {c for c in range(cols) if not hit[c]}


PLINKO_CLIFF_RATIO = 0.97
PLINKO_BRIDGE_CLIFF_RATIO = 0.80
PLINKO_FLOOR_RATIO = 0.55
PLINKO_SEED_MIN = 0.0
PLINKO_MAX_ROUNDS = 10
PLINKO_BRIDGE_BONUS = 0.01
PLINKO_SIDES = ("down", "right", "up", "left")
PLINKO_ARROW = {"up": "↑", "down": "↓", "left": "←", "right": "→"}


def _strip_box(
    gb: Tuple[int, int, int, int],
    side: str,
    rows: int,
    cols: int,
) -> Optional[Tuple[int, int, int, int]]:
    r0, r1, c0, c1 = (int(v) for v in gb)
    if side == "up":
        return (r0 - 1, r0 - 1, c0, c1) if r0 > 0 else None
    if side == "down":
        return (r1 + 1, r1 + 1, c0, c1) if r1 + 1 < rows else None
    if side == "left":
        return (r0, r1, c0 - 1, c0 - 1) if c0 > 0 else None
    if side == "right":
        return (r0, r1, c1 + 1, c1 + 1) if c1 + 1 < cols else None
    return None


def _bridge_count(
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]],
    grid: VisionPatchGrid,
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> int:
    if not text_boxes:
        return 0
    ax0, ay0, ax1, ay1 = grid.region_bbox(*a)
    bx0, by0, bx1, by1 = grid.region_bbox(*b)
    hits = 0
    for (x0, y0, x1, y1) in text_boxes:
        in_a = x0 < ax1 and x1 > ax0 and y0 < ay1 and y1 > ay0
        if not in_a:
            continue
        in_b = x0 < bx1 and x1 > bx0 and y0 < by1 and y1 > by0
        if in_b:
            hits += 1
    return hits


def plinko_grow_region(
    gb: Tuple[int, int, int, int],
    heatmap: CategoryHeatmap,
    grid: VisionPatchGrid,
    patch_matrix: np.ndarray,
    content_ok: np.ndarray,
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    area_cap: int = 0,
    cliff_ratio: float = PLINKO_CLIFF_RATIO,
    max_rounds: int = PLINKO_MAX_ROUNDS,
) -> Tuple[Tuple[int, int, int, int], List[str], str]:
    rows = max(1, int(grid.rows))
    cols = max(1, int(grid.cols))
    total = rows * cols

    dim = int(patch_matrix.shape[-1]) if patch_matrix.ndim == 2 else 0
    anchor = getattr(heatmap, "anchor", None)
    if anchor is not None:
        anchor = np.asarray(anchor, dtype=np.float32).reshape(-1)
        if dim <= 0 or int(anchor.shape[-1]) != dim:
            anchor = None

    fallback = np.asarray(
        getattr(heatmap, "affinity", heatmap.scores), dtype=np.float32
    )
    axis = "조인트 코사인" if anchor is not None else "친화도 평균"

    def _score(box: Tuple[int, int, int, int]) -> float:
        cells = [i for i in _box_patches(box, cols) if i < total]
        if not cells:
            return float("-inf")
        if anchor is not None and int(patch_matrix.shape[0]) >= total:
            v = patch_matrix[cells].mean(axis=0)
            norm = float(np.linalg.norm(v))
            if norm < 1e-8:
                return float("-inf")
            return float(np.dot(v / norm, anchor))
        vals = [
            float(fallback[i]) for i in cells
            if i < fallback.size and np.isfinite(fallback[i])
        ]
        return float(sum(vals) / len(vals)) if vals else float("-inf")

    cap = int(area_cap) if area_cap and area_cap > 0 else total
    cur = tuple(int(v) for v in gb)
    cur_s = _score(cur)
    seed_s = cur_s
    trace: List[str] = []

    if not np.isfinite(cur_s):
        return cur, trace, axis
    if cur_s <= PLINKO_SEED_MIN:
        return cur, trace, axis

    floor = seed_s * float(PLINKO_FLOOR_RATIO)

    for _round in range(max(1, int(max_rounds))):
        best = None
        for side in PLINKO_SIDES:
            strip = _strip_box(cur, side, rows, cols)
            if strip is None:
                continue
            cells = [i for i in _box_patches(strip, cols) if i < total]
            if not cells:
                continue

            cand = (
                min(cur[0], strip[0]), max(cur[1], strip[1]),
                min(cur[2], strip[2]), max(cur[3], strip[3]),
            )
            area = (cand[1] - cand[0] + 1) * (cand[3] - cand[2] + 1)
            if area > cap:
                continue

            inked = sum(
                1 for i in cells
                if i < int(content_ok.size) and bool(content_ok[i])
            )
            bridge = _bridge_count(text_boxes, grid, cur, strip)
            if inked <= 0 and bridge <= 0:
                continue

            s = _score(cand)
            if not np.isfinite(s):
                continue
            if s <= 0.0:
                continue
            if s < floor:
                continue

            ratio = (
                float(PLINKO_BRIDGE_CLIFF_RATIO) if bridge > 0
                else float(cliff_ratio)
            )
            if s < cur_s * ratio:
                continue

            gain = (s - cur_s) + PLINKO_BRIDGE_BONUS * float(bridge)
            if best is None or gain > best[0]:
                best = (gain, side, cand, s, bridge, inked)

        if best is None:
            break

        _gain, side, cand, s, bridge, inked = best
        why = (
            f"검출박스 {bridge}개가 경계를 가로지름"
            if bridge else f"내용 {inked}칸"
        )
        trace.append(
            f"{PLINKO_ARROW[side]} {cur_s:+.4f}→{s:+.4f} ({why})"
        )
        cur, cur_s = cand, s

    return cur, trace, axis


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


CROP_PAD_LINES_Y = 0.55
CROP_PAD_LINES_X = 1.30
CROP_PAD_FLOOR_PX = 4.0

MERGE_FILL_DROP = 0.65
MERGE_PAGE_RATIO = 0.55
MERGE_ROW_GAP = 1
MULTI_CROP_LIMIT = 3
CROP_MERGE_IOU = 0.25


def crop_pad_px(
    grid: Optional[VisionPatchGrid],
    text_h: float,
) -> Tuple[float, float]:
    unit = float(text_h)
    if unit <= 1.0 and grid is not None:
        unit = max(1.0, float(grid.cell_height()) * 0.33)
    if unit <= 1.0:
        unit = 12.0
    return (
        max(CROP_PAD_FLOOR_PX, unit * CROP_PAD_LINES_X),
        max(CROP_PAD_FLOOR_PX, unit * CROP_PAD_LINES_Y),
    )


def _pad_grid_box(
    gb: Tuple[int, int, int, int],
    rows: int,
    cols: int,
    grid: Optional[VisionPatchGrid] = None,
    pad_px: Optional[Tuple[float, float]] = None,
) -> Tuple[int, int, int, int]:
    r0, r1, c0, c1 = (int(v) for v in gb)

    pr, pc = 0, 0
    if grid is not None and pad_px is not None:
        ch = max(1.0, float(grid.cell_height()))
        cw = max(1.0, float(grid.cell_width()))
        pr = int(float(pad_px[1]) // ch)
        pc = int(float(pad_px[0]) // cw)

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
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
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

    text_h = 0.0
    if text_boxes:
        from .text_boxes import median_text_height
        text_h = float(median_text_height(text_boxes))
    pad_grid = crop_pad_px(grid, text_h)

    pmat = np.asarray(grid.embeddings, dtype=np.float32)
    if pmat.ndim == 1:
        pmat = pmat.reshape(1, -1)
    if pmat.size:
        pmat = pmat / np.maximum(
            np.linalg.norm(pmat, axis=1, keepdims=True), 1e-8
        )
    anchored = sum(
        1 for h in heatmaps if getattr(h, "anchor", None) is not None
    )

    band_gutter_cache: Dict[Tuple[int, int], set] = {}

    def _band_gutters(br0: int, br1: int) -> set:
        key = (int(br0), int(br1))
        got = band_gutter_cache.get(key)
        if got is not None:
            return got
        out = band_gutters_from_boxes(text_boxes, grid, br0, br1)
        band_gutter_cache[key] = out
        return out

    def _tight(px: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        if not text_boxes:
            return px
        from .text_boxes import box_tighten
        fitted = box_tighten(
            text_boxes, px,
            pad_x=pad_grid[0], pad_y=pad_grid[1],
            bounds=(grid.orig_width, grid.orig_height),
        )
        return fitted if fitted is not None else px

    has_legibility = (
        legibility is not None and int(getattr(legibility, "size", 0)) == n
    )

    def _region_ink(
        gb: Tuple[int, int, int, int],
        px_box: Tuple[int, int, int, int],
    ) -> Tuple[int, int, int]:
        cells = [i for i in _box_patches(gb, cols) if i < n]
        inked_n = sum(1 for i in cells if bool(content_ok[i]))

        legible_n = -1
        if has_legibility:
            legible_n = sum(
                1 for i in cells if bool(legibility.legible[i])
            )

        boxed_n = 0
        if text_boxes:
            bx0, by0, bx1, by1 = (int(v) for v in px_box)
            boxed_n = sum(
                1 for b in text_boxes
                if b[0] < bx1 and b[2] > bx0 and b[1] < by1 and b[3] > by0
            )
        return boxed_n, legible_n, inked_n

    def _region_blank(
        gb: Tuple[int, int, int, int],
        px_box: Tuple[int, int, int, int],
    ) -> Tuple[bool, int, int, int]:
        boxed_n, legible_n, inked_n = _region_ink(gb, px_box)
        if boxed_n > 0:
            return False, boxed_n, legible_n, inked_n
        if legible_n >= 0:
            return legible_n <= 0, boxed_n, legible_n, inked_n
        return inked_n <= 0, boxed_n, legible_n, inked_n

    if log is not None:
        log.append(
            f"  🧪 [BLANK GATE] 후보 영역 판정 기준을 STEP 4 의 "
            f"EMPTY CROP SKIP 과 일치시킵니다 — 교차 검출 박스 0개 이고 "
            f"{'판독 가능 패치' if has_legibility else '잉크 패치'} 0개면 "
            f"크롭을 만들지 않습니다. content_ok 는 판독불가 잉크까지 "
            f"포함하므로 plan 단계에서만 통과시키는 원인이었습니다."
        )

    if log is not None:
        log.append(
            f"  📊 [PLAN_CROPS INPUT] 히트맵 {len(heatmaps)}개 "
            f"(존재 판정 통과 {len(present)}개) | 격자 {rows}x{cols}={n} "
            f"| area_cap={area_cap} | content 활성 {int(content_ok.sum())}/{n}"
        )
        log.append(
            f"  🎰 [PLINKO READY] 카테고리 앵커 {anchored}/{len(heatmaps)}개 "
            f"확보 | 패치 행렬 {tuple(pmat.shape)} — 조각을 아래·오른쪽·위·"
            f"왼쪽 네 방향으로 한 줄씩 붙여 보고 코사인이 절벽에 닿는 곳에서 "
            f"멈춥니다. 표의 가로·세로 방향은 이득이 큰 쪽으로 자동 결정됩니다."
        )
        log.append(
            f"  📐 [CROP PAD] 검출 박스 중앙 높이 {text_h:.0f}px 기준 — "
            f"가로 ±{pad_grid[0]:.0f}px / 세로 ±{pad_grid[1]:.0f}px "
            f"(격자 한 칸 {grid.cell_width():.0f}x{grid.cell_height():.0f}px "
            f"→ 격자 패딩 {int(pad_grid[1] // max(1.0, grid.cell_height()))}행"
            f"/{int(pad_grid[0] // max(1.0, grid.cell_width()))}열, "
            f"나머지는 픽셀 단계에서 글자 경계에 맞춥니다)"
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
        plinko_hits = 0

        for comp in comps:
            targets = [comp]
            if comp.area > area_cap:
                subs = split_oversized(
                    h.scores, comp, rows, cols, gate, area_cap, bc, br
                )
                if len(subs) > 1:
                    targets = subs

            for s in targets:
                local = _band_gutters(s.r_min, s.r_max)
                gb = expand_row_band(
                    s, content, content_gate, cols,
                    bc | local, max_crop_cols,
                )
                if (gb[2], gb[3]) != (s.c_min, s.c_max):
                    band_hits += 1
                    if log is not None:
                        wall = (
                            f" | 밴드 여백 열 {sorted(local)} 에서 멈춤"
                            if local else ""
                        )
                        log.append(
                            f"    ↔️ [ROW BAND] '{h.category}' | "
                            f"c{s.c_min}~{s.c_max} → c{gb[2]}~{gb[3]} "
                            f"(같은 행 밴드의 값 셀 편입){wall}"
                        )

                grown, trace, axis = plinko_grow_region(
                    gb, h, grid, pmat, content_ok,
                    text_boxes=text_boxes, area_cap=area_cap,
                )
                if trace:
                    plinko_hits += 1
                    if log is not None:
                        log.append(
                            f"    🎰 [PLINKO] '{h.category}' grid{gb} → "
                            f"grid{grown} | {len(trace)}라운드 | 축 {axis} "
                            f"— 점수가 절벽에 닿을 때까지만 이어 붙입니다."
                        )
                        for step in trace:
                            log.append(f"       {step}")
                    gb = grown

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
            padded = _pad_grid_box(gb, rows, cols, grid, pad_grid)
            px = grid.region_bbox(*padded)
            plan = CropPlan(category, px, peak, 0.0, cnt, padded)
            plan.top_field = top_field_of.get(category, category)
            raw.append(plan)

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

    cw_px = max(1.0, float(grid.cell_width()))
    ch_px = max(1.0, float(grid.cell_height()))
    page_px = float(max(1, grid.orig_width * grid.orig_height))

    def _emit(cand: CropPlan, tag: str = "CROP PLAN"):
        x0, y0, x1, y1 = cand.bbox
        act = active_count.get(cand.category, 0)
        hit = 0
        for h in heatmaps:
            if h.category != cand.category:
                continue
            fin = np.isfinite(h.scores)
            for i in range(min(int(h.scores.size), n)):
                if not fin[i] or h.scores[i] <= 0.0:
                    continue
                r, c = divmod(i, cols)
                cx = (c + 0.5) * cw_px
                cy = (r + 0.5) * ch_px
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    hit += 1
            break
        cand.coverage = (hit / act) if act else 0.0
        if log is not None:
            pct = int(round(cand.coverage * 100))
            share = int(round(
                100.0 * max(0, x1 - x0) * max(0, y1 - y0) / page_px
            ))
            log.append(
                f"    📊 [CROP COVERAGE] '{cand.category}' 히트맵 활성 "
                f"{act}개 중 {hit}개 커버 ({pct}%) | 지면 점유 {share}%"
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

    empty_skipped: Dict[str, int] = {}

    for cand in raw:
        if cand.category in taken_categories:
            continue

        if cand.margin < margin_threshold:
            continue

        cand.bbox = ensure_min_size(
            cand.bbox, grid.orig_width, grid.orig_height, min_w, min_h
        )

        blank, boxed_n, legible_n, inked_n = _region_blank(
            cand.grid_box, cand.bbox
        )
        if blank:
            empty_skipped[cand.category] = (
                empty_skipped.get(cand.category, 0) + 1
            )
            if log is not None:
                log.append(
                    f"    ⛔ [EMPTY REGION SKIP] '{cand.category}' 후보 "
                    f"grid{cand.grid_box} px{cand.bbox} — 교차 검출 박스 "
                    f"{boxed_n}개 / 판독 가능 패치 {max(0, legible_n)}개 "
                    f"/ 잉크 패치 {inked_n}개. 히트맵 봉우리가 여백이나 "
                    f"판독불가 얼룩에 찍힌 것이므로 다음 후보 영역으로 "
                    f"넘어갑니다 — 빈 크롭을 VLM 에 보내면 빈 사고 블록이 "
                    f"돌아와 남은 전체 크롭의 토큰 예산이 3배로 뜁니다."
                )
            continue

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
            if hit.category != cand.category:
                cand.source = f"twin-of:{hit.category}"
                winners.append(cand)
                taken_categories.add(cand.category)
                if log is not None:
                    log.append(
                        f"  👯 [TWIN KEEP] '{cand.category}' 와 "
                        f"'{hit.category}' 의 좌표가 IoU {hit_iou:.2f} 로 "
                        f"겹칩니다. 서로 다른 축이므로 흡수하지 않고 같은 "
                        f"좌표를 공유한 채 남깁니다 — 흡수하면 스키마 "
                        f"카테고리 하나가 통째로 사라지고, 뒤이은 SPLIT 이 "
                        f"거의 같은 자리에 크롭을 다시 만들어 같은 그림을 "
                        f"두 번 보내게 됩니다."
                    )
                _emit(cand)
                continue

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
                    f"→ 같은 카테고리의 '{hit.category}' 영역에 흡수"
                )
            continue

        winners.append(cand)
        taken_categories.add(cand.category)
        _emit(cand)

    if log is not None and empty_skipped:
        brief = ", ".join(
            f"{k}({v}건)" for k, v in sorted(empty_skipped.items())
        )
        log.append(
            f"    ⛔ [EMPTY REGION SKIP] 글자가 전혀 없는 후보 영역 "
            f"{sum(empty_skipped.values())}건을 건너뛰었습니다 — {brief}. "
            f"해당 카테고리는 다음 후보나 SPLIT/RESCUE 크롭으로 대체되며, "
            f"끝내 못 찾으면 이 문서에 없는 축입니다."
        )

    by_cat_regions: Dict[str, List[Tuple[int, int, int, int]]] = {}
    for category, boxes, _peaks, _counts in per_cat:
        by_cat_regions[category] = list(boxes)

    COVERAGE_FLOOR = 0.70
    col_gap_tol = max(1, cols // 6)
    page_cells = float(max(1, rows * cols))
    extra: List[CropPlan] = []

    def _fill_ratio(gb: Tuple[int, int, int, int]) -> float:
        cells = _box_patches(gb, cols)
        if not cells:
            return 0.0
        got = sum(1 for i in cells if i < n and bool(content_ok[i]))
        return got / float(len(cells))

    def _adjacent(a, b) -> bool:
        ar0, ar1, ac0, ac1 = a
        br0, br1, bc0, bc1 = b
        row_gap = max(0, max(ar0, br0) - min(ar1, br1))
        col_gap = max(0, max(ac0, bc0) - min(ac1, bc1))
        return row_gap <= MERGE_ROW_GAP and col_gap <= col_gap_tol

    for w in list(winners):
        if w.coverage >= COVERAGE_FLOOR:
            continue
        regions = [
            _pad_grid_box(gb, rows, cols, grid, pad_grid)
            for gb in (by_cat_regions.get(w.category) or [])
        ]
        regions = [gb for gb in regions if gb != w.grid_box]
        if not regions:
            continue

        base_fill = _fill_ratio(w.grid_box)
        merged_box = w.grid_box
        joined = 0
        left: List[Tuple[int, int, int, int]] = []

        for gb in sorted(
            regions,
            key=lambda g: abs(((g[0] + g[1]) / 2.0)
                              - ((w.grid_box[0] + w.grid_box[1]) / 2.0)),
        ):
            if not _adjacent(merged_box, gb):
                left.append(gb)
                continue
            cand = (
                min(merged_box[0], gb[0]), max(merged_box[1], gb[1]),
                min(merged_box[2], gb[2]), max(merged_box[3], gb[3]),
            )
            area = float((cand[1] - cand[0] + 1) * (cand[3] - cand[2] + 1))
            if area > page_cells * MERGE_PAGE_RATIO:
                left.append(gb)
                if log is not None:
                    log.append(
                        f"    ⛔ [MERGE SKIP] '{w.category}' 두 영역을 합치면 "
                        f"지면의 {area / page_cells:.0%} 를 차지합니다. "
                        f"합치지 않고 별도 크롭으로 남깁니다."
                    )
                continue
            cand_fill = _fill_ratio(cand)
            if cand_fill < base_fill * MERGE_FILL_DROP:
                left.append(gb)
                if log is not None:
                    log.append(
                        f"    ⛔ [MERGE SKIP] '{w.category}' 병합 사각형의 "
                        f"내용 밀도가 {cand_fill:.0%} 로 원본 {base_fill:.0%} "
                        f"대비 급락합니다. 두 영역 사이가 여백이라는 뜻이므로 "
                        f"합치지 않습니다."
                    )
                continue
            merged_box = cand
            joined += 1

        if joined:
            w.grid_box = merged_box
            w.bbox = ensure_min_size(
                grid.region_bbox(*merged_box),
                grid.orig_width, grid.orig_height, min_w, min_h,
            )
            if log is not None:
                log.append(
                    f"    🧲 [ADJACENT MERGE] '{w.category}' 커버리지 "
                    f"{int(w.coverage * 100)}% < {int(COVERAGE_FLOOR * 100)}% "
                    f"→ 행이 겹치고 열 간격 {col_gap_tol}칸 이내인 이웃 영역 "
                    f"{joined}개만 합쳤습니다. grid{merged_box}"
                )
            _emit(w, tag="CROP REPLAN")

        for gb in left[: max(0, MULTI_CROP_LIMIT - 1)]:
            grown, trace, _axis = plinko_grow_region(
                gb, next(
                    (h for h in heatmaps if h.category == w.category),
                    None,
                ) or w, grid, pmat, content_ok,
                text_boxes=text_boxes, area_cap=area_cap,
            )
            if trace:
                gb = grown
            px = _tight(ensure_min_size(
                grid.region_bbox(*gb),
                grid.orig_width, grid.orig_height, min_w, min_h,
            ))

            inside = 0
            if text_boxes:
                from .text_boxes import boxes_in_region
                inside = len(boxes_in_region(text_boxes, px))
                if inside <= 0:
                    if log is not None:
                        log.append(
                            f"    ⛔ [SPLIT SKIP] '{w.category}' 잔여 영역 "
                            f"grid{gb} 안에 검출된 글자가 없습니다. 빈 크롭을 "
                            f"만들지 않습니다."
                        )
                    continue

            if any(
                compute_iou(px, p.bbox) >= iou_threshold
                for p in list(winners) + extra
            ):
                continue
            side = CropPlan(
                w.category, px, w.score, w.margin,
                len(_box_patches(gb, cols)), gb, source="split",
            )
            side.top_field = w.top_field
            extra.append(side)
            if log is not None:
                tail = (
                    f" | PLINKO {len(trace)}라운드 성장" if trace else ""
                )
                log.append(
                    f"    ➕ [SPLIT CROP] '{w.category}' 는 떨어진 영역이 "
                    f"남아 있어 하나로 합치는 대신 크롭을 하나 더 만듭니다. "
                    f"grid{gb} → px{px} (글자 {inside}덩이){tail} — 행 원장이 "
                    f"있어 같은 줄을 두 번 읽지 않습니다."
                )
            _emit(side, tag="CROP PLAN")

    if extra:
        winners.extend(extra)

    if text_boxes:
        fitted_n = 0
        saved_px = 0
        for w in winners:
            fitted = _tight(w.bbox)
            if fitted == w.bbox:
                continue
            before_a = max(1, (w.bbox[2] - w.bbox[0]) * (w.bbox[3] - w.bbox[1]))
            after_a = max(1, (fitted[2] - fitted[0]) * (fitted[3] - fitted[1]))
            if log is not None:
                log.append(
                    f"    📐 [TEXT FIT] '{w.category}' 크롭을 안쪽 글자 "
                    f"경계에 맞춥니다. px{w.bbox} → px{fitted} "
                    f"(면적 {after_a / before_a:.0%})"
                )
            saved_px += max(0, before_a - after_a)
            w.bbox = fitted
            fitted_n += 1
        if log is not None and fitted_n:
            log.append(
                f"    📐 [TEXT FIT] {fitted_n}건의 크롭이 글자 경계 안쪽으로 "
                f"정렬되었습니다 — 여백 {saved_px / 1000.0:.0f}K px² 제거. "
                f"경계 밖으로는 글자 높이에서 계산한 "
                f"가로 {pad_grid[0]:.0f}px / 세로 {pad_grid[1]:.0f}px 만 "
                f"넘어갑니다."
            )

    for h in heatmaps:
        if h.category not in present or h.category in taken_categories:
            continue
        idx = peak_index_of.get(h.category, -1)
        if idx < 0:
            continue
        r, c = divmod(int(idx), cols)
        seed = Component(
            [int(idx)], r, r, c, c, float(h.top_score), float(h.top_score)
        )
        band_box = expand_row_band(
            seed, content, content_gate, cols,
            bc | _band_gutters(r, r), max_crop_cols,
        )
        band_box, rescue_trace, _axis = plinko_grow_region(
            band_box, h, grid, pmat, content_ok,
            text_boxes=text_boxes, area_cap=area_cap,
        )
        if rescue_trace and log is not None:
            log.append(
                f"    🎰 [PLINKO] '{h.category}' 구제 크롭을 "
                f"{len(rescue_trace)}라운드 성장시켰습니다 → grid{band_box}"
            )
        gb = _pad_grid_box(band_box, rows, cols, grid, pad_grid)
        px = _tight(ensure_min_size(
            grid.region_bbox(*gb),
            grid.orig_width, grid.orig_height, min_w, min_h,
        ))

        blank, boxed_n, legible_n, inked_n = _region_blank(gb, px)
        if blank:
            if log is not None:
                log.append(
                    f"    ⛔ [RESCUE SKIP] '{h.category}' 구제 크롭 grid{gb} "
                    f"px{px} 에도 글자가 없습니다 (검출 박스 {boxed_n} / "
                    f"판독 가능 {max(0, legible_n)} / 잉크 {inked_n}). "
                    f"최고 봉우리조차 여백에 찍혔다는 뜻이므로 이 축은 이 "
                    f"문서에 없는 것으로 두고 빈 크롭을 만들지 않습니다."
                )
            continue

        plan = CropPlan(
            h.category, px, float(h.top_score),
            float(h.top_score), 1, gb, source="rescue",
        )
        plan.top_field = top_field_of.get(h.category, h.category)
        winners.append(plan)
        taken_categories.add(h.category)
        if log is not None:
            log.append(
                f"    🛟 [STARVATION RESCUE] '{h.category}' 는 영역을 "
                f"선점당했지만 자기 최고 봉우리({h.top_score:+.4f})가 있는 "
                f"행 밴드 r{band_box[0]}~{band_box[1]} "
                f"c{band_box[2]}~{band_box[3]} 만 독립 크롭합니다."
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
            ic0, ic1 = cols, 0
            for rr in range(ib_start, ib_end + 1):
                for c in range(cols):
                    i = rr * cols + c
                    if i < n and bool(content_ok[i]):
                        ic0 = min(ic0, c)
                        ic1 = max(ic1, c)
            if ic0 > ic1:
                ic0, ic1 = 0, cols - 1

            gb = _pad_grid_box(
                (ib_start, ib_end, ic0, ic1), rows, cols, grid, pad_grid
            )
            px = _tight(ensure_min_size(
                grid.region_bbox(*gb),
                grid.orig_width, grid.orig_height, min_w, min_h,
            ))
            peak = 0.0
            for h in heatmaps:
                if h.category == ident:
                    peak = float(h.top_score)
                    break
            plan = CropPlan(
                ident, px, peak, peak,
                len(band_content), gb, source="identity-band",
            )
            plan.top_field = top_field_of.get(ident, ident)
            winners.append(plan)
            if log is not None:
                log.append(
                    f"    🪪 [IDENTITY BAND GUARANTEE] '{ident}' 가 식별 밴드 "
                    f"내용을 {got}/{need} 밖에 못 담아 전용 크롭을 추가합니다. "
                    f"r{ib_start}~{ib_end} c{ic0}~{ic1} → px{px} "
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
        bands: List[Tuple[int, int]] = []
        run = [missing[0], missing[0]]
        for r in missing[1:]:
            if r == run[1] + 1:
                run[1] = r
                continue
            bands.append((run[0], run[1]))
            run = [r, r]
        bands.append((run[0], run[1]))

        added = 0
        for br0, br1 in bands:
            rc0, rc1 = cols, 0
            for rr in range(br0, br1 + 1):
                for c in range(cols):
                    i = rr * cols + c
                    if i < n and bool(content_ok[i]):
                        rc0 = min(rc0, c)
                        rc1 = max(rc1, c)
            if rc0 > rc1:
                rc0, rc1 = 0, cols - 1

            owner = ""
            owner_field = ""
            best_score = -np.inf
            for h in heatmaps:
                fin = np.isfinite(h.scores)
                for rr in range(br0, br1 + 1):
                    for c in range(rc0, rc1 + 1):
                        i = rr * cols + c
                        if i >= h.scores.size or not fin[i]:
                            continue
                        if float(h.scores[i]) > best_score:
                            best_score = float(h.scores[i])
                            owner = h.category
                            owner_field = top_field_of.get(
                                h.category, h.category
                            )
            if not owner:
                continue

            gb = _pad_grid_box(
                (br0, br1, rc0, rc1), rows, cols, grid, pad_grid
            )
            px = _tight(ensure_min_size(
                grid.region_bbox(*gb),
                grid.orig_width, grid.orig_height, min_w, min_h,
            ))
            if any(compute_iou(px, w.bbox) >= iou_threshold for w in winners):
                continue

            plan = CropPlan(
                owner, px, best_score, 0.0,
                (br1 - br0 + 1) * (rc1 - rc0 + 1),
                gb, source="coverage-rescue",
            )
            plan.top_field = owner_field
            winners.append(plan)
            added += 1
            if log is not None:
                log.append(
                    f"    🩹 [COVERAGE RESCUE] 미커버 행 밴드 r{br0}~{br1} "
                    f"(c{rc0}~{rc1}) 를 '{owner}' 소유로 전용 크롭합니다. "
                    f"→ px{px} | Peak: {best_score:+.4f} — 기존 크롭을 "
                    f"넓히지 않습니다."
                )
        if log is not None:
            log.append(
                f"    🩹 [COVERAGE RESCUE] 누락 행 {len(missing)}개를 밴드 "
                f"{len(bands)}개로 묶어 전용 크롭 {added}건을 추가했습니다."
            )

    seen_boxes: Dict[Tuple[int, int, int, int], str] = {}
    twins: List[str] = []
    for w in sorted(winners, key=lambda p: -p.score):
        owner = seen_boxes.get(w.bbox)
        if owner is None:
            seen_boxes[w.bbox] = w.category
            continue
        twins.append(w.category)
        w.source = f"twin-of:{owner}"

    if log is not None and twins:
        log.append(
            f"    👯 [TWIN CROP] 좌표가 완전히 같은 크롭 {len(twins)}건 "
            f"({', '.join(twins)}) — 점수가 낮은 쪽은 배열을 만들지 "
            f"않습니다. 같은 표에서 두 카테고리가 같은 값을 복제하는 것을 "
            f"막습니다."
        )

    merged: List[CropPlan] = []
    for w in sorted(winners, key=lambda p: (p.category, -p.area())):
        same = None
        for m in merged:
            if m.category != w.category:
                continue
            if _shares_table(m, w):
                continue
            if compute_iou(m.bbox, w.bbox) >= CROP_MERGE_IOU:
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
        new_area = float(
            max(0, new_box[2] - new_box[0]) * max(0, new_box[3] - new_box[1])
        )
        if new_area > page_px * MERGE_PAGE_RATIO:
            if log is not None:
                log.append(
                    f"    ⛔ [MERGE SKIP] '{w.category}' 두 크롭을 합치면 "
                    f"지면의 {new_area / page_px:.0%} 를 차지합니다. "
                    f"별도 크롭으로 유지합니다."
                )
            merged.append(w)
            continue
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