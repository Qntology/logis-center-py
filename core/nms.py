import math
from typing import List, Dict, Sequence, Tuple, Optional

import numpy as np


class Candidate:
    def __init__(
        self,
        field_name: str,
        bbox: Tuple[int, int, int, int],
        score: float,
        row: int,
        col: int,
        status: str = "alive",
        region_key: Optional[Tuple[int, int]] = None,
    ):
        self.field_name = field_name
        self.bbox = tuple(bbox)
        self.score = float(score)
        self.row = int(row)
        self.col = int(col)
        self.status = status
        self.region_key = region_key if region_key is not None else (int(row), int(col))
        self.margin = 0.0
        self.rival = ""
        self.absorbed: List[dict] = []

    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)

    def to_dict(self):
        return {
            "field_name": self.field_name,
            "bbox": list(self.bbox),
            "score": self.score,
            "margin": self.margin,
            "rival": self.rival,
            "row": self.row,
            "col": self.col,
            "status": self.status,
            "absorbed": list(self.absorbed),
        }


class NMSProcessor:
    def __init__(
        self,
        iou_threshold: float = 0.8,
        margin_threshold: float = 0.28,
    ):
        self.iou_threshold = iou_threshold
        self.margin_threshold = margin_threshold
        self.log: List[str] = []

    def compute_iou(
        self,
        box_a: Tuple[int, int, int, int],
        box_b: Tuple[int, int, int, int],
    ) -> float:
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b

        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)

        if ix0 >= ix1 or iy0 >= iy1:
            return 0.0

        inter = (ix1 - ix0) * (iy1 - iy0)
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter

        if union <= 0:
            return 0.0
        return inter / union

    def compute_margins(self, candidates: List[Candidate]) -> None:
        by_region: Dict[Tuple[int, int], List[Candidate]] = {}
        for c in candidates:
            by_region.setdefault(c.region_key, []).append(c)

        for _key, group in by_region.items():
            group.sort(key=lambda x: x.score, reverse=True)
            top = group[0]
            if len(group) > 1:
                second = group[1]
                top.margin = top.score - second.score
                top.rival = second.field_name
            else:
                top.margin = top.score
                top.rival = ""
            for loser in group[1:]:
                loser.margin = loser.score - top.score
                loser.rival = top.field_name

    def suppress_overlaps(
        self,
        candidates: List[Candidate],
    ) -> List[Candidate]:
        pool = [c for c in candidates if c.status == "alive"]
        pool.sort(key=lambda c: (c.score, c.area()), reverse=True)

        winners: List[Candidate] = []

        for cand in pool:
            absorbed_by: Optional[Candidate] = None
            hit_iou = 0.0

            for win in winners:
                iou = self.compute_iou(win.bbox, cand.bbox)
                if iou >= self.iou_threshold:
                    absorbed_by = win
                    hit_iou = iou
                    break

            if absorbed_by is None:
                winners.append(cand)
                continue

            cand.status = "suppressed"
            absorbed_by.absorbed.append({
                "field_name": cand.field_name,
                "score": cand.score,
                "bbox": list(cand.bbox),
                "row": cand.row,
                "col": cand.col,
            })
            ax0, ay0, ax1, ay1 = absorbed_by.bbox
            bx0, by0, bx1, by1 = cand.bbox
            absorbed_by.bbox = (
                min(ax0, bx0),
                min(ay0, by0),
                max(ax1, bx1),
                max(ay1, by1),
            )
            self.log.append(
                f"🚫 IoU 억제 r{cand.row}c{cand.col} '{cand.field_name}' "
                f"(IoU={hit_iou:.2f}, score={cand.score:+.4f}) "
                f"→ '{absorbed_by.field_name}' 에 흡수"
            )

        return pool

    def exclusive_assign(
        self,
        candidates: List[Candidate],
    ) -> List[Candidate]:
        alive = [c for c in candidates if c.status == "alive"]
        self.compute_margins(alive)
        alive.sort(key=lambda c: (c.score, c.margin), reverse=True)

        assigned_fields = set()
        assigned_regions = set()
        assigned: List[Candidate] = []

        for c in alive:
            if c.field_name in assigned_fields:
                c.status = "exclusive_blocked"
                self.log.append(
                    f"🔒 '{c.field_name}' 이미 선점됨 → r{c.row}c{c.col} 차단"
                )
                continue

            if c.region_key in assigned_regions:
                c.status = "region_taken"
                self.log.append(
                    f"🔒 r{c.row}c{c.col} 영역은 이미 다른 필드가 소유 → "
                    f"'{c.field_name}' 차단"
                )
                continue

            if c.margin >= self.margin_threshold:
                c.status = "confirmed"
                assigned_fields.add(c.field_name)
                assigned_regions.add(c.region_key)
                assigned.append(c)
                rival = c.rival if c.rival else "-"
                self.log.append(
                    f"✨ r{c.row}c{c.col} 확정 ('{c.field_name}', "
                    f"score={c.score:+.4f}, margin={c.margin:+.4f} vs '{rival}')"
                )
            else:
                c.status = "below-margin"
                self.log.append(
                    f"⚠️ '{c.field_name}' margin 부족 "
                    f"({c.margin:+.4f} < {self.margin_threshold:.4f}) → 폐기"
                )

        return assigned

    def process(
        self,
        candidates: List[Candidate],
    ) -> Tuple[List[Candidate], List[str]]:
        self.log = []
        if not candidates:
            self.log.append("⚪ 후보가 없어 NMS 를 수행하지 않았습니다.")
            return [], self.log

        self.log.append(
            f"▶ NMS 시작: 후보 {len(candidates)}개 "
            f"(IoU={self.iou_threshold:.2f}, margin={self.margin_threshold:.2f})"
        )
        suppressed = self.suppress_overlaps(candidates)
        alive_cnt = sum(1 for c in suppressed if c.status == "alive")
        self.log.append(f"▶ IoU 억제 후 생존 {alive_cnt}개")

        assigned = self.exclusive_assign(suppressed)
        self.log.append(f"▶ 배타 배정 확정 {len(assigned)}개")
        return assigned, self.log


def gumbel_expected_z(n: int) -> float:
    if n <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(float(n)))


def bank_internal_cohesion(vectors: Sequence[np.ndarray]) -> float:
    rows: List[np.ndarray] = []
    dim = 0
    for v in (vectors or []):
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.any(arr):
            continue
        if dim == 0:
            dim = int(arr.shape[-1])
        if int(arr.shape[-1]) != dim:
            continue
        rows.append(arr)

    if len(rows) < 2:
        return 0.0

    mat = np.stack(rows).astype(np.float32)
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
    sims = mat @ mat.T
    iu = np.triu_indices(sims.shape[0], k=1)
    val = float(np.mean(sims[iu]))
    if not math.isfinite(val):
        return 0.0
    return max(0.0, min(1.0, val))


def prejudice_dominates(own: float, prej: float, cohesion: float) -> bool:
    if not math.isfinite(float(own)) or float(own) <= 0.0:
        return True
    relief = max(0.0, min(0.5, float(cohesion)))
    return float(prej) > float(own) * (1.0 + relief)


def decisive_margin(scores: Sequence[float]) -> Tuple[float, float, bool]:
    vals: List[float] = []
    for s in (scores or []):
        try:
            f = float(s)
        except Exception:
            continue
        if math.isfinite(f):
            vals.append(f)

    if len(vals) < 2:
        return 0.0, 0.0, True

    vals.sort(reverse=True)
    m12 = vals[0] - vals[1]

    tail = vals[1:]
    mean = sum(tail) / len(tail)
    var = sum((v - mean) * (v - mean) for v in tail) / len(tail)
    band = math.sqrt(max(var, 0.0))

    if band <= 1e-6:
        return m12, 0.0, m12 > 0.0
    return m12, band, m12 >= band