from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


class VisionPatchGrid:
    def __init__(
        self,
        embeddings: np.ndarray,
        rows: int,
        cols: int,
        orig_width: int,
        orig_height: int,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        patch_size: int = 16,
        source: str = "",
    ):
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.rows = int(rows)
        self.cols = int(cols)
        self.orig_width = int(orig_width)
        self.orig_height = int(orig_height)
        self.scale_x = float(scale_x)
        self.scale_y = float(scale_y)
        self.patch_size = int(patch_size)
        self.source = source
        self.num_patches = self.rows * self.cols

    @property
    def dim(self) -> int:
        if self.embeddings.size == 0:
            return 0
        return int(self.embeddings.shape[-1])

    def index_of(self, row: int, col: int) -> int:
        return int(row) * self.cols + int(col)

    def row_col(self, index: int) -> Tuple[int, int]:
        cols = max(1, self.cols)
        return int(index) // cols, int(index) % cols

    def vector(self, index: int) -> np.ndarray:
        return self.embeddings[int(index)]

    def cell_width(self) -> float:
        return self.orig_width / float(max(1, self.cols))

    def cell_height(self) -> float:
        return self.orig_height / float(max(1, self.rows))

    def patch_bbox(self, row: int, col: int) -> Tuple[int, int, int, int]:
        cw = self.cell_width()
        ch = self.cell_height()
        x0 = int(col * cw)
        y0 = int(row * ch)
        x1 = int((col + 1) * cw)
        y1 = int((row + 1) * ch)
        x0 = max(0, min(x0, self.orig_width - 1))
        y0 = max(0, min(y0, self.orig_height - 1))
        x1 = min(max(x1, x0 + 1), self.orig_width)
        y1 = min(max(y1, y0 + 1), self.orig_height)
        return (x0, y0, x1, y1)

    def index_bbox(self, index: int) -> Tuple[int, int, int, int]:
        r, c = self.row_col(index)
        return self.patch_bbox(r, c)

    def region_bbox(self, r0: int, r1: int, c0: int, c1: int) -> Tuple[int, int, int, int]:
        r0 = max(0, min(int(r0), self.rows - 1))
        r1 = max(0, min(int(r1), self.rows - 1))
        c0 = max(0, min(int(c0), self.cols - 1))
        c1 = max(0, min(int(c1), self.cols - 1))
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0
        tx0, ty0, _, _ = self.patch_bbox(r0, c0)
        _, _, bx1, by1 = self.patch_bbox(r1, c1)
        return (tx0, ty0, max(bx1, tx0 + 1), max(by1, ty0 + 1))

    def patch_center(self, index: int) -> Tuple[float, float]:
        r, c = self.row_col(index)
        cw = self.cell_width()
        ch = self.cell_height()
        return ((c + 0.5) * cw, (r + 0.5) * ch)

    def indices_in_bbox(self, bbox: Tuple[int, int, int, int]) -> List[int]:
        x0, y0, x1, y1 = bbox
        out: List[int] = []
        for i in range(self.num_patches):
            cx, cy = self.patch_center(i)
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                out.append(i)
        return out

    def to_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "num_patches": self.num_patches,
            "dim": self.dim,
            "patch_size": self.patch_size,
            "orig_width": self.orig_width,
            "orig_height": self.orig_height,
            "scale_x": round(self.scale_x, 6),
            "scale_y": round(self.scale_y, 6),
            "source": self.source,
        }

    def __repr__(self) -> str:
        return (
            f"<VisionPatchGrid {self.rows}x{self.cols}={self.num_patches} "
            f"dim={self.dim} src={self.source}>"
        )


def _l2_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    return mat / np.maximum(norms, eps)


def _resample_to_grid(feats: np.ndarray, target: int) -> np.ndarray:
    n = feats.shape[0]
    if n == target:
        return feats
    if n == 0:
        return np.zeros((target, 1), dtype=np.float32)
    idx = np.linspace(0, n - 1, target)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = (idx - lo).reshape(-1, 1).astype(np.float32)
    return feats[lo] * (1.0 - w) + feats[hi] * w


def build_patch_grid(
    image: Image.Image,
    embedder=None,
    ocr=None,
    prefer: str = "ocr",
    align_to: Optional[VisionPatchGrid] = None,
    joint=None,
) -> VisionPatchGrid:
    orig_w, orig_h = image.size

    if joint is not None and prefer in ("joint", "siglip", "siglip2"):
        out = joint.embed_image_patches(image)
        feats = np.asarray(out["valid_features"], dtype=np.float32)
        rows = int(out["rows"])
        cols = int(out["cols"])
        expected = max(1, rows * cols)
        if feats.shape[0] != expected:
            feats = _resample_to_grid(feats, expected)
        feats = _l2_rows(feats)
        return VisionPatchGrid(
            embeddings=feats,
            rows=rows,
            cols=cols,
            orig_width=orig_w,
            orig_height=orig_h,
            patch_size=getattr(joint, "patch_size", 16),
            source="siglip2",
        )

    if prefer == "ocr" and ocr is not None:
        out = ocr.embed_image_patches(image)
        feats = np.asarray(out["valid_features"], dtype=np.float32)
        rows = int(out["rows"])
        cols = int(out["cols"])
        expected = max(1, rows * cols)
        if feats.shape[0] != expected:
            feats = _resample_to_grid(feats, expected)
        feats = _l2_rows(feats)
        return VisionPatchGrid(
            embeddings=feats,
            rows=rows,
            cols=cols,
            orig_width=orig_w,
            orig_height=orig_h,
            patch_size=getattr(ocr, "patch_size", 16),
            source="hayai",
        )

    if embedder is not None:
        pg = embedder.embed_image(image)
        feats = np.asarray(pg.embeddings, dtype=np.float32)
        grid = VisionPatchGrid(
            embeddings=feats,
            rows=pg.grid_rows,
            cols=pg.grid_cols,
            orig_width=pg.orig_width,
            orig_height=pg.orig_height,
            scale_x=pg.scale_x,
            scale_y=pg.scale_y,
            patch_size=pg.patch_size,
            source="ax-ve",
        )
        if align_to is not None and grid.num_patches != align_to.num_patches:
            grid.embeddings = _l2_rows(
                _resample_to_grid(grid.embeddings, align_to.num_patches)
            )
            grid.rows = align_to.rows
            grid.cols = align_to.cols
            grid.num_patches = align_to.num_patches
        return grid

    if ocr is not None:
        out = ocr.embed_image_patches(image)
        feats = np.asarray(out["valid_features"], dtype=np.float32)
        rows = int(out["rows"])
        cols = int(out["cols"])
        expected = max(1, rows * cols)
        if feats.shape[0] != expected:
            feats = _resample_to_grid(feats, expected)
        feats = _l2_rows(feats)
        return VisionPatchGrid(
            embeddings=feats,
            rows=rows,
            cols=cols,
            orig_width=orig_w,
            orig_height=orig_h,
            patch_size=getattr(ocr, "patch_size", 16),
            source="hayai",
        )

    raise RuntimeError(
        "패치 격자를 만들 수 있는 모델이 없습니다. (joint / embedder / ocr 전부 None)"
    )


def row_band_profile(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ink = 255.0 - gray
    return ink.mean(axis=1)


def ink_ratio_per_row(image: Image.Image, rows: int) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h = gray.shape[0]
    if h == 0 or rows <= 0:
        return np.zeros((max(1, rows),), dtype=np.float32)
    ink = 255.0 - gray
    thr = float(ink.mean() + ink.std() * 0.25)
    out = np.zeros((rows,), dtype=np.float32)
    step = h / float(rows)
    for r in range(rows):
        y0 = int(r * step)
        y1 = max(y0 + 1, int((r + 1) * step))
        seg = ink[y0:min(y1, h), :]
        if seg.size == 0:
            continue
        out[r] = float((seg > thr).mean())
    return out


def ink_ratio_per_col(image: Image.Image, cols: int) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    w = gray.shape[1]
    if w == 0 or cols <= 0:
        return np.zeros((max(1, cols),), dtype=np.float32)
    ink = 255.0 - gray
    thr = float(ink.mean() + ink.std() * 0.25)
    out = np.zeros((cols,), dtype=np.float32)
    step = w / float(cols)
    for c in range(cols):
        x0 = int(c * step)
        x1 = max(x0 + 1, int((c + 1) * step))
        seg = ink[:, x0:min(x1, w)]
        if seg.size == 0:
            continue
        out[c] = float((seg > thr).mean())
    return out


def _gutter_indices(
    profile: np.ndarray,
    rel_floor: float = 0.30,
    hard_floor: float = 0.020,
    max_ratio: float = 0.45,
) -> set:
    if profile.size == 0:
        return set()

    active = profile[profile > hard_floor]
    if active.size == 0:
        return set()

    gate = max(hard_floor, float(active.mean()) * float(rel_floor))
    cand = [int(i) for i in range(profile.size) if float(profile[i]) <= gate]

    limit = max(0, int(profile.size * max_ratio))
    if len(cand) > limit:
        cand.sort(key=lambda i: float(profile[i]))
        cand = cand[:limit]

    return set(cand)


def column_gutters(
    image: Image.Image,
    cols: int,
    rel_floor: float = 0.30,
    hard_floor: float = 0.020,
) -> set:
    return _gutter_indices(
        ink_ratio_per_col(image, cols),
        rel_floor=rel_floor,
        hard_floor=hard_floor,
    )


def row_gutters(
    image: Image.Image,
    rows: int,
    rel_floor: float = 0.22,
    hard_floor: float = 0.015,
) -> set:
    return _gutter_indices(
        ink_ratio_per_row(image, rows),
        rel_floor=rel_floor,
        hard_floor=hard_floor,
        max_ratio=0.35,
    )


def _otsu(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float(v.mean()) if v.size else 0.0
    lo = float(v.min())
    hi = float(v.max())
    if hi - lo < 1e-9:
        return lo
    hist, edges = np.histogram(v, bins=64, range=(lo, hi))
    total = float(hist.sum())
    if total <= 0:
        return lo
    centers = (edges[:-1] + edges[1:]) * 0.5
    w = np.cumsum(hist).astype(np.float64)
    m = np.cumsum(hist.astype(np.float64) * centers)
    mt = float(m[-1])
    denom = w * (total - w)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w / total - m) ** 2 / denom
    if not np.any(np.isfinite(between)):
        return lo
    between = np.where(np.isfinite(between), between, -np.inf)
    return float(centers[int(np.argmax(between))])


class LegibilityMap:
    def __init__(
        self,
        rows: int,
        cols: int,
        ink: np.ndarray,
        sharp: np.ndarray,
        ink_gate: float,
        sharp_gate: float,
    ):
        self.rows = int(rows)
        self.cols = int(cols)
        self.ink = np.asarray(ink, dtype=np.float32)
        self.sharp = np.asarray(sharp, dtype=np.float32)
        self.ink_gate = float(ink_gate)
        self.sharp_gate = float(sharp_gate)

        self.blank = self.ink <= self.ink_gate
        inked = ~self.blank
        self.legible = inked & (self.sharp > self.sharp_gate)
        self.illegible = inked & ~self.legible

    @property
    def size(self) -> int:
        return int(self.ink.size)

    def counts(self) -> Tuple[int, int, int]:
        return (
            int(self.blank.sum()),
            int(self.illegible.sum()),
            int(self.legible.sum()),
        )

    def row_legible(self) -> np.ndarray:
        m = self.legible.reshape(self.rows, self.cols)
        return m.sum(axis=1).astype(np.int32)

    def row_inked(self) -> np.ndarray:
        m = (~self.blank).reshape(self.rows, self.cols)
        return m.sum(axis=1).astype(np.int32)

    def report(self) -> str:
        b, i, l = self.counts()
        ok = "분리 성공" if (l > 0 and b > 0) else "분리 실패"
        return (
            f"  🔎 [LEGIBILITY] 여백 {b} | 판독불가 {i} | 판독가능 {l} / "
            f"{self.size} | ink_gate {self.ink_gate:.2f} "
            f"| sharp_gate {self.sharp_gate:.3f} | {ok}"
        )


def legibility_map(
    image: Image.Image,
    rows: int,
    cols: int,
) -> LegibilityMap:
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    ink_img = 255.0 - gray

    ink = np.zeros((rows * cols,), dtype=np.float32)
    sharp = np.zeros((rows * cols,), dtype=np.float32)

    ystep = h / float(rows)
    xstep = w / float(cols)

    for r in range(rows):
        y0 = int(r * ystep)
        y1 = max(y0 + 1, int((r + 1) * ystep))
        for c in range(cols):
            x0 = int(c * xstep)
            x1 = max(x0 + 1, int((c + 1) * xstep))
            seg = ink_img[y0:min(y1, h), x0:min(x1, w)]
            idx = r * cols + c
            if seg.size == 0:
                continue
            ink[idx] = float(seg.mean())
            if seg.shape[0] >= 2 and seg.shape[1] >= 2:
                gx = float(np.abs(np.diff(seg, axis=1)).mean())
                gy = float(np.abs(np.diff(seg, axis=0)).mean())
                sharp[idx] = float((gx + gy) * 0.5 / 255.0)

    ink_gate = _otsu(ink)
    inked = ink > ink_gate
    sharp_gate = _otsu(sharp[inked]) if int(inked.sum()) >= 2 else 0.0

    return LegibilityMap(rows, cols, ink, sharp, ink_gate, sharp_gate)


def table_row_band(
    legibility: LegibilityMap,
    min_span: int = 2,
) -> Tuple[int, int, int]:
    inked = legibility.row_inked()
    if inked.size == 0:
        return -1, -1, 0
    active = inked[inked > 0]
    if active.size < 2:
        return -1, -1, 0

    gate = _otsu(active.astype(np.float64))
    dense = inked >= max(gate, float(np.median(active)))

    best = (-1, -1, 0)
    start = -1
    for r in range(int(inked.size)):
        if dense[r] and start < 0:
            start = r
        elif not dense[r] and start >= 0:
            span = r - start
            if span > best[2]:
                best = (start, r - 1, span)
            start = -1
    if start >= 0:
        span = int(inked.size) - start
        if span > best[2]:
            best = (start, int(inked.size) - 1, span)

    if best[2] < int(min_span):
        return -1, -1, 0
    return best