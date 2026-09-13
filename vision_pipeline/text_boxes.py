from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

MIN_BOX_W = 6
MIN_BOX_H = 6
MAX_BOXES = 400
DILATE_X = 3
DILATE_Y = 1
MERGE_GAP_X = 12
MERGE_GAP_Y = 4


def _binarize(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    ink = 255.0 - gray
    mu = float(ink.mean())
    sd = float(ink.std())
    gate = mu + sd * 0.55
    return ink > max(gate, 12.0)


def _dilate(mask: np.ndarray, rx: int, ry: int) -> np.ndarray:
    out = mask.copy()
    for dx in range(1, int(rx) + 1):
        out[:, dx:] |= mask[:, :-dx]
        out[:, :-dx] |= mask[:, dx:]
    base = out.copy()
    for dy in range(1, int(ry) + 1):
        out[dy:, :] |= base[:-dy, :]
        out[:-dy, :] |= base[dy:, :]
    return out


def _components(mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: List[Tuple[int, int, int, int]] = []

    ys_all, xs_all = np.nonzero(mask)
    for k in range(ys_all.size):
        sy = int(ys_all[k])
        sx = int(xs_all[k])
        if visited[sy, sx]:
            continue
        stack = [(sy, sx)]
        visited[sy, sx] = True
        y0 = y1 = sy
        x0 = x1 = sx
        while stack:
            cy, cx = stack.pop()
            if cy < y0:
                y0 = cy
            if cy > y1:
                y1 = cy
            if cx < x0:
                x0 = cx
            if cx > x1:
                x1 = cx
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = cy + dy, cx + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w:
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))
        if (x1 - x0 + 1) < MIN_BOX_W or (y1 - y0 + 1) < MIN_BOX_H:
            continue
        out.append((x0, y0, x1 + 1, y1 + 1))
        if len(out) >= MAX_BOXES:
            break
    return out


def _merge(
    boxes: Sequence[Tuple[int, int, int, int]],
    max_h: int = 0,
) -> List[Tuple[int, int, int, int]]:
    items = sorted(boxes, key=lambda b: (b[1], b[0]))
    out: List[Tuple[int, int, int, int]] = []
    cap = int(max_h) if max_h and max_h > 0 else 0

    for b in items:
        hit = -1
        for i, a in enumerate(out):
            vy = min(a[3], b[3]) - max(a[1], b[1])
            gx = max(a[0], b[0]) - min(a[2], b[2])
            if vy <= 0 or gx > MERGE_GAP_X:
                continue
            merged_h = max(a[3], b[3]) - min(a[1], b[1])
            if cap and merged_h > cap:
                continue
            hit = i
            break

        if hit < 0:
            out.append(tuple(b))
            continue

        a = out[hit]
        out[hit] = (
            min(a[0], b[0]), min(a[1], b[1]),
            max(a[2], b[2]), max(a[3], b[3]),
        )
    return out


def detect_text_boxes(
    image: Image.Image,
    ocr=None,
    log: Optional[List[str]] = None,
) -> List[Tuple[int, int, int, int]]:
    if image is None or image.width < 8 or image.height < 8:
        return []

    if ocr is not None and hasattr(ocr, "detect_boxes"):
        try:
            boxes = ocr.detect_boxes(image)
        except Exception:
            boxes = []
        if boxes:
            if log is not None:
                log.append(
                    f"  🔍 [TEXT BOXES] PP-OCRv5 det 로 {len(boxes)}개 검출"
                )
            return [tuple(int(v) for v in b) for b in boxes]

    mask = _dilate(_binarize(image), DILATE_X, DILATE_Y)
    cap = max(24, int(image.height * 0.22))
    boxes = _merge(_components(mask), max_h=cap)

    kept: List[Tuple[int, int, int, int]] = []
    dropped = 0
    for b in boxes:
        bw = b[2] - b[0]
        bh = b[3] - b[1]
        if bw >= image.width * 0.95 and bh >= image.height * 0.80:
            dropped += 1
            continue
        kept.append(b)

    if log is not None:
        log.append(
            f"  🔍 [TEXT BOXES] 영상처리 폴백으로 {len(kept)}개 검출 "
            f"(적응 이진화 + 형태학 팽창 + 연결요소 | 행 높이 상한 {cap}px)"
        )
        if dropped:
            log.append(
                f"  🚫 [TEXT BOXES] 화면 전체를 덮는 박스 {dropped}개를 "
                f"버립니다 — 크롭 스냅이 원본 전체로 번지는 것을 막습니다."
            )
    return kept


def box_union(
    boxes: Sequence[Tuple[int, int, int, int]],
    region: Tuple[int, int, int, int],
    overlap_ratio: float = 0.20,
) -> Optional[Tuple[int, int, int, int]]:
    rx0, ry0, rx1, ry1 = (int(v) for v in region)
    hits: List[Tuple[int, int, int, int]] = []

    for (x0, y0, x1, y1) in boxes:
        ix0 = max(rx0, x0)
        iy0 = max(ry0, y0)
        ix1 = min(rx1, x1)
        iy1 = min(ry1, y1)
        if ix0 >= ix1 or iy0 >= iy1:
            continue
        inter = float((ix1 - ix0) * (iy1 - iy0))
        area = float(max(1, (x1 - x0) * (y1 - y0)))
        if inter / area >= float(overlap_ratio):
            hits.append((x0, y0, x1, y1))

    if not hits:
        return None

    return (
        min(b[0] for b in hits),
        min(b[1] for b in hits),
        max(b[2] for b in hits),
        max(b[3] for b in hits),
    )


def median_text_height(
    boxes: Sequence[Tuple[int, int, int, int]]
) -> float:
    hs = [float(b[3] - b[1]) for b in boxes if b[3] > b[1]]
    if not hs:
        return 0.0
    return float(np.median(np.asarray(hs, dtype=np.float32)))