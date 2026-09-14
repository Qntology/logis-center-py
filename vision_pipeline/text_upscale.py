import math
from typing import Optional, Tuple

import numpy as np
from PIL import Image

VISION_PATCH_PX = 28.0
MAX_UPSCALE = 4.0
MAX_UPSCALE_SPARSE = 2.0
MAX_SIDE_PX = 2048
VLM_MAX_SIDE_PX = 1024
VLM_MAX_PIXELS = 640_000

OVERLAP_LINE_UNITS = 1.4
OVERLAP_MIN_PX = 8.0
OVERLAP_MAX_RATIO = 0.20
GUTTER_SEARCH_UNITS = 1.2
GUTTER_INK_QUANTILE = 25.0

TILE_SIDE_BUDGET = float(MAX_SIDE_PX)
TILE_PIXEL_BUDGET = float(MAX_SIDE_PX) * float(MAX_SIDE_PX)
TILE_SAFETY = 0.92


def crop_region(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = bbox
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(image.width, int(x1))
    y1 = min(image.height, int(y1))
    if x1 <= x0 or y1 <= y0:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    return image.crop((x0, y0, x1, y1))


def estimate_text_height(image: Image.Image) -> Optional[float]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    if h < 24 or w < 16:
        return None

    profile = (255.0 - gray).mean(axis=1)
    profile = profile - float(profile.mean())
    energy = float(np.dot(profile, profile))
    if energy <= 1e-6:
        return None

    max_lag = min(int(h // 2), 160)
    if max_lag <= 6:
        return None

    corr = np.zeros((max_lag,), dtype=np.float64)
    for lag in range(1, max_lag):
        a = profile[: h - lag]
        b = profile[lag:]
        if a.size < 8:
            break
        na = float(np.dot(a, a))
        nb = float(np.dot(b, b))
        if na <= 1e-9 or nb <= 1e-9:
            continue
        corr[lag] = float(np.dot(a, b)) / math.sqrt(na * nb)

    start = 0
    for lag in range(1, max_lag):
        if corr[lag] <= 0.0:
            start = lag + 1
            break
    if start == 0 or start >= max_lag - 2:
        return None

    best_lag = 0
    best = 0.0
    for lag in range(start + 1, max_lag - 1):
        v = float(corr[lag])
        if v <= best:
            continue
        if v < corr[lag - 1] or v < corr[lag + 1]:
            continue
        best = v
        best_lag = lag

    if best_lag <= 0 or best < 0.20:
        return None

    return float(best_lag) * 0.6


def text_aware_upscale(
    image: Image.Image,
    target_px: float = VISION_PATCH_PX,
    log: Optional[list] = None,
    text_boxes: Optional[list] = None,
) -> Tuple[Image.Image, float, str]:
    th = None
    mode = ""

    if text_boxes:
        from .text_boxes import median_text_height
        bh = median_text_height(text_boxes)
        if bh > 2.0:
            th = float(bh)
            mode = "det-box"
            if log is not None:
                log.append(
                    f"    📏 검출 박스 {len(text_boxes)}개 중앙 높이 "
                    f"{th:.1f}px 를 글자 높이로 씁니다."
                )

    if th is None:
        th = estimate_text_height(image)
        if th is not None and th > 0.5:
            mode = "line-pitch"
            if log is not None:
                log.append(
                    f"    📏 추정 글자 높이 {th:.1f}px "
                    f"(자기상관) → 배율 계산"
                )

    if th is not None and th > 0.5:
        factor = float(np.clip(target_px / th, 1.0, MAX_UPSCALE))
        if log is not None:
            log.append(
                f"    📏 배율 {factor:.2f}x (목표 {int(target_px)}px / "
                f"모드 {mode})"
            )
    else:
        short = float(min(image.width, image.height))
        factor = float(np.clip(target_px * 6.0 / max(1.0, short), 1.4, MAX_UPSCALE_SPARSE))
        mode = "floor"
        if log is not None:
            log.append(
                f"    📏 글자 높이를 못 구해 최소 배율 {factor:.2f}x 를 "
                f"강제합니다 (1.0x 로 두면 저해상도 그대로 전달됩니다)."
            )

    if factor <= 1.01:
        return image, 1.0, mode

    nw = min(MAX_SIDE_PX, max(1, int(round(image.width * factor))))
    nh = min(MAX_SIDE_PX, max(1, int(round(image.height * factor))))
    return image.resize((nw, nh), Image.LANCZOS), factor, mode


def prepare_crop(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_px: float = VISION_PATCH_PX,
    log: Optional[list] = None,
    text_boxes: Optional[list] = None,
) -> Tuple[Image.Image, float, str]:
    cropped = crop_region(image, bbox)

    local = None
    if text_boxes:
        bx0, by0, bx1, by1 = (int(v) for v in bbox)
        local = [
            (b[0] - bx0, b[1] - by0, b[2] - bx0, b[3] - by0)
            for b in text_boxes
            if b[0] < bx1 and b[2] > bx0 and b[1] < by1 and b[3] > by0
        ]

    return text_aware_upscale(
        cropped, target_px=target_px, log=log, text_boxes=local
    )


def fit_for_vlm(
    image: Image.Image,
    max_side: int = VLM_MAX_SIDE_PX,
    max_pixels: int = VLM_MAX_PIXELS,
    log: Optional[list] = None,
) -> Image.Image:
    w, h = image.size
    if w <= 0 or h <= 0:
        return image

    scale = 1.0
    longest = max(w, h)
    if longest > max_side:
        scale = float(max_side) / float(longest)
    if w * h * scale * scale > max_pixels:
        scale = min(scale, (float(max_pixels) / float(w * h)) ** 0.5)

    if scale >= 0.999:
        return image

    nw = max(16, int(round(w * scale)))
    nh = max(16, int(round(h * scale)))
    if log is not None:
        log.append(
            f"    📉 [VLM FIT] {w}x{h} → {nw}x{nh} "
            f"(장변 {max_side}px / 화소 {max_pixels // 1000}K 상한)"
        )
    return image.resize((nw, nh), Image.LANCZOS)


def _line_height_for(
    image: Optional[Image.Image],
    bbox: Tuple[int, int, int, int],
    text_boxes: Optional[list] = None,
) -> Tuple[float, str]:
    if text_boxes:
        x0, y0, x1, y1 = (int(v) for v in bbox)
        inside = [
            b for b in text_boxes
            if b[0] < x1 and b[2] > x0 and b[1] < y1 and b[3] > y0
        ]
        if inside:
            try:
                from .text_boxes import median_text_height
                bh = float(median_text_height(inside))
            except Exception:
                bh = 0.0
            if bh > 2.0:
                return bh, "det-box"

    if image is not None:
        th = estimate_text_height(crop_region(image, bbox))
        if th is not None and th > 0.5:
            return float(th) / 0.6, "line-pitch"

    return 0.0, ""


def _row_gutters(
    image: Optional[Image.Image],
    bbox: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    if image is None:
        return None
    crop = crop_region(image, bbox)
    if crop.height < 24 or crop.width < 16:
        return None
    gray = np.asarray(crop.convert("L"), dtype=np.float32)
    ink = 255.0 - gray
    prof = ink.mean(axis=1)
    if prof.size < 8:
        return None
    return prof


def _snap_to_gutter(
    prof: Optional[np.ndarray],
    y_local: float,
    unit: float,
    y_min: float,
    y_max: float,
) -> float:
    if prof is None or unit <= 1.0:
        return y_local

    span = int(max(2.0, unit * GUTTER_SEARCH_UNITS))
    lo = int(max(y_min, y_local - span))
    hi = int(min(y_max, y_local + span))
    if hi <= lo + 1:
        return y_local

    seg = prof[lo:hi]
    if seg.size < 2:
        return y_local

    gate = float(np.percentile(prof, GUTTER_INK_QUANTILE))
    quiet = np.nonzero(seg <= gate)[0]
    if quiet.size == 0:
        return float(lo + int(np.argmin(seg)))

    target = y_local - lo
    best = quiet[int(np.argmin(np.abs(quiet - target)))]
    return float(lo + int(best))


def plan_overlap_tiles(
    bbox: Tuple[int, int, int, int],
    tile_count: int,
    overlap_ratio: float = 0.25,
    image: Optional[Image.Image] = None,
    text_boxes: Optional[list] = None,
    log: Optional[list] = None,
    category: str = "",
) -> list:
    x0, y0, x1, y1 = bbox
    if tile_count <= 1 or y1 <= y0:
        return [tuple(bbox)]

    h = float(y1 - y0)
    n = float(tile_count)

    unit, src = _line_height_for(image, bbox, text_boxes)

    if unit > 1.0:
        overlap_px = max(OVERLAP_MIN_PX, unit * OVERLAP_LINE_UNITS)
        overlap_px = min(overlap_px, h * OVERLAP_MAX_RATIO)
        eff_ratio = overlap_px / max(1.0, h / n)
        eff_ratio = float(np.clip(eff_ratio, 0.0, OVERLAP_MAX_RATIO))
        mode = f"글자높이 {unit:.0f}px×{OVERLAP_LINE_UNITS:.1f} ({src})"
    else:
        eff_ratio = float(max(0.0, min(overlap_ratio, OVERLAP_MAX_RATIO)))
        overlap_px = (h / n) * eff_ratio
        mode = "고정 비율 (글자 높이 미검출)"

    denom = n - (n - 1.0) * eff_ratio
    if denom <= 0:
        return [tuple(bbox)]

    t = h / denom
    step = t * (1.0 - eff_ratio)

    prof = _row_gutters(image, bbox) if unit > 1.0 else None
    snapped = 0

    out = []
    for i in range(tile_count):
        ty0 = float(y0) + step * i
        ty1 = min(ty0 + t, float(y1))

        if prof is not None and i > 0:
            moved = _snap_to_gutter(
                prof, ty0 - y0, unit, 0.0, h - 1.0
            ) + y0
            if abs(moved - ty0) >= 1.0:
                snapped += 1
            ty0 = moved
        if prof is not None and i + 1 < tile_count:
            moved = _snap_to_gutter(
                prof, ty1 - y0, unit, 0.0, h - 1.0
            ) + y0
            if abs(moved - ty1) >= 1.0:
                snapped += 1
            ty1 = moved

        if ty1 <= ty0 + 1.0:
            continue
        out.append((x0, int(ty0), x1, int(ty1)))

    if log is not None:
        tail = (
            f" | 경계 {snapped}곳을 행 사이 여백에 스냅"
            if snapped else ""
        )
        log.append(
            f"    📐 [TILE OVERLAP] '{category}' 겹침 {overlap_px:.0f}px "
            f"({eff_ratio:.0%}) — {mode}{tail}"
        )

    return out if out else [tuple(bbox)]


def decide_tile_count(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    max_tiles: int = 3,
    table_rows: int = 0,
    legible: int = 0,
    patches: int = 0,
    text_boxes: Optional[list] = None,
) -> Tuple[int, str]:
    x0, y0, x1, y1 = bbox
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)

    unit, src = _line_height_for(image, bbox, text_boxes)
    if unit > 0.5:
        factor = float(np.clip(VISION_PATCH_PX / unit, 1.0, MAX_UPSCALE))
        basis = f"글자높이 {unit:.0f}px ({src})"
    else:
        short = float(min(width, height))
        factor = float(np.clip(
            VISION_PATCH_PX * 6.0 / max(1.0, short), 1.4, MAX_UPSCALE_SPARSE
        ))
        basis = "글자높이 미검출"

    up_w = float(width) * factor
    up_h = float(height) * factor

    side_need = up_h / (TILE_SIDE_BUDGET * TILE_SAFETY)
    px_need = (up_w * up_h) / (TILE_PIXEL_BUDGET * TILE_SAFETY)
    need = max(side_need, px_need)

    if need <= 1.0:
        return 1, (
            f"업스케일 {factor:.2f}x 후 {int(up_w)}x{int(up_h)} 가 "
            f"{int(TILE_SIDE_BUDGET)}px 예산 이내 ({basis}) — 나눌 이유 없음"
        )

    tiles = int(np.clip(math.ceil(need), 1, max_tiles))
    return tiles, (
        f"업스케일 {factor:.2f}x 후 {int(up_w)}x{int(up_h)} 가 예산 초과 "
        f"(필요 {need:.2f}배 / {basis})"
    )