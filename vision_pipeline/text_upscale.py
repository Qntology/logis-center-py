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
) -> Tuple[Image.Image, float, str]:
    th = estimate_text_height(image)

    if th is not None and th > 0.5:
        factor = float(np.clip(target_px / th, 1.0, MAX_UPSCALE))
        mode = "line-pitch"
        if log is not None:
            log.append(
                f"    📏 추정 글자 높이 {th:.1f}px → 배율 {factor:.2f}x "
                f"(목표 {int(target_px)}px)"
            )
    else:
        short = float(min(image.width, image.height))
        factor = float(np.clip(target_px * 4.0 / max(1.0, short), 1.0, MAX_UPSCALE_SPARSE))
        mode = "conservative"
        if log is not None:
            log.append(f"    📏 라인 주기 미검출 → 보수적 배율 {factor:.2f}x")

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
) -> Tuple[Image.Image, float, str]:
    cropped = crop_region(image, bbox)
    return text_aware_upscale(cropped, target_px=target_px, log=log)


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


def plan_overlap_tiles(
    bbox: Tuple[int, int, int, int],
    tile_count: int,
    overlap_ratio: float = 0.25,
) -> list:
    x0, y0, x1, y1 = bbox
    if tile_count <= 1 or y1 <= y0:
        return [tuple(bbox)]

    h = float(y1 - y0)
    n = float(tile_count)
    denom = n - (n - 1.0) * overlap_ratio
    if denom <= 0:
        return [tuple(bbox)]

    t = h / denom
    step = t * (1.0 - overlap_ratio)

    out = []
    for i in range(tile_count):
        ty0 = y0 + step * i
        ty1 = min(ty0 + t, float(y1))
        if ty1 <= ty0 + 1.0:
            continue
        out.append((x0, int(ty0), x1, int(ty1)))

    return out if out else [tuple(bbox)]


def decide_tile_count(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    max_tiles: int = 3,
    table_rows: int = 0,
    legible: int = 0,
    patches: int = 0,
) -> Tuple[int, str]:
    if int(table_rows) > 0:
        tiles = int(np.clip(round(int(table_rows) / 3.0), 1, max_tiles))
        return tiles, "표행밀도"

    x0, y0, x1, y1 = bbox
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)

    if patches > 0 and legible * 4 < patches:
        return 1, "내용희소"

    cropped = crop_region(image, bbox)
    th = estimate_text_height(cropped)
    if th is None or th <= 0.5:
        return 1, "라인 주기 미검출"

    est_lines = height / max(1.0, th / 0.6)
    if est_lines <= 3.0:
        return 1, "행수 부족"
    if height < width:
        return 1, "가로형 밴드"

    return int(np.clip(round(est_lines / 3.0), 1, max_tiles)), "행수밀도"