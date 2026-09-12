from typing import Optional, Tuple

import numpy as np
from PIL import Image

VISION_PATCH_PX = 28.0
MAX_UPSCALE = 4.0
MAX_UPSCALE_SPARSE = 2.0
MAX_SIDE_PX = 2048


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
    if h < 16 or w < 16:
        return None

    profile = (255.0 - gray).mean(axis=1)
    profile = profile - float(profile.mean())

    max_lag = min(int(h // 2), 120)
    if max_lag <= 4:
        return None

    best_lag = 0
    best = -np.inf
    for lag in range(4, max_lag):
        seg_a = profile[: h - lag]
        seg_b = profile[lag:]
        if seg_a.size == 0:
            continue
        val = float(np.dot(seg_a, seg_b) / seg_a.size)
        if val > best:
            best = val
            best_lag = lag

    if best_lag == 0 or best <= 0.0:
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
) -> int:
    x0, y0, x1, y1 = bbox
    height = max(1, y1 - y0)
    width = max(1, x1 - x0)

    cropped = crop_region(image, bbox)
    th = estimate_text_height(cropped)
    if th is None or th <= 0.5:
        return 1

    est_lines = height / max(1.0, th / 0.6)
    if est_lines <= 3.0:
        return 1
    if height < width:
        return 1

    return int(np.clip(round(est_lines / 3.0), 1, max_tiles))