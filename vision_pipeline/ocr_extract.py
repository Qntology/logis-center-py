from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .text_upscale import decide_tile_count, plan_overlap_tiles, prepare_crop
from .vision_nms import CropPlan


class ExtractedField:
    def __init__(
        self,
        category: str,
        bbox: Tuple[int, int, int, int],
        score: float,
        margin: float,
        ocr_text: str = "",
        refined: Optional[dict] = None,
        upscale: float = 1.0,
        upscale_mode: str = "",
        tiles: int = 1,
        crop_image: Optional[Image.Image] = None,
    ):
        self.category = category
        self.bbox = tuple(int(v) for v in bbox)
        self.score = float(score)
        self.margin = float(margin)
        self.ocr_text = ocr_text
        self.refined = refined or {}
        self.upscale = float(upscale)
        self.upscale_mode = upscale_mode
        self.tiles = int(tiles)
        self.crop_image = crop_image
        self.ocr_raw = ocr_text
        self.lemma = ""
        self.nlp_meta: dict = {}

    @property
    def value(self) -> str:
        if self.refined:
            for key in ("value", "text", self.category):
                v = self.refined.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
        return self.ocr_text.strip()

    def to_dict(self, include_image: bool = False) -> dict:
        out = {
            "category": self.category,
            "field_name": self.category,
            "bbox": list(self.bbox),
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "ocr_text": self.ocr_text,
            "ocr_raw": self.ocr_raw,
            "lemma": self.lemma,
            "nlp": self.nlp_meta,
            "value": self.value,
            "refined": self.refined,
            "upscale": round(self.upscale, 3),
            "upscale_mode": self.upscale_mode,
            "tiles": self.tiles,
        }
        if include_image and self.crop_image is not None:
            import base64
            import io
            buf = io.BytesIO()
            try:
                self.crop_image.save(buf, format="PNG")
                out["crop_base64"] = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                out["crop_base64"] = ""
        return out

    def __repr__(self) -> str:
        preview = self.value[:40].replace("\n", " ")
        return f"<Extracted {self.category} '{preview}' score={self.score:+.4f}>"


def _ocr_tiles(
    ocr,
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_px: float,
    max_tiles: int,
    log: Optional[List[str]] = None,
) -> Tuple[str, float, str, int, Optional[Image.Image]]:
    tiles = decide_tile_count(image, bbox, max_tiles=max_tiles)

    if tiles <= 1:
        crop, factor, mode = prepare_crop(image, bbox, target_px=target_px, log=log)
        try:
            text = ocr.ocr_image(crop)
        except Exception as e:
            if log is not None:
                log.append(f"    ⚠ OCR 실패: {e}")
            text = ""
        return text, factor, mode, 1, crop

    if log is not None:
        log.append(f"    🧱 {tiles}타일 분할 (겹침 25%)")

    boxes = plan_overlap_tiles(bbox, tiles, overlap_ratio=0.25)
    chunks: List[str] = []
    factor = 1.0
    mode = ""
    first_crop: Optional[Image.Image] = None

    for i, tb in enumerate(boxes):
        crop, f, m = prepare_crop(image, tb, target_px=target_px, log=None)
        if i == 0:
            first_crop = crop
            factor, mode = f, m
        try:
            txt = ocr.ocr_image(crop)
        except Exception:
            txt = ""
        txt = (txt or "").strip()
        if txt and txt not in chunks:
            chunks.append(txt)

    return "\n".join(chunks), factor, mode, len(boxes), first_crop


def nlp_clean_ocr(
    text: str,
    nlp=None,
    log: Optional[List[str]] = None,
    min_content_ratio: float = 0.20,
) -> Tuple[str, dict]:
    raw = (text or "").strip()
    meta = {"lines_in": 0, "lines_out": 0, "dropped": 0, "lemma": ""}

    if not raw:
        return raw, meta

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    meta["lines_in"] = len(lines)

    if nlp is None or not getattr(nlp, "available", False):
        meta["lines_out"] = len(lines)
        return "\n".join(lines), meta

    kept: List[str] = []
    for ln in lines:
        verdict = nlp.analyze(ln)
        if verdict is None:
            kept.append(ln)
            continue
        if verdict.is_entity or verdict.content_ratio >= min_content_ratio:
            kept.append(ln)
            continue
        meta["dropped"] += 1
        if log is not None:
            log.append(f"    🧬 NLP 노이즈 제거 '{ln[:40]}' ({verdict.reason})")

    if not kept:
        kept = lines
        meta["dropped"] = 0

    meta["lines_out"] = len(kept)
    cleaned = "\n".join(kept)

    try:
        meta["lemma"] = nlp.canonicalize(cleaned)
    except Exception:
        meta["lemma"] = ""

    return cleaned, meta


def extract_from_crops(
    image: Image.Image,
    plans: Sequence[CropPlan],
    ocr,
    refine_fn: Optional[Callable[[str, str, Image.Image], dict]] = None,
    target_px: float = 28.0,
    max_tiles: int = 3,
    keep_crop: bool = True,
    nlp=None,
    log: Optional[List[str]] = None,
) -> List[ExtractedField]:
    out: List[ExtractedField] = []

    for plan in plans:
        if log is not None:
            log.append(f"  ✂️ '{plan.category}' 크롭 px{plan.bbox}")

        text, factor, mode, tiles, crop = _ocr_tiles(
            ocr, image, plan.bbox, target_px, max_tiles, log=log
        )

        cleaned, nlp_meta = nlp_clean_ocr(text, nlp=nlp, log=log)
        if nlp_meta["dropped"] and log is not None:
            log.append(
                f"    🧬 NLP 게이트: {nlp_meta['lines_in']} → "
                f"{nlp_meta['lines_out']}줄 ({nlp_meta['dropped']}줄 제거)"
            )

        refined = {}
        if refine_fn is not None and cleaned.strip():
            try:
                refined = refine_fn(plan.category, cleaned, crop) or {}
            except Exception as e:
                if log is not None:
                    log.append(f"    ⚠ 정제 추출 실패: {e}")

        field = ExtractedField(
            category=plan.category,
            bbox=plan.bbox,
            score=plan.score,
            margin=plan.margin,
            ocr_text=cleaned,
            refined=refined,
            upscale=factor,
            upscale_mode=mode,
            tiles=tiles,
            crop_image=crop if keep_crop else None,
        )
        field.ocr_raw = text
        field.lemma = nlp_meta.get("lemma", "")
        field.nlp_meta = nlp_meta
        out.append(field)

        if log is not None:
            preview = field.value[:60].replace("\n", " / ")
            log.append(f"    📝 '{plan.category}' = {preview or '(공백)'}")

    return out


def fields_to_record(
    fields: Sequence[ExtractedField],
    schema: Optional[dict] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {}
    for f in fields:
        val = f.value
        if not val:
            continue
        if f.category in record:
            prev = record[f.category]
            if isinstance(prev, list):
                prev.append(val)
            else:
                record[f.category] = [prev, val]
        else:
            record[f.category] = val

    if schema:
        for name in (schema.get("fields", {}) or {}).keys():
            record.setdefault(name, None)
    return record