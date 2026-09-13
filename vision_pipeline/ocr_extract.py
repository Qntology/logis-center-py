from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .text_upscale import crop_region

LINE_READ_MIN_LINES = 2

from .text_upscale import (
    decide_tile_count,
    fit_for_vlm,
    plan_overlap_tiles,
    prepare_crop,
)
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
        self.field_values: Dict[str, str] = {}
        self.rows: List[dict] = []
        self.line_texts: List[str] = []
        self.line_windows: int = 0

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
            "lines": list(self.line_texts),
            "windows": self.line_windows,
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
    category: str = "",
    table_rows: int = 0,
    legible: int = 0,
    patches: int = 0,
    text_boxes: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[str, float, str, int, Optional[Image.Image]]:
    tiles, reason = decide_tile_count(
        image, bbox, max_tiles=max_tiles,
        table_rows=table_rows, legible=legible, patches=patches,
        text_boxes=text_boxes,
    )

    if log is not None:
        log.append(
            f"    🧱 [TILE PLAN] '{category}' → {tiles}타일 (겹침 25%) "
            f"| 사유: {reason} | 표행 {table_rows} "
            f"| 판독가능 {legible}/{patches}"
        )

    usable = ocr is not None and getattr(ocr, "available", False)

    if tiles <= 1:
        crop, factor, mode = prepare_crop(
            image, bbox, target_px=target_px, log=log, text_boxes=text_boxes
        )
        text = ""
        if usable:
            try:
                text = ocr.ocr_image(crop)
            except Exception as e:
                if log is not None:
                    log.append(f"    ⚠ OCR 실패: {e}")
        elif log is not None:
            log.append(
                f"    ⏭ [{category}] PP-OCRv5 rec 미가동 — OCR 초안 없이 "
                f"VLM 판독에 맡깁니다."
            )
        return text, factor, mode, 1, crop

    boxes = plan_overlap_tiles(bbox, tiles, overlap_ratio=0.25)
    chunks: List[str] = []
    factor = 1.0
    mode = ""
    first_crop: Optional[Image.Image] = None

    for i, tb in enumerate(boxes):
        crop, f, m = prepare_crop(
            image, tb, target_px=target_px, log=None, text_boxes=text_boxes
        )
        if i == 0:
            first_crop = crop
            factor, mode = f, m
        if not usable:
            continue
        try:
            txt = ocr.ocr_image(crop)
        except Exception:
            txt = ""
        txt = (txt or "").strip()
        if txt and txt not in chunks:
            chunks.append(txt)

    return "\n".join(chunks), factor, mode, len(boxes), first_crop


def read_by_lines(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    refiner,
    category: str,
    lang_code: str = "",
    script: str = "",
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    span: int = 1,
    stride: int = 1,
    overlap: int = 1,
    ocr=None,
    log: Optional[List[str]] = None,
) -> Tuple[str, List[str], int]:
    if refiner is None or not getattr(refiner, "vision", False):
        return "", [], 0

    try:
        from core.line_reader import read_lines
    except Exception as e:
        if log is not None:
            log.append(f"    ⏭ 행 판독 모듈 로드 실패 ({e})")
        return "", [], 0

    cropped = crop_region(image, bbox)
    if cropped.width < 8 or cropped.height < 8:
        return "", [], 0

    bx0, by0, bx1, by1 = (int(v) for v in bbox)
    local: List[Tuple[int, int, int, int]] = []
    for b in (text_boxes or []):
        if b[0] >= bx1 or b[2] <= bx0 or b[1] >= by1 or b[3] <= by0:
            continue
        local.append((
            max(0, b[0] - bx0), max(0, b[1] - by0),
            min(cropped.width, b[2] - bx0), min(cropped.height, b[3] - by0),
        ))

    calls = {"n": 0}

    def _reader(win_img: Image.Image, start: int, end: int) -> str:
        calls["n"] += 1
        span_n = max(1, end - start)
        hint = (
            f"{category} — line {start + 1}"
            if span_n == 1
            else f"{category} — lines {start + 1}~{end}"
        )
        try:
            return refiner.read_raw(
                win_img, hint=hint, max_chars=160,
                lang_code=lang_code, script=script,
            )
        except Exception:
            return ""

    ocr_fn = None
    if ocr is not None and getattr(ocr, "available", False):
        if hasattr(ocr, "recognize_lines"):
            def ocr_fn(crops):
                return ocr.recognize_lines(crops)

    merged, lines, windows = read_lines(
        cropped, _reader, boxes=local,
        span=span, stride=stride, overlap=overlap,
        ocr_fn=ocr_fn, log=log,
    )

    if log is not None:
        src = "VLM + 전용 인식기" if ocr_fn is not None else "VLM 단독"
        log.append(
            f"    🔁 [LINE READ] '{category}' VLM 호출 {calls['n']}회 "
            f"({src}) — 한 번에 긴 문장을 읽지 않고 행 단위로 끊었습니다."
        )

    return merged, [l.text for l in lines if l.text], len(windows)


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
    array_categories: Optional[set] = None,
    text_boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    line_read_fn: Optional[Callable] = None,
) -> List[ExtractedField]:
    out: List[ExtractedField] = []
    arrays = set(array_categories or ())
    tboxes = list(text_boxes or [])

    for plan in plans:
        if log is not None:
            log.append(f"  ✂️ '{plan.category}' 크롭 px{plan.bbox}")

        text, factor, mode, tiles, crop = _ocr_tiles(
            ocr, image, plan.bbox, target_px, max_tiles, log=log,
            category=plan.category,
            table_rows=int(getattr(plan, "table_rows", 0)),
            legible=int(getattr(plan, "legible", 0)),
            patches=int(getattr(plan, "patches_total", 0)),
            text_boxes=tboxes,
        )

        cleaned, nlp_meta = nlp_clean_ocr(text, nlp=nlp, log=log)
        if nlp_meta["dropped"] and log is not None:
            log.append(
                f"    🧬 NLP 게이트: {nlp_meta['lines_in']} → "
                f"{nlp_meta['lines_out']}줄 ({nlp_meta['dropped']}줄 제거)"
            )

        line_texts: List[str] = []
        windows_n = 0
        if line_read_fn is not None:
            try:
                merged, line_texts, windows_n = line_read_fn(
                    image, plan.bbox, plan.category, tboxes, ocr, log
                )
            except Exception as e:
                if log is not None:
                    log.append(f"    ⚠ 행 판독 실패 ({e})")
                merged = ""
            if merged and len(line_texts) >= LINE_READ_MIN_LINES:
                if log is not None:
                    log.append(
                        f"    🔀 [DRAFT SWAP] '{plan.category}' 초안을 행 "
                        f"단위 판독 결과로 교체합니다 "
                        f"({len(cleaned)}자 → {len(merged)}자 / "
                        f"{len(line_texts)}행)"
                    )
                cleaned = merged
            elif merged and not cleaned.strip():
                cleaned = merged

        if not line_texts and cleaned.strip():
            line_texts = [
                ln.strip() for ln in cleaned.splitlines() if ln.strip()
            ]

        refined: Dict[str, object] = {}
        values: Dict[str, str] = {}
        rows_out: List[dict] = []

        if refine_fn is not None:
            vlm_crop = crop
            if vlm_crop is None:
                vlm_crop = crop_region(image, plan.bbox)
            try:
                vlm_crop = fit_for_vlm(vlm_crop, log=log)
            except Exception:
                pass
            try:
                refined = refine_fn(
                    plan.category, cleaned, vlm_crop, plan.top_field
                ) or {}
            except TypeError:
                try:
                    refined = refine_fn(plan.category, cleaned, vlm_crop) or {}
                except Exception as e:
                    if log is not None:
                        log.append(f"    ⚠ 정제 추출 실패: {e}")
            except Exception as e:
                if log is not None:
                    log.append(f"    ⚠ 정제 추출 실패: {e}")

            raw_alt = refined.get("__raw__") if isinstance(refined, dict) else ""
            if isinstance(raw_alt, str) and raw_alt.strip():
                if log is not None:
                    log.append(
                        f"    👁 [RAW READ] '{plan.category}' OCR 원문을 VLM "
                        f"판독으로 교체합니다: {raw_alt[:48]!r}"
                    )
                cleaned = raw_alt.strip()
            if isinstance(refined, dict):
                inner = refined.get("__fields__")
                if isinstance(inner, dict):
                    values = {
                        str(k): str(v) for k, v in inner.items()
                        if isinstance(v, str) and v.strip()
                    }
                rowset = refined.get("__rows__")
                if isinstance(rowset, list):
                    rows_out = [r for r in rowset if isinstance(r, dict) and r]

        if refine_fn is None and plan.category in arrays and not rows_out:
            field_name = str(plan.top_field or plan.category)
            seen_rows: set = set()
            for row_text in (line_texts or cleaned.splitlines()):
                t = str(row_text or "").strip()
                if len(t) < 2:
                    continue
                key = "".join(ch for ch in t.lower() if ch.isalnum())
                if not key or key in seen_rows:
                    continue
                seen_rows.add(key)
                rows_out.append({field_name: t})
            if rows_out and log is not None:
                log.append(
                    f"    🛟 [OCR ONLY] 정제 LLM 없이 '{plan.category}' 의 "
                    f"OCR 행 {len(rows_out)}건을 '{field_name}' 값으로 "
                    f"직접 승격했습니다."
                )

        if refine_fn is None and plan.category not in arrays and not values:
            field_name = str(plan.top_field or plan.category)
            body = cleaned.strip()
            if len(body) >= 2:
                values[field_name] = body
                if log is not None:
                    log.append(
                        f"    🛟 [OCR ONLY] 정제 LLM 없이 '{plan.category}' 의 "
                        f"OCR 원문을 '{field_name}' 값으로 직접 씁니다."
                    )

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
        field.field_values = values
        field.rows = rows_out if plan.category in arrays else []
        field.line_texts = list(line_texts)
        field.line_windows = int(windows_n)
        out.append(field)

        if log is not None:
            if values:
                brief = " | ".join(
                    f"{k}={str(v)[:24]}" for k, v in list(values.items())[:5]
                )
                log.append(f"    📝 [{plan.category}] {brief}")
            else:
                preview = field.value[:60].replace("\n", " / ")
                log.append(f"    📝 '{plan.category}' = {preview or '(공백)'}")

    return out


def fields_to_record(
    fields: Sequence[ExtractedField],
    schema: Optional[dict] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {}
    raw_text: Dict[str, str] = {}
    schema_fields = set((schema or {}).get("fields", {}) or {})

    for f in fields:
        if getattr(f, "rows", None):
            prev = record.get(f.category)
            merged = list(prev) if isinstance(prev, list) else []
            for r in f.rows:
                if r not in merged:
                    merged.append(r)
            record[f.category] = merged
            continue

        if f.field_values:
            for key, val in f.field_values.items():
                if not val:
                    continue
                if schema_fields and key not in schema_fields:
                    continue
                if key in record and record[key]:
                    continue
                record[key] = val
            continue

        val = f.value
        if not val:
            continue
        if schema_fields and f.category not in schema_fields:
            prev = raw_text.get(f.category, "")
            raw_text[f.category] = (prev + "\n" + val).strip() if prev else val
            continue
        if f.category in record:
            prev = record[f.category]
            if isinstance(prev, list):
                prev.append(val)
            else:
                record[f.category] = [prev, val]
        else:
            record[f.category] = val

    for name in schema_fields:
        record.setdefault(name, None)

    if raw_text:
        record["__ocr_by_category__"] = raw_text
    return record


def record_to_json(
    record: Dict[str, object],
    schema: Optional[dict] = None,
    drop_empty: bool = True,
    source_path: str = "",
    lang_code: str = "",
) -> Dict[str, object]:
    clean: Dict[str, object] = {}
    for key, val in record.items():
        if key.startswith("__"):
            continue
        if drop_empty and (val is None or val == "" or val == []):
            continue
        clean[key] = val

    ocr_raw = record.get("__ocr_by_category__")
    ocr_raw = ocr_raw if isinstance(ocr_raw, dict) else {}

    def _fallback(reason: str) -> Dict[str, object]:
        fields = (schema or {}).get("fields", {}) or {}
        grouped: Dict[str, Dict[str, object]] = {}
        for key, val in clean.items():
            definition = fields.get(key)
            cat = (
                str(definition.get("category") or "misc")
                if isinstance(definition, dict) else "misc"
            )
            grouped.setdefault(cat, {})[key] = val
        out: Dict[str, object] = {
            "doc_type": (schema or {}).get("code") or "",
            "domain": (schema or {}).get("domain") or "",
            "fields": clean,
            "by_category": grouped,
            "shape_error": reason,
        }
        if ocr_raw:
            out["ocr_by_category"] = ocr_raw
        return out

    try:
        from core.record_shape import build_record, relay_plan
    except Exception as e:
        return _fallback(f"{type(e).__name__}: {e}")

    try:
        payload = build_record(
            clean, schema or {},
            source_path=source_path,
            lang_code=lang_code,
            has_vision=True,
            ocr_pool=ocr_raw,
        )
    except TypeError as e:
        try:
            payload = build_record(
                clean, schema or {},
                source_path=source_path,
                lang_code=lang_code,
                has_vision=True,
            )
            payload["shape_warning"] = (
                f"ocr_pool 미지원 build_record — {type(e).__name__}: {e}"
            )
        except Exception as e2:
            return _fallback(f"{type(e2).__name__}: {e2}")
    except Exception as e:
        return _fallback(f"{type(e).__name__}: {e}")

    try:
        plan = relay_plan(payload, str(payload.get("doc_type") or ""))
    except Exception:
        plan = []
    if plan:
        payload["relay_plan"] = plan

    if ocr_raw:
        payload["ocr_by_category"] = ocr_raw

    return payload