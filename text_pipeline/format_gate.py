import re
from typing import Callable, Dict, List, Optional, Sequence

DIGIT_RE = re.compile(r"\d")
DIGITS_ONLY_RE = re.compile(r"\d")
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)

DATE_LITERAL_RE = re.compile(
    r"(\d{4}\s*[^\w\s]?\s*\d{1,2}\s*[^\w\s]?\s*\d{1,2})"
    r"|(\d{1,2}\s*[^\w\s]\s*\d{1,2}\s*[^\w\s]\s*\d{2,4})"
    r"|(\d{1,2}\s+[^\W\d_]{3,12}\.?\s+\d{2,4})"
    r"|([^\W\d_]{3,12}\.?\s+\d{1,2},?\s+\d{4})",
    re.UNICODE,
)

LINK_RE = re.compile(
    r"(https?://)|(^\s*/)|(\s/[^\s]{2,})|((?:www|ftp)\.)",
    re.IGNORECASE | re.UNICODE,
)

EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^\W\d_]{2,}", re.UNICODE)

CODE_TOKEN_RE = re.compile(
    r"[^\W\d_]{1,6}\s*[-_/]?\s*\d{3,}|\d{3,}\s*[-_/]\s*[^\W\d_]{1,6}",
    re.UNICODE,
)

TRACKING_SHAPE_RE = re.compile(
    r"\b(?:[^\W\d_]{2,4}\d{6,}|\d{9,}|\d{4}[- ]\d{4}[- ]\d{4})\b",
    re.UNICODE,
)

PHONE_SHAPE_RE = re.compile(
    r"(?:\+?\d{1,4}[\s\-.()]{0,3})?(?:\d[\s\-.()]{0,3}){6,14}\d",
    re.UNICODE,
)

NUMERIC_VALUE_RE = re.compile(r"[-+]?\d[\d\s,.\u00A0'\u066B\u066C]*\d|\d", re.UNICODE)

PURE_NUMERIC_RE = re.compile(
    r"^[-+]?[\d\s,.\u00A0'\u066B\u066C]*\d[\d\s,.\u00A0'\u066B\u066C]*"
    r"\s*[%‰°]?\s*[^\W\d_]{0,4}$",
    re.UNICODE,
)

ANCHOR_MARGIN_THRESHOLD = 0.015


class FieldFormat:
    DATE = "date"
    TRACKING_CODE = "tracking_code"
    IDENTIFIER = "identifier"
    LINK = "link"
    NUMERIC = "numeric"
    ENUM = "enum"
    PHONE = "phone"
    ADDRESS = "address"
    SYNTHESIS = "synthesis"
    TEXT = "text"

    ALL = (
        DATE, TRACKING_CODE, IDENTIFIER, LINK, NUMERIC,
        ENUM, PHONE, ADDRESS, SYNTHESIS, TEXT,
    )


FORMAT_ANCHORS: Dict[str, str] = {
    FieldFormat.DATE: "a calendar date value such as year month day",
    FieldFormat.PHONE: "a telephone or fax contact number",
    FieldFormat.LINK: "a web address url or hyperlink path",
    FieldFormat.NUMERIC: "a numeric measured quantity amount or monetary value",
    FieldFormat.IDENTIFIER: "an alphanumeric reference identifier code or serial number",
    FieldFormat.TRACKING_CODE: "a shipment tracking or waybill barcode number",
    FieldFormat.ENUM: "a short status state or category label",
    FieldFormat.ADDRESS: "a postal street address or geographic location",
    FieldFormat.SYNTHESIS: "a long free form descriptive sentence or paragraph",
    FieldFormat.TEXT: "a plain name or free text label",
}

VALUE_FORMAT_PROBES = (
    (FieldFormat.LINK, lambda v: bool(LINK_RE.search(v)) or bool(EMAIL_RE.search(v))),
    (FieldFormat.DATE, lambda v: bool(DATE_LITERAL_RE.search(v))),
    (FieldFormat.PHONE, lambda v: bool(PHONE_SHAPE_RE.search(v))),
    (FieldFormat.TRACKING_CODE, lambda v: bool(TRACKING_SHAPE_RE.search(v))),
    (FieldFormat.IDENTIFIER, lambda v: bool(CODE_TOKEN_RE.search(v))),
    (FieldFormat.NUMERIC, lambda v: bool(PURE_NUMERIC_RE.match(v.strip()))),
)


def infer_format_from_values(values: Sequence[str]) -> Optional[str]:
    samples = [str(v).strip() for v in (values or []) if v is not None and str(v).strip()]
    if not samples:
        return None

    tally: Dict[str, int] = {}
    for v in samples:
        for fmt, probe in VALUE_FORMAT_PROBES:
            try:
                if probe(v):
                    tally[fmt] = tally.get(fmt, 0) + 1
                    break
            except Exception:
                continue

    if not tally:
        avg_len = sum(len(v) for v in samples) / float(len(samples))
        if avg_len >= 40:
            return FieldFormat.SYNTHESIS
        return None

    best = max(tally.items(), key=lambda kv: kv[1])
    if best[1] * 2 >= len(samples):
        return best[0]
    return None


def infer_format_by_anchor(
    field_name: str,
    definition: Optional[dict],
    embed_fn: Optional[Callable[[List[str]], "np.ndarray"]] = None,
    cache: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    if embed_fn is None:
        return None
    if cache is not None and field_name in cache:
        return cache[field_name]

    import numpy as np

    parts = [str(field_name).replace("_", " ")]
    if definition:
        from .field_bank import flatten_text_values
        for sem in flatten_text_values(definition.get("semantic")):
            if sem:
                parts.append(sem)
                break
    probe_text = " ".join(parts).strip()
    if not probe_text:
        return None

    fmt_keys = list(FORMAT_ANCHORS.keys())
    try:
        mat = embed_fn([probe_text] + [FORMAT_ANCHORS[k] for k in fmt_keys])
    except Exception:
        return None
    if mat is None:
        return None

    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1 or mat.shape[0] < len(fmt_keys) + 1:
        return None

    def _unit(v):
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-8 else v

    probe = _unit(mat[0])
    sims = [float(np.dot(probe, _unit(mat[i + 1]))) for i in range(len(fmt_keys))]

    order = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
    top = order[0]
    margin = sims[top] - (sims[order[1]] if len(order) > 1 else 0.0)

    result = fmt_keys[top] if margin >= ANCHOR_MARGIN_THRESHOLD else None
    if cache is not None and result:
        cache[field_name] = result
    return result


def detect_field_format(
    field_name: str,
    definition: Optional[dict] = None,
    embed_fn: Optional[Callable[[List[str]], "np.ndarray"]] = None,
    anchor_cache: Optional[Dict[str, str]] = None,
) -> str:
    if definition:
        explicit = str(definition.get("format", "") or "").strip().lower()
        if explicit in FieldFormat.ALL:
            return explicit

        for key in ("value", "bias", "examples"):
            raw = definition.get(key)
            if not raw:
                continue
            from .field_bank import flatten_text_values
            inferred = infer_format_from_values(flatten_text_values(raw))
            if inferred:
                return inferred

    anchored = infer_format_by_anchor(field_name, definition, embed_fn, anchor_cache)
    if anchored:
        return anchored

    return FieldFormat.TEXT


def digit_count(value: str) -> int:
    return len(DIGITS_ONLY_RE.findall(value or ""))


def has_date_literal(value: str) -> bool:
    return bool(DATE_LITERAL_RE.search(value or ""))


def value_matches_format(value: str, fmt: str) -> bool:
    v = (value or "").strip()
    if not v:
        return False

    if fmt == FieldFormat.NUMERIC:
        return bool(NUMERIC_VALUE_RE.search(v))

    if fmt == FieldFormat.DATE:
        if has_date_literal(v):
            return True
        return digit_count(v) >= 4

    if fmt == FieldFormat.PHONE:
        return digit_count(v) >= 7

    if fmt == FieldFormat.LINK:
        return bool(LINK_RE.search(v)) or bool(EMAIL_RE.search(v))

    if fmt == FieldFormat.TRACKING_CODE:
        if bool(TRACKING_SHAPE_RE.search(v)):
            return True
        return digit_count(v) >= 6

    if fmt == FieldFormat.IDENTIFIER:
        if bool(CODE_TOKEN_RE.search(v)):
            return True
        return digit_count(v) >= 3

    if fmt == FieldFormat.ADDRESS:
        return len(v) >= 4

    if fmt == FieldFormat.ENUM:
        return len(v) >= 1

    if fmt == FieldFormat.SYNTHESIS:
        if len(v) < 4:
            return False
        return bool(ALPHA_RE.search(v))

    if fmt == FieldFormat.TEXT:
        return bool(ALPHA_RE.search(v))

    return True


def format_gate_for_indexing(
    chunks,
    bank=None,
    formats: Optional[Dict[str, str]] = None,
    log: Optional[List[str]] = None,
) -> List[dict]:
    formats = dict(formats or {})
    report: List[dict] = []

    for chunk in chunks:
        prop = getattr(chunk, "property", "") or ""
        if not prop or prop == "unclassified":
            continue

        definition = None
        if bank is not None:
            entry = bank.get(prop)
            definition = entry.definition if entry else None

        if chunk.confirmed:
            chunk.property_format = formats.get(prop) or detect_field_format(
                prop, definition
            )
            report.append({
                "text": chunk.text,
                "property": prop,
                "format": chunk.property_format,
                "verdict": "confirmed-bypass",
            })
            continue

        fmt = formats.get(prop) or detect_field_format(prop, definition)
        chunk.property_format = fmt

        target = chunk.value_part or chunk.text
        ok = value_matches_format(target, fmt)

        if ok:
            report.append({
                "text": chunk.text,
                "property": prop,
                "format": fmt,
                "verdict": "pass",
            })
            continue

        chunk.property = "unclassified"
        chunk.status = "format-rejected"
        report.append({
            "text": chunk.text,
            "property": prop,
            "format": fmt,
            "verdict": "reject",
        })
        if log is not None:
            log.append(
                f"🚧 FORMAT GATE 탈락 '{chunk.text}' "
                f"({prop} / {fmt}) → unclassified"
            )

    return report


def build_format_map(
    bank,
    embed_fn: Optional[Callable[[List[str]], "np.ndarray"]] = None,
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if bank is None:
        return out
    anchor_cache: Dict[str, str] = {}
    for name in bank.field_names:
        entry = bank.get(name)
        out[name] = detect_field_format(
            name,
            entry.definition if entry else None,
            embed_fn=embed_fn,
            anchor_cache=anchor_cache,
        )
    return out