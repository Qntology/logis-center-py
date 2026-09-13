import hashlib
import re
import struct
import time
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

__all__ = [
    "build_record", "relay_plan", "resolve_doc_identity",
    "json_to_natural_language", "normalize_identifier",
    "normalize_numeric_homoglyphs", "digest_of", "relay_index",
]

ZERO_ADDR = "0x" + "0" * 40

HOST_TRADING = "trading"
MODE_BY_DOMAIN = {
    "trade": "shipping",
    "logistics": "tracking",
    "commerce": "goods",
    "comics": "comics",
}

ROOT_PROMOTE_BY_DOMAIN = {
    "comics": (
        "title", "episode_number", "page_number", "credits",
        "publisher", "character_name", "dialogue", "narration",
        "sound_effect", "sign_text",
    ),
}

CRC32_POLY = 0xEDB88320

WS_RE = re.compile(r"\s+")
NON_ALNUM_RE = re.compile(r"[^0-9A-Za-z\uAC00-\uD7A3]+")

FULLWIDTH_OFFSET = 0xFEE0

NUMERIC_HOMOGLYPH = {
    "O": "0", "o": "0", "Q": "0", "D": "0",
    "I": "1", "l": "1", "|": "1",
    "S": "5", "s": "5",
    "B": "8",
    "Z": "2", "z": "2",
    "G": "6",
}


def _crc32(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ (CRC32_POLY if crc & 1 else 0)
    return crc ^ 0xFFFFFFFF


def _addr(prefix: str, payload: str) -> str:
    raw = f"{prefix}\x1f{payload}".encode("utf-8")
    return "0x" + hashlib.sha1(raw).hexdigest()


def normalize_identifier(value: str) -> str:
    s = unicodedata.normalize("NFKC", str(value or "")).strip()
    out = []
    for ch in s:
        cp = ord(ch)
        if 0xFF01 <= cp <= 0xFF5E:
            ch = chr(cp - FULLWIDTH_OFFSET)
        out.append(ch)
    s = "".join(out)
    return NON_ALNUM_RE.sub("", s).upper()


def normalize_numeric_homoglyphs(value: str) -> str:
    s = normalize_identifier(value)
    digits = sum(1 for c in s if c.isdigit())
    if digits * 2 < len(s):
        return s
    return "".join(NUMERIC_HOMOGLYPH.get(c, c) for c in s)


def digest_of(record: Dict[str, object]) -> str:
    parts: List[str] = []
    for key in sorted(record.keys()):
        if key.startswith("__"):
            continue
        val = record[key]
        if val is None or val == "" or val == [] or val == {}:
            continue
        parts.append(f"{key}={_stringify(val)}")
    body = WS_RE.sub(" ", "|".join(parts)).strip().lower()
    return _addr("digest", body)


def _stringify(val) -> str:
    if isinstance(val, dict):
        return "{" + ",".join(
            f"{k}:{_stringify(v)}" for k, v in sorted(val.items())
        ) + "}"
    if isinstance(val, (list, tuple)):
        return "[" + ",".join(_stringify(v) for v in val) + "]"
    return str(val)


def relay_index(key: str) -> int:
    norm = normalize_numeric_homoglyphs(key)
    if not norm:
        return 0
    return int(_crc32(norm.encode("utf-8")))


def entity_address(kind: str, value: str) -> str:
    norm = normalize_identifier(value)
    if not norm:
        return ZERO_ADDR
    return _addr(kind, norm)


def trading_cc(doc_type: str, doc_number: str) -> str:
    return entity_address("cc", f"{HOST_TRADING}/{doc_type}/{doc_number}")


def trading_ref(doc_number: str) -> str:
    return entity_address("ref", f"{HOST_TRADING}/{doc_number}")


def entity_bcc(mode: str, doc_type: str, digest: str) -> str:
    return entity_address("bcc", f"{mode}/{doc_type}/{digest}")


IDENT_SHAPE_RE = re.compile(
    r"\b([A-Z]{1,6}[-/ ]?\d{3,}[A-Z0-9\-/]*|\d{6,})\b"
)

IDENT_SCAN_CATEGORIES = ("header", "financials", "logistics", "conditions")


def _task_id(fallback: str) -> str:
    seed = normalize_identifier(fallback) or str(int(time.time() * 1000))
    return f"task_{_crc32(seed.encode('utf-8')):010d}"


def resolve_doc_identity(
    record: Dict[str, object],
    doc_type: str,
    fallback: str = "",
    ocr_pool: Optional[Dict[str, str]] = None,
) -> Tuple[str, bool]:
    for key in ("doc_number", "no", "reference_number"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip(), False

    for cat in IDENT_SCAN_CATEGORIES:
        node = record.get(cat)
        if isinstance(node, dict):
            for key in ("doc_number", "no", "reference_number"):
                v = node.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip(), False

    if ocr_pool:
        for cat in IDENT_SCAN_CATEGORIES:
            raw = ocr_pool.get(cat)
            if not isinstance(raw, str) or not raw.strip():
                continue
            m = IDENT_SHAPE_RE.search(raw.upper())
            if m:
                return normalize_numeric_homoglyphs(m.group(1)), True

        for raw in ocr_pool.values():
            if not isinstance(raw, str) or not raw.strip():
                continue
            m = IDENT_SHAPE_RE.search(raw.upper())
            if m:
                return normalize_numeric_homoglyphs(m.group(1)), True

    return _task_id(fallback), True


def humanize(name: str) -> str:
    return str(name or "").replace("_", " ").strip()


def json_to_natural_language(
    grouped: Dict[str, Dict[str, object]],
    roots: Dict[str, object],
    arrays: Optional[Dict[str, list]] = None,
    ocr_pool: Optional[Dict[str, str]] = None,
) -> str:
    chunks: List[str] = []

    amount = roots.get("amount")
    currency = roots.get("currency")

    cats = set(grouped.keys()) | set((arrays or {}).keys())
    cats |= set((ocr_pool or {}).keys())

    for cat in sorted(cats):
        sentences = [f"Regarding {humanize(cat)},"]
        wrote = False

        body = grouped.get(cat)
        if isinstance(body, dict):
            for key in sorted(body.keys()):
                val = body[key]
                if val is None or val == "" or val == []:
                    continue
                if isinstance(val, (dict, list)):
                    continue
                sentences.append(f"Its {humanize(key)} is {val}.")
                wrote = True

        rows = (arrays or {}).get(cat)
        if isinstance(rows, list) and rows:
            for idx, row in enumerate(rows[:8], start=1):
                if not isinstance(row, dict):
                    continue
                parts = [
                    f"{humanize(k)} is {v}"
                    for k, v in sorted(row.items())
                    if v not in (None, "", [])
                    and not isinstance(v, (dict, list))
                ]
                if parts:
                    sentences.append(f"Line {idx} has " + ", ".join(parts) + ".")
                    wrote = True

        if not wrote:
            raw = (ocr_pool or {}).get(cat)
            if isinstance(raw, str) and raw.strip():
                snippet = WS_RE.sub(" ", raw).strip()[:200]
                sentences.append(f"Its raw text reads {snippet}.")

        chunks.append(" ".join(sentences))

    if amount not in (None, "", []):
        unit = f" {currency}" if currency else ""
        chunks.insert(0, f"The amount is {amount}{unit}.")

    return " ".join(c for c in chunks if c).strip()


def build_record(
    fields: Dict[str, object],
    schema: Dict[str, object],
    source_path: str = "",
    lang_code: str = "",
    has_vision: bool = True,
    ocr_pool: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    defs = (schema or {}).get("fields", {}) or {}
    domain = str((schema or {}).get("domain") or "trade")
    doc_type = str(
        (schema or {}).get("code") or (schema or {}).get("doc_type") or ""
    )
    mode = MODE_BY_DOMAIN.get(domain, domain or "shipping")

    grouped: Dict[str, Dict[str, object]] = {}
    arrays: Dict[str, list] = {}
    roots: Dict[str, object] = {}

    for key, val in fields.items():
        if key.startswith("__"):
            continue
        if val is None or val == "" or val == []:
            continue

        if isinstance(val, list):
            arrays[key] = val
            continue

        definition = defs.get(key)
        cat = "misc"
        if isinstance(definition, dict):
            cat = str(definition.get("category") or "misc")
        grouped.setdefault(cat, {})[key] = val

        if key in _promote_set(domain):
            roots[key] = val

    for cat in list(arrays.keys()):
        grouped.pop(cat, None)

    flat = {k: v for g in grouped.values() for k, v in g.items()}
    identity, fallback_used = resolve_doc_identity(
        {**roots, **flat, **grouped},
        doc_type,
        fallback=source_path,
        ocr_pool=ocr_pool,
    )

    record: Dict[str, object] = {}

    record["doc_type"] = doc_type
    record["type"] = doc_type
    record["code"] = ""
    record["mode"] = mode
    record["no"] = identity
    record["doc_number"] = identity

    for cat in sorted(grouped.keys()):
        record[cat] = grouped[cat]
    for cat in sorted(arrays.keys()):
        record[cat] = arrays[cat]
        if cat == "items":
            record["line_items"] = list(arrays[cat])

    for key, val in roots.items():
        record.setdefault(key, val)

    record["status"] = 0
    record["order"] = 0
    record["embed"] = 0
    record["goods"] = 0
    record["tracking"] = 0
    record["has_vision"] = 1 if has_vision else 0
    record["barcode"] = ""
    record["stock_keeping_unit"] = ""
    record["tracking_number"] = str(record.get("reference_bl") or "")
    record["tags"] = []
    record["lang"] = str(lang_code or "")
    record["source"] = str(source_path or "")

    text = json_to_natural_language(
        grouped, roots, arrays=arrays, ocr_pool=ocr_pool
    )
    record["text"] = text
    record["masked_text"] = text

    dg = digest_of(record)
    record["digest"] = dg
    record["id"] = _addr("id", dg)
    record["cc"] = trading_cc(doc_type, identity)
    record["ref"] = trading_ref(identity)
    record["bcc"] = entity_bcc(mode, doc_type, dg)
    record["from"] = ZERO_ADDR
    record["to"] = entity_address("to", str(record.get("recipient_name") or identity))
    record["index"] = relay_index(identity)

    now = int(time.time() * 1000)
    record["created_at"] = now
    record["updated_at"] = now

    record["identity_fallback"] = bool(fallback_used)

    return record


def _promote_set(domain: str = "") -> set:
    override = ROOT_PROMOTE_BY_DOMAIN.get(str(domain or "").strip().lower())
    if override:
        return set(override)
    from .trade_schema import ROOT_PROMOTE_FIELDS
    return set(ROOT_PROMOTE_FIELDS)


def relay_plan(
    record: Dict[str, object],
    doc_type: str,
) -> List[Dict[str, str]]:
    if bool(record.get("identity_fallback")):
        return []

    identity = str(record.get("doc_number") or "").strip()
    bl = str(record.get("reference_bl") or "").strip()

    out: List[Dict[str, str]] = []
    if bl:
        out.append({
            "target": "BL", "role": "transport",
            "key": "doc_number", "value": bl,
        })
    if identity and not identity.startswith("task_"):
        for target in (
            "PL", "CINV", "CSI", "CO", "ED", "ID", "FI", "SOA", "PO", "LC",
        ):
            if target == doc_type:
                continue
            out.append({
                "target": target, "role": "reference_invoice",
                "key": "reference_invoice", "value": identity,
            })
    return out