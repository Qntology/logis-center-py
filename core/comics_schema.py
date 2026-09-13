import re
from typing import Dict, List, Optional, Sequence, Tuple

from .trade_schema import (
    TYPE_MARKER_RE,
    desc_to_anchor,
    desc_to_format,
    load_bias_dictionary,
)

COMICS_DOMAIN = "comics"
COMICS_BIAS_NODE = "comics"
COMICS_SCHEMA_NODE = "comics_schema"

COMICS_ARRAY_CATEGORIES = ("speech", "thought", "effect", "signage", "cast")
COMICS_TABLE_CATEGORIES: tuple = ()

COMICS_ROOT_PROMOTE = (
    "title", "episode_number", "page_number", "credits",
    "publisher", "character_name", "dialogue", "narration",
    "sound_effect", "sign_text",
)

COMICS_LANG_ALIAS = {
    "kor": "ko", "eng": "en", "jpn": "ja", "zho": "zh",
}


def comics_codes(bias: dict) -> List[str]:
    node = (bias or {}).get(COMICS_SCHEMA_NODE) or {}
    ov = node.get("overlay")
    return sorted(ov.keys()) if isinstance(ov, dict) else []


def comics_schema_triples(bias: dict, code: str) -> List[Tuple[str, str, str]]:
    node = (bias or {}).get(COMICS_SCHEMA_NODE) or {}
    out: List[Tuple[str, str, str]] = []
    index: Dict[Tuple[str, str], int] = {}

    def absorb(cat_obj):
        if not isinstance(cat_obj, dict):
            return
        for category, fields in cat_obj.items():
            if not isinstance(fields, dict):
                continue
            for field, desc in fields.items():
                text = str(desc or "")
                key = (category, field)
                if key in index:
                    if text.strip():
                        out[index[key]] = (category, field, text)
                    continue
                index[key] = len(out)
                out.append((category, field, text))

    absorb(node.get("base"))
    ov = node.get("overlay")
    if isinstance(ov, dict):
        absorb(ov.get(code))
    return out


def _split_phrases(raw) -> List[str]:
    if not isinstance(raw, str):
        return []
    out: List[str] = []
    seen = set()
    for part in re.split(r"[,;|\n]+", raw):
        t = part.strip()
        if not t:
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
    return out


def comics_bridge_phrases(bias: dict, field: str) -> List[str]:
    out: List[str] = []
    seen = set()
    sb = (bias or {}).get("search_bridge") or {}

    mva = sb.get("multilingual_value_anchor")
    if isinstance(mva, dict):
        entry = mva.get(f"{COMICS_BIAS_NODE}.{field}")
        if isinstance(entry, dict):
            for k in ("semantic", "bias"):
                for t in _split_phrases(entry.get(k)):
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        out.append(t)
    return out


def comics_lang_entry(bias: dict, lang_code: str, field: str) -> dict:
    code = COMICS_LANG_ALIAS.get(str(lang_code or "").strip().lower(), "")
    for c in [c for c in (code, "ko", "en") if c]:
        node = (bias or {}).get(c)
        if not isinstance(node, dict):
            continue
        dom = node.get(COMICS_BIAS_NODE)
        if not isinstance(dom, dict):
            continue
        entry = dom.get(field)
        if isinstance(entry, dict):
            return entry
    return {}


def comics_doc_anchors(bias: dict, code: str) -> List[str]:
    node = (bias or {}).get(COMICS_SCHEMA_NODE) or {}
    ov = node.get("overlay")
    spec = ov.get(code) if isinstance(ov, dict) else None

    out: List[str] = []
    seen = set()
    if isinstance(spec, dict):
        for _cat, fields in spec.items():
            if not isinstance(fields, dict):
                continue
            for _f, desc in fields.items():
                t = desc_to_anchor(desc)
                if not t or t.lower() in seen:
                    continue
                seen.add(t.lower())
                out.append(t)
    if not out:
        out.append(f"a {str(code).replace('_', ' ').lower()} comic page")
    return out


def build_comics_schema(bias: dict, code: str, lang_code: str = "") -> dict:
    fields: Dict[str, dict] = {}

    for category, field, desc in comics_schema_triples(bias, code):
        anchor = desc_to_anchor(desc)
        entry: Dict[str, object] = {
            "category": category,
            "semantic": anchor or str(field).replace("_", " "),
        }

        fmt = desc_to_format(desc)
        if fmt:
            entry["format"] = fmt
        if category in COMICS_ARRAY_CATEGORIES:
            entry["array"] = True
        if COMICS_TABLE_CATEGORIES and category in COMICS_TABLE_CATEGORIES:
            entry["table"] = True
        if field in ("title", "episode_number"):
            entry["top_region"] = True

        loc = comics_lang_entry(bias, lang_code, field)
        for key in ("bias", "prejudice", "label"):
            v = loc.get(key)
            if isinstance(v, str) and v.strip():
                entry[key] = v

        bridge = comics_bridge_phrases(bias, field)
        if bridge:
            prev = str(entry.get("bias") or "").strip()
            parts = ([prev] if prev else []) + bridge
            entry["bias"] = ", ".join(parts)

        shape = str(entry.get("semantic") or "").strip()
        if shape:
            entry["label"] = shape

        fields[field] = entry

    return {
        "domain": COMICS_DOMAIN,
        "doc_type": code,
        "code": code,
        "doc_anchors": comics_doc_anchors(bias, code),
        "fields": fields,
    }


def load_comics_schemas(
    path,
    lang_code: str = "",
) -> Tuple[Dict[str, dict], List[str]]:
    bias, diag = load_bias_dictionary(path)
    codes = comics_codes(bias)

    out: Dict[str, dict] = {}
    for c in codes:
        out[c] = build_comics_schema(bias, c, lang_code)

    if out:
        avg = sum(len(s["fields"]) for s in out.values()) // max(1, len(out))
        diag.append(
            f"  🎨 만화 스키마 {len(out)}종 조립 (base+overlay) "
            f"| 종당 평균 필드 {avg}개"
        )
    elif bias:
        diag.append(
            "  ⏭ comics_schema.overlay 노드가 없어 만화 스키마를 "
            "건너뜁니다."
        )

    return out, diag