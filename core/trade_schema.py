import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

TYPE_MARKER_RE = re.compile(r"\{(String|Number|Boolean|Array)\}")

TRADE_DOMAIN = "trade"
TRADE_BIAS_NODE = "shipping_doc"

ARRAY_CATEGORIES = (
    "items", "containers", "parties", "other_parties", "charges",
    "test_results", "findings_and_damage", "account_ledger",
    "adjustments", "packing_details", "licensed_items", "purchased_items",
)

LANG_NODE_ALIAS: Dict[str, str] = {
    "kor": "ko", "eng": "en", "jpn": "ja", "zho": "zh",
    "deu": "de", "fra": "fr", "spa": "es", "por": "pt",
    "ita": "it", "nld": "nl", "ara": "ar", "ces": "cs",
}

_CACHE: Dict[str, dict] = {}


def _repair_json(text: str) -> Tuple[Optional[str], str]:
    depth = 0
    in_str = False
    esc = False
    safe = -1
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 1:
                safe = i
    if in_str or safe < 0:
        return None, "문자열이 닫히지 않아 복구 지점을 찾지 못했습니다."
    body = text[: safe + 1].rstrip()
    if body.endswith(","):
        body = body[:-1]
    return body + "\n}", f"닫히지 않은 괄호 {depth}단 — 마지막 온전한 최상위 항목까지만 복구"


def load_bias_dictionary(path) -> Tuple[dict, List[str]]:
    p = Path(path)
    key = str(p)
    if key in _CACHE:
        return _CACHE[key], []

    diag: List[str] = []
    if not p.exists():
        diag.append(f"  ❌ 사전 파일이 없습니다: {p}")
        _CACHE[key] = {}
        return {}, diag

    raw = p.read_text(encoding="utf-8", errors="ignore")
    data = None
    try:
        data = json.loads(raw)
        diag.append(f"  📖 사전 적재 {p.name} ({len(raw) / 1000:.0f}KB)")
    except json.JSONDecodeError as e:
        lines = raw.splitlines()
        bad = lines[e.lineno - 1][:140] if 0 < e.lineno <= len(lines) else ""
        diag.append(
            f"  ⚠ {p.name} JSON 구문 오류 — line {e.lineno} col {e.colno}: {e.msg}"
        )
        if bad:
            diag.append(f"      → {bad}")
        fixed, why = _repair_json(raw)
        if fixed is not None:
            try:
                data = json.loads(fixed)
                diag.append(f"  🩹 부분 복구 성공 — {why}")
                diag.append(
                    "      복구분 이후 노드는 유실됩니다. 원본 괄호를 맞춰 주세요."
                )
            except Exception as e2:
                diag.append(f"  ❌ 부분 복구 실패: {e2}")

    if not isinstance(data, dict):
        data = {}
    _CACHE[key] = data
    return data, diag


def desc_to_anchor(desc) -> str:
    return " ".join(TYPE_MARKER_RE.sub(" ", str(desc or "")).split())


def desc_to_format(desc) -> str:
    return "numeric" if "{Number}" in str(desc or "") else ""


def trade_doc_codes(bias: dict) -> List[str]:
    node = (bias or {}).get("trade_schema") or {}
    ov = node.get("overlay")
    return sorted(ov.keys()) if isinstance(ov, dict) else []


def trade_schema_triples(bias: dict, doc_type: str) -> List[Tuple[str, str, str]]:
    node = (bias or {}).get("trade_schema") or {}
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
        absorb(ov.get(doc_type))
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


def bridge_phrases(bias: dict, field: str) -> List[str]:
    out: List[str] = []
    seen = set()
    sb = (bias or {}).get("search_bridge") or {}

    mva = sb.get("multilingual_value_anchor")
    if isinstance(mva, dict):
        entry = mva.get(f"{TRADE_BIAS_NODE}.{field}")
        if isinstance(entry, dict):
            for k in ("semantic", "bias"):
                for t in _split_phrases(entry.get(k)):
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        out.append(t)

    ab = sb.get("abstract_bridge")
    if isinstance(ab, dict):
        for target, entry in ab.items():
            if str(target).rsplit(".", 1)[-1] != field:
                continue
            if not isinstance(entry, dict):
                continue
            for k in ("semantic", "bias"):
                for t in _split_phrases(entry.get(k)):
                    if t.lower() not in seen:
                        seen.add(t.lower())
                        out.append(t)
    return out


def lang_node_entry(bias: dict, lang_code: str, field: str) -> dict:
    code = LANG_NODE_ALIAS.get(str(lang_code or "").strip().lower(), "")
    for c in [c for c in (code, "en", "ko") if c]:
        node = (bias or {}).get(c)
        if not isinstance(node, dict):
            continue
        dom = node.get(TRADE_BIAS_NODE)
        if not isinstance(dom, dict):
            continue
        entry = dom.get(field)
        if isinstance(entry, dict):
            return entry
    return {}


def doc_type_anchors(bias: dict, doc_type: str) -> List[str]:
    node = (bias or {}).get("trade_schema") or {}
    ov = node.get("overlay")
    spec = ov.get(doc_type) if isinstance(ov, dict) else None

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
        out.append(f"a {str(doc_type).replace('_', ' ')} trade document")
    return out


def build_doc_schema(bias: dict, doc_type: str, lang_code: str = "") -> dict:
    fields: Dict[str, dict] = {}

    for category, field, desc in trade_schema_triples(bias, doc_type):
        anchor = desc_to_anchor(desc)
        entry: Dict[str, object] = {
            "category": category,
            "semantic": anchor or str(field).replace("_", " "),
        }

        fmt = desc_to_format(desc)
        if fmt:
            entry["format"] = fmt
        if category in ARRAY_CATEGORIES:
            entry["array"] = True

        loc = lang_node_entry(bias, lang_code, field)
        for key in ("bias", "prejudice", "label"):
            v = loc.get(key)
            if isinstance(v, str) and v.strip():
                entry[key] = v

        bridge = bridge_phrases(bias, field)
        if bridge:
            prev = str(entry.get("bias") or "").strip()
            parts = ([prev] if prev else []) + bridge
            entry["bias"] = ", ".join(parts)

        fields[field] = entry

    return {
        "domain": TRADE_DOMAIN,
        "doc_type": doc_type,
        "code": doc_type,
        "doc_anchors": doc_type_anchors(bias, doc_type),
        "fields": fields,
    }


def load_trade_schemas(
    path,
    lang_code: str = "",
) -> Tuple[Dict[str, dict], List[str]]:
    bias, diag = load_bias_dictionary(path)
    codes = trade_doc_codes(bias)

    out: Dict[str, dict] = {}
    for c in codes:
        out[c] = build_doc_schema(bias, c, lang_code)

    if out:
        avg = sum(len(s["fields"]) for s in out.values()) // max(1, len(out))
        anchored = sum(1 for s in out.values() if len(s["doc_anchors"]) > 1)
        diag.append(
            f"  🚢 무역 서식 스키마 {len(out)}종 조립 (base+overlay) "
            f"| 서식당 평균 필드 {avg}개 | 고유 앵커 보유 {anchored}종"
        )
    elif bias:
        diag.append(
            "  ⚠ trade_schema.overlay 노드를 찾지 못했습니다. "
            "bias.json 구조를 확인하세요."
        )

    return out, diag