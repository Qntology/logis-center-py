import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CORE_DIR = Path(__file__).resolve().parent
BIAS_PATH = CORE_DIR / "bias_dictionary.json"

PHRASE_SPLIT_RE = re.compile(r"[,;|\n]+")

LANG_NODE_ALIAS: Dict[str, str] = {
    "kor": "ko",
    "eng": "en",
    "ko": "ko",
    "en": "en",
}

DOMAIN_NODE: Dict[str, Tuple[str, ...]] = {
    "logistics": ("tracking",),
    "commerce": ("goods", "order", "review", "coupon", "event"),
    "trade": ("shipping_doc",),
    "comics": ("comics",),
}

FIELD_ALIAS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "logistics": {
        "tracking_number": ("id,link",),
        "recipient_name": ("recipient_name",),
        "recipient_address": ("recipient_address",),
        "shipping_status": ("status",),
        "carrier": ("carrier",),
        "sender_name": ("sender_name",),
        "delivery_date": ("shipping_date", "registration_date"),
        "package_weight": ("weight",),
        "service_type": ("shipping_method",),
        "shipping_fee": ("shipping_fee",),
    },
    "commerce": {
        "goods_title": ("title",),
        "price": ("sale_price",),
        "order_id": ("id,link",),
        "quantity": ("quantity",),
        "stock_status": ("status",),
        "brand": ("brand_name",),
        "option_spec": ("options",),
        "discount_rate": ("discount",),
        "category": ("tags",),
        "review_score": ("completed",),
    },
    "trade": {
        "invoice_number": ("doc_number",),
        "shipper_name": ("sender_name", "sender_address"),
        "consignee_name": ("recipient_name", "recipient_address"),
        "description_of_goods": ("description",),
        "total_amount": ("amount",),
        "issue_date": ("issue_date",),
        "transport_document_number": ("id,link",),
        "export_reference": ("no",),
        "hs_code": ("hs_code",),
        "incoterm": ("incoterms",),
        "total_weight": ("weight_gross",),
        "package_count": ("package_count",),
        "country_of_origin": ("country_of_manufacture",),
        "destination_country": ("pod",),
    },
    "comics": {
        "title": ("title",),
        "episode_number": ("episode_number", "id,link"),
        "dialogue": ("dialogue",),
        "thought_balloon": ("thought_balloon",),
        "narration": ("narration",),
        "sound_effect": ("sound_effect",),
        "character_name": ("character_name",),
        "credits": ("credits",),
    },
}


@lru_cache(maxsize=1)
def load_bias_dictionary() -> dict:
    if not BIAS_PATH.exists():
        return {}
    try:
        with open(BIAS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def available() -> bool:
    return bool(load_bias_dictionary())


def _split(raw) -> List[str]:
    if not isinstance(raw, str):
        return []
    out: List[str] = []
    seen = set()
    for part in PHRASE_SPLIT_RE.split(raw):
        t = part.strip()
        if not t:
            continue
        if len(t) == 1 and t.isascii():
            continue
        low = t.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(t)
    return out


def _lang_nodes(lang_code: str) -> List[str]:
    code = LANG_NODE_ALIAS.get(str(lang_code or "").strip().lower(), "")
    if code:
        return [code]
    return ["ko", "en"]


def _field_sources(domain: str, field: str) -> Tuple[str, ...]:
    table = FIELD_ALIAS.get(domain) or {}
    mapped = table.get(field)
    if mapped:
        return mapped
    return (field,)


def field_phrases(
    domain: str,
    field: str,
    lang_code: str = "",
) -> Tuple[List[str], List[str]]:
    root = load_bias_dictionary()
    if not root:
        return [], []

    nodes = DOMAIN_NODE.get(domain) or ()
    if not nodes:
        return [], []

    sources = _field_sources(domain, field)

    bias: List[str] = []
    prej: List[str] = []
    seen_b = set()
    seen_p = set()

    for lang in _lang_nodes(lang_code):
        lang_node = root.get(lang)
        if not isinstance(lang_node, dict):
            continue
        for node in nodes:
            dom_node = lang_node.get(node)
            if not isinstance(dom_node, dict):
                continue
            for src in sources:
                entry = dom_node.get(src)
                if not isinstance(entry, dict):
                    continue
                for t in _split(entry.get("bias")):
                    low = t.lower()
                    if low in seen_b:
                        continue
                    seen_b.add(low)
                    bias.append(t)
                for t in _split(entry.get("semantic")):
                    low = t.lower()
                    if low in seen_b:
                        continue
                    seen_b.add(low)
                    bias.append(t)
                for t in _split(entry.get("prejudice")):
                    low = t.lower()
                    if low in seen_p:
                        continue
                    seen_p.add(low)
                    prej.append(t)

    return bias, prej


def value_anchor_phrases(domain: str, field: str) -> List[str]:
    root = load_bias_dictionary()
    if not root:
        return []

    node = (
        root.get("search_bridge", {})
        .get("multilingual_value_anchor")
    )
    if not isinstance(node, dict):
        return []

    nodes = DOMAIN_NODE.get(domain) or ()
    sources = _field_sources(domain, field)

    out: List[str] = []
    seen = set()
    for dom in nodes:
        for src in sources:
            entry = node.get(f"{dom}.{src}")
            if not isinstance(entry, dict):
                continue
            for t in _split(entry.get("bias")):
                low = t.lower()
                if low in seen:
                    continue
                seen.add(low)
                out.append(t)
    return out


def report(domain: str, fields: List[str], lang_code: str = "") -> List[str]:
    if not available():
        return [f"  ⏭ 보조 사전이 없습니다: {BIAS_PATH}"]

    lines = [
        f"  📖 보조 사전 연결 — 도메인 '{domain}' "
        f"| 언어 노드 {', '.join(_lang_nodes(lang_code))}"
    ]
    hit = 0
    miss: List[str] = []
    for f in fields:
        b, p = field_phrases(domain, f, lang_code)
        v = value_anchor_phrases(domain, f)
        if b or p or v:
            hit += 1
            lines.append(
                f"    · {f:<26} bias+{len(b):3d} prej+{len(p):3d} value+{len(v):3d}"
            )
        else:
            miss.append(f)
    lines.insert(1, f"  📖 증강 {hit}/{len(fields)}개 필드")
    if miss:
        lines.append(
            f"    ⚪ 보조 사전에 대응 노드가 없는 필드: {', '.join(miss)}"
        )
    return lines