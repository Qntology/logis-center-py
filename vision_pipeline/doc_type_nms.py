import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .patch_grid import VisionPatchGrid

CHROME_ANCHORS: Tuple[str, ...] = (
    "company logo emblem brand graphic",
    "official stamp seal signature mark",
    "blank white margin empty paper area",
    "table border ruling line grid frame",
    "barcode qr code square block",
    "photograph picture illustration image",
    "page number footer header decoration",
)


def gumbel_expected_z(n: int) -> float:
    if n <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(float(n)))


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def _z(values: Sequence[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return [0.0] * len(values)
    return [float((v - mean) / std) for v in values]


class GroupSpec:
    def __init__(self, name: str, anchors: Sequence[str], codes: Dict[str, Sequence[str]]):
        self.name = name
        self.anchors = list(anchors)
        self.codes = {k: list(v) for k, v in (codes or {}).items()}

    def code_names(self) -> List[str]:
        return list(self.codes.keys())


class DocTypeVerdict:
    def __init__(
        self,
        group: str = "",
        group_score: float = 0.0,
        group_margin: float = 0.0,
        code: str = "",
        code_score: float = 0.0,
        code_margin: float = 0.0,
        schema_file: str = "",
        candidates: Optional[List[Tuple[str, float]]] = None,
        needs_llm: bool = False,
        logs: Optional[List[str]] = None,
    ):
        self.group = group
        self.group_score = float(group_score)
        self.group_margin = float(group_margin)
        self.code = code
        self.code_score = float(code_score)
        self.code_margin = float(code_margin)
        self.schema_file = schema_file
        self.candidates = candidates or []
        self.needs_llm = bool(needs_llm)
        self.logs = logs or []

    def to_dict(self) -> dict:
        return {
            "group": self.group,
            "group_score": round(self.group_score, 4),
            "group_margin": round(self.group_margin, 4),
            "code": self.code,
            "code_score": round(self.code_score, 4),
            "code_margin": round(self.code_margin, 4),
            "schema_file": self.schema_file,
            "needs_llm": self.needs_llm,
            "candidates": [
                {"code": c, "score": round(s, 4)} for c, s in self.candidates[:10]
            ],
        }

    def __repr__(self) -> str:
        return (
            f"<DocType {self.group}/{self.code} "
            f"gm={self.group_margin:+.3f} cm={self.code_margin:+.3f}>"
        )


def load_doc_type_specs(schema_dir) -> Tuple[List[GroupSpec], Dict[str, str]]:
    schema_dir = Path(schema_dir)
    groups: Dict[str, Dict] = {}
    file_map: Dict[str, str] = {}

    for f in sorted(schema_dir.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue

        domain = str(data.get("domain", "") or f.stem)
        doc_type = str(data.get("doc_type", "") or f.stem)
        code = str(data.get("code", "") or doc_type)

        anchors = data.get("doc_anchors")
        if not anchors:
            anchors = [
                f"a {doc_type.replace('_', ' ')} document",
                f"a {domain.replace('_', ' ')} business form",
            ]
        elif isinstance(anchors, str):
            anchors = [anchors]

        g = groups.setdefault(domain, {"anchors": set(), "codes": {}})
        g["anchors"].add(f"a {domain.replace('_', ' ')} document")
        g["codes"][code] = list(anchors)
        file_map[code] = f.name

    specs = [
        GroupSpec(name, sorted(payload["anchors"]), payload["codes"])
        for name, payload in groups.items()
    ]
    return specs, file_map


class _AnchorCache:
    def __init__(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        expect_dim: int = 0,
    ):
        self.embed_fn = embed_fn
        self.expect_dim = int(expect_dim)
        self.store: Dict[str, np.ndarray] = {}
        self.rejected: int = 0
        self.seen_dim: int = 0

    def get_many(self, phrases: Sequence[str]) -> Dict[str, np.ndarray]:
        need = [p for p in phrases if p and p not in self.store]
        if need:
            try:
                mat = self.embed_fn(need)
            except Exception:
                mat = None
            if mat is not None:
                mat = np.asarray(mat, dtype=np.float32)
                if mat.ndim == 1:
                    mat = mat.reshape(1, -1)
                self.seen_dim = int(mat.shape[-1]) if mat.size else 0
                if self.expect_dim and self.seen_dim != self.expect_dim:
                    self.rejected += len(need)
                    return {p: self.store[p] for p in phrases if p in self.store}
                for i, p in enumerate(need):
                    if i >= mat.shape[0]:
                        break
                    v = _unit(mat[i])
                    if float(np.linalg.norm(v)) > 1e-8:
                        self.store[p] = v
        return {p: self.store[p] for p in phrases if p in self.store}

    def report(self) -> str:
        if not self.rejected:
            return ""
        return (
            f"임베딩 공간 불일치 — 패치 격자 {self.expect_dim}차원 vs "
            f"텍스트 앵커 {self.seen_dim}차원 (앵커 {self.rejected}구 폐기)"
        )


def _max_pool_over_patches(grid: VisionPatchGrid, vec: np.ndarray) -> float:
    if grid.embeddings.size == 0 or vec is None:
        return 0.0
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[-1] != int(grid.embeddings.shape[-1]):
        return 0.0
    sims = grid.embeddings @ v
    return float(np.max(sims))


def _bank_net_score(
    grid: VisionPatchGrid,
    bias_vecs: Sequence[np.ndarray],
    prej_vecs: Sequence[np.ndarray],
) -> Tuple[float, int]:
    if not bias_vecs:
        return 0.0, 0
    best_bias = max(_max_pool_over_patches(grid, v) for v in bias_vecs)
    best_prej = 0.0
    if prej_vecs:
        best_prej = max(0.0, max(_max_pool_over_patches(grid, v) for v in prej_vecs))
    return best_bias - best_prej, len(bias_vecs)


def classify_doc_type(
    grid: VisionPatchGrid,
    specs: Sequence[GroupSpec],
    embed_fn: Callable[[List[str]], np.ndarray],
    file_map: Optional[Dict[str, str]] = None,
    margin_threshold: float = 0.28,
    log: Optional[List[str]] = None,
) -> DocTypeVerdict:
    logs: List[str] = []

    def _say(m: str):
        logs.append(m)
        if log is not None:
            log.append(m)

    if not specs:
        _say("⚪ 문서 유형 스펙이 없습니다.")
        return DocTypeVerdict(logs=logs)

    cache = _AnchorCache(embed_fn, expect_dim=int(grid.dim))

    all_phrases: List[str] = list(CHROME_ANCHORS)
    for spec in specs:
        all_phrases.extend(spec.anchors)
        for anchors in spec.codes.values():
            all_phrases.extend(anchors)
    vecs = cache.get_many(sorted(set(all_phrases)))

    if not vecs:
        msg = cache.report() or "텍스트 앵커 임베딩을 얻지 못했습니다."
        _say(f"❌ 문서 유형 분류 중단 — {msg}")
        return DocTypeVerdict(logs=logs)

    chrome_vecs = [vecs[p] for p in CHROME_ANCHORS if p in vecs]

    _say("═══ Doc Type NMS: Depth 1 (그룹) ═══")

    group_raw: List[Tuple[str, float, int]] = []
    for spec in specs:
        bias = [vecs[p] for p in spec.anchors if p in vecs]
        for anchors in spec.codes.values():
            bias.extend(vecs[p] for p in anchors if p in vecs)

        prej = list(chrome_vecs)
        for other in specs:
            if other.name == spec.name:
                continue
            prej.extend(vecs[p] for p in other.anchors if p in vecs)

        net, n = _bank_net_score(grid, bias, prej)
        group_raw.append((spec.name, net, n))

    if not group_raw:
        _say("⚪ 그룹 채점 결과가 없습니다.")
        return DocTypeVerdict(logs=logs)

    zs = _z([v for _n, v, _c in group_raw])
    group_scores = [
        (name, z - gumbel_expected_z(max(1, cnt)))
        for (name, _raw, cnt), z in zip(group_raw, zs)
    ]
    group_scores.sort(key=lambda kv: kv[1], reverse=True)

    for name, sc in group_scores:
        _say(f"  📐 그룹 {name:<16} {sc:+.4f}")

    best_group, best_gscore = group_scores[0]
    gmargin = best_gscore - (group_scores[1][1] if len(group_scores) > 1 else best_gscore)
    _say(f"  👑 그룹 확정 '{best_group}' (margin {gmargin:+.4f})")

    _say("═══ Doc Type NMS: Depth 2 (코드) ═══")

    pool: Dict[str, List[str]] = {}
    for spec in specs:
        if spec.name == best_group:
            pool.update(spec.codes)
    for name, sc in group_scores[1:]:
        if sc <= 0.0:
            continue
        for spec in specs:
            if spec.name != name:
                continue
            for code, anchors in spec.codes.items():
                pool.setdefault(code, anchors)

    if not pool:
        _say("⚪ 후보 코드가 없습니다.")
        return DocTypeVerdict(
            group=best_group, group_score=best_gscore, group_margin=gmargin, logs=logs
        )

    code_raw: List[Tuple[str, float, int]] = []
    raw_lookup: Dict[str, float] = {}
    for code, anchors in pool.items():
        bias = [vecs[p] for p in anchors if p in vecs]
        prej = list(chrome_vecs)
        for other_code, other_anchors in pool.items():
            if other_code == code:
                continue
            prej.extend(vecs[p] for p in other_anchors if p in vecs)
        net, n = _bank_net_score(grid, bias, prej)
        code_raw.append((code, net, n))
        raw_lookup[code] = float(net)

    zs = _z([v for _c, v, _n in code_raw])
    code_scores = [
        (code, z - gumbel_expected_z(max(1, cnt)))
        for (code, _raw, cnt), z in zip(code_raw, zs)
    ]
    code_scores.sort(key=lambda kv: kv[1], reverse=True)

    for code, sc in code_scores[:8]:
        _say(f"  📐 코드 {code:<16} z={sc:+.4f} | raw={raw_lookup.get(code, 0.0):+.4f}")

    best_code, best_cscore = code_scores[0]
    cmargin = best_cscore - (code_scores[1][1] if len(code_scores) > 1 else best_cscore)

    raw_margin = 0.0
    if len(code_scores) > 1:
        raw_margin = raw_lookup.get(best_code, 0.0) - raw_lookup.get(
            code_scores[1][0], 0.0
        )

    if raw_margin <= 0.0 and len(code_scores) > 1:
        _say(
            f"  🚧 '{best_code}' 의 원시 코사인 우위가 없습니다 "
            f"(raw 차 {raw_margin:+.4f}) → z-정규화가 만든 허위 마진"
        )
        needs_llm = True
    else:
        needs_llm = cmargin < margin_threshold

    if needs_llm:
        _say(
            f"  ⚠ 코드 마진 z={cmargin:+.4f} raw={raw_margin:+.4f} "
            f"(요구 {margin_threshold:.2f}) → LLM 재판정 후보 나열"
        )
    else:
        _say(
            f"  👑 코드 확정 '{best_code}' "
            f"(z마진 {cmargin:+.4f} / raw마진 {raw_margin:+.4f})"
        )

    schema_file = (file_map or {}).get(best_code, "")

    return DocTypeVerdict(
        group=best_group,
        group_score=best_gscore,
        group_margin=gmargin,
        code=best_code,
        code_score=best_cscore,
        code_margin=cmargin,
        schema_file=schema_file,
        candidates=code_scores,
        needs_llm=needs_llm,
        logs=logs,
    )