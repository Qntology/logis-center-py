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


SCHEMA_FIELD_ANCHOR_LIMIT = 12


def _schema_field_anchors(data: dict, lang_code: str = "") -> List[str]:
    fields = data.get("fields")
    if not isinstance(fields, dict) or not fields:
        return []

    try:
        from text_pipeline.field_bank import is_junk_phrase, pick_lang_values
    except Exception:
        return []

    out: List[str] = []
    seen = set()

    for name, definition in fields.items():
        picked = ""
        if isinstance(definition, dict):
            for key in ("semantic", "label"):
                for raw in pick_lang_values(definition.get(key), lang_code):
                    text = str(raw or "").strip()
                    if not text or is_junk_phrase(text):
                        continue
                    picked = text
                    break
                if picked:
                    break
        if not picked:
            picked = str(name).replace("_", " ").strip()
        if not picked:
            continue

        low = picked.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(picked)
        if len(out) >= SCHEMA_FIELD_ANCHOR_LIMIT:
            break

    return out


def _absorb_schema(
    data: dict,
    stem: str,
    lang_code: str,
    groups: Dict[str, Dict],
    file_map: Dict[str, str],
    ref: str,
) -> None:
    domain = str(data.get("domain", "") or stem)
    doc_type = str(data.get("doc_type", "") or stem)
    code = str(data.get("code", "") or doc_type)

    declared = data.get("doc_anchors")
    if isinstance(declared, str):
        declared = [declared]
    anchors = [str(a).strip() for a in (declared or []) if str(a or "").strip()]

    if not anchors:
        anchors = [
            f"a {doc_type.replace('_', ' ')} document",
            f"a {domain.replace('_', ' ')} business form",
        ]
        for a in _schema_field_anchors(data, lang_code):
            if a not in anchors:
                anchors.append(a)

    g = groups.setdefault(domain, {"anchors": set(), "codes": {}})
    g["anchors"].add(f"a {domain.replace('_', ' ')} document")
    g["codes"][code] = list(anchors)
    file_map[code] = ref


def load_doc_type_specs(
    schema_dir,
    lang_code: str = "",
    log: Optional[List[str]] = None,
) -> Tuple[List[GroupSpec], Dict[str, str]]:
    schema_dir = Path(schema_dir)
    groups: Dict[str, Dict] = {}
    file_map: Dict[str, str] = {}
    skipped: List[str] = []
    dictionaries: List[Path] = []

    for f in sorted(schema_dir.glob("*.json")):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            dictionaries.append(f)
            skipped.append(f"{f.name}(개별 스키마 아님 → 사전 후보)")
            continue

        if not isinstance(data, dict):
            skipped.append(f"{f.name}(객체 아님)")
            continue

        has_fields = isinstance(data.get("fields"), dict) and bool(data.get("fields"))
        has_ident = bool(data.get("doc_type") or data.get("code"))
        if not has_fields and not has_ident:
            dictionaries.append(f)
            skipped.append(f"{f.name}(개별 스키마 아님 → 사전 후보)")
            continue

        _absorb_schema(data, f.stem, lang_code, groups, file_map, f.name)

    if not groups and dictionaries:
        try:
            from core.trade_schema import load_trade_schemas
        except Exception as e:
            load_trade_schemas = None
            if log is not None:
                log.append(f"    ⏭ 무역 사전 로더를 불러오지 못했습니다: {e}")

        if load_trade_schemas is not None:
            for f in dictionaries:
                schemas, diag = load_trade_schemas(f, lang_code)
                if log is not None:
                    for line in diag:
                        log.append(line)
                for code, sch in schemas.items():
                    _absorb_schema(
                        sch, code, lang_code, groups, file_map, f"{f.name}#{code}"
                    )
                if schemas:
                    break

    specs = [
        GroupSpec(name, sorted(payload["anchors"]), payload["codes"])
        for name, payload in groups.items()
    ]

    if log is not None:
        total_codes = sum(len(s.codes) for s in specs)
        log.append(
            f"  📚 스키마 적재 — 그룹 {len(specs)}개 / 코드 {total_codes}개 "
            f"(경로 {schema_dir})"
        )
        for s in specs:
            log.append(
                f"    · {s.name:<14} 코드 {len(s.codes)}개: "
                + ", ".join(sorted(s.codes.keys())[:12])
            )
        if skipped:
            log.append(f"    ⏭ 개별 스키마 제외 {len(skipped)}개: "
                       + ", ".join(skipped[:6]))
        if not specs:
            log.append(
                "    ❌ 문서 유형 스펙이 0개입니다. schema/ 에 "
                "'domain'/'doc_type'/'fields' 스키마 JSON 을 두거나, "
                "'trade_schema.overlay' 를 가진 bias.json 을 두십시오."
            )

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
        self.failed: int = 0
        self.seen_dim: int = 0
        self.last_error: str = ""

    def get_many(
        self,
        phrases: Sequence[str],
        batch: int = 64,
    ) -> Dict[str, np.ndarray]:
        need = [p for p in phrases if p and p not in self.store]
        step = max(1, int(batch))

        for i in range(0, len(need), step):
            chunk = need[i: i + step]
            try:
                mat = self.embed_fn(chunk)
            except Exception as e:
                self.failed += len(chunk)
                self.last_error = str(e)
                continue

            if mat is None:
                self.failed += len(chunk)
                continue

            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.size == 0:
                self.failed += len(chunk)
                continue

            self.seen_dim = int(mat.shape[-1])
            if self.expect_dim and self.seen_dim != self.expect_dim:
                self.rejected += len(chunk)
                continue

            for j, p in enumerate(chunk):
                if j >= mat.shape[0]:
                    break
                v = _unit(mat[j])
                if float(np.linalg.norm(v)) > 1e-8:
                    self.store[p] = v

        return {p: self.store[p] for p in phrases if p in self.store}

    def report(self) -> str:
        parts: List[str] = []
        if self.rejected:
            parts.append(
                f"임베딩 공간 불일치 — 패치 격자 {self.expect_dim}차원 vs "
                f"텍스트 앵커 {self.seen_dim}차원 (앵커 {self.rejected}구 폐기)"
            )
        if self.failed:
            msg = f"임베딩 호출 실패 {self.failed}구"
            if self.last_error:
                msg += f" ({self.last_error})"
            parts.append(msg)
        return " | ".join(parts)


def _max_pool_over_patches(grid: VisionPatchGrid, vec: np.ndarray) -> float:
    if grid.embeddings.size == 0 or vec is None:
        return 0.0
    v = np.asarray(vec, dtype=np.float32).reshape(-1)
    if v.shape[-1] != int(grid.embeddings.shape[-1]):
        return 0.0
    sims = grid.embeddings @ v
    return float(np.max(sims))


TITLE_BAND_RATIO = 0.28
TITLE_GATE_WEIGHT = 2.0


def _grid_matrix(grid: VisionPatchGrid) -> Optional[np.ndarray]:
    emb = np.asarray(grid.embeddings, dtype=np.float32)
    if emb.size == 0:
        return None
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)


def title_band_matrix(grid: VisionPatchGrid) -> Tuple[Optional[np.ndarray], int, int]:
    pmat = _grid_matrix(grid)
    if pmat is None:
        return None, 0, 0
    rows = max(1, int(grid.rows))
    cols = max(1, int(grid.cols))
    band = max(2, int(round(rows * TITLE_BAND_RATIO)))
    band = min(band, rows)
    cut = min(int(pmat.shape[0]), band * cols)
    if cut <= 0:
        return None, 0, rows
    return pmat[:cut], band, rows


def _bank_profile(
    pmat: np.ndarray,
    vecs: Sequence[np.ndarray],
) -> Optional[np.ndarray]:
    dim = int(pmat.shape[-1])
    rows: List[np.ndarray] = []
    for v in (vecs or []):
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float32).reshape(-1)
        if int(arr.shape[-1]) != dim:
            continue
        rows.append(arr)
    if not rows:
        return None
    bank = np.stack(rows).astype(np.float32)
    bank = bank / np.maximum(np.linalg.norm(bank, axis=1, keepdims=True), 1e-8)
    return np.max(pmat @ bank.T, axis=1).astype(np.float32)


def bank_neutral_key_matrix(
    grid: VisionPatchGrid,
    banks: Dict[str, Sequence[np.ndarray]],
    prejudice: Optional[Sequence[np.ndarray]] = None,
    patch_matrix: Optional[np.ndarray] = None,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    empty = np.zeros((0, 0), dtype=np.float32)
    pmat = patch_matrix if patch_matrix is not None else _grid_matrix(grid)
    if pmat is None or pmat.size == 0 or not banks:
        return [], empty, empty

    keys: List[str] = []
    rows: List[np.ndarray] = []
    for k, vecs in banks.items():
        prof = _bank_profile(pmat, vecs)
        if prof is None:
            continue
        keys.append(k)
        rows.append(prof)

    if not rows:
        return [], empty, empty

    raw = np.stack(rows).astype(np.float32)
    mu = raw.mean(axis=1, keepdims=True)
    sd = float(np.sqrt(np.mean((raw - mu) ** 2)))
    sd = sd if sd > 1e-6 else 1.0
    net = (raw - mu) / sd

    prej = _bank_profile(pmat, prejudice or [])
    if prej is not None and int(prej.shape[0]) == int(raw.shape[1]):
        zp = np.maximum((prej - float(prej.mean())) / sd, 0.0)
        net = net - zp[None, :]

    if net.shape[0] >= 2:
        net = net - net.mean(axis=0, keepdims=True)

    return keys, net.astype(np.float32), raw.astype(np.float32)


def _text_axis_scores(
    banks: Dict[str, Sequence[str]],
    text_sample: str,
    text_embed_fn: Optional[Callable[[List[str]], np.ndarray]],
) -> Tuple[Dict[str, float], int, str]:
    sample = " ".join(str(text_sample or "").split())
    if not sample or text_embed_fn is None or not banks:
        return {}, 0, ""

    phrases: List[str] = []
    for anchors in banks.values():
        phrases.extend(str(a).strip() for a in (anchors or []) if str(a or "").strip())
    uniq = sorted(set(phrases))
    if not uniq:
        return {}, 0, ""

    try:
        doc = text_embed_fn([sample[:2000]])
        mat = text_embed_fn(uniq)
    except Exception as e:
        return {}, 0, f"텍스트 임베딩 실패: {e}"

    if doc is None or mat is None:
        return {}, 0, "텍스트 임베딩이 비었습니다."

    doc = np.asarray(doc, dtype=np.float32)
    mat = np.asarray(mat, dtype=np.float32)
    if doc.ndim == 1:
        doc = doc.reshape(1, -1)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if doc.size == 0 or mat.size == 0:
        return {}, 0, "텍스트 임베딩이 비었습니다."
    if int(doc.shape[-1]) != int(mat.shape[-1]) or int(mat.shape[0]) < len(uniq):
        return {}, 0, (
            f"텍스트 임베딩 불일치 (문서 {int(doc.shape[-1])}차원 / "
            f"앵커 {int(mat.shape[-1])}차원 × {int(mat.shape[0])}구)"
        )

    dv = doc[0]
    dv = dv / max(float(np.linalg.norm(dv)), 1e-8)
    mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
    sims = (mat @ dv).astype(np.float32)
    lookup = {p: float(sims[i]) for i, p in enumerate(uniq)}

    out: Dict[str, float] = {}
    for code, anchors in banks.items():
        vals = [lookup[str(a).strip()] for a in (anchors or [])
                if str(a).strip() in lookup]
        if vals:
            out[code] = float(max(vals))
    return out, len(uniq), ""


def classify_doc_type(
    grid: VisionPatchGrid,
    specs: Sequence[GroupSpec],
    embed_fn: Callable[[List[str]], np.ndarray],
    file_map: Optional[Dict[str, str]] = None,
    margin_threshold: float = 0.28,
    log: Optional[List[str]] = None,
    text_sample: str = "",
    text_embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> DocTypeVerdict:
    from core.nms import decisive_margin

    logs: List[str] = []

    def _say(m: str):
        logs.append(m)
        if log is not None:
            log.append(m)

    if not specs:
        _say(
            "⚪ 문서 유형 스펙이 없습니다 — schema/ 폴더에 "
            "'domain' / 'doc_type' / 'fields' 를 가진 스키마 JSON 이 필요합니다."
        )
        return DocTypeVerdict(logs=logs)

    cache = _AnchorCache(embed_fn, expect_dim=int(grid.dim))

    all_phrases: List[str] = list(CHROME_ANCHORS)
    for spec in specs:
        all_phrases.extend(spec.anchors)
        for anchors in spec.codes.values():
            all_phrases.extend(anchors)
    uniq = sorted(set(p for p in all_phrases if p))
    vecs = cache.get_many(uniq)

    _say(
        f"  📖 앵커 임베딩 {len(vecs)}/{len(uniq)}구 × {grid.dim}차원 "
        f"| 그룹 {len(specs)}개 | 패치 {grid.num_patches}개"
    )
    note = cache.report()
    if note:
        _say(f"  ⚠ {note}")

    try:
        from core import diagnostics
        diagnostics.describe_matrix("doc_type.patch_grid", grid.embeddings, _say)
        if vecs:
            diagnostics.describe_matrix(
                "doc_type.anchors",
                np.stack([v for v in list(vecs.values())[:64]]),
                _say,
            )
            diagnostics.describe_joint_space(
                grid.embeddings, vecs, _say, title="문서 유형 앵커 ↔ 패치"
            )
    except Exception as e:
        _say(f"  ⏭ 진단 생략: {e}")

    if not vecs:
        _say(
            "❌ 문서 유형 분류 중단 — 텍스트 앵커 임베딩을 한 구도 얻지 "
            "못했습니다. 위 차원 표기를 확인하세요."
        )
        return DocTypeVerdict(logs=logs)

    chrome_vecs = [vecs[p] for p in CHROME_ANCHORS if p in vecs]

    _say("═══ Doc Type NMS: Depth 1 (그룹) ═══")

    group_banks: Dict[str, List[np.ndarray]] = {}
    for spec in specs:
        bank = [vecs[p] for p in spec.anchors if p in vecs]
        for anchors in spec.codes.values():
            bank.extend(vecs[p] for p in anchors if p in vecs)
        if bank:
            group_banks[spec.name] = bank

    gkeys, gnet, graw = bank_neutral_key_matrix(grid, group_banks, chrome_vecs)
    if not gkeys:
        _say("⚪ 그룹 채점 결과가 없습니다.")
        return DocTypeVerdict(logs=logs)

    group_scores = sorted(
        ((k, float(gnet[i].max())) for i, k in enumerate(gkeys)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    graw_lookup = {k: float(graw[i].max()) for i, k in enumerate(gkeys)}

    for name, sc in group_scores:
        _say(
            f"  📐 그룹 {name:<14} net={sc:+.4f} "
            f"| raw={graw_lookup.get(name, 0.0):+.4f} "
            f"| 앵커 {len(group_banks.get(name, []))}구"
        )

    best_group, best_gscore = group_scores[0]
    gm12, gband, gdecisive = decisive_margin([s for _n, s in group_scores])
    _say(
        f"  👑 그룹 '{best_group}' — 마진 {gm12:+.4f} / 잡음대 {gband:.4f} "
        f"→ {'결정적' if gdecisive else '동률(하위 그룹도 후보 유지)'}"
    )

    _say("═══ Doc Type NMS: Depth 2 (코드) ═══")

    pool: Dict[str, List[str]] = {}
    for spec in specs:
        if spec.name == best_group:
            pool.update(spec.codes)

    if not gdecisive:
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
            group=best_group, group_score=best_gscore,
            group_margin=gm12, logs=logs,
        )

    code_banks: Dict[str, List[np.ndarray]] = {}
    for code, anchors in pool.items():
        bank = [vecs[p] for p in anchors if p in vecs]
        if bank:
            code_banks[code] = bank

    ckeys, cnet, craw = bank_neutral_key_matrix(grid, code_banks, chrome_vecs)
    if not ckeys:
        _say("⚪ 코드 앵커 임베딩을 얻지 못해 그룹 판정만 유지합니다.")
        return DocTypeVerdict(
            group=best_group, group_score=best_gscore,
            group_margin=gm12, logs=logs,
        )

    raw_lookup = {k: float(craw[i].max()) for i, k in enumerate(ckeys)}
    bank_n = {k: len(code_banks.get(k, [])) for k in ckeys}
    code_net = {k: float(cnet[i].max()) for i, k in enumerate(ckeys)}

    for code, sc in sorted(code_net.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        _say(
            f"  📐 [VISION CODE] {code:<14} Surprisal: {sc:+.4f} "
            f"| raw={raw_lookup.get(code, 0.0):+.4f} | 앵커 {bank_n.get(code, 0)}구"
        )
    _cm12, _cband, _cdec = decisive_margin(list(code_net.values()))
    _say(f"  📐 [VISION CODE MARGIN] 1·2위 격차 {_cm12:+.4f} | 잡음대 {_cband:.4f}")

    tmat, band_rows, total_rows = title_band_matrix(grid)
    title_net: Dict[str, float] = {}
    if tmat is not None:
        _say(
            f"  🔍 [TITLE GATE SCAN] 상단 밴드: {band_rows}행 / 전체 "
            f"{total_rows}행 | 스캔 패치 {int(tmat.shape[0])}개"
        )
        tkeys, tnet, _traw = bank_neutral_key_matrix(
            grid, code_banks, chrome_vecs, patch_matrix=tmat
        )
        title_net = {k: float(tnet[i].max()) for i, k in enumerate(tkeys)}
        for code, sc in sorted(
            title_net.items(), key=lambda kv: kv[1], reverse=True
        )[:10]:
            _say(f"     📐 [TITLE GATE] {code:<14} Surprisal: {sc:+.4f}")
        tm12, tband, _tdec = decisive_margin(list(title_net.values()))
        _say(
            f"     📐 [TITLE GATE MARGIN] 1·2위 격차 {tm12:+.4f} "
            f"| 잡음대 {tband:.4f}"
        )
    else:
        _say("  ⏭ [TITLE GATE] 상단 밴드를 추출하지 못해 생략합니다.")

    text_raw, text_phrases, text_err = _text_axis_scores(
        {k: pool.get(k, []) for k in ckeys}, text_sample, text_embed_fn
    )
    text_z: Dict[str, float] = {}
    if text_raw:
        tk = [k for k in ckeys if k in text_raw]
        tz = _z([text_raw[k] for k in tk])
        text_z = {k: tz[i] for i, k in enumerate(tk)}
        _say(
            f"  🔤 텍스트 축 가동 — OCR 표본 "
            f"{len(' '.join(str(text_sample or '').split()))}자 × 앵커 "
            f"{text_phrases}구 (비전과 독립된 다국어 코사인 축)"
        )
    elif text_err:
        _say(f"  ⏭ 텍스트 축 생략 — {text_err}")
    else:
        _say("  ⏭ 텍스트 축 생략 — OCR 표본 또는 텍스트 임베더가 없습니다.")

    fused = {
        k: code_net.get(k, 0.0)
        + TITLE_GATE_WEIGHT * title_net.get(k, 0.0)
        + text_z.get(k, 0.0)
        for k in ckeys
    }
    code_scores = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

    for code, sc in code_scores[:8]:
        _say(
            f"  📐 코드 {code:<14} 합={sc:+.4f} "
            f"| 코드 {code_net.get(code, 0.0):+.4f} "
            f"| 제목×{TITLE_GATE_WEIGHT:.0f} "
            f"{TITLE_GATE_WEIGHT * title_net.get(code, 0.0):+.4f} "
            f"| 텍스트 {text_z.get(code, 0.0):+.4f} "
            f"| raw={raw_lookup.get(code, 0.0):+.4f}"
        )

    best_code, best_cscore = code_scores[0]
    cm12, cband, cdecisive = decisive_margin([s for _c, s in code_scores])

    raw_margin = 0.0
    if len(code_scores) > 1:
        raw_margin = raw_lookup.get(best_code, 0.0) - raw_lookup.get(
            code_scores[1][0], 0.0
        )

    if cband <= 1e-6:
        needs_llm = cm12 < float(margin_threshold)
    else:
        needs_llm = not cdecisive

    if not text_z and raw_margin <= 0.0 and len(code_scores) > 1:
        _say(
            f"  🚧 '{best_code}' 는 원시 코사인 우위가 없습니다 "
            f"(raw 차 {raw_margin:+.4f}). 정규화가 만든 허위 마진이므로 "
            f"확정하지 않고 후보로만 남깁니다."
        )
        needs_llm = True

    if needs_llm:
        _say(
            f"  ⚠ 코드 마진 {cm12:+.4f} < 잡음대 {cband:.4f} "
            f"(raw 차 {raw_margin:+.4f}) → 동률입니다. "
            f"'{best_code}' 를 잠정 채택하고 후보를 남깁니다."
        )
    else:
        _say(
            f"  👑 코드 확정 '{best_code}' — 마진 {cm12:+.4f} ≥ "
            f"잡음대 {cband:.4f} (raw 차 {raw_margin:+.4f})"
        )

    schema_file = (file_map or {}).get(best_code, "")
    if not schema_file:
        _say(f"  ⚠ '{best_code}' 에 대응하는 스키마 파일을 찾지 못했습니다.")

    return DocTypeVerdict(
        group=best_group,
        group_score=best_gscore,
        group_margin=gm12,
        code=best_code,
        code_score=best_cscore,
        code_margin=cm12,
        schema_file=schema_file,
        candidates=code_scores,
        needs_llm=needs_llm,
        logs=logs,
    )