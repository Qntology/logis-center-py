from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .doc_type_nms import gumbel_expected_z
from .patch_grid import VisionPatchGrid


class GroundingClaim:
    def __init__(
        self,
        category: str,
        field: str,
        value: str,
        bbox: Tuple[int, int, int, int],
    ):
        self.category = category
        self.field = field
        self.value = value
        self.bbox = tuple(int(v) for v in bbox)


class GroundingVerdict:
    def __init__(
        self,
        category: str,
        field: str,
        value: str,
        surprisal_in: float,
        surprisal_out: float,
        accepted: bool,
        reason: str = "",
        top_patch: int = -1,
    ):
        self.category = category
        self.field = field
        self.value = value
        self.surprisal_in = float(surprisal_in)
        self.surprisal_out = float(surprisal_out)
        self.accepted = bool(accepted)
        self.reason = reason
        self.top_patch = int(top_patch)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "field": self.field,
            "value": self.value,
            "surprisal_in": round(self.surprisal_in, 4),
            "surprisal_out": round(self.surprisal_out, 4),
            "accepted": self.accepted,
            "reason": self.reason,
            "top_patch": self.top_patch,
        }

    def __repr__(self) -> str:
        mark = "OK" if self.accepted else "DROP"
        return (
            f"<Grounding[{mark}] {self.field}='{self.value[:24]}' "
            f"in={self.surprisal_in:+.3f} out={self.surprisal_out:+.3f}>"
        )


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > eps else v


def verify_claims(
    claims: Sequence[GroundingClaim],
    grid: VisionPatchGrid,
    embed_fn: Callable[[List[str]], np.ndarray],
    nlp=None,
    log: Optional[List[str]] = None,
) -> List[GroundingVerdict]:
    out: List[GroundingVerdict] = []

    if not claims or grid.embeddings.size == 0:
        return out

    texts: List[str] = []
    normalized = 0
    for c in claims:
        v = c.value.strip()
        if v and nlp is not None and getattr(nlp, "available", False):
            try:
                lemma = nlp.canonicalize(v)
            except Exception:
                lemma = ""
            if lemma and lemma != v:
                normalized += 1
                v = f"{v} {lemma}"
        texts.append(v)

    if normalized and log is not None:
        log.append(f"  🧬 접지 검증: {normalized}건 lemma 정규화 적용")

    valid_idx = [i for i, t in enumerate(texts) if t]
    if not valid_idx:
        return out

    try:
        mat = embed_fn([texts[i] for i in valid_idx])
    except Exception as e:
        if log is not None:
            log.append(f"  ⚠ 접지 검증 임베딩 실패: {e}")
        mat = None

    if mat is None:
        for c in claims:
            out.append(
                GroundingVerdict(
                    c.category, c.field, c.value, 0.0, 0.0,
                    accepted=True, reason="임베딩 실패 — 검증 보류",
                )
            )
        return out

    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)

    if mat.size and int(mat.shape[-1]) != int(grid.embeddings.shape[-1]):
        if log is not None:
            log.append(
                f"  ⏭ 접지 검증 생략 — 패치 격자 {int(grid.embeddings.shape[-1])}차원 vs "
                f"텍스트 임베딩 {int(mat.shape[-1])}차원 불일치"
            )
        for c in claims:
            out.append(
                GroundingVerdict(
                    c.category, c.field, c.value, 0.0, 0.0,
                    accepted=True, reason="임베딩 공간 불일치 — 검증 보류",
                )
            )
        return out

    vec_map: Dict[int, np.ndarray] = {}
    for k, i in enumerate(valid_idx):
        if k < mat.shape[0]:
            vec_map[i] = _unit(mat[k])

    dropped = 0

    for i, claim in enumerate(claims):
        if not claim.value.strip():
            continue

        vec = vec_map.get(i)
        if vec is None:
            out.append(
                GroundingVerdict(
                    claim.category, claim.field, claim.value, 0.0, 0.0,
                    accepted=True, reason="벡터 없음 — 검증 보류",
                )
            )
            continue

        sims = grid.embeddings @ vec
        mean = float(sims.mean())
        std = max(float(sims.std()), 1e-6)

        inside = grid.indices_in_bbox(claim.bbox)
        if not inside:
            out.append(
                GroundingVerdict(
                    claim.category, claim.field, claim.value, 0.0, 0.0,
                    accepted=True, reason="대응 패치 없음 — 검증 보류",
                )
            )
            continue

        inside_set = set(inside)
        outside = [j for j in range(sims.size) if j not in inside_set]

        in_arr = sims[inside]
        top_local = int(np.argmax(in_arr))
        top_patch = inside[top_local]
        s_in = (float(in_arr[top_local]) - mean) / std - gumbel_expected_z(len(inside))

        if outside:
            out_arr = sims[outside]
            s_out = (
                (float(np.max(out_arr)) - mean) / std - gumbel_expected_z(len(outside))
            )
        else:
            s_out = -np.inf

        if s_in <= 0.0:
            dropped += 1
            if log is not None:
                log.append(
                    f"  🚫 UNGROUNDED '{claim.field}' = \"{claim.value[:30]}\" "
                    f"(in {s_in:+.4f} ≤ 0) → 폐기"
                )
            out.append(
                GroundingVerdict(
                    claim.category, claim.field, claim.value, s_in, s_out,
                    accepted=False, reason="크롭 내부 접지 실패", top_patch=top_patch,
                )
            )
            continue

        reason = "접지 확인"
        if s_out > s_in:
            reason = "다른 영역 소유 가능"
            if log is not None:
                log.append(
                    f"  ⚠️ CROSS-CROP '{claim.field}' = \"{claim.value[:30]}\" "
                    f"(in {s_in:+.4f} < out {s_out:+.4f})"
                )

        out.append(
            GroundingVerdict(
                claim.category, claim.field, claim.value, s_in, s_out,
                accepted=True, reason=reason, top_patch=top_patch,
            )
        )

    if log is not None:
        log.append(
            f"  ✅ VALUE GROUNDING 검증 {len(out)}건 / "
            f"유지 {len(out) - dropped} / 폐기 {dropped}"
        )

    return out


def apply_verdicts(
    record: Dict[str, object],
    verdicts: Sequence[GroundingVerdict],
) -> Dict[str, object]:
    drop = {v.field for v in verdicts if not v.accepted}
    out = dict(record)
    for f in drop:
        if f in out:
            out[f] = None
    return out