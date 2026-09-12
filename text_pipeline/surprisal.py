import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .field_bank import FieldBank, FieldEntry

BANK_PENALTY_DAMP = 0.25


def gumbel_expected_z(n: int) -> float:
    if n <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(float(n)))


def l2_normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def weighted_max_pool(
    vec: np.ndarray,
    bank: np.ndarray,
    weights: np.ndarray,
) -> Tuple[float, int]:
    if bank.size == 0 or bank.shape[0] == 0:
        return 0.0, -1
    sims = bank @ vec
    if weights.size == bank.shape[0]:
        sims = sims * weights
    idx = int(np.argmax(sims))
    return float(sims[idx]), idx


class SurprisalScore:
    def __init__(
        self,
        field_name: str,
        raw_bias: float,
        raw_prejudice: float,
        net: float,
        z: float,
        penalty: float,
        surprisal: float,
        bank_size: int,
        best_phrase: str = "",
        raw_label: float = 0.0,
        label_phrase: str = "",
    ):
        self.field_name = field_name
        self.raw_bias = float(raw_bias)
        self.raw_prejudice = float(raw_prejudice)
        self.net = float(net)
        self.z = float(z)
        self.penalty = float(penalty)
        self.surprisal = float(surprisal)
        self.bank_size = int(bank_size)
        self.best_phrase = best_phrase
        self.raw_label = float(raw_label)
        self.label_phrase = label_phrase

    @property
    def label_dominant(self) -> bool:
        return self.raw_label > self.raw_bias

    def to_dict(self) -> dict:
        return {
            "field_name": self.field_name,
            "raw_bias": round(self.raw_bias, 6),
            "raw_prejudice": round(self.raw_prejudice, 6),
            "raw_label": round(self.raw_label, 6),
            "net": round(self.net, 6),
            "z": round(self.z, 6),
            "penalty": round(self.penalty, 6),
            "surprisal": round(self.surprisal, 6),
            "bank_size": self.bank_size,
            "best_phrase": self.best_phrase,
            "label_phrase": self.label_phrase,
            "label_dominant": self.label_dominant,
        }

    def __repr__(self) -> str:
        mark = " [LABEL]" if self.label_dominant else ""
        return (
            f"<Surprisal {self.field_name} net={self.net:+.4f} "
            f"z={self.z:+.4f} −{self.penalty:+.4f} = {self.surprisal:+.4f}{mark}>"
        )


def surprisal_dual_scores(
    vec: np.ndarray,
    bank: FieldBank,
    field_names: Optional[Sequence[str]] = None,
) -> List[SurprisalScore]:
    if vec is None or vec.size == 0:
        return []

    v = l2_normalize(np.asarray(vec, dtype=np.float32))
    names = list(field_names) if field_names else bank.field_names

    raw_nets: List[float] = []
    details: List[Tuple[str, float, float, int, str, float, str]] = []

    for name in names:
        entry: Optional[FieldEntry] = bank.get(name)
        if entry is None:
            continue
        bmat, bwts = entry.bias_matrix()
        if bmat.size == 0:
            continue
        bias_sim, bidx = weighted_max_pool(v, bmat, bwts)
        best_phrase = entry.bias[bidx].text if 0 <= bidx < len(entry.bias) else ""

        pmat, pwts = entry.prejudice_matrix()
        prej_sim = 0.0
        if pmat.size:
            prej_sim, _pidx = weighted_max_pool(v, pmat, pwts)
            prej_sim = max(0.0, prej_sim)

        lmat, lwts = entry.label_matrix()
        label_sim = 0.0
        label_phrase = ""
        if lmat.size:
            label_sim, lidx = weighted_max_pool(v, lmat, lwts)
            label_sim = max(0.0, label_sim)
            if 0 <= lidx < len(entry.label):
                label_phrase = entry.label[lidx].text

        net = bias_sim - prej_sim
        raw_nets.append(net)
        details.append((
            name, bias_sim, prej_sim, entry.bank_size(),
            best_phrase, label_sim, label_phrase,
        ))

    if not details:
        return []

    arr = np.asarray(raw_nets, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())

    penalties = np.asarray(
        [gumbel_expected_z(max(1, d[3])) for d in details], dtype=np.float64
    )
    penalty_mu = float(penalties.mean())

    out: List[SurprisalScore] = []
    for idx, (
        name, bias_sim, prej_sim, bank_size, best_phrase, label_sim, label_phrase
    ) in enumerate(details):
        net = raw_nets[idx]
        z = 0.0 if std < 1e-8 else (net - mean) / std
        penalty = (float(penalties[idx]) - penalty_mu) * BANK_PENALTY_DAMP
        out.append(
            SurprisalScore(
                field_name=name,
                raw_bias=bias_sim,
                raw_prejudice=prej_sim,
                net=net,
                z=z,
                penalty=penalty,
                surprisal=z - penalty,
                bank_size=bank_size,
                best_phrase=best_phrase,
                raw_label=label_sim,
                label_phrase=label_phrase,
            )
        )

    out.sort(key=lambda s: s.surprisal, reverse=True)
    return out


def strip_label_prefix(text: str, label: str) -> str:
    body = (text or "").strip()
    lab = (label or "").strip()
    if not body or not lab:
        return body

    low_body = body.lower()
    low_lab = lab.lower()

    if low_body == low_lab:
        return ""

    if low_body.startswith(low_lab):
        rest = body[len(lab):]
        rest = rest.lstrip(" \t:：=-—·|,")
        return rest.strip()

    return body


def score_chunks(
    chunks,
    vectors: np.ndarray,
    bank: FieldBank,
    gate: float = 0.0,
) -> Dict[int, List[SurprisalScore]]:
    results: Dict[int, List[SurprisalScore]] = {}
    if vectors is None or vectors.size == 0:
        return results

    for idx, chunk in enumerate(chunks):
        if idx >= vectors.shape[0]:
            break
        scores = surprisal_dual_scores(vectors[idx], bank)
        if not scores:
            continue

        top = scores[0]
        margin = (
            top.surprisal - scores[1].surprisal if len(scores) > 1 else top.surprisal
        )

        passed = [s for s in scores if s.surprisal > gate]
        if not passed:
            chunk.margin = margin
            continue

        results[idx] = passed
        chunk.score = top.surprisal
        chunk.property = top.field_name
        chunk.margin = margin

        if not chunk.value_part:
            chunk.value_part = strip_label_prefix(chunk.text, top.label_phrase)

    return results


def build_score_matrix(
    chunks,
    scored: Dict[int, List[SurprisalScore]],
    field_names: Sequence[str],
) -> np.ndarray:
    mat = np.full((len(chunks), len(field_names)), -np.inf, dtype=np.float32)
    index = {name: i for i, name in enumerate(field_names)}
    for ci, scores in scored.items():
        for s in scores:
            fi = index.get(s.field_name)
            if fi is None:
                continue
            mat[ci, fi] = s.surprisal
    return mat