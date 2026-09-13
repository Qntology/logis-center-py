import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ABSENT_ROUNDS_FLOOR = 1
MIN_TERRITORY_PATCHES = 2
DEFAULT_PATCH_MARGIN = 0.12
DEFAULT_ROUNDS = 3
SUPPRESSION_DECAY = 0.55


def gumbel_expected_z(n: int) -> float:
    if n <= 1:
        return 0.0
    return math.sqrt(2.0 * math.log(float(n)))


class PatchVerdict:
    def __init__(
        self,
        index: int,
        winner: str,
        winner_score: float,
        rival: str,
        rival_score: float,
        chrome_score: float,
    ):
        self.index = int(index)
        self.winner = winner
        self.winner_score = float(winner_score)
        self.rival = rival
        self.rival_score = float(rival_score)
        self.chrome_score = float(chrome_score)

    @property
    def margin(self) -> float:
        return self.winner_score - self.rival_score

    @property
    def chrome_margin(self) -> float:
        return self.winner_score - self.chrome_score

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "winner": self.winner,
            "winner_score": round(self.winner_score, 4),
            "rival": self.rival,
            "rival_score": round(self.rival_score, 4),
            "margin": round(self.margin, 4),
            "chrome_margin": round(self.chrome_margin, 4),
        }

    def __repr__(self) -> str:
        return (
            f"<Patch#{self.index} {self.winner}({self.winner_score:+.3f}) "
            f"vs {self.rival}({self.rival_score:+.3f}) m={self.margin:+.3f}>"
        )


class FieldTerritory:
    def __init__(self, category: str):
        self.category = category
        self.patches: List[int] = []
        self.margins: List[float] = []
        self.peak: float = -np.inf
        self.peak_index: int = -1
        self.rivals: Dict[str, int] = {}
        self.absent: bool = False
        self.absent_reason: str = ""

    def add(self, verdict: PatchVerdict):
        self.patches.append(verdict.index)
        self.margins.append(verdict.margin)
        if verdict.winner_score > self.peak:
            self.peak = verdict.winner_score
            self.peak_index = verdict.index
        if verdict.rival:
            self.rivals[verdict.rival] = self.rivals.get(verdict.rival, 0) + 1

    @property
    def size(self) -> int:
        return len(self.patches)

    @property
    def mean_margin(self) -> float:
        if not self.margins:
            return 0.0
        return float(sum(self.margins) / len(self.margins))

    @property
    def top_rival(self) -> str:
        if not self.rivals:
            return ""
        return max(self.rivals.items(), key=lambda kv: kv[1])[0]

    def mask(self, n: int) -> np.ndarray:
        m = np.zeros((n,), dtype=bool)
        for i in self.patches:
            if 0 <= i < n:
                m[i] = True
        return m

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "patches": self.size,
            "peak": round(float(self.peak), 4) if self.peak > -np.inf else 0.0,
            "peak_index": self.peak_index,
            "mean_margin": round(self.mean_margin, 4),
            "top_rival": self.top_rival,
            "absent": self.absent,
            "absent_reason": self.absent_reason,
        }

    def __repr__(self) -> str:
        mark = "ABSENT" if self.absent else "OWN"
        return (
            f"<Territory[{mark}] {self.category} n={self.size} "
            f"peak={self.peak:+.3f} m̄={self.mean_margin:+.3f}>"
        )


class ArenaResult:
    def __init__(self):
        self.verdicts: List[PatchVerdict] = []
        self.territories: Dict[str, FieldTerritory] = {}
        self.present: List[str] = []
        self.absent: List[str] = []
        self.rounds: int = 0
        self.unclaimed: int = 0

    def territory_of(self, category: str) -> Optional[FieldTerritory]:
        return self.territories.get(category)

    def to_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "unclaimed": self.unclaimed,
            "present": list(self.present),
            "absent": list(self.absent),
            "territories": {
                k: v.to_dict() for k, v in self.territories.items()
            },
        }


def _stack_scores(
    categories: Sequence[str],
    score_map: Dict[str, np.ndarray],
    n: int,
) -> np.ndarray:
    mat = np.full((len(categories), n), -np.inf, dtype=np.float32)
    for r, c in enumerate(categories):
        arr = score_map.get(c)
        if arr is None:
            continue
        m = min(n, int(arr.shape[0]))
        mat[r, :m] = arr[:m]
    return mat


def compete_patches(
    categories: Sequence[str],
    score_map: Dict[str, np.ndarray],
    chrome: Optional[np.ndarray],
    n: int,
    patch_margin: float = DEFAULT_PATCH_MARGIN,
    suppressed: Optional[np.ndarray] = None,
) -> Tuple[List[PatchVerdict], int]:
    cats = list(categories)
    if not cats or n <= 0:
        return [], n

    mat = _stack_scores(cats, score_map, n)

    if suppressed is not None and suppressed.shape == mat.shape:
        mat = np.where(suppressed, -np.inf, mat)

    ch = np.zeros((n,), dtype=np.float32)
    if chrome is not None and chrome.size:
        m = min(n, int(chrome.shape[0]))
        ch[:m] = chrome[:m]

    out: List[PatchVerdict] = []
    unclaimed = 0

    order = np.argsort(-mat, axis=0)

    for i in range(n):
        col = mat[:, i]
        first = int(order[0, i])
        top = float(col[first])

        if not np.isfinite(top):
            unclaimed += 1
            continue

        second = ""
        second_score = -np.inf
        if len(cats) > 1:
            sec = int(order[1, i])
            val = float(col[sec])
            if np.isfinite(val):
                second = cats[sec]
                second_score = val

        if second_score == -np.inf:
            second_score = float(ch[i])

        chrome_val = float(ch[i])
        floor = max(second_score, chrome_val)

        if (top - floor) < float(patch_margin):
            unclaimed += 1
            continue

        out.append(
            PatchVerdict(
                index=i,
                winner=cats[first],
                winner_score=top,
                rival=second,
                rival_score=second_score,
                chrome_score=chrome_val,
            )
        )

    return out, unclaimed


def run_arena(
    categories: Sequence[str],
    score_map: Dict[str, np.ndarray],
    chrome: Optional[np.ndarray],
    n: int,
    patch_margin: float = DEFAULT_PATCH_MARGIN,
    min_territory: int = MIN_TERRITORY_PATCHES,
    rounds: int = DEFAULT_ROUNDS,
    log: Optional[List[str]] = None,
) -> ArenaResult:
    result = ArenaResult()
    cats = [c for c in categories if c in score_map]
    if not cats or n <= 0:
        if log is not None:
            log.append("  ⚪ NMS ARENA: 경쟁 대상이 없습니다.")
        return result

    suppressed = np.zeros((len(cats), n), dtype=bool)
    index_of = {c: i for i, c in enumerate(cats)}

    verdicts: List[PatchVerdict] = []
    unclaimed = n
    eliminated: List[Tuple[str, str]] = []

    for rnd in range(1, max(1, int(rounds)) + 1):
        verdicts, unclaimed = compete_patches(
            cats, score_map, chrome, n,
            patch_margin=patch_margin,
            suppressed=suppressed,
        )

        terr: Dict[str, FieldTerritory] = {c: FieldTerritory(c) for c in cats}
        for v in verdicts:
            terr[v.winner].add(v)

        weak = [
            c for c in cats
            if terr[c].size < int(min_territory)
        ]

        if log is not None:
            owned = sum(1 for c in cats if terr[c].size >= int(min_territory))
            log.append(
                f"  🥊 ROUND {rnd}: 소유 패치 {len(verdicts)}/{n} "
                f"| 무주공산 {unclaimed} | 영토 확보 필드 {owned}/{len(cats)}"
            )

        if not weak or rnd >= int(rounds):
            result.territories = terr
            break

        for c in weak:
            row = index_of[c]
            suppressed[row, :] = True
            eliminated.append((c, f"영토 {terr[c].size} < {min_territory}"))
            if log is not None:
                log.append(
                    f"  🚫 ABSENT '{c}' — 경쟁 영토 {terr[c].size}패치 "
                    f"(최소 {min_territory}) → 라운드 {rnd} 에서 탈락"
                )

        remaining = [c for c in cats if not suppressed[index_of[c], :].all()]
        if len(remaining) < 2:
            result.territories = terr
            break

        result.rounds = rnd

    result.rounds = max(result.rounds, 1)
    result.verdicts = verdicts
    result.unclaimed = unclaimed

    if not result.territories:
        result.territories = {c: FieldTerritory(c) for c in cats}
        for v in verdicts:
            result.territories[v.winner].add(v)

    drop_reason = dict(eliminated)
    for c, t in result.territories.items():
        if c in drop_reason:
            t.absent = True
            t.absent_reason = drop_reason[c]
        elif t.size < int(min_territory):
            t.absent = True
            t.absent_reason = f"영토 {t.size} < {min_territory}"

    result.present = [c for c, t in result.territories.items() if not t.absent]
    result.absent = [c for c, t in result.territories.items() if t.absent]

    if log is not None:
        log.append(
            f"  ✅ NMS ARENA 종료 — 존재 {len(result.present)}개 / "
            f"부재 {len(result.absent)}개 / 라운드 {result.rounds}"
        )
        for c in sorted(result.present):
            t = result.territories[c]
            log.append(
                f"     👑 {c:<26} 영토 {t.size:3d}패치 "
                f"| peak {t.peak:+.4f} | 평균마진 {t.mean_margin:+.4f} "
                f"| 최대경쟁자 '{t.top_rival or '-'}'"
            )
        for c in sorted(result.absent):
            t = result.territories[c]
            log.append(f"     ⚪ {c:<26} 부재 — {t.absent_reason}")

    return result


def territory_scores(
    result: ArenaResult,
    category: str,
    n: int,
) -> np.ndarray:
    out = np.full((n,), -np.inf, dtype=np.float32)
    t = result.territory_of(category)
    if t is None:
        return out
    lookup = {v.index: v for v in result.verdicts if v.winner == category}
    for i in t.patches:
        v = lookup.get(i)
        if v is None:
            continue
        out[i] = float(v.margin)
    return out