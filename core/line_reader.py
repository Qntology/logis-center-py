import re
import unicodedata
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

LINE_MIN_H = 8
LINE_PAD_Y = 3
LINE_PAD_X = 4
LINE_MAX = 40

BAND_GATE_RATIO = 0.22
COL_GATE_RATIO = 0.30

BAND_SPLIT_RATIO = 2.2
BAND_SPLIT_ROUNDS = 3
BAND_GATE_STEP = 0.12
BAND_SPLIT_MIN_H = LINE_MIN_H * 2

WINDOW_LINES = 1
WINDOW_STRIDE = 1
WINDOW_OVERLAP = 1

MIN_LINE_UPSCALE = 1.6
MAX_LINE_UPSCALE = 4.0
TARGET_LINE_PX = 40.0

MERGE_MIN_OVERLAP = 2
VOTE_MIN_LEN = 2

WS_RE = re.compile(r"[\s\u00a0]+")


class TextLine:
    def __init__(
        self,
        index: int,
        bbox: Tuple[int, int, int, int],
        image: Image.Image,
    ):
        self.index = int(index)
        self.bbox = tuple(int(v) for v in bbox)
        self.image = image
        self.text = ""
        self.score = 0.0
        self.source = ""

    @property
    def height(self) -> int:
        return int(self.bbox[3] - self.bbox[1])

    @property
    def width(self) -> int:
        return int(self.bbox[2] - self.bbox[0])

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "bbox": list(self.bbox),
            "text": self.text,
            "score": round(self.score, 4),
            "source": self.source,
        }

    def __repr__(self) -> str:
        return f"<Line#{self.index} {self.bbox} '{self.text[:24]}'>"


class ReadWindow:
    def __init__(self, start: int, end: int, lines: Sequence[TextLine]):
        self.start = int(start)
        self.end = int(end)
        self.lines = list(lines)
        self.text = ""
        self.raw = ""

    @property
    def span(self) -> int:
        return self.end - self.start

    def bbox(self) -> Tuple[int, int, int, int]:
        xs0 = min(l.bbox[0] for l in self.lines)
        ys0 = min(l.bbox[1] for l in self.lines)
        xs1 = max(l.bbox[2] for l in self.lines)
        ys1 = max(l.bbox[3] for l in self.lines)
        return (xs0, ys0, xs1, ys1)

    def crop(self, image: Image.Image) -> Image.Image:
        x0, y0, x1, y1 = self.bbox()
        x0 = max(0, x0 - LINE_PAD_X)
        y0 = max(0, y0 - LINE_PAD_Y)
        x1 = min(image.width, x1 + LINE_PAD_X)
        y1 = min(image.height, y1 + LINE_PAD_Y)
        if x1 <= x0 or y1 <= y0:
            return Image.new("RGB", (1, 1), (255, 255, 255))
        return image.crop((x0, y0, x1, y1))

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "bbox": list(self.bbox()),
            "text": self.text,
        }

    def __repr__(self) -> str:
        return f"<Window[{self.start}:{self.end}] '{self.text[:28]}'>"


def _bands_from_profile(
    prof: np.ndarray,
    height: int,
    gate: float,
) -> List[Tuple[int, int]]:
    hot = prof > gate
    out: List[Tuple[int, int]] = []
    start = -1
    for y in range(int(height)):
        if hot[y] and start < 0:
            start = y
        elif not hot[y] and start >= 0:
            if y - start >= LINE_MIN_H:
                out.append((start, y))
            start = -1
    if start >= 0 and int(height) - start >= LINE_MIN_H:
        out.append((start, int(height)))
    return out


def _refine_tall_bands(
    prof: np.ndarray,
    bands: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    cur = list(bands)

    for _rnd in range(BAND_SPLIT_ROUNDS):
        heights = sorted(b[1] - b[0] for b in cur)
        if not heights:
            break
        med = float(heights[len(heights) // 2])
        if med <= 0.0:
            break

        limit = med * BAND_SPLIT_RATIO
        if not any((b[1] - b[0]) > limit for b in cur):
            break

        nxt: List[Tuple[int, int]] = []
        changed = False

        for (y0, y1) in cur:
            span = y1 - y0
            if span <= limit or span < BAND_SPLIT_MIN_H:
                nxt.append((y0, y1))
                continue

            seg = prof[y0:y1]
            sb = float(np.percentile(seg, 20))
            sp = float(seg.max())
            if sp - sb < 2.0:
                nxt.append((y0, y1))
                continue

            best: List[Tuple[int, int]] = []
            for k in range(1, 4):
                gate = sb + (sp - sb) * (BAND_GATE_RATIO + BAND_GATE_STEP * k)
                cand = _bands_from_profile(seg, span, gate)
                if len(cand) > len(best):
                    best = cand
                if len(best) >= 2:
                    break

            if len(best) >= 2:
                nxt.extend((y0 + a, y0 + b) for a, b in best)
                changed = True
            else:
                nxt.append((y0, y1))

        cur = sorted(nxt, key=lambda b: b[0])
        if not changed:
            break

    return cur


def _row_bands(image: Image.Image) -> List[Tuple[int, int]]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    if h < LINE_MIN_H * 2 or w < 8:
        return []

    ink = 255.0 - gray
    prof = ink.mean(axis=1)

    base = float(np.percentile(prof, 12))
    peak = float(prof.max())
    if peak - base < 3.0:
        return []

    gate = base + (peak - base) * BAND_GATE_RATIO
    bands = _bands_from_profile(prof, h, gate)
    if not bands:
        return []

    return _refine_tall_bands(prof, bands)


def _trim_columns(
    image: Image.Image,
    y0: int,
    y1: int,
) -> Tuple[int, int]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = gray.shape
    y0 = max(0, min(int(y0), h - 1))
    y1 = max(y0 + 1, min(int(y1), h))

    seg = 255.0 - gray[y0:y1, :]
    if seg.size == 0:
        return 0, w

    prof = seg.mean(axis=0)
    base = float(np.percentile(prof, 12))
    peak = float(prof.max())
    if peak - base < 2.0:
        return 0, w

    gate = base + (peak - base) * COL_GATE_RATIO
    cols = np.nonzero(prof > gate)[0]
    if cols.size == 0:
        return 0, w

    return int(cols[0]), int(cols[-1]) + 1


def split_lines(
    image: Image.Image,
    boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    log: Optional[List[str]] = None,
) -> List[TextLine]:
    out: List[TextLine] = []

    usable: List[Tuple[int, int, int, int]] = []
    for b in (boxes or []):
        x0, y0, x1, y1 = (int(v) for v in b)
        bw = x1 - x0
        bh = y1 - y0
        if bw < 4 or bh < LINE_MIN_H:
            continue
        if bw >= image.width * 0.95 and bh >= image.height * 0.80:
            continue
        usable.append((x0, y0, x1, y1))

    if usable:
        usable.sort(key=lambda b: (b[1], b[0]))
        for i, (x0, y0, x1, y1) in enumerate(usable[:LINE_MAX]):
            px0 = max(0, x0 - LINE_PAD_X)
            py0 = max(0, y0 - LINE_PAD_Y)
            px1 = min(image.width, x1 + LINE_PAD_X)
            py1 = min(image.height, y1 + LINE_PAD_Y)
            line = TextLine(i, (px0, py0, px1, py1), image.crop((px0, py0, px1, py1)))
            line.source = "detector"
            out.append(line)
        if log is not None:
            log.append(
                f"    📚 [LINE SPLIT] 검출 박스 {len(out)}개를 그대로 "
                f"행으로 씁니다."
            )
        return out

    bands = _row_bands(image)
    if not bands:
        line = TextLine(0, (0, 0, image.width, image.height), image)
        line.source = "whole"
        if log is not None:
            log.append(
                "    📚 [LINE SPLIT] 행 밴드를 찾지 못해 크롭 전체를 "
                "한 행으로 처리합니다."
            )
        return [line]

    for i, (y0, y1) in enumerate(bands[:LINE_MAX]):
        cx0, cx1 = _trim_columns(image, y0, y1)
        px0 = max(0, cx0 - LINE_PAD_X)
        py0 = max(0, y0 - LINE_PAD_Y)
        px1 = min(image.width, cx1 + LINE_PAD_X)
        py1 = min(image.height, y1 + LINE_PAD_Y)
        if px1 - px0 < 4 or py1 - py0 < 4:
            continue
        line = TextLine(i, (px0, py0, px1, py1), image.crop((px0, py0, px1, py1)))
        line.source = "profile"
        out.append(line)

    if log is not None:
        hs = [l.height for l in out]
        avg = int(sum(hs) / max(1, len(hs))) if hs else 0
        log.append(
            f"    📚 [LINE SPLIT] 수평 잉크 투영 + 다층 재분할로 행 "
            f"{len(out)}개 분리 (평균 높이 {avg}px) — 각 행을 따로 판독해 "
            f"긴 문장 누적 오류를 끊습니다."
        )
    return out


def upscale_line(
    line: TextLine,
    target_px: float = TARGET_LINE_PX,
) -> Tuple[Image.Image, float]:
    h = max(1, line.height)
    factor = float(np.clip(target_px / float(h), MIN_LINE_UPSCALE, MAX_LINE_UPSCALE))
    if factor <= 1.01:
        return line.image, 1.0
    nw = max(8, int(round(line.image.width * factor)))
    nh = max(8, int(round(line.image.height * factor)))
    return line.image.resize((nw, nh), Image.LANCZOS), factor


def build_windows(
    lines: Sequence[TextLine],
    span: int = WINDOW_LINES,
    stride: int = WINDOW_STRIDE,
    overlap: int = WINDOW_OVERLAP,
    log: Optional[List[str]] = None,
) -> List[ReadWindow]:
    items = list(lines)
    n = len(items)
    if n == 0:
        return []

    span = max(1, int(span))
    stride = max(1, int(stride))
    overlap = max(0, int(overlap))

    width = span + overlap
    out: List[ReadWindow] = []
    seen = set()

    i = 0
    while i < n:
        j = min(n, i + width)
        key = (i, j)
        if key not in seen:
            seen.add(key)
            out.append(ReadWindow(i, j, items[i:j]))
        if j >= n:
            break
        i += stride

    if log is not None:
        if overlap > 0:
            log.append(
                f"    🪟 [READ WINDOW] 행 {n}개 → 창 {len(out)}개 "
                f"(창당 {width}행 / 보폭 {stride}행 / 겹침 {overlap}행) — "
                f"겹친 구간은 교차 검증으로 확정합니다."
            )
        else:
            log.append(
                f"    🪟 [READ WINDOW] 행 {n}개 → 창 {len(out)}개 "
                f"(창당 {width}행 / 겹침 없음)"
            )
    return out


def normalize_for_match(text: str) -> str:
    s = unicodedata.normalize("NFKC", str(text or ""))
    return WS_RE.sub("", s)


def _lcs_suffix_prefix(a: str, b: str, min_len: int = MERGE_MIN_OVERLAP) -> int:
    la, lb = len(a), len(b)
    limit = min(la, lb)
    for k in range(limit, min_len - 1, -1):
        if a[la - k:] == b[:k]:
            return k
    return 0


def stitch_windows(
    windows: Sequence[ReadWindow],
    log: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    parts: List[str] = []
    notes: List[str] = []

    for w in windows:
        t = str(w.text or "").strip()
        if not t:
            continue
        if not parts:
            parts.append(t)
            continue

        prev = parts[-1]
        k = _lcs_suffix_prefix(
            normalize_for_match(prev), normalize_for_match(t)
        )
        if k >= MERGE_MIN_OVERLAP:
            tail = normalize_for_match(t)[:k]
            cut = 0
            acc = 0
            for ch in t:
                if acc >= k:
                    break
                if not WS_RE.match(ch):
                    acc += 1
                cut += 1
            merged = prev + t[cut:]
            parts[-1] = merged
            notes.append(f"겹침 {k}자 병합: …{tail}")
            continue

        parts.append(t)

    joined = "\n".join(parts)

    if log is not None and notes:
        log.append(
            f"    🧵 [STITCH] 창 경계 {len(notes)}곳을 겹침 문자열로 "
            f"이어 붙였습니다."
        )
        for nt in notes[:6]:
            log.append(f"       · {nt}")

    return joined, notes


def vote_lines(
    lines: Sequence[TextLine],
    windows: Sequence[ReadWindow],
    log: Optional[List[str]] = None,
) -> Dict[int, str]:
    bucket: Dict[int, List[str]] = {}

    for w in windows:
        rows = [r for r in str(w.text or "").splitlines() if r.strip()]
        if not rows:
            continue
        if len(rows) == w.span:
            for off, row in enumerate(rows):
                bucket.setdefault(w.start + off, []).append(row.strip())
        else:
            bucket.setdefault(w.start, []).append(" ".join(r.strip() for r in rows))

    out: Dict[int, str] = {}
    disputes = 0

    for idx, cands in bucket.items():
        clean = [c for c in cands if len(normalize_for_match(c)) >= VOTE_MIN_LEN]
        if not clean:
            continue
        if len(clean) == 1:
            out[idx] = clean[0]
            continue

        tally: Dict[str, int] = {}
        rep: Dict[str, str] = {}
        for c in clean:
            key = normalize_for_match(c)
            tally[key] = tally.get(key, 0) + 1
            rep.setdefault(key, c)

        best_key = max(tally.items(), key=lambda kv: (kv[1], len(kv[0])))[0]
        out[idx] = rep[best_key]

        if len(tally) > 1:
            disputes += 1
            if log is not None:
                variants = " | ".join(
                    f"{rep[k][:20]}({v}표)"
                    for k, v in sorted(tally.items(), key=lambda kv: -kv[1])[:3]
                )
                log.append(
                    f"       🗳 행 {idx}: {variants} → '{out[idx][:24]}' 채택"
                )

    if log is not None and disputes:
        log.append(
            f"    🗳 [CROSS VOTE] 겹친 행 {disputes}곳에서 판독이 갈려 "
            f"다수결로 확정했습니다."
        )
    return out


def read_lines(
    image: Image.Image,
    reader: Callable[[Image.Image, int, int], str],
    boxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    span: int = WINDOW_LINES,
    stride: int = WINDOW_STRIDE,
    overlap: int = WINDOW_OVERLAP,
    target_px: float = TARGET_LINE_PX,
    log: Optional[List[str]] = None,
) -> Tuple[str, List[TextLine], List[ReadWindow]]:
    lines = split_lines(image, boxes=boxes, log=log)
    if not lines:
        return "", [], []

    windows = build_windows(
        lines, span=span, stride=stride, overlap=overlap, log=log
    )

    for w in windows:
        crop = w.crop(image)
        h = max(1, crop.height // max(1, w.span))
        factor = float(
            np.clip(target_px / float(h), MIN_LINE_UPSCALE, MAX_LINE_UPSCALE)
        )
        if factor > 1.01:
            crop = crop.resize(
                (
                    max(8, int(round(crop.width * factor))),
                    max(8, int(round(crop.height * factor))),
                ),
                Image.LANCZOS,
            )
        try:
            w.raw = reader(crop, w.start, w.end) or ""
        except Exception as e:
            if log is not None:
                log.append(f"       ⚠ 창[{w.start}:{w.end}] 판독 실패 ({e})")
            w.raw = ""
        w.text = w.raw.strip()

    voted = vote_lines(lines, windows, log=log)
    for ln in lines:
        if ln.index in voted:
            ln.text = voted[ln.index]

    if voted:
        ordered = [voted[i] for i in sorted(voted.keys()) if voted[i].strip()]
        merged = "\n".join(ordered)
    else:
        merged, _notes = stitch_windows(windows, log=log)

    if log is not None:
        log.append(
            f"    📖 [LINE READ] 창 {len(windows)}개 판독 → 행 "
            f"{len([l for l in lines if l.text])}개 확정"
        )
        for ln in lines[:8]:
            if ln.text:
                log.append(f"       {ln.index:>2}. {ln.text[:52]}")

    return merged, lines, windows