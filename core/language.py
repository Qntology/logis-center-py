import math
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .lang_codes import (
    NOISE_ANCHORS,
    NOISE_QUALITY_FLOOR,
    PEER_DECISION_MARGIN,
    REFERENCE_DECISION_MARGIN,
    REFERENCE_LANGUAGE,
    SCRIPT_ANCHORS,
    SCRIPT_CHROME_ANCHORS,
    STRONG_REFERENCE_MARGIN,
    candidates_for_script,
    dominant_script,
    exclusive_language_of,
    lang_anchor_phrases,
    language_name,
    normalize_lang_code,
    priors_for_script,
    reference_language_of,
    script_histogram,
)
from .nms import gumbel_expected_z

WORD_SPLIT_RE = re.compile(r"[^\w\u00C0-\uFFFF]+", re.UNICODE)

DEFAULT_MARGIN_THRESHOLD = 0.28
DEFAULT_MAX_BANDS = 8
DEFAULT_MIN_BAND_PX = 10

EMBED_BATCH = 32
DOC_SAMPLE_CHARS = 2000
CONFIDENCE_GAIN = 24.0
FINALIST_KEEP = 10

UNRESOLVED_STAGES: Tuple[str, ...] = (
    "no-input",
    "fallback",
    "prior-fallback",
    "noise-fallback",
    "reference-fallback",
    "low-margin",
)


def _z_scores(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    if std < 1e-8:
        return [0.0] * len(values)
    return [float((v - mean) / std) for v in values]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.shape[-1] != b.shape[-1]:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _unit(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(arr))
    if n < eps:
        return arr
    return arr / n


class LanguageVerdict:
    def __init__(
        self,
        code: str,
        script: Optional[str],
        confidence: float,
        margin: float,
        stage: str,
        candidates: List[Tuple[str, float]],
        sample_text: str = "",
        logs: Optional[List[str]] = None,
    ):
        self.code = code
        self.script = script
        self.confidence = float(confidence)
        self.margin = float(margin)
        self.stage = stage
        self.candidates = candidates
        self.sample_text = sample_text
        self.logs = logs or []

    @property
    def name(self) -> str:
        return language_name(self.code)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "name": self.name,
            "script": self.script or "",
            "confidence": round(self.confidence, 4),
            "margin": round(self.margin, 4),
            "stage": self.stage,
            "candidates": [
                {"code": c, "name": language_name(c), "score": round(s, 4)}
                for c, s in self.candidates[:8]
            ],
            "sample_text": self.sample_text[:400],
        }

    def __repr__(self) -> str:
        return (
            f"<LanguageVerdict {self.code}({self.name}) script={self.script} "
            f"conf={self.confidence:.3f} margin={self.margin:+.3f} stage={self.stage}>"
        )


class TextEvidence:
    def __init__(self):
        self.script: Optional[str] = None
        self.script_ratio: float = 0.0
        self.exclusive: Optional[str] = None
        self.reference: str = REFERENCE_LANGUAGE
        self.ref_cos: float = 0.0
        self.noise_cos: float = 0.0
        self.quality: float = 0.0
        self.absolute: Dict[str, float] = {}
        self.relative: Dict[str, float] = {}
        self.ranked: List[Tuple[str, float]] = []
        self.usable: bool = False

    def to_dict(self) -> dict:
        return {
            "script": self.script or "",
            "script_ratio": round(self.script_ratio, 4),
            "exclusive": self.exclusive or "",
            "reference": self.reference,
            "ref_cos": round(self.ref_cos, 4),
            "noise_cos": round(self.noise_cos, 4),
            "quality": round(self.quality, 4),
            "usable": self.usable,
            "ranked": [
                {"code": c, "relative": round(v, 4)} for c, v in self.ranked[:8]
            ],
        }


class LanguageDetector:
    def __init__(
        self,
        ocr=None,
        embed_fn=None,
        log: Optional[Callable[[str], None]] = None,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
        max_bands: int = DEFAULT_MAX_BANDS,
        max_new_tokens: int = 96,
        nlp=None,
        reference_margin: float = REFERENCE_DECISION_MARGIN,
        peer_margin: float = PEER_DECISION_MARGIN,
        noise_floor: float = NOISE_QUALITY_FLOOR,
    ):
        self.ocr = ocr
        self.embed_fn = embed_fn
        self._log_fn = log or (lambda m: None)
        self.margin_threshold = float(margin_threshold)
        self.max_bands = int(max_bands)
        self.max_new_tokens = int(max_new_tokens)
        self.nlp = nlp
        self.reference_margin = float(reference_margin)
        self.peer_margin = float(peer_margin)
        self.noise_floor = float(noise_floor)
        self._anchor_cache: Dict[str, np.ndarray] = {}
        self._text_cache: Dict[str, np.ndarray] = {}
        self._text_dim: int = 0
        self.logs: List[str] = []

    def _log(self, msg: str):
        self.logs.append(msg)
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def _anchor_vec(self, phrase: str) -> Optional[np.ndarray]:
        if self.ocr is None:
            return None
        if phrase in self._anchor_cache:
            return self._anchor_cache[phrase]
        try:
            vec = self.ocr.embed_text(phrase)
        except Exception:
            return None
        if vec is None or not np.any(vec):
            return None
        vec = np.asarray(vec, dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return None
        vec = vec / norm
        self._anchor_cache[phrase] = vec
        return vec

    def _embed_batch(self, texts: Sequence[str]) -> Optional[np.ndarray]:
        items = [str(t) for t in texts if t is not None]
        if not items:
            return None

        if self.embed_fn is not None:
            rows: List[np.ndarray] = []
            ok = True
            for i in range(0, len(items), EMBED_BATCH):
                chunk = items[i: i + EMBED_BATCH]
                try:
                    mat = self.embed_fn(chunk)
                except Exception as e:
                    self._log(f"  ⚠ 다국어 임베딩 실패({e}) → Hayai 임베딩으로 대체")
                    ok = False
                    break
                if mat is None:
                    ok = False
                    break
                mat = np.asarray(mat, dtype=np.float32)
                if mat.ndim == 1:
                    mat = mat.reshape(1, -1)
                if mat.shape[0] < len(chunk) or mat.shape[-1] <= 1:
                    ok = False
                    break
                if rows and rows[0].shape[-1] != mat.shape[-1]:
                    ok = False
                    break
                rows.append(mat[: len(chunk)])
            if ok and rows:
                return np.concatenate(rows, axis=0)

        if self.ocr is None:
            return None

        fallback: List[np.ndarray] = []
        for t in items:
            try:
                v = self.ocr.embed_text(t)
            except Exception:
                return None
            if v is None:
                return None
            fallback.append(np.asarray(v, dtype=np.float32).reshape(-1))
        if not fallback:
            return None
        return np.stack(fallback).astype(np.float32)

    def _text_anchor_vecs(self, phrases: Sequence[str]) -> Dict[str, np.ndarray]:
        uniq = [p for p in dict.fromkeys(phrases) if p]
        need = [p for p in uniq if p not in self._text_cache]

        if need:
            mat = self._embed_batch(need)
            if mat is not None and mat.size:
                dim = int(mat.shape[-1])
                if self._text_dim and dim != self._text_dim:
                    self._log("  ⚠ 임베딩 공간이 변경되어 앵커 캐시를 초기화합니다.")
                    self._text_cache.clear()
                self._text_dim = dim
                for i, p in enumerate(need):
                    if i >= mat.shape[0]:
                        break
                    v = _unit(mat[i])
                    if float(np.linalg.norm(v)) > 1e-8:
                        self._text_cache[p] = v

        return {p: self._text_cache[p] for p in uniq if p in self._text_cache}

    def _doc_vec(self, text: str) -> Optional[np.ndarray]:
        sample = (text or "").strip()
        if not sample:
            return None
        mat = self._embed_batch([sample[:DOC_SAMPLE_CHARS]])
        if mat is None or mat.size == 0:
            return None
        v = _unit(mat[0])
        return v if float(np.linalg.norm(v)) > 1e-8 else None

    def ink_bands(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if h < DEFAULT_MIN_BAND_PX or w < DEFAULT_MIN_BAND_PX:
            return [(0, 0, w, h)]

        ink = 255.0 - gray
        row_profile = ink.mean(axis=1)
        thr = float(row_profile.mean() + row_profile.std() * 0.25)

        bands: List[Tuple[int, int]] = []
        start = -1
        for y in range(h):
            hot = row_profile[y] > thr
            if hot and start < 0:
                start = y
            elif not hot and start >= 0:
                if y - start >= DEFAULT_MIN_BAND_PX:
                    bands.append((start, y))
                start = -1
        if start >= 0 and h - start >= DEFAULT_MIN_BAND_PX:
            bands.append((start, h))

        if not bands:
            return [(0, 0, w, h)]

        scored = []
        for (y0, y1) in bands:
            seg = ink[y0:y1, :]
            col_profile = seg.mean(axis=0)
            cthr = float(col_profile.mean() + col_profile.std() * 0.1)
            cols = np.nonzero(col_profile > cthr)[0]
            if cols.size == 0:
                x0, x1 = 0, w
            else:
                x0 = int(max(0, cols[0] - 4))
                x1 = int(min(w, cols[-1] + 5))
            if x1 - x0 < DEFAULT_MIN_BAND_PX:
                x0, x1 = 0, w
            scored.append((float(seg.mean()), (x0, y0, x1, y1)))

        scored.sort(key=lambda kv: kv[0], reverse=True)
        return [box for _s, box in scored[: self.max_bands]]

    def collect_text(self, image: Image.Image) -> str:
        if self.ocr is None:
            return ""
        boxes = self.ink_bands(image)
        self._log(f"  📐 텍스트 후보 밴드 {len(boxes)}개 추출")

        chunks: List[str] = []
        noise = 0

        for idx, box in enumerate(boxes):
            try:
                txt = self.ocr.ocr_crop(
                    image,
                    box,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=1,
                    repetition_penalty=1.05,
                )
            except Exception as e:
                self._log(f"  ⚠ 밴드 {idx} OCR 실패: {e}")
                continue

            txt = (txt or "").strip()
            if not txt:
                continue

            if self.nlp is not None and getattr(self.nlp, "available", False):
                ratio = self.nlp.content_ratio(txt, default=1.0)
                if ratio < 0.15:
                    noise += 1
                    continue

            chunks.append(txt)

        if noise:
            self._log(f"  🧬 NLP 노이즈 밴드 {noise}개 제외")

        merged = "\n".join(chunks)
        if merged:
            preview = merged.replace("\n", " / ")[:120]
            self._log(f"  📝 OCR 표본 {len(merged)}자: {preview}")
        else:
            self._log("  ⚠ OCR 표본을 얻지 못했습니다.")
        return merged

    def stage_a_visual(self, image: Image.Image) -> Dict[str, float]:
        if self.ocr is None:
            return {}
        try:
            patches = self.ocr.embed_image_for_scoring(image)
        except Exception as e:
            self._log(f"  ⚠ 시각 스크립트 판정 실패: {e}")
            return {}
        if patches is None or patches.size == 0:
            return {}

        chrome_vecs = []
        for ph in SCRIPT_CHROME_ANCHORS:
            v = self._anchor_vec(ph)
            if v is not None:
                chrome_vecs.append(v)

        raw: Dict[str, float] = {}
        for script, phrase in SCRIPT_ANCHORS.items():
            av = self._anchor_vec(phrase)
            if av is None:
                continue
            best = -1.0
            for i in range(patches.shape[0]):
                s = _cos(patches[i], av)
                if s > best:
                    best = s
            prej = 0.0
            for cv in chrome_vecs:
                pbest = -1.0
                for i in range(patches.shape[0]):
                    s = _cos(patches[i], cv)
                    if s > pbest:
                        pbest = s
                if pbest > prej:
                    prej = pbest
            raw[script] = best - prej

        if not raw:
            return {}

        keys = list(raw.keys())
        zs = _z_scores([raw[k] for k in keys])
        penalty = gumbel_expected_z(len(keys))
        out = {k: (z - penalty) for k, z in zip(keys, zs)}

        top = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:4]
        brief = " | ".join(f"{k}={v:+.3f}" for k, v in top)
        self._log(f"  👁 시각 스크립트 후보: {brief}")
        return out

    def _cosine_profile(
        self,
        doc_vec: np.ndarray,
        codes: Sequence[str],
        keep: int = FINALIST_KEEP,
    ) -> Tuple[Dict[str, float], float]:
        base_map: Dict[str, str] = {}
        for c in codes:
            phrases = lang_anchor_phrases(c)
            if phrases:
                base_map[c] = phrases[0]

        probe = list(base_map.values()) + list(NOISE_ANCHORS)
        vecs = self._text_anchor_vecs(probe)
        if not vecs:
            return {}, 0.0

        coarse: Dict[str, float] = {}
        for c, phrase in base_map.items():
            v = vecs.get(phrase)
            if v is not None:
                coarse[c] = _cos(doc_vec, v)

        if not coarse:
            return {}, 0.0

        noise_sims = [_cos(doc_vec, vecs[p]) for p in NOISE_ANCHORS if p in vecs]
        noise = float(max(noise_sims)) if noise_sims else 0.0

        ranked = sorted(coarse.items(), key=lambda kv: kv[1], reverse=True)
        finalists = [c for c, _v in ranked[: max(2, int(keep))]]
        if REFERENCE_LANGUAGE in coarse and REFERENCE_LANGUAGE not in finalists:
            finalists.append(REFERENCE_LANGUAGE)

        fine_probe: List[str] = []
        for c in finalists:
            fine_probe.extend(lang_anchor_phrases(c))
        fine = self._text_anchor_vecs(fine_probe)

        out: Dict[str, float] = dict(coarse)
        for c in finalists:
            sims = [
                _cos(doc_vec, fine[p])
                for p in lang_anchor_phrases(c) if p in fine
            ]
            if sims:
                out[c] = float(sum(sims) / len(sims))

        return out, noise

    def stage_b_text(self, text: str) -> TextEvidence:
        ev = TextEvidence()

        if not text or not text.strip():
            self._log("  ⚠ 판별할 텍스트 표본이 없습니다.")
            return ev

        script, ratio, hist = dominant_script(text)
        ev.script = script
        ev.script_ratio = float(ratio)

        if script is None:
            self._log("  ⚠ OCR 텍스트에서 식별 가능한 스크립트가 없습니다.")
            return ev

        hist_brief = " | ".join(
            f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:4]
        )
        self._log(f"  🔤 문자 스크립트: {script} ({ratio:.2%}) [{hist_brief}]")

        ev.exclusive = exclusive_language_of(script)
        if script == "Han" and hist.get("Kana", 0) > 0:
            ev.exclusive = "jpn"
        if ev.exclusive:
            self._log(
                f"  🎯 스크립트 배타 언어: {ev.exclusive}"
                f"({language_name(ev.exclusive)})"
            )

        doc_vec = self._doc_vec(text)
        if doc_vec is None:
            self._log("  ⚠ 문서 임베딩을 얻지 못해 코사인 판별을 건너뜁니다.")
            return ev

        codes = list(candidates_for_script(script))
        for extra in (reference_language_of(script), REFERENCE_LANGUAGE):
            if extra and extra not in codes:
                codes.append(extra)

        absolute, noise = self._cosine_profile(doc_vec, codes)
        if not absolute:
            self._log("  ⚠ 언어 앵커 임베딩을 얻지 못했습니다.")
            return ev

        base_code = (
            REFERENCE_LANGUAGE if REFERENCE_LANGUAGE in absolute
            else reference_language_of(script)
        )
        if base_code not in absolute:
            base_code = max(absolute.items(), key=lambda kv: kv[1])[0]

        ev.absolute = absolute
        ev.noise_cos = float(noise)
        ev.reference = base_code
        ev.ref_cos = float(absolute.get(base_code, 0.0))
        ev.relative = {c: v - ev.ref_cos for c, v in absolute.items()}
        ev.ranked = sorted(ev.relative.items(), key=lambda kv: kv[1], reverse=True)
        ev.quality = float(max(absolute.values()) - noise)
        ev.usable = True

        self._log(
            f"  🧭 기준 언어 '{base_code}' 절대 코사인 {ev.ref_cos:+.4f} "
            f"| 노이즈 앵커 {noise:+.4f} | 언어 품질 {ev.quality:+.4f}"
        )
        self._log(
            "  📚 영어 기준 상대 코사인: "
            + " | ".join(f"{c}={v:+.4f}" for c, v in ev.ranked[:6])
        )
        return ev

    def fuse(
        self,
        visual_scores: Dict[str, float],
        ev: TextEvidence,
    ) -> Tuple[str, Optional[str], float, float, List[Tuple[str, float]], str]:
        script = ev.script
        if script is None and visual_scores:
            script = max(visual_scores.items(), key=lambda kv: kv[1])[0]

        visual_boost = 0.0
        if script and visual_scores:
            visual_boost = float(visual_scores.get(script, 0.0))

        exclusive = ev.exclusive or (exclusive_language_of(script) if script else None)
        fallback = exclusive or reference_language_of(script)
        ranked = ev.ranked if ev.ranked else [(fallback, 0.0)]

        if not ev.usable:
            if exclusive:
                self._log(
                    f"  🎯 텍스트 코사인 없이 스크립트 배타 언어 '{exclusive}' 로 확정"
                )
                return (
                    exclusive, script, max(0.0, visual_boost),
                    max(0.0, visual_boost), ranked, "visual-exclusive",
                )
            self._log(
                f"  ⚪ 코사인 근거가 없어 기준 언어 '{fallback}' 로 보류합니다."
            )
            return fallback, script, 0.0, 0.0, ranked, "fallback"

        if exclusive:
            return (
                exclusive, script, ev.absolute.get(exclusive, 0.0),
                max(ev.quality, 0.0), ranked, "script-exclusive",
            )

        if ev.quality < self.noise_floor:
            self._log(
                f"  🚧 언어 품질 {ev.quality:+.4f} < {self.noise_floor:+.4f} — "
                f"노이즈 앵커가 더 강해 '{fallback}' 로 보류합니다."
            )
            return fallback, script, ev.quality, ev.quality, ranked, "noise-fallback"

        best_code, best_rel = ranked[0]
        second_code = ranked[1][0] if len(ranked) > 1 else best_code
        second_rel = ranked[1][1] if len(ranked) > 1 else best_rel
        peer_margin = best_rel - second_rel

        if best_code == ev.reference:
            self._log(
                f"  👑 기준 언어 '{ev.reference}' 가 최상위 — "
                f"동급 격차 {peer_margin:+.4f}"
            )
            return (
                ev.reference, script, best_rel, max(peer_margin, 0.0),
                ranked, "reference-locked",
            )

        if best_rel < self.reference_margin:
            self._log(
                f"  ⚖ '{best_code}' 가 기준 언어 '{ev.reference}' 를 "
                f"{best_rel:+.4f} 만큼만 앞섭니다 "
                f"(요구 {self.reference_margin:.3f}) → 기준 언어 유지"
            )
            return (
                ev.reference, script, best_rel, best_rel,
                ranked, "reference-fallback",
            )

        if peer_margin < self.peer_margin:
            priors = priors_for_script(script)
            pool = [c for c, _v in ranked[:3] if c != ev.reference]
            pick = max(pool, key=lambda c: priors.get(c, 0.0)) if pool else fallback

            if best_rel >= STRONG_REFERENCE_MARGIN and script != "Latin":
                self._log(
                    f"  🌍 '{best_code}' vs '{second_code}' 격차 {peer_margin:+.4f} "
                    f"이지만 기준 언어 대비 {best_rel:+.4f} 로 뚜렷 → "
                    f"문자체계 우선 언어 '{pick}' 확정"
                )
                return pick, script, best_rel, peer_margin, ranked, "script-prior"

            self._log(
                f"  ⚖ '{best_code}' vs '{second_code}' 격차 {peer_margin:+.4f} < "
                f"{self.peer_margin:.3f} → 기준 언어 '{fallback}' 로 보류"
            )
            return fallback, script, best_rel, peer_margin, ranked, "low-margin"

        self._log(
            f"  👑 '{best_code}' 확정 — 기준 언어 대비 {best_rel:+.4f}, "
            f"차상위 '{second_code}' 대비 {peer_margin:+.4f}"
        )
        return best_code, script, best_rel, peer_margin, ranked, "cosine-decided"

    def detect(self, image: Image.Image) -> LanguageVerdict:
        self.logs = []
        self._log("═══ 언어 판별 시작 ═══")

        visual = self.stage_a_visual(image)
        text = self.collect_text(image)
        ev = self.stage_b_text(text)

        code, final_script, score, margin, ranked, stage = self.fuse(visual, ev)
        code = normalize_lang_code(code)

        unresolved = stage in UNRESOLVED_STAGES
        conf = (
            0.0 if unresolved
            else 1.0 / (1.0 + math.exp(-float(margin) * CONFIDENCE_GAIN))
        )

        mark = "⚠️ 보류" if unresolved else "✅ 확정"
        self._log(
            f"  {mark}: {code}({language_name(code)}) "
            f"script={final_script} margin={margin:+.4f} conf={conf:.3f} stage={stage}"
        )

        return LanguageVerdict(
            code=code,
            script=final_script,
            confidence=conf,
            margin=margin,
            stage=stage,
            candidates=ranked,
            sample_text=text,
            logs=list(self.logs),
        )

    def detect_from_text(self, text: str) -> LanguageVerdict:
        self.logs = []
        self._log("═══ 언어 판별 시작 (텍스트 입력) ═══")
        ev = self.stage_b_text(text)
        code, final_script, score, margin, ranked, stage = self.fuse({}, ev)
        code = normalize_lang_code(code)
        unresolved = stage in UNRESOLVED_STAGES
        conf = (
            0.0 if unresolved
            else 1.0 / (1.0 + math.exp(-float(margin) * CONFIDENCE_GAIN))
        )
        mark = "⚠️ 보류" if unresolved else "✅ 확정"
        self._log(
            f"  {mark}: {code}({language_name(code)}) "
            f"script={final_script} margin={margin:+.4f} conf={conf:.3f} stage={stage}"
        )
        return LanguageVerdict(
            code=code,
            script=final_script,
            confidence=conf,
            margin=margin,
            stage=stage,
            candidates=ranked,
            sample_text=text,
            logs=list(self.logs),
        )


def detect_language(
    image: Image.Image,
    ocr=None,
    embed_fn=None,
    log=None,
    **kwargs,
) -> LanguageVerdict:
    return LanguageDetector(
        ocr=ocr, embed_fn=embed_fn, log=log, **kwargs
    ).detect(image)


def detect_language_from_text(
    text: str,
    embed_fn=None,
    log=None,
    **kwargs,
) -> LanguageVerdict:
    return LanguageDetector(
        ocr=None, embed_fn=embed_fn, log=log, **kwargs
    ).detect_from_text(text)


def script_summary(text: str) -> dict:
    script, ratio, hist = dominant_script(text)
    return {
        "script": script or "",
        "ratio": round(ratio, 4),
        "histogram": script_histogram(text),
    }