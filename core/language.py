import math
import re
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from .lang_codes import (
    CODEPOINT_SIGNATURES,
    LANG_ANCHOR_TEMPLATE,
    SCRIPT_ANCHORS,
    SCRIPT_CHROME_ANCHORS,
    candidates_for_script,
    dominant_script,
    exclusive_language_of,
    language_name,
    normalize_lang_code,
    priors_for_script,
    script_histogram,
)
from .nms import gumbel_expected_z

WORD_SPLIT_RE = re.compile(r"[^\w\u00C0-\uFFFF]+", re.UNICODE)

DEFAULT_MARGIN_THRESHOLD = 0.28
DEFAULT_MAX_BANDS = 8
DEFAULT_MIN_BAND_PX = 10


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
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


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


class LanguageDetector:
    def __init__(
        self,
        ocr=None,
        log: Optional[Callable[[str], None]] = None,
        margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
        max_bands: int = DEFAULT_MAX_BANDS,
        max_new_tokens: int = 96,
        nlp=None,
    ):
        self.ocr = ocr
        self._log_fn = log or (lambda m: None)
        self.margin_threshold = float(margin_threshold)
        self.max_bands = int(max_bands)
        self.max_new_tokens = int(max_new_tokens)
        self.nlp = nlp
        self._anchor_cache: Dict[str, np.ndarray] = {}
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

    def _codepoint_signature_scores(
        self,
        text: str,
        candidates: List[str],
    ) -> Dict[str, float]:
        if not text:
            return {}
        cps = [ord(ch) for ch in text if not ch.isspace() and not ch.isdigit()]
        if not cps:
            return {}
        total = float(len(cps))

        out: Dict[str, float] = {}
        for code in candidates:
            ranges = CODEPOINT_SIGNATURES.get(code)
            if not ranges:
                continue
            hits = 0
            for cp in cps:
                for lo, hi in ranges:
                    if lo <= cp <= hi:
                        hits += 1
                        break
            if hits:
                out[code] = hits / total
        return out

    def _embedding_language_scores(
        self,
        text: str,
        candidates: List[str],
    ) -> Dict[str, float]:
        if self.ocr is None or not text.strip() or not candidates:
            return {}

        try:
            doc_vec = self.ocr.embed_text(text[:2000])
        except Exception as e:
            self._log(f"  ⚠ 문서 텍스트 임베딩 실패: {e}")
            return {}
        if doc_vec is None or not np.any(doc_vec):
            return {}

        doc_vec = np.asarray(doc_vec, dtype=np.float32)
        norm = float(np.linalg.norm(doc_vec))
        if norm < 1e-8:
            return {}
        doc_vec = doc_vec / norm

        out: Dict[str, float] = {}
        for code in candidates:
            phrase = LANG_ANCHOR_TEMPLATE.format(name=language_name(code))
            av = self._anchor_vec(phrase)
            if av is None:
                continue
            out[code] = _cos(doc_vec, av)
        return out

    def stage_b_text(self, text: str) -> Tuple[Optional[str], Dict[str, float]]:
        if not text or not text.strip():
            return None, {}

        script, ratio, hist = dominant_script(text)
        if script is None:
            self._log("  ⚠ OCR 텍스트에서 식별 가능한 스크립트가 없습니다.")
            return None, {}

        hist_brief = " | ".join(
            f"{k}:{v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])[:4]
        )
        self._log(f"  🔤 문자 스크립트: {script} ({ratio:.2%}) [{hist_brief}]")

        excl = exclusive_language_of(script)
        if excl:
            self._log(f"  🎯 스크립트 배타 언어 확정: {excl}({language_name(excl)})")
            return script, {excl: 1.0}

        cands = candidates_for_script(script)
        priors = priors_for_script(script)

        sig_scores = self._codepoint_signature_scores(text, cands)
        emb_scores = self._embedding_language_scores(text, cands)

        if sig_scores:
            top_sig = sorted(sig_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
            self._log(
                "  🔡 코드포인트 시그니처: "
                + " | ".join(f"{k}={v:.3%}" for k, v in top_sig)
            )
        if emb_scores:
            top_emb = sorted(emb_scores.items(), key=lambda kv: kv[1], reverse=True)[:4]
            self._log(
                "  🧭 임베딩 언어 코사인: "
                + " | ".join(f"{k}={v:+.4f}" for k, v in top_emb)
            )

        emb_z: Dict[str, float] = {}
        if emb_scores:
            keys = list(emb_scores.keys())
            zs = _z_scores([emb_scores[k] for k in keys])
            emb_z = dict(zip(keys, zs))

        raw: Dict[str, float] = {}
        for code in cands:
            sig = sig_scores.get(code, 0.0)
            emb = emb_z.get(code, 0.0)
            prior = priors.get(code, 0.10)
            raw[code] = (sig * 4.00) + (emb * 1.00) + (prior * 0.25)

        if script == "Han" and hist.get("Kana", 0) > 0:
            raw["jpn"] = raw.get("jpn", 0.0) + 1.00

        keys = list(raw.keys())
        zs = _z_scores([raw[k] for k in keys])
        penalty = gumbel_expected_z(len(keys))
        out = {k: (z - penalty) for k, z in zip(keys, zs)}

        top = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:5]
        brief = " | ".join(f"{k}={v:+.3f}" for k, v in top)
        self._log(f"  📚 텍스트 언어 후보: {brief}")
        return script, out

    def fuse(
        self,
        visual_scores: Dict[str, float],
        text_script: Optional[str],
        text_scores: Dict[str, float],
    ) -> Tuple[str, Optional[str], float, float, List[Tuple[str, float]], str]:
        if not text_scores and not visual_scores:
            return "eng", None, 0.0, 0.0, [("eng", 0.0)], "fallback"

        script = text_script
        if script is None and visual_scores:
            script = max(visual_scores.items(), key=lambda kv: kv[1])[0]

        visual_boost = 0.0
        if script and visual_scores:
            visual_boost = visual_scores.get(script, 0.0)

        if not text_scores:
            excl = exclusive_language_of(script) if script else None
            if excl:
                return excl, script, max(0.0, visual_boost), visual_boost, [(excl, visual_boost)], "visual-exclusive"
            cands = candidates_for_script(script)
            priors = priors_for_script(script)
            ranked = sorted(
                ((c, priors.get(c, 0.1)) for c in cands),
                key=lambda kv: kv[1],
                reverse=True,
            )
            best = ranked[0][0] if ranked else "eng"
            margin = (ranked[0][1] - ranked[1][1]) if len(ranked) > 1 else ranked[0][1] if ranked else 0.0
            return best, script, max(0.0, visual_boost), margin, ranked, "visual-prior"

        fused = {k: v + (visual_boost * 0.25) for k, v in text_scores.items()}
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)

        best_code, best_score = ranked[0]
        margin = best_score - ranked[1][1] if len(ranked) > 1 else best_score
        stage = "fused" if margin >= self.margin_threshold else "fused-low-margin"

        if margin < self.margin_threshold:
            excl = exclusive_language_of(script) if script else None
            if excl:
                best_code = excl
                stage = "script-fallback"
            else:
                priors = priors_for_script(script)
                top_prior = sorted(priors.items(), key=lambda kv: kv[1], reverse=True)
                if top_prior:
                    best_code = top_prior[0][0]
                    stage = "prior-fallback"

        return best_code, script, best_score, margin, ranked, stage

    def detect(self, image: Image.Image) -> LanguageVerdict:
        self.logs = []
        self._log("═══ 언어 판별 시작 ═══")

        visual = self.stage_a_visual(image)
        text = self.collect_text(image)
        script, text_scores = self.stage_b_text(text)

        code, final_script, score, margin, ranked, stage = self.fuse(visual, script, text_scores)
        code = normalize_lang_code(code)

        conf = 1.0 / (1.0 + math.exp(-float(margin) * 2.0))

        self._log(
            f"  ✅ 확정: {code}({language_name(code)}) "
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
        script, text_scores = self.stage_b_text(text)
        code, final_script, score, margin, ranked, stage = self.fuse({}, script, text_scores)
        code = normalize_lang_code(code)
        conf = 1.0 / (1.0 + math.exp(-float(margin) * 2.0))
        self._log(
            f"  ✅ 확정: {code}({language_name(code)}) "
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


def detect_language(image: Image.Image, ocr=None, log=None, **kwargs) -> LanguageVerdict:
    return LanguageDetector(ocr=ocr, log=log, **kwargs).detect(image)


def detect_language_from_text(text: str, log=None, **kwargs) -> LanguageVerdict:
    return LanguageDetector(ocr=None, log=log, **kwargs).detect_from_text(text)


def script_summary(text: str) -> dict:
    script, ratio, hist = dominant_script(text)
    return {
        "script": script or "",
        "ratio": round(ratio, 4),
        "histogram": script_histogram(text),
    }