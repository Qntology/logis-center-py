from typing import Dict, List, Optional, Sequence, Tuple

from .lang_codes import iso1_of, normalize_lang_code
from .model_manager import (
    STANZA_CORE_PROCESSORS,
    STANZA_ROOT,
    stanza_lang_dir,
    stanza_ready,
)

FUNCTION_UPOS = frozenset({"ADP", "DET", "AUX", "CCONJ", "SCONJ", "PART", "PUNCT", "SYM", "X"})
CONTENT_UPOS = frozenset({"NOUN", "PROPN", "NUM", "ADJ", "VERB", "SYM"})
ENTITY_UPOS = frozenset({"PROPN", "NOUN", "NUM"})

ENTITY_DEPREL = frozenset({
    "nsubj", "obj", "iobj", "obl", "nmod", "appos", "flat",
    "compound", "conj", "root", "nummod", "amod",
})


class MorphVerdict:
    def __init__(
        self,
        text: str,
        is_entity: bool,
        content_ratio: float,
        upos: List[str],
        deprels: List[str],
        lemmas: List[str],
        reason: str = "",
    ):
        self.text = text
        self.is_entity = bool(is_entity)
        self.content_ratio = float(content_ratio)
        self.upos = upos
        self.deprels = deprels
        self.lemmas = lemmas
        self.reason = reason

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "is_entity": self.is_entity,
            "content_ratio": round(self.content_ratio, 4),
            "upos": self.upos,
            "deprels": self.deprels,
            "lemmas": self.lemmas,
            "reason": self.reason,
        }

    def __repr__(self) -> str:
        mark = "ENT" if self.is_entity else "FUN"
        return f"<Morph[{mark}] '{self.text[:24]}' upos={self.upos}>"


class StanzaNLP:
    def __init__(
        self,
        lang_code: str,
        processors: Optional[Sequence[str]] = None,
        use_gpu: bool = False,
        log=None,
    ):
        self.code = normalize_lang_code(lang_code)
        self.iso1 = iso1_of(self.code)
        self.available = False
        self.pipeline = None
        self.processors: List[str] = []
        self._log_fn = log or (lambda m: None)
        self._cache: Dict[str, MorphVerdict] = {}

        if not stanza_ready(self.iso1):
            self._log(f"  ⏭ Stanza '{self.iso1}' 모델 미준비 → NLP 게이트 비활성")
            return

        try:
            import stanza
        except ImportError:
            self._log("  ⏭ stanza 패키지가 설치되어 있지 않습니다 → NLP 게이트 비활성")
            return

        wanted = list(processors or STANZA_CORE_PROCESSORS)
        available = self._detect_processors()
        self.processors = [p for p in wanted if p in available]

        if "tokenize" not in self.processors:
            self._log(f"  ⏭ Stanza '{self.iso1}' tokenize 프로세서 없음 → 비활성")
            return

        try:
            self.pipeline = stanza.Pipeline(
                lang=self.iso1,
                dir=str(STANZA_ROOT),
                processors=",".join(self.processors),
                download_method=None,
                use_gpu=bool(use_gpu),
                logging_level="ERROR",
                verbose=False,
            )
            self.available = True
            self._log(
                f"  ✅ Stanza '{self.iso1}' 로드 완료 "
                f"(processors: {', '.join(self.processors)})"
            )
        except Exception as e:
            self._log(f"  ⚠ Stanza '{self.iso1}' 로드 실패: {e} → NLP 게이트 비활성")
            self.pipeline = None
            self.available = False

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def _detect_processors(self) -> List[str]:
        d = stanza_lang_dir(self.iso1)
        if not d.is_dir():
            return []
        out = []
        for p in sorted(d.iterdir()):
            if p.is_dir() and any(p.glob("*.pt")):
                out.append(p.name)
        return out

    def analyze(self, text: str) -> Optional[MorphVerdict]:
        if not self.available or not text or not text.strip():
            return None

        key = text.strip()
        if key in self._cache:
            return self._cache[key]

        try:
            doc = self.pipeline(key)
        except Exception:
            return None

        upos: List[str] = []
        deprels: List[str] = []
        lemmas: List[str] = []

        for sent in getattr(doc, "sentences", []) or []:
            for word in getattr(sent, "words", []) or []:
                u = getattr(word, "upos", None) or ""
                d = getattr(word, "deprel", None) or ""
                l = getattr(word, "lemma", None) or ""
                if u:
                    upos.append(u)
                if d:
                    deprels.append(d)
                if l:
                    lemmas.append(l)

        if not upos:
            return None

        content = sum(1 for u in upos if u in CONTENT_UPOS)
        ratio = content / float(len(upos))

        has_entity_pos = any(u in ENTITY_UPOS for u in upos)
        has_entity_dep = any(d.split(":")[0] in ENTITY_DEPREL for d in deprels) if deprels else True
        all_function = all(u in FUNCTION_UPOS for u in upos)

        if all_function:
            is_entity = False
            reason = "전 토큰이 기능어(UPOS)"
        elif not has_entity_pos:
            is_entity = False
            reason = "개체 후보 UPOS 없음"
        elif not has_entity_dep:
            is_entity = False
            reason = "개체 후보 DEPREL 없음"
        else:
            is_entity = True
            reason = "개체 후보"

        verdict = MorphVerdict(key, is_entity, ratio, upos, deprels, lemmas, reason)
        self._cache[key] = verdict
        return verdict

    def is_entity_candidate(self, text: str, default: bool = True) -> bool:
        v = self.analyze(text)
        if v is None:
            return default
        return v.is_entity

    def content_ratio(self, text: str, default: float = 1.0) -> float:
        v = self.analyze(text)
        if v is None:
            return default
        return v.content_ratio

    def canonicalize(self, text: str) -> str:
        v = self.analyze(text)
        if v is None or not v.lemmas:
            return text
        return " ".join(v.lemmas)

    def stats(self) -> dict:
        return {
            "code": self.code,
            "iso1": self.iso1,
            "available": self.available,
            "processors": self.processors,
            "cached": len(self._cache),
        }


def load_stanza(lang_code: str, use_gpu: bool = False, log=None) -> StanzaNLP:
    return StanzaNLP(lang_code, use_gpu=use_gpu, log=log)