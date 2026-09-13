import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .chunker import Chunk, ChunkerConfig, split_natural_language_to_chunks
from .field_bank import flatten_text_values
from .exclusive_assign import (
    Assignment,
    assignments_to_record,
    exclusive_assign_for_indexing,
)
from .field_bank import FieldBank, build_field_bank
from .format_gate import build_format_map, format_gate_for_indexing
from .nms_battle import (
    bridge_gaps,
    coverage_ratio,
    nms_battle_for_indexing,
    winners_to_chunks,
)
from .plinko import plinko_game_for_indexing, plinko_override
from .surprisal import score_chunks


class TextPipelineConfig:
    def __init__(
        self,
        min_words: int = 1,
        max_words: int = 6,
        max_windows: int = 48,
        dual_track: bool = True,
        surprisal_gate: float = 0.0,
        margin_threshold: float = 0.0,
        cliff_ratio: float = 0.75,
        enable_plinko: bool = True,
        enable_gap_bridge: bool = True,
        cross_prejudice: bool = True,
        embed_batch_size: int = 32,
        multi_value_fields: Optional[Sequence[str]] = None,
        lang_code: str = "",
        external_dictionary: bool = True,
    ):
        self.min_words = int(min_words)
        self.max_words = int(max_words)
        self.max_windows = int(max_windows)
        self.dual_track = bool(dual_track)
        self.surprisal_gate = float(surprisal_gate)
        self.margin_threshold = float(margin_threshold)
        self.cliff_ratio = float(cliff_ratio)
        self.enable_plinko = bool(enable_plinko)
        self.enable_gap_bridge = bool(enable_gap_bridge)
        self.cross_prejudice = bool(cross_prejudice)
        self.embed_batch_size = int(embed_batch_size)
        self.multi_value_fields = list(multi_value_fields or [])
        self.lang_code = str(lang_code or "")
        self.external_dictionary = bool(external_dictionary)

    def chunker(self) -> ChunkerConfig:
        return ChunkerConfig(
            min_words=self.min_words,
            max_words=self.max_words,
            max_windows=self.max_windows,
            dual_track=self.dual_track,
        )


class TextPipelineResult:
    def __init__(self):
        self.ok: bool = False
        self.error: str = ""
        self.words: List[str] = []
        self.chunks: List[Chunk] = []
        self.winners: List = []
        self.assignments: List[Assignment] = []
        self.leftovers: List[Chunk] = []
        self.record: Dict[str, object] = {}
        self.format_report: List[dict] = []
        self.plinko: List = []
        self.log: List[str] = []
        self.bank_stats: Dict = {}
        self.elapsed: float = 0.0
        self.coverage: float = 0.0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "record": self.record,
            "words": len(self.words),
            "chunks": len(self.chunks),
            "winners": [w.to_dict() for w in self.winners],
            "assignments": [a.to_dict() for a in self.assignments],
            "leftovers": [c.to_dict() for c in self.leftovers],
            "format_report": self.format_report,
            "plinko": [p.to_dict() for p in self.plinko],
            "bank": self.bank_stats,
            "coverage": round(self.coverage, 4),
            "elapsed": round(self.elapsed, 3),
            "log": self.log,
        }


def _match_field_by_label(bank, label: str) -> str:
    target = "".join(ch for ch in str(label or "").lower() if ch.isalnum())
    if not target:
        return ""

    best = ""
    best_len = 0
    for name in bank.field_names:
        entry = bank.get(name)
        if entry is None:
            continue
        cands = [name.replace("_", " ")]
        for p in list(entry.label) + list(entry.bias):
            cands.append(getattr(p, "text", ""))
        for cand in cands:
            c = "".join(ch for ch in str(cand or "").lower() if ch.isalnum())
            if not c:
                continue
            if c == target:
                return name
            if (c in target or target in c) and len(c) > best_len:
                best = name
                best_len = len(c)
    return best


class TextPipeline:
    def __init__(
        self,
        schema: dict,
        embed_fn: Callable[[List[str]], np.ndarray],
        config: Optional[TextPipelineConfig] = None,
        log: Optional[Callable[[str], None]] = None,
        nlp=None,
    ):
        self.schema = schema or {}
        self.embed_fn = embed_fn
        self.config = config or TextPipelineConfig()
        self._log_fn = log or (lambda m: None)
        self.nlp = nlp
        self.bank: Optional[FieldBank] = None
        self.logs: List[str] = []

    def _log(self, msg: str):
        self.logs.append(msg)
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def prepare_bank(self) -> FieldBank:
        if self.bank is not None:
            return self.bank
        self._log("═══ 필드 뱅크 구축 ═══")
        self.bank = build_field_bank(
            self.schema,
            embed_fn=self.embed_fn,
            cross_prejudice=self.config.cross_prejudice,
            batch_size=self.config.embed_batch_size,
            lang_code=self.config.lang_code,
            external_dictionary=self.config.external_dictionary,
        )
        for line in self.bank.report_lines():
            self._log(line)

        if self.config.external_dictionary:
            try:
                from core import bias_bridge
                for line in bias_bridge.report(
                    self.bank.domain,
                    self.bank.field_names,
                    self.config.lang_code,
                ):
                    self._log(line)
            except Exception as e:
                self._log(f"  ⏭ 보조 사전 진단 생략: {e}")

        return self.bank

    def _embed_chunks(self, chunks: Sequence[Chunk]) -> np.ndarray:
        texts = [c.text for c in chunks]
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        out: List[np.ndarray] = []
        bs = self.config.embed_batch_size
        for i in range(0, len(texts), bs):
            batch = texts[i: i + bs]
            try:
                mat = self.embed_fn(batch)
            except Exception as e:
                self._log(f"  ⚠ 청크 임베딩 실패(batch {i}): {e}")
                mat = None
            if mat is None:
                dim = out[0].shape[1] if out else 1
                out.append(np.zeros((len(batch), dim), dtype=np.float32))
                continue
            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            out.append(mat)

        if not out:
            return np.zeros((0, 0), dtype=np.float32)

        dim = max(m.shape[1] for m in out)
        fixed = []
        for m in out:
            if m.shape[1] == dim:
                fixed.append(m)
            else:
                pad = np.zeros((m.shape[0], dim), dtype=np.float32)
                pad[:, : m.shape[1]] = m
                fixed.append(pad)
        return np.concatenate(fixed, axis=0)

    def run(self, text: str) -> TextPipelineResult:
        result = TextPipelineResult()
        started = time.time()
        self.logs = []

        if not text or not text.strip():
            result.error = "입력 텍스트가 비어 있습니다."
            result.log = self.logs
            return result

        try:
            bank = self.prepare_bank()
            result.bank_stats = bank.stats()

            self._log("═══ PHASE A: 입력 정제 + 청크 분할 ═══")

            label_bank: List[str] = []
            for _name, _d in (self.schema.get("fields", {}) or {}).items():
                if not isinstance(_d, dict):
                    continue
                for _k in ("label", "semantic"):
                    label_bank.extend(flatten_text_values(_d.get(_k)))

            try:
                from core.text_prep import (
                    collect_label_value_pairs,
                    enrich_chunks_with_metadata,
                    pairs_to_confirmed_chunks,
                    sanitize_llm_input,
                )
            except Exception as e:
                self._log(f"  ⏭ 전처리 모듈 로드 실패({e}) — 원문으로 진행합니다.")
                sanitize_llm_input = None

            if sanitize_llm_input is not None:
                before_len = len(text)
                text = sanitize_llm_input(text)
                self._log(
                    f"  🧼 [SANITIZE] 제어문자·BOM·중복공백 제거 "
                    f"{before_len}자 → {len(text)}자"
                )

            chunks, words = split_natural_language_to_chunks(text, self.config.chunker())
            result.words = words
            result.chunks = chunks
            self._log(f"  단어 {len(words)}개 → 윈도우 {len(chunks)}개")

            if not chunks:
                result.error = "청크를 생성하지 못했습니다."
                result.log = self.logs
                return result

            if sanitize_llm_input is not None:
                pairs = collect_label_value_pairs(
                    text, label_bank=label_bank, log=self.logs
                )
                confirmed = pairs_to_confirmed_chunks(pairs, words)
                if confirmed:
                    chunks.extend(confirmed)
                    chunks.sort(key=lambda c: (c.start, -(c.end - c.start)))
                    result.chunks = chunks
                    self._log(
                        f"  ✅ [CONFIRMED PROMOTE] 라벨↔값 쌍 {len(confirmed)}건을 "
                        f"확정 청크로 승격했습니다."
                    )
                enrich_chunks_with_metadata(
                    chunks, label_bank=label_bank, log=self.logs
                )

            self._log("═══ PHASE B: SURPRISAL 채점 ═══")
            vectors = self._embed_chunks(chunks)
            scored = score_chunks(chunks, vectors, bank, gate=self.config.surprisal_gate)
            self._log(f"  게이트 통과 청크 {len(scored)}/{len(chunks)}개")

            self._log("═══ PHASE C: NMS 배틀 + 흡수 ═══")
            winners = nms_battle_for_indexing(chunks, log=self.logs)
            self._log(f"  승자 {len(winners)}개")

            if self.config.enable_gap_bridge and winners:
                winners = bridge_gaps(winners, words, log=self.logs)
                result.coverage = coverage_ratio(winners, len(words))
                self._log(f"  갭 브리징 후 커버리지 {result.coverage:.1%}")

            result.winners = winners
            active = winners_to_chunks(winners)

            self._log("═══ PHASE D: FORMAT GATE ═══")
            formats = build_format_map(bank, embed_fn=self.embed_fn)
            fmt_brief = " | ".join(f"{k}:{v}" for k, v in list(formats.items())[:8])
            self._log(f"  형식 추론: {fmt_brief}")
            result.format_report = format_gate_for_indexing(
                active, bank=bank, formats=formats, log=self.logs
            )
            rejected = sum(1 for r in result.format_report if r["verdict"] == "reject")
            self._log(f"  검증 {len(result.format_report)}건 / 탈락 {rejected}건")

            if self.config.enable_plinko:
                self._log("═══ PHASE E: PLINKO GAME ═══")
                plinko = plinko_game_for_indexing(
                    words,
                    bank,
                    self.embed_fn,
                    confirmed_chunks=active,
                    max_window=self.config.max_words,
                    cliff_ratio=self.config.cliff_ratio,
                    gate=self.config.surprisal_gate,
                    log=self.logs,
                )
                result.plinko = plinko
                active = plinko_override(active, plinko, log=self.logs)
                self._log(f"  PLINKO 결과 {len(plinko)}건")

            self._log("═══ PHASE F: 배타 배정 ═══")

            resolved = 0
            for c in active:
                if not getattr(c, "confirmed", False):
                    continue
                if c.property and c.property != "unclassified":
                    continue
                hints = getattr(c, "bias_phrases", None) or []
                if not hints:
                    continue
                pick = _match_field_by_label(bank, hints[0])
                if pick:
                    c.property = pick
                    c.score = max(c.score, 1.0)
                    resolved += 1
            if resolved:
                self._log(
                    f"  🏷 [LABEL BIND] 확정 청크 {resolved}건을 라벨 코사인으로 "
                    f"필드에 결속했습니다."
                )

            assignments, leftovers = exclusive_assign_for_indexing(
                active,
                bank.field_names,
                scored=None,
                margin_threshold=self.config.margin_threshold,
                multi_value_fields=self.config.multi_value_fields,
                nlp=self.nlp,
                log=self.logs,
            )
            result.assignments = assignments
            result.leftovers = leftovers
            result.record = assignments_to_record(assignments, bank.field_names)
            self._log(f"  확정 {len(assignments)}건 / 잔여 {len(leftovers)}건")

            result.ok = True

        except Exception as e:
            import traceback
            result.error = str(e)
            self._log(f"❌ 텍스트 파이프라인 오류:\n{traceback.format_exc()}")

        result.elapsed = time.time() - started
        result.log = list(self.logs)
        return result


def run_text_pipeline(
    text: str,
    schema: dict,
    embed_fn: Callable[[List[str]], np.ndarray],
    config: Optional[TextPipelineConfig] = None,
    log: Optional[Callable[[str], None]] = None,
    nlp=None,
) -> TextPipelineResult:
    return TextPipeline(schema, embed_fn, config=config, log=log, nlp=nlp).run(text)