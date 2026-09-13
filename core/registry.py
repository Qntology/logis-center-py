import threading
from typing import Callable, Dict, List, Optional, Sequence

from .lang_codes import iso1_of, language_name, normalize_lang_code
from .model_manager import (
    BOOTSTRAP_LANGUAGES,
    CORE_MODEL_KEYS,
    DEFAULT_LANGUAGE,
    LANG_KIND_PRIORITY,
    LANG_MODEL_KINDS,
    LANG_REPO_TEMPLATES,
    MODEL_KEYS,
    MODEL_SPECS,
    OPTIONAL_MODEL_KEYS,
    any_embedder_ready,
    check_all_models,
    delete_lang_model,
    delete_model,
    delete_stanza,
    describe_lang_model,
    describe_lang_models,
    describe_model,
    describe_stanza,
    embedder_ready_codes,
    is_manual_only,
    is_model_ready,
    lang_model_ready,
    lang_models_ready,
    missing_core_models,
    missing_lang_models,
    stanza_ready,
)

ROLE_LABELS = {
    "llm": "LLM / 임베딩",
    "vision": "비전 인코더",
    "ocr": "OCR",
    "support": "보조 설정",
    "optional": "선택 (수동 배치)",
}

STEP_REQUIREMENTS: Dict[str, dict] = {
    "bootstrap": {
        "label": "부트스트랩 임베딩",
        "base": [],
        "lang": ["ppocr", "qwen3emb"],
        "bootstrap_lang": True,
        "stanza": False,
    },
    "patch_grid": {
        "label": "패치 임베딩 격자",
        "base": ["ax-ve"],
        "lang": ["siglip2"],
        "bootstrap_lang": True,
        "stanza": False,
    },
    "doc_type": {
        "label": "문서 유형 분류",
        "base": [],
        "lang": ["siglip2", "qwen3emb"],
        "bootstrap_lang": True,
        "stanza": False,
    },
    "language": {
        "label": "언어 판별",
        "base": [],
        "lang": ["ppocr", "qwen3emb"],
        "bootstrap_lang": True,
        "stanza": False,
    },
    "field_heatmap": {
        "label": "필드 히트맵",
        "base": [],
        "lang": ["siglip2", "qwen3emb"],
        "stanza": False,
    },
    "ocr_extract": {
        "label": "크롭 OCR 추출",
        "base": ["ppocr-det"],
        "lang": ["ppocr"],
        "stanza": True,
    },
    "refine": {
        "label": "LLM 정제 추출",
        "base": [],
        "lang": ["qwen35"],
        "stanza": False,
    },
    "text_pipeline": {
        "label": "텍스트 파이프라인",
        "base": [],
        "lang": ["qwen3emb"],
        "stanza": True,
    },
}

PREREQUISITES: Dict[str, tuple] = {
    "base:siglip2-naflex": (),
    "base:ax-ve": (),
    "base:ppocr-det": (),
    "lang:ppocr": (),
    "lang:siglip2": (),
    "lang:qwen3emb": (),
    "lang:qwen35": ("lang:qwen3emb",),
    "stanza": (),
}

DOWNLOAD_ORDER = (
    "base:siglip2-naflex",
    "base:ax-ve",
    "base:ppocr-det",
    "lang:ppocr",
    "lang:siglip2",
    "lang:qwen3emb",
    "lang:qwen35",
    "stanza",
)


def _prereq_key(target_id: str) -> str:
    if target_id.startswith("base:"):
        return target_id
    if target_id.startswith("lang:"):
        parts = target_id.split(":")
        return f"lang:{parts[1]}" if len(parts) > 1 else target_id
    if target_id.startswith("stanza"):
        return "stanza"
    return target_id


def _order_rank(target_id: str) -> int:
    key = _prereq_key(target_id)
    try:
        return DOWNLOAD_ORDER.index(key)
    except ValueError:
        return len(DOWNLOAD_ORDER)


class ModelRegistry:
    def __init__(
        self,
        log: Optional[Callable[[str], None]] = None,
        event: Optional[Callable[[dict], None]] = None,
        notify: Optional[Callable[[dict], None]] = None,
        auto_fetch: bool = True,
    ):
        self._log_fn = log or (lambda m: None)
        self._event_fn = event or (lambda p: None)
        self._notify_fn = notify or (lambda p: None)
        self.auto_fetch = bool(auto_fetch)
        self.lang_code = DEFAULT_LANGUAGE
        self.language_resolved = False
        self._lock = threading.RLock()
        self._busy = False
        self._current = ""
        self._cancel_flag = False

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def _event(self, payload: dict):
        try:
            self._event_fn(payload)
        except Exception:
            pass

    def _notify(self, payload: dict):
        try:
            self._notify_fn(payload)
        except Exception:
            pass

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def current(self) -> str:
        return self._current

    def set_language(self, code: str, resolved: bool = False):
        self.lang_code = normalize_lang_code(code or DEFAULT_LANGUAGE)
        if resolved:
            self.language_resolved = True

    def reset_language_resolution(self):
        self.language_resolved = False

    def status(self) -> dict:
        code = self.lang_code
        iso1 = iso1_of(code)

        base = []
        for key in MODEL_KEYS:
            info = describe_model(key)
            info["role_label"] = ROLE_LABELS.get(info["role"], info["role"])
            info["scope"] = "base"
            info["id"] = f"base:{key}"
            base.append(info)

        lang = []
        for kind in LANG_MODEL_KINDS:
            info = describe_lang_model(kind, code)
            info["scope"] = "lang"
            info["id"] = f"lang:{kind}"
            info["role_label"] = "언어 스코프"
            lang.append(info)

        st = describe_stanza(iso1)
        st["scope"] = "stanza"
        st["id"] = "stanza"
        st["role_label"] = "NLP 게이트"

        boot = []
        if not self.language_resolved:
            for c in BOOTSTRAP_LANGUAGES:
                if c == code:
                    continue
                for kind in LANG_KIND_PRIORITY:
                    info = describe_lang_model(kind, c)
                    info["scope"] = "bootstrap"
                    info["id"] = f"lang:{kind}:{c}"
                    info["role_label"] = "부트스트랩"
                    boot.append(info)

        return {
            "lang_code": code,
            "lang_name": language_name(code),
            "iso1": iso1,
            "language_resolved": self.language_resolved,
            "bootstrap_languages": list(BOOTSTRAP_LANGUAGES),
            "embedder_ready": embedder_ready_codes(),
            "base": base,
            "lang": lang,
            "bootstrap": boot,
            "stanza": st,
            "busy": self._busy,
            "current": self._current,
            "auto_fetch": self.auto_fetch,
            "missing_core": missing_core_models(),
            "missing_lang": missing_lang_models(code),
            "total_bytes": (
                sum(i["bytes"] for i in base)
                + sum(i["bytes"] for i in lang)
                + sum(i["bytes"] for i in boot)
                + int(st.get("bytes", 0) or 0)
            ),
        }

    def cancel(self):
        self._cancel_flag = True
        try:
            from .model_fetcher import BaseModelFetcher
        except Exception:
            pass
        self._event({"status": "cancel_requested", "label": "취소 요청됨"})

    def download_base(self, key: str) -> dict:
        with self._lock:
            if self._busy:
                return {"ok": False, "error": "이미 다운로드가 진행 중입니다."}
            self._busy = True
            self._current = f"base:{key}"

        try:
            from .model_fetcher import BaseModelFetcher
            fetcher = BaseModelFetcher(progress_callback=self._event)
            if self._cancel_flag:
                fetcher.cancel()
            res = fetcher.download(key)
            self._log(
                f"  {'✅' if res.get('ok') else '❌'} "
                f"[{MODEL_SPECS[key]['label']}] "
                f"{'준비 완료' if res.get('ok') else res.get('error', '실패')}"
            )
            return res
        finally:
            self._busy = False
            self._current = ""
            self._cancel_flag = False

    def download_lang(self, kind: str, code: Optional[str] = None) -> dict:
        code = normalize_lang_code(code or self.lang_code)
        with self._lock:
            if self._busy:
                return {"ok": False, "error": "이미 다운로드가 진행 중입니다."}
            self._busy = True
            self._current = f"lang:{kind}:{code}"

        try:
            from .model_fetcher import LanguageModelFetcher
            fetcher = LanguageModelFetcher(progress_callback=self._event)
            ok = fetcher.download_kind(kind, code)
            return {"ok": bool(ok), "kind": kind, "code": code}
        finally:
            self._busy = False
            self._current = ""

    def download_lang_all(self, code: Optional[str] = None) -> dict:
        code = normalize_lang_code(code or self.lang_code)
        with self._lock:
            if self._busy:
                return {"ok": False, "error": "이미 다운로드가 진행 중입니다."}
            self._busy = True
            self._current = f"lang:*:{code}"

        try:
            from .model_fetcher import LanguageModelFetcher
            fetcher = LanguageModelFetcher(progress_callback=self._event)
            return fetcher.download_language(code)
        finally:
            self._busy = False
            self._current = ""

    def download_stanza(self, code: Optional[str] = None) -> dict:
        code = normalize_lang_code(code or self.lang_code)
        iso1 = iso1_of(code)
        with self._lock:
            if self._busy:
                return {"ok": False, "error": "이미 다운로드가 진행 중입니다."}
            self._busy = True
            self._current = f"stanza:{iso1}"

        try:
            from .model_fetcher import StanzaFetcher
            return StanzaFetcher(progress_callback=self._event).download(iso1)
        finally:
            self._busy = False
            self._current = ""

    def download(self, target_id: str) -> dict:
        if target_id.startswith("base:"):
            return self.download_base(target_id.split(":", 1)[1])
        if target_id.startswith("lang:"):
            parts = target_id.split(":")
            kind = parts[1] if len(parts) > 1 else ""
            code = parts[2] if len(parts) > 2 else self.lang_code
            if kind == "*":
                return self.download_lang_all(code)
            return self.download_lang(kind, code)
        if target_id.startswith("stanza"):
            parts = target_id.split(":")
            code = parts[1] if len(parts) > 1 else self.lang_code
            return self.download_stanza(code)
        return {"ok": False, "error": f"알 수 없는 대상: {target_id}"}

    def download_async(self, target_id: str) -> dict:
        if self._busy:
            return {"ok": False, "error": "이미 다운로드가 진행 중입니다."}
        t = threading.Thread(target=self.download, args=(target_id,), daemon=True)
        t.start()
        return {"ok": True, "started": target_id}

    def download_all_missing_async(self) -> dict:
        if self._busy:
            return {"ok": False, "error": "이미 다운로드가 진행 중입니다."}

        def _run():
            for key in DOWNLOAD_ORDER:
                if not key.startswith("base:"):
                    continue
                k = key.split(":", 1)[1]
                if is_manual_only(k):
                    continue
                if not is_model_ready(k):
                    self.download_base(k)

            codes = [self.lang_code]
            if not self.language_resolved:
                for c in BOOTSTRAP_LANGUAGES:
                    if c not in codes:
                        codes.append(c)

            for kind in LANG_KIND_PRIORITY:
                for c in codes:
                    if not lang_model_ready(kind, c):
                        self.download_lang(kind, c)

            for c in codes:
                if not stanza_ready(iso1_of(c)):
                    self.download_stanza(c)

            self._event({"status": "all_done", "label": "누락 모델 전체 준비 완료"})

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"ok": True, "started": "all"}

    def delete(self, target_id: str) -> dict:
        if self._busy:
            return {"ok": False, "error": "다운로드 중에는 삭제할 수 없습니다."}

        if target_id.startswith("base:"):
            key = target_id.split(":", 1)[1]
            res = delete_model(key)
            self._log(
                f"  🗑 [{MODEL_SPECS[key]['label']}] 가중치 {res['removed']}개 삭제 "
                f"({res['freed'] / 1e6:.1f} MB 확보)"
            )
            self._event({"status": "deleted", "key": key, "label": f"{key} 삭제 완료"})
            return res

        if target_id.startswith("lang:"):
            parts = target_id.split(":")
            kind = parts[1] if len(parts) > 1 else ""
            code = parts[2] if len(parts) > 2 else self.lang_code
            if kind == "*":
                total = 0
                for k in LANG_MODEL_KINDS:
                    total += delete_lang_model(k, code).get("freed", 0)
                self._event({"status": "deleted", "key": target_id, "label": f"{code} 언어 모델 삭제"})
                return {"ok": True, "freed": total}
            res = delete_lang_model(kind, code)
            self._event({"status": "deleted", "key": target_id, "label": f"{kind} 삭제 완료"})
            return res

        if target_id.startswith("stanza"):
            parts = target_id.split(":")
            code = parts[1] if len(parts) > 1 else self.lang_code
            res = delete_stanza(iso1_of(code))
            self._event({"status": "deleted", "key": target_id, "label": "Stanza 삭제 완료"})
            return res

        return {"ok": False, "error": f"알 수 없는 대상: {target_id}"}

    def delete_all(self) -> dict:
        if self._busy:
            return {"ok": False, "error": "다운로드 중에는 삭제할 수 없습니다."}

        freed = 0
        for key in MODEL_KEYS:
            freed += delete_model(key).get("freed", 0)
        for kind in LANG_MODEL_KINDS:
            freed += delete_lang_model(kind, self.lang_code).get("freed", 0)
        freed += delete_stanza(iso1_of(self.lang_code)).get("freed", 0)

        self._log(f"  🗑 전체 모델 삭제 완료 ({freed / 1e6:.1f} MB 확보)")
        self._event({"status": "deleted", "key": "all", "label": "전체 모델 삭제 완료"})
        return {"ok": True, "freed": freed}

    def target_codes(self, step: str, code: Optional[str] = None) -> List[str]:
        req = STEP_REQUIREMENTS.get(step) or {}
        code = normalize_lang_code(code or self.lang_code)

        if not self.language_resolved:
            codes = list(BOOTSTRAP_LANGUAGES)
            if req.get("bootstrap_lang"):
                return codes
            ready = [c for c in codes if lang_model_ready("qwen3emb", c)]
            if ready:
                return ready
            fallback = embedder_ready_codes()
            if fallback:
                return fallback
            return codes

        return [code]

    def missing_for_step(self, step: str, code: Optional[str] = None) -> List[dict]:
        req = STEP_REQUIREMENTS.get(step)
        if req is None:
            return []

        codes = self.target_codes(step, code)
        out: List[dict] = []
        seen = set()

        def _push(item: dict):
            if item["id"] in seen:
                return
            seen.add(item["id"])
            out.append(item)

        for key in req["base"]:
            if is_manual_only(key):
                continue
            if not is_model_ready(key):
                _push({
                    "id": f"base:{key}",
                    "scope": "base",
                    "key": key,
                    "code": "",
                    "label": MODEL_SPECS[key]["label"],
                })

        for kind in req["lang"]:
            for c in codes:
                if not lang_model_ready(kind, c):
                    _push({
                        "id": f"lang:{kind}:{c}",
                        "scope": "lang",
                        "key": kind,
                        "code": c,
                        "label": f"{LANG_REPO_TEMPLATES[kind]['label']} [{c}]",
                    })

        if req.get("stanza"):
            for c in codes:
                if not stanza_ready(iso1_of(c)):
                    _push({
                        "id": f"stanza:{c}",
                        "scope": "stanza",
                        "key": "stanza",
                        "code": c,
                        "label": f"Stanza NLP [{iso1_of(c)}]",
                    })

        return self.expand_with_prereqs(out, codes)

    def expand_with_prereqs(
        self,
        targets: List[dict],
        codes: Optional[Sequence[str]] = None,
    ) -> List[dict]:
        codes = list(codes or [self.lang_code])
        result: List[dict] = []
        seen = set()

        def _describe(target_id: str) -> Optional[dict]:
            if target_id.startswith("base:"):
                key = target_id.split(":", 1)[1]
                if key not in MODEL_SPECS or is_manual_only(key):
                    return None
                if is_model_ready(key):
                    return None
                return {
                    "id": target_id,
                    "scope": "base",
                    "key": key,
                    "code": "",
                    "label": MODEL_SPECS[key]["label"],
                    "prereq": True,
                }
            if target_id.startswith("lang:"):
                parts = target_id.split(":")
                kind = parts[1]
                c = parts[2] if len(parts) > 2 else codes[0]
                if kind not in LANG_REPO_TEMPLATES:
                    return None
                if lang_model_ready(kind, c):
                    return None
                return {
                    "id": f"lang:{kind}:{c}",
                    "scope": "lang",
                    "key": kind,
                    "code": c,
                    "label": f"{LANG_REPO_TEMPLATES[kind]['label']} [{c}]",
                    "prereq": True,
                }
            return None

        def _visit(item: dict):
            tid = item["id"]
            if tid in seen:
                return
            seen.add(tid)

            key = _prereq_key(tid)
            for dep_key in PREREQUISITES.get(key, ()):
                if dep_key.startswith("lang:"):
                    dep_kind = dep_key.split(":", 1)[1]
                    for c in ([item.get("code")] if item.get("code") else codes):
                        if not c:
                            continue
                        dep = _describe(f"lang:{dep_kind}:{c}")
                        if dep:
                            _visit(dep)
                else:
                    dep = _describe(dep_key)
                    if dep:
                        _visit(dep)

            result.append(item)

        for t in sorted(targets, key=lambda x: _order_rank(x["id"])):
            _visit(t)

        result.sort(key=lambda x: _order_rank(x["id"]))
        return result

    def ensure_step(
        self,
        step: str,
        code: Optional[str] = None,
        required: bool = True,
    ) -> dict:
        req = STEP_REQUIREMENTS.get(step)
        step_label = req["label"] if req else step
        codes = self.target_codes(step, code)
        missing = self.missing_for_step(step, code)

        if not missing:
            return {
                "ok": True, "step": step, "downloaded": False,
                "missing": [], "codes": codes,
            }

        prereq = [m for m in missing if m.get("prereq")]
        direct = [m for m in missing if not m.get("prereq")]
        names = ", ".join(m["label"] for m in missing)

        multi = len(codes) > 1
        lang_note = ""
        if multi:
            lang_note = (
                f"\n언어 기준이 아직 확정되지 않아 기본 언어 "
                f"{', '.join(language_name(c) + f'({c})' for c in codes)} "
                f"모델을 함께 준비합니다."
            )

        prereq_note = ""
        if prereq:
            prereq_note = (
                f"\n선행 필요: {', '.join(m['label'] for m in prereq)} "
                f"(먼저 내려받습니다)"
            )

        if not self.auto_fetch:
            self._notify({
                "action": "open_settings",
                "step": step,
                "step_label": step_label,
                "required": required,
                "auto": False,
                "codes": codes,
                "models": missing,
                "message": (
                    f"'{step_label}' 단계에 필요한 모델이 없습니다.\n"
                    f"필요 모델: {names}"
                    f"{prereq_note}{lang_note}\n"
                    f"환경설정 → 모델 관리에서 내려받아 주세요."
                ),
            })
            return {
                "ok": False, "step": step, "missing": missing,
                "downloaded": False, "codes": codes,
            }

        self._notify({
            "action": "open_settings",
            "step": step,
            "step_label": step_label,
            "required": required,
            "auto": True,
            "codes": codes,
            "models": missing,
            "message": (
                f"'{step_label}' 단계에 필요한 모델이 없어 자동으로 내려받습니다.\n"
                f"대상 {len(missing)}개: {names}"
                f"{prereq_note}{lang_note}\n"
                f"다운로드가 끝나면 작업이 이어서 진행됩니다."
            ),
        })

        self._log(f"📥 '{step_label}' 자동 취득 시작 — 총 {len(missing)}개")
        if prereq:
            self._log(f"   ① 선행 {len(prereq)}개: {', '.join(m['label'] for m in prereq)}")
        if direct:
            self._log(f"   ② 본 단계 {len(direct)}개: {', '.join(m['label'] for m in direct)}")

        failed: List[dict] = []
        done: List[dict] = []

        for idx, m in enumerate(missing, 1):
            self._log(f"   [{idx}/{len(missing)}] {m['label']}")
            res = self.download(m["id"])
            if res.get("ok"):
                done.append(m)
            else:
                failed.append(m)
                if m.get("prereq"):
                    self._log(
                        f"   ⛔ 선행 모델 '{m['label']}' 실패 → "
                        f"후속 다운로드를 중단합니다."
                    )
                    remaining = missing[idx:]
                    failed.extend(remaining)
                    break

        ok = not failed
        if not ok and required:
            ok = self._degraded_ok(step, codes)

        self._notify({
            "action": "fetch_done" if ok else "fetch_failed",
            "step": step,
            "step_label": step_label,
            "codes": codes,
            "models": missing,
            "done": done,
            "failed": failed,
            "message": (
                f"'{step_label}' 모델 준비 완료 ({len(done)}개). 작업을 이어서 진행합니다."
                if ok else
                f"'{step_label}' 모델 준비 실패: "
                + ", ".join(f["label"] for f in failed)
            ),
        })

        return {
            "ok": ok,
            "step": step,
            "downloaded": True,
            "missing": failed,
            "done": done,
            "codes": codes,
        }

    def _degraded_ok(self, step: str, codes: Sequence[str]) -> bool:
        req = STEP_REQUIREMENTS.get(step) or {}

        for key in req.get("base", []):
            if is_manual_only(key):
                continue
            if key == "ppocr-det" and not is_model_ready(key):
                self._log(
                    "   ↩ PP-OCRv5 검출 모델이 없어 영상처리 폴백 검출로 "
                    "진행합니다."
                )
                continue
            if not is_model_ready(key):
                return False

        for kind in req.get("lang", []):
            if kind == "ppocr":
                if any(lang_model_ready(kind, c) for c in codes):
                    continue
                self._log(
                    "   ↩ 요청 언어 PP-OCRv5 rec 는 실패했습니다. "
                    "OCR 없이 VLM 직접 판독으로 진행합니다."
                )
                continue
            if kind != "qwen3emb":
                continue
            if any(lang_model_ready(kind, c) for c in codes):
                continue
            if any_embedder_ready():
                self._log(
                    "   ↩ 요청 언어 임베딩은 실패했지만 다른 언어 임베딩이 있어 "
                    "그것으로 진행합니다."
                )
                continue
            return False

        return True

    def ensure_bootstrap(self, code: Optional[str] = None) -> dict:
        codes = list(BOOTSTRAP_LANGUAGES)
        if code:
            c = normalize_lang_code(code)
            if c not in codes:
                codes.insert(0, c)

        self._log(
            f"🌐 부트스트랩 언어 준비: "
            f"{', '.join(language_name(c) + f'({c})' for c in codes)}"
        )

        if any(lang_model_ready("qwen3emb", c) for c in codes):
            ready = [c for c in codes if lang_model_ready("qwen3emb", c)]
            self._log(f"  ✅ 임베딩 준비됨: {', '.join(ready)}")
            return {"ok": True, "codes": codes, "ready": ready, "downloaded": False}

        return self.ensure_step("bootstrap", code=code, required=True)

    def report_lines(self) -> List[str]:
        s = self.status()
        scope = f"{s['lang_code']} ({s['lang_name']})"
        if not s["language_resolved"]:
            scope += f" · 미확정 → 부트스트랩 {', '.join(s['bootstrap_languages'])}"
        lines = [
            f"  언어 스코프: {scope} | 총 {s['total_bytes'] / 1e9:.2f} GB"
        ]
        for info in s["base"]:
            mark = "OK " if info["ready"] else "MISS"
            lines.append(
                f"    [{mark}] {info['label']:<38} "
                f"{info['bytes'] / 1e6:9.1f} MB  {info['repo'] or '-'}"
            )
        for info in s["lang"]:
            mark = "OK " if info["ready"] else "MISS"
            lines.append(
                f"    [{mark}] {info['label']:<38} "
                f"{info['bytes'] / 1e6:9.1f} MB  {info['repo']}"
            )
        for info in s.get("bootstrap", []):
            mark = "OK " if info["ready"] else "MISS"
            lines.append(
                f"    [{mark}] {info['label']:<38} "
                f"{info['bytes'] / 1e6:9.1f} MB  {info['repo']}"
            )
        st = s["stanza"]
        mark = "OK " if st.get("ready") else "MISS"
        lines.append(
            f"    [{mark}] {st.get('label', 'Stanza'):<38} "
            f"{st.get('bytes', 0) / 1e6:9.1f} MB  {st.get('repo', '-')}"
        )
        return lines