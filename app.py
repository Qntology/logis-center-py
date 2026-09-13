import argparse
import base64
import io
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = BASE_DIR / "schema"
UI_DIR = BASE_DIR / "ui"
OUTPUT_DIR = BASE_DIR / "output"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core import devtools as devtools_mod
from core.crossover import (
    CrossoverSwitch,
    PHASE_EMBEDDING,
    PHASE_GENERATION,
    PHASE_IDLE,
    SLOT_EMBEDDER,
    SLOT_JOINT,
    SLOT_NLP,
    SLOT_OCR,
    SLOT_REFINER,
    SLOT_VISION,
)
from core.device import detect_accelerator, get_vram_info, select_dtype
from core.lang_codes import (
    REFERENCE_LANGUAGE,
    iso1_of,
    language_name,
    normalize_lang_code,
)
from core.language import (
    UNRESOLVED_STAGES as LANG_UNRESOLVED_STAGES,
    LanguageDetector,
    LanguageVerdict,
)
from core.llm import (
    EmbeddingRouter,
    RefinerLLM,
    TextEmbedder,
    resolve_embedder_path,
    resolve_embedder_paths,
    resolve_joint_path,
    resolve_refiner_path,
)
from core import memory as memory_mod
from core import paddle_bootstrap
from core.phrase_cache import CachedEmbedder
from core.model_manager import (
    BOOTSTRAP_LANGUAGES,
    DEFAULT_LANGUAGE,
    LANG_MODEL_KINDS,
    MODELS_ROOT,
    MissingModelError,
    active_language_code,
    any_embedder_ready,
    check_all_models,
    describe_lang_models,
    describe_stanza,
    embedder_ready_codes,
    ensure_model_dir,
    format_lang_report,
    format_model_report,
    format_stanza_report,
    installed_language_codes,
    lang_model_ready,
    lang_models_ready,
    missing_core_models,
    missing_lang_models,
    missing_models,
    paddle_ocr_note,
    paddle_ocr_script_hint,
    paddle_ocr_slug,
    ppocr_det_ready,
    ppocr_ready,
    save_active_language,
    stanza_ready,
)
from core.registry import ModelRegistry
from text_pipeline import TextPipelineConfig, run_text_pipeline
from vision_pipeline import VisionPipelineConfig, VisionPipeline

DOCUMENT_EXT = (".pdf",)
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
TEXT_EXT = (".txt", ".md", ".csv", ".json")


class NMSOcrApp:
    LOW_VRAM_THRESHOLD_GB = 6.0

    def __init__(
        self,
        log: Optional[Callable[[str], None]] = None,
        crossover_enabled: bool = True,
        auto_fetch: bool = True,
        vram_budget: float = 0.0,
        force_low_vram: bool = False,
        paddle_autoinstall: bool = True,
    ):
        self._external_log = log
        self.log_lines: List[str] = []
        self.paddle_autoinstall = bool(paddle_autoinstall)
        if not self.paddle_autoinstall:
            os.environ[paddle_bootstrap.ENV_AUTO] = "0"
        paddle_bootstrap.configure_runtime()

        self.vram_budget = float(vram_budget or 0.0)
        if self.vram_budget <= 0.0:
            info = get_vram_info()
            self.vram_budget = float(info.get("total_gb", 0.0) or 0.0)

        self.low_vram = bool(force_low_vram) or (
            0.0 < self.vram_budget < self.LOW_VRAM_THRESHOLD_GB
        )

        self.crossover = CrossoverSwitch(
            log=self._log,
            enabled=crossover_enabled,
            budget_gb=self.vram_budget,
        )
        self.registry = ModelRegistry(
            log=self._log,
            event=self._download_progress,
            notify=self._model_notify,
            auto_fetch=auto_fetch,
        )

        self.text_embedder: Optional[TextEmbedder] = None
        self.refiner: Optional[RefinerLLM] = None
        self.nlp = None

        self.router = EmbeddingRouter(log=self._log, name="text-router")
        self.vision_router = EmbeddingRouter(log=self._log, name="joint-router")
        self.cached_router: Optional[CachedEmbedder] = None
        self.cached_vision: Optional[CachedEmbedder] = None
        self.language: Optional[LanguageVerdict] = None
        self.saved_lang_code: str = normalize_lang_code(
            active_language_code(DEFAULT_LANGUAGE)
        )
        self.lang_code: str = DEFAULT_LANGUAGE
        self.language_resolved: bool = False
        self.active_codes: List[str] = list(BOOTSTRAP_LANGUAGES)
        self._embedder_slots: Dict[str, str] = {}
        self._primary_embedder_slot: str = ""
        self._ocr_code: str = ""
        self._doc_script: str = ""
        self._joint_registered: bool = False
        self.joint_label: str = ""
        self.joint_dim: int = 0
        self.prefer_grid: str = "joint"

        self.current_image: Optional[Image.Image] = None
        self.current_path: str = ""
        self.current_text: str = ""

        self.last_result: Optional[dict] = None
        self._download_lock = threading.Lock()
        self._models_lock = threading.RLock()
        self.base_ready: bool = False
        self.models_ready: bool = False
        self._language_pending: bool = True

    @property
    def embedder(self):
        return self.crossover.get(SLOT_VISION)

    @property
    def ocr(self):
        return self.crossover.get(SLOT_OCR)

    @property
    def joint(self):
        return self.crossover.get(SLOT_JOINT)

    def embed_fn(self, texts):
        if self.cached_router is not None:
            return self.cached_router(texts)
        return self.router(texts)

    def vision_embed_fn(self, texts):
        if self.cached_vision is not None:
            return self.cached_vision(texts)
        return self.vision_router(texts)

    def _log(self, msg: str):
        self.log_lines.append(msg)
        if self._external_log is not None:
            try:
                self._external_log(msg)
            except Exception:
                pass
        else:
            print(msg, flush=True)
        self._push_ui(f"appendLog({json.dumps(msg)})")

    def _push_ui(self, script: str):
        try:
            import webview
            if webview.windows:
                webview.windows[0].evaluate_js(script)
        except Exception:
            pass

    def _progress(self, pct: int, label: str = ""):
        self._push_ui(f"updateProgress({int(pct)}, {json.dumps(label)})")

    def _download_progress(self, info: dict):
        self._push_ui(f"onModelEvent({json.dumps(info)})")
        status = info.get("status", "")
        if status in ("repo_start", "repo_done", "repo_error", "lang_start",
                      "lang_done", "lang_error", "cancelled", "skip",
                      "deleted", "all_done"):
            self._log(f"  📦 {info.get('label', '')}")

    def _model_notify(self, payload: dict):
        self._push_ui(f"onModelRequired({json.dumps(payload)})")
        msg = payload.get("message", "")
        if msg:
            for line in str(msg).splitlines():
                self._log(f"  🔔 {line}")

    def _sync_registry_language(self):
        self.registry.set_language(self.lang_code, resolved=self.language_resolved)
        if not self.language_resolved:
            self.registry.reset_language_resolution()

    def ensure_step(self, step: str, required: bool = True) -> bool:
        self._sync_registry_language()
        res = self.registry.ensure_step(step, code=self.lang_code, required=required)
        codes = res.get("codes") or []
        if codes:
            self.active_codes = list(codes)
        if not res.get("ok") and required:
            names = ", ".join(m["label"] for m in res.get("missing", []))
            self._log(f"❌ '{step}' 단계 모델 준비 실패: {names}")
        return bool(res.get("ok"))

    def ensure_bootstrap(self) -> bool:
        self._sync_registry_language()

        if any(lang_model_ready("qwen3emb", c) for c in self.active_codes):
            return True

        self._log("═══ 부트스트랩 임베딩 준비 ═══")
        self._log(
            "  ℹ 언어 기준이 아직 없어 기본 언어 "
            + ", ".join(f"{language_name(c)}({c})" for c in BOOTSTRAP_LANGUAGES)
            + " 임베딩을 함께 준비합니다."
        )

        res = self.registry.ensure_bootstrap(code=self.lang_code)
        codes = res.get("codes") or list(BOOTSTRAP_LANGUAGES)
        self.active_codes = list(codes)

        ready = [c for c in codes if lang_model_ready("qwen3emb", c)]
        if ready:
            self._log(f"  ✅ 사용 가능한 임베딩 언어: {', '.join(ready)}")
            return True

        if any_embedder_ready():
            fallback = embedder_ready_codes()
            self.active_codes = fallback
            self._log(f"  ↩ 다른 언어 임베딩으로 진행합니다: {', '.join(fallback)}")
            return True

        self._log("  ❌ 사용 가능한 임베딩 모델이 없습니다.")
        return False

    def get_gpu_info(self) -> dict:
        device, accel = detect_accelerator()
        return {
            "device": str(device),
            "accel_label": accel,
            "dtype": str(select_dtype(device)),
            "vram": get_vram_info(),
            "budget_gb": round(self.vram_budget, 2),
            "low_vram": self.low_vram,
        }

    def _log_vram_profile(self):
        for line in memory_mod.report_lines():
            self._log(line)

        room = memory_mod.usable_ram_gb()
        if 0.0 < room < 6.0:
            self._log(
                f"  ⚠️ 가용 시스템 RAM 이 {room:.1f} GB 뿐입니다. "
                f"4B 정제 LLM 은 가중치를 RAM 에 먼저 펼치므로 "
                f"로드가 중단될 수 있습니다."
            )
            self._log(
                "     다른 프로그램을 닫거나, Windows 가상 메모리를 "
                "늘리거나, 더 작은 모델을 쓰세요."
            )

        if self.vram_budget <= 0.0:
            self._log("  💻 CPU 모드 — VRAM 프로파일을 적용하지 않습니다.")
            return

        self._log(f"  📊 VRAM 예산 {self.vram_budget:.1f} GB")
        if self.low_vram:
            self._log(
                "  ⚠️ 저VRAM 모드: 정제 LLM 은 지연 로드 + 양자화/오프로드로 실행합니다."
            )
            self._log(
                "     Qwen3.5-4B(fp16 ≈ 8.6GB / 4bit ≈ 3.1GB)가 예산을 "
                "초과하면 OCR 원문으로 자동 폴백합니다."
            )
            self._log(
                "     4B 는 2B 보다 유사 글자 변별이 좋지만 4GB 카드에서는 "
                "4bit 양자화가 필수입니다: pip install bitsandbytes"
            )

    def check_models(self) -> dict:
        self._sync_registry_language()
        out = self.registry.status()
        out["crossover"] = self.crossover.stats()
        out["active_codes"] = list(self.active_codes)
        out["base_ready"] = self.base_ready
        out["models_ready"] = self.models_ready
        out["language_pending"] = self._language_pending
        return out

    def download_model(self, target_id: str) -> dict:
        self._sync_registry_language()
        return self.registry.download_async(target_id)

    def download_all_missing(self) -> dict:
        self._sync_registry_language()
        return self.registry.download_all_missing_async()

    def download_bootstrap(self) -> dict:
        self._sync_registry_language()
        import threading as _t
        _t.Thread(
            target=self.registry.ensure_bootstrap,
            kwargs={"code": self.lang_code},
            daemon=True,
        ).start()
        return {"ok": True, "started": "bootstrap", "codes": list(BOOTSTRAP_LANGUAGES)}

    def delete_model(self, target_id: str) -> dict:
        return self.registry.delete(target_id)

    def delete_all_models(self) -> dict:
        return self.registry.delete_all()

    def cancel_download(self) -> dict:
        self.registry.cancel()
        return {"ok": True}

    def set_auto_fetch(self, enabled: bool) -> dict:
        self.registry.auto_fetch = bool(enabled)
        self._log(f"  ⚙ 모델 자동 취득: {'켬' if enabled else '끔'}")
        return {"ok": True, "auto_fetch": self.registry.auto_fetch}

    def print_model_report(self):
        self._log("═══ 모델 상태 ═══")
        self._sync_registry_language()
        for line in self.registry.report_lines():
            self._log(line)

    def _register_slots(self):
        def _load_vision():
            from core.embedding import AXVEEmbedder
            return AXVEEmbedder(str(ensure_model_dir("ax-ve")))

        joint_path, joint_label, joint_code = resolve_joint_path(
            self.lang_code, self.active_codes
        )

        if joint_path:
            if SLOT_JOINT not in self.crossover.slots:
                def _load_joint(p=joint_path, l=joint_label):
                    import traceback
                    try:
                        from core.siglip_joint import Siglip2Joint
                        return Siglip2Joint(p, label=l, log=self._log)
                    except Exception as e:
                        self._log(
                            f"  ❌ [{l}] 조인트 생성 실패 "
                            f"({type(e).__name__}: {e})"
                        )
                        for line in traceback.format_exc().splitlines()[-10:]:
                            self._log(f"      {line}")
                        raise

                self.crossover.register(
                    SLOT_JOINT, _load_joint, label=joint_label, est_gb=1.6
                )
                self.joint_label = joint_label
                self.prefer_grid = "joint"
                self._log(
                    f"  🪢 SigLIP2 조인트 슬롯 등록 — '{joint_label}' "
                    f"(언어 {joint_code}) | 패치 격자와 텍스트 앵커를 "
                    f"동일 대조 공간에서 계산합니다."
                )
            if SLOT_VISION in self.crossover.slots:
                self.crossover.release(SLOT_VISION)
                self.crossover.slots.pop(SLOT_VISION, None)
                self._log("  ⏭ A.X-VE 는 SigLIP2 조인트로 대체되어 등록하지 않습니다.")
        else:
            self.prefer_grid = "vision"
            if SLOT_VISION not in self.crossover.slots:
                self.crossover.register(SLOT_VISION, _load_vision, est_gb=0.9)
            self._log(
                "  ⚠ SigLIP2 언어 텍스트 타워를 찾지 못해 A.X-VE 격자로 "
                "내려갑니다. PP-OCRv5 rec 는 인식 전용이라 패치 격자를 "
                "제공하지 않습니다."
            )

        self._register_ocr_slot()

    def _pick_ocr_code(self) -> str:
        pool: List[str] = []
        if self.language_resolved and self.lang_code:
            pool.append(self.lang_code)
        pool.extend(self.active_codes)
        pool.extend(BOOTSTRAP_LANGUAGES)

        for c in pool:
            if c and ppocr_ready(c):
                return normalize_lang_code(c)
        return normalize_lang_code(pool[0] if pool else DEFAULT_LANGUAGE)

    def _register_ocr_slot(self):
        if SLOT_OCR in self.crossover.slots:
            return

        code = self._ocr_code or self._pick_ocr_code()
        self._ocr_code = code

        def _load_ocr(c=code):
            from core.ppocr import PaddleOCRRec
            rec = PaddleOCRRec(c, log=self._log)
            try:
                rec.attach_detector()
            except Exception as e:
                self._log(f"  ⏭ 검출기 결합 생략 ({e})")
            return rec

        slug = paddle_ocr_slug(code)
        self.crossover.register(
            SLOT_OCR,
            _load_ocr,
            label=f"PP-OCRv5 rec [{slug or 'ch'}]",
            est_gb=0.25,
        )

        note = paddle_ocr_note(code)
        if note:
            self._log(f"  ℹ OCR 언어 대체: {note}")
        if not ppocr_det_ready():
            self._log(
                "  ℹ PP-OCRv5 검출 모델(base:ppocr-det)이 없어 크롭 경계 "
                "스냅은 영상처리 폴백으로 동작합니다. 환경설정에서 받으면 "
                "글자 잘림이 더 줄어듭니다."
            )

    def _rebind_ocr(self, code: str) -> bool:
        code = normalize_lang_code(code or DEFAULT_LANGUAGE)
        if code == self._ocr_code and self.crossover.get(SLOT_OCR) is not None:
            return True

        self.crossover.release(SLOT_OCR)
        self.crossover.slots.pop(SLOT_OCR, None)
        self._ocr_code = code
        self._register_ocr_slot()

        try:
            self.crossover.acquire(SLOT_OCR)
            return True
        except Exception as e:
            self._log(f"  ⚠ PP-OCRv5 rec '{code}' 로드 실패: {e}")
            return False

    def _pick_detection_ocr(self, image) -> str:
        codes = [
            c for c in dict.fromkeys(
                list(self.active_codes) + list(BOOTSTRAP_LANGUAGES)
            ) if c and ppocr_ready(c)
        ]

        if not codes:
            self._log(
                "  ⏭ 설치된 PP-OCRv5 rec 가 없어 OCR 표본을 만들 수 없습니다."
            )
            return ""

        if len(codes) == 1 or image is None:
            self._rebind_ocr(codes[0])
            return codes[0]

        self._log(
            f"  🔤 OCR 후보 {len(codes)}종({', '.join(codes)})을 "
            f"CTC 인식 확신도로 비교합니다."
        )

        best = ""
        best_score = -1.0
        for c in codes:
            if not self._rebind_ocr(c):
                continue
            ocr = self.crossover.get(SLOT_OCR)
            if ocr is None or not getattr(ocr, "available", False):
                continue
            try:
                score = float(ocr.probe_confidence(image))
            except Exception as e:
                self._log(f"     · {c}: 확신도 측정 실패 ({e})")
                continue
            self._log(f"     · {c}({paddle_ocr_slug(c)}): 평균 확신도 {score:.4f}")
            if score > best_score:
                best = c
                best_score = score

        if best:
            self._rebind_ocr(best)
            self._log(
                f"  🥇 OCR 언어 '{best}' 선택 (확신도 {best_score:.4f})"
            )
        return best

    def job_queue(self):
        q = getattr(self, "_job_queue", None)
        if q is None:
            from core.jobqueue import JobQueue

            def _emit(info: dict):
                self._push_ui(f"onJobState({json.dumps(info)})")

            q = JobQueue(log=self._log, name="pipeline", event=_emit)
            self._job_queue = q
        return q

    def job_slot(self, title: str):
        return self.job_queue().slot(title)

    def job_stats(self) -> dict:
        return self.job_queue().stats()

    def _register_embed_provider(self):
        joint_on = SLOT_JOINT in self.crossover.slots

        if joint_on and not self._joint_registered:
            def _joint_embed(texts):
                jnt = self.crossover.get(SLOT_JOINT)
                if jnt is None:
                    jnt = self.crossover.acquire(SLOT_JOINT)
                if jnt is None:
                    return None
                return jnt.encode_text(texts)

            label = self.joint_label or "siglip2"

            stale = [
                n for n in self.vision_router.providers()
                if n != label and (n.startswith("siglip2") or n == "text-router")
            ]
            for n in stale:
                self.vision_router.unregister(n)
            if stale:
                self._log(
                    f"  🧹 [joint-router] 구버전 제공자 {stale} 를 "
                    f"제거하고 '{label}' 단독으로 재구성합니다."
                )

            self.vision_router.register(label, _joint_embed, priority=200)
            self.vision_router.promote(label, priority=200)

            jnt = self.crossover.get(SLOT_JOINT)
            if jnt is None:
                try:
                    jnt = self.crossover.acquire(SLOT_JOINT)
                except Exception as e:
                    self._log(f"  ⚠ 조인트 획득 실패({e})")
                    jnt = None
            jdim = int(getattr(jnt, "dim", 0) or 0) if jnt is not None else 0
            if jdim > 0:
                self.vision_router.lock_space(jdim, owner=label)
            self._joint_registered = True

        if not joint_on:
            self.vision_router.lock_space(0)
            if "text-router" not in self.vision_router.providers():
                self.vision_router.register(
                    "text-router", lambda texts: self.router(texts), priority=100
                )
                self._log(
                    "  ⚠ SigLIP2 조인트가 없어 비전 앵커를 텍스트 임베딩으로 "
                    "대체합니다. 패치 격자와 차원이 다르면 문서 유형 분류가 "
                    "중단됩니다."
                )

        recipe = (
            f"{self.joint_label or 'siglip2'}-joint"
            if self._joint_registered else "text-router-joint"
        )

        probe = self.vision_router(["__dim_probe__"])
        dim = int(probe.shape[-1]) if probe is not None and probe.size else 0
        self.joint_dim = dim

        if self._joint_registered and dim <= 1:
            self._log(
                "  ❌ 조인트 텍스트 인코더가 동작하지 않습니다. "
                "텍스트 임베딩 공간으로 되돌립니다."
            )
            self._demote_joint()
            return self._register_embed_provider()

        self.cached_vision = CachedEmbedder(
            self.vision_router, recipe=recipe, dim=dim, log=self._log
        )
        self._log(
            f"  🪢 비전-텍스트 공동 임베딩 채널 준비 "
            f"({self.vision_router.active or recipe}, dim={dim}) — "
            f"패치 격자와 동일 공간에서만 코사인을 계산합니다."
        )
        for line in self.vision_router.report_lines():
            self._log(line)

    def _demote_joint(self):
        label = self.joint_label or "siglip2"
        try:
            self.crossover.release(SLOT_JOINT)
        except Exception:
            pass
        self.crossover.slots.pop(SLOT_JOINT, None)
        self.vision_router.unregister(label)
        self.vision_router.lock_space(0)
        self._joint_registered = False
        self.joint_label = ""
        self.joint_dim = 0
        self.prefer_grid = "vision"
        if SLOT_VISION not in self.crossover.slots:
            def _load_vision():
                from core.embedding import AXVEEmbedder
                return AXVEEmbedder(str(ensure_model_dir("ax-ve")))
            self.crossover.register(SLOT_VISION, _load_vision, est_gb=0.9)
        if "text-router" not in self.vision_router.providers():
            self.vision_router.register(
                "text-router", lambda texts: self.router(texts), priority=100
            )
        self._log(
            "  ↩ 조인트 슬롯을 내리고 A.X-VE 격자 + 텍스트 앵커로 "
            "복귀했습니다."
        )

    def load_base_models(self) -> dict:
        missing = missing_core_models()
        if missing:
            self._log(f"📥 필수 모델 {len(missing)}개 누락 → 자동 취득을 시도합니다.")
            if not self.ensure_step("patch_grid", required=True):
                return {
                    "ok": False,
                    "error": (
                        f"필수 모델 누락: {', '.join(missing)}\n"
                        f"환경설정 → 모델 관리에서 내려받거나 "
                        f"models/ 하위에 직접 배치해 주세요."
                    ),
                }

        if not self.ensure_bootstrap():
            return {
                "ok": False,
                "error": (
                    "임베딩 모델을 확보하지 못했습니다.\n"
                    f"기본 언어 {', '.join(BOOTSTRAP_LANGUAGES)} 중 "
                    f"최소 하나의 Qwen3-Embedding 이 필요합니다.\n"
                    "환경설정 → 모델 관리에서 내려받아 주세요."
                ),
            }

        self._register_slots()

        if not paddle_bootstrap.ready():
            self._log("═══ PP-OCRv5 실행 백엔드 확인 ═══")
            paddle_bootstrap.ensure_paddle(
                log=self._log, auto=self.paddle_autoinstall
            )
        for line in paddle_bootstrap.report_lines():
            self._log(line)

        if SLOT_JOINT in self.crossover.slots:
            try:
                self._log("🔄 SigLIP2 조인트(비전-텍스트) 로드 중...")
                self._progress(10, "SigLIP2 조인트 로드 중")
                self.crossover.acquire(SLOT_JOINT)
                self._progress(30, "SigLIP2 조인트 완료")
            except Exception as e:
                self._log(f"⚠ SigLIP2 조인트 로드 실패({e}) → Hayai 공간으로 진행합니다.")
                from core import diagnostics as _diag
                if _diag.enabled(2):
                    import traceback
                    for line in traceback.format_exc().splitlines()[-8:]:
                        self._log(f"      {line}")
                self._demote_joint()
        else:
            try:
                self._log("🔄 A.X-VE 비전 인코더 로드 중...")
                self._progress(10, "A.X-VE 로드 중")
                self.crossover.acquire(SLOT_VISION)
                self._progress(30, "A.X-VE 완료")
            except Exception as e:
                self._log(f"❌ A.X-VE 로드 실패: {e}")
                return {"ok": False, "error": str(e)}

        try:
            self._log(
                f"🔄 PP-OCRv5 rec 로드 중... "
                f"(언어 {self._ocr_code or '-'} / "
                f"{paddle_ocr_slug(self._ocr_code or DEFAULT_LANGUAGE)})"
            )
            self._progress(45, "PP-OCRv5 rec 로드 중")
            self.crossover.acquire(SLOT_OCR)
            self._progress(60, "PP-OCRv5 rec 완료")
        except Exception as e:
            self._log(
                f"⚠ PP-OCRv5 rec 로드 실패({e}) → OCR 초안 없이 "
                f"VLM 직접 판독으로 진행합니다."
            )

        self._register_embed_provider()
        self.crossover.phase = PHASE_EMBEDDING
        self._log_vram_profile()
        self._progress(65, "기본 모델 로드 완료")
        self.base_ready = True
        return {"ok": True}

    UNRESOLVED_STAGES = LANG_UNRESOLVED_STAGES

    def detect_language(self, image: Optional[Image.Image] = None) -> LanguageVerdict:
        img = image if image is not None else self.current_image

        self._log("═══ 언어 판별 ═══")
        self.ensure_bootstrap()
        self.ensure_step("language", required=False)

        judge_codes: List[str] = [REFERENCE_LANGUAGE]
        for c in list(BOOTSTRAP_LANGUAGES) + list(self.active_codes):
            if c and c not in judge_codes:
                judge_codes.append(c)

        entries = self._register_text_embedders(judge_codes, promote=True)
        if entries:
            self._log(
                f"  ⚖ 판별 기준 임베딩 '{entries[0][2]}' | 후보 "
                + ", ".join(lbl for _c, _p, lbl in entries)
            )
        else:
            self._log("  ⏭ 다국어 임베딩이 없어 Hayai 임베딩으로 판별합니다.")

        self._pick_detection_ocr(img)

        if self.ocr is None or not getattr(self.ocr, "available", False):
            self._log(
                "  ⚠️ PP-OCRv5 rec 가 준비되지 않아 OCR 표본 없이 판별합니다. "
                "유니코드 블록 확정이 불가능하면 기본 언어로 진행합니다."
            )

        installed = list(installed_language_codes())
        complete = [c for c in installed if lang_models_ready(c)]
        partial = [c for c in installed if c not in complete]

        served = list(dict.fromkeys(
            [c for c in judge_codes if c]
            + list(BOOTSTRAP_LANGUAGES)
            + complete
        ))
        self._log(
            "  🎯 서비스 가능한 언어 스코프: " + ", ".join(served)
        )
        if partial:
            self._log(
                f"  ⏭ 부분 설치 언어 {', '.join(partial)} 는 스코프에서 "
                f"제외합니다 (중단된 다운로드 흔적일 수 있습니다)."
            )

        detector = LanguageDetector(
            ocr=self.ocr,
            embed_fn=self.embed_fn,
            log=self._log,
            nlp=self.nlp,
            served_codes=served,
        )

        if img is not None:
            verdict = detector.detect(img)
        elif self.current_text:
            verdict = detector.detect_from_text(self.current_text)
        else:
            verdict = LanguageVerdict(
                code=DEFAULT_LANGUAGE, script=None, confidence=0.0,
                margin=0.0, stage="no-input", candidates=[],
            )

        self.language = verdict
        self._language_pending = False

        resolved = (
            verdict.stage not in self.UNRESOLVED_STAGES
            and bool(verdict.script)
        )

        block_decided = verdict.stage == "unicode-block"

        if block_decided:
            self._log(
                f"  🧱 유니코드 블록으로 확정된 언어입니다 — "
                f"마진 검사를 면제합니다."
            )
        elif resolved and verdict.code not in served and verdict.margin < 0.05:
            self._log(
                f"  🚧 '{verdict.code}' 는 미설치 언어이고 마진이 "
                f"{verdict.margin:+.4f} 로 약합니다. 수 GB 다운로드를 "
                f"피하기 위해 기본 언어로 진행합니다."
            )
            self._log(
                f"     이 언어를 쓰시려면 환경설정 → 모델 관리에서 "
                f"'{verdict.code}' 를 직접 받아 주세요."
            )
            resolved = False

        if resolved:
            self.lang_code = verdict.code
            self.language_resolved = True
            self.active_codes = [verdict.code]
            if verdict.script:
                self._doc_script = str(verdict.script)
            save_active_language(
                verdict.code, {"source": "detector", "stage": verdict.stage}
            )
            self._log(
                f"  🌐 확정 언어: {verdict.code} ({verdict.name}) "
                f"| script={verdict.script} | margin={verdict.margin:+.4f}"
            )
            self._log(
                f"  🔤 문서 스크립트 래치 '{self._doc_script}' — 행 판독 "
                f"프롬프트에 이 문자체계를 강제합니다."
            )
        else:
            self.language_resolved = False
            self.active_codes = list(BOOTSTRAP_LANGUAGES)
            self.lang_code = verdict.code or DEFAULT_LANGUAGE
            self._log(
                f"  ⚠️ 언어를 확정하지 못했습니다 (stage={verdict.stage}). "
                f"기본 언어 {', '.join(BOOTSTRAP_LANGUAGES)} 로 진행합니다."
            )

        self._sync_registry_language()
        return verdict

    def ensure_language_models(self, code: Optional[str] = None, fetch: bool = True) -> dict:
        code = normalize_lang_code(code or self.lang_code or DEFAULT_LANGUAGE)

        if lang_models_ready(code):
            self._log(f"  ✅ '{code}' 언어 모델 3종 준비됨")
            self.lang_code = code
            return {"ok": True, "code": code, "downloaded": False}

        missing = missing_lang_models(code)
        self._log(f"  📥 '{code}' 누락 모델: {', '.join(missing)}")

        if not fetch:
            return {"ok": False, "code": code, "error": "자동 취득이 비활성화되었습니다."}

        with self._download_lock:
            from core.model_fetcher import LanguageModelFetcher
            fetcher = LanguageModelFetcher(progress_callback=self._download_progress)
            res = fetcher.ensure_language(code, fallback=DEFAULT_LANGUAGE, log=self._log)

        if res.get("ok"):
            self.lang_code = res.get("code", code)
            self._log(f"  ✅ '{self.lang_code}' 언어 모델 준비 완료")
        else:
            self._log(f"  ⚠ 언어 모델 준비 실패: {res.get('error', '')}")
        return res

    def ensure_stanza(self, code: Optional[str] = None, fetch: bool = True) -> dict:
        code = normalize_lang_code(code or self.lang_code or DEFAULT_LANGUAGE)
        iso1 = iso1_of(code)

        if stanza_ready(iso1):
            self._log(f"  ✅ Stanza '{iso1}' 준비됨")
            return {"ok": True, "iso1": iso1, "downloaded": False}

        if not fetch:
            return {"ok": False, "iso1": iso1, "error": "자동 취득이 비활성화되었습니다."}

        with self._download_lock:
            from core.model_fetcher import StanzaFetcher
            res = StanzaFetcher(progress_callback=self._download_progress).ensure(
                iso1, log=self._log
            )
        return res

    def _register_text_embedders(
        self,
        codes: Sequence[str],
        promote: bool = False,
    ) -> List[tuple]:
        entries = resolve_embedder_paths(list(codes))
        if not entries:
            return []

        for idx, (ecode, epath, elabel) in enumerate(entries):
            key = ecode or "default"
            slot_name = self._embedder_slots.get(key)

            if slot_name is None:
                slot_name = (
                    SLOT_EMBEDDER if not self._primary_embedder_slot
                    else f"{SLOT_EMBEDDER}:{key}"
                )
                self._embedder_slots[key] = slot_name
                if not self._primary_embedder_slot:
                    self._primary_embedder_slot = slot_name

                def _make_loader(p=epath, l=elabel):
                    return lambda: TextEmbedder(p, label=l, log=self._log)

                def _make_embed(sn=slot_name):
                    def _fn(texts):
                        obj = self.crossover.get(sn)
                        if obj is None:
                            obj = self.crossover.acquire(sn)
                        if obj is None:
                            return None
                        if sn == self._primary_embedder_slot:
                            self.text_embedder = obj
                        return obj.encode(texts)
                    return _fn

                self.crossover.register(
                    slot_name, _make_loader(), label=elabel, est_gb=1.4
                )
                self.router.register(
                    elabel, _make_embed(), priority=50 - idx
                )

            if promote and idx == 0 and slot_name != self._primary_embedder_slot:
                if self.router.promote(elabel, priority=80):
                    self._primary_embedder_slot = slot_name

        return entries

    def load_language_models(self, fetch: bool = True) -> dict:
        code = self.lang_code or DEFAULT_LANGUAGE

        self._log("═══ 언어 스코프 모델 준비 ═══")
        self.registry.auto_fetch = bool(fetch)
        self._sync_registry_language()

        if self.language_resolved:
            self.ensure_language_models(code, fetch=fetch)
            code = self.lang_code
            self.active_codes = [code]
        else:
            self.ensure_bootstrap()
            if not self.active_codes:
                self.active_codes = list(BOOTSTRAP_LANGUAGES)
            code = self.active_codes[0]
            self.lang_code = code
            self._log(
                "  ℹ 언어 미확정 — 기본 언어 "
                + ", ".join(self.active_codes)
                + f" 모델을 사용합니다. (스코프 기준 '{code}')"
            )
        self._sync_registry_language()

        want_ocr = self._pick_ocr_code()
        if want_ocr and want_ocr != self._ocr_code:
            self._log(
                f"  🔤 OCR 언어 재바인딩 '{self._ocr_code or '-'}' → "
                f"'{want_ocr}' ({paddle_ocr_slug(want_ocr)})"
            )
            self._rebind_ocr(want_ocr)

        jp, jl, jc = resolve_joint_path(self.lang_code, self.active_codes)
        if jp and jl != self.joint_label:
            old = self.joint_label or "-"
            self.crossover.release(SLOT_JOINT)
            self.crossover.slots.pop(SLOT_JOINT, None)
            self._joint_registered = False
            self.joint_label = ""
            self.vision_router.lock_space(0)
            self._register_slots()
            self._register_embed_provider()
            self._log(
                f"  🪢 조인트 공간 재바인딩 '{old}' → '{jl}' (언어 {jc}) "
                f"| 활성 '{self.vision_router.active}' "
                f"| 고정 {self.vision_router.space_dim}차원 "
                f"| prefer_grid={self.prefer_grid}"
            )

        entries = self._register_text_embedders(self.active_codes, promote=True)
        if entries:
            self._log(
                f"  🔤 임베딩 후보 {len(entries)}개: "
                + ", ".join(lbl for _c, _p, lbl in entries)
            )

            if self._primary_embedder_slot:
                try:
                    self._progress(75, "텍스트 임베딩 로드 중")
                    self.text_embedder = self.crossover.acquire(
                        self._primary_embedder_slot
                    )
                except Exception as e:
                    self._log(f"  ⚠ 텍스트 임베딩 로드 실패: {e}")
        else:
            self._log("  ⏭ 텍스트 임베딩 모델을 찾지 못해 Hayai 임베딩만 사용합니다.")

        if not self._ensure_refiner_slot():
            self._log("  ⏭ 정제 LLM 을 찾지 못해 OCR 원문을 그대로 사용합니다.")

        self._log("═══ NLP 게이트 준비 ═══")
        self.ensure_stanza(code, fetch=fetch)
        try:
            from core.nlp import StanzaNLP
            self.nlp = StanzaNLP(code, log=self._log)
        except Exception as e:
            self._log(f"  ⚠ Stanza 초기화 실패: {e}")
            self.nlp = None

        probe = self.router(["__dim_probe__"])
        dim = int(probe.shape[-1]) if probe is not None and probe.size else 0
        recipe = f"{self.router.active or 'router'}-{'+'.join(self.active_codes)}"
        self.cached_router = CachedEmbedder(
            self.router, recipe=recipe, dim=dim, log=self._log
        )
        self._log(f"  💾 앵커 캐시 활성 (recipe={recipe}, dim={dim})")

        self._progress(100, "모든 모델 준비 완료")
        self._log(f"  🔀 임베딩 제공자: {', '.join(self.router.providers())}")
        for line in self.crossover.report_lines():
            self._log(line)
        self.models_ready = True
        return {
            "ok": True,
            "code": code,
            "codes": list(self.active_codes),
            "resolved": self.language_resolved,
            "providers": self.router.providers(),
        }

    def load_models(self, fetch: bool = True) -> dict:
        with self._models_lock:
            self.registry.auto_fetch = bool(fetch)

            base = self.load_base_models()
            if not base.get("ok"):
                self.models_ready = False
                return base

            if self.current_image is not None or self.current_text:
                self.detect_language()
            else:
                self._log(
                    "  ℹ 입력이 없어 언어를 판별하지 않았습니다. "
                    f"기본 언어 {', '.join(BOOTSTRAP_LANGUAGES)} 로 준비합니다."
                )
                self.language_resolved = False
                self.active_codes = list(BOOTSTRAP_LANGUAGES)

            return self.load_language_models(fetch=fetch)

    def ensure_models_ready(self, fetch: bool = True) -> dict:
        with self._models_lock:
            has_input = self.current_image is not None or bool(self.current_text)

            need_base = not self.base_ready
            if not need_base:
                if SLOT_OCR not in self.crossover.slots:
                    need_base = True
                elif SLOT_JOINT not in self.crossover.slots and \
                        SLOT_VISION not in self.crossover.slots:
                    need_base = True

            if need_base or not self.models_ready:
                self._log(
                    f"🧩 모델이 준비되지 않아 실행과 함께 자동 로드합니다. "
                    f"(base_ready={self.base_ready} / "
                    f"models_ready={self.models_ready})"
                )
                res = dict(self.load_models(fetch=fetch) or {})
                res["auto_loaded"] = True
                return res

            if has_input and self._language_pending and not self.language_resolved:
                self._log("🌐 입력이 바뀌어 언어를 다시 판별합니다.")
                self.detect_language()
                res = dict(self.load_language_models(fetch=fetch) or {})
                res["auto_loaded"] = True
                return res

            return {"ok": True, "auto_loaded": False}

    def open_file_dialog(self) -> Optional[str]:
        try:
            import webview
            result = webview.windows[0].create_file_dialog(
                webview.OPEN_DIALOG,
                directory=str(BASE_DIR),
                file_types=(
                    "지원 파일 (*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.pdf;*.txt;*.md;*.csv)",
                    "모든 파일 (*.*)",
                ),
            )
            if result:
                return result[0]
        except Exception:
            pass
        return None

    def load_input(self, path: str = "") -> dict:
        if not path:
            path = self.open_file_dialog() or ""
        if not path:
            return {"ok": False, "error": "파일이 선택되지 않았습니다."}

        p = Path(path)
        if not p.exists():
            return {"ok": False, "error": f"파일이 없습니다: {path}"}

        self.current_path = str(p)
        self.current_image = None
        self.current_text = ""
        self.language_resolved = False
        self.language = None
        self._language_pending = True
        self._doc_script = ""
        self._sync_registry_language()
        ext = p.suffix.lower()

        self._log(f"📂 입력 로드: {p.name}")

        try:
            if ext in TEXT_EXT:
                self.current_text = p.read_text(encoding="utf-8", errors="ignore")
                self._log(f"  ✅ 텍스트 {len(self.current_text)}자")
                return {"ok": True, "mode": "text", "chars": len(self.current_text)}

            if ext in DOCUMENT_EXT:
                self.current_image = self._render_pdf(p)
            else:
                self.current_image = Image.open(p).convert("RGB")

            self._log(
                f"  ✅ 이미지 {self.current_image.width}x{self.current_image.height}"
            )
            return {
                "ok": True,
                "mode": "image",
                "width": self.current_image.width,
                "height": self.current_image.height,
            }
        except Exception as e:
            self._log(f"❌ 입력 로드 실패: {e}")
            return {"ok": False, "error": str(e)}

    def _render_pdf(self, path: Path) -> Image.Image:
        try:
            import fitz
            doc = fitz.open(str(path))
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()
            return img
        except ImportError:
            from pdf2image import convert_from_path
            return convert_from_path(str(path), first_page=1, last_page=1, dpi=150)[0].convert("RGB")

    def list_schemas(self) -> dict:
        out = []
        dictionaries = []
        for f in sorted(SCHEMA_DIR.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except Exception:
                dictionaries.append(f)
                continue
            if not isinstance(data, dict):
                dictionaries.append(f)
                continue
            fields = data.get("fields")
            if not isinstance(fields, dict) or not fields:
                dictionaries.append(f)
                continue
            out.append({
                "filename": f.name,
                "domain": data.get("domain", "unknown"),
                "doc_type": data.get("doc_type", "unknown"),
                "field_count": len(fields),
            })

        if not out:
            scope = self.lang_code if self.language_resolved else ""
            for f in dictionaries:
                hit = False
                for mod, fn in (
                    ("core.trade_schema", "load_trade_schemas"),
                    ("core.comics_schema", "load_comics_schemas"),
                ):
                    try:
                        loader = getattr(__import__(mod, fromlist=[fn]), fn)
                        schemas, _diag = loader(f, scope)
                    except Exception as e:
                        self._log(
                            f"  ⏭ {mod} 스키마 조립 실패 "
                            f"({type(e).__name__}: {e})"
                        )
                        continue
                    for code, sch in schemas.items():
                        out.append({
                            "filename": f"{f.name}#{code}",
                            "domain": sch.get("domain", "trade"),
                            "doc_type": code,
                            "field_count": len(sch.get("fields", {}) or {}),
                        })
                    if schemas:
                        hit = True
                if hit:
                    break

        return {"ok": True, "schemas": out}

    def load_schema(self, filename: str) -> dict:
        name = str(filename or "")
        ref_code = ""
        if "#" in name:
            name, ref_code = name.split("#", 1)

        p = SCHEMA_DIR / name
        if not p.exists():
            return {"ok": False, "error": f"스키마 없음: {filename}"}

        try:
            if ref_code:
                scope = self.lang_code if self.language_resolved else ""
                schema = None
                for mod, fn in (
                    ("core.trade_schema", "load_trade_schemas"),
                    ("core.comics_schema", "load_comics_schemas"),
                ):
                    try:
                        loader = getattr(__import__(mod, fromlist=[fn]), fn)
                    except Exception:
                        continue
                    try:
                        schemas, diag = loader(p, scope)
                    except Exception:
                        continue
                    if ref_code in schemas:
                        for line in diag:
                            self._log(line)
                        schema = schemas[ref_code]
                        break
                if schema is None:
                    return {
                        "ok": False,
                        "error": f"'{name}' 안에 '{ref_code}' 서식이 없습니다.",
                    }
                self._log(
                    f"  📋 '{ref_code}' 서식 스키마 로드 — "
                    f"도메인 {schema.get('domain')} | "
                    f"필드 {len(schema.get('fields', {}) or {})}개"
                )
            else:
                with open(p, "r", encoding="utf-8") as f:
                    schema = json.load(f)

            return {
                "ok": True,
                "domain": schema.get("domain", ""),
                "doc_type": schema.get("doc_type", ""),
                "fields": list((schema.get("fields", {}) or {}).keys()),
                "schema": schema,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def classify_document(self, skip_ready: bool = False) -> dict:
        with self.job_slot("문서 유형 분류"):
            return self._classify_document_body(skip_ready=skip_ready)

    def _classify_document_body(self, skip_ready: bool = False) -> dict:
        if self.current_image is None:
            return {"ok": False, "error": "이미지가 로드되지 않았습니다."}

        if not skip_ready:
            ready = self.ensure_models_ready()
            if not ready.get("ok"):
                return {
                    "ok": False,
                    "error": ready.get("error", "모델 자동 로드에 실패했습니다."),
                }

        if not self.ensure_step("doc_type", required=True):
            return {"ok": False, "error": "문서 유형 분류에 필요한 모델이 없습니다."}

        self.crossover.enter_embedding_phase()
        if self.joint is None and self.embedder is None:
            if SLOT_JOINT not in self.crossover.slots and \
                    SLOT_VISION not in self.crossover.slots:
                return {
                    "ok": False,
                    "error": (
                        "패치 격자 모델이 없습니다. "
                        "SigLIP2 조인트(lang:siglip2) 또는 A.X-VE(base:ax-ve)를 "
                        "준비하세요."
                    ),
                }

        self._log("═══ 문서 유형 분류 (Doc Type NMS) ═══")

        joint_obj = self.joint
        if joint_obj is None and SLOT_JOINT in self.crossover.slots:
            try:
                joint_obj = self.crossover.acquire(
                    SLOT_JOINT, protect=[SLOT_OCR]
                )
            except Exception as e:
                self._log(f"  ⚠ 조인트 선획득 실패({e}) → Hayai 격자로 진행")
                joint_obj = None

        prefer = self.prefer_grid
        if joint_obj is None and prefer == "joint":
            self._log(
                "  🚧 조인트 모델이 메모리에 없어 격자를 A.X-VE 로 되돌립니다. "
                "앵커 공간도 함께 되돌려 반쪽 폴백을 막습니다."
            )
            self._demote_joint()
            self._register_embed_provider()
            prefer = self.prefer_grid

        pipeline = VisionPipeline(
            self.vision_embed_fn, ocr=self.ocr, embedder=self.embedder,
            config=VisionPipelineConfig(
                lang_code=self.lang_code if self.language_resolved else "",
                prefer_grid=prefer,
            ),
            log=self._log, nlp=self.nlp,
            crossover=self.crossover, joint=joint_obj,
        )
        sample = ""
        if self.language is not None:
            sample = str(getattr(self.language, "sample_text", "") or "")
        if not sample:
            sample = self.current_text or ""
        verdict = pipeline.classify(
            self.current_image, SCHEMA_DIR,
            text_sample=sample,
            text_embed_fn=self.embed_fn,
        )

        if not verdict.code:
            return {
                "ok": False,
                "error": (
                    "문서 유형을 판정하지 못했습니다.\n"
                    f"  스키마 경로 : {SCHEMA_DIR}\n"
                    "  위 로그의 '📚 스키마 적재' 와 '📖 앵커 임베딩' 줄에서 "
                    "스펙 수 / 임베딩 차원을 확인하세요."
                ),
            }

        if verdict.needs_llm:
            self._log(
                f"  ⚠️ 문서 유형 '{verdict.group}/{verdict.code}' 는 마진 미달입니다. "
                f"스키마를 직접 선택하면 정확도가 크게 올라갑니다."
            )

        return {
            "ok": True,
            "schema": verdict.schema_file,
            "verdict": verdict.to_dict(),
            "low_confidence": bool(verdict.needs_llm),
        }

    REFINER_EST_GB_4BIT = 3.1
    REFINER_EST_GB_FP16 = 8.6

    def _ensure_refiner_slot(self) -> bool:
        if SLOT_REFINER in self.crossover.slots:
            return True

        ref_path, ref_label = resolve_refiner_path(
            self.lang_code, codes=self.active_codes
        )
        if not ref_path:
            return False

        low = self.low_vram
        budget = self.vram_budget

        def _load_refiner():
            memory_mod.reclaim(log=self._log, label="정제 LLM 적재 전")
            return RefinerLLM(
                ref_path,
                label=ref_label,
                log=self._log,
                low_vram=low,
                budget_gb=budget,
            )

        est = self.REFINER_EST_GB_4BIT if low else self.REFINER_EST_GB_FP16
        disk = memory_mod.model_disk_gb(ref_path)

        self.crossover.register(
            SLOT_REFINER, _load_refiner, label=ref_label, est_gb=est
        )
        mode = "저VRAM(양자화/오프로드)" if low else "표준"
        self._log(
            f"  ⏳ [{ref_label}] 정제 LLM 지연 로드 예약 — {mode} "
            f"| 예상 VRAM {est:.1f} GB | 가중치 {disk:.1f} GB"
        )

        stage = disk * 1.2 + memory_mod.RAM_SAFETY_GB
        room = memory_mod.usable_ram_gb()
        if room > 0.0 and room < stage:
            self._log(
                f"  ⚠️ [RAM] 현재 가용 {room:.1f} GB < 적재에 필요한 "
                f"{stage:.1f} GB — 정제 LLM 로드 시점에 다른 모델을 "
                f"먼저 반납합니다."
            )
        return True

    def _refine_fn(self, hint: str = "", schema: Optional[dict] = None):
        if SLOT_REFINER not in self.crossover.slots:
            return None

        fields = (schema or {}).get("fields", {}) or {}
        specs: Dict[str, Dict[str, str]] = {}
        labels: List[str] = []
        array_cats: set = set()

        for name, definition in fields.items():
            d = definition if isinstance(definition, dict) else {}
            cat = str(d.get("category") or "misc")
            desc = str(d.get("semantic") or name.replace("_", " "))
            specs.setdefault(cat, {})[name] = desc[:90]
            if d.get("array"):
                array_cats.add(cat)
            for key in ("semantic", "label"):
                v = d.get(key)
                if isinstance(v, str) and v.strip():
                    labels.append(v.strip())

        claimed: Dict[str, str] = {}
        rows_by_cat: Dict[str, List[dict]] = {}

        lang = self.lang_code if self.language_resolved else ""
        probe_src = self.current_image
        state = {"script": self._doc_script, "probed": bool(self._doc_script)}

        def _probe_script(obj) -> str:
            if state["probed"]:
                return state["script"]
            state["probed"] = True

            if not getattr(obj, "vision", False) or probe_src is None:
                return ""

            try:
                from vision_pipeline.text_upscale import fit_for_vlm
                small = fit_for_vlm(probe_src.convert("RGB"))
            except Exception:
                small = probe_src

            self._log(
                "    🔤 [SCRIPT PROBE] 문서 전체를 1회 판독해 표기 문자체계를 "
                "확정합니다. (언어 판별이 실패해 프롬프트에 넣을 스크립트가 "
                "없습니다)"
            )
            try:
                raw = obj.read_raw(small, hint="writing system probe", max_chars=160)
            except Exception as e:
                self._log(f"    ⚠ [SCRIPT PROBE] 실패 ({e})")
                return ""

            if not raw:
                self._log("    ⏭ [SCRIPT PROBE] 표본을 얻지 못했습니다.")
                return ""

            try:
                from core.lang_codes import block_census
                census = block_census(raw)
            except Exception:
                census = {}

            if not census:
                return ""

            top = max(census.items(), key=lambda kv: kv[1])[0]
            state["script"] = top
            self._doc_script = top
            brief = " | ".join(
                f"{k}:{v}" for k, v in sorted(census.items(), key=lambda kv: -kv[1])[:4]
            )
            self._log(
                f"    🔤 [SCRIPT PROBE] 확정 '{top}' — 이후 모든 크롭 "
                f"프롬프트에 이 문자체계를 강제합니다. ({brief})"
            )
            return top

        def _fn(category: str, raw_text: str, crop_image, top_field: str = "") -> dict:
            obj = self.crossover.get(SLOT_REFINER)
            if obj is None:
                obj = self.crossover.acquire(SLOT_REFINER)
            if obj is None:
                return {}
            self.refiner = obj

            script = _probe_script(obj)
            local_hint = hint
            if script:
                local_hint = (
                    f"{hint} | detected script: {script}" if hint
                    else f"detected script: {script}"
                )

            spec = specs.get(category)
            if not spec:
                return obj.refine_field(category, raw_text, hint=local_hint)

            out: dict = {}

            if category in array_cats:
                prev = rows_by_cat.setdefault(category, [])
                if getattr(obj, "vision", False) and crop_image is not None:
                    self._log(
                        f"    📤 [{category}] "
                        f"{crop_image.width}x{crop_image.height} 표 크롭 전송"
                    )
                rows = obj.refine_array(
                    category, spec, raw_text,
                    existing=prev, label_bank=labels, hint=local_hint,
                    image=crop_image, primary_hint=top_field, lang_code=lang,
                    script=script,
                )
                prev.extend(rows)
                out["__rows__"] = list(prev)
                got_any = bool(rows)
            else:
                if claimed:
                    self._log(
                        f"    🔒 [ALREADY CLAIMED] 확정값 {len(claimed)}건을 "
                        f"금지 목록으로 전달합니다."
                    )
                if getattr(obj, "vision", False) and crop_image is not None:
                    self._log(
                        f"    📤 [{category}] "
                        f"{crop_image.width}x{crop_image.height} 크롭 이미지 전송"
                    )
                got = obj.refine_category(
                    category, spec, raw_text,
                    claimed=claimed, label_bank=labels, hint=local_hint,
                    image=crop_image, lang_code=lang, script=script,
                )
                for k, v in got.items():
                    claimed.setdefault(k, v)
                out["__fields__"] = got
                got_any = bool(got)

            if not got_any and getattr(obj, "vision", False):
                ok, why = obj.draft_matches_language(raw_text, lang)
                if not ok or not str(raw_text or "").strip():
                    raw = obj.read_raw(
                        crop_image, hint=category,
                        lang_code=lang, script=script,
                    )
                    if raw:
                        out["__raw__"] = raw
                    elif not ok:
                        self._log(f"    ⚠ [RAW READ] 판독 실패 — {why}")

            return out

        return _fn

    def _line_read_fn(self):
        from vision_pipeline.ocr_extract import read_by_lines

        lang = self.lang_code if self.language_resolved else ""
        span = int(os.environ.get("NMS_LINE_SPAN", "1") or 1)
        stride = int(os.environ.get("NMS_LINE_STRIDE", "1") or 1)
        overlap = int(os.environ.get("NMS_LINE_OVERLAP", "1") or 1)

        def _fn(image, bbox, category, text_boxes, ocr, log):
            obj = self.crossover.get(SLOT_REFINER)
            if obj is None:
                obj = self.crossover.acquire(SLOT_REFINER)
            if obj is None:
                return "", [], 0
            self.refiner = obj

            rec = ocr if ocr is not None else self.ocr
            if rec is None:
                rec = self.crossover.get(SLOT_OCR)

            return read_by_lines(
                image, bbox, obj, category,
                lang_code=lang,
                script=self._doc_script,
                text_boxes=text_boxes,
                span=span, stride=stride, overlap=overlap,
                ocr=rec,
                log=log,
            )

        return _fn

    def run_pipeline(
        self,
        schema_json: str = "",
        schema_file: str = "",
        iou_threshold: float = 0.80,
        margin_threshold: float = 0.28,
        use_refiner: bool = True,
    ) -> dict:
        with self.job_slot("파이프라인 실행"):
            self.log_lines = []

            if self.current_image is None and not self.current_text:
                return {"ok": False, "error": "입력이 로드되지 않았습니다."}

            return self._run_pipeline_body(
                schema_json=schema_json,
                schema_file=schema_file,
                iou_threshold=iou_threshold,
                margin_threshold=margin_threshold,
                use_refiner=use_refiner,
            )

    def _run_pipeline_body(
        self,
        schema_json: str = "",
        schema_file: str = "",
        iou_threshold: float = 0.80,
        margin_threshold: float = 0.28,
        use_refiner: bool = True,
    ) -> dict:

        ready = self.ensure_models_ready()
        if not ready.get("ok"):
            return {
                "ok": False,
                "error": ready.get("error", "모델 자동 로드에 실패했습니다."),
                "log": self.log_lines,
                "models_ready": self.models_ready,
            }

        schema: dict = {}
        if schema_json:
            try:
                schema = json.loads(schema_json)
            except Exception as e:
                return {"ok": False, "error": f"스키마 JSON 파싱 실패: {e}"}
        elif schema_file:
            loaded = self.load_schema(schema_file)
            if not loaded.get("ok"):
                return loaded
            schema = loaded["schema"]
        else:
            cls = self._classify_document_body(skip_ready=True)
            if not cls.get("ok"):
                return cls
            loaded = self.load_schema(cls["schema"])
            if not loaded.get("ok"):
                return loaded
            schema = loaded["schema"]
            self._log(f"  📋 스키마 자동 선택: {cls['schema']}")

        if self.current_text:
            return self._run_text(schema, margin_threshold)

        if self.current_image is None:
            return {"ok": False, "error": "입력이 로드되지 않았습니다."}

        return self._run_vision(
            schema, iou_threshold, margin_threshold, use_refiner=use_refiner
        )

    def _run_text(self, schema: dict, margin_threshold: float) -> dict:
        self._log("═══ 텍스트 파이프라인 ═══")
        self._progress(10, "텍스트 파이프라인 시작")

        if not self.ensure_step("text_pipeline", required=True):
            return {"ok": False, "error": "텍스트 파이프라인에 필요한 모델이 없습니다."}

        self.crossover.enter_embedding_phase()

        cfg = TextPipelineConfig(
            margin_threshold=margin_threshold,
            lang_code=self.lang_code if self.language_resolved else "",
        )
        res = run_text_pipeline(
            self.current_text, schema, self.embed_fn,
            config=cfg, log=self._log, nlp=self.nlp,
        )

        self.crossover.mark_idle()

        self._progress(100, "완료" if res.ok else "실패")
        payload = res.to_dict()
        payload["mode"] = "text"
        payload["language"] = self.language.to_dict() if self.language else {}
        payload["crossover"] = self.crossover.stats()
        payload["models_ready"] = self.models_ready

        try:
            from vision_pipeline.ocr_extract import record_to_json
            payload["json"] = record_to_json(
                res.record, schema,
                source_path=self.current_path,
                lang_code=self.lang_code if self.language_resolved else "",
            )
            import json as _json
            self._log("═══ 추출 결과 (JSON) ═══")
            for line in _json.dumps(
                payload["json"], ensure_ascii=False, indent=2
            ).splitlines():
                self._log(f"  {line}")
        except Exception as e:
            self._log(f"  ⏭ JSON 요약 생략: {e}")

        self.last_result = payload
        return payload

    def _run_vision(
        self,
        schema: dict,
        iou_threshold: float,
        margin_threshold: float,
        use_refiner: bool,
    ) -> dict:
        if not self.ensure_step("field_heatmap", required=True):
            return {"ok": False, "error": "필드 히트맵에 필요한 모델이 없습니다."}

        self.ensure_step("ocr_extract", required=False)
        if use_refiner:
            if not self.ensure_step("refine", required=False):
                self._log("  ⏭ 정제 LLM 준비 실패 → OCR 원문으로 진행합니다.")
                use_refiner = False
            elif not self._ensure_refiner_slot():
                self._log("  ⏭ 정제 LLM 슬롯을 등록하지 못해 OCR 원문으로 진행합니다.")
                use_refiner = False
            else:
                try:
                    self.crossover.acquire(SLOT_REFINER, protect=[SLOT_OCR])
                except MissingModelError as e:
                    self._log(f"  ⏭ 정제 LLM 적재 불가 — OCR 원문으로 진행합니다.")
                    for ln in str(e).splitlines():
                        self._log(f"     {ln}")
                    self.crossover.slots.pop(SLOT_REFINER, None)
                    use_refiner = False
                except Exception as e:
                    self._log(
                        f"  ⏭ 정제 LLM 적재 실패({type(e).__name__}: {e}) "
                        f"→ OCR 원문으로 진행합니다."
                    )
                    self.crossover.slots.pop(SLOT_REFINER, None)
                    use_refiner = False

        self.crossover.enter_embedding_phase()
        if self.joint is None and self.embedder is None:
            if SLOT_JOINT not in self.crossover.slots and \
                    SLOT_VISION not in self.crossover.slots:
                return {"ok": False, "error": "패치 격자 모델이 로드되지 않았습니다."}

        if self.ocr is None or not getattr(self.ocr, "available", False):
            self._log(
                "  ⚠ PP-OCRv5 rec 를 쓸 수 없어 OCR 초안 없이 "
                "VLM 직접 판독에 의존합니다."
            )

        self._log("═══ 비전 파이프라인 ═══")
        self._progress(10, "비전 파이프라인 시작")

        joint_obj = self.joint
        if joint_obj is None and SLOT_JOINT in self.crossover.slots:
            try:
                joint_obj = self.crossover.acquire(
                    SLOT_JOINT, protect=[SLOT_OCR]
                )
            except Exception as e:
                self._log(f"  ⚠ 조인트 선획득 실패({e}) → A.X-VE 격자로 진행")
                joint_obj = None

        prefer = self.prefer_grid
        if joint_obj is None and prefer == "joint":
            self._log(
                "  🚧 조인트 모델이 메모리에 없어 격자를 A.X-VE 로 되돌립니다."
            )
            self._demote_joint()
            self._register_embed_provider()
            prefer = self.prefer_grid

        for line in memory_mod.report_lines():
            self._log(line)

        if SLOT_OCR in self.crossover.slots:
            rec = self.ocr
            if rec is None:
                try:
                    rec = self.crossover.acquire(SLOT_OCR)
                except Exception as e:
                    self._log(f"  ⚠ 전용 인식기 획득 실패({e})")
                    rec = None

            if rec is not None and getattr(rec, "available", False):
                self._log(
                    f"  🔤 행 판독에 전용 인식기를 함께 씁니다 — "
                    f"{getattr(rec, 'label', 'PP-OCRv5 rec')} "
                    f"(백엔드 {getattr(rec, 'backend', '-')})"
                )
            else:
                self._log(
                    "  ⚠ 전용 인식기가 동작하지 않아 행 판독이 VLM 단독으로 "
                    "진행됩니다."
                )
        else:
            self._log(
                "  ⚠ 전용 인식기 슬롯이 없어 행 판독이 VLM 단독으로 "
                "진행됩니다."
            )

        if not use_refiner:
            self._log(
                "  🛟 정제 LLM 이 없으므로 PP-OCRv5 인식 결과를 그대로 "
                "필드 값으로 씁니다. 구조화 품질은 낮아지지만 판독 "
                "정확도는 유지됩니다."
            )

        line_on = str(os.environ.get("NMS_LINE_READ", "1")).strip().lower() \
            not in ("0", "false", "no", "off")

        cfg = VisionPipelineConfig(
            iou_threshold=iou_threshold,
            margin_threshold=margin_threshold,
            lang_code=self.lang_code if self.language_resolved else "",
            prefer_grid=prefer,
            doc_code=str(schema.get("code") or schema.get("doc_type") or ""),
            source_path=self.current_path,
            line_read=line_on,
            line_span=int(os.environ.get("NMS_LINE_SPAN", "1") or 1),
            line_stride=int(os.environ.get("NMS_LINE_STRIDE", "1") or 1),
            line_overlap=int(os.environ.get("NMS_LINE_OVERLAP", "1") or 1),
        )
        pipeline = VisionPipeline(
            self.vision_embed_fn, ocr=self.ocr, embedder=self.embedder,
            config=cfg, log=self._log, nlp=self.nlp,
            crossover=self.crossover, joint=joint_obj,
            line_read_fn=self._line_read_fn() if use_refiner else None,
        )

        if self.language_resolved:
            hint = f"document language: {language_name(self.lang_code)}"
        else:
            hint = ""
            self._log(
                "  🚧 언어가 확정되지 않아 'document language' 힌트를 "
                "비웁니다. 잘못된 언어를 넣으면 VLM 이 원문을 로마자로 "
                "음차합니다."
            )

        res = pipeline.run(
            self.current_image, schema,
            refine_fn=self._refine_fn(hint, schema) if use_refiner else None,
        )

        self._progress(100, "완료" if res.ok else "실패")
        payload = res.to_dict(include_images=True)
        payload["mode"] = "vision"
        payload["language"] = self.language.to_dict() if self.language else {}
        payload["crossover"] = self.crossover.stats()
        payload["models_ready"] = self.models_ready
        payload["schema_code"] = str(
            schema.get("code") or schema.get("doc_type") or ""
        )
        absent = payload.get("absent_fields") or []
        if absent:
            self._log(
                f"  ⚪ 이 문서에 존재하지 않는 필드 {len(absent)}개: "
                + ", ".join(sorted(absent))
            )
        self.last_result = payload

        for line in self.crossover.report_lines():
            self._log(line)
        return payload

    def get_result_json(self, pretty: bool = True) -> dict:
        payload = self.last_result
        if not payload:
            return {"ok": False, "error": "아직 추출 결과가 없습니다."}

        data = payload.get("json")
        if not data:
            data = {"fields": payload.get("record") or {}}

        try:
            text = json.dumps(
                data, ensure_ascii=False, indent=2 if pretty else None
            )
        except Exception as e:
            return {"ok": False, "error": f"직렬화 실패: {e}"}

        return {"ok": True, "json": data, "text": text}

    def print_result_json(self) -> dict:
        res = self.get_result_json(pretty=True)
        if not res.get("ok"):
            self._log(f"  ⚠ {res.get('error')}")
            return res
        self._log("═══ 추출 결과 (JSON) ═══")
        for line in str(res["text"]).splitlines():
            self._log(f"  {line}")
        return res

    def save_results(self, results_json: str = "") -> dict:
        payload = self.last_result
        if results_json:
            try:
                payload = json.loads(results_json)
            except Exception as e:
                return {"ok": False, "error": f"JSON 파싱 실패: {e}"}
        if not payload:
            return {"ok": False, "error": "저장할 결과가 없습니다."}

        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            for item in payload.get("results", []) or []:
                b64 = item.get("crop_base64")
                if not b64:
                    continue
                name = str(item.get("field_name") or item.get("category") or "crop")
                safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
                try:
                    img = Image.open(io.BytesIO(base64.b64decode(b64)))
                    img.save(OUTPUT_DIR / f"{safe}.png")
                except Exception:
                    continue

            slim = dict(payload)
            slim["results"] = [
                {k: v for k, v in item.items() if k != "crop_base64"}
                for item in (payload.get("results") or [])
            ]

            with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
                json.dump(slim, f, ensure_ascii=False, indent=2)

            record = payload.get("record") or {}
            with open(OUTPUT_DIR / "record.json", "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)

            extracted = payload.get("json") or {"fields": record}
            with open(OUTPUT_DIR / "extracted.json", "w", encoding="utf-8") as f:
                json.dump(extracted, f, ensure_ascii=False, indent=2)

            indexed = self.index_to_zvec(payload)

            self._log(f"💾 결과 저장 완료: {OUTPUT_DIR}")
            return {
                "ok": True,
                "output_dir": str(OUTPUT_DIR),
                "zvec": indexed,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def index_to_zvec(self, payload: dict) -> dict:
        from core.zvec import CATALOG

        results = payload.get("results") or []
        record = payload.get("record") or {}
        shaped = payload.get("json") or {}
        doc_type = str(
            shaped.get("doc_type")
            or (payload.get("doc_type") or {}).get("code")
            or ""
        )
        doc_id = str(
            shaped.get("id") or self.current_path or f"doc:{doc_type}"
        )

        texts: List[str] = []
        rows: List[tuple] = []

        for item in results:
            value = str(item.get("value") or "").strip()
            if not value:
                continue
            field = str(item.get("field_name") or item.get("category") or "")
            leaf = f"{field.replace('_', ' ')} {value}".strip()
            texts.append(leaf)
            rows.append((f"{doc_id}#{field}", {
                "kind": "field",
                "doc_id": doc_id,
                "doc_type": doc_type,
                "lang": self.lang_code,
                "field": field,
                "value": value,
                "bbox": item.get("bbox") or [],
                "score": item.get("score", 0.0),
            }))

        summary = str(shaped.get("text") or "").strip()
        if not summary:
            summary = " ".join(
                f"{k.replace('_', ' ')} {v}"
                for k, v in record.items()
                if isinstance(v, str) and v.strip()
            ).strip()
        if summary:
            texts.append(summary[:2000])
            rows.append((doc_id, {
                "kind": "doc",
                "doc_id": doc_id,
                "doc_type": doc_type,
                "lang": self.lang_code,
                "digest": shaped.get("digest", ""),
                "cc": shaped.get("cc", ""),
                "bcc": shaped.get("bcc", ""),
                "ref": shaped.get("ref", ""),
                "index": shaped.get("index", 0),
                "no": shaped.get("no", ""),
                "record": shaped or record,
            }))

        if not texts:
            self._log("  ⏭ [zvec] 적재할 값이 없습니다.")
            return {"fields": 0, "docs": 0}

        try:
            mat = self.embed_fn(texts)
        except Exception as e:
            self._log(f"  ⚠ [zvec] 임베딩 실패: {e}")
            return {"fields": 0, "docs": 0}

        if mat is None:
            self._log("  ⚠ [zvec] 임베딩이 비어 적재를 건너뜁니다.")
            return {"fields": 0, "docs": 0}

        mat = np.asarray(mat, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)

        dim = int(mat.shape[-1])
        recipe = f"{self.router.active or 'router'}-{'+'.join(self.active_codes)}"
        field_store = CATALOG.store("fields", dim, recipe)
        doc_store = CATALOG.store("docs", dim, recipe)

        f_items, d_items = [], []
        for i, (rid, meta) in enumerate(rows):
            if i >= mat.shape[0]:
                break
            if meta.get("kind") == "doc":
                d_items.append((rid, mat[i], meta))
            else:
                f_items.append((rid, mat[i], meta))

        n_f = field_store.upsert_many(f_items)
        n_d = doc_store.upsert_many(d_items)

        self._log(
            f"  💽 [zvec] 적재 — 필드 {n_f}건 / 문서 {n_d}건 "
            f"(dim={dim}, recipe={recipe})"
        )
        for line in (field_store.stats(), doc_store.stats()):
            self._log(
                f"     · {line['namespace']:<8} {line['entries']}건 "
                f"/ 묘비 {line['tombstones']} / {line['bytes'] / 1e6:.2f} MB"
            )
        return {"fields": n_f, "docs": n_d}

    def zvec_search(self, query: str, top_k: int = 10, namespace: str = "fields") -> dict:
        from core.zvec import CATALOG

        q = str(query or "").strip()
        if not q:
            return {"ok": False, "error": "질의가 비었습니다."}

        try:
            mat = self.embed_fn([q])
        except Exception as e:
            return {"ok": False, "error": f"임베딩 실패: {e}"}
        if mat is None:
            return {"ok": False, "error": "임베딩이 비었습니다."}

        mat = np.asarray(mat, dtype=np.float32)
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)

        dim = int(mat.shape[-1])
        recipe = f"{self.router.active or 'router'}-{'+'.join(self.active_codes)}"
        store = CATALOG.store(namespace, dim, recipe)
        hits = store.search(mat[0], top_k=int(top_k))

        return {
            "ok": True,
            "namespace": namespace,
            "hits": [
                {"id": rid, "score": round(score, 6), "payload": meta}
                for rid, score, meta in hits
            ],
        }

    def zvec_stats(self) -> dict:
        from core.zvec import CATALOG
        return {"ok": True, "stores": CATALOG.stats()}

    def zvec_purge(self) -> dict:
        from core.zvec import CATALOG
        CATALOG.purge_all()
        self.cached_router = None
        self.cached_vision = None
        self._log("  🗑 [zvec] 전 네임스페이스를 비웠습니다.")
        return {"ok": True}

    def toggle_devtools(self) -> dict:
        errors = []

        native = devtools_mod.try_native_devtools()
        if native.get("ok"):
            return native
        if native.get("error"):
            errors.append(native["error"])

        port = int(os.environ.get("NMS_DEVTOOLS_PORT", devtools_mod.DEFAULT_PORT))
        if devtools_mod.is_port_open(port):
            res = devtools_mod.open_in_browser(port)
            if res.get("ok"):
                self._log(f"  🛠 원격 DevTools 열기: {res['url']}")
                return {"ok": True, "mode": "remote", "url": res["url"]}
            errors.append(res.get("error", ""))
        else:
            errors.append(f"원격 디버깅 포트 {port} 가 닫혀 있습니다.")

        st = devtools_mod.status(port)
        return {
            "ok": False,
            "mode": "inapp",
            "fallback": True,
            "status": st,
            "error": "\n".join(e for e in errors if e),
            "hint": st.get("hint", ""),
        }

    def devtools_status(self) -> dict:
        port = int(os.environ.get("NMS_DEVTOOLS_PORT", devtools_mod.DEFAULT_PORT))
        return devtools_mod.status(port)

    def debug_snapshot(self) -> dict:
        import platform

        try:
            import torch
            torch_ver = torch.__version__
            cuda = torch.cuda.is_available()
            cuda_ver = getattr(torch.version, "cuda", "") or ""
        except Exception:
            torch_ver, cuda, cuda_ver = "", False, ""

        try:
            import transformers
            tf_ver = transformers.__version__
        except Exception:
            tf_ver = ""

        try:
            import numpy
            np_ver = numpy.__version__
        except Exception:
            np_ver = ""

        return {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch_ver,
            "cuda": cuda,
            "cuda_version": cuda_ver,
            "transformers": tf_ver,
            "numpy": np_ver,
            "gpu": self.get_gpu_info(),
            "crossover": self.crossover.stats(),
            "vram_budget": self.vram_budget,
            "low_vram": self.low_vram,
            "language": self.language.to_dict() if self.language else {},
            "lang_code": self.lang_code,
            "saved_lang_code": self.saved_lang_code,
            "providers": self.router.providers(),
            "vision_providers": self.vision_router.providers(),
            "vision_cache": self.cached_vision.stats() if self.cached_vision else {},
            "ocr": {
                "code": self._ocr_code,
                "slug": paddle_ocr_slug(self._ocr_code or DEFAULT_LANGUAGE),
                "script_hint": paddle_ocr_script_hint(
                    self._ocr_code or DEFAULT_LANGUAGE
                ),
                "backend": getattr(self.ocr, "backend", ""),
                "available": bool(getattr(self.ocr, "available", False)),
                "charset": int(getattr(self.ocr, "charset_size", 0) or 0),
                "mean_score": float(getattr(self.ocr, "mean_score", 0.0) or 0.0),
                "det_ready": ppocr_det_ready(),
                "det_backend": getattr(
                    getattr(self.ocr, "detector", None), "backend", ""
                ),
            },
            "doc_script": self._doc_script,
            "paddle": paddle_bootstrap.status(),
            "ram": memory_mod.ram_info(),
            "ram_usable_gb": round(memory_mod.usable_ram_gb(), 2),
            "joint": {
                "label": self.joint_label,
                "dim": self.joint_dim,
                "registered": self._joint_registered,
                "prefer_grid": self.prefer_grid,
                "loaded": self.joint is not None,
                "space_dim": self.vision_router.space_dim,
                "provider_dims": dict(self.vision_router.dims),
                "text_provider_dims": dict(self.router.dims),
            },
            "devtools": self.devtools_status(),
            "input": {
                "path": self.current_path,
                "image": bool(self.current_image),
                "text_chars": len(self.current_text),
            },
            "log_lines": len(self.log_lines),
        }

    def js_console(self, level: str, message: str) -> dict:
        tag = {
            "error": "🟥 JS",
            "warn": "🟨 JS",
            "info": "🟦 JS",
            "log": "⬜ JS",
        }.get(str(level).lower(), "⬜ JS")
        for line in str(message).splitlines():
            self._log(f"  {tag} {line}")
        return {"ok": True}

    def get_crossover(self) -> dict:
        return self.crossover.stats()

    def unload(self):
        self.crossover.mark_idle()
        self.refiner = None
        self.text_embedder = None
        self.nlp = None
        self.cached_router = None
        self.cached_vision = None
        self._ocr_code = ""
        self._doc_script = ""
        self.base_ready = False
        self.models_ready = False


def run_cli(args) -> int:
    os.environ["NMS_LINE_SPAN"] = str(max(1, int(args.line_span)))
    os.environ["NMS_LINE_OVERLAP"] = str(max(0, int(args.line_overlap)))
    if args.no_line_read:
        os.environ["NMS_LINE_READ"] = "0"
    if args.paddle_device:
        os.environ[paddle_bootstrap.ENV_DEVICE] = str(args.paddle_device)
    if args.paddle_mkldnn:
        os.environ[paddle_bootstrap.ENV_MKLDNN] = "1"
    if args.ram_limit > 0.0:
        os.environ[memory_mod.ENV_RAM_LIMIT] = f"{float(args.ram_limit):.2f}"
    if args.allow_low_ram:
        os.environ[memory_mod.ENV_ALLOW_LOW_RAM] = "1"
    paddle_bootstrap.configure_runtime()

    app = NMSOcrApp(
        crossover_enabled=not args.no_crossover,
        auto_fetch=not args.no_fetch,
        vram_budget=args.vram_budget,
        force_low_vram=args.low_vram,
        paddle_autoinstall=not args.no_paddle_install,
    )
    app.print_model_report()
    app._log_vram_profile()

    if args.fetch_all:
        if args.lang:
            app.lang_code = normalize_lang_code(args.lang)
            app.language_resolved = True
            app.active_codes = [app.lang_code]
        app._sync_registry_language()
        app.download_all_missing()
        while app.registry.busy:
            time.sleep(0.5)
        time.sleep(1.0)
        app.print_model_report()
        return 0

    if args.check_only:
        return 0

    if args.lang:
        app.lang_code = normalize_lang_code(args.lang)
        app.language_resolved = True
        app.active_codes = [app.lang_code]
        app._language_pending = False
        app._sync_registry_language()
        app._log(f"🌐 언어 강제 지정: {app.lang_code} ({language_name(app.lang_code)})")
    else:
        app._log(
            "🌐 언어 미지정 — 기본 언어 "
            + ", ".join(f"{language_name(c)}({c})" for c in BOOTSTRAP_LANGUAGES)
            + " 로 부트스트랩합니다."
        )

    loaded = app.load_base_models()
    if not loaded.get("ok"):
        print(f"\n❌ {loaded.get('error', '')}", file=sys.stderr)
        return 1

    if args.input:
        res = app.load_input(args.input)
        if not res.get("ok"):
            print(f"\n❌ {res.get('error', '')}", file=sys.stderr)
            return 1

    if not args.lang:
        app.detect_language()

    app.load_language_models(fetch=not args.no_fetch)

    if not args.input:
        app.print_model_report()
        return 0

    result = app.run_pipeline(
        schema_file=args.schema or "",
        iou_threshold=args.iou,
        margin_threshold=args.margin,
        use_refiner=not args.no_refine,
    )

    if not result.get("ok"):
        print(f"\n❌ {result.get('error', '')}", file=sys.stderr)
        return 1

    print("\n═══ 추출 결과 ═══")
    print(json.dumps(
        result.get("json") or result.get("record", {}),
        ensure_ascii=False, indent=2,
    ))

    if args.save:
        saved = app.save_results()
        if saved.get("ok"):
            print(f"\n💾 저장: {saved['output_dir']}")

    app.unload()
    return 0


def run_ui(args) -> int:
    dev_info = None
    if not args.no_devtools:
        port = devtools_mod.find_free_port(int(args.devtools_port))
        dev_info = devtools_mod.enable_remote_debugging(port)
        print("=" * 60)
        print("  🛠 원격 디버깅 활성화")
        print(f"     크롬 주소창에 입력:  http://127.0.0.1:{port}")
        print(f"     또는 chrome://inspect 에서 localhost:{port} 추가")
        print("     끄려면: python app.py --no-devtools")
        print("=" * 60)

    try:
        import webview
    except ImportError:
        print("pywebview 가 설치되어 있지 않습니다. `pip install pywebview`", file=sys.stderr)
        return 1

    caps = devtools_mod.webview_capabilities()
    print(
        f"  pywebview {caps.get('version') or '?'} "
        f"| 내장 DevTools {'O' if caps.get('toggle_devtools') else 'X'}"
    )
    if not caps.get("toggle_devtools"):
        print("  ℹ 내장 DevTools 가 없습니다. 원격 디버깅 또는 인앱 패널(F12)을 사용하세요.")
        print('     내장 DevTools 사용: pip install -U "pywebview>=5.0"')

    os.environ["NMS_LINE_SPAN"] = str(max(1, int(args.line_span)))
    os.environ["NMS_LINE_OVERLAP"] = str(max(0, int(args.line_overlap)))
    if args.no_line_read:
        os.environ["NMS_LINE_READ"] = "0"
    if args.paddle_device:
        os.environ[paddle_bootstrap.ENV_DEVICE] = str(args.paddle_device)
    if args.paddle_mkldnn:
        os.environ[paddle_bootstrap.ENV_MKLDNN] = "1"
    if args.ram_limit > 0.0:
        os.environ[memory_mod.ENV_RAM_LIMIT] = f"{float(args.ram_limit):.2f}"
    if args.allow_low_ram:
        os.environ[memory_mod.ENV_ALLOW_LOW_RAM] = "1"
    paddle_bootstrap.configure_runtime()

    app = NMSOcrApp(
        crossover_enabled=not args.no_crossover,
        auto_fetch=not args.no_fetch,
        vram_budget=args.vram_budget,
        force_low_vram=args.low_vram,
        paddle_autoinstall=not args.no_paddle_install,
    )
    if dev_info:
        app._log(f"🛠 원격 DevTools: {dev_info['url']} ({dev_info['platform']})")
    app._log(
        f"🖥 pywebview {caps.get('version') or '?'} "
        f"| 내장 DevTools {'사용 가능' if caps.get('toggle_devtools') else '없음 → 인앱 패널'}"
    )

    class JSApi:
        def check_models(self):
            return app.check_models()

        def get_gpu_info(self):
            return app.get_gpu_info()

        def load_models(self):
            with app.job_slot("모델 로드"):
                return app.load_models(fetch=not args.no_fetch)

        def load_image(self, path: str = ""):
            with app.job_slot("입력 로드"):
                return app.load_input(path)

        def load_input(self, path: str = ""):
            with app.job_slot("입력 로드"):
                return app.load_input(path)

        def detect_language(self):
            with app.job_slot("언어 판별"):
                v = app.detect_language()
                return {"ok": True, "language": v.to_dict()}

        def classify_document(self):
            with app.job_slot("문서 유형 분류"):
                return app.classify_document()

        def list_schemas(self):
            return app.list_schemas()

        def load_schema(self, filename: str):
            return app.load_schema(filename)

        def run_pipeline(
            self,
            schema_json: str = "",
            iou_threshold: float = 0.80,
            margin_threshold: float = 0.28,
            text_embed_mode: str = "",
            schema_file: str = "",
            use_refiner: bool = True,
        ):
            return app.run_pipeline(
                schema_json=schema_json,
                schema_file=schema_file,
                iou_threshold=float(iou_threshold),
                margin_threshold=float(margin_threshold),
                use_refiner=bool(use_refiner),
            )

        def save_results(self, results_json: str = ""):
            with app.job_slot("결과 저장"):
                return app.save_results(results_json)

        def get_result_json(self, pretty: bool = True):
            return app.get_result_json(bool(pretty))

        def print_result_json(self):
            return app.print_result_json()

        def job_stats(self):
            return app.job_stats()

        def zvec_search(self, query: str = "", top_k: int = 10, namespace: str = "fields"):
            with app.job_slot("벡터 검색"):
                return app.zvec_search(query, int(top_k), namespace)

        def zvec_stats(self):
            return app.zvec_stats()

        def zvec_purge(self):
            with app.job_slot("벡터 저장소 비우기"):
                return app.zvec_purge()

        def toggle_devtools(self):
            return app.toggle_devtools()

        def get_crossover(self):
            return app.get_crossover()

        def unload_models(self):
            with app.job_slot("모델 반납"):
                app.unload()
                return {"ok": True, "crossover": app.get_crossover()}

        def devtools_status(self):
            return app.devtools_status()

        def paddle_status(self):
            return paddle_bootstrap.status()

        def ram_status(self):
            info = memory_mod.ram_info()
            info["usable_gb"] = round(memory_mod.usable_ram_gb(), 2)
            return info

        def reclaim_memory(self):
            with app.job_slot("메모리 회수"):
                after = memory_mod.reclaim(log=app._log, label="수동 회수")
                return {"ok": True, "usable_gb": round(after, 2)}

        def install_paddle(self):
            with app.job_slot("PaddleOCR 백엔드 설치"):
                return paddle_bootstrap.ensure_paddle(
                    log=app._log, auto=True, force=True
                )

        def debug_snapshot(self):
            return app.debug_snapshot()

        def js_console(self, level: str = "log", message: str = ""):
            return app.js_console(level, message)

        def download_model(self, target_id: str):
            return app.download_model(target_id)

        def download_all_missing(self):
            return app.download_all_missing()

        def download_bootstrap(self):
            return app.download_bootstrap()

        def delete_model(self, target_id: str):
            return app.delete_model(target_id)

        def delete_all_models(self):
            return app.delete_all_models()

        def cancel_download(self):
            return app.cancel_download()

        def set_auto_fetch(self, enabled: bool):
            return app.set_auto_fetch(bool(enabled))

    webview.create_window(
        title="NMS-OCR — Vision / Text NMS Extraction",
        url=str(UI_DIR / "index.html"),
        js_api=JSApi(),
        width=1280,
        height=860,
        min_size=(960, 640),
    )

    start_kwargs = {"debug": bool(args.debug or not args.no_devtools)}
    if args.gui:
        start_kwargs["gui"] = args.gui

    try:
        webview.start(**start_kwargs)
    except TypeError:
        webview.start(debug=start_kwargs["debug"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="app.py",
        description="NMS-OCR — 다국어 문서 구조화 추출 (Vision / Text NMS)",
    )
    p.add_argument("input", nargs="?", default="", help="이미지 / PDF / 텍스트 파일 경로")
    p.add_argument("--schema", default="", help="스키마 파일명 (미지정 시 자동 분류)")
    p.add_argument("--lang", default="", help="언어 코드 강제 지정 (ISO 639-1/3)")
    p.add_argument("--iou", type=float, default=0.80, help="IoU 억제 임계값")
    p.add_argument("--margin", type=float, default=0.28, help="마진 임계값")
    p.add_argument("--save", action="store_true", help="결과를 output/ 에 저장")
    p.add_argument("--no-fetch", action="store_true", help="모델 자동 취득 비활성화")
    p.add_argument("--fetch-all", action="store_true", help="누락 모델 전체 내려받고 종료")
    p.add_argument("--no-refine", action="store_true", help="LLM 정제 추출 비활성화")
    p.add_argument("--check-only", action="store_true", help="모델 상태만 출력하고 종료")
    p.add_argument("--no-ui", action="store_true", help="UI 없이 CLI 로 실행")
    p.add_argument("--debug", action="store_true", help="pywebview 디버그 모드")
    p.add_argument(
        "--no-devtools",
        action="store_true",
        help="크롬 원격 디버깅 비활성화 (기본은 활성)",
    )
    p.add_argument(
        "--devtools-port",
        type=int,
        default=devtools_mod.DEFAULT_PORT,
        help="원격 디버깅 포트 (기본 9222)",
    )
    p.add_argument(
        "--gui",
        default="",
        help="pywebview GUI 백엔드 강제 지정 (edgechromium / qt / gtk / cef)",
    )
    p.add_argument(
        "--vram-budget",
        type=float,
        default=0.0,
        help="VRAM 예산(GB) 강제 지정. 0 이면 자동 감지",
    )
    p.add_argument(
        "--low-vram",
        action="store_true",
        help="저VRAM 모드 강제 (LLM 양자화 / CPU 오프로드)",
    )
    p.add_argument(
        "--no-crossover",
        action="store_true",
        help="페이즈 전환(모델 스위칭) 비활성화 — 전부 상주",
    )
    p.add_argument(
        "--no-paddle-install",
        action="store_true",
        help="PaddleOCR 백엔드 자동 설치 비활성화",
    )
    p.add_argument(
        "--line-span",
        type=int,
        default=1,
        help="행 판독 창 하나에 담을 행 수 (기본 1)",
    )
    p.add_argument(
        "--line-overlap",
        type=int,
        default=1,
        help="창 사이 겹칠 행 수 (기본 1, 0 이면 겹침 없음)",
    )
    p.add_argument(
        "--no-line-read",
        action="store_true",
        help="행 단위 판독 비활성화 — 크롭 전체를 한 번에 읽습니다",
    )
    p.add_argument(
        "--paddle-device",
        default="",
        help="PaddleOCR 실행 장치 (cpu / gpu:0). 기본 cpu",
    )
    p.add_argument(
        "--paddle-mkldnn",
        action="store_true",
        help="PaddleOCR oneDNN 가속 강제 활성화 (기본 비활성)",
    )
    p.add_argument(
        "--ram-limit",
        type=float,
        default=0.0,
        help="사용할 시스템 RAM 상한(GB). 0 이면 자동 감지",
    )
    p.add_argument(
        "--allow-low-ram",
        action="store_true",
        help="RAM 부족 경고를 무시하고 강행 (프로세스 종료 위험)",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.no_ui or args.input or args.check_only:
        return run_cli(args)
    return run_ui(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n중단되었습니다.", file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        sys.exit(1)