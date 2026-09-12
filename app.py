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

from core.crossover import (
    CrossoverSwitch,
    PHASE_EMBEDDING,
    PHASE_GENERATION,
    PHASE_IDLE,
    SLOT_EMBEDDER,
    SLOT_NLP,
    SLOT_OCR,
    SLOT_REFINER,
    SLOT_VISION,
)
from core.device import detect_accelerator, get_vram_info, select_dtype
from core.lang_codes import iso1_of, language_name, normalize_lang_code
from core.language import LanguageDetector, LanguageVerdict
from core.llm import (
    EmbeddingRouter,
    RefinerLLM,
    TextEmbedder,
    resolve_embedder_path,
    resolve_refiner_path,
)
from core.phrase_cache import CachedEmbedder
from core.model_manager import (
    DEFAULT_LANGUAGE,
    LANG_MODEL_KINDS,
    MODELS_ROOT,
    MissingModelError,
    active_language_code,
    check_all_models,
    describe_lang_models,
    describe_stanza,
    ensure_model_dir,
    format_lang_report,
    format_model_report,
    format_stanza_report,
    installed_language_codes,
    lang_models_ready,
    missing_lang_models,
    missing_models,
    save_active_language,
    stanza_ready,
)
from text_pipeline import TextPipelineConfig, run_text_pipeline
from vision_pipeline import VisionPipelineConfig, VisionPipeline

DOCUMENT_EXT = (".pdf",)
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
TEXT_EXT = (".txt", ".md", ".csv", ".json")


class NMSOcrApp:
    def __init__(
        self,
        log: Optional[Callable[[str], None]] = None,
        crossover_enabled: bool = True,
    ):
        self._external_log = log
        self.log_lines: List[str] = []

        self.crossover = CrossoverSwitch(log=self._log, enabled=crossover_enabled)

        self.text_embedder: Optional[TextEmbedder] = None
        self.refiner: Optional[RefinerLLM] = None
        self.nlp = None

        self.router = EmbeddingRouter(log=self._log)
        self.cached_router: Optional[CachedEmbedder] = None
        self.language: Optional[LanguageVerdict] = None
        self.lang_code: str = active_language_code(DEFAULT_LANGUAGE)

        self.current_image: Optional[Image.Image] = None
        self.current_path: str = ""
        self.current_text: str = ""

        self.last_result: Optional[dict] = None
        self._download_lock = threading.Lock()

    @property
    def embedder(self):
        return self.crossover.get(SLOT_VISION)

    @property
    def ocr(self):
        return self.crossover.get(SLOT_OCR)

    def embed_fn(self, texts):
        if self.cached_router is not None:
            return self.cached_router(texts)
        return self.router(texts)

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
        self._push_ui(f"onModelDownload({json.dumps(info)})")
        status = info.get("status", "")
        if status in ("repo_start", "repo_done", "repo_error", "lang_start",
                      "lang_done", "lang_error", "cancelled", "skip"):
            self._log(f"  📦 {info.get('label', '')}")

    def get_gpu_info(self) -> dict:
        device, accel = detect_accelerator()
        return {
            "device": str(device),
            "accel_label": accel,
            "dtype": str(select_dtype(device)),
            "vram": get_vram_info(),
        }

    def check_models(self) -> dict:
        base = check_all_models()
        out = {
            "base": base,
            "language": {},
            "stanza": {},
            "lang_code": self.lang_code,
            "lang_name": language_name(self.lang_code) if self.lang_code else "",
            "crossover": self.crossover.stats(),
        }
        if self.lang_code:
            out["language"] = describe_lang_models(self.lang_code)
            out["stanza"] = describe_stanza(iso1_of(self.lang_code))
        return out

    def print_model_report(self):
        self._log("═══ 모델 상태 ═══")
        for line in format_model_report().splitlines():
            self._log(line)
        if self.lang_code:
            for line in format_lang_report(self.lang_code).splitlines():
                self._log(line)
            for line in format_stanza_report(iso1_of(self.lang_code)).splitlines():
                self._log(line)

    def _register_slots(self):
        def _load_vision():
            from core.embedding import AXVEEmbedder
            return AXVEEmbedder(str(ensure_model_dir("ax-ve")))

        def _load_ocr():
            from core.ocr import HayaiOCR
            return HayaiOCR(str(ensure_model_dir("hayai")))

        self.crossover.register(SLOT_VISION, _load_vision)
        self.crossover.register(SLOT_OCR, _load_ocr)

    def _register_embed_provider(self):
        def _hayai_embed(texts):
            ocr = self.crossover.get(SLOT_OCR)
            if ocr is None:
                ocr = self.crossover.acquire(SLOT_OCR)
            if ocr is None:
                return None
            return ocr.embed_text_batch(texts)

        self.router.register("hayai", _hayai_embed, priority=10)

    def load_base_models(self) -> dict:
        missing = missing_models()
        if missing:
            self._log("❌ 필수 모델이 준비되지 않았습니다:")
            for line in format_model_report().splitlines():
                self._log(line)
            return {
                "ok": False,
                "error": (
                    f"모델 누락: {', '.join(missing)}\n"
                    f"models/ 하위에 직접 배치해 주세요. (자동 다운로드 없음)"
                ),
            }

        self._register_slots()

        try:
            self._log("🔄 A.X-VE 비전 인코더 로드 중...")
            self._progress(10, "A.X-VE 로드 중")
            self.crossover.acquire(SLOT_VISION)
            self._progress(30, "A.X-VE 완료")
        except Exception as e:
            self._log(f"❌ A.X-VE 로드 실패: {e}")
            return {"ok": False, "error": str(e)}

        try:
            self._log("🔄 Hayai OCR 로드 중...")
            self._progress(45, "Hayai OCR 로드 중")
            self.crossover.acquire(SLOT_OCR)
            self._progress(60, "Hayai OCR 완료")
        except Exception as e:
            self._log(f"❌ Hayai OCR 로드 실패: {e}")
            return {"ok": False, "error": str(e)}

        self._register_embed_provider()
        self.crossover.phase = PHASE_EMBEDDING
        self._progress(65, "기본 모델 로드 완료")
        return {"ok": True}

    def detect_language(self, image: Optional[Image.Image] = None) -> LanguageVerdict:
        img = image if image is not None else self.current_image

        self._log("═══ 언어 판별 ═══")
        if self.crossover.get(SLOT_OCR) is None:
            self.crossover.acquire(SLOT_OCR)
        detector = LanguageDetector(ocr=self.ocr, log=self._log, nlp=self.nlp)

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
        self.lang_code = verdict.code
        save_active_language(verdict.code, {"source": "detector", "stage": verdict.stage})
        self._log(
            f"  🌐 확정 언어: {verdict.code} ({verdict.name}) "
            f"| script={verdict.script} | margin={verdict.margin:+.4f}"
        )
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

    def load_language_models(self, fetch: bool = True) -> dict:
        code = self.lang_code or DEFAULT_LANGUAGE

        self._log("═══ 언어 스코프 모델 준비 ═══")
        self.ensure_language_models(code, fetch=fetch)
        code = self.lang_code

        emb_path, emb_label = resolve_embedder_path(code)
        if emb_path:
            def _load_embedder():
                return TextEmbedder(emb_path, label=emb_label, log=self._log)

            def _embed(texts):
                obj = self.crossover.get(SLOT_EMBEDDER)
                if obj is None:
                    obj = self.crossover.acquire(SLOT_EMBEDDER)
                if obj is None:
                    return None
                self.text_embedder = obj
                return obj.encode(texts)

            self.crossover.register(SLOT_EMBEDDER, _load_embedder, label=emb_label)
            self.router.register(emb_label, _embed, priority=50)

            try:
                self._progress(75, "텍스트 임베딩 로드 중")
                self.text_embedder = self.crossover.acquire(SLOT_EMBEDDER)
            except Exception as e:
                self._log(f"  ⚠ 텍스트 임베딩 로드 실패({emb_label}): {e}")
        else:
            self._log("  ⏭ 텍스트 임베딩 모델을 찾지 못해 Hayai 임베딩만 사용합니다.")

        ref_path, ref_label = resolve_refiner_path(code)
        if ref_path:
            def _load_refiner():
                return RefinerLLM(ref_path, label=ref_label, log=self._log)

            self.crossover.register(SLOT_REFINER, _load_refiner, label=ref_label)
            self._log(f"  ⏳ [{ref_label}] 정제 LLM 은 생성 페이즈에서 지연 로드합니다.")
        else:
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
        recipe = f"{self.router.active or 'router'}-{code}"
        self.cached_router = CachedEmbedder(
            self.router, recipe=recipe, dim=dim, log=self._log
        )
        self._log(f"  💾 앵커 캐시 활성 (recipe={recipe}, dim={dim})")

        self._progress(100, "모든 모델 준비 완료")
        self._log(f"  🔀 임베딩 제공자: {', '.join(self.router.providers())}")
        for line in self.crossover.report_lines():
            self._log(line)
        return {"ok": True, "code": code, "providers": self.router.providers()}

    def load_models(self, fetch: bool = True) -> dict:
        base = self.load_base_models()
        if not base.get("ok"):
            return base
        if self.current_image is not None or self.current_text:
            self.detect_language()
        return self.load_language_models(fetch=fetch)

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
        for f in sorted(SCHEMA_DIR.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                out.append({
                    "filename": f.name,
                    "domain": data.get("domain", "unknown"),
                    "doc_type": data.get("doc_type", "unknown"),
                    "field_count": len(data.get("fields", {}) or {}),
                })
            except Exception:
                continue
        return {"ok": True, "schemas": out}

    def load_schema(self, filename: str) -> dict:
        p = SCHEMA_DIR / filename
        if not p.exists():
            return {"ok": False, "error": f"스키마 없음: {filename}"}
        try:
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

    def classify_document(self) -> dict:
        if self.current_image is None:
            return {"ok": False, "error": "이미지가 로드되지 않았습니다."}

        self.crossover.enter_embedding_phase()
        if self.ocr is None:
            return {"ok": False, "error": "모델이 로드되지 않았습니다."}

        self._log("═══ 문서 유형 분류 (Doc Type NMS) ═══")
        pipeline = VisionPipeline(
            self.embed_fn, ocr=self.ocr, embedder=self.embedder,
            log=self._log, nlp=self.nlp,
        )
        verdict = pipeline.classify(self.current_image, SCHEMA_DIR)

        if not verdict.code:
            return {"ok": False, "error": "문서 유형을 판정하지 못했습니다."}

        return {
            "ok": True,
            "schema": verdict.schema_file,
            "verdict": verdict.to_dict(),
        }

    def _refine_fn(self, hint: str = ""):
        if SLOT_REFINER not in self.crossover.slots:
            return None

        def _fn(field_name: str, raw_text: str, crop_image) -> dict:
            obj = self.crossover.get(SLOT_REFINER)
            if obj is None:
                obj = self.crossover.acquire(SLOT_REFINER)
            if obj is None:
                return {}
            self.refiner = obj
            return obj.refine_field(field_name, raw_text, hint=hint)

        return _fn

    def run_pipeline(
        self,
        schema_json: str = "",
        schema_file: str = "",
        iou_threshold: float = 0.80,
        margin_threshold: float = 0.28,
        use_refiner: bool = True,
    ) -> dict:
        self.log_lines = []

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
            cls = self.classify_document()
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

        self.crossover.enter_embedding_phase()

        cfg = TextPipelineConfig(margin_threshold=margin_threshold)
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
        self.last_result = payload
        return payload

    def _run_vision(
        self,
        schema: dict,
        iou_threshold: float,
        margin_threshold: float,
        use_refiner: bool,
    ) -> dict:
        self.crossover.enter_embedding_phase()
        if self.ocr is None:
            return {"ok": False, "error": "모델이 로드되지 않았습니다."}

        self._log("═══ 비전 파이프라인 ═══")
        self._progress(10, "비전 파이프라인 시작")

        cfg = VisionPipelineConfig(
            iou_threshold=iou_threshold,
            margin_threshold=margin_threshold,
        )
        pipeline = VisionPipeline(
            self.embed_fn, ocr=self.ocr, embedder=self.embedder,
            config=cfg, log=self._log, nlp=self.nlp,
            crossover=self.crossover,
        )

        hint = f"document language: {language_name(self.lang_code)}"
        res = pipeline.run(
            self.current_image, schema,
            refine_fn=self._refine_fn(hint) if use_refiner else None,
        )

        self._progress(100, "완료" if res.ok else "실패")
        payload = res.to_dict(include_images=True)
        payload["mode"] = "vision"
        payload["language"] = self.language.to_dict() if self.language else {}
        payload["crossover"] = self.crossover.stats()
        self.last_result = payload

        for line in self.crossover.report_lines():
            self._log(line)
        return payload

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

            self._log(f"💾 결과 저장 완료: {OUTPUT_DIR}")
            return {"ok": True, "output_dir": str(OUTPUT_DIR)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def toggle_devtools(self) -> dict:
        try:
            import webview
            webview.windows[0].toggle_devtools()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_crossover(self) -> dict:
        return self.crossover.stats()

    def unload(self):
        self.crossover.mark_idle()
        self.refiner = None
        self.text_embedder = None
        self.nlp = None
        self.cached_router = None


def run_cli(args) -> int:
    app = NMSOcrApp(crossover_enabled=not args.no_crossover)
    app.print_model_report()

    if args.check_only:
        return 0

    if args.lang:
        app.lang_code = normalize_lang_code(args.lang)
        app._log(f"🌐 언어 강제 지정: {app.lang_code} ({language_name(app.lang_code)})")

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
    print(json.dumps(result.get("record", {}), ensure_ascii=False, indent=2))

    if args.save:
        saved = app.save_results()
        if saved.get("ok"):
            print(f"\n💾 저장: {saved['output_dir']}")

    app.unload()
    return 0


def run_ui(args) -> int:
    try:
        import webview
    except ImportError:
        print("pywebview 가 설치되어 있지 않습니다. `pip install pywebview`", file=sys.stderr)
        return 1

    app = NMSOcrApp(crossover_enabled=not args.no_crossover)

    class JSApi:
        def check_models(self):
            return app.check_models()

        def get_gpu_info(self):
            return app.get_gpu_info()

        def load_models(self):
            return app.load_models(fetch=not args.no_fetch)

        def load_image(self, path: str = ""):
            return app.load_input(path)

        def load_input(self, path: str = ""):
            return app.load_input(path)

        def detect_language(self):
            v = app.detect_language()
            return {"ok": True, "language": v.to_dict()}

        def classify_document(self):
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
        ):
            return app.run_pipeline(
                schema_json=schema_json,
                iou_threshold=float(iou_threshold),
                margin_threshold=float(margin_threshold),
            )

        def save_results(self, results_json: str = ""):
            return app.save_results(results_json)

        def toggle_devtools(self):
            return app.toggle_devtools()

        def get_crossover(self):
            return app.get_crossover()

        def unload_models(self):
            app.unload()
            return {"ok": True, "crossover": app.get_crossover()}

    webview.create_window(
        title="NMS-OCR — Vision / Text NMS Extraction",
        url=str(UI_DIR / "index.html"),
        js_api=JSApi(),
        width=1280,
        height=860,
        min_size=(960, 640),
    )
    webview.start(debug=bool(args.debug))
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
    p.add_argument("--no-fetch", action="store_true", help="언어 모델 자동 취득 비활성화")
    p.add_argument("--no-refine", action="store_true", help="LLM 정제 추출 비활성화")
    p.add_argument("--check-only", action="store_true", help="모델 상태만 출력하고 종료")
    p.add_argument("--no-ui", action="store_true", help="UI 없이 CLI 로 실행")
    p.add_argument("--debug", action="store_true", help="UI DevTools 활성화")
    p.add_argument(
        "--no-crossover",
        action="store_true",
        help="페이즈 전환(모델 스위칭) 비활성화 — 전부 상주",
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