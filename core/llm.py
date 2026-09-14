from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .device import configure_backends, detect_accelerator, select_dtype
from .memory import (
    RamGuardAbort,
    RamWatchdog,
    can_stage,
    commit_free_gb,
    cpu_share_gb,
    headroom_gb,
    make_room,
    model_disk_gb,
    reclaim,
    staging_need_gb,
    usable_ram_gb,
)
from .quant_bootstrap import (
    build_config as build_quant_config,
    capability as quant_capability,
    stage_ratio as quant_stage_ratio,
)
from .model_manager import (
    BOOTSTRAP_LANGUAGES,
    LLM_PATH,
    MissingModelError,
    embedder_ready_codes,
    ensure_lang_model_dir,
    ensure_model_dir,
    lang_model_dir,
    lang_model_ready,
)


import re

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
TYPE_MARKER_RE = re.compile(r"\{(String|Number|Boolean|Array)\}", re.IGNORECASE)
FIELD_ECHO_RE = re.compile(r"^(field|value|val|text|key|item|raw)([ _\-]?\d+)?$")

SCHEMA_ECHO_TOKENS = frozenset({
    "", "-", "--", "...", "n/a", "na", "null", "none", "undefined", "unknown",
    "string", "number", "boolean", "array", "object", "schema", "region",
    "hint", "spec", "format", "example", "placeholder", "todo", "tbd",
    "value", "val", "field", "text", "key", "item", "raw",
    "cleaned value", "empty string", "cleaned value or empty string",
    "copy from ocr text or empty string",
})

SPEC_ECHO_RATIO = 0.60
SPEC_ECHO_MIN_TOKENS = 2

INDEX_KEY_RE = re.compile(r"^(.*?)[ _\-]?(\d{1,2})$")
BRACELESS_PAIR_RE = re.compile(r'"[A-Za-z_][A-Za-z0-9_ \-]*"\s*:')
KEY_SIM_FLOOR = 0.72

VISION_PLACEHOLDERS = (
    ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>"),
    ("<|vision_start|>", "<|video_pad|>", "<|vision_end|>"),
    ("<image>", "", ""),
)

SCRIPT_OF_LANG: Dict[str, tuple] = {
    "kor": ("Hangul", "Korean", "한국어 예시: 안녕하세요"),
    "jpn": ("Kana and Kanji", "Japanese", "日本語の例: こんにちは"),
    "zho": ("Han", "Chinese", "中文示例: 你好"),
    "rus": ("Cyrillic", "Russian", "Пример: Привет"),
    "ukr": ("Cyrillic", "Ukrainian", "Приклад: Привіт"),
    "ara": ("Arabic", "Arabic", "مثال: مرحبا"),
    "tha": ("Thai", "Thai", "ตัวอย่าง: สวัสดี"),
    "hin": ("Devanagari", "Hindi", "उदाहरण: नमस्ते"),
    "ell": ("Greek", "Greek", "Παράδειγμα: Γειά"),
    "heb": ("Hebrew", "Hebrew", "דוגמה: שלום"),
    "tam": ("Tamil", "Tamil", "எடுத்துக்காட்டு: வணக்கம்"),
    "tel": ("Telugu", "Telugu", "ఉదాహరణ: నమస్కారం"),
    "eng": ("Latin", "English", ""),
}

SCRIPT_BY_BLOCK: Dict[str, tuple] = {
    "Hangul": ("Hangul", "Korean", "한국어 예시: 안녕하세요"),
    "Kana": ("Kana and Kanji", "Japanese", "日本語の例: こんにちは"),
    "Han": ("Han", "Chinese", "中文示例: 你好"),
    "Cyrillic": ("Cyrillic", "Russian", "Пример: Привет"),
    "Arabic": ("Arabic", "Arabic", "مثال: مرحبا"),
    "Thai": ("Thai", "Thai", "ตัวอย่าง: สวัสดี"),
    "Devanagari": ("Devanagari", "Hindi", "उदाहरण: नमस्ते"),
    "Greek": ("Greek", "Greek", "Παράδειγμα: Γειά"),
    "Hebrew": ("Hebrew", "Hebrew", "דוגמה: שלום"),
    "Tamil": ("Tamil", "Tamil", "எடுத்துக்காட்டு: வணக்கம்"),
    "Telugu": ("Telugu", "Telugu", "ఉదాహరణ: నమస్కారం"),
    "Latin": ("Latin", "English", ""),
}

NON_LATIN_SCRIPTS = frozenset({
    "Hangul", "Kana", "Han", "Cyrillic", "Arabic", "Thai",
    "Devanagari", "Greek", "Hebrew", "Tamil", "Telugu",
})

ROMANIZE_RATIO = 0.55

PROMPT_ECHO_MARKERS = (
    "script rule",
    "scriptrule",
    "do not romanize",
    "do not translate",
    "do not transliterate",
    "correct output style",
    "writing system",
    "transcribe now",
    "reproduce every character",
    "omit it rather than",
    "is a failure",
    "separate distinct text blocks",
    "context:",
    "region:",
    "schema:",
    "hint:",
)

PROMPT_ECHO_MIN_HITS = 1

REASONING_OPENERS = (
    "the user wants",
    "the user is asking",
    "the user asks",
    "i need to",
    "i should",
    "let me",
    "okay,",
    "first,",
    "looking at the image",
    "we need to",
)

THINK_BUDGET_MULTIPLIER = 3.0
THINK_BUDGET_FLOOR = 512
THINK_BUDGET_CEIL = 3072

NO_THINK_TAG = "/no_think"
JSON_ACTION_OBJECT = "[ACTION] JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON."
JSON_ACTION_ARRAY = "[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON."
JSON_PREFILL_OBJECT = "{"
JSON_PREFILL_ARRAY = "["
JSON_BALANCE_TAIL_TOKENS = 2


VISION_ARCH_MARKERS = (
    "imagetexttotext",
    "vision2seq",
    "conditionalgeneration",
    "vlforconditional",
    "visionencoderdecoder",
)

PLAN_GPU_DISK = "gpu+disk"
PLAN_GPU_CPU_DISK = "gpu+cpu+disk"
PLAN_DISK_HEAVY = "disk-heavy"

LOAD_PLANS = (PLAN_GPU_DISK, PLAN_DISK_HEAVY)
LOAD_PLANS_WITH_CPU = (PLAN_GPU_CPU_DISK, PLAN_GPU_DISK, PLAN_DISK_HEAVY)

PLAN_GPU_RATIO = {
    PLAN_GPU_CPU_DISK: 1.00,
    PLAN_GPU_DISK: 1.00,
    PLAN_DISK_HEAVY: 0.55,
}

PLAN_NOTE = {
    PLAN_GPU_CPU_DISK: (
        "GPU 우선 + CPU 여유분 + 디스크 오프로드 (RAM 여유가 넉넉할 때)"
    ),
    PLAN_GPU_DISK: (
        "GPU 우선 + 디스크 오프로드 (CPU 몫 0 — 호스트 RAM 을 쓰지 않습니다)"
    ),
    PLAN_DISK_HEAVY: (
        "GPU 절반만 + 나머지 전량 디스크 (가장 느리지만 가장 안전)"
    ),
}


def _load_with_dtype(loader, path, dtype, **kwargs):
    try:
        return loader.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return loader.from_pretrained(path, torch_dtype=dtype, **kwargs)


def _peek_config(model_path: str) -> dict:
    import json
    from pathlib import Path as _P

    cfg = _P(model_path) / "config.json"
    if not cfg.exists():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}


def _looks_vision_model(meta: dict) -> Tuple[bool, str]:
    if not isinstance(meta, dict):
        return False, ""

    archs = meta.get("architectures")
    if isinstance(archs, str):
        archs = [archs]
    if isinstance(archs, (list, tuple)):
        for a in archs:
            low = str(a).lower()
            for marker in VISION_ARCH_MARKERS:
                if marker in low:
                    return True, str(a)

    for key in ("vision_config", "image_token_id", "vision_start_token_id"):
        if meta.get(key) is not None:
            return True, f"config.{key}"

    return False, ""


class TextEmbedder:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        label: str = "embedder",
        log=None,
    ):
        from transformers import AutoModel, AutoTokenizer

        self.model_path = str(model_path)
        self.label = label
        self.max_length = int(max_length)
        self._log_fn = log or (lambda m: None)

        self.device, self.accel_label = detect_accelerator(device)
        self.dtype = select_dtype(self.device)
        configure_backends(self.device)

        self.weight_gb = model_disk_gb(self.model_path)
        ok, why = can_stage(self.weight_gb, log=self._log, label=self.label)
        if not ok:
            raise MissingModelError(
                f"[{self.label}] 시스템 메모리가 부족해 로드를 중단했습니다.\n"
                f"  {why}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = _load_with_dtype(
            AutoModel, self.model_path, self.dtype,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()

        self.dim = int(getattr(self.model.config, "hidden_size", 0) or 0)
        self._log(
            f"  ✅ [{self.label}] 임베딩 모델 로드 "
            f"({self.model_path} | dim={self.dim} | {self.device}/{self.dtype})"
        )

        from . import diagnostics
        diagnostics.describe_config(self.model.config, self.label, self._log)
        diagnostics.describe_module(self.model, self.label, self._log)
        diagnostics.probe_space(self.label, self.encode, self._log)

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    @torch.no_grad()
    def encode(self, texts: Sequence[str], l2: bool = True) -> np.ndarray:
        items = [str(t) if t is not None else "" for t in texts]
        if not items:
            return np.zeros((0, max(1, self.dim)), dtype=np.float32)

        enc = self.tokenizer(
            items,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        out = self.model(**enc)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0]

        mask = enc.get("attention_mask")
        if mask is not None:
            m = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = hidden.mean(dim=1)

        mat = pooled.float().cpu().numpy().astype(np.float32)
        if l2:
            norms = np.linalg.norm(mat, axis=-1, keepdims=True)
            mat = mat / np.maximum(norms, 1e-8)
        return mat

    def unload(self):
        self.model = None
        self.tokenizer = None
        reclaim(log=self._log, label=f"{self.label} 반납")


class RefinerLLM:
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        label: str = "refiner",
        log=None,
        low_vram: bool = False,
        budget_gb: float = 0.0,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = str(model_path)
        self.label = label
        self.max_new_tokens = int(max_new_tokens)
        self._log_fn = log or (lambda m: None)
        self.low_vram = bool(low_vram)
        self.budget_gb = float(budget_gb or 0.0)
        self.load_mode = "standard"
        self.load_plan = ""
        self.vision = False
        self.processor = None
        self._vision_marker_logged = False
        self.thinking = False
        self._think_probed = False
        self._no_think_kwarg = None
        self._json_marker_logged = False

        self.device, self.accel_label = detect_accelerator(device)
        self.dtype = select_dtype(self.device)
        configure_backends(self.device)

        self.weight_gb = model_disk_gb(self.model_path)
        self.quant_mode = ""

        quant_ready = False
        if self.low_vram and self.device.type == "cuda":
            mode = quant_capability(log=self._log)
            quant_ready = bool(mode)
            if quant_ready:
                self.quant_mode = mode

        ratio = quant_stage_ratio() if quant_ready else 1.0
        fallback = max(1.5, self.weight_gb * ratio)
        measured, why = staging_need_gb(self.model_path, fallback_gb=fallback)

        if quant_ready:
            self.stage_gb = min(fallback, measured)
            self._log(
                f"  🧮 [{self.label}] {self.quant_mode} 양자화는 텐서 단위로 "
                f"스트리밍되어 전량이 동시에 RAM 에 있지 않습니다. "
                f"스테이징 {self.stage_gb:.1f} GB — {why}"
            )
        else:
            self.stage_gb = measured if measured > 0.0 else self.weight_gb
            self._log(
                f"  🧮 [{self.label}] 스테이징 {self.stage_gb:.1f} GB — {why}"
            )
            if self.low_vram and self.device.type == "cuda":
                self._log(
                    f"  ⚠ [{self.label}] 양자화 없이 원본 {self.weight_gb:.1f} GB 를 "
                    f"GPU 에 펼쳐야 합니다. VRAM 이 부족하면 실패합니다."
                )

        ok, why = can_stage(self.stage_gb, log=self._log, label=self.label)

        if not ok:
            need = self.stage_gb * (1.0 + 0.15) + 1.2
            self._log(
                f"  🪜 [{self.label}] 1차 점검 실패 — 강한 회수를 시도한 뒤 "
                f"다시 판정합니다."
            )
            got, room = make_room(
                need, release_fn=None, log=self._log, label=self.label
            )
            if got:
                ok = True
                why = f"회수 후 가용 {room:.1f} GB ≥ 필요 {need:.1f} GB"

        if not ok:
            from .quant_bootstrap import manual_command as quant_cmd
            tip = (
                f"    · 4bit 양자화를 켜면 스테이징이 약 1/3 로 줄어듭니다.\n"
                f"      {quant_cmd()}\n"
                if not quant_ready else ""
            )
            raise MissingModelError(
                f"[{self.label}] 시스템 메모리가 부족해 로드를 중단했습니다.\n"
                f"  {why}\n"
                f"  가중치 {self.weight_gb:.1f} GB 를 GPU 로 올리려면\n"
                f"  시스템 RAM 에 약 {self.stage_gb:.1f} GB 의 작업 공간이\n"
                f"  필요합니다.\n"
                f"  해결 방법:\n"
                f"{tip}"
                f"    · 다른 프로그램을 닫아 RAM 을 확보하세요.\n"
                f"    · Windows 가상 메모리(페이지 파일)를 늘리세요.\n"
                f"    · 더 작은 모델을 쓰세요.\n"
                f"    · 강행하려면 set NMS_ALLOW_LOW_RAM=1"
            )
        self._log(
            f"  📊 [{self.label}] 메모리 사전 점검 통과 — {why} "
            f"| 가중치 {self.weight_gb:.1f} GB / 스테이징 {self.stage_gb:.1f} GB"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        quant = None
        if self.device.type == "cuda" and self.low_vram:
            quant = self._try_quant_config()
            self.load_mode = (
                (self.quant_mode or "4bit") if quant is not None else "offload"
            )

        offload_dir = ""
        if self.device.type == "cuda" and self.low_vram:
            probe = Path(self.model_path).parent / ".offload"
            try:
                probe.mkdir(parents=True, exist_ok=True)
                offload_dir = str(probe)
            except Exception as e:
                self._log(f"  ⏭ 오프로드 폴더 준비 실패 ({e})")

        cpu_room = cpu_share_gb()
        plans = list(
            LOAD_PLANS_WITH_CPU if cpu_room > 0.0 else LOAD_PLANS
        )
        if self.device.type != "cuda" or not self.low_vram:
            plans = [""]

        if len(plans) > 1:
            self._log(
                f"  🪜 [{self.label}] 적재를 {len(plans)}단계로 쪼갭니다 — "
                f"{' → '.join(plans)} (앞 단계가 실패하면 더 얇은 계획으로 "
                f"자동 강등합니다)"
            )
            if cpu_room <= 0.0:
                self._log(
                    f"  🚧 [{self.label}] 커밋 여유 {headroom_gb():.1f} GB 로는 "
                    f"CPU 배치 몫을 둘 수 없습니다. CPU 상한을 0 으로 두고 "
                    f"GPU + 디스크로만 나눕니다 — 이 설정이 프로세스 강제 "
                    f"종료를 막습니다."
                )

        def _build_kwargs(plan: str) -> dict:
            kw = {"trust_remote_code": True, "low_cpu_mem_usage": True}
            if self.device.type != "cuda":
                return kw
            if not self.low_vram:
                kw["device_map"] = "auto"
                return kw

            if quant is not None:
                kw["quantization_config"] = quant
            kw["device_map"] = "auto"

            if self.budget_gb > 0:
                ratio = float(PLAN_GPU_RATIO.get(plan, 1.0))
                cap = max(0.6, (self.budget_gb - 0.9) * ratio)
                limits = {0: f"{cap:.1f}GiB"}
                if plan == PLAN_GPU_CPU_DISK and cpu_room > 0.0:
                    limits["cpu"] = f"{cpu_room:.1f}GiB"
                else:
                    limits["cpu"] = "0GiB"
                kw["max_memory"] = limits
                self._log(
                    f"  🧮 [{self.label}] '{plan}' 메모리 상한 — GPU "
                    f"{cap:.1f} GiB / CPU {limits['cpu']} | "
                    f"{PLAN_NOTE.get(plan, '')}"
                )

            if offload_dir:
                kw["offload_folder"] = offload_dir
                kw["offload_state_dict"] = True
                kw["offload_buffers"] = True
            return kw

        meta = _peek_config(self.model_path)
        want_vision, arch = _looks_vision_model(meta)

        self.model = None
        last_error: Optional[BaseException] = None

        for attempt, plan in enumerate(plans, start=1):
            kwargs = _build_kwargs(plan)
            tag = plan or "standard"

            if attempt > 1:
                room = reclaim(
                    log=self._log,
                    label=f"{self.label} {attempt}단계 전 회수",
                    rounds=3,
                )
                self._log(
                    f"  🔁 [{self.label}] {attempt}단계 '{tag}' 재시도 — "
                    f"커밋 여유 {room:.1f} GB"
                )

            guard = RamWatchdog(label=f"{self.label}/{tag}", log=self._log)
            try:
                with guard:
                    if want_vision:
                        import transformers as _tf
                        for loader_name in (
                            "AutoModelForImageTextToText",
                            "AutoModelForVision2Seq",
                        ):
                            loader = getattr(_tf, loader_name, None)
                            if loader is None:
                                continue
                            try:
                                self.model = _load_with_dtype(
                                    loader, self.model_path, self.dtype,
                                    **kwargs
                                )
                                guard.check()
                                self.vision = True
                                self._log(
                                    f"  👁 [{self.label}] 비전-언어 모델로 "
                                    f"로드했습니다 ({loader_name} | config "
                                    f"{arch}) — 크롭 이미지를 직접 읽습니다."
                                )
                                break
                            except RamGuardAbort:
                                raise
                            except Exception as e:
                                self._log(
                                    f"  ⏭ [{self.label}] {loader_name} 로드 "
                                    f"불가 ({type(e).__name__}: {str(e)[:90]})"
                                )
                                self.model = None
                                reclaim(
                                    log=self._log,
                                    label=f"{self.label} 로더 실패 후",
                                )
                    elif attempt == 1:
                        self._log(
                            f"  📄 [{self.label}] config 에 비전 아키텍처 "
                            f"표시가 없어 텍스트 전용 경로로 바로 로드합니다 "
                            f"— 불필요한 중복 적재를 건너뜁니다."
                        )

                    if self.model is None:
                        self.model = _load_with_dtype(
                            AutoModelForCausalLM, self.model_path,
                            self.dtype, **kwargs
                        )
                        guard.check()
                        self.vision = False
                        self._log(
                            f"  📄 [{self.label}] 텍스트 전용으로 "
                            f"로드했습니다 — OCR 원문만 정제합니다."
                        )

                if self.model is not None:
                    if plan:
                        self.load_plan = plan
                        self._log(
                            f"  ✅ [{self.label}] '{tag}' 계획으로 적재 성공 "
                            f"({attempt}/{len(plans)}단계)"
                        )
                    break

            except RamGuardAbort as e:
                last_error = e
                self.model = None
                self._log(f"  🚨 [{self.label}] {e}")
                reclaim(
                    log=self._log,
                    label=f"{self.label} 중단 후 회수",
                    rounds=3,
                )
            except (MemoryError, OSError) as e:
                last_error = e
                self.model = None
                self._log(
                    f"  ⚠ [{self.label}] '{tag}' 적재 실패 "
                    f"({type(e).__name__}: {str(e)[:110]})"
                )
                reclaim(
                    log=self._log,
                    label=f"{self.label} 실패 후 회수",
                    rounds=3,
                )
            except Exception as e:
                low = str(e).lower()
                if "out of memory" not in low and "alloc" not in low:
                    raise
                last_error = e
                self.model = None
                self._log(
                    f"  ⚠ [{self.label}] '{tag}' 메모리 부족 "
                    f"({type(e).__name__}: {str(e)[:110]})"
                )
                reclaim(
                    log=self._log,
                    label=f"{self.label} 실패 후 회수",
                    rounds=3,
                )

        if self.model is None:
            raise MissingModelError(
                f"[{self.label}] {len(plans)}단계 적재 계획을 모두 시도했지만 "
                f"메모리가 부족합니다.\n"
                f"  마지막 오류: {type(last_error).__name__ if last_error else '?'}"
                f": {str(last_error)[:160]}\n"
                f"  커밋 여유 {commit_free_gb():.1f} GB\n"
                f"  해결 방법:\n"
                f"    · 다른 프로그램을 닫아 RAM 을 확보하세요.\n"
                f"    · Windows 가상 메모리(페이지 파일)를 늘리세요.\n"
                f"    · 더 작은 모델을 쓰세요."
            )

        reclaim(log=self._log, label=f"{self.label} 로드 후")

        if self.vision:
            try:
                from transformers import AutoProcessor
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path, trust_remote_code=True
                )
                self._log(
                    f"  👁 [{self.label}] 프로세서 "
                    f"{type(self.processor).__name__} 준비"
                )
            except Exception as e:
                self._log(
                    f"  ⚠ [{self.label}] 프로세서 로드 실패({e}) "
                    f"→ 텍스트 경로로 되돌립니다."
                )
                self.vision = False
                self.processor = None

        if self.vision and self.processor is not None:
            try:
                from PIL import Image as _Img, ImageDraw as _Draw
                probe = _Img.new("RGB", (448, 224), (255, 255, 255))
                _Draw.Draw(probe).text((20, 90), "ZX7QK", fill=(0, 0, 0))

                enc = self._encode_via_template("Read the text.", probe)
                mode = "template"
                if enc is None:
                    enc, err = self._encode_manual("Read the text.", probe)
                    mode = "manual"
                    if enc is None:
                        raise RuntimeError(f"인코딩 실패: {err}")

                keys = sorted(enc.keys())
                ids = enc.get("input_ids")
                n_tok = int(ids.shape[-1]) if ids is not None else 0
                self._log(
                    f"  🧭 [{self.label}] 비전 인코딩 자가검진 — 경로 {mode} "
                    f"| 키 {keys} | 토큰 {n_tok}"
                )

                got = self.generate_with_image(
                    "Read the printed text and reply with it only.",
                    probe, max_new_tokens=64,
                )

                if self.looks_reasoning(got):
                    self.note_thinking(got)
                    self._log(
                        f"  🔁 [{self.label}] 사고 단계에서 잘린 응답입니다. "
                        f"예산을 {self.budget_for(64)}토큰으로 늘려 "
                        f"재검진합니다."
                    )
                    got = self.generate_with_image(
                        "Read the printed text and reply with it only.",
                        probe, max_new_tokens=64,
                    )

                low = str(got or "").strip().lower()
                bad = low in ("", "user", "assistant", "system")
                if bad:
                    self._log(
                        f"  ⚠ [{self.label}] 비전 경로 응답이 역할 토큰"
                        f"({got[:24]!r}) — 텍스트 경로를 사용합니다."
                    )
                    self.vision = False
                elif self.looks_reasoning(got):
                    self._log(
                        f"  ⚠ [{self.label}] 예산을 늘려도 사고 텍스트만 "
                        f"나옵니다 ({got[:36]!r}). 비전 판독을 끄고 "
                        f"PP-OCRv5 결과를 그대로 씁니다 — 쓸모없는 호출로 "
                        f"수 분을 낭비하지 않습니다."
                    )
                    self.vision = False
                else:
                    self._log(
                        f"  ✅ [{self.label}] 비전 경로 자가검진 통과 "
                        f"(응답 {got[:24]!r})"
                    )
            except Exception as e:
                self._log(
                    f"  ⚠ [{self.label}] 비전 경로 자가검진 실패 "
                    f"({type(e).__name__}: {e}) — 텍스트 경로를 사용합니다."
                )
                from . import diagnostics as _diag
                if _diag.enabled(2):
                    import traceback
                    for line in traceback.format_exc().splitlines()[-10:]:
                        self._log(f"      {line}")
                self.vision = False

        if self.device.type != "cuda" and self.load_mode == "standard":
            self.model.to(self.device)
        self.model.eval()

        placement = ""
        dev_map = getattr(self.model, "hf_device_map", None)
        if isinstance(dev_map, dict):
            tally: Dict[str, int] = {}
            for v in dev_map.values():
                key = str(v)
                tally[key] = tally.get(key, 0) + 1
            placement = " | 배치: " + ", ".join(
                f"{k}×{n}" for k, n in sorted(tally.items())
            )
            if int(tally.get("cpu", 0)) > 0:
                self._log(
                    f"  ⚠ [{self.label}] 레이어 {tally['cpu']}개가 CPU 에 "
                    f"상주합니다. 호스트 RAM 을 계속 점유하므로 다음 "
                    f"적재에서 압박이 커집니다."
                )
            if int(tally.get("disk", 0)) > 0:
                self._log(
                    f"  💽 [{self.label}] 레이어 {tally['disk']}개를 디스크에서 "
                    f"읽습니다. 생성 속도가 느려지지만 RAM 은 쓰지 않습니다."
                )

        plan_tail = f" | 계획 {self.load_plan}" if self.load_plan else ""
        self._log(
            f"  ✅ [{self.label}] 정제 LLM 로드 "
            f"({self.load_mode} | {self.dtype}{placement}{plan_tail})"
        )

        from . import diagnostics as _diag
        from . import fp8 as _fp8

        _diag.describe_config(self.model.config, self.label, self._log)
        _diag.describe_module(self.model, self.label, self._log)

        try:
            emb = self.model.get_input_embeddings()
            head = self.model.get_output_embeddings()
            tied = (
                emb is not None and head is not None
                and emb.weight.data_ptr() == head.weight.data_ptr()
            )
            if emb is not None:
                self._log(
                    f"  🧮 [{self.label}] embed_tokens "
                    f"{list(emb.weight.shape)} "
                    f"{str(emb.weight.dtype).replace('torch.', '')} "
                    f"({emb.weight.numel() * emb.weight.element_size() / 1e6:.0f} MB) "
                    f"| lm_head {'tied — 중복 없음' if tied else '별도 가중치'}"
                )
        except Exception:
            pass

        self.kv_plan = _fp8.plan_kv_cache(
            self.model, 4096, label=self.label, log=self._log
        )
        self._kv_adapter = _fp8.Fp8KVCacheAdapter(self.label, log=self._log)
        self.fp8_kv = bool(self._kv_adapter.build() is not None)
        if self.fp8_kv and self.vision:
            self._log(
                f"  ⏭ [{self.label}] 멀티모달 생성에서는 fp8 KV 를 쓰지 "
                f"않습니다 (위치 인덱싱이 캐시 길이에 의존)."
            )

    def _try_quant_config(self):
        cfg, mode = build_quant_config(log=self._log)

        if cfg is None:
            self._log(
                f"  ⏭ [{self.label}] 양자화를 쓸 수 없어 CPU 오프로드로 "
                f"진행합니다. 속도가 크게 느려집니다."
            )
            return None

        self.quant_mode = mode
        label = "4bit(NF4 + double quant)" if mode == "4bit" else "8bit(LLM.int8)"
        self._log(
            f"  🧊 [{self.label}] {label} 양자화를 적용합니다 — "
            f"가중치를 {'약 1/4' if mode == '4bit' else '약 1/2'} 로 "
            f"줄여 GPU 에 올립니다."
        )
        return cfg

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    @staticmethod
    def looks_reasoning(text: str) -> bool:
        s = str(text or "").strip().lower()
        if not s:
            return False
        if "<think>" in s:
            return True
        for opener in REASONING_OPENERS:
            if s.startswith(opener):
                return True
        return False

    def note_thinking(self, sample: str = "") -> None:
        if self.thinking:
            return
        if not self.looks_reasoning(sample):
            return
        self.thinking = True
        self._log(
            f"  🧠 [{self.label}] 사고형(thinking) 모델로 판정했습니다 "
            f"(응답이 '{str(sample).strip()[:28]}' 로 시작). 토큰 예산을 "
            f"{THINK_BUDGET_MULTIPLIER:.0f}배로 늘리고, 가능하면 사고 "
            f"단계를 끕니다."
        )

    def budget_for(self, base: int) -> int:
        n = max(1, int(base))
        if not self.thinking:
            return n
        scaled = int(n * THINK_BUDGET_MULTIPLIER)
        return int(min(THINK_BUDGET_CEIL, max(THINK_BUDGET_FLOOR, scaled)))

    @staticmethod
    def json_action(array: bool = False) -> str:
        head = JSON_ACTION_ARRAY if array else JSON_ACTION_OBJECT
        return f"\n{head} {NO_THINK_TAG}"

    def _json_stopper(self, opener: str, start_len: int):
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList
        except Exception:
            return None

        tok = getattr(self.processor, "tokenizer", None) or self.tokenizer
        if tok is None:
            return None

        close = "}" if opener == JSON_PREFILL_OBJECT else "]"
        label = self.label

        class _Balanced(StoppingCriteria):
            def __init__(self):
                self.depth = 1
                self.in_str = False
                self.esc = False
                self.seen = start_len
                self.tail = 0

            def __call__(self, input_ids, scores, **kw):
                seq = input_ids[0]
                n = int(seq.shape[-1])
                if n <= self.seen:
                    return False
                try:
                    piece = tok.decode(
                        seq[self.seen:n], skip_special_tokens=True
                    )
                except Exception:
                    self.seen = n
                    return False
                self.seen = n

                for ch in piece:
                    if self.in_str:
                        if self.esc:
                            self.esc = False
                        elif ch == "\\":
                            self.esc = True
                        elif ch == '"':
                            self.in_str = False
                        continue
                    if ch == '"':
                        self.in_str = True
                    elif ch == opener:
                        self.depth += 1
                    elif ch == close:
                        self.depth -= 1
                        if self.depth <= 0:
                            return True
                return False

        return StoppingCriteriaList([_Balanced()])

    def _prefill_ids(self, enc: dict, opener: str):
        if not opener:
            return enc, 0
        tok = getattr(self.processor, "tokenizer", None) or self.tokenizer
        ids = enc.get("input_ids")
        if tok is None or ids is None:
            return enc, 0

        try:
            import torch as _t
            seed = tok(opener, add_special_tokens=False, return_tensors="pt")
            seed_ids = seed["input_ids"].to(ids.device)
            merged = dict(enc)
            merged["input_ids"] = _t.cat([ids, seed_ids], dim=-1)
            mask = enc.get("attention_mask")
            if mask is not None:
                pad = _t.ones_like(seed_ids)
                merged["attention_mask"] = _t.cat(
                    [mask, pad.to(mask.device)], dim=-1
                )
            mm = enc.get("mm_token_type_ids")
            if mm is not None:
                zero = _t.zeros_like(seed_ids)
                merged["mm_token_type_ids"] = _t.cat(
                    [mm, zero.to(mm.device)], dim=-1
                )
            return merged, int(seed_ids.shape[-1])
        except Exception as e:
            self._log(
                f"    ⏭ [{self.label}] JSON 프리필 주입 실패 "
                f"({type(e).__name__}) — 프롬프트 지시만으로 진행합니다."
            )
            return enc, 0

    def _template_kwargs(self) -> dict:
        if self._no_think_kwarg is not None:
            return dict(self._no_think_kwarg)

        out: dict = {}
        tok = getattr(self.processor, "tokenizer", None) or self.tokenizer
        src = getattr(tok, "chat_template", "") or ""
        if not src:
            src = getattr(self.processor, "chat_template", "") or ""
        if "enable_thinking" in str(src):
            out["enable_thinking"] = False
            self._log(
                f"  🧠 [{self.label}] chat_template 이 enable_thinking 을 "
                f"지원합니다 — 사고 단계를 끄고 바로 답만 받습니다."
            )
        self._no_think_kwarg = dict(out)
        return dict(out)

    def _encode_via_template(self, prompt: str, pil):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ],
        }]
        extra = self._template_kwargs()
        try:
            enc = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                **extra,
            )
        except TypeError:
            try:
                enc = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            except Exception as e:
                if not self._vision_marker_logged:
                    self._log(
                        f"  ⏭ [{self.label}] 프로세서 템플릿 경로 불가 "
                        f"({type(e).__name__}: {str(e)[:70]}) → 수동 확장"
                    )
                return None
        except Exception as e:
            if not self._vision_marker_logged:
                self._log(
                    f"  ⏭ [{self.label}] 프로세서 템플릿 경로 불가 "
                    f"({type(e).__name__}: {str(e)[:70]}) → 수동 확장"
                )
            return None

        if enc is None:
            return None
        keys = set(enc.keys()) if hasattr(enc, "keys") else set()
        if "pixel_values" not in keys and "pixel_values_videos" not in keys:
            return None
        if not self._vision_marker_logged:
            self._log(
                f"  👁 [{self.label}] 프로세서 템플릿 경로 사용 "
                f"| 키 {sorted(keys)}"
            )
            self._vision_marker_logged = True
        return enc

    def _image_pad_count(self, img_enc) -> int:
        for key in ("image_grid_thw", "image_sizes"):
            thw = img_enc.get(key) if hasattr(img_enc, "get") else None
            if thw is None:
                continue
            try:
                arr = thw[0]
                vals = [int(v) for v in (arr.tolist() if hasattr(arr, "tolist") else arr)]
            except Exception:
                continue
            if not vals:
                continue
            total = 1
            for v in vals:
                total *= max(1, v)
            merge = 1
            ip = getattr(self.processor, "image_processor", None)
            m = getattr(ip, "merge_size", None) or getattr(
                self.processor, "merge_size", None
            )
            if isinstance(m, int) and m > 0:
                merge = m * m
            return max(1, total // merge)

        pv = img_enc.get("pixel_values") if hasattr(img_enc, "get") else None
        if pv is not None and hasattr(pv, "shape"):
            try:
                return max(1, int(pv.shape[0]))
            except Exception:
                return 1
        return 1

    def _encode_manual(self, prompt: str, pil):
        ip = getattr(self.processor, "image_processor", None)
        tok = getattr(self.processor, "tokenizer", None) or self.tokenizer
        if ip is None or tok is None:
            return None, RuntimeError("image_processor 또는 tokenizer 없음")

        try:
            img_enc = ip(images=[pil], return_tensors="pt")
        except Exception as e:
            return None, e

        vocab = set()
        try:
            vocab = set(tok.get_vocab().keys())
        except Exception:
            vocab = set()

        start, pad, end = "<|vision_start|>", "<|image_pad|>", "<|vision_end|>"
        for s, p, e in VISION_PLACEHOLDERS:
            probe = p or s
            if probe and probe in vocab:
                start, pad, end = s, p, e
                break

        n_pad = self._image_pad_count(img_enc)
        marker = f"{start}{pad * n_pad}{end}" if pad else start

        if not self._vision_marker_logged:
            self._log(
                f"  👁 [{self.label}] 수동 확장 — 플레이스홀더 '{pad or start}' "
                f"× {n_pad}개"
            )
            self._vision_marker_logged = True

        messages = [{"role": "user", "content": f"{marker}\n{prompt}"}]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = f"{marker}\n{prompt}"
        if marker not in text:
            text = f"{marker}\n{text}"

        try:
            txt_enc = tok([text], return_tensors="pt")
        except Exception as e:
            return None, e

        enc = dict(txt_enc)
        for k, v in (img_enc.items() if hasattr(img_enc, "items") else []):
            enc[k] = v
        return enc, None

    @torch.no_grad()
    def generate_with_image(
        self,
        prompt: str,
        image,
        max_new_tokens: Optional[int] = None,
        kwargs_json_opener: str = "",
    ) -> str:
        if not self.vision or self.processor is None or image is None:
            return self.generate(
                prompt, max_new_tokens=max_new_tokens,
                kwargs_json_opener=kwargs_json_opener,
            )

        pil = image.convert("RGB")
        last = None

        enc = self._encode_via_template(prompt, pil)
        if enc is None:
            enc, last = self._encode_manual(prompt, pil)

        if enc is None:
            self._log(
                f"  ⚠ [{self.label}] 이미지 인코딩 실패"
                f"({type(last).__name__ if last else '?'}: "
                f"{str(last)[:110]}) → 텍스트 경로"
            )
            from . import diagnostics as _diag
            if _diag.enabled(2):
                import traceback
                for line in traceback.format_exc().splitlines()[-10:]:
                    self._log(f"      {line}")
            return self.generate(prompt, max_new_tokens=max_new_tokens)

        enc = {
            k: (v.to(self.model.device) if hasattr(v, "to") else v)
            for k, v in enc.items()
        }

        opener = str(kwargs_json_opener or "")
        base_len = 0
        ids0 = enc.get("input_ids")
        if ids0 is not None:
            base_len = int(ids0.shape[-1])

        seeded = 0
        if opener:
            enc, seeded = self._prefill_ids(enc, opener)

        gen_kwargs = {
            "max_new_tokens": self.budget_for(
                int(max_new_tokens or self.max_new_tokens)
            ),
            "do_sample": False,
        }

        if opener and seeded > 0:
            stopper = self._json_stopper(opener, base_len + seeded)
            if stopper is not None:
                gen_kwargs["stopping_criteria"] = stopper

        try:
            out = self.model.generate(**enc, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log(f"  ⚠ [{self.label}] VRAM 부족으로 이미지 생성을 건너뜁니다.")
            return ""
        except Exception as e:
            self._log(
                f"  ⚠ [{self.label}] 이미지 생성 실패 "
                f"({type(e).__name__}: {e}) → 텍스트 경로"
            )
            from . import diagnostics as _diag
            if _diag.enabled(2):
                import traceback
                for line in traceback.format_exc().splitlines()[-8:]:
                    self._log(f"      {line}")
            return self.generate(prompt, max_new_tokens=max_new_tokens)

        ids = enc.get("input_ids")
        start = int(ids.shape[-1]) if ids is not None else 0
        gen = out[0][start:]
        try:
            text = self.processor.decode(gen, skip_special_tokens=True).strip()
        except Exception:
            text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()

        if opener and seeded > 0:
            text = opener + text
            if not self._json_marker_logged:
                self._json_marker_logged = True
                self._log(
                    f"    🔒 [JSON FORCE] '{opener}' 를 프리필로 주입해 "
                    f"모델이 JSON 중간부터 생성하도록 강제했습니다. "
                    f"사고 서두가 물리적으로 나올 수 없습니다."
                )

        self.note_thinking(text)
        return text

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        kwargs_json_opener: str = "",
    ) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        opener = str(kwargs_json_opener or "")
        if opener:
            text = text + opener

        enc = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": self.budget_for(
                int(max_new_tokens or self.max_new_tokens)
            ),
            "do_sample": False,
        }
        if opener:
            ids0 = enc.get("input_ids")
            if ids0 is not None:
                stopper = self._json_stopper(opener, int(ids0.shape[-1]))
                if stopper is not None:
                    gen_kwargs["stopping_criteria"] = stopper
        if getattr(self, "fp8_kv", False):
            cache = self._kv_adapter.build()
            if cache is not None:
                gen_kwargs["past_key_values"] = cache

        try:
            out = self.model.generate(**enc, **gen_kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            self._log(f"  ⚠ [{self.label}] VRAM 부족으로 생성을 건너뜁니다.")
            return ""
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                self._log(f"  ⚠ [{self.label}] VRAM 부족으로 생성을 건너뜁니다.")
                return ""
            raise

        gen = out[0][len(enc["input_ids"][0]):]
        text = self.tokenizer.decode(gen, skip_special_tokens=True).strip()
        if opener:
            text = opener + text
        self.note_thinking(text)
        return text

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        s = str(text or "")
        s = THINK_BLOCK_RE.sub(" ", s)
        idx = s.rfind("</think>")
        if idx >= 0:
            s = s[idx + len("</think>"):]
        idx = s.rfind("<think>")
        if idx >= 0:
            s = s[:idx]
        s = s.strip()

        if not s:
            return s

        low = s.lower()
        if any(low.startswith(o) for o in REASONING_OPENERS):
            for marker in ("\n{", "\n[", "\n```"):
                cut = s.find(marker)
                if cut >= 0:
                    return s[cut:].strip()
            for opener in ("{", "["):
                cut = s.find(opener)
                if cut > 0:
                    return s[cut:].strip()
            return ""

        return s

    @staticmethod
    def is_schema_echo(value: str, field_name: str = "") -> bool:
        v = str(value or "").strip()
        if not v:
            return True
        low = v.lower().strip(" .:\"'`")
        if low in SCHEMA_ECHO_TOKENS:
            return True
        if TYPE_MARKER_RE.search(v):
            return True
        if FIELD_ECHO_RE.match(low):
            return True
        fname = str(field_name or "").strip().lower()
        if fname and low in (fname, fname.replace("_", " ")):
            return True
        return False

    @staticmethod
    def _compact(text: str) -> str:
        return "".join(ch for ch in str(text or "").lower() if ch.isalnum())

    @staticmethod
    def _word_set(text: str) -> set:
        out = set()
        buf = []
        for ch in str(text or "").lower():
            if ch.isalnum():
                buf.append(ch)
                continue
            if buf:
                out.add("".join(buf))
                buf = []
        if buf:
            out.add("".join(buf))
        return {w for w in out if len(w) >= 2}

    @classmethod
    def echoes_spec(cls, value: str, spec: str) -> bool:
        v = str(value or "").strip()
        s = str(spec or "").strip()
        if not v or not s:
            return False

        cv = "".join(ch for ch in v.lower() if ch.isalnum())
        cs = "".join(ch for ch in s.lower() if ch.isalnum())
        if not cv or not cs:
            return False
        if cv == cs or cv in cs or cs in cv:
            return True

        vw = cls._word_set(v)
        sw = cls._word_set(s)
        if len(sw) < SPEC_ECHO_MIN_TOKENS or len(vw) < SPEC_ECHO_MIN_TOKENS:
            return False

        hit = len(vw & sw)
        return (hit / float(len(sw))) >= SPEC_ECHO_RATIO

    @staticmethod
    def _key_sim(a: str, b: str) -> float:
        sa = "".join(ch for ch in str(a).lower() if ch.isalnum())
        sb = "".join(ch for ch in str(b).lower() if ch.isalnum())
        if not sa or not sb:
            return 0.0
        if sa == sb:
            return 1.0
        m, n = len(sa), len(sb)
        prev = [0] * (n + 1)
        for i in range(1, m + 1):
            cur = [0] * (n + 1)
            ca = sa[i - 1]
            for j in range(1, n + 1):
                if ca == sb[j - 1]:
                    cur[j] = prev[j - 1] + 1
                else:
                    cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
            prev = cur
        return (2.0 * prev[n]) / float(m + n)

    @classmethod
    def _repair_key(cls, key: str, fields: Sequence[str]) -> Tuple[str, int]:
        raw = str(key or "").strip()
        if not raw:
            return "", 0

        idx = 0
        m = INDEX_KEY_RE.match(raw)
        if m and m.group(1):
            base = m.group(1)
            try:
                idx = int(m.group(2))
            except Exception:
                idx = 0
        else:
            base = raw

        if base in fields:
            return base, idx

        best = ""
        best_score = 0.0
        for f in fields:
            s = cls._key_sim(base, f)
            if s > best_score:
                best_score = s
                best = f
        if best and best_score >= KEY_SIM_FLOOR:
            return best, idx
        return "", idx

    @staticmethod
    def _primary_field(
        category: str,
        fields: Sequence[str],
        hint: str = "",
    ) -> str:
        names = list(fields)
        if not names:
            return ""
        h = "".join(ch for ch in str(hint or "").lower() if ch.isalnum())
        if h:
            for f in names:
                if "".join(ch for ch in f.lower() if ch.isalnum()) == h:
                    return f
        stem = "".join(ch for ch in str(category or "").lower() if ch.isalnum())
        stem = stem.rstrip("s")
        if stem:
            for f in names:
                if stem in "".join(ch for ch in f.lower() if ch.isalnum()):
                    return f
        for f in names:
            if f.endswith("_text") or f.endswith("_name"):
                return f
        return names[0]

    @staticmethod
    def _repair_braceless_array(body: str) -> Optional[str]:
        inner = str(body or "").strip()
        if not inner.startswith("[") or not inner.endswith("]"):
            return None
        core = inner[1:-1].strip()
        if not core or core.startswith("{") or core.startswith("["):
            return None
        if len(BRACELESS_PAIR_RE.findall(core)) < 1:
            return None
        return "[{" + core.rstrip(",") + "}]"

    @staticmethod
    def _claimable(value: str) -> bool:
        v = str(value or "").strip()
        if len(v) < 3:
            return False
        cv = "".join(ch for ch in v.lower() if ch.isalnum())
        if not cv:
            return False
        if cv.isdigit() and len(cv) <= 4:
            return False
        return True

    @staticmethod
    def _has_identity(row: Dict[str, str]) -> bool:
        vals = [str(v).strip() for v in row.values() if str(v or "").strip()]
        if not vals:
            return False
        if len(vals) >= 2:
            return True
        return any(any(ch.isalpha() for ch in v) for v in vals)

    def refine_array(
        self,
        category: str,
        field_specs: Dict[str, str],
        raw_text: str,
        existing: Optional[List[dict]] = None,
        label_bank: Optional[Sequence[str]] = None,
        hint: str = "",
        image=None,
        primary_hint: str = "",
        lang_code: str = "",
        script: str = "",
    ) -> List[dict]:
        import json

        use_vision = bool(self.vision and image is not None)
        body = str(raw_text or "").strip()
        if not field_specs:
            return []
        if not body and not use_vision:
            return []

        if use_vision and body:
            ok, why = self.draft_matches_language(body, lang_code)
            if not ok:
                self._log(
                    f"    🚯 [DRAFT DROP] [{category}] OCR 초안을 프롬프트에서 "
                    f"제외합니다 — {why}."
                )
                body = ""

        lines = [f'    "{k}": <{v or "value"} or null>' for k, v in field_specs.items()]

        directive = self.script_directive(lang_code, script)

        if use_vision:
            prompt = (
                "You read one cropped region of a document or comic image and "
                "extract every separate occurrence as its own row.\n"
                "Copy values exactly as printed. Never invent a value.\n"
                + directive
                + "Printed column headers are NOT values.\n"
                "If the same field appears several times, emit one object per "
                "occurrence. Do NOT number the keys.\n"
                "Every row MUST be wrapped in braces. "
                'Correct: [{"a": "1"}, {"a": "2"}]  '
                'Wrong: ["a": "1", "a": "2"]\n'
                "Count the printed rows before you write. If you see three "
                "rows, the array has three elements. The example below is a "
                "shape, not a row count.\n"
                "Return ONLY a JSON array, no markdown, no reasoning.\n\n"
                f"REGION: {category}\n"
                + (f"HINT: {hint}\n" if hint else "")
                + (f"OCR DRAFT (may be wrong):\n{body}\n\n" if body else "")
                + "SCHEMA: [\n  {\n" + ",\n".join(lines) + "\n  }\n]"
                + self.json_action(array=True)
            )
        else:
            prompt = (
                "You extract repeated table rows from noisy OCR text of one region "
                "of a business document.\n"
                "Copy values verbatim from the OCR TEXT. Never invent a value.\n"
                + directive
                + "Printed column headers are NOT values. Return one object per row.\n"
                "Count the printed rows before you write. Never merge two rows "
                "into one element. Never split one row into two.\n"
                "Return ONLY a JSON array, no markdown, no reasoning.\n\n"
                f"REGION: {category}\n"
                + (f"HINT: {hint}\n" if hint else "")
                + f"OCR TEXT:\n{body}\n\n"
                "SCHEMA: [\n  {\n" + ",\n".join(lines) + "\n  }\n]"
                + self.json_action(array=True)
            )

        per_field = 56 if use_vision else 32
        budget = min(2560, 256 + per_field * len(field_specs))
        try:
            if use_vision:
                out = self.generate_with_image(
                    prompt, image, max_new_tokens=budget,
                    kwargs_json_opener=JSON_PREFILL_ARRAY,
                )
            else:
                out = self.generate(
                    prompt, max_new_tokens=budget,
                    kwargs_json_opener=JSON_PREFILL_ARRAY,
                )
        except Exception:
            return []

        cleaned = self._strip_reasoning(out)
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()

        parsed = None
        a0 = cleaned.find("[")
        a1 = cleaned.rfind("]")
        if a0 != -1 and a1 > a0:
            try:
                parsed = json.loads(cleaned[a0: a1 + 1])
            except Exception:
                parsed = None

        if parsed is None and a0 != -1 and a1 > a0:
            fixed = self._repair_braceless_array(cleaned[a0: a1 + 1])
            if fixed:
                try:
                    parsed = json.loads(fixed)
                except Exception:
                    parsed = None
                if isinstance(parsed, list):
                    self._log(
                        f"    🔧 [BRACE REPAIR] [{category}] 배열 안에 객체 "
                        f"중괄호가 빠져 있어 자동으로 감쌌습니다."
                    )

        if parsed is None:
            o0 = cleaned.find("{")
            o1 = cleaned.rfind("}")
            if o0 != -1 and o1 > o0:
                try:
                    obj = json.loads(cleaned[o0: o1 + 1])
                except Exception:
                    obj = None
                if isinstance(obj, dict):
                    self._log(
                        f"    🔧 [ARRAY COERCE] [{category}] 단일 객체 응답을 "
                        f"원소 1개 배열로 승격합니다."
                    )
                    parsed = [obj]

        if not isinstance(parsed, list):
            objs: List[dict] = []
            depth = 0
            start_at = -1
            for i, ch in enumerate(cleaned):
                if ch == "{":
                    if depth == 0:
                        start_at = i
                    depth += 1
                elif ch == "}":
                    if depth > 0:
                        depth -= 1
                    if depth == 0 and start_at >= 0:
                        try:
                            frag = json.loads(cleaned[start_at: i + 1])
                        except Exception:
                            frag = None
                        if isinstance(frag, dict):
                            objs.append(frag)
                        start_at = -1
            if objs:
                self._log(
                    f"    🩹 [ARRAY SALVAGE] '{category}' 배열 파싱은 "
                    f"실패했지만 완성된 객체 {len(objs)}건을 구조합니다."
                )
                parsed = objs

        if not isinstance(parsed, list):
            head = cleaned[:80].replace("\n", " ")
            self._log(
                f"    🚫 [{category}] 배열 응답 파싱 실패 — 폐기합니다. "
                f"(응답 선두: {head!r})"
            )
            return []

        labels = {self._compact(t) for t in (label_bank or []) if t}
        cb = self._compact(body)
        ground = (not use_vision) and bool(cb)
        names = list(field_specs.keys())
        primary = self._primary_field(category, names, hint=primary_hint)

        seen = {json.dumps(r, sort_keys=True, ensure_ascii=False)
                for r in (existing or [])}
        kept: List[dict] = []
        dropped_ident = 0
        dropped_dup = 0
        repaired = 0
        split_rows = 0
        bare_rows = 0

        romanized = 0

        def _accept(v) -> str:
            nonlocal romanized
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                v = str(v)
            if not isinstance(v, str):
                return ""
            s = v.strip()
            if not s:
                return ""
            if self.looks_romanized(s, lang_code, script):
                romanized += 1
                return ""
            cs = self._compact(s)
            if not cs or cs in labels:
                return ""
            if ground and cs not in cb:
                return ""
            return s

        def _push(row: Dict[str, str]) -> None:
            nonlocal dropped_ident, dropped_dup
            if not self._has_identity(row):
                dropped_ident += 1
                return
            sig = json.dumps(row, sort_keys=True, ensure_ascii=False)
            if sig in seen:
                dropped_dup += 1
                return
            seen.add(sig)
            kept.append(row)

        for item in parsed:
            if isinstance(item, (str, int, float)) and primary:
                s = _accept(item)
                if s and not self.is_schema_echo(s, primary):
                    bare_rows += 1
                    _push({primary: s})
                continue

            if isinstance(item, list):
                for sub in item:
                    s = _accept(sub)
                    if s and primary and not self.is_schema_echo(s, primary):
                        bare_rows += 1
                        _push({primary: s})
                continue

            if not isinstance(item, dict):
                continue

            buckets: Dict[int, Dict[str, str]] = {}
            for key, val in item.items():
                field, idx = self._repair_key(key, names)
                if not field:
                    continue
                if field != str(key).strip():
                    repaired += 1
                s = _accept(val)
                if not s or self.is_schema_echo(s, field):
                    continue
                buckets.setdefault(int(idx), {})[field] = s

            if not buckets:
                continue
            if len(buckets) > 1:
                split_rows += len(buckets) - 1
                self._log(
                    f"    🔢 [INDEX SPLIT] [{category}] 같은 필드의 값이 "
                    f"{len(buckets)}개라 행 {len(buckets)}건으로 나눕니다 "
                    f"(인덱스 {sorted(buckets.keys())})."
                )
            for _idx in sorted(buckets.keys()):
                _push(buckets[_idx])

        extra = []
        if repaired:
            extra.append(f"키 복원 {repaired}건")
        if split_rows:
            extra.append(f"인덱스 분리 {split_rows}행")
        if bare_rows:
            extra.append(f"값 나열 승격 {bare_rows}건")
        if romanized:
            extra.append(f"음차 폐기 {romanized}건")
            name, _lang, _s = self.script_profile(lang_code, script)
            self._log(
                f"    🔤 [ROMANIZE BLOCK] [{category}] {name or '원문'} 스크립트 "
                f"대신 로마자로 답한 값 {romanized}건을 폐기했습니다."
            )
        tail = (" | " + " | ".join(extra)) if extra else ""

        self._log(
            f"    ➕ [{category}] 배열 신규 {len(kept)}건 | 겹침 중복 "
            f"{dropped_dup}건 제거 | 정체 없는 행 {dropped_ident}건 폐기"
            f"{tail}"
        )
        return kept

    def refine_category(
        self,
        category: str,
        field_specs: Dict[str, str],
        raw_text: str,
        claimed: Optional[Dict[str, str]] = None,
        label_bank: Optional[Sequence[str]] = None,
        hint: str = "",
        image=None,
        lang_code: str = "",
        script: str = "",
    ) -> Dict[str, str]:
        import json

        use_vision = bool(self.vision and image is not None)
        body = str(raw_text or "").strip()
        if not field_specs:
            return {}
        if not body and not use_vision:
            return {}

        if use_vision and body:
            ok, why = self.draft_matches_language(body, lang_code)
            if not ok:
                self._log(
                    f"    🚯 [DRAFT DROP] [{category}] OCR 초안을 프롬프트에서 "
                    f"제외합니다 — {why}."
                )
                body = ""

        lines = [f'  "{k}": <{v or "value"} or null>' for k, v in field_specs.items()]
        banned = ""
        ban_vals = [
            v for v in (claimed or {}).values()
            if v and self._claimable(v)
        ][:20]
        if ban_vals:
            banned = (
                "ALREADY CLAIMED (do NOT return these values again):\n"
                + "\n".join(f"  - {v}" for v in ban_vals)
                + "\n\n"
            )

        directive = self.script_directive(lang_code, script)

        if use_vision:
            prompt = (
                "You read one cropped region of a business document image and "
                "extract structured fields.\n"
                "Copy values exactly as printed. Never invent a value.\n"
                + directive
                + "Printed form labels are NOT values. If a field is absent, "
                "use null.\n"
                "Return ONLY a JSON object, no markdown, no reasoning.\n\n"
                f"REGION: {category}\n"
                + (f"HINT: {hint}\n" if hint else "")
                + banned
                + (f"OCR DRAFT (may be wrong):\n{body}\n\n" if body else "")
                + "SCHEMA: {\n" + ",\n".join(lines) + "\n}"
                + self.json_action(array=False)
            )
        else:
            prompt = (
                "You extract structured fields from noisy OCR text of one region "
                "of a business document.\n"
                "Copy values verbatim from the OCR TEXT. Never invent a value.\n"
                + directive
                + "Printed form labels are NOT values. If a field is absent, use null.\n"
                "Return ONLY a JSON object, no markdown, no reasoning.\n\n"
                f"REGION: {category}\n"
                + (f"HINT: {hint}\n" if hint else "")
                + banned
                + f"OCR TEXT:\n{body}\n\n"
                "SCHEMA: {\n" + ",\n".join(lines) + "\n}"
                + self.json_action(array=False)
            )

        per_field = 40 if use_vision else 24
        budget = min(2048, 192 + per_field * len(field_specs))
        try:
            if use_vision:
                out = self.generate_with_image(
                    prompt, image, max_new_tokens=budget,
                    kwargs_json_opener=JSON_PREFILL_OBJECT,
                )
            else:
                out = self.generate(
                    prompt, max_new_tokens=budget,
                    kwargs_json_opener=JSON_PREFILL_OBJECT,
                )
        except Exception:
            return {}

        cleaned = self._strip_reasoning(out)
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        parsed = None
        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start: end + 1])
            except Exception:
                parsed = None

        if not isinstance(parsed, dict) and start != -1:
            frag = cleaned[start:]
            salvaged: Dict[str, str] = {}
            for m in re.finditer(
                r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*"([^"\\]{1,160})"', frag
            ):
                salvaged[m.group(1)] = m.group(2)
            for m in re.finditer(
                r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:\s*(-?\d+(?:\.\d+)?)\s*[,}]', frag
            ):
                salvaged.setdefault(m.group(1), m.group(2))
            if salvaged:
                self._log(
                    f"    🩹 [JSON SALVAGE] '{category}' 응답이 잘렸지만 "
                    f"완성된 쌍 {len(salvaged)}건을 구조해 냅니다."
                )
                parsed = salvaged

        if not isinstance(parsed, dict):
            head = cleaned[:80].replace("\n", " ")
            self._log(
                f"  🚫 [{self.label}] '{category}' JSON 응답 폐기 "
                f"— OCR 원문을 유지합니다. (응답 선두: {head!r})"
            )
            return {}

        labels = {self._compact(t) for t in (label_bank or []) if t}
        claimed_c = {
            self._compact(v) for v in (claimed or {}).values()
            if v and self._claimable(v)
        }
        cb = self._compact(body)
        ground = (not use_vision) and bool(cb)
        spec_echo = 0
        inner_dup = 0
        echoed_dup = ""
        used_local: Dict[str, str] = {}

        kept: Dict[str, str] = {}
        echo = 0
        label_echo = 0
        halluc = 0
        dup = 0
        romanized = 0

        for key, val in parsed.items():
            if key not in field_specs:
                continue
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                val = str(val)
            if not isinstance(val, str):
                continue
            v = val.strip()
            if not v:
                continue
            if self.is_schema_echo(v, key):
                echo += 1
                continue
            if self.looks_romanized(v, lang_code, script):
                romanized += 1
                self._log(
                    f"    🔤 [ROMANIZE BLOCK] [{category}] '{key}' = "
                    f"\"{v[:36]}\" 는 로마자 음차입니다 — 폐기합니다."
                )
                continue
            cv = self._compact(v)
            if not cv:
                continue
            if cv in labels:
                label_echo += 1
                self._log(
                    f"    🚫 [LABEL ECHO] [{category}] '{key}' = \"{v[:36]}\" "
                    f"— 서식의 인쇄 라벨이므로 폐기합니다."
                )
                continue
            if cv in claimed_c:
                dup += 1
                self._log(
                    f"    ⚠️ [CLAIM VIOLATION] [{category}] '{key}' = "
                    f"\"{v[:36]}\" 는 이미 다른 축이 확정한 값입니다 "
                    f"— 이 카테고리에서는 비워 둡니다."
                )
                echoed_dup = v if not echoed_dup else echoed_dup
                continue
            if ground and cv not in cb:
                halluc += 1
                continue
            if self.echoes_spec(v, field_specs.get(key, "")):
                spec_echo += 1
                continue
            if self._claimable(v):
                if cv in used_local:
                    inner_dup += 1
                    self._log(
                        f"    ♻️ [DUP FIELD] [{category}] '{key}' 가 "
                        f"'{used_local[cv]}' 와 같은 값 \"{v[:24]}\" 을 "
                        f"반환해 폐기합니다."
                    )
                    continue
                used_local[cv] = key
            kept[key] = v

        self._log(
            f"    ✅ [{category}] 신규 {len(kept)}건 | 스키마 에코 폐기 "
            f"{echo}건 | 설명문 에코 {spec_echo}건 | 라벨 에코 "
            f"{label_echo}건 | 환각 {halluc}건 | 선점 중복 {dup}건 "
            f"| 자체 중복 {inner_dup}건 | 음차 폐기 {romanized}건"
        )
        if not kept and echoed_dup:
            self._log(
                f"    ℹ️ [{category}] 추출값이 전부 선점 중복이라 원문 "
                f"\"{echoed_dup[:30]}\" 만 남깁니다."
            )
        return kept

    @staticmethod
    def script_profile(
        lang_code: str = "",
        script: str = "",
    ) -> Tuple[str, str, str]:
        if script and script in SCRIPT_BY_BLOCK:
            return SCRIPT_BY_BLOCK[script]
        code = str(lang_code or "").strip().lower()
        if code in SCRIPT_OF_LANG:
            return SCRIPT_OF_LANG[code]
        return "", "", ""

    @classmethod
    def script_directive(
        cls,
        lang_code: str = "",
        script: str = "",
    ) -> str:
        name, lang, sample = cls.script_profile(lang_code, script)
        if not name:
            return (
                "SCRIPT RULE: Reproduce every character exactly as drawn, "
                "in the writing system shown in the image. "
                "Do NOT romanize. Do NOT transliterate. Do NOT translate. "
                "Do NOT append a translation in parentheses.\n"
            )

        head = (
            f"SCRIPT RULE: The text in this image is written in the {name} "
            f"script ({lang}). Every value you return MUST be written in "
            f"{name} characters, byte for byte as printed.\n"
            "  - Do NOT romanize. 'annyeonghaseyo' style output is WRONG.\n"
            "  - Do NOT translate into English.\n"
            "  - Do NOT append a translation in parentheses.\n"
            "  - If you cannot read a character, omit it rather than "
            "guessing a Latin substitute.\n"
        )
        if sample:
            head += f"  - Correct output style — {sample}\n"
        return head

    @staticmethod
    def looks_prompt_echo(text: str) -> Tuple[bool, str]:
        s = str(text or "").strip().lower()
        if not s:
            return False, ""

        flat = "".join(ch for ch in s if not ch.isspace())
        for marker in PROMPT_ECHO_MARKERS:
            probe = marker.replace(" ", "")
            if probe and probe in flat:
                return True, marker
        return False, ""

    @staticmethod
    def looks_romanized(text: str, lang_code: str = "", script: str = "") -> bool:
        s = str(text or "").strip()
        if not s:
            return False

        expect = script
        if not expect:
            expect = {
                "kor": "Hangul", "jpn": "Kana", "zho": "Han",
                "rus": "Cyrillic", "ukr": "Cyrillic", "ara": "Arabic",
                "tha": "Thai", "hin": "Devanagari", "ell": "Greek",
                "heb": "Hebrew", "tam": "Tamil", "tel": "Telugu",
            }.get(str(lang_code or "").strip().lower(), "")

        if expect not in NON_LATIN_SCRIPTS:
            return False

        try:
            from .lang_codes import block_census
        except Exception:
            return False

        census = block_census(s)
        total = sum(census.values())
        if total < 4:
            return False

        latin = int(census.get("Latin", 0))
        return (latin / float(total)) >= ROMANIZE_RATIO

    @staticmethod
    def draft_matches_language(draft: str, lang_code: str) -> Tuple[bool, str]:
        s = str(draft or "").strip()
        code = str(lang_code or "").strip().lower()
        if not s or not code:
            return True, ""

        try:
            from .lang_codes import block_census
        except Exception:
            return True, ""

        census = block_census(s)
        if not census:
            return True, ""

        expect = {
            "kor": "Hangul", "jpn": "Kana", "zho": "Han",
            "rus": "Cyrillic", "ara": "Arabic", "tha": "Thai",
            "hin": "Devanagari", "ell": "Greek", "heb": "Hebrew",
        }.get(code, "Latin")

        total = sum(census.values())
        hit = int(census.get(expect, 0))
        if expect == "Latin":
            hit += int(census.get("Han", 0)) if code == "zho" else 0
        ratio = hit / float(max(1, total))

        if ratio >= 0.25:
            return True, ""

        top = max(census.items(), key=lambda kv: kv[1])[0]
        return False, (
            f"문서 언어 '{code}'({expect}) 인데 OCR 초안은 '{top}' 위주 "
            f"({expect} {ratio:.0%})"
        )

    def read_raw(
        self,
        image,
        hint: str = "",
        max_chars: int = 400,
        lang_code: str = "",
        script: str = "",
    ) -> str:
        if not self.vision or image is None:
            return ""

        prompt = (
            "Transcribe every piece of text visible in this image.\n"
            "Write the characters exactly as drawn. Do NOT explain.\n"
            + self.script_directive(lang_code, script)
            + "Separate distinct text blocks with ' / '.\n"
            "If there is no text, reply with an empty line."
            + (f"\nCONTEXT: {hint}" if hint else "")
            + f"\n{NO_THINK_TAG}"
        )

        try:
            out = self.generate_with_image(prompt, image, max_new_tokens=256)
        except Exception as e:
            self._log(f"    ⚠ [RAW READ] 실패({type(e).__name__}: {e})")
            return ""

        text = self._strip_reasoning(out)
        text = text.replace("```", "").strip()

        echo, marker = self.looks_prompt_echo(text)
        if echo:
            self._log(
                f"    🚯 [PROMPT ECHO] 지시문 '{marker}' 을 그대로 되뱉었습니다 "
                f"— 판독 실패로 처리합니다: {text[:32]!r}"
            )
            return ""

        for junk in ("here is", "the text", "transcription:", "i see"):
            if text.lower().startswith(junk):
                cut = text.find(":")
                if 0 <= cut < 40:
                    text = text[cut + 1:].strip()
                break
        if len(text) > max_chars:
            text = text[:max_chars]
        text = text.strip()

        alnum = sum(1 for ch in text if ch.isalnum())
        if alnum < 2:
            self._log(
                f"    ⏭ [RAW READ] 판독 결과에 글자가 {alnum}자뿐이라 "
                f"버립니다 ({text[:20]!r})."
            )
            return ""

        low = text.lower().strip(" .!/")
        if low in ("", "no text", "none", "empty", "n/a", "nothing"):
            self._log("    ⏭ [RAW READ] 글자 없음 응답 — 버립니다.")
            return ""

        if self.looks_romanized(text, lang_code, script):
            name, _lang, _s = self.script_profile(lang_code, script)
            self._log(
                f"    🔤 [RAW READ] 로마자 음차 응답이라 {name or '원문'} "
                f"스크립트로 재판독합니다: {text[:32]!r}"
            )
            retry = (
                f"The image contains {name or 'non-Latin'} characters.\n"
                "Output ONLY those characters. A romanized answer is a "
                "failure. Do not write any Latin letters unless they are "
                "literally printed in the image.\n"
                "Transcribe now:"
            )
            try:
                again = self.generate_with_image(retry, image, max_new_tokens=256)
            except Exception:
                again = ""
            again = self._strip_reasoning(again).replace("```", "").strip()

            echo2, marker2 = self.looks_prompt_echo(again)
            if echo2:
                self._log(
                    f"    🚯 [PROMPT ECHO] 재판독도 지시문 '{marker2}' 을 "
                    f"되뱉어 버립니다."
                )
                return ""

            if again and not self.looks_romanized(again, lang_code, script):
                return again[:max_chars].strip()
            self._log("    ⏭ [RAW READ] 재판독도 음차라 버립니다.")
            return ""

        return text

    def refine_field(self, field_name: str, raw_text: str, hint: str = "") -> dict:
        import json

        body = str(raw_text or "").strip()
        if not body:
            return {}

        prompt = (
            "You extract one field value from noisy OCR text of a business document.\n"
            "Copy the value verbatim from the OCR TEXT. Never invent a value.\n"
            "If the OCR TEXT does not contain this field, return an empty string.\n"
            "Return ONLY a JSON object. No markdown, no explanation, no reasoning.\n\n"
            f"FIELD: {field_name}\n"
            + (f"HINT: {hint}\n" if hint else "")
            + f"OCR TEXT:\n{body}\n\n"
            'SCHEMA: {"value": "<copy from OCR TEXT or empty string>"}'
            + self.json_action(array=False)
        )

        try:
            out = self.generate(
                prompt, max_new_tokens=128,
                kwargs_json_opener=JSON_PREFILL_OBJECT,
            )
        except Exception:
            return {}

        cleaned = self._strip_reasoning(out)
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        if not cleaned:
            return {}

        value = ""
        start = cleaned.find("{")
        end = cleaned.rfind("}")

        if start != -1 and end > start:
            try:
                parsed = json.loads(cleaned[start: end + 1])
            except Exception:
                parsed = None

            if not isinstance(parsed, dict):
                self._log(
                    f"  🚫 [{self.label}] '{field_name}' JSON 파싱 실패 응답 폐기 "
                    f"— OCR 원문을 유지합니다."
                )
                return {}

            for key in ("value", "text", field_name):
                v = parsed.get(key)
                if isinstance(v, str) and v.strip():
                    value = v.strip()
                    break

            if not value:
                self._log(
                    f"  🚫 [{self.label}] '{field_name}' 스키마 에코 응답 폐기 "
                    f"(키 {list(parsed.keys())[:4]}) — OCR 원문을 유지합니다."
                )
                return {}
        else:
            if "{" in cleaned or '"' in cleaned:
                self._log(
                    f"  🚫 [{self.label}] '{field_name}' 잘린 JSON 응답 폐기 "
                    f"— OCR 원문을 유지합니다."
                )
                return {}
            value = cleaned

        if self.is_schema_echo(value, field_name):
            self._log(
                f"  🚫 [{self.label}] '{field_name}' 플레이스홀더 "
                f"'{value[:24]}' 폐기 — OCR 원문을 유지합니다."
            )
            return {}

        cv = self._compact(value)
        cb = self._compact(body)
        if cv and cb and cv not in cb:
            self._log(
                f"  🚫 [{self.label}] '{field_name}' 응답 '{value[:24]}' 이 "
                f"OCR 원문에 없습니다 → 환각으로 보고 폐기합니다."
            )
            return {}

        return {"value": value}

    def unload(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._kv_adapter = None
        reclaim(log=self._log, label=f"{self.label} 반납")


class EmbeddingRouter:
    def __init__(self, log=None, name: str = "router"):
        self._providers: List[tuple] = []
        self._log_fn = log or (lambda m: None)
        self.active = ""
        self.name = str(name)
        self.space_dim = 0
        self.last_failed = False
        self.dims: Dict[str, int] = {}
        self._failed: Dict[str, int] = {}
        self._rejected: Dict[str, int] = {}

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def lock_space(self, dim: int, owner: str = "") -> None:
        self.space_dim = int(dim or 0)
        if self.space_dim > 0:
            self._log(
                f"  🔒 [{self.name}] 임베딩 공간 {self.space_dim}차원으로 고정"
                + (f" (기준 '{owner}')" if owner else "")
                + " — 다른 차원의 제공자는 사용하지 않습니다."
            )

    def register(self, name: str, fn: Callable[[List[str]], np.ndarray], priority: int = 0):
        self._providers.append((priority, name, fn))
        self._providers.sort(key=lambda t: t[0], reverse=True)
        if not self.active:
            self.active = name

    def unregister(self, name: str) -> bool:
        before = len(self._providers)
        self._providers = [t for t in self._providers if t[1] != name]
        if self.active == name:
            self.active = self._providers[0][1] if self._providers else ""
        return len(self._providers) != before

    def promote(self, name: str, priority: int = 80) -> bool:
        found = False
        rebuilt: List[tuple] = []
        for prio, n, fn in self._providers:
            if n == name:
                rebuilt.append((int(priority), n, fn))
                found = True
            else:
                rebuilt.append((prio, n, fn))
        if not found:
            return False
        rebuilt.sort(key=lambda t: t[0], reverse=True)
        self._providers = rebuilt
        self._log(f"  🔝 임베딩 제공자 우선순위 승격 → '{name}'")
        return True

    def _note_failure(self, name: str, err: Exception) -> None:
        cnt = self._failed.get(name, 0) + 1
        self._failed[name] = cnt
        if cnt <= 2:
            self._log(
                f"  ⚠ [{self.name}] 임베딩 제공자 '{name}' 실패 "
                f"({type(err).__name__}: {err})"
            )
            from . import diagnostics
            if diagnostics.enabled(2):
                import traceback
                for line in traceback.format_exc().splitlines()[-6:]:
                    self._log(f"      {line}")
        elif cnt == 3:
            self._log(
                f"  ⚠ [{self.name}] '{name}' 반복 실패 — 이후 동일 오류는 "
                f"요약만 남깁니다."
            )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        self.last_failed = False
        for _prio, name, fn in self._providers:
            try:
                mat = fn(items)
            except Exception as e:
                self._note_failure(name, e)
                continue
            if mat is None:
                continue
            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.shape[0] == 0:
                continue

            dim = int(mat.shape[-1])
            if self.dims.get(name) != dim:
                self.dims[name] = dim

            if self.space_dim and dim != self.space_dim:
                cnt = self._rejected.get(name, 0) + 1
                self._rejected[name] = cnt
                if cnt <= 2:
                    self._log(
                        f"  🚫 [{self.name}] '{name}' 공간 불일치 "
                        f"({dim}차원 ≠ 고정 {self.space_dim}차원) → 사용하지 않습니다."
                    )
                continue

            if self.active != name:
                self.active = name
                self._log(f"  🔀 [{self.name}] 임베딩 제공자 전환 → '{name}' (dim={dim})")
            return mat

        self._log(
            f"  ❌ [{self.name}] 사용할 수 있는 임베딩 제공자가 없습니다 "
            f"(등록 {len(self._providers)}개 / 실패 {sum(self._failed.values())}회 "
            f"/ 공간 거부 {sum(self._rejected.values())}회)"
        )
        self.last_failed = True
        if self.space_dim > 0:
            return np.zeros((len(items), self.space_dim), dtype=np.float32)
        return None

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def providers(self) -> List[str]:
        return [name for _p, name, _f in self._providers]

    def report_lines(self) -> List[str]:
        lines = [
            f"  🔀 [{self.name}] 활성 '{self.active or '-'}' "
            f"| 고정 공간 {self.space_dim or '자유'}차원"
        ]
        for prio, name, _fn in self._providers:
            lines.append(
                f"      · {name:<20} prio={prio:>4} "
                f"dim={self.dims.get(name, '?')} "
                f"실패={self._failed.get(name, 0)} "
                f"거부={self._rejected.get(name, 0)}"
            )
        return lines


def _alphaedge_available() -> bool:
    if not LLM_PATH.is_dir():
        return False
    if not (LLM_PATH / "config.json").exists():
        return False
    weights = [p for p in LLM_PATH.glob("*.safetensors") if p.is_file()]
    if not weights:
        weights = [p for p in LLM_PATH.glob("*.bin") if p.is_file()]
    return bool(weights)


def resolve_embedder_path(lang_code: str) -> tuple:
    if lang_model_ready("qwen3emb", lang_code):
        return str(lang_model_dir("qwen3emb", lang_code)), f"qwen3emb/{lang_code}"
    if _alphaedge_available():
        return str(LLM_PATH), "alphaedge-ai"
    return "", ""


def resolve_embedder_paths(codes: Sequence[str]) -> List[tuple]:
    out: List[tuple] = []
    seen = set()

    for code in codes:
        if not code or code in seen:
            continue
        seen.add(code)
        if lang_model_ready("qwen3emb", code):
            out.append((
                code,
                str(lang_model_dir("qwen3emb", code)),
                f"qwen3emb/{code}",
            ))

    if not out:
        for code in embedder_ready_codes():
            if code in seen:
                continue
            seen.add(code)
            out.append((
                code,
                str(lang_model_dir("qwen3emb", code)),
                f"qwen3emb/{code}",
            ))

    if not out and _alphaedge_available():
        out.append(("", str(LLM_PATH), "alphaedge-ai"))

    return out


def resolve_refiner_path(
    lang_code: str,
    codes: Optional[Sequence[str]] = None,
) -> tuple:
    wanted: List[str] = []
    for c in [lang_code] + list(codes or []):
        if c and c not in wanted:
            wanted.append(c)

    for code in wanted:
        if lang_model_ready("qwen35", code):
            return str(lang_model_dir("qwen35", code)), f"qwen35/{code}"

    if _alphaedge_available():
        return str(LLM_PATH), "alphaedge-ai"
    return "", ""


def script_of_ocr_lang(lang_code: str) -> str:
    from .model_manager import paddle_ocr_slug
    return {
        "korean": "Hangul",
        "": "Han",
        "en": "Latin",
        "latin": "Latin",
        "eslav": "Cyrillic",
        "cyrillic": "Cyrillic",
        "arabic": "Arabic",
        "devanagari": "Devanagari",
        "ta": "Tamil",
        "te": "Telugu",
        "el": "Greek",
        "th": "Thai",
    }.get(paddle_ocr_slug(lang_code), "")


def resolve_joint_path(
    lang_code: str = "",
    codes: Optional[Sequence[str]] = None,
) -> tuple:
    wanted: List[str] = []
    for c in [lang_code] + list(codes or []) + list(BOOTSTRAP_LANGUAGES):
        if c and c not in wanted:
            wanted.append(c)

    for code in wanted:
        if lang_model_ready("siglip2", code):
            return str(lang_model_dir("siglip2", code)), f"siglip2/{code}", code

    return "", "", ""