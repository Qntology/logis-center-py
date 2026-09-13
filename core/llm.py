from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .device import configure_backends, detect_accelerator, select_dtype
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
    "string", "number", "boolean", "array", "object",
    "value", "val", "field", "text", "key", "item", "raw",
    "cleaned value", "empty string", "cleaned value or empty string",
    "copy from ocr text or empty string",
})


def _load_with_dtype(loader, path, dtype, **kwargs):
    try:
        return loader.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return loader.from_pretrained(path, torch_dtype=dtype, **kwargs)


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

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = _load_with_dtype(
            AutoModel, self.model_path, self.dtype, trust_remote_code=True
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
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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

        self.device, self.accel_label = detect_accelerator(device)
        self.dtype = select_dtype(self.device)
        configure_backends(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        kwargs = {"trust_remote_code": True}

        if self.device.type == "cuda":
            if self.low_vram:
                quant = self._try_quant_config()
                if quant is not None:
                    kwargs["quantization_config"] = quant
                    kwargs["device_map"] = "auto"
                    self.load_mode = "4bit"
                else:
                    kwargs["device_map"] = "auto"
                    kwargs["low_cpu_mem_usage"] = True
                    if self.budget_gb > 0:
                        cap = max(1.0, self.budget_gb - 0.8)
                        kwargs["max_memory"] = {0: f"{cap:.1f}GiB", "cpu": "16GiB"}
                    self.load_mode = "offload"
            else:
                kwargs["device_map"] = "auto"

        self.model = _load_with_dtype(
            AutoModelForCausalLM, self.model_path, self.dtype, **kwargs
        )

        if self.device.type != "cuda" and self.load_mode == "standard":
            self.model.to(self.device)
        self.model.eval()

        placement = ""
        dev_map = getattr(self.model, "hf_device_map", None)
        if isinstance(dev_map, dict):
            devs = sorted(set(str(v) for v in dev_map.values()))
            placement = " | 배치: " + ", ".join(devs)
            if any(d in ("cpu", "disk") for d in devs):
                self._log(f"  ⚠ [{self.label}] 일부 레이어가 CPU/디스크로 오프로드되었습니다.")

        self._log(
            f"  ✅ [{self.label}] 정제 LLM 로드 "
            f"({self.load_mode} | {self.dtype}{placement})"
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

    def _try_quant_config(self):
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            self._log(
                f"  ⏭ [{self.label}] bitsandbytes 미설치 → 4bit 양자화 대신 오프로드를 씁니다.\n"
                f"     설치하면 4GB VRAM 에서도 안정적으로 동작합니다: "
                f"pip install bitsandbytes"
            )
            return None

        try:
            from transformers import BitsAndBytesConfig
            import torch as _t
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=_t.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            self._log(f"  ⏭ [{self.label}] 양자화 설정 실패({e}) → 오프로드로 진행합니다.")
            return None

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = prompt

        enc = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        gen_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "do_sample": False,
        }
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
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

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
        return s.strip()

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

    def refine_category(
        self,
        category: str,
        field_specs: Dict[str, str],
        raw_text: str,
        claimed: Optional[Dict[str, str]] = None,
        label_bank: Optional[Sequence[str]] = None,
        hint: str = "",
    ) -> Dict[str, str]:
        import json

        body = str(raw_text or "").strip()
        if not body or not field_specs:
            return {}

        lines = [f'  "{k}": <{v or "value"} or null>' for k, v in field_specs.items()]
        banned = ""
        if claimed:
            banned = (
                "ALREADY CLAIMED (do NOT return these values again):\n"
                + "\n".join(f"  - {v}" for v in list(claimed.values())[:20])
                + "\n\n"
            )

        prompt = (
            "You extract structured fields from noisy OCR text of one region "
            "of a business document.\n"
            "Copy values verbatim from the OCR TEXT. Never invent a value.\n"
            "Printed form labels are NOT values. If a field is absent, use null.\n"
            "Return ONLY a JSON object, no markdown, no reasoning.\n\n"
            f"REGION: {category}\n"
            + (f"HINT: {hint}\n" if hint else "")
            + banned
            + f"OCR TEXT:\n{body}\n\n"
            "SCHEMA: {\n" + ",\n".join(lines) + "\n}"
        )

        budget = min(1024, 64 + 24 * len(field_specs))
        try:
            out = self.generate(prompt, max_new_tokens=budget)
        except Exception:
            return {}

        cleaned = self._strip_reasoning(out)
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            self._log(
                f"  🚫 [{self.label}] '{category}' 잘린 JSON 응답 폐기 "
                f"— OCR 원문을 유지합니다."
            )
            return {}

        try:
            parsed = json.loads(cleaned[start: end + 1])
        except Exception:
            self._log(f"  🚫 [{self.label}] '{category}' JSON 파싱 실패 응답 폐기")
            return {}
        if not isinstance(parsed, dict):
            return {}

        labels = {self._compact(t) for t in (label_bank or []) if t}
        claimed_c = {self._compact(v) for v in (claimed or {}).values() if v}
        cb = self._compact(body)

        kept: Dict[str, str] = {}
        echo = 0
        label_echo = 0
        halluc = 0
        dup = 0

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
                    f"\"{v[:36]}\" 는 이미 다른 축이 확정한 값입니다."
                )
                continue
            if cv not in cb:
                halluc += 1
                continue
            kept[key] = v

        self._log(
            f"    ✅ [{category}] 신규 {len(kept)}건 | 스키마 에코 폐기 {echo}건 "
            f"| 라벨 에코 {label_echo}건 | 환각 {halluc}건 | 중복 {dup}건"
        )
        return kept

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
        )

        try:
            out = self.generate(prompt, max_new_tokens=128)
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
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EmbeddingRouter:
    def __init__(self, log=None, name: str = "router"):
        self._providers: List[tuple] = []
        self._log_fn = log or (lambda m: None)
        self.active = ""
        self.name = str(name)
        self.space_dim = 0
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
        return np.zeros((len(items), max(1, self.space_dim)), dtype=np.float32)

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