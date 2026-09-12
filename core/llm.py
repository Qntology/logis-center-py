from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .device import configure_backends, detect_accelerator, select_dtype
from .model_manager import (
    LLM_PATH,
    MissingModelError,
    ensure_lang_model_dir,
    ensure_model_dir,
    lang_model_dir,
    lang_model_ready,
)


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
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = str(model_path)
        self.label = label
        self.max_new_tokens = int(max_new_tokens)
        self._log_fn = log or (lambda m: None)

        self.device, self.accel_label = detect_accelerator(device)
        self.dtype = select_dtype(self.device)
        configure_backends(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.model = _load_with_dtype(
            AutoModelForCausalLM,
            self.model_path,
            self.dtype,
            trust_remote_code=True,
            device_map="auto" if self.device.type == "cuda" else None,
        )
        if self.device.type != "cuda":
            self.model.to(self.device)
        self.model.eval()

        self._log(f"  ✅ [{self.label}] 정제 LLM 로드 ({self.model_path} | {self.dtype})")

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
        out = self.model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens or self.max_new_tokens),
            do_sample=False,
        )
        gen = out[0][len(enc["input_ids"][0]):]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def refine_field(self, field_name: str, raw_text: str, hint: str = "") -> dict:
        import json

        prompt = (
            "You extract one field value from noisy OCR text of a business document.\n"
            "Return ONLY a JSON object, no markdown, no explanation.\n\n"
            f"FIELD: {field_name}\n"
            + (f"HINT: {hint}\n" if hint else "")
            + f"OCR TEXT:\n{raw_text}\n\n"
            'SCHEMA: {"value": "<cleaned value or empty string>"}'
        )

        try:
            out = self.generate(prompt, max_new_tokens=128)
        except Exception:
            return {}

        cleaned = out.replace("```json", "").replace("```", "").strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {"value": cleaned}
        try:
            return json.loads(cleaned[start: end + 1])
        except Exception:
            return {"value": cleaned}

    def unload(self):
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class EmbeddingRouter:
    def __init__(self, log=None):
        self._providers: List[tuple] = []
        self._log_fn = log or (lambda m: None)
        self.active = ""

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def register(self, name: str, fn: Callable[[List[str]], np.ndarray], priority: int = 0):
        self._providers.append((priority, name, fn))
        self._providers.sort(key=lambda t: t[0], reverse=True)
        if not self.active:
            self.active = name

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        items = list(texts)
        for _prio, name, fn in self._providers:
            try:
                mat = fn(items)
            except Exception as e:
                self._log(f"  ⚠ 임베딩 제공자 '{name}' 실패: {e}")
                continue
            if mat is None:
                continue
            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
            if mat.shape[0] == 0:
                continue
            if self.active != name:
                self.active = name
                self._log(f"  🔀 임베딩 제공자 전환 → '{name}'")
            return mat
        return np.zeros((len(items), 1), dtype=np.float32)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def providers(self) -> List[str]:
        return [name for _p, name, _f in self._providers]


def resolve_embedder_path(lang_code: str) -> tuple:
    if lang_model_ready("qwen3emb", lang_code):
        return str(lang_model_dir("qwen3emb", lang_code)), f"qwen3emb/{lang_code}"
    if LLM_PATH.is_dir() and (LLM_PATH / "config.json").exists():
        return str(LLM_PATH), "alphaedge-ai"
    return "", ""


def resolve_refiner_path(lang_code: str) -> tuple:
    if lang_model_ready("qwen35", lang_code):
        return str(lang_model_dir("qwen35", lang_code)), f"qwen35/{lang_code}"
    if LLM_PATH.is_dir() and (LLM_PATH / "config.json").exists():
        return str(LLM_PATH), "alphaedge-ai"
    return "", ""