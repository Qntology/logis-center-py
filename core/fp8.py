import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

FP8_E4M3 = getattr(torch, "float8_e4m3fn", None)
FP8_E5M2 = getattr(torch, "float8_e5m2", None)

E4M3_MAX = 448.0
E5M2_MAX = 57344.0

MIN_FP8_NUMEL = 262144


def fp8_storage_available() -> bool:
    return FP8_E4M3 is not None


def fp8_native_matmul() -> bool:
    if not torch.cuda.is_available() or FP8_E4M3 is None:
        return False
    try:
        major, minor = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return (major, minor) >= (8, 9)


def capability_report() -> str:
    if not torch.cuda.is_available():
        return "CPU — fp8 미적용"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        name = torch.cuda.get_device_name(0)
    except Exception:
        return "GPU 정보를 읽지 못했습니다."
    native = fp8_native_matmul()
    return (
        f"{name} (sm_{major}{minor}) | fp8 dtype "
        f"{'있음' if fp8_storage_available() else '없음'} | "
        f"fp8 텐서코어 {'지원' if native else '미지원 → 저장 전용(업캐스트 연산)'}"
    )


def quantize_fp8(t: torch.Tensor, axis: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    src = t.detach().to(torch.float32)
    amax = src.abs().amax(dim=axis, keepdim=True).clamp_(min=1e-8)
    scale = (amax / E4M3_MAX).to(torch.float32)
    q = (src / scale).clamp_(-E4M3_MAX, E4M3_MAX).to(FP8_E4M3)
    return q, scale


def dequantize_fp8(q: torch.Tensor, scale: torch.Tensor, dtype) -> torch.Tensor:
    return (q.to(torch.float32) * scale).to(dtype)


class Fp8Linear(nn.Module):
    def __init__(self, src: nn.Linear):
        super().__init__()
        self.in_features = int(src.in_features)
        self.out_features = int(src.out_features)
        self.compute_dtype = src.weight.dtype

        q, scale = quantize_fp8(src.weight.data, axis=1)
        self.register_buffer("weight_fp8", q, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)

        if src.bias is not None:
            self.register_buffer(
                "bias", src.bias.data.detach().clone(), persistent=False
            )
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = dequantize_fp8(self.weight_fp8, self.weight_scale, x.dtype)
        return torch.nn.functional.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"storage=float8_e4m3fn"
        )


def convert_linear_to_fp8(
    root: nn.Module,
    skip_names: Sequence[str] = (),
    min_numel: int = MIN_FP8_NUMEL,
    log=None,
) -> Dict[str, object]:
    stats = {"converted": 0, "skipped": 0, "saved_bytes": 0, "names": []}
    if not fp8_storage_available():
        if log is not None:
            log(
                "  ⏭ [FP8] torch.float8_e4m3fn 이 없어 fp8 저장을 적용하지 "
                "않습니다. (PyTorch 2.1 이상 필요)"
            )
        return stats

    skip = tuple(skip_names or ())
    targets: List[Tuple[nn.Module, str, nn.Linear]] = []

    for mod_name, module in root.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if any(s in full for s in skip):
                stats["skipped"] += 1
                continue
            if child.weight.numel() < int(min_numel):
                stats["skipped"] += 1
                continue
            targets.append((module, child_name, child))

    for parent, child_name, child in targets:
        try:
            before = child.weight.numel() * child.weight.element_size()
            new = Fp8Linear(child)
            setattr(parent, child_name, new)
            after = new.weight_fp8.numel() + new.weight_scale.numel() * 4
            stats["converted"] += 1
            stats["saved_bytes"] += max(0, before - after)
            stats["names"].append(child_name)
        except Exception:
            stats["skipped"] += 1

    if log is not None:
        log(
            f"  🧊 [FP8] Linear {stats['converted']}개 fp8(E4M3) 저장 전환 "
            f"| 건너뜀 {stats['skipped']}개 "
            f"| 절감 {stats['saved_bytes'] / 1e6:.0f} MB"
        )
        log(f"  🧊 [FP8] {capability_report()}")
    return stats


def plan_kv_cache(
    model,
    max_tokens: int,
    label: str = "",
    log=None,
) -> Dict[str, object]:
    cfg = getattr(model, "config", None)
    layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    heads_kv = int(
        getattr(cfg, "num_key_value_heads", 0)
        or getattr(cfg, "num_attention_heads", 0)
        or 0
    )
    hidden = int(getattr(cfg, "hidden_size", 0) or 0)
    heads = int(getattr(cfg, "num_attention_heads", 0) or 0)
    head_dim = int(getattr(cfg, "head_dim", 0) or 0)
    if head_dim <= 0 and heads > 0 and hidden > 0:
        head_dim = hidden // heads

    try:
        dtype = next(model.parameters()).dtype
        elem = torch.tensor([], dtype=dtype).element_size()
    except Exception:
        elem = 2

    per_token = 2 * layers * heads_kv * head_dim * elem
    need_mb = per_token * max(1, int(max_tokens)) / 1e6

    free_mb = 0.0
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            free_mb = (
                props.total_memory - torch.cuda.memory_reserved(0)
            ) / 1e6
        except Exception:
            free_mb = 0.0

    place = "vram" if free_mb > need_mb + 512 else "cpu"
    out = {
        "layers": layers,
        "kv_heads": heads_kv,
        "head_dim": head_dim,
        "elem": elem,
        "per_token": per_token,
        "need_mb": need_mb,
        "free_mb": free_mb,
        "place": place,
    }

    if log is not None:
        log(
            f"  📐 [KV-PLAN] {label or 'model'} | 층 {layers} × KV헤드 "
            f"{heads_kv} × head_dim {head_dim} × {elem}B "
            f"| 토큰당 {per_token}B | 최대 {max_tokens}토큰 = "
            f"{need_mb:.2f} MB | VRAM 여유 {free_mb:.0f} MB → {place}"
        )
    return out


class Fp8KVCacheAdapter:
    def __init__(self, label: str = "", log=None):
        self.label = label
        self._log_fn = log or (lambda m: None)
        self.enabled = False
        self.reason = ""

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def build(self):
        if not fp8_storage_available():
            self.reason = "torch.float8_e4m3fn 없음"
            self._log(f"  ⏭ [FP8-KV] {self.reason} → 기본 캐시 사용")
            return None
        try:
            from transformers.cache_utils import DynamicCache
        except Exception as e:
            self.reason = f"DynamicCache 임포트 실패({e})"
            self._log(f"  ⏭ [FP8-KV] {self.reason} → 기본 캐시 사용")
            return None

        outer = self

        class _Fp8Cache(DynamicCache):
            def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                dtype = key_states.dtype
                qk, sk = quantize_fp8(key_states, axis=-1)
                qv, sv = quantize_fp8(value_states, axis=-1)
                k = dequantize_fp8(qk, sk, dtype)
                v = dequantize_fp8(qv, sv, dtype)
                return super().update(k, v, layer_idx, cache_kwargs)

        try:
            cache = _Fp8Cache()
        except Exception as e:
            self.reason = f"캐시 생성 실패({e})"
            self._log(f"  ⏭ [FP8-KV] {self.reason} → 기본 캐시 사용")
            return None

        self.enabled = True
        self._log(
            f"  🧊 [FP8-KV] {self.label} KV 캐시를 E4M3 양자화-복원 경로로 "
            f"전환했습니다. (Ampere 는 텐서코어 미지원이라 정밀도 절감 효과만)"
        )
        return cache