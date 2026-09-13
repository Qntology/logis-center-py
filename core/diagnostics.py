import os
from typing import Callable, Dict, Optional, Sequence

import numpy as np

try:
    import torch
except Exception:
    torch = None

DIAG_LEVEL = int(os.environ.get("NMS_DIAG", "1") or 0)

PROBE_POSITIVE = (
    ("commercial invoice number", "the invoice number printed under the title"),
    ("port of loading", "the departure port where the cargo is loaded"),
    ("total gross weight in kilograms", "gross weight kg of this shipment"),
)

PROBE_NEGATIVE = (
    "a photograph of a cat sleeping on a sofa",
    "assembly instructions for a wooden chair",
    "quarterly earnings call of a software company",
)

CONFIG_KEYS = (
    "model_type", "architectures", "hidden_size", "intermediate_size",
    "num_hidden_layers", "num_attention_heads", "vocab_size",
    "projection_size", "image_size", "patch_size", "max_position_embeddings",
    "torch_dtype", "d_model", "num_channels",
)


def enabled(level: int = 1) -> bool:
    return DIAG_LEVEL >= int(level)


def _emit(emit: Optional[Callable[[str], None]], msg: str) -> None:
    if emit is None:
        print(msg, flush=True)
        return
    try:
        emit(msg)
    except Exception:
        pass


def _flatten_config(cfg, prefix: str = "") -> Dict[str, object]:
    out: Dict[str, object] = {}
    if cfg is None:
        return out
    try:
        keys = list(vars(cfg).keys())
    except Exception:
        keys = []
    for k in keys:
        v = getattr(cfg, k, None)
        if v is None:
            continue
        if hasattr(v, "to_dict") or hasattr(v, "__dict__") and not isinstance(
            v, (str, int, float, bool, list, tuple, dict)
        ):
            out.update(_flatten_config(v, prefix=f"{prefix}{k}."))
            continue
        if k in CONFIG_KEYS:
            out[f"{prefix}{k}"] = v
    return out


def describe_config(cfg, title: str, emit=None, level: int = 1) -> Dict[str, object]:
    flat = _flatten_config(cfg)
    if not enabled(level):
        return flat
    if not flat:
        _emit(emit, f"  🔧 [{title}] 설정을 읽지 못했습니다.")
        return flat
    items = sorted(flat.items())
    _emit(emit, f"  🔧 [{title}] 설정 {len(items)}항목")
    row: list = []
    for k, v in items:
        row.append(f"{k}={v}")
        if len(row) == 4:
            _emit(emit, "      " + " | ".join(row))
            row = []
    if row:
        _emit(emit, "      " + " | ".join(row))
    return flat


def describe_module(model, title: str, emit=None, top: int = 6, level: int = 1) -> dict:
    info = {"tensors": 0, "params": 0, "dtypes": {}, "devices": {}, "bad": []}
    try:
        params = list(model.named_parameters())
    except Exception as e:
        _emit(emit, f"  🔍 [{title}] 파라미터 조회 실패: {e}")
        return info

    for name, p in params:
        info["tensors"] += 1
        try:
            info["params"] += int(p.numel())
        except Exception:
            continue
        d = str(getattr(p, "dtype", "")).replace("torch.", "")
        info["dtypes"][d] = info["dtypes"].get(d, 0) + 1
        dev = str(getattr(p, "device", ""))
        info["devices"][dev] = info["devices"].get(dev, 0) + 1

    if torch is not None:
        for name, p in params[:32]:
            try:
                if not bool(torch.isfinite(p.detach()).all().item()):
                    info["bad"].append(name)
            except Exception:
                continue

    if enabled(level):
        _emit(
            emit,
            f"  🔍 [{title}] 텐서 {info['tensors']}개 "
            f"| 파라미터 {info['params']:,} ({info['params'] / 1e6:.1f}M) "
            f"| dtype {info['dtypes']} | device {info['devices']}"
        )
        if info["bad"]:
            _emit(emit, f"      ⚠ NaN/Inf 파라미터: {', '.join(info['bad'][:4])}")

    if enabled(2):
        for name, p in params[:top]:
            _emit(
                emit,
                f"      ├─ {name}: {list(p.shape)} "
                f"{str(p.dtype).replace('torch.', '')}"
            )
        if len(params) > top:
            _emit(emit, f"      └─ … 그 외 {len(params) - top}개")

    return info


def describe_tensor(name: str, t, emit=None, level: int = 1) -> None:
    if not enabled(level) or t is None:
        return
    if torch is not None and torch.is_tensor(t):
        try:
            f = t.detach().float()
            finite = torch.isfinite(f)
            bad = int((~finite).sum().item())
            g = f[finite] if bad else f
            if g.numel() == 0:
                _emit(emit, f"    · {name}: 전부 비정상 값입니다.")
                return
            _emit(
                emit,
                f"    · {name}: shape={tuple(t.shape)} "
                f"dtype={str(t.dtype).replace('torch.', '')} dev={t.device} "
                f"min={float(g.min()):+.4f} max={float(g.max()):+.4f} "
                f"mean={float(g.mean()):+.4f} std={float(g.std()):.4f}"
                + (f" ⚠ 비정상 {bad}개" if bad else "")
            )
            return
        except Exception as e:
            _emit(emit, f"    · {name}: 텐서 요약 실패({e})")
            return
    describe_matrix(name, t, emit, level)


def describe_matrix(name: str, mat, emit=None, level: int = 1) -> None:
    if not enabled(level) or mat is None:
        return
    try:
        a = np.asarray(mat, dtype=np.float32)
    except Exception as e:
        _emit(emit, f"    · {name}: 배열 변환 실패({e})")
        return
    if a.size == 0:
        _emit(emit, f"    · {name}: 비어 있음")
        return

    finite = np.isfinite(a)
    bad = int((~finite).sum())
    g = a[finite]
    if g.size == 0:
        _emit(emit, f"    · {name}: 전부 비정상 값입니다.")
        return

    if a.ndim >= 2:
        norms = np.linalg.norm(a.reshape(a.shape[0], -1), axis=1)
    else:
        norms = np.asarray([float(np.linalg.norm(a))], dtype=np.float32)

    _emit(
        emit,
        f"    · {name}: shape={tuple(a.shape)} dtype={a.dtype} "
        f"min={float(g.min()):+.4f} max={float(g.max()):+.4f} "
        f"mean={float(g.mean()):+.4f} std={float(g.std()):.4f} "
        f"|L2| 중앙 {float(np.median(norms)):.4f}"
        + (f" ⚠ 비정상 {bad}개" if bad else "")
    )


def probe_space(label: str, encode_fn, emit=None, level: int = 1) -> dict:
    out = {"label": label, "ok": False, "dim": 0,
           "pos": 0.0, "neg": 0.0, "margin": 0.0, "verdict": ""}
    if not enabled(level):
        return out

    texts: list = []
    for a, b in PROBE_POSITIVE:
        texts.extend([a, b])
    texts.extend(PROBE_NEGATIVE)

    try:
        mat = encode_fn(texts)
    except Exception as e:
        _emit(emit, f"  🧪 [{label}] 공간 자가검진 실패: {e}")
        return out

    if mat is None:
        _emit(emit, f"  🧪 [{label}] 공간 자가검진 — 인코더가 None 을 반환했습니다.")
        return out

    m = np.asarray(mat, dtype=np.float32)
    if m.ndim == 1:
        m = m.reshape(1, -1)
    if m.shape[0] < len(texts):
        _emit(
            emit,
            f"  🧪 [{label}] 공간 자가검진 — 반환 행 {m.shape[0]} < 요청 {len(texts)}"
        )
        return out

    m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-8)
    k = len(PROBE_POSITIVE)

    pos = [float(m[2 * i] @ m[2 * i + 1]) for i in range(k)]
    negs = m[2 * k:]
    neg = [
        float(m[2 * i] @ negs[j])
        for i in range(k) for j in range(negs.shape[0])
    ]

    p = sum(pos) / max(1, len(pos))
    n = sum(neg) / max(1, len(neg))
    margin = p - n

    verdict = "양호" if margin >= 0.10 else ("빈약" if margin > 0.02 else "붕괴")
    out.update(ok=True, dim=int(m.shape[-1]), pos=p, neg=n,
               margin=margin, verdict=verdict)

    _emit(
        emit,
        f"  🧪 [{label}] 공간 자가검진 dim={out['dim']} "
        f"| 동의쌍 {p:+.4f} | 무관쌍 {n:+.4f} | 변별폭 {margin:+.4f} → {verdict}"
    )
    if verdict == "붕괴":
        _emit(
            emit,
            "     ⚠ 이 인코더는 의미가 다른 문장을 구분하지 못합니다. "
            "필드 친화도 맵이 잡음이 됩니다."
        )
    if enabled(2):
        for i, (a, b) in enumerate(PROBE_POSITIVE):
            _emit(emit, f"      · '{a[:28]}' ↔ '{b[:28]}' = {pos[i]:+.4f}")
    return out


def describe_joint_space(
    patch_matrix,
    anchors: Dict[str, np.ndarray],
    emit=None,
    title: str = "조인트 공간",
    sample: int = 48,
    level: int = 1,
) -> dict:
    out = {"ok": False}
    if not enabled(level):
        return out

    P = np.asarray(patch_matrix, dtype=np.float32)
    names = [k for k in list(anchors.keys())[:sample]]
    if P.size == 0 or not names:
        _emit(emit, f"  🔬 [{title}] 비교할 패치 또는 앵커가 없습니다.")
        return out

    A = np.stack([
        np.asarray(anchors[n], dtype=np.float32).reshape(-1) for n in names
    ])
    if int(A.shape[-1]) != int(P.shape[-1]):
        _emit(
            emit,
            f"  🔬 [{title}] 차원 불일치 — 패치 {int(P.shape[-1])} vs "
            f"앵커 {int(A.shape[-1])}"
        )
        return out

    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-8)
    An = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-8)
    S = An @ Pn.T
    peak = S.max(axis=1)

    _emit(
        emit,
        f"  🔬 [{title}] 패치 {P.shape[0]}개 × 앵커 {A.shape[0]}구 "
        f"| 코사인 min {float(S.min()):+.4f} / 평균 {float(S.mean()):+.4f} "
        f"/ max {float(S.max()):+.4f} / σ {float(S.std()):.4f}"
    )
    _emit(
        emit,
        f"     앵커별 최고 코사인 — 중앙 {float(np.median(peak)):+.4f} "
        f"| 최저 {float(peak.min()):+.4f} | 최고 {float(peak.max()):+.4f} "
        f"| 앵커 간 산포 {float(peak.std()):.4f}"
    )
    if float(peak.std()) < 0.01:
        _emit(
            emit,
            "     ⚠ 앵커마다 최고 코사인이 사실상 같습니다 — "
            "필드 변별이 물리적으로 불가능합니다."
        )

    out.update(
        ok=True,
        mean=float(S.mean()),
        std=float(S.std()),
        peak_spread=float(peak.std()),
    )
    return out