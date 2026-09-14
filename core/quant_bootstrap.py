import importlib
import os
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Optional, Tuple

MIN_VERSION = "0.43.0"
PACKAGE = f"bitsandbytes>={MIN_VERSION}"

ENV_AUTO = "NMS_QUANT_AUTOINSTALL"
ENV_MODE = "NMS_QUANT_MODE"
ENV_TIMEOUT = "NMS_QUANT_TIMEOUT"

DEFAULT_TIMEOUT = 600

MODE_AUTO = "auto"
MODE_4BIT = "4bit"
MODE_8BIT = "8bit"
MODE_OFF = "off"

PROBE_DIM = 64
PROBE_BATCH = 2

_lock = threading.RLock()
_state: Dict[str, object] = {
    "attempted": False,
    "tested": False,
    "installed": False,
    "version": "",
    "capability": "",
    "reason": "",
    "selftest_4bit": None,
    "selftest_8bit": None,
    "compute_dtype": "",
}


def _log_to(log: Optional[Callable[[str], None]], msg: str):
    if log is None:
        return
    try:
        log(msg)
    except Exception:
        pass


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def in_venv() -> bool:
    return (
        hasattr(sys, "real_prefix")
        or (getattr(sys, "base_prefix", sys.prefix) != sys.prefix)
        or bool(os.environ.get("VIRTUAL_ENV"))
    )


def requested_mode() -> str:
    raw = str(os.environ.get(ENV_MODE, MODE_AUTO)).strip().lower()
    if raw in (MODE_AUTO, MODE_4BIT, MODE_8BIT, MODE_OFF):
        return raw
    return MODE_AUTO


def auto_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = str(os.environ.get(ENV_AUTO, "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def device_capability() -> Tuple[int, int]:
    try:
        import torch
        if not torch.cuda.is_available():
            return (0, 0)
        return tuple(int(v) for v in torch.cuda.get_device_capability(0))
    except Exception:
        return (0, 0)


def supports_bf16() -> bool:
    major, _minor = device_capability()
    if major >= 8:
        return True
    try:
        import torch
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def compute_dtype():
    try:
        import torch
    except Exception:
        return None
    return torch.bfloat16 if supports_bf16() else torch.float16


def compute_dtype_name() -> str:
    dt = compute_dtype()
    return str(dt).replace("torch.", "") if dt is not None else ""


def installed_version() -> str:
    if not has_module("bitsandbytes"):
        return ""
    try:
        import bitsandbytes
        return str(getattr(bitsandbytes, "__version__", "") or "")
    except Exception:
        return ""


def _version_tuple(raw: str) -> Tuple[int, ...]:
    parts: List[int] = []
    for chunk in str(raw or "").split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def version_ok(raw: str = "") -> bool:
    got = _version_tuple(raw or installed_version())
    want = _version_tuple(MIN_VERSION)
    return got >= want


def manual_command() -> str:
    return f'"{sys.executable}" -m pip install "{PACKAGE}"'


def _run_pip(
    args: List[str],
    log: Optional[Callable[[str], None]],
    timeout: int,
) -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    cmd.extend(args)
    _log_to(log, "     $ " + " ".join(cmd[2:]))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(timeout),
        )
    except subprocess.TimeoutExpired:
        return False, f"{timeout}초 안에 끝나지 않았습니다."
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    text = ""
    try:
        text = proc.stdout.decode("utf-8", errors="ignore")
    except Exception:
        text = ""

    if proc.returncode == 0:
        return True, ""

    tail = [ln for ln in text.splitlines() if ln.strip()][-6:]
    return False, "\n".join(tail) if tail else f"pip 종료코드 {proc.returncode}"


def _probe_4bit(log: Optional[Callable[[str], None]]) -> Tuple[bool, str]:
    try:
        import torch
        import bitsandbytes as bnb
    except Exception as e:
        return False, f"임포트 실패: {type(e).__name__}: {e}"

    dtype = compute_dtype()
    layer = None
    try:
        layer = bnb.nn.Linear4bit(
            PROBE_DIM,
            PROBE_DIM,
            bias=False,
            compute_dtype=dtype,
            quant_type="nf4",
        )
        layer = layer.to("cuda")
        x = torch.randn(PROBE_BATCH, PROBE_DIM, device="cuda", dtype=dtype)
        with torch.no_grad():
            y = layer(x)
        if not bool(torch.isfinite(y.float()).all().item()):
            return False, "출력에 NaN/Inf 가 섞였습니다."
        if tuple(y.shape) != (PROBE_BATCH, PROBE_DIM):
            return False, f"출력 형상 이상 {tuple(y.shape)}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        layer = None
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

    return True, ""


def _probe_8bit(log: Optional[Callable[[str], None]]) -> Tuple[bool, str]:
    try:
        import torch
        import bitsandbytes as bnb
    except Exception as e:
        return False, f"임포트 실패: {type(e).__name__}: {e}"

    layer = None
    try:
        layer = bnb.nn.Linear8bitLt(
            PROBE_DIM, PROBE_DIM, bias=False, has_fp16_weights=False
        )
        layer = layer.to("cuda")
        x = torch.randn(
            PROBE_BATCH, PROBE_DIM, device="cuda", dtype=torch.float16
        )
        with torch.no_grad():
            y = layer(x)
        if not bool(torch.isfinite(y.float()).all().item()):
            return False, "출력에 NaN/Inf 가 섞였습니다."
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:160]}"
    finally:
        layer = None
        try:
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()
        except Exception:
            pass

    return True, ""


def selftest(
    log: Optional[Callable[[str], None]] = None,
    force: bool = False,
) -> str:
    with _lock:
        if _state["tested"] and not force:
            return str(_state["capability"] or "")
        _state["tested"] = True

    if not cuda_available():
        _state["capability"] = ""
        _state["reason"] = "CUDA 를 쓸 수 없습니다."
        return ""

    if not has_module("bitsandbytes"):
        _state["capability"] = ""
        _state["reason"] = "bitsandbytes 가 없습니다."
        return ""

    mode = requested_mode()
    ver = installed_version()
    major, minor = device_capability()

    _log_to(
        log,
        f"  🧪 [QUANT] bitsandbytes {ver or '?'} 자가검진 — "
        f"sm_{major}{minor} | 계산 dtype {compute_dtype_name()}",
    )

    if mode == MODE_OFF:
        _state["capability"] = ""
        _state["reason"] = f"{ENV_MODE}=off 로 비활성화되었습니다."
        _log_to(log, f"  ⏭ [QUANT] {_state['reason']}")
        return ""

    if mode in (MODE_AUTO, MODE_4BIT):
        ok, why = _probe_4bit(log)
        _state["selftest_4bit"] = bool(ok)
        if ok:
            _state["capability"] = MODE_4BIT
            _state["reason"] = ""
            _state["compute_dtype"] = compute_dtype_name()
            _log_to(
                log,
                f"  ✅ [QUANT] 4bit(NF4) 자가검진 통과 — "
                f"Linear4bit 순전파 정상",
            )
            return MODE_4BIT
        _log_to(log, f"  ⚠ [QUANT] 4bit 자가검진 실패 — {why}")
        if mode == MODE_4BIT:
            _state["capability"] = ""
            _state["reason"] = f"4bit 강제 모드인데 실패: {why}"
            return ""

    if mode in (MODE_AUTO, MODE_8BIT):
        ok, why = _probe_8bit(log)
        _state["selftest_8bit"] = bool(ok)
        if ok:
            _state["capability"] = MODE_8BIT
            _state["reason"] = "4bit 실패로 8bit 로 내려갔습니다."
            _state["compute_dtype"] = "float16"
            _log_to(
                log,
                "  ✅ [QUANT] 8bit(LLM.int8) 자가검진 통과 — 4bit 대신 "
                "8bit 로 진행합니다. 메모리 절감폭은 절반입니다.",
            )
            return MODE_8BIT
        _log_to(log, f"  ⚠ [QUANT] 8bit 자가검진 실패 — {why}")

    _state["capability"] = ""
    _state["reason"] = "4bit/8bit 모두 동작하지 않습니다."
    _log_to(
        log,
        "  ❌ [QUANT] bitsandbytes 가 설치되어 있지만 실제로 동작하지 "
        "않습니다. CUDA 바이너리가 PyTorch 와 맞지 않을 수 있습니다.\n"
        "     재설치: pip install --force-reinstall "
        f'"{PACKAGE}"',
    )
    return ""


def ensure_bitsandbytes(
    log: Optional[Callable[[str], None]] = None,
    auto: Optional[bool] = None,
    force: bool = False,
) -> Dict[str, object]:
    with _lock:
        if _state["attempted"] and not force:
            return status()
        _state["attempted"] = True

    if force:
        with _lock:
            _state["tested"] = False
            _state["capability"] = ""

    if requested_mode() == MODE_OFF:
        _state["tested"] = True
        _state["reason"] = f"{ENV_MODE}=off"
        _log_to(log, f"  ⏭ [QUANT] 양자화가 꺼져 있습니다 ({ENV_MODE}=off).")
        return status()

    if not cuda_available():
        _state["tested"] = True
        _state["reason"] = "CUDA 없음 — 양자화가 필요하지 않습니다."
        _log_to(log, f"  ⏭ [QUANT] {_state['reason']}")
        return status()

    ver = installed_version()

    if ver and version_ok(ver):
        _state["installed"] = True
        _state["version"] = ver
        selftest(log=log)
        return status()

    if ver and not version_ok(ver):
        _log_to(
            log,
            f"  ⬆ [QUANT] bitsandbytes {ver} 는 최소 요구 {MIN_VERSION} 보다 "
            f"낮습니다. 업그레이드합니다.",
        )
    else:
        _log_to(
            log,
            f"  📦 [QUANT] 4bit 양자화 백엔드를 자동 설치합니다 "
            f"({PACKAGE})",
        )

    if not auto_enabled(auto):
        _state["reason"] = "자동 설치가 꺼져 있습니다."
        _log_to(
            log,
            f"  ⏭ [QUANT] 자동 설치 비활성 ({ENV_AUTO}=0). 수동 설치:\n"
            f"     {manual_command()}",
        )
        return status()

    if not in_venv():
        _log_to(
            log,
            "  ⚠ [QUANT] 가상환경 밖에서 실행 중입니다. "
            "setup_and_run.bat 으로 실행하면 venv 안에 설치됩니다.",
        )

    timeout = DEFAULT_TIMEOUT
    try:
        timeout = int(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT))
    except Exception:
        timeout = DEFAULT_TIMEOUT

    args = [PACKAGE]
    if ver:
        args.insert(0, "--upgrade")

    ok, err = _run_pip(args, log, timeout)
    if not ok:
        _state["reason"] = f"설치 실패: {err}"
        _log_to(log, f"  ❌ [QUANT] 설치 실패\n     {err}")
        _log_to(log, f"     수동 설치: {manual_command()}")
        return status()

    importlib.invalidate_caches()
    for name in list(sys.modules.keys()):
        if name == "bitsandbytes" or name.startswith("bitsandbytes."):
            sys.modules.pop(name, None)

    new_ver = installed_version()
    _state["installed"] = bool(new_ver)
    _state["version"] = new_ver

    if not new_ver:
        _state["reason"] = "설치 후에도 모듈을 찾지 못했습니다."
        _log_to(log, f"  ❌ [QUANT] {_state['reason']}")
        return status()

    _log_to(log, f"  ✅ [QUANT] bitsandbytes {new_ver} 설치 완료")
    selftest(log=log)
    return status()


def capability(log: Optional[Callable[[str], None]] = None) -> str:
    with _lock:
        if _state["tested"]:
            return str(_state["capability"] or "")
        if _state["attempted"]:
            return ""
    ensure_bitsandbytes(log=log)
    return str(_state.get("capability") or "")


def build_config(log: Optional[Callable[[str], None]] = None):
    mode = capability(log=log)
    if not mode:
        return None, ""

    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:
        _log_to(log, f"  ⏭ [QUANT] BitsAndBytesConfig 임포트 실패 ({e})")
        return None, ""

    dtype = compute_dtype()

    try:
        if mode == MODE_4BIT:
            cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_storage=dtype,
            )
            return cfg, MODE_4BIT

        cfg = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        return cfg, MODE_8BIT
    except TypeError:
        try:
            if mode == MODE_4BIT:
                cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                return cfg, MODE_4BIT
            return BitsAndBytesConfig(load_in_8bit=True), MODE_8BIT
        except Exception as e:
            _log_to(log, f"  ⏭ [QUANT] 설정 생성 실패 ({e})")
            return None, ""
    except Exception as e:
        _log_to(log, f"  ⏭ [QUANT] 설정 생성 실패 ({e})")
        return None, ""


def stage_ratio() -> float:
    mode = str(_state.get("capability") or "")
    if mode == MODE_4BIT:
        return 0.35
    if mode == MODE_8BIT:
        return 0.60
    return 1.0


def vram_ratio() -> float:
    mode = str(_state.get("capability") or "")
    if mode == MODE_4BIT:
        return 0.42
    if mode == MODE_8BIT:
        return 0.70
    return 1.15


def status() -> Dict[str, object]:
    with _lock:
        out = dict(_state)
    out["present"] = has_module("bitsandbytes")
    out["version"] = installed_version()
    out["version_ok"] = version_ok(out["version"])
    out["cuda"] = cuda_available()
    out["sm"] = "sm_{}{}".format(*device_capability())
    out["bf16"] = supports_bf16()
    out["mode_requested"] = requested_mode()
    out["stage_ratio"] = stage_ratio()
    out["vram_ratio"] = vram_ratio()
    out["verified"] = bool(out.get("tested"))
    return out


def report_lines() -> List[str]:
    s = status()

    if not s["cuda"]:
        return ["  [--  ] 양자화 불필요 — CPU 모드"]

    cap = str(s.get("capability") or "")
    mark = "OK " if cap else "MISS"

    out = [
        f"  [{mark}] 양자화 백엔드 | bitsandbytes {s['version'] or '없음'} "
        f"| {s['sm']} | bf16 {'지원' if s['bf16'] else '미지원'} "
        f"| 모드 {cap or '비활성'}"
    ]

    if cap:
        out.append(
            f"         계산 dtype {s.get('compute_dtype') or '-'} "
            f"| VRAM 배율 {s['vram_ratio']:.2f} "
            f"| RAM 스테이징 배율 {s['stage_ratio']:.2f}"
        )
    else:
        if s.get("reason"):
            out.append(f"         사유: {s['reason']}")
        out.append(f"         설치: {manual_command()}")

    return out