import importlib
import os
import subprocess
import sys
import threading
from typing import Callable, Dict, List, Optional, Tuple

PADDLE_INDEX_CPU = "https://www.paddlepaddle.org.cn/packages/stable/cpu/"

PADDLE_INDEX_CUDA: Tuple[Tuple[Tuple[int, int], str], ...] = (
    ((12, 9), "https://www.paddlepaddle.org.cn/packages/stable/cu129/"),
    ((12, 6), "https://www.paddlepaddle.org.cn/packages/stable/cu126/"),
    ((11, 8), "https://www.paddlepaddle.org.cn/packages/stable/cu118/"),
)

PADDLE_TRUSTED_HOST = "www.paddlepaddle.org.cn"

PY_MIN = (3, 9)
PY_MAX = (3, 13)

ENV_AUTO = "NMS_PADDLE_AUTOINSTALL"
ENV_TIMEOUT = "NMS_PADDLE_TIMEOUT"
ENV_DEVICE = "NMS_PADDLE_DEVICE"
ENV_MKLDNN = "NMS_PADDLE_MKLDNN"
ENV_THREADS = "NMS_PADDLE_THREADS"

DEFAULT_TIMEOUT = 900

RUNTIME_FLAGS: Dict[str, str] = {
    "FLAGS_use_mkldnn": "0",
    "FLAGS_call_stack_level": "1",
    "GLOG_minloglevel": "2",
}

_runtime_applied: Dict[str, str] = {}

_lock = threading.RLock()
_state: Dict[str, object] = {
    "attempted": False,
    "ok": False,
    "paddle": False,
    "paddleocr": False,
    "reason": "",
    "index": "",
    "package": "",
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


def python_supported() -> bool:
    v = sys.version_info[:2]
    return PY_MIN <= v <= PY_MAX


def python_label() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def detect_cuda() -> Optional[Tuple[int, int]]:
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    raw = getattr(torch.version, "cuda", "") or ""
    parts = str(raw).split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def pick_target() -> Tuple[str, str, str]:
    cu = detect_cuda()
    if cu is None:
        return "paddlepaddle", PADDLE_INDEX_CPU, "CPU"

    best_url = ""
    best_key: Optional[Tuple[int, int]] = None
    for key, url in PADDLE_INDEX_CUDA:
        if key <= cu:
            if best_key is None or key > best_key:
                best_key = key
                best_url = url

    if best_key is None:
        oldest_key, oldest_url = PADDLE_INDEX_CUDA[-1]
        return (
            "paddlepaddle-gpu",
            oldest_url,
            f"CUDA {cu[0]}.{cu[1]} → cu{oldest_key[0]}{oldest_key[1]} (하위 호환)",
        )

    return (
        "paddlepaddle-gpu",
        best_url,
        f"CUDA {cu[0]}.{cu[1]} → cu{best_key[0]}{best_key[1]}",
    )


def manual_command() -> str:
    pkg, index, _label = pick_target()
    return (
        f'"{sys.executable}" -m pip install {pkg} '
        f"-i {index} --trusted-host {PADDLE_TRUSTED_HOST}\n"
        f'"{sys.executable}" -m pip install paddleocr'
    )


def _run_pip(
    args: List[str],
    log: Optional[Callable[[str], None]],
    timeout: int,
) -> Tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"] + args
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


_late_warned = {"done": False}


def configure_runtime(log: Optional[Callable[[str], None]] = None) -> Dict[str, str]:
    with _lock:
        fresh: Dict[str, str] = {}

        for key, val in RUNTIME_FLAGS.items():
            if os.environ.get(key) is None:
                os.environ[key] = val
                fresh[key] = val
                _runtime_applied[key] = val

        late = has_module("paddle") and "paddle" in sys.modules

        if late and fresh and not _late_warned["done"]:
            _late_warned["done"] = True
            _log_to(
                log,
                "  ⚠ [PADDLE] paddle 이 이미 로드된 뒤라 환경 플래그가 "
                "반영되지 않을 수 있습니다. 예측기 인자로 다시 강제합니다. "
                "(이 경고는 한 번만 표시합니다)",
            )

        if fresh:
            brief = " | ".join(f"{k}={v}" for k, v in fresh.items())
            _log_to(
                log,
                f"  ⚙ [PADDLE] 런타임 플래그 적용 — {brief}\n"
                f"     oneDNN 경로는 PP-OCRv5 det 의 double 배열 속성을 "
                f"변환하지 못해 기본으로 끕니다.",
            )

        return dict(_runtime_applied)


def device_preference() -> str:
    raw = str(os.environ.get(ENV_DEVICE, "") or "").strip()
    return raw if raw else "cpu"


def mkldnn_enabled() -> bool:
    raw = str(os.environ.get(ENV_MKLDNN, "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def cpu_threads() -> int:
    try:
        n = int(os.environ.get(ENV_THREADS, "0") or 0)
    except Exception:
        n = 0
    if n > 0:
        return n
    try:
        return max(1, min(8, int(os.cpu_count() or 4)))
    except Exception:
        return 4


def predictor_kwargs() -> Dict[str, object]:
    return {
        "device": device_preference(),
        "enable_mkldnn": mkldnn_enabled(),
        "cpu_threads": cpu_threads(),
    }


def kwargs_variants(base: Dict[str, object]) -> List[Dict[str, object]]:
    opts = predictor_kwargs()

    full = dict(base)
    full.update(opts)

    mid = dict(base)
    mid["device"] = opts["device"]
    mid["enable_mkldnn"] = opts["enable_mkldnn"]

    slim = dict(base)
    slim["enable_mkldnn"] = opts["enable_mkldnn"]

    out: List[Dict[str, object]] = []
    for cand in (full, mid, slim, dict(base)):
        if cand not in out:
            out.append(cand)
    return out


def describe_kwargs(kw: Dict[str, object]) -> str:
    bits = []
    if "device" in kw:
        bits.append(f"device={kw['device']}")
    if "enable_mkldnn" in kw:
        bits.append(f"mkldnn={'on' if kw['enable_mkldnn'] else 'off'}")
    if "cpu_threads" in kw:
        bits.append(f"threads={kw['cpu_threads']}")
    return " ".join(bits) if bits else "기본값"


def status() -> Dict[str, object]:
    with _lock:
        out = dict(_state)
        out["runtime_flags"] = dict(_runtime_applied)
    out["paddle_present"] = has_module("paddle")
    out["paddleocr_present"] = has_module("paddleocr")
    out["python"] = python_label()
    out["venv"] = in_venv()
    out["executable"] = sys.executable
    out["device"] = device_preference()
    out["mkldnn"] = mkldnn_enabled()
    return out


def ready() -> bool:
    return has_module("paddle") and has_module("paddleocr")


def auto_enabled(explicit: Optional[bool] = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    raw = str(os.environ.get(ENV_AUTO, "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def ensure_paddle(
    log: Optional[Callable[[str], None]] = None,
    auto: Optional[bool] = None,
    force: bool = False,
) -> Dict[str, object]:
    configure_runtime(log=log)

    with _lock:
        if ready():
            _state["ok"] = True
            _state["paddle"] = True
            _state["paddleocr"] = True
            return status()

        if _state["attempted"] and not force:
            return status()

        _state["attempted"] = True

    if not auto_enabled(auto):
        _state["reason"] = "자동 설치가 꺼져 있습니다."
        _log_to(
            log,
            f"  ⏭ [PADDLE] 자동 설치 비활성 ({ENV_AUTO}=0). 수동 설치:\n"
            + "\n".join(f"     {ln}" for ln in manual_command().splitlines()),
        )
        return status()

    if not python_supported():
        _state["reason"] = (
            f"Python {python_label()} 은 PaddlePaddle 지원 범위 "
            f"{PY_MIN[0]}.{PY_MIN[1]}~{PY_MAX[0]}.{PY_MAX[1]} 밖입니다."
        )
        _log_to(log, f"  ❌ [PADDLE] {_state['reason']}")
        return status()

    if not in_venv():
        _log_to(
            log,
            "  ⚠ [PADDLE] 가상환경 밖에서 실행 중입니다. "
            "setup_and_run.bat 으로 실행하면 venv 안에 설치됩니다.\n"
            f"     현재 인터프리터: {sys.executable}",
        )

    timeout = DEFAULT_TIMEOUT
    try:
        timeout = int(os.environ.get(ENV_TIMEOUT, DEFAULT_TIMEOUT))
    except Exception:
        timeout = DEFAULT_TIMEOUT

    pkg, index, label = pick_target()
    _state["index"] = index
    _state["package"] = pkg

    _log_to(
        log,
        f"  📦 [PADDLE] PP-OCRv5 실행 백엔드를 자동 설치합니다 — {label}\n"
        f"     PyPI 에는 paddlepaddle 이 없어 전용 인덱스를 씁니다: {index}",
    )

    if not has_module("paddle"):
        ok, err = _run_pip(
            [pkg, "-i", index, "--trusted-host", PADDLE_TRUSTED_HOST],
            log, timeout,
        )
        if not ok:
            _state["reason"] = f"{pkg} 설치 실패: {err}"
            _log_to(log, f"  ❌ [PADDLE] {pkg} 설치 실패\n     {err}")
            _log_to(
                log,
                "     수동 설치:\n"
                + "\n".join(f"     {ln}" for ln in manual_command().splitlines()),
            )
            return status()
        _state["paddle"] = True
        _log_to(log, f"  ✅ [PADDLE] {pkg} 설치 완료")
    else:
        _state["paddle"] = True

    if not has_module("paddleocr"):
        ok, err = _run_pip(["paddleocr"], log, timeout)
        if not ok:
            _state["reason"] = f"paddleocr 설치 실패: {err}"
            _log_to(log, f"  ❌ [PADDLE] paddleocr 설치 실패\n     {err}")
            return status()
        _state["paddleocr"] = True
        _log_to(log, "  ✅ [PADDLE] paddleocr 설치 완료")
    else:
        _state["paddleocr"] = True

    importlib.invalidate_caches()

    if ready():
        _state["ok"] = True
        _state["reason"] = ""
        _log_to(log, "  ✅ [PADDLE] PP-OCRv5 실행 백엔드 준비 완료")
    else:
        _state["reason"] = "설치 후에도 모듈을 찾지 못했습니다."
        _log_to(log, f"  ❌ [PADDLE] {_state['reason']}")

    return status()


def report_lines() -> List[str]:
    s = status()
    mark = "OK " if (s["paddle_present"] and s["paddleocr_present"]) else "MISS"
    out = [
        f"  [{mark}] PaddleOCR 백엔드 | paddle={s['paddle_present']} "
        f"paddleocr={s['paddleocr_present']} | Python {s['python']} "
        f"| venv {'예' if s['venv'] else '아니오'}"
    ]
    out.append(
        f"         실행 옵션: device={s.get('device', 'cpu')} "
        f"| oneDNN {'on' if s.get('mkldnn') else 'off'} "
        f"| threads={cpu_threads()}"
    )
    flags = s.get("runtime_flags") or {}
    if flags:
        out.append(
            "         플래그: " + " | ".join(f"{k}={v}" for k, v in flags.items())
        )
    if s.get("index"):
        out.append(f"         인덱스: {s['index']} ({s.get('package', '')})")
    if s.get("reason"):
        out.append(f"         사유: {s['reason']}")
    if not (s["paddle_present"] and s["paddleocr_present"]):
        for ln in manual_command().splitlines():
            out.append(f"         {ln}")
    return out