import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

BYTES_GB = 1024.0 ** 3

RAM_SAFETY_GB = 1.2
RAM_HEADROOM_RATIO = 0.15

ENV_RAM_LIMIT = "NMS_RAM_LIMIT_GB"
ENV_ALLOW_LOW_RAM = "NMS_ALLOW_LOW_RAM"


def _has_psutil() -> bool:
    try:
        import psutil  # noqa: F401
        return True
    except Exception:
        return False


def _windows_memory() -> Optional[Dict[str, float]]:
    if not sys.platform.startswith("win"):
        return None
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return None

        return {
            "total_gb": stat.ullTotalPhys / BYTES_GB,
            "available_gb": stat.ullAvailPhys / BYTES_GB,
            "commit_total_gb": stat.ullTotalPageFile / BYTES_GB,
            "commit_available_gb": stat.ullAvailPageFile / BYTES_GB,
            "load_percent": float(stat.dwMemoryLoad),
            "source": "GlobalMemoryStatusEx",
        }
    except Exception:
        return None


def ram_info() -> Dict[str, float]:
    win = _windows_memory()
    if win is not None:
        return win

    if _has_psutil():
        try:
            import psutil
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
            return {
                "total_gb": vm.total / BYTES_GB,
                "available_gb": vm.available / BYTES_GB,
                "commit_total_gb": (vm.total + sw.total) / BYTES_GB,
                "commit_available_gb": (vm.available + sw.free) / BYTES_GB,
                "load_percent": float(vm.percent),
                "source": "psutil",
            }
        except Exception:
            pass

    return {
        "total_gb": 0.0,
        "available_gb": 0.0,
        "commit_total_gb": 0.0,
        "commit_available_gb": 0.0,
        "load_percent": 0.0,
        "source": "unknown",
    }


def usable_ram_gb() -> float:
    info = ram_info()
    avail = float(info.get("available_gb", 0.0) or 0.0)
    commit = float(info.get("commit_available_gb", 0.0) or 0.0)

    if avail <= 0.0 and commit <= 0.0:
        return 0.0
    if commit <= 0.0:
        return avail
    if avail <= 0.0:
        return commit
    return min(avail, commit)


def ram_limit_gb() -> float:
    try:
        raw = float(os.environ.get(ENV_RAM_LIMIT, "0") or 0.0)
    except Exception:
        raw = 0.0
    return raw if raw > 0.0 else 0.0


def allow_low_ram() -> bool:
    raw = str(os.environ.get(ENV_ALLOW_LOW_RAM, "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def model_disk_gb(path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        try:
            return p.stat().st_size / BYTES_GB
        except Exception:
            return 0.0

    total = 0
    for f in p.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in (".safetensors", ".bin", ".pt", ".pth", ".gguf"):
            continue
        try:
            total += f.stat().st_size
        except Exception:
            continue
    return total / BYTES_GB


def free_disk_gb(path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / BYTES_GB
    except Exception:
        return 0.0


def can_stage(
    need_gb: float,
    log: Optional[Callable[[str], None]] = None,
    label: str = "",
) -> Tuple[bool, str]:
    need = float(need_gb) * (1.0 + RAM_HEADROOM_RATIO) + RAM_SAFETY_GB
    usable = usable_ram_gb()
    cap = ram_limit_gb()

    if cap > 0.0:
        usable = min(usable, cap)

    if usable <= 0.0:
        return True, "가용 RAM 을 측정하지 못해 진행합니다."

    if usable >= need:
        return True, (
            f"가용 {usable:.1f} GB ≥ 필요 {need:.1f} GB"
        )

    reason = (
        f"가용 RAM {usable:.1f} GB < 필요 {need:.1f} GB "
        f"(가중치 {float(need_gb):.1f} GB + 여유 {RAM_SAFETY_GB:.1f} GB)"
    )

    if allow_low_ram():
        if log is not None:
            log(
                f"  ⚠ [RAM] {label} — {reason}\n"
                f"     {ENV_ALLOW_LOW_RAM}=1 이라 강행합니다. "
                f"프로세스가 강제 종료될 수 있습니다."
            )
        return True, reason

    return False, reason


def report_lines() -> list:
    info = ram_info()
    src = str(info.get("source", "unknown"))
    if src == "unknown":
        return [
            "  [MISS] 시스템 RAM 측정 불가 — psutil 설치를 권장합니다: "
            "pip install psutil"
        ]
    return [
        f"  [OK ] 시스템 RAM {info['total_gb']:.1f} GB "
        f"| 가용 {info['available_gb']:.1f} GB "
        f"| 커밋 여유 {info['commit_available_gb']:.1f} GB "
        f"| 사용률 {info['load_percent']:.0f}% ({src})"
    ]


def reclaim(log: Optional[Callable[[str], None]] = None, label: str = "") -> float:
    before = usable_ram_gb()

    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    gc.collect()

    after = usable_ram_gb()
    gained = max(0.0, after - before)

    if log is not None and gained >= 0.2:
        log(
            f"  🧹 [RAM] {label or '회수'} — {before:.1f} → {after:.1f} GB "
            f"(+{gained:.1f} GB)"
        )
    return after