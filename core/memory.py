import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

BYTES_GB = 1024.0 ** 3

RAM_SAFETY_GB = 1.2
RAM_HEADROOM_RATIO = 0.15

RECLAIM_ROUNDS = 3
RECLAIM_SETTLE_SEC = 0.15
TRIM_MIN_GAIN_GB = 0.15

ENV_RAM_LIMIT = "NMS_RAM_LIMIT_GB"
ENV_ALLOW_LOW_RAM = "NMS_ALLOW_LOW_RAM"
ENV_NO_TRIM = "NMS_NO_WORKING_SET_TRIM"


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


def trim_working_set() -> bool:
    if str(os.environ.get(ENV_NO_TRIM, "0")).strip().lower() in (
        "1", "true", "yes", "on"
    ):
        return False
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.EmptyWorkingSet(handle)
        if not ok:
            ok = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx(
                handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1), 0
            )
        return bool(ok)
    except Exception:
        return False


def _cuda_purge():
    try:
        import torch
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    except Exception:
        pass


def reclaim(
    log: Optional[Callable[[str], None]] = None,
    label: str = "",
    rounds: int = RECLAIM_ROUNDS,
    aggressive: bool = True,
) -> float:
    import time

    before = usable_ram_gb()
    trimmed = False

    for i in range(max(1, int(rounds))):
        gc.collect()
        _cuda_purge()
        gc.collect()

        if aggressive and i == 0:
            trimmed = trim_working_set()

        if i + 1 < int(rounds):
            time.sleep(RECLAIM_SETTLE_SEC)
            mid = usable_ram_gb()
            if before > 0.0 and (mid - before) < TRIM_MIN_GAIN_GB:
                continue

    after = usable_ram_gb()
    gained = max(0.0, after - before)

    if log is not None and gained >= TRIM_MIN_GAIN_GB:
        tail = " | 작업 집합 트리밍" if trimmed else ""
        log(
            f"  🧹 [RAM] {label or '회수'} — {before:.1f} → {after:.1f} GB "
            f"(+{gained:.1f} GB){tail}"
        )
    return after


def make_room(
    need_gb: float,
    release_fn: Optional[Callable[[], bool]] = None,
    log: Optional[Callable[[str], None]] = None,
    label: str = "",
    max_rounds: int = 6,
) -> Tuple[bool, float]:
    need = float(need_gb)
    room = usable_ram_gb()

    if room <= 0.0 or room >= need:
        return True, room

    if log is not None:
        log(
            f"  🪜 [RAM STAGE] {label} — 필요 {need:.1f} GB / 현재 "
            f"{room:.1f} GB. 단계적으로 확보합니다."
        )

    room = reclaim(log=log, label=f"{label} 1단계 회수")
    if room >= need:
        return True, room

    if release_fn is None:
        return False, room

    for i in range(max(1, int(max_rounds))):
        try:
            released = bool(release_fn())
        except Exception:
            released = False

        if not released:
            break

        room = reclaim(
            log=log, label=f"{label} {i + 2}단계 회수", rounds=2
        )
        if log is not None:
            log(f"     · 회차 {i + 1}: 가용 {room:.1f} GB / 목표 {need:.1f} GB")
        if room >= need:
            return True, room

    return room >= need, room