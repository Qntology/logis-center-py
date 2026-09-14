import gc
import json
import os
import shutil
import struct
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
ENV_ABORT_FLOOR = "NMS_RAM_ABORT_FLOOR_GB"
ENV_CPU_SHARE_RATIO = "NMS_CPU_SHARE_RATIO"

ABORT_FLOOR_GB = 0.35
WATCHDOG_INTERVAL_SEC = 0.35
WATCHDOG_CONFIRM_HITS = 3
WATCHDOG_RECHECK_SEC = 0.6
CPU_SHARE_MIN_GB = 1.6
CPU_SHARE_RATIO = 0.35
CPU_SHARE_CAP_GB = 2.0

QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x00000002
QUOTA_LIMITS_HARDWS_MAX_DISABLE = 0x00000008

CACHE_RAM_BUDGET_BYTES = 64 * 1024 * 1024


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


SAFETENSORS_HEADER_MAX = 128 * 1024 * 1024
STAGE_PEAK_MULTIPLIER = 4.0
STAGE_FLOOR_GB = 1.0


def _safetensors_max_tensor_bytes(path: Path) -> int:
    try:
        with open(path, "rb") as fh:
            head_len_raw = fh.read(8)
            if len(head_len_raw) < 8:
                return 0
            (head_len,) = struct.unpack("<Q", head_len_raw)
            if head_len <= 0 or head_len > SAFETENSORS_HEADER_MAX:
                return 0
            head = fh.read(int(head_len))
    except Exception:
        return 0

    try:
        meta = json.loads(head.decode("utf-8", errors="ignore"))
    except Exception:
        return 0
    if not isinstance(meta, dict):
        return 0

    biggest = 0
    for key, spec in meta.items():
        if key == "__metadata__" or not isinstance(spec, dict):
            continue
        offsets = spec.get("data_offsets")
        if not isinstance(offsets, (list, tuple)) or len(offsets) != 2:
            continue
        try:
            size = int(offsets[1]) - int(offsets[0])
        except Exception:
            continue
        if size > biggest:
            biggest = size
    return biggest


def model_peak_tensor_gb(path) -> float:
    p = Path(path)
    if not p.exists():
        return 0.0

    files = []
    if p.is_file():
        if p.suffix == ".safetensors":
            files = [p]
    else:
        files = sorted(f for f in p.glob("*.safetensors") if f.is_file())

    peak = 0
    for f in files:
        got = _safetensors_max_tensor_bytes(f)
        if got > peak:
            peak = got
    return peak / BYTES_GB


def staging_need_gb(path, fallback_gb: float = 0.0) -> Tuple[float, str]:
    peak = model_peak_tensor_gb(path)
    if peak <= 0.0:
        need = float(fallback_gb)
        return need, (
            f"safetensors 헤더를 읽지 못해 디스크 추정 {need:.1f} GB 를 씁니다."
        )
    need = max(STAGE_FLOOR_GB, peak * STAGE_PEAK_MULTIPLIER)
    return need, (
        f"safetensors 최대 텐서 {peak * 1024.0:.0f} MB × "
        f"{STAGE_PEAK_MULTIPLIER:.0f} = {need:.1f} GB "
        f"(mmap 스트리밍 실측 — 파일 전체가 동시에 RAM 에 오르지 않습니다)"
    )


def free_disk_gb(path) -> float:
    try:
        return shutil.disk_usage(str(path)).free / BYTES_GB
    except Exception:
        return 0.0


def safety_margin_gb(need_gb: float) -> float:
    size = float(need_gb)
    if size <= 0.5:
        return 0.25
    if size <= 1.5:
        return 0.45
    if size <= 3.0:
        return 0.80
    return RAM_SAFETY_GB


def can_stage(
    need_gb: float,
    log: Optional[Callable[[str], None]] = None,
    label: str = "",
) -> Tuple[bool, str]:
    margin = safety_margin_gb(need_gb)
    need = float(need_gb) * (1.0 + RAM_HEADROOM_RATIO) + margin
    usable = headroom_gb()
    cap = ram_limit_gb()

    if cap > 0.0:
        usable = min(usable, cap)

    if usable <= 0.0:
        return True, "가용 RAM 을 측정하지 못해 진행합니다."

    floor = abort_floor_gb()
    if usable >= need + floor:
        return True, (
            f"커밋 여유 {usable:.1f} GB ≥ 필요 {need:.1f} GB "
            f"+ 종료 임계 {floor:.2f} GB"
        )

    reason = (
        f"커밋 여유 {usable:.1f} GB < 필요 {need:.1f} GB "
        f"(가중치 {float(need_gb):.1f} GB + 여유 {margin:.2f} GB "
        f"+ 종료 임계 {floor:.2f} GB)"
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
        flags = (
            QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE
        )

        ok = False
        try:
            fn = ctypes.windll.kernel32.SetProcessWorkingSetSizeEx
            fn.restype = ctypes.c_int
            ok = bool(fn(
                handle,
                ctypes.c_size_t(-1),
                ctypes.c_size_t(-1),
                ctypes.c_uint(flags),
            ))
        except Exception:
            ok = False

        if not ok:
            try:
                ok = bool(ctypes.windll.psapi.EmptyWorkingSet(handle))
            except Exception:
                ok = False

        if not ok:
            try:
                ok = bool(ctypes.windll.kernel32.SetProcessWorkingSetSize(
                    handle, ctypes.c_size_t(-1), ctypes.c_size_t(-1)
                ))
            except Exception:
                ok = False

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
    trim_rounds = 0

    for i in range(max(1, int(rounds))):
        gc.collect()
        _cuda_purge()
        gc.collect()

        if aggressive and i == 0:
            trimmed = trim_working_set()
            if trimmed:
                trim_rounds += 1

        if i + 1 < int(rounds):
            time.sleep(RECLAIM_SETTLE_SEC)
            mid = usable_ram_gb()
            if before > 0.0 and (mid - before) < TRIM_MIN_GAIN_GB:
                if aggressive:
                    if trim_working_set():
                        trim_rounds += 1
                        trimmed = True
                continue

    after = usable_ram_gb()
    gained = max(0.0, after - before)

    if log is not None and gained >= TRIM_MIN_GAIN_GB:
        tail = f" | 작업 집합 트리밍 {trim_rounds}회" if trimmed else ""
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


def commit_free_gb() -> float:
    info = ram_info()
    return float(info.get("commit_available_gb", 0.0) or 0.0)


def headroom_gb() -> float:
    commit = commit_free_gb()
    if commit > 0.0:
        return commit
    return float(ram_info().get("available_gb", 0.0) or 0.0)


def physical_free_gb() -> float:
    return float(ram_info().get("available_gb", 0.0) or 0.0)


def pressure_summary() -> str:
    info = ram_info()
    return (
        f"물리 가용 {float(info.get('available_gb', 0.0) or 0.0):.1f} GB "
        f"| 커밋 여유 {float(info.get('commit_available_gb', 0.0) or 0.0):.1f} GB "
        f"| 사용률 {float(info.get('load_percent', 0.0) or 0.0):.0f}%"
    )


def abort_floor_gb() -> float:
    try:
        raw = float(os.environ.get(ENV_ABORT_FLOOR, "0") or 0.0)
    except Exception:
        raw = 0.0
    return raw if raw > 0.0 else ABORT_FLOOR_GB


def cpu_share_ratio() -> float:
    try:
        raw = float(os.environ.get(ENV_CPU_SHARE_RATIO, "0") or 0.0)
    except Exception:
        raw = 0.0
    if raw > 0.0:
        return min(1.0, raw)
    return CPU_SHARE_RATIO


def cpu_share_gb(reserve_gb: float = 1.2) -> float:
    room = headroom_gb() - float(reserve_gb)
    if room < CPU_SHARE_MIN_GB:
        return 0.0

    phys = physical_free_gb() - float(reserve_gb)
    if phys > 0.0:
        room = min(room, phys)

    share = room * cpu_share_ratio()
    share = min(share, CPU_SHARE_CAP_GB)

    if share < CPU_SHARE_MIN_GB:
        return 0.0
    return share


def cache_budget_entries(entry_bytes: int) -> int:
    n = max(1, int(entry_bytes))
    return max(64, CACHE_RAM_BUDGET_BYTES // n)


class RamGuardAbort(MemoryError):
    pass


class RamWatchdog:
    def __init__(
        self,
        label: str = "",
        floor_gb: float = 0.0,
        log: Optional[Callable[[str], None]] = None,
        interval: float = WATCHDOG_INTERVAL_SEC,
    ):
        self.label = str(label or "")
        self.floor = float(floor_gb) if floor_gb > 0.0 else abort_floor_gb()
        self._log = log
        self.interval = float(interval)
        self.tripped = False
        self.low_water = 0.0
        self.low_physical = 0.0
        self.hits = 0
        self.warned = False
        self._stop = None
        self._thread = None

    def _emit(self, msg: str) -> None:
        if self._log is None:
            return
        try:
            self._log(msg)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            room = commit_free_gb()
            phys = physical_free_gb()

            if phys > 0.0 and (self.low_physical <= 0.0 or phys < self.low_physical):
                self.low_physical = phys

            if room > 0.0:
                if self.low_water <= 0.0 or room < self.low_water:
                    self.low_water = room

                if room < self.floor:
                    self.hits += 1
                    if not self.warned:
                        self.warned = True
                        self._emit(
                            f"  ⚠ [RAM GUARD] {self.label} 커밋 여유 "
                            f"{room:.2f} GB < 임계 {self.floor:.2f} GB "
                            f"— {WATCHDOG_CONFIRM_HITS}회 연속 확인 중입니다 "
                            f"(물리 가용 {phys:.2f} GB)"
                        )
                    if self.hits >= WATCHDOG_CONFIRM_HITS and not self.tripped:
                        self.tripped = True
                        self._emit(
                            f"  🚨 [RAM GUARD] {self.label} 커밋 여유가 "
                            f"{WATCHDOG_CONFIRM_HITS}회 연속 임계 미달입니다 "
                            f"({room:.2f} GB < {self.floor:.2f} GB). "
                            f"프로세스 강제 종료를 막기 위해 중단 신호를 "
                            f"올립니다."
                        )
                else:
                    self.hits = 0

            self._stop.wait(self.interval)

    def __enter__(self) -> "RamWatchdog":
        import threading as _th

        self._stop = _th.Event()
        self.low_water = commit_free_gb()
        self.low_physical = physical_free_gb()
        self._thread = _th.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        if self.low_water > 0.0:
            tail = ""
            if self.low_physical > 0.0:
                tail = (
                    f" | 최저 물리 가용 {self.low_physical:.2f} GB "
                    f"(mmap 페이지 캐시는 회수 가능하므로 판정에 쓰지 "
                    f"않습니다)"
                )
            self._emit(
                f"  📉 [RAM GUARD] {self.label} 최저 커밋 여유 "
                f"{self.low_water:.2f} GB (임계 {self.floor:.2f} GB){tail}"
            )
        return False

    def check(self, settled: bool = True) -> None:
        if not self.tripped:
            return

        if settled:
            import time as _t

            _t.sleep(WATCHDOG_RECHECK_SEC)
            room = commit_free_gb()
            if room >= self.floor:
                self.tripped = False
                self.hits = 0
                self._emit(
                    f"  ✅ [RAM GUARD] {self.label} 적재 완료 후 커밋 여유가 "
                    f"{room:.2f} GB 로 회복되었습니다. 중단 신호를 "
                    f"취소하고 이 모델을 그대로 씁니다."
                )
                return

        raise RamGuardAbort(
            f"{self.label} 적재 중 커밋 여유가 임계 {self.floor:.2f} GB "
            f"아래로 떨어져 중단했습니다 (최저 {self.low_water:.2f} GB)."
        )