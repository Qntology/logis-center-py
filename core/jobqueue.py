import threading
import time
from typing import Callable, List, Optional


class JobQueue:
    def __init__(self, log=None, name: str = "job", event=None):
        self._lock = threading.RLock()
        self._owner: Optional[int] = None
        self._depth = 0
        self._waiting = 0
        self._seq = 0
        self._log_fn = log or (lambda m: None)
        self._event_fn = event
        self.name = str(name)
        self.current: str = ""
        self.history: List[dict] = []

    def _emit(self, phase: str, title: str):
        if self._event_fn is None:
            return
        try:
            self._event_fn({
                "phase": phase,
                "title": title,
                "busy": self.busy(),
                "waiting": int(self._waiting),
                "seq": int(self._seq),
            })
        except Exception:
            pass

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def busy(self) -> bool:
        return self._owner is not None and self._depth > 0

    def stats(self) -> dict:
        return {
            "busy": self.busy(),
            "waiting": int(self._waiting),
            "done": len(self.history),
            "depth": int(self._depth),
            "current": self.current,
            "recent": list(self.history[-5:]),
        }

    class _Slot:
        def __init__(self, q: "JobQueue", title: str):
            self.q = q
            self.title = title
            self.tid = threading.get_ident()
            self.t0 = 0.0
            self.reentrant = False

        def __enter__(self):
            q = self.q
            if q._owner == self.tid and q._depth > 0:
                self.reentrant = True
                q._depth += 1
                return self

            if q.busy():
                q._waiting += 1
                q._log(
                    f"  ⏳ [QUEUE] '{self.title}' 대기 중 — 앞선 작업이 "
                    f"끝나면 이어서 실행합니다. (대기 {q._waiting}건)"
                )
                waited = True
            else:
                waited = False

            if waited:
                q._emit("wait", self.title)

            q._lock.acquire()
            if waited:
                q._waiting = max(0, q._waiting - 1)
            q._owner = self.tid
            q._depth = 1
            q._seq += 1
            q.current = self.title
            self.t0 = time.time()
            if waited:
                q._log(f"  ▶️ [QUEUE] '{self.title}' 대기 해제 — 시작 (#{q._seq})")
            q._emit("start", self.title)
            return self

        def __exit__(self, exc_type, exc, tb):
            q = self.q
            q._depth = max(0, q._depth - 1)
            if self.reentrant:
                return False
            dt = time.time() - self.t0
            q._owner = None
            q.current = ""
            try:
                q._lock.release()
            except Exception:
                pass
            mark = "완료" if exc_type is None else f"중단({exc_type.__name__})"
            if dt >= 1.0 or exc_type is not None or q._waiting:
                q._log(f"  ⏹ [QUEUE] '{self.title}' {mark} — {dt:.1f}s")
            q.history.append({
                "title": self.title,
                "seconds": round(dt, 2),
                "ok": exc_type is None,
            })
            if len(q.history) > 64:
                del q.history[: len(q.history) - 64]
            q._emit("done", self.title)
            return False

    def slot(self, title: str = "") -> "JobQueue._Slot":
        return JobQueue._Slot(self, title or self.name)

    def wrap(self, fn: Callable, title: str = "") -> Callable:
        label = title or getattr(fn, "__name__", self.name)

        def _inner(*args, **kwargs):
            with self.slot(label):
                return fn(*args, **kwargs)

        _inner.__name__ = getattr(fn, "__name__", "wrapped")
        return _inner