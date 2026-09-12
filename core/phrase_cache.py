import hashlib
import json
import struct
import threading
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .model_manager import MODELS_ROOT

CACHE_MAGIC = 0x50485243
CACHE_VERSION = 1
CACHE_ROOT = MODELS_ROOT / "cache" / "phrase_embeds"
MAX_RECORDS = 200_000


def _key_of(phrase: str) -> int:
    h = hashlib.blake2b(phrase.encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", h)[0]


class PhraseCache:
    def __init__(self, recipe: str, dim: int, root: Optional[Path] = None):
        self.recipe = recipe
        self.dim = int(dim)
        self.root = Path(root or CACHE_ROOT)
        self.path = self.root / f"{self._safe(recipe)}-d{self.dim}.bin"
        self.mem: Dict[int, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
        self._loaded = False

    @staticmethod
    def _safe(name: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)

    def _record_size(self) -> int:
        return 8 + self.dim * 4

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True

            if not self.path.exists():
                return

            try:
                raw = self.path.read_bytes()
            except Exception:
                return

            if len(raw) < 16:
                return

            magic, version, dim = struct.unpack("<III", raw[:12])
            if magic != CACHE_MAGIC or version != CACHE_VERSION or dim != self.dim:
                try:
                    self.path.unlink()
                except Exception:
                    pass
                return

            rec = self._record_size()
            body = raw[16:]
            n = len(body) // rec
            for i in range(n):
                off = i * rec
                key = struct.unpack_from("<Q", body, off)[0]
                vec = np.frombuffer(body, dtype=np.float32, count=self.dim, offset=off + 8)
                self.mem[key] = vec.copy()

            if n:
                print(
                    f"[PHRASE-CACHE] 복원 {n}구 "
                    f"({len(raw) / 1e6:.1f} MB) | {self.path.name}"
                )

    def get(self, phrase: str) -> Optional[np.ndarray]:
        self.ensure_loaded()
        with self._lock:
            v = self.mem.get(_key_of(phrase))
            if v is None:
                self.misses += 1
                return None
            self.hits += 1
            return v

    def get_many(self, phrases: Sequence[str]) -> Dict[str, np.ndarray]:
        self.ensure_loaded()
        out: Dict[str, np.ndarray] = {}
        with self._lock:
            for p in phrases:
                v = self.mem.get(_key_of(p))
                if v is None:
                    self.misses += 1
                else:
                    self.hits += 1
                    out[p] = v
        return out

    def all_cached(self, phrases: Sequence[str]) -> bool:
        self.ensure_loaded()
        with self._lock:
            return all(_key_of(p) in self.mem for p in phrases)

    def put_batch(self, items: Sequence) -> int:
        self.ensure_loaded()
        fresh: List = []

        with self._lock:
            if len(self.mem) >= MAX_RECORDS:
                return 0
            for phrase, vec in items:
                v = np.asarray(vec, dtype=np.float32).reshape(-1)
                if v.shape[0] != self.dim:
                    continue
                key = _key_of(phrase)
                if key in self.mem:
                    continue
                self.mem[key] = v
                fresh.append((key, v))

        if not fresh:
            return 0

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            is_new = not self.path.exists()
            with open(self.path, "ab") as f:
                if is_new:
                    f.write(struct.pack("<IIII", CACHE_MAGIC, CACHE_VERSION, self.dim, 0))
                for key, v in fresh:
                    f.write(struct.pack("<Q", key))
                    f.write(v.astype(np.float32).tobytes())
        except Exception as e:
            print(f"[PHRASE-CACHE] 디스크 기록 실패(메모리는 유지): {e}")

        return len(fresh)

    def stats(self) -> dict:
        return {
            "recipe": self.recipe,
            "dim": self.dim,
            "path": str(self.path),
            "entries": len(self.mem),
            "hits": self.hits,
            "misses": self.misses,
        }

    def clear(self) -> None:
        with self._lock:
            self.mem.clear()
            self.hits = 0
            self.misses = 0
            try:
                if self.path.exists():
                    self.path.unlink()
            except Exception:
                pass


class CachedEmbedder:
    def __init__(
        self,
        encode_fn: Callable[[List[str]], np.ndarray],
        recipe: str,
        dim: int = 0,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.encode_fn = encode_fn
        self.recipe = recipe
        self.dim = int(dim)
        self.cache: Optional[PhraseCache] = PhraseCache(recipe, dim) if dim > 0 else None
        self._log_fn = log or (lambda m: None)

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def _ensure_cache(self, dim: int):
        if self.cache is None and dim > 0:
            self.dim = dim
            self.cache = PhraseCache(self.recipe, dim)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        items = [str(t) if t is not None else "" for t in texts]
        if not items:
            return np.zeros((0, max(1, self.dim)), dtype=np.float32)

        cached: Dict[str, np.ndarray] = {}
        if self.cache is not None:
            cached = self.cache.get_many(items)

        missing = [t for t in items if t not in cached]

        if missing:
            mat = self.encode_fn(missing)
            if mat is None:
                mat = np.zeros((len(missing), max(1, self.dim)), dtype=np.float32)
            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)

            self._ensure_cache(int(mat.shape[-1]))

            fresh = []
            for i, t in enumerate(missing):
                if i >= mat.shape[0]:
                    break
                cached[t] = mat[i]
                fresh.append((t, mat[i]))

            if self.cache is not None and fresh:
                n = self.cache.put_batch(fresh)
                if n:
                    self._log(f"  💾 앵커 캐시 신규 {n}구 기록")
        elif items:
            self._log(f"  ⚡ 앵커 캐시 전량 히트 ({len(items)}구) — 인코더 호출 생략")

        dim = self.dim if self.dim > 0 else (
            int(next(iter(cached.values())).shape[-1]) if cached else 1
        )
        out = np.zeros((len(items), dim), dtype=np.float32)
        for i, t in enumerate(items):
            v = cached.get(t)
            if v is not None and v.shape[-1] == dim:
                out[i] = v
        return out

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def all_cached(self, phrases: Sequence[str]) -> bool:
        if self.cache is None:
            return False
        return self.cache.all_cached(phrases)

    def stats(self) -> dict:
        return self.cache.stats() if self.cache else {"entries": 0}