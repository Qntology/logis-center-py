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
        from .zvec import CATALOG, ZvecStore

        self.recipe = recipe
        self.dim = int(dim)
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

        if root is None:
            self.store = CATALOG.store("phrases", self.dim, self.recipe)
        else:
            self.store = ZvecStore(
                "phrases", self.dim, self.recipe, root=Path(root)
            )
        self.root = self.store.root
        self.path = self.store.path

    def ensure_loaded(self) -> None:
        self.store.ensure_loaded()

    def get(self, phrase: str) -> Optional[np.ndarray]:
        v = self.store.get(phrase)
        with self._lock:
            if v is None:
                self.misses += 1
            else:
                self.hits += 1
        return v

    def get_many(self, phrases: Sequence[str]) -> Dict[str, np.ndarray]:
        found = self.store.get_many(list(phrases))
        with self._lock:
            self.hits += len(found)
            self.misses += max(0, len(list(phrases)) - len(found))
        return found

    def all_cached(self, phrases: Sequence[str]) -> bool:
        found = self.store.get_many(list(phrases))
        return len(found) == len(set(phrases))

    def put_batch(self, items: Sequence) -> int:
        payload = []
        for phrase, vec in items:
            payload.append((str(phrase), vec, {"kind": "anchor"}))
        if not payload:
            return 0
        return self.store.upsert_many(payload)

    def stats(self) -> dict:
        out = self.store.stats()
        out["hits"] = self.hits
        out["misses"] = self.misses
        return out

    def clear(self) -> None:
        self.store.purge()
        with self._lock:
            self.hits = 0
            self.misses = 0


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

    MIN_CACHE_DIM = 8

    def _ensure_cache(self, dim: int):
        if self.cache is not None:
            return
        if dim < self.MIN_CACHE_DIM:
            self._log(
                f"  ⏭ 앵커 캐시 생성 보류 — 임베딩 차원 {dim} 은 유효하지 "
                f"않습니다 (최소 {self.MIN_CACHE_DIM}). 영벡터를 디스크에 "
                f"기록하지 않습니다."
            )
            return
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
            degraded = mat is None
            if degraded:
                mat = np.zeros((len(missing), max(1, self.dim)), dtype=np.float32)
            mat = np.asarray(mat, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)

            if not degraded:
                self._ensure_cache(int(mat.shape[-1]))

            fresh = []
            for i, t in enumerate(missing):
                if i >= mat.shape[0]:
                    break
                vec = mat[i]
                cached[t] = vec
                if degraded:
                    continue
                if not np.any(vec):
                    continue
                fresh.append((t, vec))

            if degraded:
                self._log(
                    f"  🚯 [ANCHOR CACHE] 임베딩 제공자가 모두 실패해 "
                    f"{len(missing)}구를 캐시에 기록하지 않습니다. "
                    f"이 축은 이번 실행에서 무효입니다."
                )
            elif self.cache is not None and fresh:
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