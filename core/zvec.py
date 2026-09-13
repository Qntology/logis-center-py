import hashlib
import json
import struct
import threading
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .model_manager import MODELS_ROOT

ZVEC_MAGIC = 0x5A564543
ZVEC_VERSION = 1
ZVEC_ROOT = MODELS_ROOT / "zvec"

FLAG_LIVE = 0
FLAG_TOMBSTONE = 1

MAX_RECORDS = 500_000
COMPACT_RATIO = 0.35


def key_of(record_id: str) -> int:
    h = hashlib.blake2b(str(record_id).encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<Q", h)[0]


def _safe(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))


class ZvecStore:
    def __init__(
        self,
        namespace: str,
        dim: int,
        recipe: str = "",
        root: Optional[Path] = None,
        log: Optional[Callable[[str], None]] = None,
    ):
        self.namespace = str(namespace)
        self.dim = int(dim)
        self.recipe = str(recipe or "")
        self.root = Path(root or ZVEC_ROOT)
        tag = hashlib.blake2b(
            f"{self.recipe}|{int(self.dim)}".encode("utf-8"), digest_size=4
        ).hexdigest()
        self.path = (
            self.root
            / f"zvec_index.{_safe(self.namespace)}.d{int(self.dim)}.{tag}.bin"
        )
        self._log_fn = log or (lambda m: None)

        self._vec: Dict[int, np.ndarray] = {}
        self._meta: Dict[int, dict] = {}
        self._rid: Dict[int, str] = {}
        self._dead = 0
        self._matrix: Optional[np.ndarray] = None
        self._order: List[int] = []
        self._lock = threading.RLock()
        self._loaded = False

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def _record_size(self, meta_len: int) -> int:
        return 8 + 4 + 4 + self.dim * 4 + int(meta_len)

    def _write_header(self, fh):
        rb = self.recipe.encode("utf-8")
        fh.write(struct.pack("<IIII", ZVEC_MAGIC, ZVEC_VERSION, self.dim, len(rb)))
        fh.write(rb)

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            self._loaded = True

            if self.dim <= 0 or not self.path.exists():
                return

            try:
                raw = self.path.read_bytes()
            except Exception as e:
                self._log(f"  ⚠ [zvec/{self.namespace}] 읽기 실패: {e}")
                return

            if len(raw) < 16:
                return

            magic, version, dim, rlen = struct.unpack_from("<IIII", raw, 0)
            off = 16 + int(rlen)
            if off > len(raw):
                return
            recipe = raw[16:off].decode("utf-8", errors="ignore")

            if (magic != ZVEC_MAGIC or version != ZVEC_VERSION
                    or dim != self.dim or recipe != self.recipe):
                self._log(
                    f"  🧹 [zvec/{self.namespace}] 세대 불일치 "
                    f"(저장 dim={dim} recipe='{recipe}' vs "
                    f"현재 dim={self.dim} recipe='{self.recipe}') → 전량 폐기"
                )
                try:
                    self.path.unlink()
                except Exception:
                    pass
                return

            vec_bytes = self.dim * 4
            n = 0
            while off + 16 <= len(raw):
                key, flags, meta_len = struct.unpack_from("<QII", raw, off)
                body = off + 16
                end = body + vec_bytes + int(meta_len)
                if end > len(raw):
                    break
                if flags == FLAG_TOMBSTONE:
                    self._vec.pop(key, None)
                    self._meta.pop(key, None)
                    self._rid.pop(key, None)
                else:
                    vec = np.frombuffer(
                        raw, dtype=np.float32, count=self.dim, offset=body
                    ).copy()
                    meta_raw = raw[body + vec_bytes: end]
                    try:
                        meta = json.loads(meta_raw.decode("utf-8")) if meta_len else {}
                    except Exception:
                        meta = {}
                    self._vec[key] = vec
                    self._meta[key] = meta if isinstance(meta, dict) else {}
                    self._rid[key] = str(self._meta[key].get("_id", key))
                off = end
                n += 1

            self._dead = max(0, n - len(self._vec))
            self._matrix = None
            if self._vec:
                self._log(
                    f"  💽 [zvec/{self.namespace}] 복원 {len(self._vec)}건 "
                    f"(레코드 {n} / 묘비 {self._dead} / {len(raw) / 1e6:.1f} MB)"
                )

    def _append(self, rows: Sequence[Tuple[int, int, np.ndarray, bytes]]) -> None:
        if not rows:
            return
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            is_new = not self.path.exists()
            with open(self.path, "ab") as fh:
                if is_new:
                    self._write_header(fh)
                for key, flags, vec, meta in rows:
                    fh.write(struct.pack("<QII", key, flags, len(meta)))
                    fh.write(np.asarray(vec, dtype=np.float32).tobytes())
                    if meta:
                        fh.write(meta)
        except Exception as e:
            self._log(f"  ⚠ [zvec/{self.namespace}] 기록 실패(메모리는 유지): {e}")

    def upsert(self, record_id: str, vector, payload: Optional[dict] = None) -> bool:
        return self.upsert_many([(record_id, vector, payload or {})]) > 0

    def upsert_many(self, items: Iterable[Tuple[str, object, dict]]) -> int:
        self.ensure_loaded()
        rows: List[Tuple[int, int, np.ndarray, bytes]] = []

        with self._lock:
            if self.dim <= 0:
                return 0
            for record_id, vector, payload in items:
                v = np.asarray(vector, dtype=np.float32).reshape(-1)
                if v.shape[0] != self.dim:
                    continue
                if len(self._vec) >= MAX_RECORDS and key_of(record_id) not in self._vec:
                    break
                key = key_of(record_id)
                meta = dict(payload or {})
                meta["_id"] = str(record_id)
                blob = json.dumps(meta, ensure_ascii=False).encode("utf-8")

                if key in self._vec:
                    self._dead += 1
                self._vec[key] = v
                self._meta[key] = meta
                self._rid[key] = str(record_id)
                rows.append((key, FLAG_LIVE, v, blob))

            self._matrix = None

        self._append(rows)
        self._maybe_compact()
        return len(rows)

    def delete(self, record_id: str) -> bool:
        self.ensure_loaded()
        key = key_of(record_id)
        with self._lock:
            if key not in self._vec:
                return False
            self._vec.pop(key, None)
            self._meta.pop(key, None)
            self._rid.pop(key, None)
            self._dead += 1
            self._matrix = None
        self._append([(key, FLAG_TOMBSTONE, np.zeros(self.dim, np.float32), b"")])
        return True

    def get(self, record_id: str) -> Optional[np.ndarray]:
        self.ensure_loaded()
        with self._lock:
            v = self._vec.get(key_of(record_id))
            return None if v is None else v.copy()

    def payload(self, record_id: str) -> Optional[dict]:
        self.ensure_loaded()
        with self._lock:
            m = self._meta.get(key_of(record_id))
            return None if m is None else dict(m)

    def get_many(self, record_ids: Sequence[str]) -> Dict[str, np.ndarray]:
        self.ensure_loaded()
        out: Dict[str, np.ndarray] = {}
        with self._lock:
            for rid in record_ids:
                v = self._vec.get(key_of(rid))
                if v is not None:
                    out[rid] = v
        return out

    def all_ids(self) -> List[str]:
        self.ensure_loaded()
        with self._lock:
            return list(self._rid.values())

    def _build_matrix(self) -> Tuple[Optional[np.ndarray], List[int]]:
        if self._matrix is not None:
            return self._matrix, self._order
        if not self._vec:
            return None, []
        order = list(self._vec.keys())
        mat = np.stack([self._vec[k] for k in order]).astype(np.float32)
        mat = mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)
        self._matrix = mat
        self._order = order
        return mat, order

    def search(
        self,
        vector,
        top_k: int = 10,
        where: Optional[Callable[[dict], bool]] = None,
    ) -> List[Tuple[str, float, dict]]:
        self.ensure_loaded()
        q = np.asarray(vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            return []
        q = q / max(float(np.linalg.norm(q)), 1e-8)

        with self._lock:
            mat, order = self._build_matrix()
            if mat is None:
                return []
            sims = mat @ q
            idx = np.argsort(-sims)

            out: List[Tuple[str, float, dict]] = []
            for i in idx:
                key = order[int(i)]
                meta = self._meta.get(key, {})
                if where is not None:
                    try:
                        if not where(meta):
                            continue
                    except Exception:
                        continue
                out.append((self._rid.get(key, str(key)), float(sims[int(i)]), dict(meta)))
                if len(out) >= int(top_k):
                    break
            return out

    def _maybe_compact(self) -> None:
        with self._lock:
            live = len(self._vec)
            if live == 0 or self._dead < 64:
                return
            if self._dead < live * COMPACT_RATIO:
                return
        self.compact()

    def compact(self) -> int:
        self.ensure_loaded()
        with self._lock:
            snapshot = [
                (k, self._vec[k], json.dumps(self._meta.get(k, {}),
                                             ensure_ascii=False).encode("utf-8"))
                for k in self._vec
            ]
            self._dead = 0

        tmp = self.path.with_suffix(".bin.tmp")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as fh:
                self._write_header(fh)
                for key, vec, meta in snapshot:
                    fh.write(struct.pack("<QII", key, FLAG_LIVE, len(meta)))
                    fh.write(vec.tobytes())
                    fh.write(meta)
            tmp.replace(self.path)
            self._log(
                f"  🧯 [zvec/{self.namespace}] 압축 완료 — {len(snapshot)}건 유지"
            )
        except Exception as e:
            self._log(f"  ⚠ [zvec/{self.namespace}] 압축 실패: {e}")
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        return len(snapshot)

    def purge(self) -> None:
        with self._lock:
            self._vec.clear()
            self._meta.clear()
            self._rid.clear()
            self._dead = 0
            self._matrix = None
            self._loaded = True
        try:
            if self.path.exists():
                self.path.unlink()
        except Exception:
            pass

    def stats(self) -> dict:
        self.ensure_loaded()
        with self._lock:
            size = self.path.stat().st_size if self.path.exists() else 0
            return {
                "namespace": self.namespace,
                "dim": self.dim,
                "recipe": self.recipe,
                "path": str(self.path),
                "entries": len(self._vec),
                "tombstones": self._dead,
                "bytes": int(size),
            }


class ZvecCatalog:
    def __init__(self, root: Optional[Path] = None, log=None):
        self.root = Path(root or ZVEC_ROOT)
        self._log_fn = log
        self._stores: Dict[str, ZvecStore] = {}
        self._lock = threading.RLock()

    def store(self, namespace: str, dim: int, recipe: str = "") -> ZvecStore:
        key = f"{namespace}|{int(dim)}|{recipe}"
        with self._lock:
            st = self._stores.get(key)
            if st is None:
                st = ZvecStore(
                    namespace, dim, recipe, root=self.root, log=self._log_fn
                )
                self._stores[key] = st
            return st

    def stats(self) -> List[dict]:
        with self._lock:
            return [s.stats() for s in self._stores.values()]

    def purge_all(self) -> None:
        with self._lock:
            for s in self._stores.values():
                s.purge()
            self._stores.clear()
        try:
            for f in self.root.glob("zvec_index.*.bin"):
                f.unlink()
        except Exception:
            pass


CATALOG = ZvecCatalog()