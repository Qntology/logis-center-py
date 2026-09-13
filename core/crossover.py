import threading
from typing import Callable, Dict, List, Optional, Sequence

PHASE_IDLE = "idle"
PHASE_EMBEDDING = "embedding"
PHASE_GENERATION = "generation"

SLOT_VISION = "vision"
SLOT_OCR = "ocr"
SLOT_EMBEDDER = "embedder"
SLOT_REFINER = "refiner"
SLOT_NLP = "nlp"
SLOT_JOINT = "joint"

PHASE_PLAN: Dict[str, Dict[str, List[str]]] = {
    PHASE_EMBEDDING: {
        "keep": [SLOT_JOINT, SLOT_VISION, SLOT_OCR, SLOT_EMBEDDER, SLOT_NLP],
        "release": [SLOT_REFINER],
    },
    PHASE_GENERATION: {
        "keep": [SLOT_OCR, SLOT_REFINER, SLOT_NLP],
        "release": [SLOT_JOINT, SLOT_VISION, SLOT_EMBEDDER],
    },
    PHASE_IDLE: {
        "keep": [],
        "release": [SLOT_JOINT, SLOT_VISION, SLOT_OCR, SLOT_EMBEDDER, SLOT_REFINER],
    },
}

SLOT_LABELS: Dict[str, str] = {
    SLOT_VISION: "A.X-VE 비전 인코더",
    SLOT_OCR: "Hayai OCR",
    SLOT_EMBEDDER: "텍스트 임베딩",
    SLOT_REFINER: "정제 LLM",
    SLOT_NLP: "Stanza NLP",
    SLOT_JOINT: "SigLIP2 조인트(비전-텍스트)",
}


def _vram_free_gb() -> float:
    try:
        import torch
        if not torch.cuda.is_available():
            return 0.0
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
        return float(total - reserved)
    except Exception:
        return 0.0


def _empty_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


class Slot:
    def __init__(
        self,
        name: str,
        loader: Callable[[], object],
        unloader: Optional[Callable[[object], None]] = None,
        label: str = "",
        est_gb: float = 0.0,
    ):
        self.name = name
        self.loader = loader
        self.unloader = unloader
        self.label = label or SLOT_LABELS.get(name, name)
        self.est_gb = float(est_gb)
        self.instance: Optional[object] = None
        self.load_count = 0
        self.release_count = 0
        self.last_error = ""

    @property
    def loaded(self) -> bool:
        return self.instance is not None

    def acquire(self) -> Optional[object]:
        if self.instance is None:
            self.instance = self.loader()
            self.load_count += 1
        return self.instance

    def release(self) -> bool:
        if self.instance is None:
            return False
        obj = self.instance
        self.instance = None
        self.release_count += 1
        try:
            if self.unloader is not None:
                self.unloader(obj)
            elif hasattr(obj, "unload"):
                obj.unload()
        except Exception:
            pass
        del obj
        _empty_cache()
        return True

    def stats(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "loaded": self.loaded,
            "loads": self.load_count,
            "releases": self.release_count,
            "est_gb": round(self.est_gb, 2),
            "last_error": self.last_error,
        }


class CrossoverSwitch:
    def __init__(
        self,
        log: Optional[Callable[[str], None]] = None,
        enabled: bool = True,
        budget_gb: float = 0.0,
        headroom_gb: float = 0.4,
    ):
        self.phase = PHASE_IDLE
        self.enabled = bool(enabled)
        self.budget_gb = float(budget_gb or 0.0)
        self.headroom_gb = float(headroom_gb)
        self.slots: Dict[str, Slot] = {}
        self._lock = threading.RLock()
        self._log_fn = log or (lambda m: None)
        self.transitions: List[dict] = []
        self.evictions: List[dict] = []

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def register(
        self,
        name: str,
        loader: Callable[[], object],
        unloader: Optional[Callable[[object], None]] = None,
        label: str = "",
        est_gb: float = 0.0,
    ) -> Slot:
        with self._lock:
            slot = Slot(name, loader, unloader, label, est_gb)
            self.slots[name] = slot
            return slot

    def loaded_estimate_gb(self) -> float:
        return sum(s.est_gb for s in self.slots.values() if s.loaded)

    def _make_room_for(self, name: str, protect: Sequence[str]) -> List[str]:
        slot = self.slots.get(name)
        if slot is None or slot.est_gb <= 0.0 or self.budget_gb <= 0.0:
            return []

        need = slot.est_gb + self.headroom_gb
        free = _vram_free_gb()
        if free >= need:
            return []

        protect_set = set(protect) | {name}
        candidates = [
            s for s in self.slots.values()
            if s.loaded and s.name not in protect_set and s.est_gb > 0.0
        ]
        candidates.sort(key=lambda s: s.est_gb, reverse=True)

        evicted: List[str] = []
        for cand in candidates:
            if _vram_free_gb() >= need:
                break
            self._log(
                f"  🧹 [VRAM 압박] '{slot.label}'({slot.est_gb:.1f} GB) 적재를 위해 "
                f"'{cand.label}'({cand.est_gb:.1f} GB) 를 먼저 반납합니다."
            )
            if self.release(cand.name):
                evicted.append(cand.name)
                self.evictions.append({
                    "for": name,
                    "evicted": cand.name,
                    "need_gb": round(need, 2),
                })

        after = _vram_free_gb()
        if after < need:
            self._log(
                f"  ⚠️ VRAM 여유 {after:.2f} GB < 필요 {need:.2f} GB. "
                f"'{slot.label}' 로드가 실패할 수 있습니다."
            )
        return evicted

    def get(self, name: str) -> Optional[object]:
        with self._lock:
            slot = self.slots.get(name)
            if slot is None:
                return None
            return slot.instance

    def acquire(self, name: str, protect: Optional[Sequence[str]] = None) -> Optional[object]:
        with self._lock:
            slot = self.slots.get(name)
            if slot is None:
                return None
            if slot.loaded:
                return slot.instance

            if self.enabled:
                self._make_room_for(name, protect or [])

            before = _vram_free_gb()
            try:
                obj = slot.acquire()
                slot.last_error = ""
            except Exception as e:
                slot.last_error = str(e)
                self._log(f"  ❌ [{slot.label}] 로드 실패: {e}")
                _empty_cache()
                raise

            after = _vram_free_gb()
            if before > 0.0:
                self._log(
                    f"  📥 [{slot.label}] 로드 "
                    f"(VRAM {before:.2f} → {after:.2f} GB)"
                )
            else:
                self._log(f"  📥 [{slot.label}] 로드")
            return obj

    def release(self, name: str) -> bool:
        with self._lock:
            slot = self.slots.get(name)
            if slot is None or not slot.loaded:
                return False
            before = _vram_free_gb()
            ok = slot.release()
            after = _vram_free_gb()
            if ok:
                if before > 0.0:
                    self._log(
                        f"  ♻️ [{slot.label}] 반납 "
                        f"(VRAM {before:.2f} → {after:.2f} GB, "
                        f"+{max(0.0, after - before):.2f} GB 확보)"
                    )
                else:
                    self._log(f"  ♻️ [{slot.label}] 반납")
            return ok

    def transition(self, phase: str, force: bool = False) -> dict:
        with self._lock:
            if phase not in PHASE_PLAN:
                return {"ok": False, "error": f"알 수 없는 페이즈: {phase}"}

            if self.phase == phase and not force:
                return {"ok": True, "phase": phase, "changed": False}

            plan = PHASE_PLAN[phase]
            prev = self.phase
            before = _vram_free_gb()

            self._log(f"🔀 [CROSSOVER] {prev} → {phase}")

            released: List[str] = []
            if self.enabled:
                for name in plan["release"]:
                    if self.release(name):
                        released.append(name)
            else:
                self._log("  ⏭ CROSSOVER 비활성 — 모델 반납을 건너뜁니다.")

            acquired: List[str] = []
            keep = list(plan["keep"])
            for name in keep:
                slot = self.slots.get(name)
                if slot is None:
                    continue
                if not slot.loaded:
                    try:
                        self.acquire(name, protect=keep)
                        acquired.append(name)
                    except Exception as e:
                        self._log(f"  ⚠ [{slot.label}] 로드 실패: {e}")

            after = _vram_free_gb()
            self.phase = phase

            record = {
                "ok": True,
                "phase": phase,
                "prev": prev,
                "changed": True,
                "released": released,
                "acquired": acquired,
                "vram_before": round(before, 3),
                "vram_after": round(after, 3),
            }
            self.transitions.append(record)

            if before > 0.0:
                self._log(
                    f"  📊 VRAM {before:.2f} → {after:.2f} GB "
                    f"| 반납 {len(released)} / 로드 {len(acquired)}"
                )
            return record

    def enter_embedding_phase(self) -> dict:
        return self.transition(PHASE_EMBEDDING)

    def switch_to_generation(self) -> dict:
        return self.transition(PHASE_GENERATION)

    def mark_idle(self) -> dict:
        return self.transition(PHASE_IDLE)

    def loaded_slots(self) -> List[str]:
        with self._lock:
            return [n for n, s in self.slots.items() if s.loaded]

    def stats(self) -> dict:
        with self._lock:
            return {
                "phase": self.phase,
                "enabled": self.enabled,
                "budget_gb": round(self.budget_gb, 2),
                "loaded_est_gb": round(self.loaded_estimate_gb(), 2),
                "vram_free_gb": round(_vram_free_gb(), 3),
                "slots": {n: s.stats() for n, s in self.slots.items()},
                "transitions": len(self.transitions),
                "evictions": len(self.evictions),
            }

    def report_lines(self) -> List[str]:
        s = self.stats()
        lines = [
            f"  페이즈: {s['phase']} | 전환 {s['transitions']}회 "
            f"| VRAM 여유 {s['vram_free_gb']:.2f} GB"
        ]
        for name, info in s["slots"].items():
            mark = "ON " if info["loaded"] else "off"
            lines.append(
                f"    [{mark}] {info['label']:<24} "
                f"로드 {info['loads']}회 / 반납 {info['releases']}회"
            )
        return lines