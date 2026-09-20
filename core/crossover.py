import threading
from typing import Callable, Dict, List, Optional, Sequence

PHASE_IDLE = "idle"
PHASE_EMBEDDING = "embedding"
PHASE_GENERATION = "generation"


SLOT_OCR = "ocr"
SLOT_EMBEDDER = "embedder"
SLOT_REFINER = "refiner"
SLOT_NLP = "nlp"
SLOT_JOINT = "joint"

PHASE_PLAN: Dict[str, Dict[str, List[str]]] = {
    PHASE_EMBEDDING: {
        "keep": [SLOT_JOINT, SLOT_OCR, SLOT_EMBEDDER, SLOT_NLP],
        "release": [SLOT_REFINER],
    },
    PHASE_GENERATION: {
        "keep": [SLOT_OCR, SLOT_REFINER, SLOT_NLP],
        "release": [SLOT_JOINT, SLOT_EMBEDDER],
    },
    PHASE_IDLE: {
        "keep": [],
        "release": [SLOT_JOINT, SLOT_OCR, SLOT_EMBEDDER, SLOT_REFINER],
    },
}

SLOT_LABELS: Dict[str, str] = {
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
        torch_view = float(total - reserved)

        try:
            free_b, _total_b = torch.cuda.mem_get_info(0)
            driver_view = float(free_b) / (1024 ** 3)
        except Exception:
            driver_view = 0.0

        if driver_view <= 0.0:
            return torch_view
        return min(torch_view, driver_view)
    except Exception:
        return 0.0


def _empty_cache():
    try:
        from .memory import reclaim
        reclaim()
        return
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def _ram_free_gb() -> float:
    try:
        from .memory import headroom_gb
        return headroom_gb()
    except Exception:
        pass
    try:
        from .memory import usable_ram_gb
        return usable_ram_gb()
    except Exception:
        return 0.0


class Slot:
    def __init__(
        self,
        name: str,
        loader: Callable[[], object],
        unloader: Optional[Callable[[object], None]] = None,
        label: str = "",
        est_gb: float = 0.0,
        stage_gb: float = 0.0,
        lazy: bool = False,
    ):
        self.name = name
        self.loader = loader
        self.unloader = unloader
        self.label = label or SLOT_LABELS.get(name, name)
        self.est_gb = float(est_gb)
        self.stage_gb = float(stage_gb or 0.0)
        self.lazy = bool(lazy)
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
        self._thrash_warned: set = set()

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
        stage_gb: float = 0.0,
        lazy: bool = False,
    ) -> Slot:
        with self._lock:
            slot = Slot(
                name, loader, unloader, label, est_gb, stage_gb, lazy
            )
            self.slots[name] = slot
            return slot

    def loaded_estimate_gb(self) -> float:
        return sum(s.est_gb for s in self.slots.values() if s.loaded)

    EVICT_LAST = (SLOT_JOINT, SLOT_OCR)

    RAM_STAGE_RATIO = 2.4
    RAM_STAGE_FLOOR = 1.5

    THRASH_WINDOW = 6
    THRASH_LIMIT = 3

    def _make_room_for(self, name: str, protect: Sequence[str]) -> List[str]:
        slot = self.slots.get(name)
        if slot is None or slot.est_gb <= 0.0 or self.budget_gb <= 0.0:
            return []

        need = slot.est_gb + self.headroom_gb
        declared = float(getattr(slot, "stage_gb", 0.0) or 0.0)
        if declared > 0.0:
            ram_need = declared
        else:
            ram_need = slot.est_gb * self.RAM_STAGE_RATIO + self.RAM_STAGE_FLOOR

        free = _vram_free_gb()
        ram = _ram_free_gb()

        vram_ok = free >= need
        ram_ok = ram <= 0.0 or ram >= ram_need

        if vram_ok and ram_ok:
            return []

        if not ram_ok:
            self._log(
                f"  🧹 [RAM 압박] '{slot.label}' 적재에는 시스템 RAM "
                f"{ram_need:.1f} GB 가 필요한데 {ram:.1f} GB 뿐입니다. "
                f"가중치는 GPU 로 가기 전에 RAM 을 먼저 거칩니다."
            )

        protect_set = set(protect) | {name}
        candidates = [
            s for s in self.slots.values()
            if s.loaded and s.name not in protect_set and s.est_gb > 0.0
        ]
        candidates.sort(
            key=lambda s: (
                1 if s.name in self.EVICT_LAST else 0,
                -s.est_gb,
            )
        )

        recent = [e.get("evicted") for e in self.evictions[-self.THRASH_WINDOW:]]

        evicted: List[str] = []
        for cand in candidates:
            if _vram_free_gb() >= need and (
                _ram_free_gb() <= 0.0 or _ram_free_gb() >= ram_need
            ):
                break

            hits = recent.count(cand.name)
            if hits >= self.THRASH_LIMIT and cand.name not in self._thrash_warned:
                self._thrash_warned.add(cand.name)
                self._log(
                    f"  🔁 [스래싱] '{cand.label}' 이 최근 {self.THRASH_WINDOW}회 "
                    f"중 {hits}번 반납되었습니다. 메모리가 모든 모델을 담기에 "
                    f"부족해 같은 모델을 반복해서 올렸다 내리는 중입니다."
                )
                self._log(
                    f"     실행 시간이 크게 늘어납니다. RAM 을 확보하거나 "
                    f"더 작은 모델을 쓰는 편이 빠릅니다."
                )

            self._log(
                f"  🧹 [메모리 압박] '{slot.label}'({slot.est_gb:.1f} GB) 적재를 "
                f"위해 '{cand.label}'({cand.est_gb:.1f} GB) 를 먼저 반납합니다."
            )
            if self.release(cand.name):
                evicted.append(cand.name)
                self.evictions.append({
                    "for": name,
                    "evicted": cand.name,
                    "need_gb": round(need, 2),
                    "ram_need_gb": round(ram_need, 2),
                })

        if _ram_free_gb() > 0.0 and _ram_free_gb() < ram_need:
            try:
                from .memory import make_room

                def _step() -> bool:
                    dropped = self.release_next(protect=protect_set)
                    if dropped:
                        evicted.append(dropped)
                        self.evictions.append({
                            "for": name,
                            "evicted": dropped,
                            "need_gb": round(need, 2),
                            "ram_need_gb": round(ram_need, 2),
                            "staged": True,
                        })
                    return bool(dropped)

                make_room(
                    ram_need,
                    release_fn=_step,
                    log=self._log,
                    label=f"'{slot.label}' 적재",
                )
            except Exception as e:
                self._log(f"  ⏭ 단계적 RAM 확보 생략 ({e})")

        after = _vram_free_gb()
        ram_after = _ram_free_gb()

        if after < need:
            self._log(
                f"  ⚠️ VRAM 여유 {after:.2f} GB < 필요 {need:.2f} GB. "
                f"'{slot.label}' 로드가 실패할 수 있습니다."
            )

        if ram_after > 0.0 and ram_after < ram_need:
            key = f"ram:{name}"
            if key not in self._thrash_warned:
                self._thrash_warned.add(key)
                self._log(
                    f"  ⚠️ 시스템 RAM 여유 {ram_after:.1f} GB < 권장 "
                    f"{ram_need:.1f} GB ('{slot.label}'). 로드가 느려지거나 "
                    f"실패할 수 있습니다. 같은 경고는 이후 생략합니다."
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
            ram_before = _ram_free_gb()

            def _guarded():
                try:
                    from .memory import RamWatchdog
                except Exception:
                    return slot.acquire()
                with RamWatchdog(label=slot.label, log=self._log) as g:
                    got = slot.acquire()
                    if got is None:
                        g.check(settled=True)
                        return None
                    if g.tripped:
                        self._log(
                            f"  🩺 [RAM GUARD] {slot.label} 적재는 끝났습니다. "
                            f"회수 후 커밋 여유를 다시 재어 실제로 위험한지 "
                            f"판정합니다."
                        )
                        _empty_cache()
                        g.check(settled=True)
                    return got

            try:
                obj = _guarded()
                slot.last_error = ""
            except Exception as e:
                slot.last_error = str(e)
                slot.instance = None
                _empty_cache()

                purged: List[str] = []
                while True:
                    dropped = self.release_next(protect=[name])
                    if not dropped:
                        break
                    purged.append(dropped)

                if purged:
                    self._log(
                        f"  🔁 [{slot.label}] 1차 적재 실패 — 상주 슬롯 "
                        f"{len(purged)}개({', '.join(purged)})를 전부 비우고 "
                        f"한 번 더 시도합니다."
                    )
                    _empty_cache()
                    try:
                        obj = _guarded()
                        slot.last_error = ""
                        after = _vram_free_gb()
                        self._log(
                            f"  📥 [{slot.label}] 재시도 로드 성공 "
                            f"(VRAM {before:.2f} → {after:.2f} GB)"
                        )
                        return obj
                    except Exception as e2:
                        slot.last_error = str(e2)
                        slot.instance = None
                        _empty_cache()
                        e = e2

                self._log(f"  ❌ [{slot.label}] 로드 실패: {e}")
                ram_after = _ram_free_gb()
                if ram_before > 0.0:
                    self._log(
                        f"  🧹 [RAM] 실패한 적재분 회수 "
                        f"{ram_before:.1f} → {ram_after:.1f} GB"
                    )
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

    def release_next(self, protect: Sequence[str] = ()) -> str:
        with self._lock:
            protect_set = set(protect or ())
            pool = [
                s for s in self.slots.values()
                if s.loaded and s.name not in protect_set and s.est_gb > 0.0
            ]
            if not pool:
                return ""
            pool.sort(
                key=lambda s: (
                    1 if s.name in self.EVICT_LAST else 0,
                    -s.est_gb,
                )
            )
            target = pool[0]
        return target.name if self.release(target.name) else ""

    def release(self, name: str) -> bool:
        with self._lock:
            slot = self.slots.get(name)
            if slot is None or not slot.loaded:
                return False
            before = _vram_free_gb()
            ram_before = _ram_free_gb()
            ok = slot.release()

        if not ok:
            return False

        try:
            from .memory import reclaim
            reclaim(rounds=2, aggressive=True)
        except Exception:
            pass

        after = _vram_free_gb()
        ram_after = _ram_free_gb()

        if before > 0.0:
            tail = ""
            if ram_before > 0.0:
                tail = (
                    f" | RAM {ram_before:.1f} → {ram_after:.1f} GB "
                    f"(+{max(0.0, ram_after - ram_before):.1f} GB)"
                )
            self._log(
                f"  ♻️ [{slot.label}] 반납 "
                f"(VRAM {before:.2f} → {after:.2f} GB, "
                f"+{max(0.0, after - before):.2f} GB 확보){tail}"
            )
        else:
            self._log(f"  ♻️ [{slot.label}] 반납")
        return True

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
            deferred: List[str] = []
            keep = list(plan["keep"])
            for name in keep:
                slot = self.slots.get(name)
                if slot is None:
                    continue
                if not slot.loaded:
                    if slot.lazy:
                        deferred.append(slot.label)
                        continue
                    try:
                        self.acquire(name, protect=keep)
                        acquired.append(name)
                    except Exception as e:
                        self._log(f"  ⚠ [{slot.label}] 로드 실패: {e}")

            if deferred:
                self._log(
                    f"  ⏳ [LAZY KEEP] {', '.join(deferred)} 는 페이즈 전환에서 "
                    f"올리지 않습니다. 처음 실제로 쓰는 시점에 적재해야 커밋 "
                    f"사전 점검과 실측 커밋 비용 기록이 작동합니다."
                )

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
                "ram_free_gb": round(_ram_free_gb(), 2),
                "slots": {n: s.stats() for n, s in self.slots.items()},
                "transitions": len(self.transitions),
                "evictions": len(self.evictions),
            }

    def report_lines(self) -> List[str]:
        s = self.stats()
        lines = [
            f"  페이즈: {s['phase']} | 전환 {s['transitions']}회 "
            f"| VRAM 여유 {s['vram_free_gb']:.2f} GB "
            f"| RAM 여유 {s['ram_free_gb']:.1f} GB"
        ]
        for name, info in s["slots"].items():
            mark = "ON " if info["loaded"] else "off"
            lines.append(
                f"    [{mark}] {info['label']:<24} "
                f"로드 {info['loads']}회 / 반납 {info['releases']}회"
            )
        return lines