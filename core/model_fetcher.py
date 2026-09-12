import json
import os
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

from .lang_codes import iso1_of, language_name, normalize_lang_code
from .model_manager import (
    DEFAULT_LANGUAGE,
    LANG_MODEL_KINDS,
    LANG_REPO_TEMPLATES,
    MODELS_ROOT,
    STANZA_CORE_PROCESSORS,
    STANZA_OPTIONAL_PROCESSORS,
    STANZA_RESOURCE_FILE,
    STANZA_ROOT,
    lang_all_files,
    lang_model_dir,
    lang_model_ready,
    lang_optional_files,
    lang_repo_id,
    lang_required_files,
    missing_lang_models,
    save_active_language,
    stanza_lang_dir,
    stanza_ready,
    stanza_repo_id,
    stanza_resources_path,
)

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN") or ""

CHUNK_SIZE = 4 * 1024 * 1024
HEAD_TIMEOUT = 15
GET_TIMEOUT = 60
MAX_RETRY = 3


class DownloadCancelled(RuntimeError):
    pass


def _headers(extra: Optional[dict] = None) -> dict:
    h = {"User-Agent": "nms-ocr/1.0 (+language-model-fetcher)"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    if extra:
        h.update(extra)
    return h


def file_url(kind: str, code: str, filename: str) -> str:
    return f"{HF_ENDPOINT}/{lang_repo_id(kind, code)}/resolve/main/{filename}"


def probe_file(kind: str, code: str, filename: str, timeout: int = HEAD_TIMEOUT) -> Optional[int]:
    url = file_url(kind, code, filename)
    try:
        resp = requests.head(url, headers=_headers(), allow_redirects=True, timeout=timeout)
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    size = resp.headers.get("x-linked-size") or resp.headers.get("content-length")
    try:
        return int(size) if size is not None else 0
    except (TypeError, ValueError):
        return 0


def probe_repo(kind: str, code: str, timeout: int = HEAD_TIMEOUT) -> bool:
    for fname in lang_required_files(kind):
        if probe_file(kind, code, fname, timeout=timeout) is None:
            return False
    return True


def probe_language(code: str, timeout: int = HEAD_TIMEOUT) -> Dict[str, bool]:
    return {kind: probe_repo(kind, code, timeout=timeout) for kind in LANG_MODEL_KINDS}


def resolve_available_language(
    code: str,
    fallback: str = DEFAULT_LANGUAGE,
    log: Optional[Callable[[str], None]] = None,
) -> str:
    def _say(m: str):
        if log:
            try:
                log(m)
            except Exception:
                pass

    code = normalize_lang_code(code)

    if all(lang_model_ready(k, code) for k in LANG_MODEL_KINDS):
        _say(f"  ✅ '{code}' 언어 모델이 이미 로컬에 준비되어 있습니다.")
        return code

    probes = probe_language(code)
    if all(probes.values()):
        _say(f"  🌐 원격에 '{code}' 언어 모델 3종이 모두 존재합니다.")
        return code

    missing = [k for k, ok in probes.items() if not ok]
    _say(
        f"  ⚠ '{code}'({language_name(code)}) 저장소 중 "
        f"{', '.join(missing)} 를 찾지 못했습니다."
    )

    fallback = normalize_lang_code(fallback)
    if fallback == code:
        return code

    if all(lang_model_ready(k, fallback) for k in LANG_MODEL_KINDS):
        _say(f"  ↩ 기본 언어 '{fallback}' 로 폴백합니다. (로컬 준비 완료)")
        return fallback

    if all(probe_language(fallback).values()):
        _say(f"  ↩ 기본 언어 '{fallback}' 로 폴백합니다.")
        return fallback

    _say(f"  ❌ 기본 언어 '{fallback}' 역시 확보할 수 없습니다.")
    return fallback


class LanguageModelFetcher:
    def __init__(self, progress_callback: Optional[Callable[[dict], None]] = None):
        self._cancel = threading.Event()
        self._progress_callback = progress_callback
        self._downloading = False
        self._lock = threading.Lock()

    def set_progress_callback(self, cb: Callable[[dict], None]):
        self._progress_callback = cb

    def cancel(self):
        self._cancel.set()

    def reset(self):
        self._cancel.clear()

    @property
    def is_downloading(self) -> bool:
        return self._downloading

    def _emit(
        self,
        status: str,
        kind: str,
        code: str,
        filename: str,
        downloaded: int,
        total: int,
        label: str,
        extra: Optional[dict] = None,
    ):
        if self._progress_callback is None:
            return
        percent = 0
        if total > 0:
            percent = min(100, int(downloaded * 100 / total))
        payload = {
            "status": status,
            "kind": kind,
            "code": code,
            "file": filename,
            "label": label,
            "downloaded": downloaded,
            "total": total,
            "percent": percent,
        }
        if extra:
            payload.update(extra)
        try:
            self._progress_callback(payload)
        except Exception:
            pass

    def _download_file(self, kind: str, code: str, filename: str, label: str) -> bool:
        target_dir = lang_model_dir(kind, code)
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        part = target_dir / (filename + ".part")

        remote_size = probe_file(kind, code, filename)
        optional = filename in lang_optional_files(kind)

        if remote_size is None:
            if optional:
                self._emit("skip", kind, code, filename, 0, 0, f"{label} — {filename} 없음(선택)")
                return True
            self._emit("error", kind, code, filename, 0, 0, f"{label} — {filename} 원격에 없음")
            return False

        if target.exists():
            local = target.stat().st_size
            if remote_size == 0 or local == remote_size:
                self._emit("exists", kind, code, filename, local, local, f"{label} — {filename} 준비됨")
                return True
            target.unlink()

        resume_from = part.stat().st_size if part.exists() else 0
        if remote_size and resume_from > remote_size:
            part.unlink()
            resume_from = 0

        for attempt in range(1, MAX_RETRY + 1):
            if self._cancel.is_set():
                raise DownloadCancelled()

            headers = _headers()
            mode = "wb"
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
                mode = "ab"

            try:
                resp = requests.get(
                    file_url(kind, code, filename),
                    headers=headers,
                    stream=True,
                    timeout=GET_TIMEOUT,
                    allow_redirects=True,
                )
                if resp.status_code == 416:
                    resume_from = 0
                    if part.exists():
                        part.unlink()
                    continue
                if resp.status_code not in (200, 206):
                    raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}")
                if resume_from > 0 and resp.status_code == 200:
                    resume_from = 0
                    mode = "wb"

                downloaded = resume_from
                total = remote_size if remote_size else (downloaded + int(resp.headers.get("content-length", 0)))

                self._emit("start", kind, code, filename, downloaded, total, f"{label} — {filename}")

                last_emit = 0.0
                with open(part, mode) as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if self._cancel.is_set():
                            raise DownloadCancelled()
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_emit >= 0.2:
                            last_emit = now
                            self._emit(
                                "progress", kind, code, filename,
                                downloaded, total, f"{label} — {filename}",
                            )

                if remote_size and part.stat().st_size != remote_size:
                    resume_from = part.stat().st_size
                    raise requests.exceptions.RequestException("불완전 전송")

                part.replace(target)
                self._emit(
                    "file_done", kind, code, filename,
                    target.stat().st_size, target.stat().st_size,
                    f"{label} — {filename} 완료",
                )
                return True

            except DownloadCancelled:
                raise
            except requests.exceptions.RequestException as e:
                if attempt >= MAX_RETRY:
                    self._emit(
                        "error", kind, code, filename, 0, 0,
                        f"{label} — {filename} 실패: {e}",
                    )
                    return optional
                resume_from = part.stat().st_size if part.exists() else 0
                self._emit(
                    "retry", kind, code, filename, resume_from, remote_size or 0,
                    f"{label} — {filename} 재시도 {attempt}/{MAX_RETRY}",
                )
                time.sleep(1.0 * attempt)

        return optional

    def download_kind(self, kind: str, code: str) -> bool:
        tpl = LANG_REPO_TEMPLATES[kind]
        label = f"{tpl['label']} [{code}]"

        if lang_model_ready(kind, code):
            self._emit("ready", kind, code, "", 0, 0, f"{label} 이미 준비됨")
            return True

        self._emit("repo_start", kind, code, "", 0, 0, f"{label} 다운로드 시작")

        ok = True
        for filename in lang_all_files(kind):
            if not self._download_file(kind, code, filename, label):
                ok = False
                if filename in lang_required_files(kind):
                    break

        if ok and lang_model_ready(kind, code):
            self._emit("repo_done", kind, code, "", 0, 0, f"{label} 준비 완료")
            return True

        self._emit("repo_error", kind, code, "", 0, 0, f"{label} 준비 실패")
        return False

    def download_language(self, code: str) -> dict:
        with self._lock:
            if self._downloading:
                return {"ok": False, "error": "이미 다운로드가 진행 중입니다.", "code": code}
            self._downloading = True

        self._cancel.clear()
        code = normalize_lang_code(code)
        result = {"ok": False, "code": code, "done": [], "failed": []}

        try:
            self._emit(
                "lang_start", "", code, "", 0, 0,
                f"'{code}' ({language_name(code)}) 언어 모델 준비 시작",
            )
            for kind in LANG_MODEL_KINDS:
                if self._cancel.is_set():
                    raise DownloadCancelled()
                if self.download_kind(kind, code):
                    result["done"].append(kind)
                else:
                    result["failed"].append(kind)

            result["ok"] = not result["failed"]
            if result["ok"]:
                save_active_language(code, {"source": "fetcher"})
                self._emit(
                    "lang_done", "", code, "", 0, 0,
                    f"'{code}' 언어 모델 3종 준비 완료",
                )
            else:
                self._emit(
                    "lang_error", "", code, "", 0, 0,
                    f"'{code}' 언어 모델 준비 실패: {', '.join(result['failed'])}",
                )
        except DownloadCancelled:
            result["error"] = "사용자가 취소했습니다."
            self._emit("cancelled", "", code, "", 0, 0, "다운로드 취소됨")
        except Exception as e:
            result["error"] = str(e)
            self._emit("lang_error", "", code, "", 0, 0, f"오류: {e}")
        finally:
            self._downloading = False

        return result

    def ensure_language(
        self,
        code: str,
        fallback: str = DEFAULT_LANGUAGE,
        log: Optional[Callable[[str], None]] = None,
    ) -> dict:
        code = normalize_lang_code(code)

        missing = missing_lang_models(code)
        if not missing:
            save_active_language(code, {"source": "local"})
            return {"ok": True, "code": code, "downloaded": False}

        resolved = resolve_available_language(code, fallback=fallback, log=log)
        if resolved != code:
            missing = missing_lang_models(resolved)
            if not missing:
                save_active_language(resolved, {"source": "local-fallback", "requested": code})
                return {"ok": True, "code": resolved, "downloaded": False, "fallback_from": code}

        res = self.download_language(resolved)
        res["fallback_from"] = code if resolved != code else ""
        res["downloaded"] = True
        return res

    def start_language_download(self, code: str) -> dict:
        if self._downloading:
            return {"ok": False, "error": "이미 다운로드 중입니다."}
        t = threading.Thread(target=self.download_language, args=(code,), daemon=True)
        t.start()
        return {"ok": True, "code": normalize_lang_code(code)}


def stanza_file_url(iso1: str, rel_path: str) -> str:
    rel = rel_path.lstrip("/")
    return f"{HF_ENDPOINT}/{stanza_repo_id(iso1)}/resolve/main/{rel}"


def probe_stanza_file(iso1: str, rel_path: str, timeout: int = HEAD_TIMEOUT) -> Optional[int]:
    try:
        resp = requests.head(
            stanza_file_url(iso1, rel_path),
            headers=_headers(),
            allow_redirects=True,
            timeout=timeout,
        )
    except requests.exceptions.RequestException:
        return None
    if resp.status_code != 200:
        return None
    size = resp.headers.get("x-linked-size") or resp.headers.get("content-length")
    try:
        return int(size) if size is not None else 0
    except (TypeError, ValueError):
        return 0


def probe_stanza(iso1: str, timeout: int = HEAD_TIMEOUT) -> bool:
    return probe_stanza_file(iso1, STANZA_RESOURCE_FILE, timeout=timeout) is not None


def _collect_stanza_targets(resources: dict, iso1: str) -> List[Tuple[str, str]]:
    lang_node = resources.get(iso1) or {}
    if not isinstance(lang_node, dict):
        return []

    defaults = lang_node.get("default_processors") or {}
    dep_map = lang_node.get("default_dependencies") or {}
    pretrain = lang_node.get("default_pretrain") or {}

    wanted = list(STANZA_CORE_PROCESSORS) + list(STANZA_OPTIONAL_PROCESSORS)

    targets: List[Tuple[str, str]] = []
    seen = set()

    def _push(processor: str, package: str):
        if not processor or not package:
            return
        rel = f"models/{processor}/{package}.pt"
        if rel in seen:
            return
        seen.add(rel)
        targets.append((processor, rel))

    for proc in wanted:
        pkg = defaults.get(proc)
        if isinstance(pkg, str):
            _push(proc, pkg)

        deps = dep_map.get(proc)
        if isinstance(deps, list):
            for d in deps:
                if not isinstance(d, dict):
                    continue
                dm = d.get("model")
                dp = d.get("package")
                if isinstance(dm, str) and isinstance(dp, str):
                    _push(dm, dp)

    for proc, pkg in pretrain.items():
        if isinstance(pkg, str):
            _push("pretrain", pkg)

    return targets


class StanzaFetcher:
    def __init__(self, progress_callback: Optional[Callable[[dict], None]] = None):
        self._cancel = threading.Event()
        self._progress_callback = progress_callback
        self._downloading = False

    def set_progress_callback(self, cb: Callable[[dict], None]):
        self._progress_callback = cb

    def cancel(self):
        self._cancel.set()

    @property
    def is_downloading(self) -> bool:
        return self._downloading

    def _emit(self, status: str, iso1: str, rel: str, downloaded: int, total: int, label: str):
        if self._progress_callback is None:
            return
        percent = 0
        if total > 0:
            percent = min(100, int(downloaded * 100 / total))
        try:
            self._progress_callback({
                "status": status,
                "kind": "stanza",
                "code": iso1,
                "file": rel,
                "label": label,
                "downloaded": downloaded,
                "total": total,
                "percent": percent,
            })
        except Exception:
            pass

    def _download_rel(self, iso1: str, rel: str, label: str, optional: bool = False) -> bool:
        target = stanza_lang_dir(iso1) / Path(rel).relative_to("models") \
            if rel.startswith("models/") else stanza_lang_dir(iso1) / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_suffix(target.suffix + ".part")

        remote_size = probe_stanza_file(iso1, rel)
        if remote_size is None:
            if optional:
                self._emit("skip", iso1, rel, 0, 0, f"{label} — {rel} 없음(선택)")
                return True
            self._emit("error", iso1, rel, 0, 0, f"{label} — {rel} 원격에 없음")
            return False

        if target.exists():
            local = target.stat().st_size
            if remote_size == 0 or local == remote_size:
                self._emit("exists", iso1, rel, local, local, f"{label} — {rel} 준비됨")
                return True
            target.unlink()

        resume_from = part.stat().st_size if part.exists() else 0
        if remote_size and resume_from > remote_size:
            part.unlink()
            resume_from = 0

        for attempt in range(1, MAX_RETRY + 1):
            if self._cancel.is_set():
                raise DownloadCancelled()

            headers = _headers()
            mode = "wb"
            if resume_from > 0:
                headers["Range"] = f"bytes={resume_from}-"
                mode = "ab"

            try:
                resp = requests.get(
                    stanza_file_url(iso1, rel),
                    headers=headers,
                    stream=True,
                    timeout=GET_TIMEOUT,
                    allow_redirects=True,
                )
                if resp.status_code == 416:
                    resume_from = 0
                    if part.exists():
                        part.unlink()
                    continue
                if resp.status_code not in (200, 206):
                    raise requests.exceptions.HTTPError(f"HTTP {resp.status_code}")
                if resume_from > 0 and resp.status_code == 200:
                    resume_from = 0
                    mode = "wb"

                downloaded = resume_from
                total = remote_size or (downloaded + int(resp.headers.get("content-length", 0)))
                self._emit("start", iso1, rel, downloaded, total, f"{label} — {rel}")

                last = 0.0
                with open(part, mode) as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if self._cancel.is_set():
                            raise DownloadCancelled()
                        if not chunk:
                            continue
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last >= 0.2:
                            last = now
                            self._emit("progress", iso1, rel, downloaded, total, f"{label} — {rel}")

                if remote_size and part.stat().st_size != remote_size:
                    resume_from = part.stat().st_size
                    raise requests.exceptions.RequestException("불완전 전송")

                part.replace(target)
                self._emit(
                    "file_done", iso1, rel,
                    target.stat().st_size, target.stat().st_size,
                    f"{label} — {rel} 완료",
                )
                return True

            except DownloadCancelled:
                raise
            except requests.exceptions.RequestException as e:
                if attempt >= MAX_RETRY:
                    self._emit("error", iso1, rel, 0, 0, f"{label} — {rel} 실패: {e}")
                    return optional
                resume_from = part.stat().st_size if part.exists() else 0
                self._emit("retry", iso1, rel, resume_from, remote_size or 0,
                           f"{label} — {rel} 재시도 {attempt}/{MAX_RETRY}")
                time.sleep(1.0 * attempt)

        return optional

    def download(self, iso1: str) -> dict:
        if self._downloading:
            return {"ok": False, "error": "이미 다운로드 중입니다.", "iso1": iso1}

        self._downloading = True
        self._cancel.clear()
        label = f"Stanza NLP [{iso1}]"
        result = {"ok": False, "iso1": iso1, "files": 0}

        try:
            if stanza_ready(iso1):
                self._emit("ready", iso1, "", 0, 0, f"{label} 이미 준비됨")
                result["ok"] = True
                return result

            self._emit("repo_start", iso1, "", 0, 0, f"{label} 다운로드 시작")

            if not self._download_rel(iso1, STANZA_RESOURCE_FILE, label):
                self._emit("repo_error", iso1, "", 0, 0, f"{label} resources.json 취득 실패")
                return result

            try:
                with open(stanza_resources_path(iso1), "r", encoding="utf-8") as f:
                    resources = json.load(f)
            except Exception as e:
                self._emit("repo_error", iso1, "", 0, 0, f"{label} resources.json 파싱 실패: {e}")
                return result

            targets = _collect_stanza_targets(resources, iso1)
            if not targets:
                self._emit("repo_error", iso1, "", 0, 0,
                           f"{label} resources.json 에서 '{iso1}' 노드를 찾지 못했습니다.")
                return result

            done = 0
            for proc, rel in targets:
                if self._cancel.is_set():
                    raise DownloadCancelled()
                optional = proc in STANZA_OPTIONAL_PROCESSORS
                if self._download_rel(iso1, rel, label, optional=optional):
                    done += 1

            result["files"] = done
            result["ok"] = stanza_ready(iso1)

            if result["ok"]:
                self._emit("repo_done", iso1, "", 0, 0, f"{label} 준비 완료 ({done}개 파일)")
            else:
                self._emit("repo_error", iso1, "", 0, 0, f"{label} 필수 프로세서 누락")

        except DownloadCancelled:
            result["error"] = "사용자가 취소했습니다."
            self._emit("cancelled", iso1, "", 0, 0, "다운로드 취소됨")
        except Exception as e:
            result["error"] = str(e)
            self._emit("repo_error", iso1, "", 0, 0, f"오류: {e}")
        finally:
            self._downloading = False

        return result

    def ensure(self, iso1: str, log: Optional[Callable[[str], None]] = None) -> dict:
        def _say(m: str):
            if log:
                try:
                    log(m)
                except Exception:
                    pass

        if stanza_ready(iso1):
            _say(f"  ✅ Stanza '{iso1}' 모델이 이미 준비되어 있습니다.")
            return {"ok": True, "iso1": iso1, "downloaded": False}

        if not probe_stanza(iso1):
            _say(f"  ⚠ stanfordnlp/stanza-{iso1} 저장소를 찾지 못했습니다. NLP 게이트를 건너뜁니다.")
            return {"ok": False, "iso1": iso1, "error": "repo not found"}

        res = self.download(iso1)
        res["downloaded"] = True
        return res


def ensure_stanza_models(
    iso1: str,
    progress_callback: Optional[Callable[[dict], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    return StanzaFetcher(progress_callback=progress_callback).ensure(iso1, log=log)


def ensure_language_models(
    code: str,
    fallback: str = DEFAULT_LANGUAGE,
    progress_callback: Optional[Callable[[dict], None]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> dict:
    fetcher = LanguageModelFetcher(progress_callback=progress_callback)
    return fetcher.ensure_language(code, fallback=fallback, log=log)