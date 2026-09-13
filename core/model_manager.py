import shutil
from pathlib import Path
from typing import Dict, List, Optional

from .lang_codes import normalize_lang_code

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_ROOT = BASE_DIR / "models"

MODELS_DIR = MODELS_ROOT

LLM_PATH = MODELS_ROOT / "alphaedge-ai"
VISION_ENC_PATH = MODELS_ROOT / "ax-ve"
OCR_PATH = MODELS_ROOT / "ppocr"
PPOCR_DET_PATH = MODELS_ROOT / "PP-OCRv5_mobile_det_safetensors"

SIGLIP2_HUB_REF = "google/siglip2-base-patch16-naflex"
SIGLIP2_LOCAL_DIR = MODELS_ROOT / "siglip2-base-patch16-naflex"

WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf")
ASSET_SUFFIXES = (".json", ".py", ".txt", ".model", ".tiktoken", ".jinja", ".yaml", ".yml")


class MissingModelError(RuntimeError):
    pass


MODEL_SPECS: Dict[str, dict] = {
    "alphaedge-ai": {
        "label": "AlphaEdge AI (선택 · 로컬 배치 전용)",
        "role": "optional",
        "dir": LLM_PATH,
        "asset_source": None,
        "min_size": 100_000_000,
        "repo": "",
        "files": (),
        "required": (),
        "optional": (),
        "manual_only": True,
        "note": (
            "원격 저장소가 없습니다. 사용하려면 models/alphaedge-ai/ 에 "
            "직접 배치하세요. 배치하지 않으면 언어 스코프 모델"
            "(Qwen3-Embedding / Qwen3.5-2B)이 그 역할을 대신합니다."
        ),
    },
    "ax-ve": {
        "label": "A.X-VE 비전 인코더",
        "role": "vision",
        "dir": VISION_ENC_PATH,
        "asset_source": BASE_DIR / "ax-ve",
        "min_size": 10_000_000,
        "repo": "skt/A.X-VE",
        "files": (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "configuration_ax_ve.py",
            "modeling_ax_ve.py",
            "image_processing_ax_ve.py",
            "processing_ax_ve.py",
        ),
        "required": ("config.json", "model.safetensors"),
        "optional": ("processing_ax_ve.py",),
    },
    "ppocr-det": {
        "label": "PP-OCRv5 mobile det (텍스트 영역 검출)",
        "role": "ocr",
        "dir": PPOCR_DET_PATH,
        "asset_source": None,
        "min_size": 1_000_000,
        "repo": "PaddlePaddle/PP-OCRv5_mobile_det_safetensors",
        "files": (
            "config.json",
            "inference.yml",
            "model.safetensors",
            "preprocessor_config.json",
        ),
        "required": ("inference.yml", "model.safetensors"),
        "optional": ("config.json", "preprocessor_config.json"),
    },
    "siglip2-naflex": {
        "label": "SigLIP2 NaFlex (Hayai 비전 설정)",
        "role": "support",
        "dir": SIGLIP2_LOCAL_DIR,
        "asset_source": None,
        "min_size": 0,
        "repo": "google/siglip2-base-patch16-naflex",
        "files": (
            "config.json",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ),
        "required": ("config.json", "preprocessor_config.json"),
        "optional": ("special_tokens_map.json", "tokenizer_config.json"),
    },
}

MODEL_KEYS = tuple(MODEL_SPECS.keys())

CORE_MODEL_KEYS = ("ax-ve",)

OPTIONAL_MODEL_KEYS = ("alphaedge-ai",)


def model_repo_id(key: str) -> str:
    return str(_spec(key).get("repo", "") or "")


def is_manual_only(key: str) -> bool:
    return bool(_spec(key).get("manual_only", False))


def model_note(key: str) -> str:
    return str(_spec(key).get("note", "") or "")


def model_all_files(key: str) -> tuple:
    return tuple(_spec(key).get("files", ()) or ())


def model_required_files(key: str) -> tuple:
    return tuple(_spec(key).get("required", ()) or ())


def model_optional_files(key: str) -> tuple:
    return tuple(_spec(key).get("optional", ()) or ())


def delete_model(key: str) -> dict:
    spec = _spec(key)
    target: Path = spec["dir"]
    removed = 0
    freed = 0

    if not target.is_dir():
        return {"ok": True, "key": key, "removed": 0, "freed": 0}

    for p in sorted(target.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix not in WEIGHT_SUFFIXES:
            continue
        try:
            freed += p.stat().st_size
            p.unlink()
            removed += 1
        except Exception:
            continue

    for p in sorted(target.rglob("*.part")):
        try:
            p.unlink()
        except Exception:
            continue

    return {"ok": True, "key": key, "removed": removed, "freed": freed}


def delete_lang_model(kind: str, code: str) -> dict:
    import shutil

    d = lang_model_dir(kind, code)
    if not d.is_dir():
        return {"ok": True, "kind": kind, "code": code, "freed": 0}

    freed = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    try:
        shutil.rmtree(d)
        return {"ok": True, "kind": kind, "code": code, "freed": freed}
    except Exception as e:
        return {"ok": False, "kind": kind, "code": code, "error": str(e)}


def delete_stanza(iso1: str) -> dict:
    import shutil

    d = stanza_lang_dir(iso1)
    if not d.is_dir():
        return {"ok": True, "iso1": iso1, "freed": 0}

    freed = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
    try:
        shutil.rmtree(d)
        return {"ok": True, "iso1": iso1, "freed": freed}
    except Exception as e:
        return {"ok": False, "iso1": iso1, "error": str(e)}


DEFAULT_LANGUAGE = "eng"

BOOTSTRAP_LANGUAGES = ("eng", "kor")

ACTIVE_LANGUAGE_FILE = MODELS_ROOT / ".active_language.json"

LANG_MODEL_KINDS = ("ppocr", "siglip2", "qwen3emb", "qwen35")

LANG_KIND_PRIORITY = ("ppocr", "qwen3emb", "siglip2", "qwen35")

LANG_REPO_TEMPLATES: Dict[str, dict] = {
    "ppocr": {
        "label": "PP-OCRv5 mobile rec (텍스트 라인 인식)",
        "owner": "PaddlePaddle",
        "repo": "{prefix}PP-OCRv5_mobile_rec_safetensors",
        "slug": "paddle_rec",
        "files": (
            "config.json",
            "inference.yml",
            "model.safetensors",
            "preprocessor_config.json",
        ),
        "required": ("inference.yml", "model.safetensors"),
        "optional": ("config.json", "preprocessor_config.json"),
        "min_size": 3_000_000,
    },
    "siglip2": {
        "label": "SigLIP2 Large (언어 텍스트 타워)",
        "owner": "alphaedge-ai",
        "repo": "siglip2-large-patch16-512-{code}-16384",
        "files": (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer.model",
            "tokenizer_config.json",
        ),
        "required": ("config.json", "model.safetensors"),
        "optional": ("tokenizer.model",),
        "min_size": 50_000_000,
    },
    "qwen3emb": {
        "label": "Qwen3 Embedding (텍스트 임베딩)",
        "owner": "alphaedge-ai",
        "repo": "Qwen3-Embedding-{code}-16384",
        "files": (
            "config.json",
            "config_sentence_transformers.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors",
            "modules.json",
            "tokenizer.json",
            "vocab.json",
        ),
        "required": ("config.json", "model.safetensors", "tokenizer.json"),
        "optional": ("generation_config.json", "modules.json", "config_sentence_transformers.json"),
        "min_size": 50_000_000,
    },
    "qwen35": {
        "label": "Qwen3.5 4B (정제 추출 LLM)",
        "owner": "alphaedge-ai",
        "repo": "Qwen3.5-4B-{code}-16384",
        "files": (
            "chat_template.jinja",
            "config.json",
            "merges.txt",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "video_preprocessor_config.json",
            "vocab.json",
        ),
        "required": ("config.json", "model.safetensors", "tokenizer.json"),
        "optional": (
            "chat_template.jinja",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
        ),
        "min_size": 400_000_000,
    },
}


PADDLE_OCR_DEFAULT_SLUG = "latin"

PADDLE_OCR_REPOS = (
    "", "korean", "en", "latin", "eslav", "cyrillic",
    "arabic", "devanagari", "ta", "te", "el", "th",
)

PADDLE_OCR_SLUG: Dict[str, str] = {
    "zho": "",
    "jpn": "",
    "kor": "korean",
    "eng": "en",
    "tha": "th",
    "ell": "el",
    "tam": "ta",
    "tel": "te",
    "rus": "eslav",
    "ukr": "eslav",
    "bel": "eslav",
    "bul": "cyrillic",
    "srp": "cyrillic",
    "mkd": "cyrillic",
    "kaz": "cyrillic",
    "mon": "cyrillic",
    "ara": "arabic",
    "fas": "arabic",
    "urd": "arabic",
    "hin": "devanagari",
    "mar": "devanagari",
    "nep": "devanagari",
    "ben": "devanagari",
}

PADDLE_OCR_SUBSTITUTED: Dict[str, str] = {
    "jpn": "PaddleOCR 에 일본어 전용 모델이 없어 한자 공용 모델(중국어)로 대체합니다.",
    "ben": "벵골 전용 모델이 없어 데바나가리 모델로 대체합니다.",
    "kan": "칸나다 전용 모델이 없어 라틴 모델로 대체합니다.",
}

PADDLE_OCR_SCRIPT_HINT: Dict[str, str] = {
    "": "Han (Chinese / Japanese kanji)",
    "korean": "Hangul (Korean)",
    "en": "Latin (English)",
    "latin": "Latin",
    "eslav": "Cyrillic (East Slavic)",
    "cyrillic": "Cyrillic",
    "arabic": "Arabic",
    "devanagari": "Devanagari",
    "ta": "Tamil",
    "te": "Telugu",
    "el": "Greek",
    "th": "Thai",
}


def paddle_ocr_slug(code: str) -> str:
    c = normalize_lang_code(code)
    slug = PADDLE_OCR_SLUG.get(c, PADDLE_OCR_DEFAULT_SLUG)
    return slug if slug in PADDLE_OCR_REPOS else PADDLE_OCR_DEFAULT_SLUG


def paddle_ocr_prefix(code: str) -> str:
    slug = paddle_ocr_slug(code)
    return f"{slug}_" if slug else ""


def paddle_ocr_note(code: str) -> str:
    return PADDLE_OCR_SUBSTITUTED.get(normalize_lang_code(code), "")


def paddle_ocr_script_hint(code: str) -> str:
    return PADDLE_OCR_SCRIPT_HINT.get(paddle_ocr_slug(code), "Latin")


def paddle_ocr_model_name(code: str) -> str:
    slug = paddle_ocr_slug(code)
    return f"{slug}_PP-OCRv5_mobile_rec" if slug else "PP-OCRv5_mobile_rec"


STANZA_OWNER = "stanfordnlp"
STANZA_REPO_TEMPLATE = "stanza-{iso1}"
STANZA_RESOURCE_FILE = "models/resources.json"
STANZA_RESOURCE_CANDIDATES = (
    "models/resources.json",
    "resources.json",
)
STANZA_ROOT = MODELS_ROOT / "stanza"
STANZA_MIN_SIZE = 1_000_000

STANZA_CORE_PROCESSORS = ("tokenize", "pos", "lemma", "depparse")
STANZA_OPTIONAL_PROCESSORS = ("mwt", "ner")


def stanza_repo_id(iso1: str) -> str:
    return f"{STANZA_OWNER}/{STANZA_REPO_TEMPLATE.format(iso1=iso1)}"


def stanza_lang_dir(iso1: str) -> Path:
    return STANZA_ROOT / iso1


def stanza_resources_path(iso1: str) -> Path:
    return stanza_lang_dir(iso1) / "resources.json"


def stanza_model_files(iso1: str) -> List[Path]:
    d = stanza_lang_dir(iso1)
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*.pt") if p.is_file())


def stanza_total_bytes(iso1: str) -> int:
    return sum(p.stat().st_size for p in stanza_model_files(iso1))


def stanza_ready(iso1: str) -> bool:
    if not stanza_resources_path(iso1).exists():
        return False
    files = stanza_model_files(iso1)
    if not files:
        return False
    has_tokenize = any("tokenize" in p.parts for p in files)
    return has_tokenize and stanza_total_bytes(iso1) >= STANZA_MIN_SIZE


def describe_stanza(iso1: str) -> dict:
    d = stanza_lang_dir(iso1)
    files = stanza_model_files(iso1)
    procs = sorted({p.parent.name for p in files})
    return {
        "iso1": iso1,
        "label": f"Stanza NLP [{iso1}]",
        "repo": stanza_repo_id(iso1),
        "dir": str(d),
        "root": str(STANZA_ROOT),
        "ready": stanza_ready(iso1),
        "resources": stanza_resources_path(iso1).exists(),
        "processors": procs,
        "files": len(files),
        "bytes": stanza_total_bytes(iso1),
    }


def format_stanza_report(iso1: str) -> str:
    info = describe_stanza(iso1)
    mark = "OK " if info["ready"] else "MISS"
    size_mb = info["bytes"] / (1024 * 1024)
    lines = [
        f"  [{mark}] {info['label']:<38} {size_mb:9.1f} MB  {info['repo']}",
        f"         프로세서: {', '.join(info['processors']) or '-'} "
        f"| 파일 {info['files']}개",
    ]
    return "\n".join(lines)


def installed_stanza_langs() -> List[str]:
    if not STANZA_ROOT.is_dir():
        return []
    out = []
    for p in sorted(STANZA_ROOT.iterdir()):
        if p.is_dir() and stanza_ready(p.name):
            out.append(p.name)
    return out


def _lang_template(kind: str) -> dict:
    tpl = LANG_REPO_TEMPLATES.get(kind)
    if tpl is None:
        raise KeyError(
            f"알 수 없는 언어 모델 종류: {kind} "
            f"(사용 가능: {', '.join(LANG_MODEL_KINDS)})"
        )
    return tpl


def lang_repo_code(kind: str, code: str) -> str:
    tpl = _lang_template(kind)
    if str(tpl.get("slug", "")) == "paddle_rec":
        return paddle_ocr_slug(code)
    return str(code)


def lang_repo_prefix(kind: str, code: str) -> str:
    tpl = _lang_template(kind)
    if str(tpl.get("slug", "")) == "paddle_rec":
        return paddle_ocr_prefix(code)
    return f"{code}_"


def lang_repo_name(kind: str, code: str) -> str:
    return _lang_template(kind)["repo"].format(
        code=lang_repo_code(kind, code),
        prefix=lang_repo_prefix(kind, code),
    )


def lang_repo_id(kind: str, code: str) -> str:
    tpl = _lang_template(kind)
    return f"{tpl['owner']}/{lang_repo_name(kind, code)}"


def lang_model_dir(kind: str, code: str) -> Path:
    return MODELS_ROOT / lang_repo_name(kind, code)


def lang_required_files(kind: str) -> tuple:
    return tuple(_lang_template(kind)["required"])


def lang_all_files(kind: str) -> tuple:
    return tuple(_lang_template(kind)["files"])


def lang_optional_files(kind: str) -> tuple:
    return tuple(_lang_template(kind).get("optional", ()))


def lang_model_ready(kind: str, code: str) -> bool:
    tpl = _lang_template(kind)
    d = lang_model_dir(kind, code)
    if not d.is_dir():
        return False
    for fname in tpl["required"]:
        p = d / fname
        if not p.exists() or p.stat().st_size <= 0:
            return False
    weights = [p for p in d.iterdir() if p.is_file() and p.suffix in WEIGHT_SUFFIXES]
    total = sum(p.stat().st_size for p in weights)
    return total >= tpl["min_size"]


def ppocr_model_dir(code: str) -> Path:
    return lang_model_dir("ppocr", code)


def ppocr_det_dir() -> Path:
    return PPOCR_DET_PATH


def ppocr_det_ready() -> bool:
    return is_model_ready("ppocr-det")


def ppocr_ready(code: str) -> bool:
    return lang_model_ready("ppocr", code)


def ppocr_ready_codes() -> List[str]:
    out: List[str] = []
    seen = set()
    for code in list(BOOTSTRAP_LANGUAGES) + installed_language_codes():
        c = normalize_lang_code(code)
        if c in seen:
            continue
        seen.add(c)
        if ppocr_ready(c):
            out.append(c)
    return out


def lang_models_ready(code: str) -> bool:
    return all(lang_model_ready(k, code) for k in LANG_MODEL_KINDS)


def missing_lang_models(code: str) -> List[str]:
    return [k for k in LANG_MODEL_KINDS if not lang_model_ready(k, code)]


def ensure_lang_model_dir(kind: str, code: str) -> Path:
    tpl = _lang_template(kind)
    d = lang_model_dir(kind, code)
    if not lang_model_ready(kind, code):
        raise MissingModelError(
            f"[{tpl['label']} / {code}] 언어 모델이 준비되지 않았습니다.\n"
            f"  기대 경로 : {d}\n"
            f"  원격 저장소 : {lang_repo_id(kind, code)}\n"
            f"  필수 파일 : {', '.join(tpl['required'])}"
        )
    return d


def describe_lang_model(kind: str, code: str) -> dict:
    tpl = _lang_template(kind)
    d = lang_model_dir(kind, code)
    present = []
    missing = []
    nbytes = 0
    for fname in tpl["files"]:
        p = d / fname
        if p.exists() and p.stat().st_size > 0:
            present.append(fname)
            nbytes += p.stat().st_size
        else:
            missing.append(fname)
    return {
        "kind": kind,
        "code": code,
        "label": tpl["label"],
        "repo": lang_repo_id(kind, code),
        "dir": str(d),
        "ready": lang_model_ready(kind, code),
        "present": present,
        "missing": missing,
        "bytes": nbytes,
    }


def describe_lang_models(code: str) -> dict:
    return {k: describe_lang_model(k, code) for k in LANG_MODEL_KINDS}


def format_lang_report(code: str) -> str:
    lines = [f"  언어 코드: {code}"]
    for kind in LANG_MODEL_KINDS:
        info = describe_lang_model(kind, code)
        mark = "OK " if info["ready"] else "MISS"
        size_mb = info["bytes"] / (1024 * 1024)
        lines.append(
            f"  [{mark}] {info['label']:<38} {size_mb:9.1f} MB  {info['repo']}"
        )
        if info["missing"]:
            lines.append(f"         누락: {', '.join(info['missing'])}")
    return "\n".join(lines)


def save_active_language(code: str, meta: Optional[dict] = None) -> None:
    import json
    import time

    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"code": code, "saved_at": int(time.time())}
    if meta:
        payload.update(meta)
    try:
        with open(ACTIVE_LANGUAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_active_language() -> Optional[dict]:
    import json

    if not ACTIVE_LANGUAGE_FILE.exists():
        return None
    try:
        with open(ACTIVE_LANGUAGE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def active_language_code(default: str = DEFAULT_LANGUAGE) -> str:
    info = load_active_language()
    if info and isinstance(info.get("code"), str) and info["code"]:
        return info["code"]
    return default


def installed_language_codes() -> List[str]:
    if not MODELS_ROOT.is_dir():
        return []
    tpl = _lang_template("siglip2")
    prefix = tpl["repo"].split("{code}")[0]
    suffix = tpl["repo"].split("{code}")[1]
    codes = []
    for p in sorted(MODELS_ROOT.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        code = name[len(prefix): len(name) - len(suffix)]
        if code and code not in codes:
            codes.append(code)
    return codes


def embedder_ready_codes() -> List[str]:
    out = []
    for code in installed_language_codes():
        if lang_model_ready("qwen3emb", code):
            out.append(code)
    for code in BOOTSTRAP_LANGUAGES:
        if code not in out and lang_model_ready("qwen3emb", code):
            out.append(code)
    return out


def any_embedder_ready() -> bool:
    return bool(embedder_ready_codes())


def bootstrap_ready(kinds: Optional[List[str]] = None) -> bool:
    kinds = list(kinds or ["qwen3emb"])
    for code in BOOTSTRAP_LANGUAGES:
        for kind in kinds:
            if not lang_model_ready(kind, code):
                return False
    return True


def missing_bootstrap(kinds: Optional[List[str]] = None) -> List[tuple]:
    kinds = list(kinds or ["qwen3emb"])
    out = []
    for code in BOOTSTRAP_LANGUAGES:
        for kind in kinds:
            if not lang_model_ready(kind, code):
                out.append((kind, code))
    return out


def resolve_working_codes(
    preferred: Optional[str] = None,
    resolved: bool = False,
) -> List[str]:
    codes: List[str] = []

    if resolved and preferred:
        p = normalize_lang_code(preferred)
        codes.append(p)

    for c in BOOTSTRAP_LANGUAGES:
        if c not in codes:
            codes.append(c)

    if not resolved and preferred:
        p = normalize_lang_code(preferred)
        if p not in codes:
            codes.insert(0, p)

    return codes


def describe_bootstrap() -> dict:
    out = {"languages": list(BOOTSTRAP_LANGUAGES), "entries": [], "ready": True}
    for code in BOOTSTRAP_LANGUAGES:
        for kind in LANG_KIND_PRIORITY:
            info = describe_lang_model(kind, code)
            info["scope"] = "bootstrap"
            info["id"] = f"lang:{kind}:{code}"
            out["entries"].append(info)
            if kind == "qwen3emb" and not info["ready"]:
                out["ready"] = False
    return out


def _spec(key: str) -> dict:
    spec = MODEL_SPECS.get(key)
    if spec is None:
        raise KeyError(f"알 수 없는 모델 키: {key} (사용 가능: {', '.join(MODEL_KEYS)})")
    return spec


def get_model_dir(key: str) -> Path:
    return _spec(key)["dir"]


def list_weight_files(key: str) -> List[Path]:
    d = get_model_dir(key)
    if not d.is_dir():
        return []
    found: List[Path] = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix in WEIGHT_SUFFIXES:
            found.append(p)
    return found


def get_model_path(key: str) -> Path:
    d = get_model_dir(key)
    primary = d / "model.safetensors"
    if primary.exists():
        return primary
    weights = list_weight_files(key)
    if weights:
        return weights[0]
    return primary


def total_weight_bytes(key: str) -> int:
    return sum(p.stat().st_size for p in list_weight_files(key))


def has_config(key: str) -> bool:
    return (get_model_dir(key) / "config.json").exists()


def is_model_ready(key: str) -> bool:
    spec = _spec(key)
    d: Path = spec["dir"]
    if not d.is_dir():
        return False

    if is_manual_only(key):
        if not (d / "config.json").exists():
            return False
        min_size = int(spec.get("min_size", 0) or 0)
        return total_weight_bytes(key) >= min_size

    for fname in model_required_files(key):
        p = d / fname
        if not p.exists() or p.stat().st_size <= 0:
            return False

    min_size = int(spec.get("min_size", 0) or 0)
    if min_size > 0 and total_weight_bytes(key) < min_size:
        return False

    return True


def missing_core_models() -> List[str]:
    return [k for k in CORE_MODEL_KEYS if not is_model_ready(k)]


def missing_models() -> List[str]:
    return [k for k in MODEL_KEYS if not is_model_ready(k)]


def sync_model_assets(key: str) -> Path:
    spec = _spec(key)
    target: Path = spec["dir"]
    target.mkdir(parents=True, exist_ok=True)

    src = spec.get("asset_source")
    if src is None:
        return target

    src = Path(src)
    if not src.is_dir():
        return target

    for p in sorted(src.iterdir()):
        if not p.is_file():
            continue
        if p.suffix in WEIGHT_SUFFIXES:
            continue
        if p.suffix not in ASSET_SUFFIXES:
            continue
        dst = target / p.name
        if dst.exists() and dst.stat().st_mtime >= p.stat().st_mtime:
            continue
        shutil.copy2(p, dst)

    return target


def _missing_message(key: str, reason: str) -> str:
    spec = _spec(key)
    return (
        f"[{spec['label']}] {reason}\n"
        f"  기대 경로 : {spec['dir']}\n"
        f"  필요 파일 : config.json + model.safetensors (샤드 파일 가능)\n"
        f"  모델 파일 배치는 사용자 책임입니다. 이 앱은 모델을 다운로드하지 않습니다."
    )


def ensure_model_dir(key: str) -> Path:
    spec = _spec(key)
    target: Path = spec["dir"]

    if not target.is_dir():
        raise MissingModelError(_missing_message(key, "모델 폴더가 존재하지 않습니다."))

    sync_model_assets(key)

    if total_weight_bytes(key) < spec["min_size"]:
        raise MissingModelError(
            _missing_message(key, "가중치 파일이 없거나 크기가 비정상입니다.")
        )

    if not has_config(key):
        raise MissingModelError(_missing_message(key, "config.json 을 찾지 못했습니다."))

    return target


def describe_model(key: str) -> dict:
    spec = _spec(key)
    d: Path = spec["dir"]

    present = []
    missing = []
    for fname in model_all_files(key):
        p = d / fname
        if p.exists() and p.stat().st_size > 0:
            present.append(fname)
        else:
            missing.append(fname)

    total_bytes = 0
    if d.is_dir():
        total_bytes = sum(p.stat().st_size for p in d.rglob("*") if p.is_file())

    return {
        "key": key,
        "label": spec["label"],
        "role": spec["role"],
        "core": key in CORE_MODEL_KEYS,
        "optional": key in OPTIONAL_MODEL_KEYS,
        "manual_only": is_manual_only(key),
        "note": model_note(key),
        "ready": is_model_ready(key),
        "config": has_config(key),
        "dir": str(d),
        "path": str(get_model_path(key)),
        "repo": model_repo_id(key),
        "bytes": total_bytes,
        "weight_bytes": total_weight_bytes(key),
        "present": present,
        "missing": missing,
        "asset_source": str(spec["asset_source"]) if spec.get("asset_source") else "",
    }


def check_all_models() -> dict:
    return {key: describe_model(key) for key in MODEL_KEYS}


def resolve_siglip2_ref() -> str:
    if SIGLIP2_LOCAL_DIR.is_dir() and (SIGLIP2_LOCAL_DIR / "config.json").exists():
        return str(SIGLIP2_LOCAL_DIR)
    return SIGLIP2_HUB_REF


def format_model_report() -> str:
    lines = []
    for key in MODEL_KEYS:
        info = describe_model(key)
        mark = "OK " if info["ready"] and info["config"] else "MISS"
        size_mb = info["bytes"] / (1024 * 1024)
        lines.append(
            f"  [{mark}] {info['label']:<38} {size_mb:9.1f} MB  {info['dir']}"
        )
    return "\n".join(lines)