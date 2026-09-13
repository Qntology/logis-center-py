import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .device import configure_backends, detect_accelerator, select_dtype
from .lang_codes import language_name, normalize_lang_code
from .line_reader import split_lines as split_text_lines
from .paddle_bootstrap import (
    configure_runtime,
    describe_kwargs,
    ensure_paddle,
    kwargs_variants,
    manual_command,
)
from .paddle_bootstrap import ready as paddle_ready
from .model_manager import (
    lang_repo_id,
    paddle_ocr_model_name,
    paddle_ocr_note,
    paddle_ocr_script_hint,
    paddle_ocr_slug,
    ppocr_det_dir,
    ppocr_det_ready,
    ppocr_model_dir,
    ppocr_ready,
)

REC_HEIGHT = 48
REC_MAX_WIDTH = 320
REC_MIN_WIDTH = 16
REC_MEAN = 0.5
REC_STD = 0.5

LINE_MIN_HEIGHT = 6
LINE_PAD_Y = 2
LINE_MAX = 32
REC_BATCH = 16

DET_LIMIT_SIDE = 960
DET_THRESH = 0.30
DET_BOX_THRESH = 0.50
DET_UNCLIP = 1.6
DET_MIN_AREA = 24
DET_MEAN = (0.485, 0.456, 0.406)
DET_STD = (0.229, 0.224, 0.225)

DET_MAX_FAILURES = 2
REC_MAX_FAILURES = 2

LIST_ITEM_RE = re.compile(r"^(\s*)-\s?(.*)$")
CHAR_KEY_RE = re.compile(r"^(\s*)character_dict\s*:\s*(.*)$")
SPACE_FLAG_RE = re.compile(r"use_space_char\s*:\s*([A-Za-z0-9]+)")
SHAPE_RE = re.compile(r"rec_image_shape\s*:\s*\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

BOOL_TRUE = ("true", "yes", "on", "1")


def _unquote(raw: str) -> str:
    s = str(raw or "").rstrip("\r\n").strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        body = s[1:-1]
        if s[0] == "'":
            return body.replace("''", "'")
        if "\\" in body:
            try:
                return json.loads(s)
            except Exception:
                return body
        return body
    return s


def _charset_from_text(text: str) -> List[str]:
    lines = text.splitlines()
    start = -1
    indent = 0
    inline = ""

    for i, line in enumerate(lines):
        m = CHAR_KEY_RE.match(line)
        if m:
            start = i
            indent = len(m.group(1))
            inline = m.group(2).strip()
            break

    if start < 0:
        return []

    if inline.startswith("["):
        body = inline.strip("[]")
        return [_unquote(p) for p in body.split(",") if p.strip()]

    out: List[str] = []
    for line in lines[start + 1:]:
        if not line.strip():
            continue
        cur = len(line) - len(line.lstrip())
        m = LIST_ITEM_RE.match(line)
        if m is None or cur <= indent:
            break
        out.append(_unquote(m.group(2)))
    return out


def _charset_from_yaml(text: str) -> List[str]:
    try:
        import yaml
    except Exception:
        return []
    try:
        data = yaml.safe_load(text)
    except Exception:
        return []

    def _dig(node):
        if isinstance(node, dict):
            for key in ("character_dict", "character_list"):
                v = node.get(key)
                if isinstance(v, list) and len(v) > 2:
                    return [str(x) for x in v]
            for v in node.values():
                got = _dig(v)
                if got:
                    return got
        if isinstance(node, list):
            for v in node:
                got = _dig(v)
                if got:
                    return got
        return []

    return _dig(data)


class PaddleOCRRec:
    def __init__(
        self,
        lang_code: str = "eng",
        model_dir: Optional[str] = None,
        device: Optional[str] = None,
        log=None,
        max_width: int = REC_MAX_WIDTH,
    ):
        self._log_fn = log or (lambda m: None)
        self.lang_code = normalize_lang_code(lang_code)
        self.slug = paddle_ocr_slug(self.lang_code)
        self.model_name = paddle_ocr_model_name(self.lang_code)
        self.repo_id = lang_repo_id("ppocr", self.lang_code)
        self.label = f"PP-OCRv5 rec [{self.slug}]"
        self.max_width = int(max_width)

        self.model_dir = Path(model_dir) if model_dir else ppocr_model_dir(self.lang_code)
        self.backend = ""
        self.available = False
        self.characters: List[str] = []
        self.charset_size = 0
        self.use_space_char = True
        self.mean_score = 0.0
        self.last_scores: List[float] = []
        self.script_hint = paddle_ocr_script_hint(self.lang_code)
        self.detector: Optional["PaddleTextDetector"] = None
        self.last_boxes: List[Tuple[int, int, int, int]] = []
        self.failures = 0
        self.last_error = ""

        note = paddle_ocr_note(self.lang_code)
        if note:
            self._log(f"  ℹ [{self.label}] {note}")

        self.device = None
        self.accel_label = ""
        self.dtype = None
        self._model = None
        self._torch = None
        self._rec = None

        if not ppocr_ready(self.lang_code) and not self.model_dir.is_dir():
            self._log(
                f"  ❌ [{self.label}] 모델이 없습니다.\n"
                f"     기대 경로 : {self.model_dir}\n"
                f"     저장소    : {self.repo_id}\n"
                f"     환경설정 → 모델 관리에서 내려받으세요."
            )
            return

        self._load_meta()

        if self._bind_torch():
            pass
        else:
            if not paddle_ready():
                ensure_paddle(log=self._log_fn)
            self._bind_paddle()

        if not self.backend:
            self._log(
                f"  ❌ [{self.label}] 실행 백엔드를 찾지 못했습니다.\n"
                f"     transformers 커스텀 코드도, paddleocr 패키지도 "
                f"사용할 수 없습니다.\n"
                f"     paddlepaddle 은 PyPI 에 없어 전용 인덱스가 필요합니다:"
            )
            for ln in manual_command().splitlines():
                self._log(f"     {ln}")
            self._log(
                "     설치할 수 없다면 OCR 초안 없이 VLM 직접 판독으로 "
                "계속 진행됩니다."
            )
            return

        if self.backend == "paddleocr":
            self.available = True
        else:
            self.available = self.charset_size > 2

        self._log(
            f"  ✅ [{self.label}] 준비 완료 "
            f"(언어 {self.lang_code}/{language_name(self.lang_code)} | "
            f"백엔드 {self.backend} | 문자 {self.charset_size}자 | "
            f"공백문자 {'포함' if self.use_space_char else '없음'})"
        )

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def _load_meta(self):
        yml = self.model_dir / "inference.yml"
        chars: List[str] = []
        if yml.exists():
            text = yml.read_text(encoding="utf-8", errors="ignore")
            chars = _charset_from_text(text)
            if len(chars) < 10:
                chars = _charset_from_yaml(text)
            flag = SPACE_FLAG_RE.search(text)
            if flag:
                self.use_space_char = flag.group(1).strip().lower() in BOOL_TRUE
            shape = SHAPE_RE.search(text)
            if shape:
                try:
                    self.max_width = max(REC_MIN_WIDTH, int(shape.group(3)))
                except Exception:
                    pass
        else:
            self._log(f"  ⚠ [{self.label}] inference.yml 이 없어 문자 사전을 읽지 못했습니다.")

        pre = self.model_dir / "preprocessor_config.json"
        if pre.exists():
            try:
                meta = json.loads(pre.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                meta = {}
            shape = meta.get("image_shape") or meta.get("rec_image_shape")
            if isinstance(shape, (list, tuple)) and len(shape) == 3:
                try:
                    self.max_width = max(REC_MIN_WIDTH, int(shape[2]))
                except Exception:
                    pass

        table = ["blank"] + [str(c) for c in chars]
        if self.use_space_char:
            table.append(" ")
        self.characters = table
        self.charset_size = len(table)

    def _bind_torch(self) -> bool:
        cfg = self.model_dir / "config.json"
        weight = self.model_dir / "model.safetensors"
        if not cfg.exists() or not weight.exists():
            return False

        try:
            meta = json.loads(cfg.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            meta = {}
        if not (meta.get("architectures") or meta.get("auto_map")):
            self._log(
                f"  ⏭ [{self.label}] config.json 에 architectures/auto_map 이 "
                f"없어 transformers 경로를 건너뜁니다."
            )
            return False

        try:
            import torch
            from transformers import AutoModel
        except Exception as e:
            self._log(f"  ⏭ [{self.label}] transformers 경로 불가 ({e})")
            return False

        try:
            self.device, self.accel_label = detect_accelerator(None)
            self.dtype = select_dtype(self.device)
            configure_backends(self.device)
            self._torch = torch
            self._model = AutoModel.from_pretrained(
                str(self.model_dir), trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()
        except Exception as e:
            self._log(f"  ⏭ [{self.label}] transformers 로드 실패 ({e})")
            self._model = None
            self._torch = None
            return False

        self.backend = "transformers"
        return True

    def _bind_paddle(self) -> bool:
        configure_runtime(log=self._log_fn)
        try:
            from paddleocr import TextRecognition
        except Exception as e:
            self._log(f"  ⏭ [{self.label}] paddleocr 경로 불가 ({e})")
            return False

        bases = (
            ("로컬 safetensors", {"model_dir": str(self.model_dir)}),
            ("로컬 + 모델명", {
                "model_name": self.model_name,
                "model_dir": str(self.model_dir),
            }),
            ("공식 모델 자동 수신", {"model_name": self.model_name}),
        )

        errors: List[str] = []
        for label, base in bases:
            for kw in kwargs_variants(base):
                tag = describe_kwargs(kw)
                try:
                    rec = TextRecognition(**kw)
                except TypeError:
                    continue
                except Exception as e:
                    errors.append(
                        f"{label} [{tag}] 생성 실패: "
                        f"{type(e).__name__}: {str(e)[:110]}"
                    )
                    continue

                self._rec = rec
                self.backend = "paddleocr"

                ok, why = self._selftest()
                if ok:
                    self._log(
                        f"  🔗 [{self.label}] paddleocr 바인딩 성공 — "
                        f"{label} | {tag}"
                    )
                    return True

                errors.append(f"{label} [{tag}] 자가검진 실패: {why}")
                self._rec = None
                self.backend = ""

        self._log(f"  ⏭ [{self.label}] paddleocr 초기화 실패")
        for msg in errors[:6]:
            self._log(f"     · {msg}")
        self._rec = None
        return False

    def _selftest(self) -> Tuple[bool, str]:
        probe = Image.new("RGB", (192, 48), (255, 255, 255))
        try:
            from PIL import ImageDraw
            ImageDraw.Draw(probe).rectangle((12, 14, 180, 34), fill=(0, 0, 0))
        except Exception:
            pass
        try:
            self._recognize_paddle([probe])
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:140]}"
        return True, ""

    def get_device(self):
        return self.device

    def attach_detector(self, detector: Optional["PaddleTextDetector"] = None):
        if detector is not None:
            self.detector = detector
            return
        if self.detector is None:
            self.detector = PaddleTextDetector(log=self._log_fn)

    def detect_boxes(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        self.attach_detector()
        det = self.detector
        if det is None or not det.available:
            return []
        if int(getattr(det, "failures", 0)) >= DET_MAX_FAILURES:
            return []
        try:
            boxes = det.detect(image)
        except Exception:
            return []
        self.last_boxes = boxes
        return boxes

    def _split_by_detection(self, image: Image.Image) -> List[Image.Image]:
        boxes = self.detect_boxes(image)
        if not boxes:
            return []

        kept: List[Tuple[int, int, int, int]] = []
        rejected = 0
        for (x0, y0, x1, y1) in boxes:
            bw = int(x1) - int(x0)
            bh = int(y1) - int(y0)
            if bw >= image.width * 0.95 and bh >= image.height * 0.80:
                rejected += 1
                continue
            kept.append((int(x0), int(y0), int(x1), int(y1)))

        if rejected:
            self._log(
                f"    🚫 [DET] 화면 전체를 덮는 박스 {rejected}개를 "
                f"버립니다 (행 분리에 쓸 수 없습니다)."
            )
        if not kept:
            return []

        kept.sort(key=lambda b: (b[1], b[0]))
        out: List[Image.Image] = []
        for (x0, y0, x1, y1) in kept[:LINE_MAX]:
            x0 = max(0, x0 - 2)
            y0 = max(0, y0 - 2)
            x1 = min(image.width, x1 + 2)
            y1 = min(image.height, y1 + 2)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            out.append(image.crop((x0, y0, x1, y1)))
        return out

    def _split_lines(self, image: Image.Image) -> List[Image.Image]:
        detected = self._split_by_detection(image)
        if detected:
            self._log(
                f"    🔍 [DET] PP-OCRv5 det 로 텍스트 박스 {len(detected)}개 "
                f"검출 — 박스 단위로 인식합니다."
            )
            return detected

        shared = split_text_lines(image, boxes=None, log=None)
        if len(shared) > 1:
            self._log(
                f"    📚 [LINE SPLIT] 수평 투영으로 행 {len(shared)}개 분리 "
                f"— 행 단위로 인식합니다."
            )
            return [l.image for l in shared]

        gray = np.asarray(image.convert("L"), dtype=np.float32)
        h, w = gray.shape
        if h < LINE_MIN_HEIGHT * 2 or w < 8:
            return [image]

        ink = 255.0 - gray
        prof = ink.mean(axis=1)
        base = float(np.percentile(prof, 15))
        peak = float(prof.max())
        if peak - base < 4.0:
            return [image]

        gate = base + (peak - base) * 0.22
        hot = prof > gate

        bands: List[Tuple[int, int]] = []
        start = -1
        for y in range(h):
            if hot[y] and start < 0:
                start = y
            elif not hot[y] and start >= 0:
                if y - start >= LINE_MIN_HEIGHT:
                    bands.append((start, y))
                start = -1
        if start >= 0 and h - start >= LINE_MIN_HEIGHT:
            bands.append((start, h))

        if not bands:
            return [image]
        if len(bands) > LINE_MAX:
            bands = bands[:LINE_MAX]

        out: List[Image.Image] = []
        for (y0, y1) in bands:
            ty0 = max(0, y0 - LINE_PAD_Y)
            ty1 = min(h, y1 + LINE_PAD_Y)
            seg = ink[ty0:ty1, :]
            colp = seg.mean(axis=0)
            cgate = float(colp.mean() * 0.35)
            cols = np.nonzero(colp > cgate)[0]
            if cols.size:
                x0 = max(0, int(cols[0]) - 2)
                x1 = min(w, int(cols[-1]) + 3)
            else:
                x0, x1 = 0, w
            if x1 - x0 < 4:
                x0, x1 = 0, w
            out.append(image.crop((x0, ty0, x1, ty1)))
        return out

    def _preprocess(self, crops: Sequence[Image.Image]) -> np.ndarray:
        resized: List[Image.Image] = []
        max_w = REC_MIN_WIDTH
        for im in crops:
            g = im.convert("RGB")
            w, h = g.size
            ratio = float(w) / max(1.0, float(h))
            rw = int(math.ceil(REC_HEIGHT * ratio))
            rw = max(REC_MIN_WIDTH, min(self.max_width, rw))
            resized.append(g.resize((rw, REC_HEIGHT), Image.BILINEAR))
            max_w = max(max_w, rw)

        batch = np.zeros((len(resized), 3, REC_HEIGHT, max_w), dtype=np.float32)
        for i, im in enumerate(resized):
            arr = np.asarray(im, dtype=np.float32) / 255.0
            arr = (arr - REC_MEAN) / REC_STD
            arr = arr.transpose(2, 0, 1)
            batch[i, :, :, : arr.shape[2]] = arr
        return batch

    def _forward_torch(self, batch: np.ndarray) -> Optional[np.ndarray]:
        torch = self._torch
        if torch is None or self._model is None:
            return None

        t = torch.from_numpy(batch).to(self.device)
        try:
            t = t.to(self.dtype)
        except Exception:
            pass

        out = None
        with torch.no_grad():
            for kwargs in ({"pixel_values": t}, {"x": t}, None):
                try:
                    out = self._model(**kwargs) if kwargs else self._model(t)
                    break
                except TypeError:
                    continue
                except Exception:
                    return None
        if out is None:
            return None

        logits = out
        for attr in ("logits", "last_hidden_state", "prediction"):
            v = getattr(out, attr, None)
            if v is not None and torch.is_tensor(v):
                logits = v
                break
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if isinstance(logits, dict):
            for v in logits.values():
                if torch.is_tensor(v):
                    logits = v
                    break
        if not torch.is_tensor(logits):
            return None

        arr = logits.detach().float().cpu().numpy()
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            return None
        n = len(self.characters)
        if arr.shape[1] == n and arr.shape[2] != n:
            arr = arr.transpose(0, 2, 1)
        return arr

    def _ctc_decode(self, logits: np.ndarray) -> List[Tuple[str, float]]:
        if float(logits.min()) < 0.0 or float(logits.max()) > 1.0:
            e = np.exp(logits - logits.max(axis=2, keepdims=True))
            logits = e / np.maximum(e.sum(axis=2, keepdims=True), 1e-8)

        idx = logits.argmax(axis=2)
        prob = logits.max(axis=2)

        out: List[Tuple[str, float]] = []
        for b in range(int(idx.shape[0])):
            chars: List[str] = []
            scores: List[float] = []
            prev = -1
            for t in range(int(idx.shape[1])):
                k = int(idx[b, t])
                if k == prev:
                    continue
                prev = k
                if k <= 0 or k >= len(self.characters):
                    continue
                chars.append(self.characters[k])
                scores.append(float(prob[b, t]))
            text = "".join(chars).strip()
            score = float(np.mean(scores)) if scores else 0.0
            out.append((text, score))
        return out

    @staticmethod
    def _pick_paddle(res) -> Tuple[str, float]:
        node = res
        if isinstance(node, dict) and "res" in node:
            node = node["res"]
        if not isinstance(node, dict):
            j = getattr(node, "json", None)
            if isinstance(j, dict):
                node = j.get("res", j)
        if not isinstance(node, dict):
            return "", 0.0
        return (
            str(node.get("rec_text") or ""),
            float(node.get("rec_score") or 0.0),
        )

    def _recognize_paddle(
        self, crops: Sequence[Image.Image]
    ) -> List[Tuple[str, float]]:
        arrs = []
        for im in crops:
            a = np.asarray(im.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
            arrs.append(np.ascontiguousarray(a))

        out: List[Tuple[str, float]] = []
        try:
            gen = self._rec.predict(
                input=arrs, batch_size=min(REC_BATCH, max(1, len(arrs)))
            )
        except TypeError:
            gen = self._rec.predict(arrs)

        for res in gen:
            out.append(self._pick_paddle(res))
        while len(out) < len(crops):
            out.append(("", 0.0))
        return out

    def recognize_lines(
        self, crops: Sequence[Image.Image]
    ) -> List[Tuple[str, float]]:
        items = [c for c in crops if c is not None and c.width > 2 and c.height > 2]
        if not items or not self.available:
            return []

        if self.backend == "paddleocr":
            if self.failures >= REC_MAX_FAILURES:
                return []
            try:
                return self._recognize_paddle(items)
            except Exception as e:
                self.failures += 1
                msg = f"{type(e).__name__}: {str(e).strip()[:160]}"
                if msg != self.last_error:
                    self.last_error = msg
                    self._log(f"  ⚠ [{self.label}] paddleocr 인식 실패 ({msg})")
                if self.failures >= REC_MAX_FAILURES:
                    self.available = False
                    self._log(
                        f"  🚫 [{self.label}] 연속 {self.failures}회 실패로 "
                        f"인식기를 끕니다. OCR 초안 없이 VLM 직접 판독으로 "
                        f"진행합니다."
                    )
                return []

        out: List[Tuple[str, float]] = []
        for i in range(0, len(items), REC_BATCH):
            chunk = items[i: i + REC_BATCH]
            try:
                logits = self._forward_torch(self._preprocess(chunk))
            except Exception as e:
                self._log(f"  ⚠ [{self.label}] 추론 실패 ({e})")
                logits = None
            if logits is None:
                out.extend(("", 0.0) for _ in chunk)
                continue
            out.extend(self._ctc_decode(logits))
        return out

    def ocr_image(
        self,
        image: Image.Image,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> str:
        if not self.available or image is None:
            return ""
        if image.width < 4 or image.height < 4:
            return ""

        lines = self._split_lines(image)
        pairs = self.recognize_lines(lines)
        if pairs:
            self.last_scores = [s for _t, s in pairs]
            self.mean_score = (
                float(np.mean(self.last_scores)) if self.last_scores else 0.0
            )
        return "\n".join(t for t, _s in pairs if t.strip())

    def ocr_crop(
        self,
        image: Image.Image,
        bbox: Sequence[int],
        max_new_tokens: Optional[int] = None,
        num_beams: int = 1,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> str:
        if image is None:
            return ""
        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(image.width, int(x1))
        y1 = min(image.height, int(y1))
        if x1 <= x0 or y1 <= y0:
            return ""
        return self.ocr_image(image.crop((x0, y0, x1, y1)))

    def probe_confidence(self, image: Image.Image, max_lines: int = 8) -> float:
        if not self.available or image is None:
            return 0.0
        lines = self._split_lines(image)
        if not lines:
            return 0.0
        lines = sorted(
            lines, key=lambda im: im.width * im.height, reverse=True
        )[: int(max_lines)]
        pairs = self.recognize_lines(lines)
        vals = [s for t, s in pairs if t.strip()]
        return float(np.mean(vals)) if vals else 0.0

    def stats(self) -> dict:
        return {
            "lang_code": self.lang_code,
            "slug": self.slug,
            "model_name": self.model_name,
            "repo": self.repo_id,
            "dir": str(self.model_dir),
            "backend": self.backend,
            "available": self.available,
            "charset": self.charset_size,
            "max_width": self.max_width,
            "mean_score": round(self.mean_score, 4),
            "script_hint": self.script_hint,
            "det_backend": getattr(self.detector, "backend", ""),
            "det_available": bool(getattr(self.detector, "available", False)),
            "det_failures": int(getattr(self.detector, "failures", 0)),
            "det_error": str(getattr(self.detector, "last_error", "")),
            "det_boxes": len(self.last_boxes),
            "rec_failures": self.failures,
            "rec_error": self.last_error,
        }

    def unload(self):
        try:
            if self._model is not None:
                self._model.to("cpu")
        except Exception:
            pass
        self._model = None
        self._rec = None
        self.available = False
        try:
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:
            pass


class PaddleTextDetector:
    def __init__(self, log=None):
        self._log_fn = log or (lambda m: None)
        self.model_dir = ppocr_det_dir()
        self.label = "PP-OCRv5 mobile det"
        self.available = False
        self.backend = ""
        self.device = None
        self.dtype = None
        self.failures = 0
        self.last_error = ""
        self._torch = None
        self._model = None
        self._det = None

        if not ppocr_det_ready():
            self._log(
                f"  ⏭ [{self.label}] 검출 모델이 없어 순수 영상처리 검출로 "
                f"진행합니다. (저장소 PaddlePaddle/"
                f"PP-OCRv5_mobile_det_safetensors)"
            )
            return

        if not paddle_ready():
            ensure_paddle(log=self._log_fn)
        configure_runtime(log=self._log_fn)

        if self._bind_paddle() or self._bind_torch():
            self.available = True
            self._log(f"  ✅ [{self.label}] 준비 완료 (백엔드 {self.backend})")
        else:
            self._log(
                f"  ⏭ [{self.label}] 실행 백엔드가 없어 영상처리 검출로 "
                f"대체합니다. 정확도는 떨어지지만 파이프라인은 계속됩니다."
            )

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def _bind_paddle(self) -> bool:
        configure_runtime(log=self._log_fn)
        try:
            from paddleocr import TextDetection
        except Exception as e:
            self._log(f"  ⏭ [{self.label}] paddleocr 경로 불가 ({e})")
            return False

        bases = (
            ("로컬 safetensors", {"model_dir": str(self.model_dir)}),
            ("공식 모델 자동 수신", {"model_name": "PP-OCRv5_mobile_det"}),
        )

        errors: List[str] = []
        for label, base in bases:
            for kw in kwargs_variants(base):
                tag = describe_kwargs(kw)
                try:
                    det = TextDetection(**kw)
                except TypeError:
                    continue
                except Exception as e:
                    errors.append(
                        f"{label} [{tag}] 생성 실패: "
                        f"{type(e).__name__}: {str(e)[:110]}"
                    )
                    continue

                self._det = det
                self.backend = "paddleocr"

                ok, why = self._selftest()
                if ok:
                    self._log(
                        f"  🔗 [{self.label}] paddleocr 바인딩 성공 — "
                        f"{label} | {tag}"
                    )
                    return True

                errors.append(f"{label} [{tag}] 자가검진 실패: {why}")
                self._det = None
                self.backend = ""

        self._log(f"  ⏭ [{self.label}] paddleocr 초기화 실패")
        for msg in errors[:6]:
            self._log(f"     · {msg}")
        if any("onednn" in m.lower() or "pir" in m.lower() for m in errors):
            self._log(
                "     💡 oneDNN 경로가 PP-OCRv5 det 의 double 배열 속성을 "
                "변환하지 못합니다.\n"
                "        set NMS_PADDLE_MKLDNN=0 상태에서도 실패하면 "
                "set FLAGS_use_mkldnn=0 을 셸에서 직접 지정해 보세요."
            )
        self._det = None
        return False

    def _selftest(self) -> Tuple[bool, str]:
        probe = Image.new("RGB", (320, 96), (255, 255, 255))
        try:
            from PIL import ImageDraw
            ImageDraw.Draw(probe).rectangle((24, 34, 296, 62), fill=(0, 0, 0))
        except Exception:
            pass
        try:
            self._run_paddle(probe)
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)[:140]}"
        return True, ""

    def _bind_torch(self) -> bool:
        cfg = self.model_dir / "config.json"
        weight = self.model_dir / "model.safetensors"
        if not cfg.exists() or not weight.exists():
            return False
        try:
            meta = json.loads(cfg.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            meta = {}
        if not (meta.get("architectures") or meta.get("auto_map")):
            return False
        try:
            import torch
            from transformers import AutoModel
        except Exception:
            return False
        try:
            self.device, _lbl = detect_accelerator(None)
            self.dtype = select_dtype(self.device)
            configure_backends(self.device)
            self._torch = torch
            self._model = AutoModel.from_pretrained(
                str(self.model_dir), trust_remote_code=True
            )
            self._model.to(self.device)
            self._model.eval()
        except Exception:
            self._model = None
            self._torch = None
            return False
        self.backend = "transformers"
        return True

    @staticmethod
    def _resize(image: Image.Image) -> Tuple[np.ndarray, float, float]:
        w, h = image.size
        scale = min(1.0, float(DET_LIMIT_SIDE) / float(max(1, max(w, h))))
        nw = max(32, int(round(w * scale / 32.0)) * 32)
        nh = max(32, int(round(h * scale / 32.0)) * 32)
        arr = np.asarray(
            image.convert("RGB").resize((nw, nh), Image.BILINEAR),
            dtype=np.float32,
        ) / 255.0
        for c in range(3):
            arr[:, :, c] = (arr[:, :, c] - DET_MEAN[c]) / DET_STD[c]
        return arr.transpose(2, 0, 1)[None, ...], w / float(nw), h / float(nh)

    def _probmap_torch(self, image: Image.Image):
        torch = self._torch
        if torch is None or self._model is None:
            return None, 1.0, 1.0
        batch, sx, sy = self._resize(image)
        t = torch.from_numpy(batch).to(self.device)
        try:
            t = t.to(self.dtype)
        except Exception:
            pass
        out = None
        with torch.no_grad():
            for kwargs in ({"pixel_values": t}, {"x": t}, None):
                try:
                    out = self._model(**kwargs) if kwargs else self._model(t)
                    break
                except TypeError:
                    continue
                except Exception:
                    return None, sx, sy
        if out is None:
            return None, sx, sy
        for attr in ("logits", "last_hidden_state", "maps"):
            v = getattr(out, attr, None)
            if v is not None and torch.is_tensor(v):
                out = v
                break
        if isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            for v in out.values():
                if torch.is_tensor(v):
                    out = v
                    break
        if not torch.is_tensor(out):
            return None, sx, sy
        arr = out.detach().float().cpu().numpy()
        while arr.ndim > 2:
            arr = arr[0]
        return arr, sx, sy

    @staticmethod
    def _boxes_from_map(
        prob: np.ndarray, sx: float, sy: float
    ) -> List[Tuple[int, int, int, int]]:
        mask = prob > DET_THRESH
        h, w = mask.shape
        if not mask.any():
            return []

        visited = np.zeros_like(mask, dtype=bool)
        out: List[Tuple[int, int, int, int]] = []

        for y0 in range(h):
            for x0 in range(w):
                if not mask[y0, x0] or visited[y0, x0]:
                    continue
                stack = [(y0, x0)]
                visited[y0, x0] = True
                ys, xs = [y0], [x0]
                while stack:
                    cy, cx = stack.pop()
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        ny, nx = cy + dy, cx + dx
                        if ny < 0 or nx < 0 or ny >= h or nx >= w:
                            continue
                        if visited[ny, nx] or not mask[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
                        ys.append(ny)
                        xs.append(nx)
                if len(ys) < 4:
                    continue
                ry0, ry1 = min(ys), max(ys) + 1
                rx0, rx1 = min(xs), max(xs) + 1
                pad_y = max(1, int((ry1 - ry0) * (DET_UNCLIP - 1.0) * 0.5))
                pad_x = max(1, int((rx1 - rx0) * (DET_UNCLIP - 1.0) * 0.5))
                bx0 = int(max(0, rx0 - pad_x) * sx)
                by0 = int(max(0, ry0 - pad_y) * sy)
                bx1 = int(min(w, rx1 + pad_x) * sx)
                by1 = int(min(h, ry1 + pad_y) * sy)
                if (bx1 - bx0) * (by1 - by0) < DET_MIN_AREA:
                    continue
                out.append((bx0, by0, bx1, by1))
        return out

    def _run_paddle(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)[:, :, ::-1]
        gen = self._det.predict(input=[np.ascontiguousarray(arr)])

        out: List[Tuple[int, int, int, int]] = []
        for res in gen:
            node = res
            if isinstance(node, dict) and "res" in node:
                node = node["res"]
            if not isinstance(node, dict):
                node = getattr(node, "json", {}) or {}
                node = node.get("res", node)
            polys = node.get("dt_polys")
            if polys is None:
                continue
            for p in polys:
                pts = np.asarray(p, dtype=np.float32).reshape(-1, 2)
                if pts.size == 0:
                    continue
                out.append((
                    int(pts[:, 0].min()), int(pts[:, 1].min()),
                    int(pts[:, 0].max()) + 1, int(pts[:, 1].max()) + 1,
                ))
        return out

    def _trip(self, err: Exception) -> None:
        self.failures += 1
        msg = f"{type(err).__name__}: {str(err).strip()[:160]}"

        if msg != self.last_error:
            self.last_error = msg
            self._log(f"  ⚠ [{self.label}] 검출 실패 ({msg})")

        if self.failures >= DET_MAX_FAILURES:
            self.available = False
            self._log(
                f"  🚫 [{self.label}] 연속 {self.failures}회 실패로 검출기를 "
                f"끕니다. 이후에는 같은 오류를 반복 출력하지 않고 영상처리 "
                f"검출로만 진행합니다."
            )

    def detect(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        if image is None or not self.available:
            return []
        if self.failures >= DET_MAX_FAILURES:
            return []

        if self.backend == "paddleocr":
            try:
                return self._run_paddle(image)
            except Exception as e:
                self._trip(e)
                return []

        try:
            prob, sx, sy = self._probmap_torch(image)
        except Exception as e:
            self._trip(e)
            return []
        if prob is None:
            return []
        return self._boxes_from_map(prob, sx, sy)

    def unload(self):
        try:
            if self._model is not None:
                self._model.to("cpu")
        except Exception:
            pass
        self._model = None
        self._det = None
        self.available = False