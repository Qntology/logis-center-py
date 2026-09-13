import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image

DEFAULT_DPI = 200
MIN_DPI = 72
MAX_DPI = 400
MAX_SIDE_PX = 4000
MAX_PAGES = 32

TEXT_LAYER_MIN_CHARS = 40
TEXT_LAYER_MIN_ALNUM = 20

ENV_DPI = "NMS_PDF_DPI"
ENV_MAX_PAGES = "NMS_PDF_MAX_PAGES"
ENV_BACKEND = "NMS_PDF_BACKEND"
ENV_FORCE_VISION = "NMS_PDF_FORCE_VISION"

BACKEND_PDFIUM = "pypdfium2"
BACKEND_PYPDF = "pypdf"
BACKEND_POPPLER = "pdf2image"


class PdfPage:
    def __init__(
        self,
        index: int,
        image: Optional[Image.Image] = None,
        text: str = "",
        width_pt: float = 0.0,
        height_pt: float = 0.0,
        dpi: int = DEFAULT_DPI,
    ):
        self.index = int(index)
        self.image = image
        self.text = str(text or "")
        self.width_pt = float(width_pt)
        self.height_pt = float(height_pt)
        self.dpi = int(dpi)

    @property
    def has_image(self) -> bool:
        return self.image is not None

    @property
    def char_count(self) -> int:
        return len(self.text.strip())

    @property
    def alnum_count(self) -> int:
        return sum(1 for ch in self.text if ch.isalnum())

    @property
    def has_text_layer(self) -> bool:
        return (
            self.char_count >= TEXT_LAYER_MIN_CHARS
            and self.alnum_count >= TEXT_LAYER_MIN_ALNUM
        )

    def size(self) -> Tuple[int, int]:
        if self.image is None:
            return (0, 0)
        return (self.image.width, self.image.height)

    def to_dict(self) -> dict:
        w, h = self.size()
        return {
            "index": self.index,
            "width": w,
            "height": h,
            "dpi": self.dpi,
            "chars": self.char_count,
            "text_layer": self.has_text_layer,
        }

    def __repr__(self) -> str:
        w, h = self.size()
        return f"<PdfPage#{self.index} {w}x{h} chars={self.char_count}>"


class PdfDocument:
    def __init__(self, path: str = "", backend: str = ""):
        self.path = str(path)
        self.backend = str(backend)
        self.pages: List[PdfPage] = []
        self.total_pages = 0
        self.truncated = False
        self.error = ""
        self.logs: List[str] = []

    @property
    def ok(self) -> bool:
        return bool(self.pages) and not self.error

    @property
    def text_pages(self) -> List[PdfPage]:
        return [p for p in self.pages if p.has_text_layer]

    @property
    def has_text_layer(self) -> bool:
        if not self.pages:
            return False
        return len(self.text_pages) * 2 >= len(self.pages)

    def merged_text(self, separator: str = "\n\n") -> str:
        parts = [p.text.strip() for p in self.pages if p.text.strip()]
        return separator.join(parts)

    def page(self, index: int) -> Optional[PdfPage]:
        for p in self.pages:
            if p.index == int(index):
                return p
        return None

    def first_image(self) -> Optional[Image.Image]:
        for p in self.pages:
            if p.image is not None:
                return p.image
        return None

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "backend": self.backend,
            "total_pages": self.total_pages,
            "rendered": len(self.pages),
            "truncated": self.truncated,
            "text_layer": self.has_text_layer,
            "chars": len(self.merged_text()),
            "error": self.error,
            "pages": [p.to_dict() for p in self.pages],
        }


def _log_to(log: Optional[Callable[[str], None]], msg: str):
    if log is None:
        return
    try:
        log(msg)
    except Exception:
        pass


def has_module(name: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def available_backends() -> List[str]:
    out: List[str] = []
    if has_module("pypdfium2"):
        out.append(BACKEND_PDFIUM)
    if has_module("pypdf"):
        out.append(BACKEND_PYPDF)
    if has_module("pdf2image"):
        out.append(BACKEND_POPPLER)
    return out


def preferred_backend() -> str:
    forced = str(os.environ.get(ENV_BACKEND, "") or "").strip()
    if forced and has_module(forced.replace("-", "_")):
        return forced

    for name in (BACKEND_PDFIUM, BACKEND_PYPDF, BACKEND_POPPLER):
        mod = name.replace("-", "_")
        if has_module(mod):
            return name
    return ""


def target_dpi() -> int:
    try:
        raw = int(os.environ.get(ENV_DPI, DEFAULT_DPI) or DEFAULT_DPI)
    except Exception:
        raw = DEFAULT_DPI
    return max(MIN_DPI, min(MAX_DPI, raw))


def page_budget() -> int:
    try:
        raw = int(os.environ.get(ENV_MAX_PAGES, MAX_PAGES) or MAX_PAGES)
    except Exception:
        raw = MAX_PAGES
    return max(1, min(512, raw))


def force_vision() -> bool:
    raw = str(os.environ.get(ENV_FORCE_VISION, "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _fit_dpi(width_pt: float, height_pt: float, dpi: int) -> int:
    longest = max(float(width_pt or 0.0), float(height_pt or 0.0))
    if longest <= 0.0:
        return int(dpi)
    px = longest * float(dpi) / 72.0
    if px <= MAX_SIDE_PX:
        return int(dpi)
    return max(MIN_DPI, int(72.0 * MAX_SIDE_PX / longest))


def _page_size(page) -> Tuple[float, float]:
    for name in ("get_size", "get_mediabox"):
        fn = getattr(page, name, None)
        if fn is None:
            continue
        try:
            box = fn()
        except Exception:
            continue
        if not box:
            continue
        vals = list(box)
        if len(vals) == 2:
            return float(vals[0]), float(vals[1])
        if len(vals) == 4:
            return abs(float(vals[2]) - float(vals[0])), abs(
                float(vals[3]) - float(vals[1])
            )
    return 0.0, 0.0


def _page_text(page) -> str:
    tp = None
    try:
        tp = page.get_textpage()
    except Exception:
        return ""

    try:
        for name in ("get_text_bounded", "get_text_range"):
            fn = getattr(tp, name, None)
            if fn is None:
                continue
            try:
                return str(fn() or "")
            except Exception:
                continue
        return ""
    finally:
        try:
            tp.close()
        except Exception:
            pass


def _bitmap_to_pil(bitmap) -> Optional[Image.Image]:
    fn = getattr(bitmap, "to_pil", None)
    if fn is not None:
        try:
            return fn().convert("RGB")
        except Exception:
            pass

    fn = getattr(bitmap, "to_numpy", None)
    if fn is not None:
        try:
            import numpy as np
            arr = np.asarray(fn())
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return Image.fromarray(arr[:, :, :3].astype("uint8"), "RGB")
            if arr.ndim == 2:
                return Image.fromarray(arr.astype("uint8"), "L").convert("RGB")
        except Exception:
            pass
    return None


def _render_pdfium(
    path: Path,
    dpi: int,
    max_pages: int,
    want_text: bool,
    log: Optional[Callable[[str], None]],
) -> PdfDocument:
    import pypdfium2 as pdfium

    doc = PdfDocument(str(path), BACKEND_PDFIUM)
    pdf = None

    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as e:
        doc.error = f"{type(e).__name__}: {e}"
        return doc

    try:
        try:
            doc.total_pages = int(len(pdf))
        except Exception:
            doc.total_pages = 0

        if doc.total_pages <= 0:
            doc.error = "페이지를 찾지 못했습니다."
            return doc

        count = min(doc.total_pages, int(max_pages))
        doc.truncated = count < doc.total_pages

        for i in range(count):
            page = None
            try:
                page = pdf[i]
            except Exception as e:
                _log_to(log, f"    ⚠ {i + 1}쪽 열기 실패 ({e})")
                continue

            try:
                w_pt, h_pt = _page_size(page)
                use_dpi = _fit_dpi(w_pt, h_pt, dpi)
                scale = float(use_dpi) / 72.0

                img: Optional[Image.Image] = None
                try:
                    bitmap = page.render(scale=scale)
                    img = _bitmap_to_pil(bitmap)
                    close = getattr(bitmap, "close", None)
                    if close is not None:
                        try:
                            close()
                        except Exception:
                            pass
                except Exception as e:
                    _log_to(log, f"    ⚠ {i + 1}쪽 렌더 실패 ({e})")

                text = _page_text(page) if want_text else ""

                doc.pages.append(
                    PdfPage(
                        index=i,
                        image=img,
                        text=text,
                        width_pt=w_pt,
                        height_pt=h_pt,
                        dpi=use_dpi,
                    )
                )
            finally:
                close = getattr(page, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:
                        pass
    finally:
        close = getattr(pdf, "close", None)
        if close is not None:
            try:
                close()
            except Exception:
                pass

    if not doc.pages and not doc.error:
        doc.error = "렌더된 페이지가 없습니다."
    return doc


def _render_pypdf(
    path: Path,
    max_pages: int,
    log: Optional[Callable[[str], None]],
) -> PdfDocument:
    from pypdf import PdfReader

    doc = PdfDocument(str(path), BACKEND_PYPDF)

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        doc.error = f"{type(e).__name__}: {e}"
        return doc

    try:
        doc.total_pages = int(len(reader.pages))
    except Exception:
        doc.total_pages = 0

    if doc.total_pages <= 0:
        doc.error = "페이지를 찾지 못했습니다."
        return doc

    count = min(doc.total_pages, int(max_pages))
    doc.truncated = count < doc.total_pages

    for i in range(count):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception as e:
            _log_to(log, f"    ⚠ {i + 1}쪽 텍스트 추출 실패 ({e})")
            text = ""
        doc.pages.append(PdfPage(index=i, image=None, text=text))

    return doc


def _render_poppler(
    path: Path,
    dpi: int,
    max_pages: int,
    log: Optional[Callable[[str], None]],
) -> PdfDocument:
    from pdf2image import convert_from_path

    doc = PdfDocument(str(path), BACKEND_POPPLER)

    _log_to(
        log,
        "  ⚠ [PDF] Poppler 경로를 사용합니다. Poppler 는 GPL 이므로 별도 "
        "프로세스로만 호출해야 하며, 배포 시 라이선스를 확인하세요.",
    )

    try:
        images = convert_from_path(
            str(path), first_page=1, last_page=int(max_pages), dpi=int(dpi)
        )
    except Exception as e:
        doc.error = f"{type(e).__name__}: {e}"
        return doc

    doc.total_pages = len(images)
    for i, img in enumerate(images):
        doc.pages.append(
            PdfPage(index=i, image=img.convert("RGB"), text="", dpi=int(dpi))
        )
    return doc


def render_pdf(
    path,
    dpi: Optional[int] = None,
    max_pages: Optional[int] = None,
    want_text: bool = True,
    log: Optional[Callable[[str], None]] = None,
) -> PdfDocument:
    p = Path(path)
    use_dpi = int(dpi) if dpi else target_dpi()
    budget = int(max_pages) if max_pages else page_budget()

    backend = preferred_backend()
    if not backend:
        doc = PdfDocument(str(p), "")
        doc.error = (
            "PDF 백엔드가 없습니다.\n"
            "  pip install pypdfium2\n"
            "  (PDFium 은 BSD-3-Clause 이며 외부 바이너리가 필요 없습니다)"
        )
        return doc

    _log_to(log, f"  📄 [PDF] 백엔드 '{backend}' | 목표 {use_dpi} DPI")

    if backend == BACKEND_PDFIUM:
        doc = _render_pdfium(p, use_dpi, budget, want_text, log)
        if doc.ok:
            _describe(doc, log)
            return doc
        _log_to(log, f"  ⚠ [PDF] pypdfium2 실패 ({doc.error}) → 폴백 시도")

    if has_module("pypdf"):
        doc = _render_pypdf(p, budget, log)
        if doc.ok:
            _log_to(
                log,
                "  ⚠ [PDF] pypdf 는 텍스트만 추출합니다. 이미지 렌더가 "
                "필요하면 pip install pypdfium2 를 실행하세요.",
            )
            _describe(doc, log)
            return doc

    if has_module("pdf2image"):
        doc = _render_poppler(p, use_dpi, budget, log)
        if doc.ok:
            _describe(doc, log)
            return doc

    out = PdfDocument(str(p), backend)
    out.error = "모든 PDF 백엔드가 실패했습니다."
    return out


def _describe(doc: PdfDocument, log: Optional[Callable[[str], None]]):
    if log is None:
        return

    imgs = sum(1 for p in doc.pages if p.has_image)
    chars = len(doc.merged_text())

    tail = f" (전체 {doc.total_pages}쪽 중 상한 적용)" if doc.truncated else ""
    _log_to(
        log,
        f"  📄 [PDF] {len(doc.pages)}쪽 처리{tail} | 렌더 {imgs}쪽 "
        f"| 텍스트 {chars}자",
    )

    if doc.has_text_layer:
        _log_to(
            log,
            f"  ✅ [PDF TEXT LAYER] {len(doc.text_pages)}/{len(doc.pages)}쪽에 "
            f"내장 텍스트가 있습니다. OCR 대신 원문을 그대로 씁니다 "
            f"— 판독 오류가 원천적으로 발생하지 않습니다.",
        )
    else:
        _log_to(
            log,
            "  🖼 [PDF SCAN] 내장 텍스트가 없어 스캔 문서로 보고 "
            "비전 파이프라인으로 진행합니다.",
        )

    for pg in doc.pages[:4]:
        w, h = pg.size()
        mark = "T" if pg.has_text_layer else "-"
        _log_to(
            log,
            f"     [{mark}] {pg.index + 1}쪽 {w}x{h} @{pg.dpi}dpi "
            f"| {pg.char_count}자",
        )


def report_lines() -> List[str]:
    backends = available_backends()
    if not backends:
        return [
            "  [MISS] PDF 백엔드 없음 — pip install pypdfium2 "
            "(BSD-3-Clause, 외부 바이너리 불필요)"
        ]

    out = [
        f"  [OK ] PDF 백엔드 {', '.join(backends)} "
        f"| 활성 '{preferred_backend()}' | {target_dpi()} DPI "
        f"| 최대 {page_budget()}쪽"
    ]
    if BACKEND_PDFIUM not in backends:
        out.append(
            "         pypdfium2 가 없어 렌더 품질이나 라이선스가 "
            "제한될 수 있습니다."
        )
    if BACKEND_POPPLER in backends and BACKEND_PDFIUM not in backends:
        out.append(
            "         ⚠ Poppler(GPL) 경로만 사용 가능합니다. 배포 시 "
            "라이선스를 확인하세요."
        )
    return out