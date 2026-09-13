import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import numpy as np

from .model_manager import (
    BASE_DIR,
    VISION_ENC_PATH,
    MissingModelError,
    ensure_model_dir,
)
from .device import detect_accelerator, select_dtype, configure_backends


AX_VE_CODE_FILES = (
    "configuration_ax_ve.py",
    "modeling_ax_ve.py",
    "image_processing_ax_ve.py",
)

AX_VE_OPTIONAL_CODE_FILES = (
    "processing_ax_ve.py",
)


def _resolve_ax_ve_code_dir() -> Path:
    candidates = [BASE_DIR / "ax-ve", VISION_ENC_PATH]
    for cand in candidates:
        cand = Path(cand)
        if not cand.is_dir():
            continue
        if all((cand / f).exists() for f in AX_VE_CODE_FILES):
            return cand.resolve()
    tried = "\n".join(f"    - {c}" for c in candidates)
    raise MissingModelError(
        "A.X-VE 모델 정의 코드를 찾지 못했습니다.\n"
        f"  필요 파일 : {', '.join(AX_VE_CODE_FILES)}\n"
        f"  탐색 경로 :\n{tried}"
    )


def _load_ax_ve():
    ax_ve_dir = _resolve_ax_ve_code_dir()

    if "ax_ve" not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec("ax_ve", None, is_package=True)
        )
        pkg.__path__ = [str(ax_ve_dir)]
        pkg.__package__ = "ax_ve"
        sys.modules["ax_ve"] = pkg

    def _load(name: str, required: bool = True):
        full = f"ax_ve.{name}"
        if full in sys.modules:
            return sys.modules[full]
        target = ax_ve_dir / f"{name}.py"
        if not target.exists():
            if required:
                raise MissingModelError(f"A.X-VE 코드 파일 누락: {target}")
            return None
        spec = importlib.util.spec_from_file_location(
            full,
            str(target),
            submodule_search_locations=[str(ax_ve_dir)],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "ax_ve"
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("configuration_ax_ve")
    _load("modeling_ax_ve")
    _load("image_processing_ax_ve")
    for opt in AX_VE_OPTIONAL_CODE_FILES:
        try:
            _load(opt[:-3], required=False)
        except Exception:
            pass

    cfg_mod = sys.modules["ax_ve.configuration_ax_ve"]
    model_mod = sys.modules["ax_ve.modeling_ax_ve"]
    proc_mod = sys.modules["ax_ve.image_processing_ax_ve"]

    return cfg_mod, model_mod, proc_mod


AXVEConfig, AXVEVisionModel, AXVEImageProcessor = None, None, None


def _ensure_ax_ve():
    global AXVEConfig, AXVEVisionModel, AXVEImageProcessor
    if AXVEConfig is None:
        cfg_mod, model_mod, proc_mod = _load_ax_ve()
        AXVEConfig = cfg_mod.AXVEConfig
        AXVEVisionModel = model_mod.AXVEVisionModel
        AXVEImageProcessor = proc_mod.AXVEImageProcessor


def _log_tensor_summary(model, name: str):
    total_params = 0
    tensor_count = 0
    for pname, param in model.named_parameters():
        total_params += param.numel()
        tensor_count += 1
    print(f"[TENSOR/{name}] tensors={tensor_count} | params={total_params:,} ({total_params/1e6:.1f}M)")
    for pname, param in list(model.named_parameters())[:5]:
        print(f"  ├─ {pname}: {list(param.shape)} {param.dtype}")
    if tensor_count > 5:
        print(f"  └─ ... ({tensor_count - 5} more)")


def load_pretrained_with_dtype(loader, path, dtype, **kwargs):
    try:
        return loader.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return loader.from_pretrained(path, torch_dtype=dtype, **kwargs)


def l2_normalize_rows(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return mat / norms


class AXVEEmbedder:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        _ensure_ax_ve()

        if model_path is None:
            model_path = str(ensure_model_dir("ax-ve"))
        else:
            model_path = str(model_path)
        self.model_path = model_path

        self.device, self.accel_label = detect_accelerator(device)
        self.dtype = select_dtype(self.device)
        configure_backends(self.device)

        self.config = AXVEConfig.from_pretrained(model_path)
        self.vision_config = self.config.vision_config
        self.model = load_pretrained_with_dtype(
            AXVEVisionModel,
            model_path,
            self.dtype,
            config=self.vision_config,
        )
        self.model.to(self.device)
        self.model.eval()
        self.image_processor = AXVEImageProcessor.from_pretrained(model_path)
        self._loaded = True
        print(f"[GPU/AX-VE] Model loaded | device={self.device} | dtype={self.dtype}")
        print(f"[GPU/AX-VE] weights={self.model_path}")
        _log_tensor_summary(self.model, "AX-VE")

    def get_device(self) -> torch.device:
        return self.device

    def get_hidden_size(self) -> int:
        return int(self.vision_config.hidden_size)

    def get_patch_size(self) -> int:
        return int(self.vision_config.patch_size)

    def get_merge_size(self) -> int:
        return int(self.vision_config.spatial_merge_size)

    def preprocess(self, image):
        return self.image_processor.preprocess(
            images=image,
            return_tensors="pt",
        )

    @torch.no_grad()
    def embed_image(self, image, l2_normalize: bool = True) -> "PatchGrid":
        inputs = self.preprocess(image)
        pixel_values = inputs["pixel_values_images"].to(self.device)
        grid_hw = inputs["image_grid_hw"].to(self.device)

        hidden_states = self.model(
            hidden_states=pixel_values,
            grid_hw=grid_hw,
        )

        patch_embeddings = hidden_states.float().cpu().numpy()
        if patch_embeddings.ndim == 3:
            patch_embeddings = patch_embeddings.reshape(-1, patch_embeddings.shape[-1])
        if l2_normalize:
            patch_embeddings = l2_normalize_rows(patch_embeddings)

        gh, gw = [int(v) for v in grid_hw[0].tolist()]
        orig_w, orig_h = image.size
        resized_h, resized_w = self.image_processor.get_size_of_processed_image(
            np.array(image)
        )
        scale_x = orig_w / float(max(1, resized_w))
        scale_y = orig_h / float(max(1, resized_h))

        expected = gh * gw
        if patch_embeddings.shape[0] != expected:
            n = min(patch_embeddings.shape[0], expected)
            fixed = np.zeros((expected, patch_embeddings.shape[1]), dtype=np.float32)
            fixed[:n] = patch_embeddings[:n]
            patch_embeddings = fixed

        return PatchGrid(
            embeddings=patch_embeddings,
            grid_rows=gh,
            grid_cols=gw,
            scale_x=scale_x,
            scale_y=scale_y,
            orig_width=orig_w,
            orig_height=orig_h,
            patch_size=self.get_patch_size(),
            merge_size=self.get_merge_size(),
            resized_width=resized_w,
            resized_height=resized_h,
        )

    def unload(self):
        self.model = None
        self.image_processor = None
        self._loaded = False
        try:
            from .memory import reclaim
            reclaim()
        except Exception:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class PatchGrid:
    def __init__(
        self,
        embeddings: np.ndarray,
        grid_rows: int,
        grid_cols: int,
        scale_x: float,
        scale_y: float,
        orig_width: int,
        orig_height: int,
        patch_size: int,
        merge_size: int,
        resized_width: int = 0,
        resized_height: int = 0,
    ):
        self.embeddings = embeddings
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.scale_x = float(scale_x)
        self.scale_y = float(scale_y)
        self.orig_width = int(orig_width)
        self.orig_height = int(orig_height)
        self.patch_size = int(patch_size)
        self.merge_size = int(merge_size)
        self.resized_width = int(resized_width)
        self.resized_height = int(resized_height)
        self.num_patches = self.grid_rows * self.grid_cols

    @property
    def dim(self) -> int:
        if self.embeddings is None or self.embeddings.size == 0:
            return 0
        return int(self.embeddings.shape[-1])

    def index_of(self, row: int, col: int) -> int:
        return int(row) * self.grid_cols + int(col)

    def row_col(self, index: int) -> Tuple[int, int]:
        cols = max(1, self.grid_cols)
        return int(index) // cols, int(index) % cols

    def vector(self, index: int) -> np.ndarray:
        return self.embeddings[int(index)]

    def get_patch_bbox(self, row: int, col: int) -> Tuple[int, int, int, int]:
        x0 = int(col * self.patch_size * self.scale_x)
        y0 = int(row * self.patch_size * self.scale_y)
        x1 = int((col + 1) * self.patch_size * self.scale_x)
        y1 = int((row + 1) * self.patch_size * self.scale_y)
        x0 = max(0, min(x0, self.orig_width - 1))
        y0 = max(0, min(y0, self.orig_height - 1))
        x1 = min(max(x1, x0 + 1), self.orig_width)
        y1 = min(max(y1, y0 + 1), self.orig_height)
        return (x0, y0, x1, y1)

    def get_index_bbox(self, index: int) -> Tuple[int, int, int, int]:
        r, c = self.row_col(index)
        return self.get_patch_bbox(r, c)

    def get_region_bbox(
        self,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
    ) -> Tuple[int, int, int, int]:
        r0 = max(0, min(int(r0), self.grid_rows - 1))
        r1 = max(0, min(int(r1), self.grid_rows - 1))
        c0 = max(0, min(int(c0), self.grid_cols - 1))
        c1 = max(0, min(int(c1), self.grid_cols - 1))
        if r1 < r0:
            r0, r1 = r1, r0
        if c1 < c0:
            c0, c1 = c1, c0
        tx0, ty0, _, _ = self.get_patch_bbox(r0, c0)
        _, _, bx1, by1 = self.get_patch_bbox(r1, c1)
        return (tx0, ty0, max(bx1, tx0 + 1), max(by1, ty0 + 1))

    def get_patch_center(self, row: int, col: int):
        x0, y0, x1, y1 = self.get_patch_bbox(row, col)
        return ((x0 + x1) / 2, (y0 + y1) / 2)

    def to_dict(self) -> dict:
        return {
            "rows": self.grid_rows,
            "cols": self.grid_cols,
            "num_patches": self.num_patches,
            "dim": self.dim,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "patch_size": self.patch_size,
            "merge_size": self.merge_size,
            "orig_width": self.orig_width,
            "orig_height": self.orig_height,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
        }