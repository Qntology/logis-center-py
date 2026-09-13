import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from . import diagnostics
from .device import configure_backends, detect_accelerator, select_dtype

SIGLIP_TEXT_MAX_LEN = 64
SIGLIP_IMAGE_MEAN = 0.5
SIGLIP_IMAGE_STD = 0.5
TEXT_BATCH = 64
DEFAULT_MAX_PATCHES = 256

TENSOR_ATTRS = (
    "pooler_output",
    "last_hidden_state",
    "text_embeds",
    "image_embeds",
)


def _as_tensor(out, prefer: Sequence[str] = ()):
    if torch.is_tensor(out):
        return out
    for attr in tuple(prefer) + TENSOR_ATTRS:
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
    if isinstance(out, (tuple, list)):
        for v in out:
            if torch.is_tensor(v):
                return v
    if isinstance(out, dict):
        for v in out.values():
            if torch.is_tensor(v):
                return v
    return None


class Siglip2Joint:
    def __init__(
        self,
        model_path,
        device: Optional[str] = None,
        label: str = "siglip2",
        log=None,
        max_patches: int = DEFAULT_MAX_PATCHES,
        fp8_weights: bool = False,
    ):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.model_path = str(model_path)
        self.label = label
        self._log_fn = log or (lambda m: None)
        self.max_patches = int(max_patches or 0)

        self.device, self.accel_label = detect_accelerator(device)
        self.dtype = select_dtype(self.device)
        configure_backends(self.device)

        self.config = AutoConfig.from_pretrained(self.model_path)
        try:
            self.model = AutoModel.from_pretrained(self.model_path, dtype=self.dtype)
        except TypeError:
            self.model = AutoModel.from_pretrained(
                self.model_path, torch_dtype=self.dtype
            )
        self.model.to(self.device)
        self.model.eval()

        self.vision_model = getattr(self.model, "vision_model", None)
        self.text_model = getattr(self.model, "text_model", None)
        if self.vision_model is None or self.text_model is None:
            raise RuntimeError(
                f"{self.model_path} 는 SigLIP 조인트 모델이 아닙니다 "
                f"(vision_model / text_model 이 없습니다)."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if getattr(self.tokenizer, "pad_token", None) is None:
            for cand in ("eos_token", "unk_token", "sep_token"):
                tok = getattr(self.tokenizer, cand, None)
                if tok:
                    self.tokenizer.pad_token = tok
                    break

        vcfg = getattr(self.config, "vision_config", None)
        tcfg = getattr(self.config, "text_config", None)

        self.patch_size = int(getattr(vcfg, "patch_size", 16) or 16)
        self.image_size = int(getattr(vcfg, "image_size", 512) or 512)
        self.rows = max(1, self.image_size // self.patch_size)
        self.cols = max(1, self.image_size // self.patch_size)

        self.dim = int(
            getattr(tcfg, "projection_size", 0)
            or getattr(tcfg, "hidden_size", 0)
            or 0
        )
        self.max_text_len = int(
            getattr(tcfg, "max_position_embeddings", SIGLIP_TEXT_MAX_LEN)
            or SIGLIP_TEXT_MAX_LEN
        )

        self._param_dtype = next(self.model.parameters()).dtype

        if fp8_weights and self.device.type == "cuda":
            from . import fp8 as _fp8
            _fp8.convert_linear_to_fp8(
                self.model,
                skip_names=("head.", "embeddings", "logit"),
                log=self._log,
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        self._log(
            f"  ✅ [{self.label}] SigLIP2 조인트 로드 "
            f"({self.model_path} | dim={self.dim} | 격자 {self.rows}x{self.cols} "
            f"| 텍스트 {self.max_text_len}토큰 | {self.device}/{self.dtype})"
        )

        try:
            import transformers as _tf
            tf_ver = getattr(_tf, "__version__", "?")
        except Exception:
            tf_ver = "?"
        self._log(
            f"  🔧 [{self.label}] transformers {tf_ver} "
            f"| 클래스 {type(self.model).__name__} "
            f"| 비전 {type(self.vision_model).__name__} "
            f"| 텍스트 {type(self.text_model).__name__} "
            f"| 헤드 {type(getattr(self.vision_model, 'head', None)).__name__}"
        )
        diagnostics.describe_config(self.config, self.label, self._log)
        diagnostics.describe_module(self.model, self.label, self._log)

        self.selftest(self._log)

    def _log(self, msg: str):
        try:
            self._log_fn(msg)
        except Exception:
            pass

    def get_embed_dim(self) -> int:
        return int(self.dim)

    def _tokenize(self, batch: Sequence[str]) -> dict:
        enc = self.tokenizer(
            list(batch),
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )
        return {
            k: v.to(self.device)
            for k, v in enc.items()
            if k in ("input_ids", "attention_mask")
        }

    @torch.no_grad()
    def _text_pooled(self, feed: dict) -> torch.Tensor:
        out = self.text_model(**feed)
        pooled = _as_tensor(out, ("pooler_output", "text_embeds"))
        if pooled is not None and pooled.dim() == 2:
            return pooled

        last = _as_tensor(out, ("last_hidden_state",))
        if last is None or last.dim() != 3:
            raise RuntimeError(
                f"SigLIP 텍스트 타워 출력에서 텐서를 찾지 못했습니다 "
                f"(type={type(out).__name__})"
            )
        pooled = last[:, -1, :]
        head = getattr(self.text_model, "head", None)
        if head is not None:
            pooled = head(pooled)
        return pooled

    @torch.no_grad()
    def encode_text(self, texts: Sequence[str], l2: bool = True) -> np.ndarray:
        items = [str(t) if t is not None else "" for t in texts]
        if not items:
            return np.zeros((0, max(1, self.dim)), dtype=np.float32)

        chunks: List[np.ndarray] = []
        for i in range(0, len(items), TEXT_BATCH):
            feed = self._tokenize(items[i: i + TEXT_BATCH])
            pooled = self._text_pooled(feed)
            chunks.append(pooled.detach().float().cpu().numpy().astype(np.float32))

        mat = np.concatenate(chunks, axis=0) if chunks else np.zeros(
            (0, max(1, self.dim)), dtype=np.float32
        )
        if mat.size and int(mat.shape[-1]) != int(self.dim):
            raise RuntimeError(
                f"SigLIP 텍스트 임베딩 차원 불일치 "
                f"({int(mat.shape[-1])} ≠ 선언 {int(self.dim)})"
            )
        if l2 and mat.size:
            mat = mat / np.maximum(
                np.linalg.norm(mat, axis=1, keepdims=True), 1e-8
            )
        return mat

    def embed_text_batch(self, texts: Sequence[str], l2_normalize: bool = True) -> np.ndarray:
        return self.encode_text(texts, l2=l2_normalize)

    def embed_text(self, text: str) -> np.ndarray:
        mat = self.encode_text([text])
        if mat.size == 0:
            return np.zeros(max(1, self.dim), dtype=np.float32)
        return mat[0]

    def _letterbox(self, image: Image.Image) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        ps = max(1, int(self.patch_size))
        side = int(self.image_size)
        w, h = image.convert("RGB").size
        scale = min(side / float(max(1, w)), side / float(max(1, h)))
        nw = max(ps, int(round(w * scale)) // ps * ps)
        nh = max(ps, int(round(h * scale)) // ps * ps)
        nw = min(nw, side)
        nh = min(nh, side)

        resized = image.convert("RGB").resize((nw, nh), Image.BICUBIC)
        canvas = Image.new("RGB", (side, side), (255, 255, 255))
        ox = ((side - nw) // 2 // ps) * ps
        oy = ((side - nh) // 2 // ps) * ps
        canvas.paste(resized, (ox, oy))

        c0 = ox // ps
        r0 = oy // ps
        c1 = c0 + nw // ps - 1
        r1 = r0 + nh // ps - 1
        return canvas, (r0, r1, c0, c1)

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        arr = (arr - SIGLIP_IMAGE_MEAN) / SIGLIP_IMAGE_STD
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t.to(device=self.device, dtype=self._param_dtype)

    def _preprocess(self, image: Image.Image) -> torch.Tensor:
        canvas, _box = self._letterbox(image)
        return self._to_tensor(canvas)

    @torch.no_grad()
    def _dense_patch_features(self, hidden: torch.Tensor) -> torch.Tensor:
        head = getattr(self.vision_model, "head", None)
        if head is None:
            return hidden[0]

        n = int(hidden.shape[1])
        d = int(hidden.shape[-1])
        tokens = hidden.reshape(n, 1, d)
        probe = head.probe.repeat(n, 1, 1)

        attn, _w = head.attention(probe, tokens, tokens, need_weights=False)
        residual = attn
        x = head.layernorm(attn)
        x = residual + head.mlp(x)
        return x[:, 0]

    @staticmethod
    def _pool_grid(
        feats: np.ndarray, rows: int, cols: int, target: int
    ) -> Tuple[np.ndarray, int, int]:
        if target <= 0 or rows * cols <= target:
            return feats, rows, cols

        dim = int(feats.shape[-1])
        cube = feats.reshape(rows, cols, dim)

        scale = math.sqrt(float(target) / float(rows * cols))
        nr = max(2, min(rows, int(round(rows * scale))))
        nc = max(2, min(cols, int(round(cols * scale))))
        while nr * nc > target and (nr > 2 or nc > 2):
            if nr >= nc and nr > 2:
                nr -= 1
            elif nc > 2:
                nc -= 1
            else:
                break

        redges = np.linspace(0, rows, nr + 1).astype(int)
        cedges = np.linspace(0, cols, nc + 1).astype(int)

        out = np.zeros((nr, nc, dim), dtype=np.float32)
        for i in range(nr):
            r0 = redges[i]
            r1 = max(r0 + 1, redges[i + 1])
            for j in range(nc):
                c0 = cedges[j]
                c1 = max(c0 + 1, cedges[j + 1])
                out[i, j] = cube[r0:r1, c0:c1, :].mean(axis=(0, 1))

        return out.reshape(nr * nc, dim), int(nr), int(nc)

    def attach_vision(self):
        if self.vision_model is None:
            return
        try:
            dev = next(self.vision_model.parameters()).device
        except StopIteration:
            return
        if dev.type != self.device.type:
            self._log(f"  🔌 [{self.label}] 비전 타워 재부착 (JIT)")
            self.vision_model.to(self.device)

    def release_vision(self, reason: str = ""):
        if self.vision_model is None:
            return
        try:
            dev = next(self.vision_model.parameters()).device
        except StopIteration:
            return
        if dev.type == "cpu":
            return
        try:
            nbytes = sum(
                p.numel() * p.element_size()
                for p in self.vision_model.parameters()
            )
        except Exception:
            nbytes = 0
        self.vision_model.to("cpu")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        self._log(
            f"  ♻️ [VISION RELEASE] {self.label} 비전 가중치를 반납했습니다 "
            f"(약 {nbytes / 1e6:.0f} MB){' — ' + reason if reason else ''}"
        )

    @torch.no_grad()
    def embed_image_patches(self, image: Image.Image) -> dict:
        self.attach_vision()
        canvas, box = self._letterbox(image)
        px = self._to_tensor(canvas)

        out = self.vision_model(pixel_values=px)
        hidden = _as_tensor(out, ("last_hidden_state",))
        if hidden is None:
            raise RuntimeError(
                f"SigLIP 비전 타워 출력에서 텐서를 찾지 못했습니다 "
                f"(type={type(out).__name__})"
            )
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)

        feats_t = self._dense_patch_features(hidden)
        feats = feats_t.detach().float().cpu().numpy().astype(np.float32)

        rows, cols = self.rows, self.cols
        if feats.shape[0] != rows * cols:
            side = int(round(float(feats.shape[0]) ** 0.5))
            if side * side == feats.shape[0]:
                rows = cols = side

        r0, r1, c0, c1 = box
        if 0 <= r0 <= r1 < rows and 0 <= c0 <= c1 < cols:
            cube = feats.reshape(rows, cols, -1)[r0:r1 + 1, c0:c1 + 1, :]
            rows = int(r1 - r0 + 1)
            cols = int(c1 - c0 + 1)
            feats = cube.reshape(rows * cols, -1)
            self._log(
                f"  🧬 [PATCH GRID] 종횡비 보존 레터박스 → 내용 격자 "
                f"{rows}x{cols}={rows * cols} (전체 {self.rows}x{self.cols})"
            )

        if self.max_patches > 0 and rows * cols > self.max_patches:
            br, bc = rows, cols
            feats, rows, cols = self._pool_grid(feats, rows, cols, self.max_patches)
            self._log(
                f"  🧬 [PATCH POOL] {br}x{bc}={br * bc} → {rows}x{cols}="
                f"{rows * cols} (상한 {self.max_patches}) — 패치 하나가 "
                f"라벨+값 셀 전체를 보도록 거칠게 만듭니다."
            )

        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(norms, 1e-8)

        return {
            "features": feats,
            "valid_features": feats,
            "mask": np.ones((feats.shape[0],), dtype=np.int64),
            "rows": int(rows),
            "cols": int(cols),
        }

    @torch.no_grad()
    def embed_image_for_scoring(self, image: Image.Image) -> np.ndarray:
        return self.embed_image_patches(image)["valid_features"]

    @torch.no_grad()
    def selftest(self, emit=None) -> dict:
        emit = emit or self._log
        report: Dict[str, object] = {"ok": False, "dim": int(self.dim)}

        emit(f"  🧭 [{self.label}] 조인트 공간 자가검진 시작")

        feed = self._tokenize(["commercial invoice"])
        emit(
            f"    · 토크나이저 {type(self.tokenizer).__name__} "
            f"| vocab={len(self.tokenizer)} "
            f"| pad='{getattr(self.tokenizer, 'pad_token', None)}' "
            f"| input_ids{tuple(feed['input_ids'].shape)}"
        )

        pooled = self._text_pooled(feed)
        diagnostics.describe_tensor(f"{self.label}.text.pooled", pooled, emit)
        if int(pooled.shape[-1]) != int(self.dim):
            raise RuntimeError(
                f"[{self.label}] 텍스트 타워 출력 {int(pooled.shape[-1])}차원이 "
                f"선언 {int(self.dim)}차원과 다릅니다."
            )

        blank = Image.new(
            "RGB", (self.image_size, self.image_size), (255, 255, 255)
        )
        px = self._preprocess(blank)
        diagnostics.describe_tensor(f"{self.label}.vision.pixel_values", px, emit)

        out = self.vision_model(pixel_values=px)
        hidden = _as_tensor(out, ("last_hidden_state",))
        if hidden is None:
            raise RuntimeError(
                f"[{self.label}] 비전 타워 출력에서 텐서를 찾지 못했습니다 "
                f"(type={type(out).__name__})"
            )
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        diagnostics.describe_tensor(
            f"{self.label}.vision.last_hidden_state", hidden, emit
        )

        dense = self._dense_patch_features(hidden)
        diagnostics.describe_tensor(f"{self.label}.vision.dense_patch", dense, emit)
        if int(dense.shape[-1]) != int(self.dim):
            raise RuntimeError(
                f"[{self.label}] 패치 dense 출력 {int(dense.shape[-1])}차원이 "
                f"텍스트 {int(self.dim)}차원과 다릅니다. 조인트 공간이 성립하지 않습니다."
            )

        n_patch = int(dense.shape[0])
        if n_patch != self.rows * self.cols:
            emit(
                f"    ⚠ 패치 수 {n_patch} ≠ 선언 격자 "
                f"{self.rows}x{self.cols}={self.rows * self.cols} — "
                f"격자 좌표를 재계산합니다."
            )

        probe = diagnostics.probe_space(self.label, self.encode_text, emit)
        report["probe"] = probe
        report["patches"] = n_patch
        report["ok"] = True

        emit(
            f"  🧭 [{self.label}] 자가검진 통과 — 텍스트/패치 모두 "
            f"{int(self.dim)}차원 단일 공간, 패치 {n_patch}개"
        )
        return report

    def unload(self):
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        self.vision_model = None
        self.text_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()