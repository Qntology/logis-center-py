import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np

from .model_manager import (
    BASE_DIR,
    OCR_PATH,
    MissingModelError,
    ensure_model_dir,
    resolve_siglip2_ref,
)


HAYAI_CODE_FILES = (
    "configuration_hayai.py",
    "modeling_hayai.py",
)


def _resolve_hayai_code_dir() -> Path:
    candidates = [BASE_DIR / "hayai", OCR_PATH]
    for cand in candidates:
        cand = Path(cand)
        if not cand.is_dir():
            continue
        if all((cand / f).exists() for f in HAYAI_CODE_FILES):
            return cand.resolve()
    tried = "\n".join(f"    - {c}" for c in candidates)
    raise MissingModelError(
        "Hayai OCR 모델 정의 코드를 찾지 못했습니다.\n"
        f"  필요 파일 : {', '.join(HAYAI_CODE_FILES)}\n"
        f"  탐색 경로 :\n{tried}"
    )


def _load_hayai():
    hayai_dir = _resolve_hayai_code_dir()

    if "hayai" not in sys.modules:
        pkg = importlib.util.module_from_spec(
            importlib.machinery.ModuleSpec("hayai", None, is_package=True)
        )
        pkg.__path__ = [str(hayai_dir)]
        pkg.__package__ = "hayai"
        sys.modules["hayai"] = pkg

    def _load(name: str):
        full = f"hayai.{name}"
        if full in sys.modules:
            return sys.modules[full]
        target = hayai_dir / f"{name}.py"
        if not target.exists():
            raise MissingModelError(f"Hayai 코드 파일 누락: {target}")
        spec = importlib.util.spec_from_file_location(
            full,
            str(target),
            submodule_search_locations=[str(hayai_dir)],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = "hayai"
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    _load("configuration_hayai")
    _load("modeling_hayai")

    cfg_mod = sys.modules["hayai.configuration_hayai"]
    model_mod = sys.modules["hayai.modeling_hayai"]

    return cfg_mod.HayaiConfig, model_mod.HayaiModel


HayaiConfig, HayaiModel = None, None


def _ensure_hayai():
    global HayaiConfig, HayaiModel
    if HayaiConfig is None:
        HayaiConfig, HayaiModel = _load_hayai()


def _log_tensor_summary_ocr(model, name: str):
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


from transformers import AutoTokenizer, AutoProcessor
from PIL import Image

from .device import detect_accelerator, select_dtype, configure_backends
from .embedding import load_pretrained_with_dtype, l2_normalize_rows


class HayaiOCR:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        siglip2_ref: Optional[str] = None,
        max_patches: int = 256,
    ):
        _ensure_hayai()

        if model_path is None:
            model_path = str(ensure_model_dir("hayai"))
        else:
            model_path = str(model_path)
        self.model_path = model_path
        self.max_patches = int(max_patches)

        self.siglip2_ref = siglip2_ref or resolve_siglip2_ref()

        self.device, self.accel_label = detect_accelerator(device)
        self.dtype = select_dtype(self.device)
        configure_backends(self.device)

        self.config = HayaiConfig.from_pretrained(
            model_path,
            siglip2_ref=self.siglip2_ref,
        )
        self.model = load_pretrained_with_dtype(
            HayaiModel,
            model_path,
            self.dtype,
            config=self.config,
        )
        self.model.to(self.device)
        self.model.eval()

        try:
            self.image_processor = AutoProcessor.from_pretrained(self.siglip2_ref)
        except Exception as e:
            raise MissingModelError(
                "SigLIP2 이미지 프로세서를 불러오지 못했습니다.\n"
                f"  참조 : {self.siglip2_ref}\n"
                f"  원인 : {e}\n"
                "  오프라인 환경이라면 models/siglip2-base-patch16-naflex/ 에\n"
                "  config.json / preprocessor_config.json / tokenizer.json 을 배치하세요."
            )

        self.tokenizer, self.tokenizer_vocab, self.tokenizer_source = \
            self._resolve_tokenizer()

        self.vocab_aligned = (
            self.tokenizer_vocab > 0
            and abs(self.tokenizer_vocab - int(self.config.vocab_size)) <= 64
        )

        self.patch_size = self._resolve_patch_size()
        self._loaded = True
        print(f"[GPU/HayaiOCR] Model loaded | device={self.device} | dtype={self.dtype}")
        print(f"[GPU/HayaiOCR] weights={self.model_path} | siglip2={self.siglip2_ref}")
        print(
            f"[GPU/HayaiOCR] tokenizer={self.tokenizer_source} "
            f"| vocab={self.tokenizer_vocab} vs model={int(self.config.vocab_size)} "
            f"| aligned={'YES' if self.vocab_aligned else 'NO'}"
        )
        if not self.vocab_aligned:
            print(
                "[HayaiOCR] ⚠ 토크나이저/모델 어휘 불일치입니다.\n"
                f"    모델 vocab_size={int(self.config.vocab_size)} 인데 "
                f"토크나이저 어휘는 {self.tokenizer_vocab} 개입니다.\n"
                "    OCR 출력이 <unused..> 같은 예약 토큰으로, 텍스트 앵커 임베딩이\n"
                "    서로 구분되지 않는 벡터로 붕괴합니다.\n"
                f"    models/hayai/ 에 학습에 쓰인 토크나이저(tokenizer.json 등)를 배치하세요."
            )
        _log_tensor_summary_ocr(self.model, "HayaiOCR")

    def _resolve_patch_size(self) -> int:
        ip = getattr(self.image_processor, "image_processor", self.image_processor)
        for attr in ("patch_size", "patch_size_"):
            val = getattr(ip, attr, None)
            if isinstance(val, int) and val > 0:
                return val
            if isinstance(val, dict):
                h = val.get("height")
                if isinstance(h, int) and h > 0:
                    return h
        vc = getattr(self.model.vision_encoder.config, "patch_size", None)
        if isinstance(vc, int) and vc > 0:
            return vc
        return 16

    @staticmethod
    def _vocab_size_of(tok) -> int:
        try:
            n = len(tok)
            if isinstance(n, int) and n > 0:
                return int(n)
        except Exception:
            pass
        try:
            return int(getattr(tok, "vocab_size", 0) or 0)
        except Exception:
            return 0

    def _resolve_tokenizer(self):
        want = int(getattr(self.config, "vocab_size", 0) or 0)

        candidates = []
        for ref in (self.model_path, str(OCR_PATH), self.siglip2_ref):
            if ref and ref not in candidates:
                candidates.append(ref)

        best = None
        best_size = 0
        best_ref = ""
        best_gap = None
        errors = []

        for ref in candidates:
            try:
                tok = AutoTokenizer.from_pretrained(ref, trust_remote_code=True)
            except Exception as e:
                errors.append(f"{ref}: {e}")
                continue

            size = self._vocab_size_of(tok)
            gap = abs(size - want) if want else 0

            if want and gap <= 64:
                return tok, size, ref

            if best is None or (best_gap is not None and gap < best_gap):
                best, best_size, best_ref, best_gap = tok, size, ref, gap

        if best is not None:
            return best, best_size, best_ref

        raise MissingModelError(
            "Hayai OCR 토크나이저를 불러오지 못했습니다.\n"
            f"  탐색 경로 : {', '.join(candidates)}\n"
            f"  원인 : {' | '.join(errors) if errors else '알 수 없음'}\n"
            "  models/hayai/ 또는 models/siglip2-base-patch16-naflex/ 에\n"
            "  tokenizer.json / tokenizer_config.json 을 배치하세요."
        )

    def get_device(self) -> torch.device:
        return self.device

    def get_embed_dim(self) -> int:
        return int(self.config.d_model)

    def get_token_embeddings(self) -> torch.Tensor:
        return self.model.decoder.token_embeddings.weight.data

    @torch.no_grad()
    def embed_text(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.config.d_model, dtype=np.float32)
        token_ids = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(self.device)
        if token_ids.numel() == 0:
            return np.zeros(self.config.d_model, dtype=np.float32)

        limit = int(self.config.vocab_size)
        flat = token_ids.reshape(-1)
        keep = flat[(flat >= 0) & (flat < limit)]
        if keep.numel() == 0:
            return np.zeros(self.config.d_model, dtype=np.float32)

        embeddings = self.model.decoder.token_embeddings(keep.unsqueeze(0))
        pooled = embeddings.mean(dim=1)
        return pooled.float().cpu().numpy()[0]

    @torch.no_grad()
    def embed_text_batch(self, texts, l2_normalize: bool = True) -> np.ndarray:
        vecs = [self.embed_text(t) for t in texts]
        if not vecs:
            return np.zeros((0, self.config.d_model), dtype=np.float32)
        mat = np.stack(vecs).astype(np.float32)
        if l2_normalize:
            mat = l2_normalize_rows(mat)
        return mat

    def _fit_to_patch_budget(self, image: Image.Image) -> Image.Image:
        ps = self.patch_size
        w, h = image.size
        w = max(w, ps)
        h = max(h, ps)
        num_patches = (w // ps) * (h // ps)
        if num_patches <= self.max_patches:
            return image
        scale = (self.max_patches / float(num_patches)) ** 0.5
        new_w = max(int(w * scale) // ps * ps, ps)
        new_h = max(int(h * scale) // ps * ps, ps)
        return image.resize((new_w, new_h), Image.LANCZOS)

    def _prepare_image_for_vision(self, image: Image.Image):
        image = self._fit_to_patch_budget(image.convert("RGB"))

        proc_inputs = self.image_processor(images=image, return_tensors="pt")

        pixel_values = proc_inputs["pixel_values"].to(self.device)
        if pixel_values.dim() == 2:
            pixel_values = pixel_values.unsqueeze(0)

        pixel_attention_mask = proc_inputs.get("pixel_attention_mask", None)
        if pixel_attention_mask is None:
            pixel_attention_mask = torch.ones(
                pixel_values.shape[:2],
                dtype=torch.long,
                device=self.device,
            )
        else:
            pixel_attention_mask = pixel_attention_mask.to(self.device)

        if pixel_attention_mask.dim() == 1:
            pixel_attention_mask = pixel_attention_mask.unsqueeze(0)
        elif pixel_attention_mask.dim() > 2:
            pixel_attention_mask = pixel_attention_mask.reshape(
                pixel_attention_mask.shape[0], -1
            )

        spatial_shapes = proc_inputs.get("spatial_shapes", None)
        if spatial_shapes is None:
            w_img, h_img = image.size
            spatial_shapes = torch.tensor(
                [[h_img // self.patch_size, w_img // self.patch_size]],
                dtype=torch.long,
            )
        if not torch.is_tensor(spatial_shapes):
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long)
        if spatial_shapes.dim() == 1:
            spatial_shapes = spatial_shapes.unsqueeze(0)
        spatial_shapes = spatial_shapes.to(dtype=torch.long, device=self.device)

        pixel_values = pixel_values.to(dtype=self.model.dtype)

        return pixel_values, pixel_attention_mask, spatial_shapes

    @torch.no_grad()
    def embed_image_patches(self, image: Image.Image) -> dict:
        pixel_values, pixel_attention_mask, spatial_shapes = \
            self._prepare_image_for_vision(image)

        vision_outputs = self.model.vision_encoder(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
        )
        visual_features = vision_outputs.last_hidden_state
        projected = self.model.decoder.projector(visual_features)

        feats = projected.squeeze(0).float().cpu().numpy()
        mask = pixel_attention_mask.squeeze(0).long().cpu().numpy()
        rows, cols = [int(v) for v in spatial_shapes[0].tolist()]

        valid = feats[mask.astype(bool)] if mask.shape[0] == feats.shape[0] else feats

        return {
            "features": feats,
            "valid_features": valid,
            "mask": mask,
            "rows": rows,
            "cols": cols,
        }

    @torch.no_grad()
    def embed_image_for_scoring(self, image: Image.Image) -> np.ndarray:
        out = self.embed_image_patches(image)
        return out["valid_features"]

    @torch.no_grad()
    def ocr_crop(
        self,
        image: Image.Image,
        bbox: tuple,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        repetition_penalty: float = 1.05,
    ) -> str:
        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(image.width, int(x1))
        y1 = min(image.height, int(y1))

        if x1 <= x0 or y1 <= y0:
            return ""

        crop = image.crop((x0, y0, x1, y1))
        return self.ocr_image(
            crop,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )

    @torch.no_grad()
    def ocr_image(
        self,
        image: Image.Image,
        max_new_tokens: int = 128,
        num_beams: int = 1,
        repetition_penalty: float = 1.05,
    ) -> str:
        if image.width < 2 or image.height < 2:
            return ""

        pixel_values, pixel_attention_mask, spatial_shapes = \
            self._prepare_image_for_vision(image)

        results = self.model.generate(
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
            spatial_shapes=spatial_shapes,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
        )

        return results[0] if results else ""

    def unload(self):
        try:
            self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()