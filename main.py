import webview
import json
import os
import sys
import threading
import traceback
import base64
import io
from pathlib import Path
from typing import Optional

from PIL import Image
import numpy as np

import warnings

warnings.warn(
    "main.py 는 더 이상 유지되지 않습니다. 새 진입점은 app.py 입니다.\n"
    "  UI  : python app.py\n"
    "  CLI : python app.py <파일> --save\n"
    "  상태: python app.py --check-only",
    DeprecationWarning,
    stacklevel=2,
)

from app import NMSOcrApp, build_parser, main as app_main

BASE_DIR = Path(__file__).resolve().parent
SCHEMA_DIR = BASE_DIR / "schema"
UI_DIR = BASE_DIR / "ui"
MODELS_DIR = BASE_DIR / "models"

AX_VE_PATH = str(MODELS_DIR / "ax-ve")
HAYAI_PATH = str(MODELS_DIR / "hayai")
LLM_PATH = str(MODELS_DIR / "alphaedge-ai")


class VisionNMSCropAPI(NMSOcrApp):
    def __init__(self):
        super().__init__()
        self.axve = None
        self.current_image_path: str = ""
        self.patch_grid = None

    def _log(self, msg: str):
        self.log_lines.append(msg)
        try:
            webview.windows[0].evaluate_js(
                f"appendLog({json.dumps(msg)})"
            )
        except Exception:
            pass

    def _update_progress(self, pct: int, label: str = ""):
        try:
            webview.windows[0].evaluate_js(
                f"updateProgress({pct}, {json.dumps(label)})"
            )
        except Exception:
            pass

    def _on_download_progress(self, info: dict):
        try:
            webview.windows[0].evaluate_js(
                f"onModelDownload({json.dumps(info)})"
            )
        except Exception:
            pass

    def start_model_download(self):
        return {
            "ok": False,
            "error": "고정 모델 자동 다운로드는 제거되었습니다. models/ 하위에 직접 배치하세요.",
        }

    def cancel_model_download(self):
        return {"ok": True}

    def get_gpu_info(self):
        device, accel_label = detect_accelerator()
        dtype = select_dtype(device)
        vram = get_vram_info()
        return {
            "device": str(device),
            "accel_label": accel_label,
            "dtype": str(dtype),
            "vram": vram,
        }

    def load_models(self):
        global AX_VE_PATH, HAYAI_PATH
        AX_VE_PATH = _resolve_model_dir(BASE_DIR / "ax-ve", MODELS_DIR / "ax-ve")
        HAYAI_PATH = _resolve_model_dir(BASE_DIR / "hayai", MODELS_DIR / "hayai")

        missing = [k for k in MODEL_REGISTRY if not is_model_ready(k)]
        if missing:
            self._log(f"📥 누락 모델 {len(missing)}개 감지 → 자동 다운로드 시작")
            self._update_progress(0, "모델 다운로드 준비 중")
            self.start_model_download()
            return {
                "ok": False,
                "downloading": True,
                "error": "모델 파일이 없어 다운로드를 시작합니다. 완료 후 다시 [모델 로드]를 눌러주세요.",
            }

        try:
            self._log(f"🔄 A.X-VE 비전 임베딩 모델 로드 중... (경로: {AX_VE_PATH})")
            self._update_progress(10, "A.X-VE 로드 중")
            self.axve = AXVEEmbedder(AX_VE_PATH)
            device_info = f"device={self.axve.device}, dtype={self.axve.dtype}"
            self._log(f"✅ A.X-VE 로드 완료 ({device_info})")
            self._update_progress(30, "A.X-VE 로드 완료")
        except Exception as e:
            self._log(f"❌ A.X-VE 로드 실패: {e}")
            return {"ok": False, "error": str(e)}

        try:
            self._log("🔄 Hayai OCR 모델 로드 중...")
            self._update_progress(50, "Hayai OCR 로드 중")
            self.ocr = HayaiOCR(HAYAI_PATH)
            device_info = f"device={self.ocr.device}, dtype={self.ocr.dtype}"
            self._log(f"✅ Hayai OCR 로드 완료 ({device_info})")
            self._update_progress(80, "Hayai OCR 로드 완료")
        except Exception as e:
            self._log(f"❌ Hayai OCR 로드 실패: {e}")
            return {"ok": False, "error": str(e)}

        self._update_progress(100, "모델 로드 완료")
        return {"ok": True}

    def toggle_devtools(self):
        try:
            window = webview.windows[0]
            window.toggle_devtools()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def open_file_dialog(self):
        file_types = (
            "이미지 파일 (*.png;*.jpg;*.jpeg;*.bmp;*.tiff;*.pdf)",
            "모든 파일 (*.*)",
        )
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            directory=str(BASE_DIR),
            file_types=file_types,
        )
        if result and len(result) > 0:
            return result[0]
        return None

    def load_image(self, path: str = ""):
        if not path:
            path = self.open_file_dialog()
            if not path:
                return {"ok": False}

        path = str(path)
        self.current_image_path = path
        self._log(f"📂 이미지 로드: {path}")

        try:
            if path.lower().endswith(".pdf"):
                self._log("📄 PDF 렌더링 중...")
                self.current_image = self._load_pdf(path)
            else:
                self.current_image = Image.open(path).convert("RGB")

            self._log(
                f"✅ 이미지 로드 완료 "
                f"({self.current_image.width}x{self.current_image.height})"
            )
            return {"ok": True, "width": self.current_image.width,
                    "height": self.current_image.height}
        except Exception as e:
            self._log(f"❌ 이미지 로드 실패: {e}")
            return {"ok": False, "error": str(e)}

    def classify_document(self):
        if self.current_image is None:
            return {"ok": False, "error": "이미지가 로드되지 않았습니다."}

        try:
            gray = np.array(self.current_image.convert("L"), dtype=np.float32)
            h, w = gray.shape

            bw_ratio = float(np.mean((gray < 50) | (gray > 205)))
            mid_ratio = float(np.mean((gray >= 80) & (gray <= 180)))

            from PIL import ImageFilter
            edge_arr = np.array(
                self.current_image.convert("L").filter(ImageFilter.FIND_EDGES),
                dtype=np.float32,
            )
            edge_density = float(np.mean(edge_arr > 40))

            unique_colors = len(self.current_image.convert("RGB").getcolors(maxcolors=65536) or [])
            color_richness = min(unique_colors / 5000.0, 1.0)

            scores = {}
            for f in SCHEMA_DIR.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    domain = data.get("domain", "")

                    if domain == "comics":
                        s = bw_ratio * 0.45 + edge_density * 0.35 + (1.0 - color_richness) * 0.20
                    elif domain == "trade":
                        s = mid_ratio * 0.40 + color_richness * 0.30 + (1.0 - bw_ratio) * 0.30
                    elif domain == "logistics":
                        s = mid_ratio * 0.35 + color_richness * 0.25 + (1.0 - edge_density) * 0.40
                    elif domain == "commerce":
                        s = mid_ratio * 0.35 + color_richness * 0.35 + (1.0 - bw_ratio) * 0.30
                    else:
                        s = 0.10

                    scores[f.name] = round(s, 4)
                except Exception:
                    pass

            if not scores:
                return {"ok": False, "error": "스키마 파일이 없습니다."}

            best = max(scores, key=scores.get)
            self._log(
                f"🔍 문서 분류: {best} (bw={bw_ratio:.2f}, "
                f"edge={edge_density:.2f}, color={color_richness:.2f})"
            )
            for k, v in sorted(scores.items(), key=lambda x: -x[1]):
                self._log(f"    {k}: {v:.4f}")

            return {"ok": True, "schema": best, "scores": scores}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _load_pdf(self, path: str) -> Image.Image:
        try:
            import fitz
            doc = fitz.open(path)
            page = doc[0]
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            doc.close()
            return img
        except ImportError:
            from pdf2image import convert_from_path
            images = convert_from_path(path, first_page=1, last_page=1, dpi=150)
            return images[0].convert("RGB")

    def list_schemas(self):
        schemas = []
        for f in SCHEMA_DIR.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                schemas.append({
                    "filename": f.name,
                    "domain": data.get("domain", "unknown"),
                    "doc_type": data.get("doc_type", "unknown"),
                    "field_count": len(data.get("fields", {})),
                })
            except Exception:
                pass
        return {"ok": True, "schemas": schemas}

    def load_schema(self, filename: str):
        schema_path = SCHEMA_DIR / filename
        if not schema_path.exists():
            return {"ok": False, "error": f"스키마 없음: {filename}"}
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema = json.load(f)
            field_names = list(schema.get("fields", {}).keys())
            return {
                "ok": True,
                "domain": schema.get("domain", ""),
                "doc_type": schema.get("doc_type", ""),
                "fields": field_names,
                "schema": schema,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def run_pipeline(
        self,
        schema_json: str,
        iou_threshold: float = 0.8,
        margin_threshold: float = 0.28,
        text_embed_mode: str = "hayai",
    ):
        self.log_lines = []

        if self.axve is None or self.ocr is None:
            return {"ok": False, "error": "모델이 로드되지 않았습니다."}

        if self.current_image is None:
            return {"ok": False, "error": "이미지가 로드되지 않았습니다."}

        try:
            return self._run_pipeline_impl(
                schema_json, iou_threshold, margin_threshold, text_embed_mode
            )
        except Exception as e:
            tb = traceback.format_exc()
            self._log(f"❌ 파이프라인 오류:\n{tb}")
            return {"ok": False, "error": str(e)}

    def _run_pipeline_impl(
        self,
        schema_json: str,
        iou_threshold: float,
        margin_threshold: float,
        text_embed_mode: str,
    ):
        schema = json.loads(schema_json)
        scorer = FieldScorer(schema)
        self.nms_processor.iou_threshold = iou_threshold
        self.nms_processor.margin_threshold = margin_threshold

        self._log("═══ STEP 1: 패치 임베딩 생성 ═══")
        self._update_progress(10, "패치 임베딩 생성 중")

        patch_grid = self.axve.embed_image(self.current_image)
        self.patch_grid = patch_grid
        self._log(
            f"  패치 격자: {patch_grid.grid_rows}x{patch_grid.grid_cols} "
            f"= {patch_grid.num_patches} 패치"
        )
        self._log(
            f"  스케일: x={patch_grid.scale_x:.3f}, "
            f"y={patch_grid.scale_y:.3f}"
        )
        self._update_progress(25, "패치 임베딩 완료")

        self._log("═══ STEP 2: 필드별 코사인 스코어링 ═══")
        self._update_progress(30, "텍스트 임베딩 생성 중")

        def embed_text_fn(text: str) -> Optional[np.ndarray]:
            try:
                return self.ocr.embed_text(text)
            except Exception:
                return None

        hayai_patches = self.ocr.embed_image_for_scoring(self.current_image)
        num_hayai = hayai_patches.shape[0]
        num_grid = patch_grid.num_patches

        if num_hayai != num_grid:
            self._log(
                f"  ⚠ 패치 수 불일치 (Hayai={num_hayai}, Grid={num_grid}) → 보간"
            )
            indices = np.linspace(0, num_hayai - 1, num_grid).astype(int)
            scoring_embeddings = hayai_patches[indices]
        else:
            scoring_embeddings = hayai_patches

        self._log(
            f"  스코어링 임베딩: {scoring_embeddings.shape} "
            f"(Hayai 512-dim)"
        )

        field_scores = scorer.score_all_patches(
            scoring_embeddings, embed_text_fn
        )
        self._log(f"  필드 {len(field_scores)}개 스코어링 완료")
        self._update_progress(50, "스코어링 완료")

        self._log("═══ STEP 3: 후보 생성 + 편견 필터링 ═══")
        self._update_progress(55, "후보 생성 중")

        candidates = []
        field_names = scorer.get_field_names()

        for field_name in field_names:
            if field_name not in field_scores:
                continue
            scores = field_scores[field_name]
            for i in range(len(scores)):
                row = i // patch_grid.grid_cols
                col = i % patch_grid.grid_cols
                net_score = scores[i]

                bias_vec = embed_text_fn(
                    scorer.get_bias_phrases(field_name)[0]
                    if scorer.get_bias_phrases(field_name)
                    else ""
                )
                prej_vec = embed_text_fn(
                    scorer.get_prejudice_phrases(field_name)[0]
                    if scorer.get_prejudice_phrases(field_name)
                    else ""
                )

                bias_sim = 0.0
                prej_sim = 0.0
                if bias_vec is not None:
                    bias_sim = scorer.cosine_similarity(
                        scoring_embeddings[i],
                        scorer.l2_normalize(bias_vec)
                    )
                if prej_vec is not None:
                    prej_sim = scorer.cosine_similarity(
                        scoring_embeddings[i],
                        scorer.l2_normalize(prej_vec)
                    )

                if prej_sim > bias_sim:
                    self._log(
                        f"  🚫 Prejudice Drop: "
                        f"{field_name} row={row},col={col} "
                        f"(prej={prej_sim:.4f} > bias={bias_sim:.4f})"
                    )
                    continue

                bbox = patch_grid.get_patch_bbox(row, col)
                candidates.append(
                    Candidate(
                        field_name=field_name,
                        bbox=bbox,
                        score=net_score,
                        row=row,
                        col=col,
                    )
                )

        self._log(f"  후보 {len(candidates)}개 생성 완료")
        self._update_progress(65, "후보 생성 완료")

        self._log("═══ STEP 4: NMS 중복 억제 ═══")
        self._update_progress(70, "NMS 실행 중")

        assigned, nms_log = self.nms_processor.process(candidates)
        for line in nms_log:
            self._log(f"  {line}")
        self._log(f"  확정 후보: {len(assigned)}개")
        self._update_progress(80, "NMS 완료")

        self._log("═══ STEP 5: Crop + OCR ═══")
        self._update_progress(85, "Crop + OCR 실행 중")

        crop_results = self.cropper.crop_all(
            self.current_image,
            assigned,
            ocr_fn=self.ocr.ocr_crop,
        )
        self._log(f"  Crop 결과: {len(crop_results)}개")
        self._update_progress(95, "Crop 완료")

        result_list = []
        for r in crop_results:
            crop_b64 = ""
            try:
                buf = io.BytesIO()
                r["crop_image"].save(buf, format="PNG")
                crop_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                crop_b64 = ""

            result_list.append({
                "field_name": r["field_name"],
                "bbox": r["bbox"],
                "score": r["score"],
                "ocr_text": r["ocr_text"],
                "crop_base64": crop_b64,
            })

        self._log("═══ 파이프라인 완료 ═══")
        self._update_progress(100, "완료")

        return {
            "ok": True,
            "results": result_list,
            "log": self.log_lines,
            "patch_grid": {
                "rows": patch_grid.grid_rows,
                "cols": patch_grid.grid_cols,
                "scale_x": patch_grid.scale_x,
                "scale_y": patch_grid.scale_y,
            },
        }

    def save_results(self, results_json: str):
        try:
            results = json.loads(results_json)
        except Exception:
            return {"ok": False, "error": "JSON 파싱 실패"}

        try:
            out_dir = str(BASE_DIR / "output")
            os.makedirs(out_dir, exist_ok=True)

            for r in results:
                field_name = r.get("field_name", "unknown")
                crop_b64 = r.get("crop_base64", "")
                if crop_b64:
                    img_data = base64.b64decode(crop_b64)
                    img = Image.open(io.BytesIO(img_data))
                    img.save(os.path.join(out_dir, f"{field_name}.png"))

            json_path = os.path.join(out_dir, "results.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self._log(f"💾 결과 저장 완료: {out_dir}")
            return {"ok": True, "output_dir": out_dir}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_overlay_data(self):
        if self.patch_grid is None:
            return {"ok": False, "error": "패치 격자 없음"}

        overlay = {
            "rows": self.patch_grid.grid_rows,
            "cols": self.patch_grid.grid_cols,
            "scale_x": self.patch_grid.scale_x,
            "scale_y": self.patch_grid.scale_y,
            "patch_size": self.patch_grid.patch_size,
        }
        return {"ok": True, "overlay": overlay}


def main():
    return app_main()


if __name__ == "__main__":
    import sys
    sys.exit(main())