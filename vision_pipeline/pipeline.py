import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from .doc_type_nms import DocTypeVerdict, classify_doc_type, load_doc_type_specs
from .field_heatmap import (
    apply_arena,
    build_field_heatmaps,
    spatial_residual,
    suppress_title_rows,
)
from .nms_arena import ArenaResult, run_arena
from .ocr_extract import (
    ExtractedField,
    extract_from_crops,
    fields_to_record,
    record_to_json,
)
from .patch_grid import (
    VisionPatchGrid,
    build_patch_grid,
    column_gutters,
    legibility_map,
    row_gutters,
)
from .text_upscale import VISION_PATCH_PX
from .value_grounding import GroundingClaim, apply_verdicts, verify_claims
from .vision_nms import CropPlan, plan_crops


class _LiveLog(list):
    def __init__(self, emit: Optional[Callable[[str], None]] = None):
        super().__init__()
        self._emit = emit

    def append(self, item):
        super().append(item)
        if self._emit is None:
            return
        try:
            self._emit(item)
        except Exception:
            pass

    def extend(self, items):
        for it in items:
            self.append(it)


class VisionPipelineConfig:
    def __init__(
        self,
        iou_threshold: float = 0.80,
        margin_threshold: float = 0.28,
        target_text_px: float = VISION_PATCH_PX,
        max_tiles: int = 3,
        title_suppress_ratio: float = 0.12,
        title_penalty: float = 0.5,
        enable_spatial_residual: bool = True,
        enable_grounding: bool = True,
        cross_prejudice: bool = False,
        keep_crop: bool = True,
        prefer_grid: str = "siglip2",
        lang_code: str = "",
        doc_code: str = "",
        source_path: str = "",
        enable_gutters: bool = True,
        max_crop_cols: int = 0,
        patch_margin: float = 0.12,
        min_territory: int = 2,
        arena_rounds: int = 3,
        line_read: bool = True,
        line_span: int = 1,
        line_stride: int = 1,
        line_overlap: int = 1,
    ):
        self.iou_threshold = float(iou_threshold)
        self.margin_threshold = float(margin_threshold)
        self.target_text_px = float(target_text_px)
        self.max_tiles = int(max_tiles)
        self.title_suppress_ratio = float(title_suppress_ratio)
        self.title_penalty = float(title_penalty)
        self.enable_spatial_residual = bool(enable_spatial_residual)
        self.enable_grounding = bool(enable_grounding)
        self.cross_prejudice = bool(cross_prejudice)
        self.keep_crop = bool(keep_crop)
        self.prefer_grid = prefer_grid
        self.lang_code = str(lang_code or "")
        self.doc_code = str(doc_code or "")
        self.source_path = str(source_path or "")
        self.enable_gutters = bool(enable_gutters)
        self.max_crop_cols = int(max_crop_cols)
        self.patch_margin = float(patch_margin)
        self.min_territory = int(min_territory)
        self.arena_rounds = int(arena_rounds)
        self.line_read = bool(line_read)
        self.line_span = int(line_span)
        self.line_stride = int(line_stride)
        self.line_overlap = int(line_overlap)


class VisionPipelineResult:
    def __init__(self):
        self.ok: bool = False
        self.error: str = ""
        self.grid: Optional[VisionPatchGrid] = None
        self.doc_type: Optional[DocTypeVerdict] = None
        self.heatmaps: List = []
        self.arena: Optional[ArenaResult] = None
        self.absent_fields: List[str] = []
        self.plans: List[CropPlan] = []
        self.fields: List[ExtractedField] = []
        self.verdicts: List = []
        self.record: Dict[str, object] = {}
        self.json: Dict[str, object] = {}
        self.log: List[str] = []
        self.elapsed: float = 0.0

    def to_dict(self, include_images: bool = False) -> dict:
        return {
            "ok": self.ok,
            "error": self.error,
            "grid": self.grid.to_dict() if self.grid else {},
            "doc_type": self.doc_type.to_dict() if self.doc_type else {},
            "heatmaps": [
                h.to_dict(self.grid.rows if self.grid else 0,
                          self.grid.cols if self.grid else 0)
                for h in self.heatmaps
            ],
            "arena": self.arena.to_dict() if self.arena else {},
            "absent_fields": list(self.absent_fields),
            "plans": [p.to_dict() for p in self.plans],
            "results": [f.to_dict(include_image=include_images) for f in self.fields],
            "grounding": [v.to_dict() for v in self.verdicts],
            "record": self.record,
            "json": self.json,
            "elapsed": round(self.elapsed, 3),
            "log": self.log,
        }


class VisionPipeline:
    def __init__(
        self,
        embed_fn: Callable[[List[str]], np.ndarray],
        ocr=None,
        embedder=None,
        config: Optional[VisionPipelineConfig] = None,
        log: Optional[Callable[[str], None]] = None,
        nlp=None,
        crossover=None,
        joint=None,
        line_read_fn: Optional[Callable] = None,
    ):
        self.embed_fn = embed_fn
        self.ocr = ocr
        self.embedder = embedder
        self.joint = joint
        self.config = config or VisionPipelineConfig()
        self._log_fn = log or (lambda m: None)
        self.nlp = nlp
        self.crossover = crossover
        self.line_read_fn = line_read_fn
        self.logs: List[str] = _LiveLog(self._log_fn)

    def _phase(self, name: str):
        if self.crossover is None:
            return
        try:
            if name == "embedding":
                self.crossover.enter_embedding_phase()
            elif name == "generation":
                self.crossover.switch_to_generation()
            elif name == "idle":
                self.crossover.mark_idle()
        except Exception as e:
            self._log(f"  ⚠ 페이즈 전환 실패({name}): {e}")

    def _refresh_slots(self):
        if self.crossover is None:
            return
        ocr = self.crossover.get("ocr")
        if ocr is not None:
            self.ocr = ocr
        jnt = self.crossover.get("joint")
        if jnt is not None:
            self.joint = jnt

    def _log(self, msg: str):
        self.logs.append(msg)

    def classify(
        self,
        image: Image.Image,
        schema_dir,
        grid: Optional[VisionPatchGrid] = None,
        text_sample: str = "",
        text_embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
    ) -> DocTypeVerdict:
        self.logs = _LiveLog(self._log_fn)
        if grid is None:
            grid = build_patch_grid(
                image, embedder=self.embedder, ocr=self.ocr,
                prefer=self.config.prefer_grid, joint=self.joint,
            )
        self._log(
            f"  격자 {grid.rows}x{grid.cols}={grid.num_patches} "
            f"| dim={grid.dim} | src={grid.source}"
        )
        if self.config.lang_code:
            self._log(f"  🈯 문서 유형 앵커 스코프: {self.config.lang_code}")
        specs, file_map = load_doc_type_specs(
            schema_dir, lang_code=self.config.lang_code, log=self.logs
        )
        return classify_doc_type(
            grid, specs, self.embed_fn, file_map=file_map,
            margin_threshold=self.config.margin_threshold,
            log=self.logs,
            text_sample=text_sample,
            text_embed_fn=text_embed_fn,
        )

    def run(
        self,
        image: Image.Image,
        schema: dict,
        refine_fn: Optional[Callable[[str, str, Image.Image], dict]] = None,
    ) -> VisionPipelineResult:
        result = VisionPipelineResult()
        started = time.time()
        self.logs = _LiveLog(self._log_fn)

        try:
            self._phase("embedding")
            self._refresh_slots()

            self._log("═══ STEP 1: 패치 임베딩 격자 ═══")
            grid = build_patch_grid(
                image, embedder=self.embedder, ocr=self.ocr,
                prefer=self.config.prefer_grid, joint=self.joint,
            )
            result.grid = grid
            self._log(
                f"  격자 {grid.rows}x{grid.cols}={grid.num_patches} "
                f"| dim={grid.dim} | src={grid.source}"
            )

            if self.joint is not None and grid.source == "siglip2":
                try:
                    self.joint.release_vision(
                        f"패치 {grid.num_patches}개를 호스트로 확보"
                    )
                except Exception as e:
                    self._log(f"  ⚠ 비전 반납 실패({e})")

            legib = None
            try:
                legib = legibility_map(image, grid.rows, grid.cols)
                self._log(legib.report())
            except Exception as e:
                self._log(f"  ⏭ 판독성 맵 생략: {e}")

            self._log("═══ STEP 2: 필드 친화도 맵 (Column Cosine) ═══")
            heatmaps, chrome_ref = build_field_heatmaps(
                grid, schema, self.embed_fn,
                cross_prejudice=self.config.cross_prejudice,
                lang_code=self.config.lang_code,
                log=self.logs,
                doc_code=self.config.doc_code or str(schema.get("code") or ""),
            )
            if not heatmaps:
                result.error = "필드 친화도 맵을 생성하지 못했습니다."
                result.log = self.logs
                return result

            heatmaps = suppress_title_rows(
                heatmaps, grid,
                top_ratio=self.config.title_suppress_ratio,
                penalty=self.config.title_penalty,
                schema=schema,
                log=self.logs,
            )

            self._log("═══ STEP 2.5: NMS ARENA (패치 단위 배타 경쟁) ═══")
            arena = run_arena(
                [h.category for h in heatmaps],
                {h.category: h.scores for h in heatmaps},
                chrome_ref,
                grid.num_patches,
                patch_margin=self.config.patch_margin,
                min_territory=self.config.min_territory,
                rounds=self.config.arena_rounds,
                log=self.logs,
            )
            result.arena = arena
            result.absent_fields = list(arena.absent)

            heatmaps = apply_arena(heatmaps, arena, grid.num_patches, log=self.logs)
            if not heatmaps:
                result.error = (
                    "모든 필드가 NMS 경쟁에서 영토를 확보하지 못했습니다. "
                    "문서 유형이 스키마와 맞지 않을 수 있습니다."
                )
                result.log = self.logs
                return result

            if self.config.enable_spatial_residual:
                heatmaps = spatial_residual(heatmaps, grid, log=self.logs)
            result.heatmaps = heatmaps

            self._log("═══ STEP 3: Vision NMS & Cropping ═══")
            cgut = set()
            rgut = set()
            if self.config.enable_gutters:
                try:
                    cgut = column_gutters(image, grid.cols)
                    rgut = row_gutters(image, grid.rows)
                except Exception as e:
                    self._log(f"  ⚠ 거터 검출 실패({e}) → 거터 없이 진행합니다.")

            tboxes: List = []
            try:
                from .text_boxes import detect_text_boxes
                tboxes = detect_text_boxes(image, ocr=self.ocr, log=self.logs)
            except Exception as e:
                self._log(f"  ⏭ 텍스트 박스 검출 생략: {e}")

            schema_fields = (schema or {}).get("fields", {}) or {}
            table_cats = set()
            array_cats = set()
            identity_cat = ""
            for _fname, _fdef in schema_fields.items():
                if not isinstance(_fdef, dict):
                    continue
                _cat = str(_fdef.get("category") or "misc")
                if _fdef.get("array"):
                    array_cats.add(_cat)
                if _fdef.get("table"):
                    table_cats.add(_cat)
                if _fname == "doc_number" and not identity_cat:
                    identity_cat = _cat
            self._log(
                f"  🧾 표 통합 대상 {sorted(table_cats)} | "
                f"배열 추출 대상 {sorted(array_cats)}"
            )

            plans = plan_crops(
                heatmaps, grid,
                iou_threshold=self.config.iou_threshold,
                margin_threshold=self.config.margin_threshold,
                col_gutters=cgut,
                row_gutters=rgut,
                max_crop_cols=self.config.max_crop_cols,
                log=self.logs,
                legibility=legib,
                table_categories=table_cats,
                identity_category=identity_cat,
                title_rows=max(1, int(grid.rows * self.config.title_suppress_ratio)),
                text_boxes=tboxes,
            )
            result.plans = plans
            self._log(f"  확정 크롭 {len(plans)}개")

            if not plans:
                result.error = "확정된 크롭 영역이 없습니다."
                result.log = self.logs
                return result

            self._phase("generation")
            self._refresh_slots()

            self._log("═══ STEP 4: Crop + Upscale + OCR ═══")
            reader = self.line_read_fn if self.config.line_read else None
            if reader is not None and refine_fn is not None:
                self._log(
                    f"  🔁 [LINE READ] 활성 — 창당 {self.config.line_span}행 "
                    f"/ 보폭 {self.config.line_stride}행 "
                    f"/ 겹침 {self.config.line_overlap}행"
                )
            elif reader is not None:
                reader = None
                self._log(
                    "  ⏭ [LINE READ] 정제 LLM 이 없어 행 판독을 생략하고 "
                    "PP-OCRv5 인식 결과를 그대로 씁니다."
                )

            fields = extract_from_crops(
                image, plans, self.ocr,
                refine_fn=refine_fn,
                target_px=self.config.target_text_px,
                max_tiles=self.config.max_tiles,
                keep_crop=self.config.keep_crop,
                nlp=self.nlp,
                log=self.logs,
                array_categories=array_cats,
                text_boxes=tboxes,
                line_read_fn=reader,
                schema=schema,
            )
            result.fields = fields

            record = fields_to_record(fields, schema, log=self.logs)
            for name in result.absent_fields:
                record[name] = None
            if result.absent_fields:
                self._log(
                    f"  ⚪ 부재 판정 필드 {len(result.absent_fields)}개는 "
                    f"null 로 기록합니다: {', '.join(sorted(result.absent_fields))}"
                )

            if self.config.enable_grounding:
                self._log("═══ STEP 5: Value Grounding ═══")
                claims = []
                for f in fields:
                    if f.field_values:
                        for key, val in f.field_values.items():
                            if str(val).strip():
                                claims.append(
                                    GroundingClaim(f.category, key, str(val), f.bbox)
                                )
                    elif f.value.strip():
                        claims.append(
                            GroundingClaim(f.category, f.category, f.value, f.bbox)
                        )
                verdicts = verify_claims(
                    claims, grid, self.embed_fn, nlp=self.nlp, log=self.logs
                )
                result.verdicts = verdicts
                record = apply_verdicts(record, verdicts)

            result.record = record

            try:
                result.json = record_to_json(
                    record, schema,
                    source_path=self.config.source_path,
                    lang_code=self.config.lang_code,
                )
            except Exception as e:
                import traceback
                self._log(
                    f"  ⚠ 레코드 정형화 실패 ({type(e).__name__}: {e}) "
                    f"— 원시 record 로 대체합니다."
                )
                for line in traceback.format_exc().splitlines()[-8:]:
                    self._log(f"      {line}")
                result.json = {
                    "doc_type": str(
                        (schema or {}).get("code")
                        or (schema or {}).get("doc_type") or ""
                    ),
                    "domain": str((schema or {}).get("domain") or ""),
                    "fields": {
                        k: v for k, v in record.items()
                        if not k.startswith("__")
                        and v not in (None, "", [])
                    },
                    "ocr_by_category": record.get("__ocr_by_category__") or {},
                    "shape_error": f"{type(e).__name__}: {e}",
                }

            result.ok = True

            self._log("═══ 추출 결과 (JSON) ═══")
            try:
                import json as _json
                dumped = _json.dumps(
                    result.json, ensure_ascii=False, indent=2
                )
            except Exception as e:
                dumped = f"(직렬화 실패: {e})"
            for line in dumped.splitlines():
                self._log(f"  {line}")

            filled = sum(
                1 for k, v in record.items()
                if not k.startswith("__") and v not in (None, "", [])
            )
            total = len((schema or {}).get("fields", {}) or {})
            self._log(
                f"  📦 추출 요약 — 채워진 필드 {filled}/{total} "
                f"| 크롭 {len(plans)}개 | 접지 유지 "
                f"{sum(1 for v in result.verdicts if v.accepted)}건"
            )

        except Exception as e:
            import traceback
            result.error = str(e)
            self._log(f"❌ 비전 파이프라인 오류:\n{traceback.format_exc()}")
            if result.record and not result.json:
                result.json = {
                    "fields": {
                        k: v for k, v in result.record.items()
                        if not k.startswith("__")
                        and v not in (None, "", [])
                    },
                    "ocr_by_category": result.record.get(
                        "__ocr_by_category__"
                    ) or {},
                    "pipeline_error": f"{type(e).__name__}: {e}",
                }
                self._log(
                    "  🩹 오류 발생 전까지 확보한 추출값을 JSON 으로 "
                    "보존했습니다."
                )
        finally:
            self._phase("idle")

        result.elapsed = time.time() - started
        result.log = list(self.logs)
        return result


def run_vision_pipeline(
    image: Image.Image,
    schema: dict,
    embed_fn: Callable[[List[str]], np.ndarray],
    ocr=None,
    embedder=None,
    config: Optional[VisionPipelineConfig] = None,
    log: Optional[Callable[[str], None]] = None,
    refine_fn: Optional[Callable[[str, str, Image.Image], dict]] = None,
    nlp=None,
    crossover=None,
    line_read_fn: Optional[Callable] = None,
) -> VisionPipelineResult:
    return VisionPipeline(
        embed_fn, ocr=ocr, embedder=embedder, config=config,
        log=log, nlp=nlp, crossover=crossover, line_read_fn=line_read_fn,
    ).run(image, schema, refine_fn=refine_fn)