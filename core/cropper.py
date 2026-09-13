from PIL import Image
from typing import List, Tuple, Dict
import os


class RegionCropper:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir

    def crop_region(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(image.width, int(x1))
        y1 = min(image.height, int(y1))
        if x1 <= x0 or y1 <= y0:
            return Image.new("RGB", (1, 1))
        return image.crop((x0, y0, x1, y1))

    def crop_all(
        self,
        image: Image.Image,
        candidates: list,
        ocr_fn=None,
    ) -> List[Dict]:
        results = []
        for c in candidates:
            if c.status != "confirmed":
                continue
            crop_img = self.crop_region(image, c.bbox)

            ocr_text = ""
            if ocr_fn is not None:
                try:
                    ocr_text = ocr_fn(image, c.bbox)
                except Exception:
                    ocr_text = ""

            results.append({
                "field_name": c.field_name,
                "bbox": list(c.bbox),
                "score": c.score,
                "ocr_text": ocr_text,
                "crop_image": crop_img,
            })

        return results

    def save_crops(
        self,
        image: Image.Image,
        crop_results: List[Dict],
        output_dir: str = None,
    ) -> List[str]:
        out_dir = output_dir or self.output_dir
        os.makedirs(out_dir, exist_ok=True)

        saved_paths = []
        for result in crop_results:
            field_name = result["field_name"]
            crop_img = result["crop_image"]
            filename = f"{field_name}.png"
            filepath = os.path.join(out_dir, filename)
            crop_img.save(filepath)
            saved_paths.append(filepath)

        return saved_paths

    def export_json(
        self,
        crop_results: List[Dict],
        output_path: str,
    ) -> None:
        import json
        export_data = []
        for r in crop_results:
            export_data.append({
                "field_name": r["field_name"],
                "bbox": r["bbox"],
                "score": r["score"],
                "ocr_text": r["ocr_text"],
            })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)