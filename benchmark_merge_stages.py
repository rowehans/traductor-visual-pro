"""Desglose fino del tiempo residual sin VLM (páginas 4-6 s).

Mide, además de las etapas de benchmark_production, el spellcheck del merge
(`_ocr_spellcheck`, llamado desde `_group_and_merge_blocks`) y el merge
completo (`_group_and_merge_blocks`), que el benchmark principal no
instrumenta. Uso: python benchmark_merge_stages.py --pages 4,43,29
"""
import argparse
import json
import time
from typing import Any

import fitz
import numpy as np

import ocr_utils
from benchmark_production import _get_orig, render_page
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"


def _time_fn(name: str, timings: dict[str, float], counts: dict[str, int]) -> None:
    fn = _get_orig(ocr_utils, name)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            timings[name] = timings.get(name, 0.0) + (time.time() - t0)
            counts[name] = counts.get(name, 0) + 1

    wrapper.__name__ = f"_wrapped_{name}"
    setattr(ocr_utils, name, wrapper)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="4,43,29")
    args = ap.parse_args()
    pages = [int(p) for p in args.pages.split(",")]

    doc = fitz.open(PDF)
    ocr = OCRManager()

    per_page: dict[str, dict[str, Any]] = {}
    for pno in pages:
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        for fn, key in [
            ("_pre_filter_image", "prefilter"),
            ("_run_rapidocr", "rapid"),
            ("_recover_regions_with_easyocr", "ruta_c"),
            ("_ocr_spellcheck", "spellcheck"),
            ("_group_and_merge_blocks", "merge"),
        ]:
            _time_fn(fn, timings, counts)

        img_bgr = render_page(doc, pno, 1.2)
        t0 = time.time()
        blocks, engine_used, _ = ocr.run_ocr(img_bgr, OCR_LANG, "fusion", prefilter=True)
        dt = time.time() - t0

        per_page[str(pno)] = {
            "total_s": round(dt, 3),
            "n_blocks": len(blocks),
            "engine": engine_used,
            "stages": {k: round(v, 3) for k, v in timings.items()},
            "calls": dict(counts),
        }
        print(f"pág {pno}: total={dt:.2f}s  bloques={len(blocks)}")
        for k, v in timings.items():
            print(f"    {k:10s} {v:6.2f}s  ({100*v/dt:4.1f}%)  calls={counts[k]}")

    doc.close()
    out = {"benchmark": "benchmark_merge_stages.py", "pages": pages, "per_page": per_page}
    path = "benchmark_results/merge_stages.json"
    import pathlib
    pathlib.Path(path).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {path}")


if __name__ == "__main__":
    main()
