"""A/B de `max_length` del daemon VLM (Fase 3.3): 2048 vs 512/1024.

Corre el pipeline REAL (fusion, scale 1.2) sobre las páginas del trigger v4.2
con `uocr_client.UOCR_MAX_LENGTH` parcheado, y captura por página: tiempo del
VLM, bloques recuperados por el VLM, bloques finales y textos. Cada variante
se corre en un PROCESO SEPARADO (caches de trigger/negativas frescos).

Reusa la instrumentación de benchmark_production.py (wrappers de timing,
trigger y recuperación del VLM).

Uso: PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_vlm_maxlen.py \
       --pages 21,16,28,36 --max_length 512 --json benchmark_results/vlm_512.json
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

import fitz

import benchmark_production as bp
import uocr_client
from ocr_engine import OCRManager


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default="21,16,28,36")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--ngram", type=int, default=None,
                    help="no_repeat_ngram_size del daemon (plan §10.2 item 2)")
    ap.add_argument("--prompt", default=None,
                    help="prompt del VLM (plan §10.2 item 2)")
    ap.add_argument("--image_size", type=int, default=None,
                    help="image_size del prefill (plan §10.2 item 5)")
    ap.add_argument("--json", default="benchmark_results/vlm_maxlen.json")
    args = ap.parse_args()

    pages = [int(x) for x in args.pages.split(",")]

    # Parchear ANTES de cualquier llamada: process_page capturó UOCR_MAX_LENGTH
    # como default en la definición — setear el atributo no surte efecto. Se
    # envuelve process_page para forzar max_length/ngram/prompt explícitos.
    _orig_process_page = uocr_client.process_page

    def _forced_process(image_path: str, **kwargs: Any) -> dict[str, Any]:
        return _orig_process_page(
            image_path, max_length=args.max_length,
            ngram=args.ngram, prompt=args.prompt,
            image_size=args.image_size, **kwargs)

    uocr_client.process_page = _forced_process  # type: ignore[assignment]
    print(f"== max_length = {args.max_length} ngram = {args.ngram} "
          f"image_size = {args.image_size} prompt = {args.prompt!r} ==  "
          f"páginas={pages}")

    doc = fitz.open(bp.PDF)
    ocr = OCRManager()
    bp._wrap_dark_ratio()

    per_page: dict[str, dict[str, Any]] = {}
    total_t = 0.0
    total_vlm = 0.0
    total_rec = 0

    for pno in pages:
        timings: dict[str, float] = {}
        counts: dict[str, int] = {}
        results: dict[str, int] = {}
        bp.DARK_RATIOS.clear()
        trigger_holder: dict[str, Any] = {"triggered": False, "reason": None}
        bp._wrap_timing("_pre_filter_image", "prefilter", timings, counts)
        bp._wrap_timing("_run_rapidocr", "rapid", timings, counts)
        bp._wrap_timing("_recover_regions_with_easyocr", "ruta_c", timings, counts)
        bp._wrap_method(ocr, "_reforzar_con_unlimited", "vlm", timings, counts, results)
        bp._wrap_method(ocr, "_reforzar_con_rapid_agresivo", "rapid_aggr", timings, counts, results)
        bp._wrap_method(ocr, "_ruta_c_yolo", "yolo", timings, counts, results)
        bp._wrap_method(ocr, "_ruta_c_ctd", "ctd", timings, counts, results)
        bp._install_trigger_capture(ocr, trigger_holder)

        img_bgr = bp.render_page(doc, pno, bp.DEFAULT_SCALE)
        t0 = time.time()
        blocks, engine_used, _ = ocr.run_ocr(img_bgr, bp.OCR_LANG, "fusion", prefilter=True)
        dt = time.time() - t0
        total_t += dt
        avg_conf = (float(sum(b.get("confidence", 0) for b in blocks)) / len(blocks)
                    if blocks else 0.0)
        vlm_s = timings.get("vlm", 0.0)
        rec = results.get("vlm_recovered", 0)
        total_vlm += vlm_s
        total_rec += rec
        texts = sorted(str(b.get("text", "")).strip() for b in blocks
                       if str(b.get("text", "")).strip())
        per_page[str(pno)] = {
            "total_s": round(dt, 3),
            "vlm_s": round(vlm_s, 3),
            "vlm_recovered": rec,
            "n_blocks": len(blocks),
            "avg_conf": round(avg_conf, 3),
            "trigger": bool(trigger_holder["triggered"]),
            "trigger_reason": trigger_holder["reason"],
            "texts": texts,
            "stages": {k: round(v, 3) for k, v in timings.items()},
        }
        print(f"pág {pno:>2}: {dt:7.2f}s  vlm={vlm_s:7.2f}s  recuperado={rec}  "
              f"bloques={len(blocks):>2}  conf={avg_conf:.2f}  "
              f"trigger={trigger_holder['reason']}")

    doc.close()
    result = {
        "benchmark": "benchmark_vlm_maxlen.py",
        "max_length": args.max_length,
        "ngram": args.ngram,
        "image_size": args.image_size,
        "prompt": args.prompt,
        "pages": pages,
        "total_s": round(total_t, 3),
        "vlm_total_s": round(total_vlm, 3),
        "vlm_recovered_total": total_rec,
        "per_page": per_page,
    }
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False),
                               encoding="utf-8")
    print(f"\nmax_length={args.max_length}: VLM total {total_vlm:.1f}s en {len(pages)} págs "
          f"| recuperado {total_rec} bloques | resultado: {args.json}")


if __name__ == "__main__":
    main()
