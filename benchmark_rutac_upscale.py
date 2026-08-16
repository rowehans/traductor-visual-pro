"""A/B del upscale de la Ruta C: 3.5× (baseline) vs 2×, con ruido controlado.

CORRECCIÓN 2026-08-15 (dos bugs del A/B original):
1. **Anidamiento de wrappers**: el benchmark viejo re-envolvía
   _recover_regions_with_easyocr en cada pasada — la pasada "2×" envolvía a
   la "3.5×" y FORZABA 3.5 otra vez → comparaba 3.5 vs 3.5 (el "−13-24 %"
   documentado era pura deriva de orden). Aquí el original se captura UNA
   sola vez (tras los wraps de timing) y cada parcheo crea un wrapper fresco
   alrededor de ese original: cada pasada corre el upscale que anuncia.
2. **Sesgo de orden**: las dos pasadas separadas confundían deriva de GPU con
   efecto del parámetro. Ahora usa el harness anti-deriva de
   benchmark_ab_utils.py: intercalado por página, orden alternado, páginas de
   control (sin Ruta C) como noise-floor y veredicto explícito. Con el daemon
   VLM detenido el noise-floor baja a ~0.02 s.

Uso: PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_rutac_upscale.py
     --pages 1,4,7,2,5 --reps 3
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, cast

import fitz
import numpy as np

import benchmark_ab_utils as abu
import ocr_utils
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
SCALE = 1.2  # resolución real de producción (pdf.js default)
BASE_UPSCALE = 3.5  # valor de producción ACTUAL (revertido 2026-08-15)
ALT_UPSCALE = 2.0  # el candidato descartado (pierde 2 bloques a tiempo neutro)

# Páginas con Ruta C + normales de control (sin Ruta C → noise-floor)
DEFAULT_PAGES = "1,4,7,8,11,2,5"

TIMING = abu.Timing()
# Original capturado TRAS _wrap_timing (en main): el wrapper de upscale lo
# llama para que el timing de "ruta_c" siga registrándose. Se reasigna por
# cada parcheo SIN anidar (siempre alrededor del mismo original).
_ORIG_RECOVER: Any = None


def _wrap_timing(name: str, key: str) -> None:
    fn = getattr(ocr_utils, name, None)
    if fn is None:
        return

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.time()
        try:
            return fn(*args, **kwargs)
        finally:
            TIMING.timings[key] = TIMING.timings.get(key, 0.0) + (time.time() - t0)
            TIMING.counts[key] = TIMING.counts.get(key, 0) + 1

    wrapper.__name__ = f"_wrapped_{name}"
    setattr(ocr_utils, name, wrapper)


def _apply_upscale(which: str) -> None:
    """Fuerza el upscale de la Ruta C sin tocar producción (sin anidar)."""
    target = BASE_UPSCALE if which == "base" else ALT_UPSCALE
    orig = _ORIG_RECOVER

    def wrapper(img_bgr: Any, regions: list[dict[str, Any]],
                lang_hint: str = "es", upscale: float = 2.0) -> list[dict[str, Any]]:
        return cast(list[dict[str, Any]],
                    orig(img_bgr, regions, lang_hint, upscale=target))

    wrapper.__name__ = f"_rutac_upscale_{target}"
    ocr_utils._recover_regions_with_easyocr = wrapper


def render_page(doc: Any, page_no: int) -> Any:
    page = doc.load_page(page_no - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[:, :, ::-1].copy()


def main() -> None:
    global _ORIG_RECOVER
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=DEFAULT_PAGES)
    ap.add_argument("--reps", type=int, default=1,
                    help="Pares base/alt por página (mediana); default 1 — "
                         "con el daemon VLM detenido el ruido es ~0.02 s")
    ap.add_argument("--json", default="benchmark_results/rutac_upscale_ab.json")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps debe ser >= 1")

    pages = [int(x) for x in args.pages.split(",")]

    doc = fitz.open(PDF)
    ocr = OCRManager()
    _wrap_timing("_pre_filter_image", "prefilter")
    _wrap_timing("_run_rapidocr", "rapid")
    _wrap_timing("_recover_regions_with_easyocr", "ruta_c")
    _ORIG_RECOVER = ocr_utils._recover_regions_with_easyocr  # versión con timing

    def run_one(pno: int) -> dict[str, Any]:
        TIMING.reset()
        img_bgr = render_page(doc, pno)
        t0 = time.time()
        blocks, engine_used, _ = ocr.run_ocr(img_bgr, OCR_LANG, "fusion", prefilter=True)
        dt = time.time() - t0
        avg_conf = (float(np.mean([b.get("confidence", 0) for b in blocks]))
                    if blocks else 0.0)
        return {
            "total_s": dt,
            "n_blocks": len(blocks),
            "avg_conf": round(avg_conf, 3),
            "engine": engine_used,
            "stages": {k: round(v, 3) for k, v in TIMING.timings.items()},
            "calls": dict(TIMING.counts),
        }

    print(f"== A/B upscale Ruta C: {BASE_UPSCALE}x (base) vs "
          f"{ALT_UPSCALE}x (alt) ==  páginas={pages}  escala pdf.js={SCALE}  "
          f"reps={args.reps}")
    print("(warmup de ambos caminos en la primera página — carga de modelos)")

    # Warmup de AMBOS valores (carga de modelos fuera de las corridas medidas)
    _apply_upscale("base")
    run_one(pages[0])
    _apply_upscale("alt")
    run_one(pages[0])
    _apply_upscale("base")

    ab = abu.run_ab(pages, _apply_upscale, run_one, reps=args.reps)

    controls, drift_mean, drift_max = abu.control_stats(pages, ab)
    affected, effect_avg = abu.effect_stats(pages, ab)
    n = len(pages)
    avg_b = sum(ab[str(p)]["base"]["total_s"] for p in pages) / n
    avg_a = sum(ab[str(p)]["alt"]["total_s"] for p in pages) / n
    v = abu.verdict(controls, drift_mean, drift_max, effect_avg, avg_b)

    print(f"\n{'pág':>4} | {'total 3.5x':>10} {'total 2x':>9} {'Δtotal':>7} | "
          f"{'blk 3.5x':>8} {'blk 2x':>7} {'Δblk':>5} | {'conf 3.5x':>9} "
          f"{'conf 2x':>8} | ruta_c?")
    print("-" * 88)
    blk_b = blk_a = 0
    for p in pages:
        b, a = ab[str(p)]["base"], ab[str(p)]["alt"]
        dt = a["total_s"] - b["total_s"]
        db = a["n_blocks"] - b["n_blocks"]
        blk_b += b["n_blocks"]
        blk_a += a["n_blocks"]
        rutac = "SÍ" if b["stages"].get("ruta_c", 0) > 0 else "no"
        print(f"{p:>4} | {b['total_s']:9.2f}s {a['total_s']:8.2f}s {dt:+6.2f}s | "
              f"{b['n_blocks']:>8} {a['n_blocks']:>7} {db:+5d} | "
              f"{b['avg_conf']:9.2f} {a['avg_conf']:8.2f} | {rutac}")
    print("-" * 88)
    for line in abu.summary_block(controls, drift_mean, drift_max,
                                  affected, effect_avg, v):
        print(line)
    print("-" * 88)
    print(f"Promedio (todas): base={avg_b:.2f}s  alt={avg_a:.2f}s  "
          f"({avg_a - avg_b:+.2f}s/pág, {(avg_a / avg_b - 1) * 100:+.1f}%)")
    print(f"Bloques: base={blk_b}  alt={blk_a}  Δ={blk_a - blk_b:+d}")

    doc.close()
    result = {
        "benchmark": "benchmark_rutac_upscale.py",
        "pdf": PDF, "scale_pdfjs": SCALE,
        "base_upscale": BASE_UPSCALE, "alt_upscale": ALT_UPSCALE,
        "reps": args.reps, "pages": pages,
        "control_pages": controls,
        "control_drift_s": round(drift_mean, 3),
        "control_drift_max_s": round(drift_max, 3),
        "effect_affected_s": round(effect_avg, 3),
        "verdict": v,
        "avg_base_s": round(avg_b, 3), "avg_alt_s": round(avg_a, 3),
        "delta_avg_s": round(avg_a - avg_b, 3),
        "pct_change": round((avg_a / avg_b - 1) * 100, 1),
        "blocks_base": blk_b, "blocks_alt": blk_a,
        "per_page": ab,
    }
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False),
                               encoding="utf-8")
    print(f"\nResultado: {args.json}")


if __name__ == "__main__":
    main()
