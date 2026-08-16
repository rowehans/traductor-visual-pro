"""A/B de recuperación de la Ruta C: bloques por crop a 3.5× vs 2×.

Mide lo que el A/B de totales puede enmascarar: cuántos bloques recupera la
Ruta C por crop (vía YOLO/CTD) y con qué confianza, texto a texto.

CORRECCIÓN 2026-08-15: el benchmark viejo NO forzaba el upscale en el wrapper
(pasaba el del caller, 2.0 de producción) → ambas pasadas corrían 2.0 y el
"34 = 34 bloques" era 2.0 vs 2.0, no validaba 3.5 vs 2. Ahora el wrapper
fuerza el valor y usa el harness anti-deriva de benchmark_ab_utils.py
(intercalado por página, orden alternado, páginas de control como
noise-floor y veredicto). Con el daemon VLM detenido el ruido baja a ~0.02 s.

Uso: PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_rutac_recovery.py
     --pages 1,4,7,8,11,2,5 --reps 3
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
SCALE = 1.2
BASE_UPSCALE = 3.5  # valor histórico (baseline)
ALT_UPSCALE = 2.0  # valor de producción actual
DEFAULT_PAGES = "1,4,7,8,11,2,5"

TIMING = abu.Timing()
_ORIG_RECOVER: Any = None
# Capturas del run actual (reset por run_one)
_CAPTURE: list[tuple[str, float]] = []


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
    """Fuerza el upscale Y captura los bloques internos de la Ruta C.

    Captura el original UNA vez (tras los wraps de timing) y crea un wrapper
    fresco por parcheo: sin anidar, cada pasada corre el upscale que anuncia.
    """
    target = BASE_UPSCALE if which == "base" else ALT_UPSCALE
    orig = _ORIG_RECOVER

    def wrapper(img_bgr: Any, regions: list[dict[str, Any]],
                lang_hint: str = "es", upscale: float = 2.0) -> list[dict[str, Any]]:
        blocks = cast(list[dict[str, Any]],
                      orig(img_bgr, regions, lang_hint, upscale=target))
        for b in blocks:
            _CAPTURE.append((str(b.get("text", "")), float(b.get("confidence", 0))))
        return blocks

    wrapper.__name__ = f"_rutac_capture_{target}"
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
    ap.add_argument("--json", default="benchmark_results/rutac_recovery_ab.json")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps debe ser >= 1")

    pages = [int(x) for x in args.pages.split(",")]

    doc = fitz.open(PDF)
    ocr = OCRManager()
    _wrap_timing("_pre_filter_image", "prefilter")
    _wrap_timing("_run_rapidocr", "rapid")
    _wrap_timing("_recover_regions_with_easyocr", "ruta_c")
    _ORIG_RECOVER = ocr_utils._recover_regions_with_easyocr

    def run_one(pno: int) -> dict[str, Any]:
        TIMING.reset()
        _CAPTURE.clear()
        img_bgr = render_page(doc, pno)
        t0 = time.time()
        blocks, engine_used, _ = ocr.run_ocr(img_bgr, OCR_LANG, "fusion", prefilter=True)
        dt = time.time() - t0
        confs = [c for _t, c in _CAPTURE]
        return {
            "total_s": dt,
            "n_blocks_final": len(blocks),
            "n_recuperados": len(_CAPTURE),
            "conf_promedio": (round(float(np.mean(confs)), 3) if confs else 0.0),
            "textos": [t for t, _c in _CAPTURE],
            "engine": engine_used,
            "stages": {k: round(v, 3) for k, v in TIMING.timings.items()},
            "calls": dict(TIMING.counts),
        }

    print(f"== A/B recuperación Ruta C: {BASE_UPSCALE}x vs {ALT_UPSCALE}x ==  "
          f"páginas={pages}  escala pdf.js={SCALE}  reps={args.reps}")
    print("(warmup de ambos caminos en la primera página)")

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

    print(f"\n{'pág':>4} | {'rec 3.5x':>8} {'rec 2x':>6} {'Δrec':>5} | "
          f"{'conf 3.5x':>9} {'conf 2x':>8} | coinciden textos")
    print("-" * 84)
    total_b = total_a = 0
    for p in pages:
        bp, ap = ab[str(p)]["base"], ab[str(p)]["alt"]
        n_b, n_a = bp["n_recuperados"], ap["n_recuperados"]
        total_b += n_b
        total_a += n_a
        set_b, set_a = set(bp["textos"]), set(ap["textos"])
        only_b = set_b - set_a
        only_a = set_a - set_b
        match = ("SÍ" if n_b == n_a and not only_b and not only_a
                 else f"no (solo3.5={len(only_b)}, solo2x={len(only_a)})")
        print(f"{p:>4} | {n_b:>8} {n_a:>6} {n_a - n_b:+5d} | "
              f"{bp['conf_promedio']:9.2f} {ap['conf_promedio']:8.2f} | {match}")
        if only_b:
            print(f"      solo 3.5x: {sorted(only_b)[:3]}")
        if only_a:
            print(f"      solo 2x:   {sorted(only_a)[:3]}")
    print("-" * 84)
    for line in abu.summary_block(controls, drift_mean, drift_max,
                                  affected, effect_avg, v):
        print(line)
    print("-" * 84)
    print(f"Recuperados totales: 3.5x={total_b}  2x={total_a}  Δ={total_a - total_b:+d}")

    doc.close()
    result = {
        "benchmark": "benchmark_rutac_recovery.py",
        "pdf": PDF, "scale_pdfjs": SCALE,
        "base_upscale": BASE_UPSCALE, "alt_upscale": ALT_UPSCALE,
        "reps": args.reps, "pages": pages,
        "control_pages": controls,
        "control_drift_s": round(drift_mean, 3),
        "control_drift_max_s": round(drift_max, 3),
        "effect_affected_s": round(effect_avg, 3),
        "verdict": v,
        "avg_base_s": round(avg_b, 3), "avg_alt_s": round(avg_a, 3),
        "total_recuperados_3.5": total_b,
        "total_recuperados_2": total_a,
        "delta": total_a - total_b,
        "per_page": ab,
    }
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False),
                               encoding="utf-8")
    print(f"\nResultado: {args.json}")


if __name__ == "__main__":
    main()
