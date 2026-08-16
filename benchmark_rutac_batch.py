"""A/B estructural de la Ruta C (2026-08-15): re-OCR por crop vs strip.

Hipótesis medida: el det DBNet por crop domina el costo del re-OCR (5 crops
sueltos = 1.33 s de det vs 1 strip 400x1200 = 0.45 s, 2.9x; rec de 5 líneas en
una llamada = 569 ms vs 5 llamadas = 700 ms). El strip amortiza el overhead
fijo del det; el rec batch elimina N-1 llamadas.

Desde la integración en producción (2026-08-15), el strip ya NO es un
prototipo: `_recover_regions_with_easyocr` lo ejecuta por defecto
(_RUTA_C_STRIP_BATCH=True). Este benchmark hace el A/B mismo-proceso usando el
propio interruptor de producción:

  baseline = producción con _RUTA_C_STRIP_BATCH=False  (re-OCR por crop)
  alt      = producción con _RUTA_C_STRIP_BATCH=True   (strip: det por chunk +
             UNA text_rec para todos los crops)

Ambos caminos reciben EXACTAMENTE los mismos crops (regiones de
_detect_text_regions_in_page, YOLO real) y conservan el fallback EasyOCR por
crop. El harness solo cronometra los sub-componentes del engine (det/rec/cls/
crop_list) y el group/spellcheck/glosario, sin tocar la lógica.

Usa el harness compartido benchmark_ab_utils.py (2026-08-15): intercalado por
página con orden alternado (par b→a, impar a→b), --reps con mediana, y
veredicto contra páginas de control (sin etapa Ruta C) — el mismo código de
intercalado/veredicto que benchmark_rutac_params.py / _upscale / _recovery.

Uso: PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_rutac_batch.py
     --pages 4,29 --json benchmark_results/rutac_batch_ab.json [--reps 2]
     [--no-spellcheck]
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import fitz
import numpy as np

import benchmark_ab_utils as abu
import ocr_utils

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
SCALE = 1.2  # resolución real de producción (pdf.js default)
UPSCALE = 3.5  # upscale actual de la Ruta C (revertido a 3.5× el 2026-08-15: el A/B del 2× estaba roto — ver benchmark_rutac_upscale.py)

# Páginas con Ruta C SIN VLM (las pesadas del A/B del pad + canónicas)
DEFAULT_PAGES = "4,29,39,7,11,52,8,1"

TIMING = abu.Timing()

_ORIG_GROUP = ocr_utils._group_and_merge_blocks


def _timed_group(blocks: list[dict[str, Any]],
                 img_h: int | None = None) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    try:
        return _ORIG_GROUP(blocks, img_h)
    finally:
        TIMING.timings["group"] = TIMING.timings.get("group", 0.0) + \
            (time.perf_counter() - t0)
        TIMING.counts["group"] = TIMING.counts.get("group", 0) + 1


ocr_utils._group_and_merge_blocks = _timed_group


def _time_module_fn(module: Any, name: str, key: str) -> None:
    orig = getattr(module, name, None)
    if orig is None:
        return

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return orig(*args, **kwargs)
        finally:
            TIMING.timings[key] = TIMING.timings.get(key, 0.0) + \
                (time.perf_counter() - t0)
            TIMING.counts[key] = TIMING.counts.get(key, 0) + 1

    wrapper.__name__ = f"_wrapped_{name}"
    setattr(module, name, wrapper)


def _identity_spellcheck(text: str) -> str:
    return text


_time_module_fn(ocr_utils, "_ocr_spellcheck", "spellcheck")
_time_module_fn(ocr_utils, "_aplicar_glosario", "glosario")


class _TimedComponent:
    """Proxy que mide tiempo y reenvía atributos al componente original.

    Reemplaza engine.text_det/text_rec/text_cls por el proxy: las llamadas se
    cronometran, y cualquier acceso a atributos (postprocess_op, session, ...)
    se reenvía al objeto original — el engine __call__ muta
    postprocess_op.box_thresh/unclip_ratio a través de este proxy y el cambio
    llega al objeto real (es el mismo objeto).
    """

    def __init__(self, key: str, fn: Any) -> None:
        self._key = key
        self._fn = fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            return self._fn(*args, **kwargs)
        finally:
            TIMING.timings[self._key] = TIMING.timings.get(self._key, 0.0) + \
                (time.perf_counter() - t0)
            TIMING.counts[self._key] = TIMING.counts.get(self._key, 0) + 1

    def __getattr__(self, item: str) -> Any:
        return getattr(self._fn, item)


def _wrap_engine(engine: Any) -> None:
    """Envuelve det/rec/cls del engine para acumular tiempo por componente."""
    for key, attr in (("det", "text_det"), ("rec", "text_rec"),
                      ("cls", "text_cls"), ("crop_list", "get_crop_img_list")):
        fn = getattr(engine, attr, None)
        if fn is None:
            continue
        setattr(engine, attr, _TimedComponent(key, fn))


def render_page(doc: Any, page_no: int) -> np.ndarray:
    page = doc.load_page(page_no - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    return img[:, :, ::-1].copy()


def run_production(img_bgr: np.ndarray,
                   regions: list[dict[str, Any]]) -> dict[str, Any]:
    """Corre la Ruta C de PRODUCCIÓN con el toggle ya aplicado por el harness.

    ``_RUTA_C_STRIP_BATCH`` lo fija apply_patch del harness antes de cada
    corrida (False → re-OCR por crop, True → batch estructural). Ambos caminos
    conservan el fallback EasyOCR por crop y el merge final — solo difiere el
    motor rapid.
    """
    TIMING.reset()
    t0 = time.perf_counter()
    blocks = ocr_utils._recover_regions_with_easyocr(
        img_bgr, regions, OCR_LANG, upscale=UPSCALE)
    total_s = time.perf_counter() - t0
    timing = dict(TIMING.timings)
    calls = dict(TIMING.counts)
    rapid_only = [b for b in blocks if b.get("engine") == "rapidocr-region"]
    return {
        "total_s": total_s,
        "timing": timing,
        "calls": calls,
        "blocks_total": len(blocks),
        "blocks_rapid": len(rapid_only),
        "texts": sorted(str(b.get("text", "")).strip()
                        for b in blocks if str(b.get("text", "")).strip()),
        "texts_rapid": sorted(str(b.get("text", "")).strip()
                              for b in rapid_only
                              if str(b.get("text", "")).strip()),
        # stages: el harness usa la clave "ruta_c" para detectar páginas
        # afectadas (control = sin etapa Ruta C).
        "stages": {"ruta_c": total_s if regions else 0.0},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", default=DEFAULT_PAGES)
    ap.add_argument("--json", default="benchmark_results/rutac_batch_ab.json")
    ap.add_argument("--reps", type=int, default=1,
                    help="Pares base/alt por página (mediana); default 1")
    ap.add_argument("--no-spellcheck", action="store_true",
                    help="Desactiva el spellcheck post-OCR en AMBOS caminos para "
                         "aislar el costo estructural del strip (el spellcheck "
                         "es un costo de recuperación, no del det/rec)")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps debe ser >= 1")
    if args.no_spellcheck:
        ocr_utils._ocr_spellcheck = _identity_spellcheck
    pages = [int(x) for x in args.pages.split(",")]

    engine = ocr_utils._get_rapid_engine()
    if engine is None:
        print("No hay engine RapidOCR")
        return
    _wrap_engine(engine)

    print(f"== A/B estructural Ruta C (producción, mismo-proceso) ==  "
          f"páginas={pages}  escala={SCALE}  upscale={UPSCALE}  reps={args.reps}")
    print("  baseline = per-crop (_RUTA_C_STRIP_BATCH=False) | "
          "alt = strip (_RUTA_C_STRIP_BATCH=True)")
    print("(corrida 1 de cada camino = warmup de modelos, se descarta "
          "con --reps > 1)")

    doc = fitz.open(PDF)
    # Inputs pre-computados UNA vez por página (render + YOLO + crops):
    # ambos caminos reciben exactamente los mismos inputs — la ventana de
    # deriva entre base y alt queda mínima (nada de OCR dentro del par).
    page_inputs: dict[int, tuple[np.ndarray, list[dict[str, Any]], int]] = {}
    for pno in pages:
        img_bgr = render_page(doc, pno)
        regions = ocr_utils._detect_text_regions_in_page(img_bgr)
        crops = sum(
            1 for c in ocr_utils._ruta_c_prepare_crops(img_bgr, regions, UPSCALE)
            if c is not None)
        page_inputs[pno] = (img_bgr, regions, crops)

    def apply_patch(which: str) -> None:
        ocr_utils._RUTA_C_STRIP_BATCH = (which == "alt")

    def run_one(pno: int) -> dict[str, Any]:
        img_bgr, regions, _ = page_inputs[pno]
        return run_production(img_bgr, regions)

    # warmup inicial de ambos caminos (carga de modelos/ONNX, se descarta)
    apply_patch("base")
    run_one(pages[0])
    apply_patch("alt")
    run_one(pages[0])
    apply_patch("base")

    ab = abu.run_ab(pages, apply_patch, run_one, reps=args.reps)

    results: dict[str, Any] = {}
    for pno in pages:
        base, alt = ab[str(pno)]["base"], ab[str(pno)]["alt"]
        _img, _regions, crops = page_inputs[pno]
        n_regions = len(_regions)

        set_b, set_a = set(base["texts"]), set(alt["texts"])
        texts_eq = set_b == set_a
        alt_dups = {t: c for t, c in Counter(alt["texts"]).items() if c > 1}

        def fmt(r: dict[str, Any]) -> str:
            return (f"{r['total_s']:.2f}s  "
                    f"({r['timing'].get('det', 0):.2f}s det, "
                    f"{r['timing'].get('rec', 0):.2f}s rec, "
                    f"{r['timing'].get('cls', 0):.2f}s cls; "
                    f"{r['calls'].get('det', 0)} det-calls, "
                    f"{r['calls'].get('rec', 0)} rec-calls)  "
                    f"{r['blocks_rapid']} bloques rapid")

        print(f"\npág {pno}: {crops} crops, {n_regions} regiones")
        print(f"  per-crop: {fmt(base)}")
        print(f"  strip   : {fmt(alt)}")
        print(f"  Δtotal {alt['total_s'] - base['total_s']:+.2f}s | textos "
              f"{'IDÉNTICOS' if texts_eq else 'DIFEREN'} "
              f"(soloB={len(set_b - set_a)}, soloA={len(set_a - set_b)}) "
              f"dups_strip={dict(alt_dups)}")

        results[str(pno)] = {
            "crops": crops,
            "regions": n_regions,
            "base_s": round(base["total_s"], 3),
            "alt_s": round(alt["total_s"], 3),
            "delta_s": round(alt["total_s"] - base["total_s"], 3),
            "base_det_s": round(base["timing"].get("det", 0), 3),
            "alt_det_s": round(alt["timing"].get("det", 0), 3),
            "base_rec_s": round(base["timing"].get("rec", 0), 3),
            "alt_rec_s": round(alt["timing"].get("rec", 0), 3),
            "base_group_s": round(base["timing"].get("group", 0), 3),
            "alt_group_s": round(alt["timing"].get("group", 0), 3),
            "base_spell_s": round(base["timing"].get("spellcheck", 0), 3),
            "alt_spell_s": round(alt["timing"].get("spellcheck", 0), 3),
            "base_det_calls": base["calls"].get("det", 0),
            "alt_det_calls": alt["calls"].get("det", 0),
            "base_blocks_total": base["blocks_total"],
            "base_blocks_rapid": base["blocks_rapid"],
            "alt_blocks": alt["blocks_total"],
            "texts_equal": texts_eq,
            "solo_base": sorted(set_b - set_a),
            "solo_alt": sorted(set_a - set_b),
            "base_texts": base["texts"],
            "alt_texts": alt["texts"],
            "base_texts_rapid": base["texts_rapid"],
            "alt_dups": alt_dups,
        }

    # ── Veredicto del harness compartido (controles = sin Ruta C) ──
    controls, drift_mean, drift_max = abu.control_stats(pages, ab)
    affected, effect_avg = abu.effect_stats(pages, ab)
    avg_base = float(np.mean([ab[str(p)]["base"]["total_s"] for p in pages]))
    verdict_str = abu.verdict(controls, drift_mean, drift_max,
                              effect_avg, avg_base)
    print("\n" + "-" * 104)
    for line in abu.summary_block(controls, drift_mean, drift_max,
                                  affected, effect_avg, verdict_str):
        print(line)
    results["_verdict"] = {
        "controls": controls,
        "control_drift_s": round(drift_mean, 3),
        "control_drift_max_s": round(drift_max, 3),
        "effect_affected_s": round(effect_avg, 3),
        "verdict": verdict_str,
    }

    doc.close()
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(
        json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado: {args.json}")


if __name__ == "__main__":
    main()
