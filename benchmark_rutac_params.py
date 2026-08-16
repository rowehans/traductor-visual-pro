"""A/B de parámetros del crop de la Ruta C: padding, interpolación, rotation_info, rapid.

Sigue el patrón de benchmark_rutac_upscale.py: por cada parámetro, corre el
pipeline REAL (fusion, scale 1.2) con el valor base y con la alternativa
(parcheando las constantes de módulo `_RUTA_C_*` / `config.EASYOCR_ROTATION_INFO`
sin tocar producción) y compara: bloques finales, confianza, textos recuperados
(set-equality por página) y tiempo.

Los parámetros viven en `_recover_regions_with_easyocr` (ocr_utils.py):
- pad: max(_RUTA_C_PAD_MIN, min(w,h) * _RUTA_C_PAD_FACTOR)   [base 0.03/6]
- interpolación del upscale: _RUTA_C_INTERP                    [base INTER_CUBIC]
- rotation_info: SOLO afecta al fallback EasyOCR (RapidOCR es primario,
  RUTA_C_RAPID_PRIMARY=True) — se parchea config.EASYOCR_ROTATION_INFO.
- rapid_box / rapid_unclip / rapid_batch: parámetros del pase rapid de la Ruta C.

CORRECCIÓN 2026-08-15 (deriva de orden): el A/B original corría TODAS las
páginas con el valor base y LUEGO TODAS con el alternativo — cualquier deriva
de estado de GPU/térmica entre las dos pasadas se confundía con el efecto del
parámetro (el "−21.5 %" del box_thresh era deriva: las páginas de control sin
Ruta C mostraban el mismo Δ que las afectadas). Ahora:
- **Intercalado por página**: base y alt se corren contiguos en la misma
  página (ventana de deriva mínima entre ambos).
- **Orden alternado**: la página par corre base→alt y la impar alt→base, de
  modo que el sesgo "la segunda corrida de un par es más rápida" se cancela
  entre páginas.
- **Páginas de control**: se detectan automáticamente las páginas sin etapa
  Ruta C en el baseline (donde el parámetro no aplica). Se reporta su Δ medio
  y su noise-floor (máx |Δ|); el Δ de las páginas AFECTADAS se compara contra
  ese noise-floor y se emite un veredicto explícito (atribuible / cautela /
  NO CONCLUYENTE). Esta máquina tiene ±0.3-0.7 s de ruido run-to-run (GPU
  compartida con el daemon VLM), así que sin controles el Δ bruto no es
  fiable.
- **--reps N**: mediana de N pares por página (default 2) para suavizar ruido
  térmico de GPU.
- El rec_batch_num del recognizer se muta IN-PLACE sobre el engine ya
  construido (la librería lo lee en __call__), eliminando el rebuild del engine
  que metía el coste de carga de ONNX en la corrida posterior al cambio.

Uso: PYTHONIOENCODING=utf-8 env/Scripts/python.exe benchmark_rutac_params.py
     [--pages 1,4,7] [--param pad] [--reps 2] [--json salida.json]
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any, Final

import cv2
import fitz
import numpy as np

import benchmark_ab_utils as abu
import ocr_utils
from ocr_engine import OCRManager

PDF = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OCR_LANG = "ja"
SCALE = 1.2  # resolución real de producción (pdf.js default)
# Upscale de la Ruta C en producción (revertido a 3.5× el 2026-08-15 — el A/B
# del 2× estaba roto y el 2× perdía 2 bloques en pág 11 a tiempo neutro, ver
# benchmark_rutac_upscale.py). Este benchmark corre el ENGINE real, cuyos
# callers de la Ruta C (_ruta_c_yolo/_ruta_c_ctd/_reforzar_…) fijan 3.5
# explícitamente; la constante es la referencia visible de que se mide con el
# mismo upscale que producción.
UPSCALE: Final[float] = 3.5

# Páginas donde la Ruta C disparó en la medición previa (1,4,7,8,11) + 2 normales
DEFAULT_PAGES = "1,4,7,8,11,2,5"

PARAMS: dict[str, dict[str, Any]] = {
    "pad": {
        "label": "pad factor del crop",
        "base": {"_RUTA_C_PAD_FACTOR": 0.03},
        "alt": {"_RUTA_C_PAD_FACTOR": 0.06},
        "desc": "0.03 (producción) → 0.06: crop más grande, más área a upscale/OCR",
        "rutac_only": True,
    },
    "interp": {
        "label": "interpolación del upscale",
        "base": {"_RUTA_C_INTERP": cv2.INTER_CUBIC},
        "alt": {"_RUTA_C_INTERP": cv2.INTER_LINEAR},
        "desc": "INTER_CUBIC → INTER_LINEAR (más rápido en CPU)",
        "rutac_only": True,
    },
    "rotation": {
        "label": "rotation_info (solo fallback EasyOCR)",
        "base": {"_EASYOCR_ROTATION_INFO": (0, 90, 180, 270)},
        "alt": {"_EASYOCR_ROTATION_INFO": (0, 180)},
        "desc": "(0,90,180,270) → (0,180): la mitad de rotaciones en fallback",
        "rutac_only": True,
    },
    "rapid_box": {
        "label": "box_thresh de RapidOCR en crops de la Ruta C",
        "base": {"_RUTA_C_RAPID_BOX_THRESH": None},
        "alt": {"_RUTA_C_RAPID_BOX_THRESH": 0.35},
        "desc": "0.5 (default) → 0.35: detecta cajas más débiles (más bloques, más lento)",
    },
    "rapid_unclip": {
        "label": "unclip_ratio de RapidOCR en crops de la Ruta C",
        "base": {"_RUTA_C_RAPID_UNCLIP_RATIO": None},
        "alt": {"_RUTA_C_RAPID_UNCLIP_RATIO": 2.2},
        "desc": "1.6 (default) → 2.2 (params agresivos): cajas más grandes tras la máscara",
    },
    "rapid_batch": {
        "label": "rec_batch_num del recognizer de RapidOCR",
        "base": {"_RAPID_REC_BATCH_NUM": 6},
        "alt": {"_RAPID_REC_BATCH_NUM": 16},
        "desc": "6 → 16 (los crops de la Ruta C tienen pocas líneas; sin cambio esperado)",
        "rec_batch_attr": True,
    },
}


def render_page(doc: Any, page_no: int) -> Any:
    page = doc.load_page(page_no - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img[:, :, ::-1].copy()


class Timing(abu.Timing):
    """Timing del harness + contador de crops que cayeron al fallback EasyOCR."""

    def __init__(self) -> None:
        super().__init__()
        self.easyocr_crops = 0  # crops al fallback EasyOCR (rotation_info)

    def reset(self) -> None:
        super().reset()
        self.easyocr_crops = 0


TIMING = Timing()


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


def _wrap_easyocr_count() -> None:
    """Cuenta las llamadas a _run_ocr_on_image que llevan rotation_info
    (es decir, los crops de la Ruta C que cayeron al fallback EasyOCR)."""
    fn = getattr(ocr_utils, "_run_ocr_on_image", None)
    if fn is None:
        return

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if "rotation_info" in kwargs:
            TIMING.easyocr_crops += 1
        return fn(*args, **kwargs)

    wrapper.__name__ = "_wrapped_run_ocr_on_image"
    setattr(ocr_utils, "_run_ocr_on_image", wrapper)


def _apply_engine_batch(value: int) -> None:
    """Aplica rec_batch_num al engine YA construido (sin rebuild).

    La librería lee self.rec_batch_num en __call__ (batch_num =
    self.rec_batch_num), así que mutar el atributo surte efecto en la siguiente
    llamada. Si el engine aún no está construido, el constructor usará la
    constante de módulo (_RAPID_REC_BATCH_NUM), que también se parchea.
    """
    engine = ocr_utils._get_rapid_engine()
    if engine is not None and getattr(engine, "text_rec", None) is not None:
        engine.text_rec.rec_batch_num = value


def _apply_patch(param: str, which: str) -> None:
    """Aplica (o restaura) los valores base/alt del parámetro."""
    values = PARAMS[param][which]
    for k, v in values.items():
        if k == "_EASYOCR_ROTATION_INFO":
            import config
            config.EASYOCR_ROTATION_INFO = v  # type: ignore[misc]  # Final en runtime
        else:
            setattr(ocr_utils, k, v)
    if PARAMS[param].get("rec_batch_attr"):
        _apply_engine_batch(int(values["_RAPID_REC_BATCH_NUM"]))


def _run_once(ocr: OCRManager, img_bgr: Any) -> dict[str, Any]:
    """Corre UNA página con los valores de parámetro YA aplicados."""
    TIMING.reset()
    t0 = time.time()
    blocks, engine_used, _ = ocr.run_ocr(img_bgr, OCR_LANG, "fusion", prefilter=True)
    dt = time.time() - t0
    avg_conf = (float(np.mean([b.get("confidence", 0) for b in blocks]))
                if blocks else 0.0)
    return {
        "total_s": dt,
        "n_blocks": len(blocks),
        "avg_conf": round(avg_conf, 3),
        "texts": sorted(str(b.get("text", "")).strip() for b in blocks
                       if str(b.get("text", "")).strip()),
        "stages": {k: round(v, 3) for k, v in TIMING.timings.items()},
        "easyocr_crops": TIMING.easyocr_crops,
    }


def compare(pages: list[int], ab: dict[str, Any], param: str,
            reps: int) -> dict[str, Any]:
    info = PARAMS[param]
    print(f"\n== {info['label']}: {info['desc']} ==  (reps={reps})")
    print(f"{'pág':>4} | {'total base':>10} {'total alt':>9} {'Δtotal':>7} | "
          f"{'blk base':>8} {'blk alt':>7} {'Δblk':>5} | conf igual | textos | ruta_c?")
    print("-" * 104)
    deltas_blocks: list[int] = []
    total_time_base = 0.0
    total_time_alt = 0.0
    for p in pages:
        b, a = ab[str(p)]["base"], ab[str(p)]["alt"]
        dt = a["total_s"] - b["total_s"]
        db = a["n_blocks"] - b["n_blocks"]
        deltas_blocks.append(db)
        conf_eq = abs(a["avg_conf"] - b["avg_conf"]) < 0.001
        set_b, set_a = set(b["texts"]), set(a["texts"])
        texts_eq = set_b == set_a
        match = "SÍ" if texts_eq else f"no (soloB={len(set_b - set_a)}, soloA={len(set_a - set_b)})"
        rutac = "SÍ" if b["stages"].get("ruta_c", 0) > 0 else "no"
        total_time_base += b["total_s"]
        total_time_alt += a["total_s"]
        print(f"{p:>4} | {b['total_s']:9.2f}s {a['total_s']:8.2f}s {dt:+6.2f}s | "
              f"{b['n_blocks']:>8} {a['n_blocks']:>7} {db:+5d} | "
              f"{'SÍ' if conf_eq else 'no':>5} | {match:>34} | {rutac}")

    n = len(pages)
    avg_b = total_time_base / n
    avg_a = total_time_alt / n
    blk_b = sum(ab[str(p)]["base"]["n_blocks"] for p in pages)
    blk_a = sum(ab[str(p)]["alt"]["n_blocks"] for p in pages)
    easy_b = sum(ab[str(p)]["base"]["easyocr_crops"] for p in pages)
    easy_a = sum(ab[str(p)]["alt"]["easyocr_crops"] for p in pages)

    # ── Deriva de control: Δ en páginas sin Ruta C (harness compartido) ──
    controls, drift_mean, drift_max = abu.control_stats(pages, ab)
    affected, effect_avg = abu.effect_stats(pages, ab)
    verdict = abu.verdict(controls, drift_mean, drift_max, effect_avg, avg_b)

    print("-" * 104)
    for line in abu.summary_block(controls, drift_mean, drift_max,
                                  affected, effect_avg, verdict):
        print(line)
    print("-" * 104)
    print(f"Promedio (todas): base={avg_b:.2f}s  alt={avg_a:.2f}s  "
          f"({avg_a - avg_b:+.2f}s/pág, {(avg_a / avg_b - 1) * 100:+.1f}%)")
    print(f"Bloques: base={blk_b}  alt={blk_a}  Δ={blk_a - blk_b:+d} | "
          f"crops fallback EasyOCR: base={easy_b} alt={easy_a}")
    return {
        "param": param,
        "label": info["label"],
        "desc": info["desc"],
        "reps": reps,
        "avg_base_s": round(avg_b, 3),
        "avg_alt_s": round(avg_a, 3),
        "delta_avg_s": round(avg_a - avg_b, 3),
        "pct_change": round((avg_a / avg_b - 1) * 100, 1),
        "control_pages": controls,
        "control_drift_s": round(drift_mean, 3),
        "control_drift_max_s": round(drift_max, 3),
        "effect_affected_s": round(effect_avg, 3),
        "verdict": verdict,
        "blocks_base": blk_b,
        "blocks_alt": blk_a,
        "delta_blocks": blk_a - blk_b,
        "easyocr_crops_base": easy_b,
        "easyocr_crops_alt": easy_a,
        "per_page": ab,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", default=DEFAULT_PAGES)
    ap.add_argument("--param", default="all", help="pad | interp | rotation | all")
    ap.add_argument("--reps", type=int, default=2,
                    help="Pares base/alt por página (mediana); default 2 — esta "
                         "máquina tiene ±0.3-0.7 s de ruido run-to-run (GPU "
                         "compartida con el daemon VLM)")
    ap.add_argument("--json", default="benchmark_results/rutac_params_ab.json")
    args = ap.parse_args()
    if args.reps < 1:
        ap.error("--reps debe ser >= 1")

    pages = [int(x) for x in args.pages.split(",")]
    params = (list(PARAMS) if args.param == "all"
              else [p.strip() for p in args.param.split(",")])

    doc = fitz.open(PDF)
    ocr = OCRManager()
    _wrap_timing("_pre_filter_image", "prefilter")
    _wrap_timing("_run_rapidocr", "rapid")
    _wrap_timing("_recover_regions_with_easyocr", "ruta_c")
    _wrap_timing("_detect_text_regions_in_page", "yolo")
    _wrap_easyocr_count()

    print(f"== A/B parámetros Ruta C ==  páginas={pages}  escala={SCALE}  "
          f"upscale={UPSCALE:g}  reps={args.reps}")
    print("(warmup de ambos caminos en la primera página — carga de modelos, "
          "se descarta)")

    results: dict[str, Any] = {}
    for param in params:
        # Warmup: primar AMBOS caminos (modelos, reader EasyOCR lazy, engine)
        # para que la carga no caiga dentro de una corrida medida.
        _apply_patch(param, "base")
        _run_once(ocr, render_page(doc, pages[0]))
        _apply_patch(param, "alt")
        _run_once(ocr, render_page(doc, pages[0]))
        _apply_patch(param, "base")
        # A/B intercalado con orden alternado por página (harness compartido).
        ab = abu.run_ab(
            pages,
            lambda which: _apply_patch(param, which),
            lambda pno: _run_once(ocr, render_page(doc, pno)),
            reps=args.reps,
        )
        results[param] = compare(pages, ab, param, reps=args.reps)

    doc.close()
    Path(args.json).parent.mkdir(exist_ok=True)
    Path(args.json).write_text(json.dumps(results, indent=1, ensure_ascii=False),
                               encoding="utf-8")
    print(f"\nResultado: {args.json}")


if __name__ == "__main__":
    main()
