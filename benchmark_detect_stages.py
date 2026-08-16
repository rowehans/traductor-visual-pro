"""
benchmark_detect_stages.py — Perfila _detect_and_ocr POR ETAPA.

Pregunta de diseño: ¿el retry de alta resolución (mag_ratio=1.8) o la fusión
con RapidOCR agregan coste evitable en páginas NORMALES? Para responderlo se
instrumenta la función REAL (monkeypatch de las etapas internas de ocr_utils)
en vez de replicar el flujo a mano — así se mide exactamente el camino caliente
con sus condiciones reales.

Etapas medidas dentro de _detect_and_ocr:
  1. prefilter   — _pre_filter_image (limpieza morfológica, SIEMPRE con prefilter=True)
  2. tier1       — _run_ocr_on_image mag default (1.3), primer readtext
  3. retry18     — _run_ocr_on_image mag_ratio=1.8 (SOLO si confianza < umbral)
  4. rapid       — _preprocess_rapid + _run_rapidocr (SOLO si página débil/vacía)
  5. fusion      — _fusionar_blocks (SOLO si RapidOCR devolvió bloques)
  6. enhanced    — _preprocess_enhanced + readtext fallback (SOLO si 0 bloques)

Fase B (separada, por fuera de _detect_and_ocr): la Ruta C
(_recover_regions_with_easyocr) vive en _run_fusion de ocr_engine.py — se mide
aquí el costo por crop con upscale real 3.5× para documentar cuánto agrega el
refuerzo de regiones.

Uso:
    python benchmark_detect_stages.py [--pages 3,11,12] [--rutac-top 3]

Output:
    Por página: tiempo de cada etapa que corrió, total, e invocaciones por
    etapa. Al final, resumen de coste evitable (retry18 + fusion en páginas
    normales) y costo de la Ruta C por crop.

Los resultados se escriben en benchmark_results/detect_stages.json.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any, Final

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(  # type: ignore[union-attr]
            encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import fitz

import ocr_utils

# Upscale de la Ruta C en producción (revertido a 3.5× el 2026-08-15 — el A/B
# del 2× estaba roto y el 2× perdía 2 bloques en pág 11 a tiempo neutro, ver
# benchmark_rutac_upscale.py). Constante explícita para que esta medición no
# dependa silenciosamente del default de _recover_regions_with_easyocr.
UPSCALE: Final[float] = 3.5


def find_pdf() -> str | None:
    """Localiza un PDF de muestra del repo para el benchmark."""
    for pat in ["*43*.pdf", "*capitulo*43*.pdf", "*villanos*.pdf", "*Olympus*.pdf"]:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    pdfs = glob.glob("*.pdf")
    return pdfs[0] if pdfs else None


def render_page(pdf: fitz.Document, page_num: int) -> np.ndarray:
    """Renderiza la página del PDF a BGR (escala 2.0, como los benchmarks del repo)."""
    page = pdf.load_page(page_num - 1)  # 0-indexed
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Instrumentación: wrappers de timing por etapa ───────────────────────
class _StageTimer:
    """Acumula tiempo e invocaciones por etapa mientras corre _detect_and_ocr."""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}
        self.calls: dict[str, int] = {}
        # flag para distinguir el readtext del fallback CLAHE (mag default)
        # del tier 1 (mag default): el fallback solo corre tras _preprocess_enhanced.
        self._enhanced_done = False
        # _preprocess_rapid llama INTERNAMENTE a _preprocess_enhanced: ese
        # enhance ya queda cubierto por el wrapper de rapid_prep y NO debe
        # contarse como fallback CLAHE de 0 bloques.
        self._in_rapid = False

    def record(self, stage: str, seconds: float) -> None:
        self.times[stage] = self.times.get(stage, 0.0) + seconds
        self.calls[stage] = self.calls.get(stage, 0) + 1


def _instrument(timer: _StageTimer) -> dict[str, Any]:
    """Envuelve las etapas internas de ocr_utils con timing. Devuelve los
    originales para restaurarlos al terminar."""
    originals: dict[str, Any] = {}

    def _wrap(name: str, stage: str | None = None) -> None:
        orig = getattr(ocr_utils, name)
        originals[name] = orig

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                timer.record(stage or name, time.perf_counter() - t0)
        setattr(ocr_utils, name, wrapper)

    def _wrap_readtext() -> None:
        """_run_ocr_on_image: distingue tier1 / retry18 / fallback CLAHE por
        mag_ratio explícito y por el flag de enhanced."""
        orig = getattr(ocr_utils, "_run_ocr_on_image")
        originals["_run_ocr_on_image"] = orig

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return orig(*args, **kwargs)
            finally:
                mag = kwargs.get("mag_ratio")
                if mag is not None and abs(float(mag) - 1.8) < 1e-6:
                    stage = "retry18"
                elif timer._enhanced_done:
                    stage = "enhanced"
                else:
                    stage = "tier1"
                timer.record(stage, time.perf_counter() - t0)
        setattr(ocr_utils, "_run_ocr_on_image", wrapper)

    def _wrap_preprocess_enhanced() -> None:
        orig = getattr(ocr_utils, "_preprocess_enhanced")
        originals["_preprocess_enhanced"] = orig

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            # Dentro de _preprocess_rapid el enhance ya se cuenta en
            # rapid_prep; solo el fallback CLAHE de 0 bloques (o el prefilter
            # directo) marca el readtext posterior como "enhanced".
            if not timer._in_rapid:
                timer._enhanced_done = True
            try:
                return orig(*args, **kwargs)
            finally:
                if not timer._in_rapid:
                    timer.record("enhanced_prep", time.perf_counter() - t0)
        setattr(ocr_utils, "_preprocess_enhanced", wrapper)

    def _wrap_preprocess_rapid() -> None:
        """_preprocess_rapid envuelve pre-filter + enhance y llama a
        _preprocess_enhanced internamente: el flag _in_rapid evita que ese
        enhance interno se cuente dos veces (rapid_prep + enhanced_prep)."""
        orig = getattr(ocr_utils, "_preprocess_rapid")
        originals["_preprocess_rapid"] = orig

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            timer._in_rapid = True
            try:
                return orig(*args, **kwargs)
            finally:
                timer._in_rapid = False
                timer.record("rapid_prep", time.perf_counter() - t0)
        setattr(ocr_utils, "_preprocess_rapid", wrapper)

    _wrap_readtext()
    _wrap("_pre_filter_image", "prefilter")
    _wrap("_run_rapidocr", "rapid")
    _wrap("_fusionar_blocks", "fusion")
    _wrap("_fusionar_blocks_multi", "fusion")
    _wrap_preprocess_enhanced()
    _wrap_preprocess_rapid()
    return originals


def _restore(originals: dict[str, Any]) -> None:
    for name, orig in originals.items():
        setattr(ocr_utils, name, orig)


def measure_page(timer: _StageTimer, img_bgr: np.ndarray) -> dict[str, Any]:
    """Corre _detect_and_ocr real con los params de producción y devuelve el
    desglose por etapa."""
    timer.times.clear()
    timer.calls.clear()
    timer._enhanced_done = False
    timer._in_rapid = False
    t0 = time.perf_counter()
    blocks = ocr_utils._detect_and_ocr(
        img_bgr, lang_hint="auto", allow_fallback=True,
        prefilter=True, use_hybrid=True, avg_conf_threshold=0.15,
    )
    total = time.perf_counter() - t0
    return {
        "total_s": round(total, 4),
        "n_blocks": len(blocks),
        "stages": {k: round(v, 4) for k, v in sorted(timer.times.items())},
        "calls": dict(sorted(timer.calls.items())),
    }


def measure_rutac(img_bgr: np.ndarray, top_n: int) -> dict[str, Any]:
    """Fase B: costo de la Ruta C (_recover_regions_with_easyocr) por crop,
    con upscale real 3.5×, sobre las regiones más grandes detectadas."""
    blocks = ocr_utils._detect_and_ocr(
        img_bgr, lang_hint="auto", allow_fallback=True,
        prefilter=True, use_hybrid=True, avg_conf_threshold=0.15,
    )
    # Regiones sustitutas: los bloques más grandes (área) — la Ruta C real
    # recibe regiones de YOLO/CTD, pero el costo por crop es lo que se mide.
    by_area = sorted(
        blocks, key=lambda b: int(b.get("w", 0)) * int(b.get("h", 0)),
        reverse=True)
    regions = [
        {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]}
        for b in by_area[:top_n]
    ]
    if not regions:
        return {"regions": 0, "total_s": 0.0, "per_crop_s": 0.0, "recovered": 0}
    t0 = time.perf_counter()
    recovered = ocr_utils._recover_regions_with_easyocr(
        img_bgr, regions, lang_hint="auto", upscale=UPSCALE)
    total = time.perf_counter() - t0
    return {
        "regions": len(regions),
        "total_s": round(total, 4),
        "per_crop_s": round(total / len(regions), 4),
        "recovered": len(recovered),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", default="3,11,12",
                    help="Páginas a medir (separadas por coma)")
    ap.add_argument("--rutac-top", type=int, default=3,
                    help="Top N regiones a re-OCRear en la fase Ruta C")
    args = ap.parse_args()

    pdf_path = find_pdf()
    if not pdf_path:
        print("[benchmark] No se encontró un PDF en el directorio")
        sys.exit(1)
    print(f"[benchmark] PDF: {pdf_path}")

    # Cargar el reader real para que _detect_and_ocr no lo recargue por página
    reader = ocr_utils._get_ocr_reader("auto")
    if reader is None:
        print("[benchmark] No se pudo cargar EasyOCR")
        sys.exit(1)

    pdf = fitz.open(pdf_path)
    pages = [int(p) for p in args.pages.split(",")]
    timer = _StageTimer()
    originals = _instrument(timer)
    try:
        results: dict[str, Any] = {}
        header = (f"{'pág':>4} {'total':>7} {'blk':>4} "
                  f"{'prefilter':>9} {'tier1':>7} {'retry18':>8} "
                  f"{'rapid':>6} {'fusion':>7} {'enhanced':>9}")
        print(header)
        for p in pages:
            img = render_page(pdf, p)
            r = measure_page(timer, img)
            results[str(p)] = r
            st = r["stages"]
            print(f"{p:>4} {r['total_s']:>7.3f} {r['n_blocks']:>4} "
                  f"{st.get('prefilter', 0):>9.3f} {st.get('tier1', 0):>7.3f} "
                  f"{st.get('retry18', 0):>8.3f} {st.get('rapid', 0):>6.3f} "
                  f"{st.get('fusion', 0):>7.3f} "
                  f"{st.get('enhanced', 0) + st.get('enhanced_prep', 0):>9.3f}")
    finally:
        _restore(originals)

    # ── Resumen de coste evitable ────────────────────────────────────────
    tot_retry = sum(r["stages"].get("retry18", 0) for r in results.values())
    tot_fusion = sum(r["stages"].get("fusion", 0) for r in results.values())
    tot_total = sum(r["total_s"] for r in results.values())
    n_retry = sum(1 for r in results.values() if r["calls"].get("retry18", 0))
    n_fusion = sum(1 for r in results.values() if r["calls"].get("fusion", 0))
    tot_prefilter = sum(r["stages"].get("prefilter", 0) for r in results.values())
    tot_tier1 = sum(r["stages"].get("tier1", 0) for r in results.values())
    tot_rapid = (sum(r["stages"].get("rapid", 0) for r in results.values())
                 + sum(r["stages"].get("rapid_prep", 0) for r in results.values()))
    print(f"\nResumen ({len(pages)} págs, total {tot_total:.2f}s):")
    print(f"  retry mag=1.8: {tot_retry:.3f}s en {n_retry}/{len(pages)} págs "
          f"({tot_retry / tot_total * 100:.1f}% del total)")
    print(f"  fusión RapidOCR: {tot_fusion:.3f}s en {n_fusion}/{len(pages)} págs "
          f"({tot_fusion / tot_total * 100:.1f}% del total)")
    retry_pct = tot_retry / tot_total * 100 if tot_total else 0
    fusion_pct = tot_fusion / tot_total * 100 if tot_total else 0
    if retry_pct < 0.5 and fusion_pct < 0.5:
        print("  → COSTE EVITABLE DESPRECIABLE: el retry no corre y la fusión"
              " cuesta <0.5% del total en estas páginas.")
        dom = max([("prefilter", tot_prefilter), ("tier1", tot_tier1),
                   ("rapid", tot_rapid)], key=lambda x: x[1])
        print(f"  → La etapa dominante es {dom[0]} ({dom[1]:.2f}s, "
              f"{dom[1] / tot_total * 100:.0f}% del total) — si se quiere "
              "acelerar páginas normales, el foco está ahí, no en retry/fusión.")
    else:
        print("  → Retry/fusión corrieron (páginas débiles): coste buscado, "
              "no evitable sin perder recuperación de texto artístico.")

    # ── Fase B: Ruta C ───────────────────────────────────────────────────
    print(f"\n=== Fase B: Ruta C (_recover_regions_with_easyocr, upscale {UPSCALE:g}×) ===")
    rutac: dict[str, Any] = {}
    for p in pages:
        img = render_page(pdf, p)
        rc = measure_rutac(img, args.rutac_top)
        rutac[str(p)] = rc
        print(f"  pág {p}: {rc['regions']} regiones, {rc['total_s']:.3f}s "
              f"({rc['per_crop_s']:.3f}s/crop), {rc['recovered']} recuperados")

    # ── Persistir ────────────────────────────────────────────────────────
    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    summary = {
        "pdf": pdf_path,
        "pages": pages,
        "rutac_top": args.rutac_top,
        "params": {
            "prefilter": True, "use_hybrid": True,
            "avg_conf_threshold": 0.15, "rutac_upscale": UPSCALE,
        },
        "per_page": results,
        "totals": {
            "total_s": round(tot_total, 4),
            "retry18_s": round(tot_retry, 4),
            "fusion_s": round(tot_fusion, 4),
            "retry18_pages": n_retry,
            "fusion_pages": n_fusion,
            "retry18_pct": round(tot_retry / tot_total * 100, 2) if tot_total else 0,
            "fusion_pct": round(tot_fusion / tot_total * 100, 2) if tot_total else 0,
        },
        "rutac": rutac,
    }
    (out / "detect_stages.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[benchmark] Resultados guardados en benchmark_results/detect_stages.json")


if __name__ == "__main__":
    main()
