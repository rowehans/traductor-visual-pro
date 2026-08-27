"""
benchmark_prefilter_selective.py — A/B del detector de líneas selectivo (vía
4.4C del plan) contra el detector actual de _pre_filter_image.

Contexto (plan §4.4): el inpaint TELEA de líneas horizontales cuesta ~0.53 s/pág
y el detector actual (kernel 1x15 + umbral 50 + dilate 5x1) marca 88-94 % del
área de la página como "línea" (falso positivo masivo: captura el arte del
manga). 4.4C = detector MÁS selectivo para que la máscara cubra solo líneas
largas reales y el inpaint corra sobre área chica (~0.03-0.1 s).

FASE 0 (diagnóstico) — variantes de detección de líneas reales:
  - current: réplica del detector actual (para medir el área que marca).
  - darklines: umbral de píxeles OSCUROS (líneas de escaneo reales son
    trazos oscuros, no brillos) + apertura 1x15 + filtro de componentes
    anchos y finos (>= 20 % del ancho de página y alto <= 8 px).
  Mide área de la máscara resultante y tiempo de inpaint por página.

FASE 2 — A/B OCR completo (pipeline real via OCRManager.run_ocr):
  - actual: prefilter tal cual (líneas + speckle + bilateral).
  - sin_lineas: réplica del prefilter SIN el paso de líneas (solo speckle +
    bilateral) — el caso límite del detector selectivo: área de líneas 0.
  Con pág 11 de control (la página débil donde el prefilter más aporta).

Uso:
    python benchmark_prefilter_selective.py [--pages 1,2,4,5,7,8,11,29,39,43,46,47,50,52]

Output: benchmark_results/prefilter_selective.json
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(  # type: ignore[union-attr]
            encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import fitz

import ocr_engine  # noqa: F401  (registra _get_ocr_reader / OCRManager)
import ocr_utils

SCALE = 1.2  # resolución real de producción (pdf.js default)
DEFAULT_PAGES = "1,2,4,5,7,8,11,29,39,43,46,47,50,52"
OCR_LANG = "es"


def find_pdf() -> str:
    for pat in ["*43*.pdf", "*capitulo*43*.pdf", "*villanos*.pdf"]:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return glob.glob("*.pdf")[0]


def render_page(pdf: fitz.Document, page_num: int) -> np.ndarray:
    page = pdf.load_page(page_num - 1)
    pix = page.get_pixmap(matrix=fitz.Matrix(SCALE, SCALE), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ── Detector actual (réplica exacta de _pre_filter_image) ─────────
def _detect_lines_current(gray: np.ndarray) -> np.ndarray:
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detect_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    _, thresh_lines = cv2.threshold(detect_lines, 50, 255, cv2.THRESH_BINARY)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.dilate(thresh_lines, kernel_dilate, iterations=1)


# ── Detector de LÍNEAS REALES (4.4C) ─────────────────────────────
# Las líneas de escaneo son trazos OSCUROS, largos y finos contra el fondo
# claro. El detector actual umbraliza lo BRILLANTE tras la apertura, que en
# un manga (fondo plano) marca la página completa. Este detector umbraliza
# lo oscuro y conserva solo componentes anchos y finos.
def _detect_lines_dark(gray: np.ndarray, w: int,
                       dark_thresh: int = 120,
                       min_run: int = 15,
                       min_frac_w: float = 0.20,
                       max_h: int = 8) -> np.ndarray:
    _, dark = cv2.threshold(gray, dark_thresh, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_run, 1))
    runs = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(runs, connectivity=8)
    mask = np.zeros_like(runs)
    min_w = w * min_frac_w
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw >= min_w and bh <= max_h:
            mask[labels == i] = 255
    return cv2.dilate(mask, cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 1)), iterations=1)


def line_mask_area(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / float(mask.size)


def measure_detectors(img_bgr: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    out: dict[str, Any] = {}
    for name, fn in [("current", lambda: _detect_lines_current(gray)),
                     ("dark", lambda: _detect_lines_dark(gray, w))]:
        mask = fn()  # type: ignore[no-untyped-call]
        t0 = time.perf_counter()
        if int(mask.max()) > 0:
            _ = cv2.inpaint(img_bgr, mask, inpaintRadius=3,
                            flags=cv2.INPAINT_TELEA)
        out[name] = {
            "mask_area_frac": round(line_mask_area(mask), 5),
            "inpaint_s": round(time.perf_counter() - t0, 4),
        }
    return out


def _prefilter_no_lines(img_bgr: np.ndarray) -> np.ndarray:
    """Réplica de _pre_filter_image SIN el paso de líneas (solo speckle +
    bilateral) — el límite del detector selectivo (área de líneas 0)."""
    h, w = img_bgr.shape[:2]
    result = img_bgr.copy()

    margin_height = max(1, int(h * 0.04))
    top_strip = img_bgr[margin_height:margin_height * 2, :, :]
    if top_strip.size > 0:
        top_fill = np.median(top_strip.reshape(-1, 3), axis=0).astype(np.uint8)
        result[:margin_height, :, :] = top_fill
    bot_strip = img_bgr[h - margin_height * 2:h - margin_height, :, :]
    if bot_strip.size > 0:
        bot_fill = np.median(bot_strip.reshape(-1, 3), axis=0).astype(np.uint8)
        result[h - margin_height:, :, :] = bot_fill

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    speckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, speckle_kernel, iterations=1)
    speckle_pixels = cv2.bitwise_xor(binary, cleaned)
    if int(speckle_pixels.max()) > 0 and np.count_nonzero(speckle_pixels) > 50:
        speckle_mask = cv2.dilate(speckle_pixels, speckle_kernel, iterations=1)
        result = cv2.inpaint(result, speckle_mask, inpaintRadius=2,
                             flags=cv2.INPAINT_TELEA)

    result = cv2.bilateralFilter(result, d=3, sigmaColor=30, sigmaSpace=30)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", default=DEFAULT_PAGES)
    ap.add_argument("--fase2-only", action="store_true",
                    help="Salta la Fase 0 (diagnóstico de detectores)")
    args = ap.parse_args()

    pdf_path = find_pdf()
    print(f"[benchmark] PDF: {pdf_path}  scale={SCALE}")
    pdf = fitz.open(pdf_path)
    pages = [int(p) for p in args.pages.split(",")]

    results: dict[str, Any] = {}

    if not args.fase2_only:
        print("\nFASE 0 — área de máscara + tiempo de inpaint por detector")
        print(f"{'pág':>4} {'current':>18} {'dark(real)':>18}")
        mask_data: dict[str, Any] = {}
        for p in pages:
            img = render_page(pdf, p)
            d = measure_detectors(img)
            mask_data[str(p)] = d
            print(f"{p:>4} "
                  f"{d['current']['mask_area_frac']*100:>7.1f}%/{d['current']['inpaint_s']:>6.2f}s "
                  f"{d['dark']['mask_area_frac']*100:>7.1f}%/{d['dark']['inpaint_s']:>6.2f}s")
        results["mask"] = mask_data

    print("\nFASE 2 — A/B OCR pipeline real: prefilter ACTUAL vs SIN líneas")
    print(f"{'pág':>4} {'actual':>12} {'sinLíneas':>12} {'Δblk':>5} {'Δconf':>6} "
          f"{'pref s':>8}")
    ocr = ocr_engine.OCRManager()
    per_page: dict[str, Any] = {}
    for p in pages:
        img = render_page(pdf, p)

        t0 = time.perf_counter()
        img_pre = ocr_utils._pre_filter_image(img)
        t_actual = time.perf_counter() - t0
        blocks_a, _, _ = ocr.run_ocr(img_pre, OCR_LANG, "fusion", prefilter=False)
        conf_a = (float(np.mean([b.get("confidence", 0) for b in blocks_a]))
                  if blocks_a else 0.0)

        t0 = time.perf_counter()
        img_nl = _prefilter_no_lines(img)
        t_nl = time.perf_counter() - t0
        blocks_b, _, _ = ocr.run_ocr(img_nl, OCR_LANG, "fusion", prefilter=False)
        conf_b = (float(np.mean([b.get("confidence", 0) for b in blocks_b]))
                  if blocks_b else 0.0)

        db = len(blocks_b) - len(blocks_a)
        dc = conf_b - conf_a
        per_page[str(p)] = {
            "actual": {"n_blocks": len(blocks_a), "avg_conf": round(conf_a, 3),
                       "prefilter_s": round(t_actual, 3)},
            "sin_lineas": {"n_blocks": len(blocks_b), "avg_conf": round(conf_b, 3),
                           "prefilter_s": round(t_nl, 3)},
            "delta_blocks": db,
            "delta_conf": round(dc, 3),
        }
        print(f"{p:>4} {len(blocks_a):>5}/{conf_a:.2f} "
              f"{len(blocks_b):>5}/{conf_b:.2f} "
              f"{db:>+5} {dc:>+6.2f} {t_actual:>5.2f}->{t_nl:.2f}")

    print("\nResumen Fase 2:")
    total_d = sum(per_page[str(p)]["delta_blocks"] for p in pages)
    print(f"  Δ bloques total (sin líneas - actual): {total_d:+d}")
    avg_a = sum(per_page[str(p)]["actual"]["prefilter_s"] for p in pages) / len(pages)
    avg_b = sum(per_page[str(p)]["sin_lineas"]["prefilter_s"] for p in pages) / len(pages)
    print(f"  prefilter medio: actual {avg_a:.2f}s -> sin líneas {avg_b:.2f}s")
    p11 = per_page.get("11")
    if p11:
        print(f"  pág 11 (control débil): actual {p11['actual']['n_blocks']}/"
              f"{p11['actual']['avg_conf']:.2f} vs sin líneas "
              f"{p11['sin_lineas']['n_blocks']}/{p11['sin_lineas']['avg_conf']:.2f}")

    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    summary = {
        "pdf": pdf_path, "scale_pdfjs": SCALE, "pages": pages,
        "per_page": per_page, **results,
    }
    (out / "prefilter_selective.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[benchmark] Resultados guardados en benchmark_results/prefilter_selective.json")


if __name__ == "__main__":
    main()
