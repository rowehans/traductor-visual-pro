"""
benchmark_prefilter.py — Mide el coste del pre-filter por sub-etapa y si
cambia el resultado OCR en páginas normales (A/B con/sin prefilter).

Hallazgo del benchmark de etapas (benchmark_detect_stages.py): _pre_filter_image
cuesta 1.4-1.6s/pág — MÁS que el propio tier1 de EasyOCR (0.8-1.0s) — y corre
SIEMPRE. Este benchmark responde dos preguntas:

  1. ¿Dónde gasta el prefilter? (bilateralFilter es no-separable y notorio por
     su coste en imágenes grandes; los inpaint TELEA también son caros).
  2. ¿Aporta algo en páginas normales? (A/B: bloques y confianza con/sin).

Uso:
    python benchmark_prefilter.py [--pages 3,11,12] [--sub-etapas]

Output:
    Por página: tiempo de cada sub-etapa + OCR con/sin prefilter (bloques,
    confianza media). Al final, si la confianza sin prefilter es comparable,
    el prefilter es candidato a hacerse condicional o a reemplazar el bilateral.

Los resultados se escriben en benchmark_results/prefilter.json.
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

import ocr_utils


def find_pdf() -> str | None:
    for pat in ["*43*.pdf", "*capitulo*43*.pdf", "*villanos*.pdf", "*Olympus*.pdf"]:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    pdfs = glob.glob("*.pdf")
    return pdfs[0] if pdfs else None


def render_page(pdf: fitz.Document, page_num: int) -> np.ndarray:
    page = pdf.load_page(page_num - 1)
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def measure_substages(img_bgr: np.ndarray) -> dict[str, float]:
    """Replica _pre_filter_image midiendo cada sub-etapa por separado.

    Es una REPLICA para medir (benchmark dev, fuera del CI): si el prefilter
    cambia, hay que actualizar esta réplica a mano. El objetivo es localizar
    dónde se van los 1.4-1.6s, no validar el pipeline de producción.
    """
    h, w = img_bgr.shape[:2]
    t = {}

    # 1. Franjas 4% (numpy puro)
    t0 = time.perf_counter()
    margin_height = max(1, int(h * 0.04))
    top_strip = img_bgr[margin_height:margin_height * 2, :, :]
    if top_strip.size > 0:
        _top_fill = np.median(top_strip.reshape(-1, 3), axis=0).astype(np.uint8)
    bot_strip = img_bgr[h - margin_height * 2:h - margin_height, :, :]
    if bot_strip.size > 0:
        _bot_fill = np.median(bot_strip.reshape(-1, 3), axis=0).astype(np.uint8)
    t["franjas"] = time.perf_counter() - t0

    # 2. Líneas horizontales (morfología + inpaint TELEA)
    t0 = time.perf_counter()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detect_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    _, thresh_lines = cv2.threshold(detect_lines, 50, 255, cv2.THRESH_BINARY)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    line_mask = cv2.dilate(thresh_lines, kernel_dilate, iterations=1)
    t["lineas_morf"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if int(line_mask.max()) > 0:
        _ = cv2.inpaint(img_bgr, line_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    t["lineas_inpaint"] = time.perf_counter() - t0

    # 3. Speckle (OTSU + MORPH_OPEN + XOR + inpaint TELEA)
    t0 = time.perf_counter()
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    speckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, speckle_kernel, iterations=1)
    speckle_pixels = cv2.bitwise_xor(binary, cleaned)
    t["speckle_morf"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    if int(speckle_pixels.max()) > 0 and np.count_nonzero(speckle_pixels) > 50:
        speckle_mask = cv2.dilate(speckle_pixels, speckle_kernel, iterations=1)
        _ = cv2.inpaint(img_bgr, speckle_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)
    t["speckle_inpaint"] = time.perf_counter() - t0

    # 4. Bilateral (no-separable, sospechoso principal)
    t0 = time.perf_counter()
    _ = cv2.bilateralFilter(img_bgr, d=3, sigmaColor=30, sigmaSpace=30)
    t["bilateral"] = time.perf_counter() - t0

    # Alternativas baratas (gaussian / median) para comparar coste
    t0 = time.perf_counter()
    _ = cv2.GaussianBlur(img_bgr, (3, 3), 0)
    t["gaussian"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = cv2.medianBlur(img_bgr, 3)
    t["median"] = time.perf_counter() - t0

    return {k: round(v, 4) for k, v in t.items()}


def ocr_stats(img: np.ndarray) -> dict[str, Any]:
    """Corre _detect_and_ocr real (prefilter=False) y devuelve bloques/conf."""
    blocks = ocr_utils._detect_and_ocr(
        img, lang_hint="auto", allow_fallback=True,
        prefilter=False, use_hybrid=True, avg_conf_threshold=0.15,
    )
    avg_conf = (float(np.mean([b.get("confidence", 0) for b in blocks]))
                if blocks else 0.0)
    return {"n_blocks": len(blocks), "avg_conf": round(avg_conf, 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", default="3,11,12")
    ap.add_argument("--sub-etapas", action="store_true",
                    help="Mide también las sub-etapas del prefilter")
    args = ap.parse_args()

    pdf_path = find_pdf()
    if not pdf_path:
        print("[benchmark] No se encontró un PDF en el directorio")
        sys.exit(1)
    print(f"[benchmark] PDF: {pdf_path}")

    reader = ocr_utils._get_ocr_reader("auto")
    if reader is None:
        print("[benchmark] No se pudo cargar EasyOCR")
        sys.exit(1)

    pdf = fitz.open(pdf_path)
    pages = [int(p) for p in args.pages.split(",")]
    results: dict[str, Any] = {}

    print(f"{'pág':>4} {'sinPref':>10} {'conPref':>9} {'Δconf':>6} "
          f"{'Δblk':>5}")
    for p in pages:
        img = render_page(pdf, p)
        # A/B: mismo pipeline, prefilter on/off (solo cambia la imagen de entrada)
        t0 = time.perf_counter()
        stats_off = ocr_stats(img)
        t_off = time.perf_counter() - t0
        t0 = time.perf_counter()
        img_pre = ocr_utils._pre_filter_image(img)
        t_pre = time.perf_counter() - t0
        t0 = time.perf_counter()
        stats_on = ocr_stats(img_pre)
        t_on = time.perf_counter() - t0

        entry: dict[str, Any] = {
            "sin_prefilter": {**stats_off, "ocr_s": round(t_off, 4)},
            "con_prefilter": {**stats_on, "ocr_s": round(t_on, 4)},
            "prefilter_s": round(t_pre, 4),
            "delta_conf": round(stats_on["avg_conf"] - stats_off["avg_conf"], 3),
            "delta_blocks": stats_on["n_blocks"] - stats_off["n_blocks"],
        }
        if args.sub_etapas:
            entry["subetapas"] = measure_substages(img)
        results[str(p)] = entry
        print(f"{p:>4} {stats_off['n_blocks']:>5}/{stats_off['avg_conf']:.2f} "
              f"{stats_on['n_blocks']:>5}/{stats_on['avg_conf']:.2f} "
              f"{entry['delta_conf']:>+6.2f} {entry['delta_blocks']:>+5} "
              f"(prefilter {t_pre:.2f}s)")

    # Resumen
    print(f"\nResumen ({len(pages)} págs):")
    for key, label in [("delta_conf", "Δ confianza media (con - sin)"),
                       ("delta_blocks", "Δ bloques (con - sin)")]:
        vals = [r[key] for r in results.values()]
        avg = sum(vals) / len(vals)
        print(f"  {label}: {avg:+.3f}")
    if args.sub_etapas:
        print("\nSub-etapas del prefilter (pág media):")
        keys = ["franjas", "lineas_morf", "lineas_inpaint", "speckle_morf",
                "speckle_inpaint", "bilateral", "gaussian", "median"]
        for k in keys:
            avg = sum(r["subetapas"][k] for r in results.values()) / len(pages)
            print(f"  {k:>16}: {avg:.3f}s")

    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    summary = {"pdf": pdf_path, "pages": pages, "per_page": results}
    (out / "prefilter.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[benchmark] Resultados guardados en benchmark_results/prefilter.json")


if __name__ == "__main__":
    main()
