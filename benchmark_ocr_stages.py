"""
benchmark_ocr_stages.py — Mide el split detector/recognizer de EasyOCR.

Prerequisito de la Fase 3.4.1 del plan de optimización: antes de cambiar de
modelo hay que saber qué etapa domina el coste del camino caliente. `readtext`
ejecuta detect() y recognize() en serie; este benchmark los llama POR
SEPARADO (igual que readtext, ver source de Reader.readtext) y reporta cuánto
paga cada etapa.

Uso:
    python benchmark_ocr_stages.py --pages 3,11,12 [--reps 2]

Output:
    Por página: t_detect, t_recognize, t_total, y el % que domina. Al final,
    promedios y la recomendación del plan (recognizer domina → probar
    recortar con RapidOCR; detector domina → dejar CRAFT).

Los resultados se escriben también en benchmark_results/ocr_stages.json para
comparar entre sesiones (mismo patrón que los demás benchmarks del repo).
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

# Mismos parámetros de producción (ocr_utils.py) para que el split sea
# representativo del camino caliente real.
CANVAS_SIZE = 2500
TEXT_THRESHOLD = 0.15
LOW_TEXT = 0.10
MIN_SIZE = 6
MAG_RATIO = 1.3
BATCH_SIZE = 8  # Fase 1 (2.2): batch del recognizer ya activo en producción


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


def measure_split(reader: Any, img_bgr: np.ndarray,
                  reps: int = 1) -> dict[str, float]:
    """Detect + recognize por separado, como hace readtext (detect(0)[0])."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    t_detect = 0.0
    t_recognize = 0.0
    n_boxes = 0
    for _ in range(reps):
        t0 = time.time()
        horizontal, free = reader.detect(
            img_rgb,
            min_size=MIN_SIZE, text_threshold=TEXT_THRESHOLD,
            low_text=LOW_TEXT, canvas_size=min(max(img_bgr.shape[:2]), CANVAS_SIZE),
            mag_ratio=MAG_RATIO,
            reformat=False,
        )
        t1 = time.time()
        horizontal = horizontal[0] if isinstance(horizontal, list) and horizontal else []
        free = free[0] if isinstance(free, list) and free else []
        n_boxes = len(horizontal) + len(free)
        reader.recognize(
            img_rgb, horizontal, free,
            batch_size=BATCH_SIZE, workers=0,
            detail=1, paragraph=False,
        )
        t2 = time.time()
        t_detect += t1 - t0
        t_recognize += t2 - t1
    return {
        "detect_s": round(t_detect / reps, 4),
        "recognize_s": round(t_recognize / reps, 4),
        "total_s": round((t_detect + t_recognize) / reps, 4),
        "n_boxes": n_boxes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", default="3,11,12",
                    help="Páginas a medir (separadas por coma)")
    ap.add_argument("--reps", type=int, default=1,
                    help="Repeticiones por página (promedia)")
    args = ap.parse_args()

    pdf_path = find_pdf()
    if not pdf_path:
        print("[benchmark] No se encontró un PDF en el directorio")
        sys.exit(1)
    print(f"[benchmark] PDF: {pdf_path}")

    from ocr_utils import _get_ocr_reader  # carga el reader real (GPU)
    reader = _get_ocr_reader("auto")
    if reader is None:
        print("[benchmark] No se pudo cargar EasyOCR")
        sys.exit(1)

    pdf = fitz.open(pdf_path)
    pages = [int(p) for p in args.pages.split(",")]
    results: dict[str, dict[str, float]] = {}
    print(f"{'pág':>4} {'detect':>8} {'recognize':>10} {'total':>8} {'boxes':>6} {'%rec':>6}")
    for p in pages:
        img = render_page(pdf, p)
        r = measure_split(reader, img, reps=args.reps)
        results[str(p)] = r
        pct_rec = (r["recognize_s"] / r["total_s"] * 100) if r["total_s"] > 0 else 0
        print(f"{p:>4} {r['detect_s']:>8.3f} {r['recognize_s']:>10.3f} "
              f"{r['total_s']:>8.3f} {r['n_boxes']:>6} {pct_rec:>5.1f}%")

    # Promedios
    avg_detect = sum(r["detect_s"] for r in results.values()) / len(results)
    avg_rec = sum(r["recognize_s"] for r in results.values()) / len(results)
    avg_total = avg_detect + avg_rec
    pct_rec_avg = (avg_rec / avg_total * 100) if avg_total > 0 else 0
    print(f"\nPromedio: detect={avg_detect:.3f}s recognize={avg_rec:.3f}s "
          f"total={avg_total:.3f}s — recognize domina {pct_rec_avg:.1f}%")
    if pct_rec_avg > 60:
        print("→ RECOGNIZER domina: probar recortar las cajas de EasyOCR y "
              "reconocer con RapidOCR (PP-OCRv4) — recomendación 4.1 del plan.")
    elif pct_rec_avg < 40:
        print("→ DETECTOR domina: dejar CRAFT como está (los detectores "
              "alternativos rinden similar en manga).")
    else:
        print("→ Split equilibrado: el cambio de recognizer tendría ganancia "
              "parcial; evaluar con analisis_calidad.py antes de adoptarlo.")

    # Persistir para comparación entre sesiones
    out = Path("benchmark_results")
    out.mkdir(exist_ok=True)
    summary = {
        "pdf": pdf_path,
        "reps": args.reps,
        "params": {"canvas_size": CANVAS_SIZE, "batch_size": BATCH_SIZE,
                   "mag_ratio": MAG_RATIO},
        "avg_detect_s": round(avg_detect, 4),
        "avg_recognize_s": round(avg_rec, 4),
        "recognize_pct": round(pct_rec_avg, 1),
        "pages": results,
    }
    (out / "ocr_stages.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[benchmark] Resultados guardados en benchmark_results/ocr_stages.json")


if __name__ == "__main__":
    main()
