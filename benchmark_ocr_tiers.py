"""
Benchmark: OCR pipeline 3 niveles vs 2 niveles (sin tier 3 - binarizacion).
Compara deteccion en paginas 3, 11, 12 del Capitulo 43 (texto artistico).
"""
import sys, os, time, json

# Check dependencies
try:
    import fitz
except ImportError:
    print("[ERROR] Se necesita PyMuPDF: pip install pymupdf")
    sys.exit(1)

sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np

from ocr_utils import (
    _get_ocr_reader, _run_ocr_on_image, _ocr_results_to_blocks,
    _pre_filter_image, _preprocess_enhanced, _binarize_image,
)

PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente _ Olympus Scanlation_compressed.pdf"
PAGES = [3, 11, 12]

SEP = "=" * 70


def render_pdf_page(pdf_path: str, page_num: int, scale: float = 1.8) -> np.ndarray:
    """Renderiza una pagina de PDF a imagen numpy BGR."""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    doc.close()
    return img_bgr


def ocr_tier1(reader, img_bgr):
    """Solo tier 1: EasyOCR directo."""
    results = _run_ocr_on_image(reader, img_bgr)
    return _ocr_results_to_blocks(results, img_bgr)


def ocr_tiers_12(reader, img_bgr):
    """Tiers 1+2: EasyOCR + CLAHE fallback (SIN tier 3)."""
    # Tier 1
    results = _run_ocr_on_image(reader, img_bgr)
    blocks = _ocr_results_to_blocks(results, img_bgr)
    if blocks:
        return blocks
    # Tier 2
    img_filtered = _pre_filter_image(img_bgr)
    img_enhanced = _preprocess_enhanced(img_filtered)
    results2 = _run_ocr_on_image(reader, img_enhanced)
    return _ocr_results_to_blocks(results2, img_enhanced)


def ocr_tiers_123(reader, img_bgr):
    """Tiers 1+2+3: Pipeline completo actual."""
    # Tier 1
    results = _run_ocr_on_image(reader, img_bgr)
    blocks = _ocr_results_to_blocks(results, img_bgr)
    if blocks:
        return blocks
    # Tier 2
    img_filtered = _pre_filter_image(img_bgr)
    img_enhanced = _preprocess_enhanced(img_filtered)
    results2 = _run_ocr_on_image(reader, img_enhanced)
    blocks2 = _ocr_results_to_blocks(results2, img_enhanced)
    if blocks2:
        return blocks2
    # Tier 3
    img_binary = _binarize_image(img_bgr)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img_morph = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, morph_kernel, iterations=1)
    results3 = _run_ocr_on_image(reader, img_morph)
    return _ocr_results_to_blocks(results3, img_morph)


def show_texts(blocks):
    """Muestra los primeros 40 chars de cada bloque."""
    if blocks:
        texts = [b.get("text", "")[:40] for b in blocks]
        for t in texts:
            print(f"       [{t}]")


def main():
    print(SEP)
    print("BENCHMARK: Pipeline OCR 3 niveles vs 2 niveles")
    print("PDF: Capitulo 43 | Paginas: 3, 11, 12")
    print(SEP)

    # Cargar reader
    print("\n[PREP] Cargando EasyOCR reader...")
    sys.stdout.flush()
    reader = _get_ocr_reader("auto")
    if reader is None:
        print("[ERROR] No se pudo cargar EasyOCR")
        return
    print("[PREP] Reader listo.")
    sys.stdout.flush()

    results = {}

    for page_num in PAGES:
        print(f"\n{SEP}")
        print(f"PAGINA {page_num}")
        print(SEP)

        # Renderizar pagina
        print(f"  Renderizando pagina {page_num}...")
        sys.stdout.flush()
        img_bgr = render_pdf_page(PDF_PATH, page_num)
        h, w = img_bgr.shape[:2]
        print(f"  Tamano: {w}x{h}")
        sys.stdout.flush()

        page_results = {}

        # Pipeline 1: Solo tier 1 (EasyOCR directo)
        print(f"\n  [1] TIER 1 SOLO (EasyOCR directo):")
        t0 = time.time()
        blocks1 = ocr_tier1(reader, img_bgr)
        t1 = time.time()
        dt1 = t1 - t0
        print(f"      Bloques: {len(blocks1)} | Tiempo: {dt1:.2f}s")
        show_texts(blocks1)
        page_results["tier1"] = {"blocks": len(blocks1), "time_s": round(dt1, 2)}

        # Pipeline 2: Tiers 1+2 (sin binarizacion)
        print(f"\n  [2] TIERS 1+2 (EasyOCR + CLAHE):")
        sys.stdout.flush()
        t0 = time.time()
        blocks12 = ocr_tiers_12(reader, img_bgr)
        t1 = time.time()
        dt12 = t1 - t0
        print(f"      Bloques: {len(blocks12)} | Tiempo: {dt12:.2f}s")
        show_texts(blocks12)
        page_results["tiers_12"] = {"blocks": len(blocks12), "time_s": round(dt12, 2)}

        # Pipeline 3: Tiers 1+2+3 (completo)
        print(f"\n  [3] TIERS 1+2+3 (EasyOCR + CLAHE + Binarizacion):")
        sys.stdout.flush()
        t0 = time.time()
        blocks123 = ocr_tiers_123(reader, img_bgr)
        t1 = time.time()
        dt123 = t1 - t0
        print(f"      Bloques: {len(blocks123)} | Tiempo: {dt123:.2f}s")
        show_texts(blocks123)
        page_results["tiers_123"] = {"blocks": len(blocks123), "time_s": round(dt123, 2)}

        # Comparacion
        n1 = len(blocks1); n12 = len(blocks12); n123 = len(blocks123)
        mejora_12 = n12 - n1
        mejora_123 = n123 - n12
        mejora_total = n123 - n1
        print(f"\n  COMPARACION PAGINA {page_num}:")
        print(f"    Tier 1 solo:     {n1:3d} bloques  ({dt1:.2f}s)")
        print(f"    Tiers 1+2:       {n12:3d} bloques  ({dt12:.2f}s)  (+{mejora_12:+d} vs Tier1)")
        print(f"    Tiers 1+2+3:     {n123:3d} bloques  ({dt123:.2f}s)  (+{mejora_123:+d} vs 1+2, total +{mejora_total:+d})")
        sys.stdout.flush()

        results[str(page_num)] = page_results

    # Resumen final
    print(f"\n{SEP}")
    print("RESUMEN FINAL")
    print(SEP)

    total_b1 = total_b12 = total_b123 = 0
    total_t1 = total_t12 = total_t123 = 0.0
    for p in PAGES:
        r = results[str(p)]
        total_b1 += r["tier1"]["blocks"]; total_t1 += r["tier1"]["time_s"]
        total_b12 += r["tiers_12"]["blocks"]; total_t12 += r["tiers_12"]["time_s"]
        total_b123 += r["tiers_123"]["blocks"]; total_t123 += r["tiers_123"]["time_s"]

    print(f"\n  Pagina   | Tier1 (bloq/seg)  | Tiers12 (bloq/seg) | Tiers123 (bloq/seg)")
    print(f"  ---------+--------------------+--------------------+--------------------")
    for p in PAGES:
        r = results[str(p)]
        print(f"  {p:>7} | {r['tier1']['blocks']:>3d} / {r['tier1']['time_s']:>5.2f}s  | "
              f"{r['tiers_12']['blocks']:>3d} / {r['tiers_12']['time_s']:>5.2f}s  | "
              f"{r['tiers_123']['blocks']:>3d} / {r['tiers_123']['time_s']:>5.2f}s")
    mej_total = total_b123 - total_b1
    mej_12 = total_b12 - total_b1
    mej_123 = total_b123 - total_b12
    print(f"  ---------+--------------------+--------------------+--------------------")
    print(f"  TOTAL    | {total_b1:>3d} / {total_t1:>5.2f}s  | "
          f"{total_b12:>3d} / {total_t12:>5.2f}s  | "
          f"{total_b123:>3d} / {total_t123:>5.2f}s")
    print(f"\n  GANANCIA vs Tier1 solo:")
    print(f"    Tiers 1+2:   +{mej_12:>3d} bloques (tier 2: CLAHE)")
    print(f"    Tiers 1+2+3: +{mej_total:>3d} bloques (tier 3: binarizacion aporta +{mej_123:+d})")
    pct = (mej_total / total_b1 * 100) if total_b1 > 0 else 0
    print(f"    Mejora total: +{mej_total} bloques ({pct:.1f}% sobre Tier1 solo)")
    print(SEP)

    # Guardar JSON
    with open("benchmark_ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en benchmark_ocr_results.json")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
