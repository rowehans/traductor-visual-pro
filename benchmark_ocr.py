"""
benchmark_ocr.py — Compara EasyOCR vs RapidOCR en paginas problematicas.
Uso: python benchmark_ocr.py --pages 3,11,12
"""
import argparse
import glob
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import unicodedata

def normalize_text(t):
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return t

def find_pdf():
    for p in ["*43*.pdf", "*capitulo*43*.pdf", "*villanos*.pdf", "*Olympus*.pdf"]:
        matches = glob.glob(p)
        if matches:
            return matches[0]
    pdfs = glob.glob("*.pdf")
    return pdfs[0] if pdfs else None

def run_easyocr(reader, img, page_num):
    t0 = time.time()
    results = reader.readtext(img, detail=1, paragraph=False, min_size=6,
                                text_threshold=0.15, low_text=0.10, canvas_size=2500)
    elapsed = time.time() - t0
    blocks = [{"text": str(t).strip(), "conf": float(c), "engine": "easyocr"}
                for b, t, c in results if str(t).strip() and c >= 0.08]
    return blocks, elapsed

def run_rapidocr(engine, img, page_num):
    t0 = time.time()
    results, elapse_list = engine(img)
    elapsed = time.time() - t0
    blocks = []
    if results:
        for r in results:
            try:
                bbox, text, conf = r
                text = str(text).strip()
                if text and conf >= 0.08:
                    blocks.append({"text": text, "conf": float(conf), "engine": "rapidocr"})
            except (ValueError, IndexError, TypeError):
                continue
    return blocks, elapsed

def main():
    parser = argparse.ArgumentParser(description="Compara EasyOCR vs RapidOCR")
    parser.add_argument("--pdf", default=None)
    parser.add_argument("--pages", default="3,11,12")
    args = parser.parse_args()

    pages = [int(p.strip()) for p in args.pages.split(",")]
    pdf_path = args.pdf or find_pdf()
    if not pdf_path or not Path(pdf_path).exists():
        print(f"ERROR: No se encontro PDF. Usa --pdf <ruta>")
        return 1

    print("=" * 60)
    print(f"BENCHMARK OCR: EasyOCR (GPU) vs RapidOCR (ONNX)")
    print(f"PDF: {Path(pdf_path).name}")
    print(f"Paginas: {pages}")
    print("=" * 60)

    # Extraer paginas
    images = {}
    try:
        import fitz
    except ImportError:
        print("ERROR: pip install pymupdf")
        return 1

    doc = fitz.open(str(pdf_path))
    try:
        for pn in pages:
            if pn < 1 or pn > len(doc):
                print(f"  Pagina {pn} fuera de rango")
                continue
            page = doc[pn - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
            img_arr = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            images[pn] = img
            print(f"  OK Pagina {pn}: {img.shape[1]}x{img.shape[0]}")
    finally:
        doc.close()

    if not images:
        print("ERROR: No se pudieron extraer imagenes")
        return 1

    all_results = {}  # page_num -> {engine -> [blocks]}

    # ─── EasyOCR (GPU) ───────────────────────────────
    print("\n" + "-" * 60)
    print("EASYOCR (GPU)")
    print("-" * 60)
    try:
        import easyocr
        print("  Cargando EasyOCR...")
        t0 = time.time()
        reader = easyocr.Reader(["es", "en"], gpu=True, verbose=False)
        print(f"  EasyOCR listo en {time.time()-t0:.1f}s")

        for pn in sorted(images.keys()):
            blocks, elapsed = run_easyocr(reader, images[pn], pn)
            if pn not in all_results:
                all_results[pn] = {}
            all_results[pn]["easyocr"] = blocks
            print(f"\n  Pagina {pn} ({elapsed:.2f}s): {len(blocks)} bloques")
            for b in blocks[:10]:
                print(f"    [{b['conf']:.2f}] {b['text'][:60]}")
            if len(blocks) > 10:
                print(f"    ... y {len(blocks)-10} mas")
    except Exception as e:
        print(f"  ERROR EasyOCR: {e}")

    # ─── RapidOCR (ONNX) ────────────────────────────
    print("\n" + "-" * 60)
    print("RAPIDOCR (ONNX - CPU)")
    print("-" * 60)
    try:
        from rapidocr_onnxruntime import RapidOCR
        print("  Cargando RapidOCR...")
        t0 = time.time()
        rapid_engine = RapidOCR()
        print(f"  RapidOCR listo en {time.time()-t0:.1f}s")

        for pn in sorted(images.keys()):
            blocks, elapsed = run_rapidocr(rapid_engine, images[pn], pn)
            if pn not in all_results:
                all_results[pn] = {}
            all_results[pn]["rapidocr"] = blocks
            print(f"\n  Pagina {pn} ({elapsed:.2f}s): {len(blocks)} bloques")
            for b in blocks[:10]:
                print(f"    [{b['conf']:.2f}] {b['text'][:60]}")
            if len(blocks) > 10:
                print(f"    ... y {len(blocks)-10} mas")
    except Exception as e:
        print(f"  ERROR RapidOCR: {e}")

    # ─── Comparacion ─────────────────────────────────
    print("\n" + "=" * 60)
    print("COMPARACION")
    print("=" * 60)

    engines_detected = set()
    for pn_results in all_results.values():
        engines_detected.update(pn_results.keys())

    totals = {e: 0 for e in engines_detected}
    only_counts = {}
    both_counts = {}

    for pn in sorted(all_results.keys()):
        print(f"\n  Pagina {pn}:")

        for e in engines_detected:
            blocks = all_results[pn].get(e, [])
            print(f"    {e}: {len(blocks)} bloques")

        if len(engines_detected) >= 2:
            engines_list = sorted(engines_detected)
            e1, e2 = engines_list[0], engines_list[1]
            texts1 = {normalize_text(b["text"]) for b in all_results[pn].get(e1, [])}
            texts2 = {normalize_text(b["text"]) for b in all_results[pn].get(e2, [])}
            common = texts1 & texts2
            only1 = texts1 - texts2
            only2 = texts2 - texts1

            print(f"    Coinciden: {len(common)}")
            print(f"    Solo {e1}: {len(only1)}")
            print(f"    Solo {e2}: {len(only2)}")

            if only2:
                print(f"    Solo {e2} detecto:")
                for t in sorted(only2)[:5]:
                    print(f"      - {t[:60]}")

            totals[e1] += len(texts1)
            totals[e2] += len(texts2)
            only_counts.setdefault(e1, 0)
            only_counts.setdefault(e2, 0)
            only_counts[e1] += len(only1)
            only_counts[e2] += len(only2)

    # ─── Resumen ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)

    if len(engines_detected) >= 2:
        engines_list = sorted(engines_detected)
        e1, e2 = engines_list[0], engines_list[1]
        print(f"  {e1}: {totals[e1]} bloques totales")
        print(f"  {e2}: {totals[e2]} bloques totales")
        print(f"  Solo {e1}: {only_counts[e1]}")
        print(f"  Solo {e2}: {only_counts[e2]}")

        if totals[e1] > 0 or totals[e2] > 0:
            denom = max(totals[e1], totals[e2])
            mejora = ((only_counts[e2] - only_counts[e1]) / denom) * 100
            print(f"\n  Mejora {e2} vs {e1}: {mejora:+.1f}%")

        if only_counts[e2] > only_counts[e1]:
            print(f"\n  >> {e2} complementa mejor a {e1}")
        elif only_counts[e1] > only_counts[e2]:
            print(f"\n  >> {e1} sigue siendo mejor que {e2}")
        else:
            print(f"\n  >> Rendimiento similar")
    else:
        for e in engines_detected:
            print(f"  {e}: {totals[e]} bloques totales")

    return 0

if __name__ == "__main__":
    sys.exit(main())
