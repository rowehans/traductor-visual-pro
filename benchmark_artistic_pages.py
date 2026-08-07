"""
benchmark_artistic_pages.py — Benchmark REAL GPU 4-bit: EasyOCR vs Unlimited-OCR
sobre las páginas ARTÍSTICAS 3, 11 y 12 del Capítulo 43 (texto estilizado,
donde EasyOCR falla más según benchmark_ocr_tiers.py).

Mide por página:
  - Tiempo de OCR (EasyOCR GPU vs U-OCR 4-bit GPU vía daemon persistente)
  - Nº de bloques / caracteres / palabras detectadas
  - Solapamiento de palabras (coincidencias / solo uno)
  - CER mutuo (precisión de caracteres cruzada)

Uso:
  env\\Scripts\\python.exe benchmark_artistic_pages.py
"""
import json
import os
import re
import sys
import time
import unicodedata

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
PAGES = [3, 11, 12]
SCALE = 2.0
OUT = "benchmark_artistic_pages_results.json"


def normalize_text(t: str) -> str:
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"!\[\]\([^)]*\)", "", t)
    t = re.sub(r"<\|[^|]*\|>", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = list(ref), list(hyp)
    n, m = len(ref), len(hyp)
    if n == 0 and m == 0:
        return 0.0
    if n == 0 or m == 0:
        return 1.0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m] / max(n, m)


def render_pdf_page(pdf_path: str, page_num: int, scale: float = 2.0) -> np.ndarray:
    import fitz
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    img_bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    doc.close()
    return img_bgr


def run_easyocr(img: np.ndarray):
    import easyocr
    reader = easyocr.Reader(["es", "en"], gpu=True, verbose=False)
    t0 = time.time()
    results = reader.readtext(img, detail=1, paragraph=False, min_size=6,
                              text_threshold=0.15, low_text=0.10, canvas_size=2500)
    elapsed = time.time() - t0
    blocks = [{"text": str(t).strip(), "conf": float(c)}
              for b, t, c in results if str(t).strip() and c >= 0.08]
    return blocks, elapsed


def run_uocr(image_path: str, max_length: int = 4096):
    """U-OCR vía daemon persistente (espera ready internamente)."""
    import uocr_client
    res = uocr_client.process_page(image_path, max_length=max_length, wait_timeout_s=1500)
    if res.get("error"):
        raise RuntimeError(res["error"])
    # Bloques de texto detectados (con coordenadas), filtrando solo tipo "image"
    blocks = [b for b in res.get("blocks", []) if b.get("text") and b.get("type") != "image"]
    return blocks, res.get("infer_s", 0.0)


def main() -> int:
    print("=" * 72)
    print("BENCHMARK PÁGINAS ARTÍSTICAS: EasyOCR (GPU) vs Unlimited-OCR (GPU 4-bit)")
    print(f"PDF: {PDF_PATH} | Páginas: {PAGES}")
    print("=" * 72)

    results = {"pages": [], "summary": {}}
    all_words_e, all_words_u = set(), set()

    for page_no in PAGES:
        print(f"\n--- Página {page_no} ---")
        img = render_pdf_page(PDF_PATH, page_no, SCALE)
        png = f"benchmark_page{page_no}.png"
        cv2.imwrite(png, img)
        h, w = img.shape[:2]
        print(f"[render] {w}x{h} -> {png}")

        # EasyOCR (GPU)
        blocks_e, t_e = run_easyocr(img)
        text_e = " ".join(b["text"] for b in blocks_e)
        print(f"[EasyOCR] {len(blocks_e)} bloques | {len(text_e)} chars | {t_e:.2f}s")
        for b in blocks_e[:10]:
            print(f"    [{b['conf']:.2f}] {b['text'][:65]}")

        # Unlimited-OCR (GPU 4-bit vía daemon)
        blocks_u, t_u = run_uocr(png)
        text_u = " ".join(b["text"] for b in blocks_u)
        print(f"[U-OCR ] {len(blocks_u)} bloques | {len(text_u)} chars | {t_u:.1f}s")
        for b in blocks_u[:10]:
            print(f"    [{b['type']}] {b['text'][:65]}")

        # Comparación
        ne, nu = normalize_text(text_e), normalize_text(text_u)
        we, wu = set(ne.split()), set(nu.split())
        common = we & wu
        all_words_e |= we
        all_words_u |= wu

        page_res = {
            "page": page_no,
            "easyocr": {"time_s": round(t_e, 2), "blocks": len(blocks_e),
                        "chars": len(text_e), "words": len(we), "text": text_e},
            "uocr": {"time_s": round(t_u, 2), "blocks": len(blocks_u),
                     "chars": len(text_u), "words": len(wu), "text": text_u},
            "overlap": {"common": len(common), "only_easy": len(we - wu),
                        "only_uocr": len(wu - we)},
        }
        if text_e and text_u:
            page_res["cer"] = {
                "cer_easy_ref": round(cer(text_e, text_u), 4),
                "cer_uocr_ref": round(cer(text_u, text_e), 4),
            }
            print(f"[compare] coinciden {len(common)} | solo EasyOCR {len(we - wu)} | "
                  f"solo U-OCR {len(wu - we)}")
            print(f"[compare] CER (Easy ref): {page_res['cer']['cer_easy_ref']:.3f} | "
                  f"CER (U-OCR ref): {page_res['cer']['cer_uocr_ref']:.3f}")
        results["pages"].append(page_res)
        print(f"[Tiempo] EasyOCR {t_e:.2f}s | U-OCR {t_u:.1f}s | ratio {t_u/max(t_e,1e-6):.1f}x")

    # Resumen agregado
    total_e = sum(p["easyocr"]["time_s"] for p in results["pages"])
    total_u = sum(p["uocr"]["time_s"] for p in results["pages"])
    results["summary"] = {
        "total_easyocr_s": round(total_e, 2),
        "total_uocr_s": round(total_u, 2),
        "speedup_uocr_over_easyocr_x": round(total_u / max(total_e, 1e-6), 2),
        "unique_words_easy": len(all_words_e),
        "unique_words_uocr": len(all_words_u),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*72}\nResultados guardados en {OUT}")
    print(f"TOTAL: EasyOCR {total_e:.1f}s | U-OCR {total_u:.1f}s ({total_u/max(total_e,1e-6):.1f}x)")
    print(f"Palabras únicas agregadas: EasyOCR {len(all_words_e)} | U-OCR {len(all_words_u)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
