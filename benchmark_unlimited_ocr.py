"""
benchmark_unlimited_ocr.py — Benchmark REAL: EasyOCR (GPU) vs Unlimited-OCR (CPU)
sobre la página 2 del Capítulo 43.

Mide:
  - Tiempo de OCR por motor (excluye carga de modelos, ya precargados)
  - Nº de bloques y caracteres detectados
  - Solapamiento de texto normalizado (coincidencias / solo uno)
  - Precisión de caracteres entre motores (CER 1-way)

Uso:
  env\\Scripts\\python.exe benchmark_unlimited_ocr.py
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np

PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
PAGE = 2
PAGE_IMAGE = "benchmark_page2.png"
UOCR_RESULT = "uocr_result.json"
UOCR_PY = "run_unlimited_ocr.py"
UOCR_PYTHON = str(Path("env_uocr/Scripts/python.exe").resolve())

LANG = ["es", "en"]


def normalize_text(t: str) -> str:
    """Normaliza: minúsculas, sin acentos, espacios colapsados, quita markdown/URLs."""
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    # quitar refs de imagen del markdown de Unlimited-OCR
    t = re.sub(r"!\[\]\([^)]*\)", "", t)
    t = re.sub(r"<\|[^|]*\|>", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate (Levenshtein a nivel de caracteres, normalizado)."""
    ref, hyp = list(reference), list(hypothesis)
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
    reader = easyocr.Reader(LANG, gpu=True, verbose=False)
    t0 = time.time()
    results = reader.readtext(img, detail=1, paragraph=False, min_size=6,
                              text_threshold=0.15, low_text=0.10, canvas_size=2500)
    elapsed = time.time() - t0
    blocks = [{"text": str(t).strip(), "conf": float(c)}
              for b, t, c in results if str(t).strip() and c >= 0.08]
    return blocks, elapsed


def run_unlimited_ocr(image_path: str, max_length: int = 4096):
    cmd = [UOCR_PYTHON, UOCR_PY, image_path, UOCR_RESULT, str(max_length)]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    wall = time.time() - t0
    if proc.returncode != 0:
        print("STDERR de Unlimited-OCR:", proc.stderr[-3000:])
        raise RuntimeError(f"Unlimited-OCR falló (exit {proc.returncode})")
    with open(UOCR_RESULT, encoding="utf-8") as f:
        data = json.load(f)
    # "infer_s" ya mide solo la inferencia; wall incluye subprocess overhead
    return data, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=4096,
                    help="max_length para Unlimited-OCR (CPU: 4096 razonable)")
    ap.add_argument("--skip-uocr", action="store_true")
    args = ap.parse_args()

    print("=" * 72)
    print("BENCHMARK REAL: EasyOCR (GPU) vs Unlimited-OCR (CPU)")
    print(f"PDF: {PDF_PATH} | Página: {PAGE}")
    print("=" * 72)

    # 1) Render página
    img = render_pdf_page(PDF_PATH, PAGE)
    cv2.imwrite(PAGE_IMAGE, img)
    h, w = img.shape[:2]
    print(f"\n[render] Página {PAGE}: {w}x{h} -> {PAGE_IMAGE}")

    # 2) EasyOCR (GPU)
    print("\n--- EasyOCR (GPU) ---")
    blocks_e, t_e = run_easyocr(img)
    text_e = " ".join(b["text"] for b in blocks_e)
    print(f"  Bloques: {len(blocks_e)} | Caracteres: {len(text_e)} | Tiempo: {t_e:.2f}s")
    for b in blocks_e[:12]:
        print(f"    [{b['conf']:.2f}] {b['text'][:70]}")

    # 3) Unlimited-OCR (CPU)
    results = {}
    if args.skip_uocr:
        print("\n--- Unlimited-OCR (CPU) — OMITIDO (--skip-uocr) ---")
        uocr_text = ""
        t_u = 0.0
    else:
        print("\n--- Unlimited-OCR (CPU, bf16) ---")
        data, wall = run_unlimited_ocr(PAGE_IMAGE, args.max_length)
        t_u = data["infer_s"]
        uocr_text = data["text"]
        print(f"  Tiempo inferencia: {t_u:.1f}s (wall {wall:.1f}s)")
        print(f"  Caracteres crudos: {len(uocr_text)}")
        # Desglosar el markdown: párrafos que son texto
        lines = [l.strip() for l in uocr_text.splitlines() if l.strip() and not l.strip().startswith("![]")]
        print(f"  Líneas de texto: {len(lines)}")
        for l in lines[:12]:
            print(f"    {l[:70]}")

    # 4) Comparación
    print("\n" + "=" * 72)
    print("COMPARACIÓN")
    print("=" * 72)

    ne = normalize_text(text_e)
    nu = normalize_text(uocr_text)

    print(f"\n  Tiempo:      EasyOCR {t_e:.2f}s | Unlimited-OCR {t_u:.1f}s | "
          f"ratio {t_u/max(t_e,1e-6):.1f}x más lento")
    print(f"  Texto chars: EasyOCR {len(text_e)} | Unlimited-OCR {len(uocr_text)}")

    # tokens (palabras) de cada motor
    words_e = set(ne.split())
    words_u = set(nu.split())
    common = words_e & words_u
    only_e = words_e - words_u
    only_u = words_u - words_e
    print(f"\n  Palabras únicas detectadas:")
    print(f"    EasyOCR:        {len(words_e)}")
    print(f"    Unlimited-OCR:  {len(words_u)}")
    print(f"    Coinciden:      {len(common)}")
    print(f"    Solo EasyOCR:   {len(only_e)}")
    print(f"    Solo U-OCR:     {len(only_u)}")
    if only_u:
        print(f"    Solo U-OCR detectó ej.: {sorted(only_u)[:12]}")

    # Precisión de caracteres: usamos el otro motor como referencia (CER mutuo)
    if text_e and uocr_text:
        cer_using_easy_as_ref = cer(text_e, uocr_text)
        cer_using_uocr_as_ref = cer(uocr_text, text_e)
        acc_e = max(0.0, 1.0 - cer_using_easy_as_ref)
        acc_u = max(0.0, 1.0 - cer_using_uocr_as_ref)
        print(f"\n  CER (EasyOCR como referencia):      {cer_using_easy_as_ref:.3f} "
              f"(precisión {acc_e*100:.1f}%)")
        print(f"  CER (Unlimited-OCR como referencia): {cer_using_uocr_as_ref:.3f} "
              f"(precisión {acc_u*100:.1f}%)")

    results = {
        "page": PAGE,
        "easyocr": {
            "time_s": round(t_e, 3), "blocks": len(blocks_e),
            "chars": len(text_e), "words": len(words_e),
            "text": text_e,
        },
        "unlimited_ocr": {
            "time_s": round(t_u, 2), "chars": len(uocr_text),
            "words": len(words_u), "text": uocr_text,
        },
        "overlap": {
            "common_words": len(common),
            "only_easyocr": len(only_e),
            "only_uocr": len(only_u),
        },
    }
    if text_e and uocr_text:
        results["cer"] = {
            "cer_easyocr_ref": round(cer_using_easy_as_ref, 4),
            "cer_uocr_ref": round(cer_using_uocr_as_ref, 4),
        }
    with open("benchmark_unlimited_ocr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResultados guardados en benchmark_unlimited_ocr_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
