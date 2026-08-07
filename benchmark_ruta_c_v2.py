"""benchmark_ruta_c_v2.py — Benchmark de la Ruta C refinada (re-OCR a nivel GLOBO).

Mide si la fusión con bubble re-OCR (detección de globos dentro de paneles
image + EasyOCR 3.5x por globo) recupera el diálogo artístico de las páginas
3 y 12 que EasyOCR full-page pierde. Pasa por el endpoint real /api/process-page
en modo fusion (daemon U-OCR + Ruta C) para validar la integración completa.
"""
import base64
import json
import re
import sys
import time
import unicodedata
import urllib.request

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cv2
import numpy as np
import fitz

PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
PAGES = [3, 12]
API = "http://127.0.0.1:5174/api/process-page"
SCALE = 2.0

# Ground truth del diálogo artístico (de benchmark_ruta_c.py)
REF = {
    3: "INCREIBLE REALMENTE",
    12: "ERA UNA PROPUESTA QUE SOLO PODIA BENEFICIARME PERO",
}


def _norm(t: str) -> str:
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-z0-9 ]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = list(_norm(ref)), list(_norm(hyp))
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
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
    img = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
    img_bgr = cv2.imdecode(img, cv2.IMREAD_COLOR)
    doc.close()
    return img_bgr


def b64_of(img_bgr) -> str:
    ok, buf = cv2.imencode(".png", img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    assert ok
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def call_api(b64: str, force_uocr: bool = True) -> dict:
    # force_uocr=true fuerza el refuerzo U-OCR + Ruta C aunque el trigger no
    # dispare (necesario para páginas que EasyOCR lee "bien" en solitario).
    body = json.dumps({"image": b64, "target": "en", "source": "auto",
                       "ocr_mode": "fusion", "force_uocr": force_uocr}).encode("utf-8")
    req = urllib.request.Request(API, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data, time.time() - t0


def main() -> None:
    out = {"pages": []}
    for pn in PAGES:
        print(f"=== PÁGINA {pn} ===", flush=True)
        img = render_pdf_page(PDF_PATH, pn, SCALE)
        h, w = img.shape[:2]
        print(f"  render {w}x{h}", flush=True)
        data, t = call_api(b64_of(img), force_uocr=True)
        blocks = data.get("blocks", [])
        engines = data.get("engines_used", [])
        srcs = " ".join(b["source"] for b in blocks)
        ref = REF.get(pn, "")
        c = cer(ref, srcs) if ref else None
        print(f"  fusion t={t:.1f}s engines={engines} bloques={len(blocks)}", flush=True)
        for b in blocks[:10]:
            print(f"    ({b['x']},{b['y']},{b['w']}x{b['h']}) conf={b['confidence']:.2f} "
                  f"'{b['source'][:60]}'", flush=True)
        if ref:
            print(f"  CER vs ground truth: {c:.4f} (ref: {ref!r})", flush=True)
        out["pages"].append({
            "page": pn, "time_s": round(t, 1), "engines": engines,
            "blocks": len(blocks), "cer": round(c, 4) if c is not None else None,
            "src_text": srcs[:400],
        })
    with open("benchmark_ruta_c_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nGuardado en benchmark_ruta_c_v2_results.json")


if __name__ == "__main__":
    main()
