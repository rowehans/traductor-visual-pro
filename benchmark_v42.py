# -*- coding: utf-8 -*-
"""Benchmark v4.2 del modo fusion — trigger selectivo + serialización GPU.

Mide sobre el PDF nuevo (53 págs):
  1. Cuántas páginas disparan U-OCR (trigger selectivo: image>15% O <3 bloques conf<0.2).
  2. Tiempo por página en modo fusion (con/sin refuerzo U-OCR).
  3. Recuperación de diálogo artístico en las páginas que antes se perdían.
"""
import sys, io, time, json, base64, urllib.request

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

API = "http://127.0.0.1:5174/api/process-page"
PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"

import fitz  # PyMuPDF

def process_page(page_no: int, force: bool = False) -> dict:
    doc = fitz.open(PDF_PATH)
    page = doc[page_no - 1]
    pix = page.get_pixmap(dpi=180)
    doc.close()
    b64 = base64.b64encode(pix.tobytes("png")).decode()
    payload = {"image": b64, "ocr_mode": "fusion", "source_lang": "auto",
               "target_lang": "en", "force_uocr": force}
    req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1200) as resp:
            data = json.loads(resp.read().decode())
        dt = time.time() - t0
        blocks = data.get("blocks", [])
        engines = data.get("engines_used", [])
        texts = [b.get("text", "") for b in blocks]
        return {"page": page_no, "t": round(dt, 1), "nblocks": len(blocks),
                "engines": engines, "texts": texts,
                "ocr_engine": data.get("ocr_engine"), "err": data.get("error")}
    except Exception as e:
        return {"page": page_no, "t": round(time.time() - t0, 1),
                "nblocks": 0, "engines": [], "texts": [], "err": str(e)}

if __name__ == "__main__":
    # 1) Probar páginas NORMALES (no deberían disparar U-OCR con trigger selectivo)
    print("=== PÁGINAS NORMALES (modo fusion, trigger selectivo v4.2) ===")
    normal = []
    for p in [1, 2, 7, 15, 30, 45]:
        r = process_page(p)
        normal.append(r)
        print(f"p{p}: {r['t']}s | {r['nblocks']} bloques | engines={r['engines']} | err={r['err']}")

    # 2) Probar páginas ARTÍSTICAS (las que tenían diálogo pintado en el PDF viejo)
    #    En el PDF nuevo el diálogo artístico vive en pág. 5 ("ERA UNA PROPUESTA")
    print("\n=== PÁGINAS ARTÍSTICAS (trigger U-OCR) ===")
    artistic = []
    for p in [3, 5, 11]:
        r = process_page(p)
        artistic.append(r)
        print(f"p{p}: {r['t']}s | {r['nblocks']} bloques | engines={r['engines']} | err={r['err']}")
        for t in r["texts"][:8]:
            print(f"    · {t}")

    uocr_fired = sum(1 for r in normal + artistic if "unlimited" in r["engines"])
    total_pages = len(normal + artistic)
    print(f"\n=== RESUMEN ===")
    print(f"Páginas evaluadas: {total_pages}")
    print(f"U-OCR disparado (trigger selectivo): {uocr_fired}/{total_pages}")
    t_normal = [r['t'] for r in normal]
    t_art = [r['t'] for r in artistic if 'unlimited' in r['engines']]
    if t_normal:
        print(f"Tiempo medio página normal: {sum(t_normal)/len(t_normal):.1f}s")
    if t_art:
        print(f"Tiempo medio página artística con U-OCR: {sum(t_art)/len(t_art):.1f}s")
