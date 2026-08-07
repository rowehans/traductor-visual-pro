# -*- coding: utf-8 -*-
"""Benchmark del overhead puro de la fusión SIN U-OCR.

Procesa el PDF nuevo completo (53 págs) en dos modos:
  A) easyocr            → línea base (EasyOCR solo, sin RapidOCR ni merge)
  B) fusion+disable_uocr → fusión real (EasyOCR+RapidOCR+merge) SIN el
                           refuerzo de Unlimited-OCR (trigger anulado)

Mide por página: tiempo, nº de bloques, y los textos OCR detectados
(para comparar calidad/cobertura sin el costo del daemon).

Modos:
  easyocr        → A) EasyOCR solo (con híbrido RapidOCR — el default de la app)
  pure_easyocr   → A') EasyOCR GPU PURO (sin RapidOCR) — línea base real
  fusion         → B) fusion + disable_uocr (EasyOCR+RapidOCR+merge, sin daemon)

Uso:
  python benchmark_fusion_overhead.py            # todos
  python benchmark_fusion_overhead.py easyocr    # solo A
  python benchmark_fusion_overhead.py pure       # solo A'
  python benchmark_fusion_overhead.py fusion     # solo B
"""
import sys, io, os, json, time, base64, urllib.request
import fitz  # PyMuPDF

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

API = "http://127.0.0.1:5174/api/process-page"
PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
OUT = "benchmark_overhead_results.json"

def process_all(mode: str) -> dict:
    doc = fitz.open(PDF_PATH)
    total_pages = doc.page_count
    doc.close()
    per_page = []
    t_total = 0.0
    for pno in range(1, total_pages + 1):
        doc = fitz.open(PDF_PATH)
        pix = doc[pno - 1].get_pixmap(dpi=180)
        doc.close()
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        # pure_easyocr NO es un valor de ocr_mode: se envía como ocr_mode="easyocr"
        # + flag pure_easyocr=True (desactiva el tier híbrido RapidOCR).
        api_mode = "easyocr" if mode == "pure" else mode
        payload = {"image": b64, "ocr_mode": api_mode,
                   "source_lang": "auto", "target_lang": "en"}
        if mode == "fusion":
            payload["disable_uocr"] = True  # medir overhead puro, sin daemon
        if mode == "pure":
            payload["pure_easyocr"] = True  # EasyOCR GPU puro (sin RapidOCR)
        req = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode())
            dt = time.time() - t0
            blocks = data.get("blocks", [])
            texts = [(b.get("source") or b.get("text") or "").strip()
                     for b in blocks]
            texts = [t for t in texts if t]
            per_page.append({"page": pno, "t": round(dt, 1),
                             "nblocks": len(blocks), "texts": texts,
                             "engines": data.get("engines_used", []),
                             "err": data.get("error")})
            t_total += dt
            print(f"  p{pno}: {dt:.1f}s | {len(blocks)} bloques | "
                  f"{data.get('engines_used')} | err={data.get('error')}",
                  flush=True)
        except Exception as e:
            per_page.append({"page": pno, "t": round(time.time() - t0, 1),
                             "nblocks": 0, "texts": [], "engines": [],
                             "err": str(e)})
            print(f"  p{pno}: ERROR {e}", flush=True)
    return {"mode": mode, "pages": len(per_page),
            "t_total": round(t_total, 1), "per_page": per_page}

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    results = {}
    if os.path.exists(OUT):
        with open(OUT, "r", encoding="utf-8") as f:
            results = json.load(f)
    if mode in ("easyocr", "pure", "both", "all"):
        print(f"=== MODO easyocr ({PDF_PATH}) ===", flush=True)
        results["easyocr"] = process_all("easyocr")
    if mode in ("pure", "both", "all"):
        print(f"=== MODO pure_easyocr (sin RapidOCR) ({PDF_PATH}) ===", flush=True)
        results["pure_easyocr"] = process_all("pure")
    if mode in ("fusion", "both", "all"):
        print(f"=== MODO fusion (disable_uocr) ({PDF_PATH}) ===", flush=True)
        results["fusion_nouocr"] = process_all("fusion")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    # Resumen inmediato
    for k, r in results.items():
        ts = [p["t"] for p in r["per_page"]]
        nbs = [p["nblocks"] for p in r["per_page"]]
        print(f"\n[{k}] total={r['t_total']}s media={sum(ts)/len(ts):.1f}s "
              f"bloques_media={sum(nbs)/len(nbs):.1f} "
              f"bloques_total={sum(nbs)}", flush=True)

if __name__ == "__main__":
    main()
