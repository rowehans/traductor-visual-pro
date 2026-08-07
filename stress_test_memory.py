"""
stress_test_memory.py — Prueba de estrés para detectar memory leaks.
Procesa N páginas consecutivas EN PARALELO y reporta errores HTTP + memoria.

Workers simultáneos: 4 (el servidor serializa OCR internamente con semáforo,
pero inpaint/translate se overlapan entre requests paralelos).
"""
import os, sys, time, json, base64, gc, psutil, concurrent.futures, threading, hashlib
sys.stdout.reconfigure(encoding='utf-8')

import fitz, cv2, numpy as np, requests
from PIL import Image
from io import BytesIO

PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
API_URL = "http://127.0.0.1:5174"
# DOC_ID (sesión 126): scope por documento de los caches de decisión — mismo
# hash que process_all_pages.py para que el stress test no contamine el scope
# de otros capítulos procesados en el mismo servidor.
DOC_ID = hashlib.md5(os.path.basename(PDF_PATH).encode("utf-8")).hexdigest()[:12]
ZOOM = 1.5
TARGET = "en"
SOURCE = "es"
TIMEOUT = 90
NUM_PAGES = 50
MAX_WORKERS = 4

# Pre-load PDF
doc = fitz.open(PDF_PATH)
total_pages = min(NUM_PAGES, len(doc))

print(f"STRESS TEST: {total_pages} paginas en paralelo ({MAX_WORKERS} workers)")
print(f"{'Pag':>5s} {'Status':>7s} {'Tiempo':>7s}")
print("-" * 30)

_lock = threading.Lock()
errors = 0
success = 0
times = []
mem_before = psutil.Process().memory_info().rss / 1024 / 1024

def render_page(pg_idx: int) -> tuple[int, str | None]:
    """Renderiza una página del PDF y retorna (page_num, b64)."""
    try:
        page = doc[pg_idx]
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
        img = Image.open(BytesIO(pix.tobytes('png')))
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.png', img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        b64 = 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()
        del img, img_cv, buf, pix, page
        return pg_idx + 1, b64
    except Exception as e:
        print(f"  ERROR render page {pg_idx+1}: {e}")
        return pg_idx + 1, None

def process_page(pg_idx: int) -> dict:
    """Renderiza + envía una página al servidor. Retorna dict con resultados."""
    page_num, b64 = render_page(pg_idx)
    if b64 is None:
        with _lock:
            print(f"{page_num:5d}   ERROR    render   -      -")
        return {"page": page_num, "ok": False, "error": "render_error", "time": 0}

    t0 = time.time()
    try:
        # Fase 4: el default del endpoint es 'fusion' (puede disparar el daemon
        # U-OCR, ~1-8 min/pág) — el stress test mide MEMORIA, no calidad, así
        # que fija el modo rápido y determinista para no alargar el run.
        resp = requests.post(f"{API_URL}/api/process-page",
            json={'image': b64, 'target': TARGET, 'source': SOURCE,
                  'ocr_mode': 'easyocr', 'doc_id': DOC_ID},
            timeout=TIMEOUT)
        elapsed = time.time() - t0
    except Exception as e:
        elapsed = time.time() - t0
        with _lock:
            print(f"{page_num:5d}   ERROR  {elapsed:6.1f}s   {str(e)[:50]}")
        return {"page": page_num, "ok": False, "error": str(e), "time": elapsed}

    del b64
    gc.collect()

    mem_now = psutil.Process().memory_info().rss / 1024 / 1024
    mem_delta = mem_now - mem_before

    if resp.status_code == 200:
        with _lock:
            print(f"{page_num:5d}      OK  {elapsed:6.1f}s    -   {mem_now:7.1f}MB ({mem_delta:+.1f})")
        return {"page": page_num, "ok": True, "time": elapsed}
    else:
        with _lock:
            print(f"{page_num:5d}  ERR{resp.status_code} {elapsed:6.1f}s    -   {mem_now:7.1f}MB ({mem_delta:+.1f})")
            print(f"  ERROR: {resp.text[:200]}")
        return {"page": page_num, "ok": False, "error": f"HTTP {resp.status_code}", "time": elapsed}


# ─── Procesar páginas en paralelo ────────────────────────────────
results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="stress") as exec:
    futures = {exec.submit(process_page, i): i for i in range(total_pages)}
    for future in concurrent.futures.as_completed(futures, timeout=TIMEOUT):
        results.append(future.result())

doc.close()

# ─── Consolidar resultados ───────────────────────────────────────
mem_after = psutil.Process().memory_info().rss / 1024 / 1024
mem_growth = mem_after - mem_before

success = sum(1 for r in results if r["ok"])
errors = sum(1 for r in results if not r["ok"])
times = [r["time"] for r in results if r["ok"]]
avg_time = sum(times) / max(len(times), 1) if times else 0.0

print(f"\nRESULTADOS:")
print(f"  Exitosas: {success}/{total_pages}")
print(f"  Errores:  {errors}/{total_pages}")
print(f"  Tiempo promedio: {avg_time:.1f}s" if times else "  N/A")
print(f"  Memoria inicial: {mem_before:.1f}MB")
print(f"  Memoria final:   {mem_after:.1f}MB")
print(f"  Crecimiento:     {mem_growth:+.1f}MB")
if errors > 0:
    print(f"\n  {errors} errores detectados - hay memory leak!")
else:
    print(f"\n  Sin errores - memoria estable")

# JSON estructurado para run_ci.py
print()
print(json.dumps({
    "__stress_result__": True,
    "success": success,
    "errors": errors,
    "total": total_pages,
    "avg_time_s": round(avg_time, 1),
    "mem_before_mb": round(mem_before, 1),
    "mem_after_mb": round(mem_after, 1),
    "mem_growth_mb": round(mem_growth, 1),
    "leak_detected": errors > 0 or mem_growth > 100,
}))
