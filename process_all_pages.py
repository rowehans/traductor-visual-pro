"""
process_all_pages.py — Procesa TODAS las 128 páginas del PDF con checkpointing.
AHORA EN PARALELO: usa ThreadPoolExecutor para enviar múltiples páginas al
servidor simultáneamente. El servidor serializa el OCR internamente con
semáforo, pero el inpainting/translate puede overlapearse entre requests.

Parametros:
  --ocr-mode auto|easyocr    Modo OCR (default: easyocr — solo EasyOCR GPU)
  --workers N                Workers paralelos (default: 3)
  --prefilter                Limpieza morfologica pre-OCR en todas las paginas
                             (lineas de escaneo, speckle, margenes). Agrega ~0.2s/pag

Tiempo estimado real (128 páginas, 3 workers, benchmark Jul 2026):
  easyocr: ~15-20 min  ← DEFAULT (EasyOCR GPU ~0.88s/pág, sin fallback)
  auto:    ~60-75 min  (con fallback CLAHE, mejor cobertura pero mas lento)

Recomendacion: usar --ocr-mode easyocr (default) para velocidad. Usar auto
solo si hay páginas donde EasyOCR no detecta texto.
"""
import os, sys, time, json, base64, threading, concurrent.futures, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import fitz, requests

# ─── Argumentos CLI ──────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Procesar PDF completo con OCR y traducción")
_parser.add_argument(
    "--ocr-mode",
    choices=["auto", "easyocr"],
    default="easyocr",
    help="Modo OCR: easyocr (default, solo EasyOCR GPU), auto (EasyOCR GPU + CLAHE fallback)"
)
_parser.add_argument(
    "--workers",
    type=int,
    default=3,
    help="Workers paralelos (default: 3). 4+ satura el semaforo OCR"
)
_parser.add_argument(
    "--prefilter",
    action="store_true",
    default=False,
    help="Aplica limpieza morfologica pre-OCR en TODAS las paginas "
         "(elimina lineas de escaneo, speckle y artefactos de margen). "
         "Agrega ~0.2s por pagina pero mejora deteccion en escaneos ruidosos"
)
_args, _ = _parser.parse_known_args()
OCR_MODE: str = _args.ocr_mode
MAX_WORKERS: int = _args.workers
PREFILTER: bool = _args.prefilter

PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente _ Olympus Scanlation_compressed.pdf"
CHECKPOINT_FILE = "resultados_progreso.json"
API_URL = "http://127.0.0.1:5174"
ZOOM = 1.2                    # 1.5→1.2: -36% pixels, misma calidad visual
TARGET = "en"
SOURCE = "es"
TIMEOUT_PER_PAGE = 120  # 120s por página (páginas densas toman ~4-5s con EasyOCR GPU, sin CLAHE)
MAX_RETRIES = 2          # reintentos en caso de timeout
RETRY_DELAY = 5          # segundos de espera entre reintentos
CHECKPOINT_EVERY = 10    # save every N pages
# MAX_WORKERS se obtiene de --workers CLI (default: 3)

# ── Sesión HTTP compartida (connection pooling, keep-alive) ────
_http_session: requests.Session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=10,
    pool_maxsize=10,
    pool_block=False,
)
_http_session.mount('http://', _adapter)
_http_session.mount('https://', _adapter)

# ── Thread-safe estado compartido ────────────────────────────────
_lock = threading.Lock()
_rendered_queue: list[tuple[int, str | None, float]] = []  # (page_num, b64_or_None, render_time)
_rendered_event = threading.Event()
_total_pages = 0

# ── checkpoint helpers (thread-safe con Lock) ────────────────────
def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            cp = json.load(f)
        with _lock:
            print(f"[checkpoint] Cargado: {len(cp.get('pages_done',[]))} páginas ya procesadas")
        return cp
    except Exception as e:
        print(f"[checkpoint] Error cargando: {e}, empezando de cero")
        return None

def save_checkpoint(cp):
    tmp = CHECKPOINT_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CHECKPOINT_FILE)

# ── Render thread: produce imágenes y las encola ─────────────────
def render_worker(doc, total_pages, pages_done):
    """
    Renderiza páginas en un hilo y las encola para los API workers.
    
    Optimizado: pix.tobytes('png') da PNG directo, solo falta base64.
    Antes: pix→PNG→PIL→numpy→cv2→imencode→base64 (5 conversiones intermedias).
    Ahora: pix→PNG→base64 (2 pasos, sin PIL/numpy/cv2).
    """
    for pg_idx in range(total_pages):
        page_num = pg_idx + 1
        if page_num in pages_done:
            continue
        t0 = time.time()
        try:
            page = doc[pg_idx]
            pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
            # PNG directo desde fitz — sin PIL/numpy/cv2 intermedios
            png_bytes = pix.tobytes('png')
            b64 = 'data:image/png;base64,' + base64.b64encode(png_bytes).decode()
            del pix, page
            render_t = time.time() - t0

            with _lock:
                _rendered_queue.append((page_num, b64, render_t))
            _rendered_event.set()
        except Exception as e:
            with _lock:
                print(f"  Render error pág {page_num}: {e}")
                _rendered_queue.append((page_num, None, 0))
                _rendered_event.set()

    # Señal de fin
    with _lock:
        _rendered_queue.append((None, None, 0))  # centinela
    _rendered_event.set()

def get_next_page(timeout=5):
    """Espera hasta que haya una página renderizada disponible."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _lock:
            if _rendered_queue:
                return _rendered_queue.pop(0)
        _rendered_event.wait(0.1)
        _rendered_event.clear()
    return (None, None, 0)


# ─── init ────────────────────────────────────────────────────────
# Verificar servidor (usando sesión HTTP compartida)
try:
    r = _http_session.get(f"{API_URL}/api/health", timeout=5)
    if not r.json().get("ok"):
        print("ERROR: Servidor no saludable"); sys.exit(1)
    print(f"Servidor OK - {r.json().get('version','?')}")
except Exception as e:
    print(f"ERROR: No se puede conectar: {e}"); sys.exit(1)

# Abrir PDF
doc = fitz.open(PDF_PATH)
_total_pages = len(doc)
print(f"\nPDF: {os.path.basename(PDF_PATH)} ({_total_pages} páginas)")
pf_label = "+prefilter" if PREFILTER else ""
print(f"Idioma: {SOURCE} → {TARGET} | OCR: {OCR_MODE}{pf_label} | Workers: {MAX_WORKERS} | Inicio: {time.strftime('%H:%M:%S')}")
print("=" * 80)

# ── estado (cargar checkpoint) ──────────────────────────────────
checkpoint = load_checkpoint()
total_pages = _total_pages
if checkpoint and checkpoint.get("total_pages") == total_pages:
    pages_done = set(checkpoint["pages_done"])
    results = checkpoint["results"]
    page_times = checkpoint["page_times"]
    stats = checkpoint["stats"]
else:
    pages_done = set()
    results = []
    page_times = []
    stats = {
        "total_blocks_found": 0,
        "total_blocks_translated": 0,
        "pages_with_text": 0,
        "pages_translated": 0,
        "pages_empty": 0,
        "pages_error": 0,
    }

# Contador de páginas procesadas (para checkpoint cada N)
_pages_processed_since_checkpoint = 0

def build_checkpoint():
    return {
        "total_pages": total_pages,
        "pages_done": sorted(pages_done),
        "results": results,
        "page_times": page_times,
        "stats": stats,
        "ocr_mode": OCR_MODE,
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

def procesar_pagina(page_num: int, b64: str | None, render_t: float):
    """Procesa una página contra el servidor y actualiza estado."""
    global _pages_processed_since_checkpoint
    if b64 is None:
        with _lock:
            print(f"  Pág {page_num:3d}: ERROR render previo")
            stats["pages_error"] += 1
            results.append({"page": page_num, "status": "render_error", "blocks": 0, "translated": 0, "time": render_t})
            pages_done.add(page_num)
            _pages_processed_since_checkpoint += 1
        return
    t0 = time.time()
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = _http_session.post(f"{API_URL}/api/process-page",
                json={'image': b64, 'target': TARGET, 'source': SOURCE,
                      'ocr_mode': OCR_MODE, 'prefilter': PREFILTER},
                timeout=TIMEOUT_PER_PAGE)
            elapsed = time.time() - t0
            break  # exito, salir del bucle de reintentos
        except requests.Timeout:
            elapsed = time.time() - t0
            with _lock:
                print(f"  Pág {page_num:3d}: TIMEOUT (intento {attempt+1}/{1+MAX_RETRIES}, {elapsed:.0f}s)")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (attempt + 1)  # backoff: 5s, 10s
                with _lock:
                    print(f"  Pág {page_num:3d}: Reintentando en {delay}s...")
                time.sleep(delay)
            else:
                with _lock:
                    print(f"  Pág {page_num:3d}: TIMEOUT definitivo (>={TIMEOUT_PER_PAGE}s x {1+MAX_RETRIES} intentos)")
                    stats["pages_error"] += 1
                    results.append({"page": page_num, "status": "timeout", "blocks": 0, "translated": 0, "time": elapsed})
                    pages_done.add(page_num)
                    _pages_processed_since_checkpoint += 1
                return
        except Exception as e:
            with _lock:
                print(f"  Pág {page_num:3d}: ERROR conexión (intento {attempt+1}): {e}")
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (attempt + 1)
                time.sleep(delay)
            else:
                with _lock:
                    print(f"  Pág {page_num:3d}: ERROR conexión definitivo: {e}")
                    stats["pages_error"] += 1
                    results.append({"page": page_num, "status": "conn_error", "blocks": 0, "translated": 0, "time": elapsed})
                    pages_done.add(page_num)
                    _pages_processed_since_checkpoint += 1
                return

    if resp.status_code != 200:
        with _lock:
            print(f"  Pág {page_num:3d}: ERROR HTTP {resp.status_code} ({elapsed:.1f}s)")
            stats["pages_error"] += 1
            results.append({"page": page_num, "status": f"http_{resp.status_code}", "blocks": 0, "translated": 0, "time": elapsed})
            pages_done.add(page_num)
            _pages_processed_since_checkpoint += 1
        return

    data = resp.json()
    blocks = data.get("blocks", [])
    n_blocks = len(blocks)
    n_translated = sum(1 for b in blocks if b.get("source","") != b.get("translated",""))

    with _lock:
        stats["total_blocks_found"] += n_blocks
        stats["total_blocks_translated"] += n_translated
        if n_blocks > 0:
            stats["pages_with_text"] += 1
            if n_translated > 0:
                stats["pages_translated"] += 1
        else:
            stats["pages_empty"] += 1

        status = ("OK" if n_translated == n_blocks and n_blocks > 0 else
                  "PARCIAL" if n_translated > 0 else
                  "SIN_TRAD" if n_blocks > 0 else "VACIO")
        if attempt > 0:
            status += f"_R{attempt}"

        pct = (len(pages_done) + 1) / total_pages * 100
        line = f"  Pág {page_num:3d} [{pct:3.0f}%] T{elapsed:5.1f}s R{render_t:4.1f}s | {n_blocks} bloq | {n_translated} trad | {status}"
        if n_blocks > 0 and n_translated > 0:
            first = blocks[0]
            src = first.get("source","")[:35]
            trl = first.get("translated","")[:35]
            if src != trl:
                line += f" | {src}→{trl}"
        print(line)

        results.append({"page": page_num, "status": status, "blocks": n_blocks,
                         "translated": n_translated, "time": elapsed,
                         "texts": [{"src": b.get("source",""), "tgt": b.get("translated","")} for b in blocks]})
        pages_done.add(page_num)
        page_times.append(elapsed)
        _pages_processed_since_checkpoint += 1

        if _pages_processed_since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(build_checkpoint())
            _pages_processed_since_checkpoint = 0


# ─── Lanzar render worker + N API workers ───────────────────────
render_thread = threading.Thread(
    target=render_worker, args=(doc, total_pages, pages_done), daemon=True
)
render_thread.start()

api_workers = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="page_api")
futures = []
pages_en_cola = 0

# Consumir páginas renderizadas y lanzar API calls en paralelo
while True:
    page_num, b64, render_t = get_next_page(timeout=60)
    if page_num is None:  # centinela de fin
        break
    pages_en_cola += 1
    fut = api_workers.submit(procesar_pagina, page_num, b64, render_t)
    futures.append(fut)

# Esperar a que todas las API calls terminen
concurrent.futures.wait(futures)
api_workers.shutdown()
doc.close()

# Guardar checkpoint final
save_checkpoint(build_checkpoint())

# ═══════════════ REPORTE FINAL ══════════════════════════════════
print("\n" + "=" * 80)
print("REPORTE FINAL - TRADUCCIÓN DE PDF COMPLETO")
print("=" * 80)
print(f"\nArchivo: {os.path.basename(PDF_PATH)}")
print(f"Total páginas: {total_pages}")
total_time = sum(page_times)
print(f"Tiempo total (suma): {total_time:.0f}s ({total_time/60:.1f} min)")
print(f"Promedio por página: {total_time/max(len(page_times),1):.1f}s")
print(f"Workers simultáneos: {MAX_WORKERS}")
print(f"Modo OCR: {OCR_MODE}")
print()
s = stats
print(f"Páginas con texto detectado: {s['pages_with_text']}")
print(f"  → Traducidas correctamente: {s['pages_translated']}")
print(f"  → Sin traducir:             {s['pages_with_text'] - s['pages_translated']}")
print(f"Páginas sin texto (vacías):   {s['pages_empty']}")
print(f"Páginas con error:            {s['pages_error']}")
print()
print(f"Total bloques detectados:  {s['total_blocks_found']}")
print(f"Total bloques traducidos:  {s['total_blocks_translated']}")
if s['total_blocks_found'] > 0:
    print(f"Tasa de traducción:        {s['total_blocks_translated']/s['total_blocks_found']*100:.1f}%")
print()

# Resumen visual
if s['pages_translated'] > 0:
    print(f"✅ {s['pages_translated']} páginas traducidas correctamente")
if s['pages_with_text'] - s['pages_translated'] > 0:
    print(f"⚠️  {s['pages_with_text'] - s['pages_translated']} páginas con texto sin traducir")
if s['pages_empty'] > 0:
    print(f"ℹ️  {s['pages_empty']} páginas sin texto (viñetas/arte)")
if s['pages_error'] > 0:
    print(f"❌ {s['pages_error']} páginas con error")
    failed = [r["page"] for r in results if r["status"] in ("timeout","render_error","conn_error") or str(r["status"]).startswith("http_")]
    if failed:
        print(f"   Páginas con error: {failed}")

print(f"\nReporte guardado en: {CHECKPOINT_FILE}")
print(f"Fin: {time.strftime('%H:%M:%S')}")
