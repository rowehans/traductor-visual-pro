"""
process_all_pages.py — Procesa TODAS las 128 páginas del PDF con checkpointing.
AHORA EN PARALELO: usa ThreadPoolExecutor para enviar múltiples páginas al
servidor simultáneamente. El servidor serializa el OCR internamente con
semáforo, pero el inpainting/translate puede overlapearse entre requests.

Parametros:
    --ocr-mode auto|easyocr|fusion
                        Modo OCR (default: fusion — EasyOCR+RapidOCR siempre,
                        Unlimited-OCR solo si la página es difícil)
    --workers N                Workers paralelos (default: 3)
    --prefilter                Limpieza morfologica pre-OCR en todas las paginas
                                (lineas de escaneo, speckle, margenes). Agrega ~0.2s/pag
    --batch-window N           Agrupa N páginas por request via /api/process-page-batch
                                (Fase 1): las páginas que disparan U-OCR van en UN
                                solo infer_multi al daemon. Benchmark real (págs 38-42,
                                daemon caliente, fuerza U-OCR, sesión 113): el batch de
                                4 págs tardó 570.9s (142.7s/pág) vs 5 single 261.9s
                                (52.4s/pág) — con la serialización GPU/degradaación CPU
                                activas, single es ~2.2x más rápido. Batch solo gana en
                                páginas de U-OCR masivo (F5: p5, 39-42, 51-53).
                                Default: 1 (una página por request, recomendado).
    --force-uocr               Fuerza el refuerzo Unlimited-OCR en TODAS las páginas
                                (skip del trigger v4.2). Útil para benchmarks
                                deterministas; requiere el daemon ready.

Tiempo estimado real (53 páginas, benchmark Ago 2026):
    fusion:  ~7.1 min con disable_uocr / ~47 min con U-OCR selectivo (páginas artísticas)
    fusion + --batch-window 4: NO más rápido con daemon caliente (single 270s vs batch
                                587s en 5 págs, sesión 113) — solo gana en páginas con
                                U-OCR masivo. Default se mantiene en 1.
    easyocr: híbrido ~7.25s/pág (EasyOCR+RapidOCR); puro ~3.68s/pág pero 90% menos bloques
    auto:    con fallback CLAHE, mejor cobertura pero mas lento

Recomendacion: usar --ocr-mode fusion (default) para mejor calidad. Usar
--ocr-mode easyocr para máxima velocidad (menos detección).
"""
import os, sys, time, json, base64, threading, concurrent.futures, argparse, hashlib

import fitz, requests

# ─── Argumentos CLI ──────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Procesar PDF completo con OCR y traducción")
_parser.add_argument(
    "--ocr-mode",
    choices=["auto", "easyocr", "fusion"],
    default="fusion",
    help="Modo OCR: fusion (default, EasyOCR+RapidOCR siempre + Unlimited-OCR solo si la "
         "página es difícil), easyocr (solo EasyOCR GPU), auto (EasyOCR GPU + CLAHE fallback)"
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
_parser.add_argument(
    "--batch-window",
    type=int,
    default=1,
    help="Fase 1: agrupa N páginas por request via /api/process-page-batch. "
            "Las páginas que disparan U-OCR van en UN solo infer_multi al daemon. "
            "Benchmark sesión 113 (daemon caliente): batch-window 4 fue MÁS LENTO que "
            "single (587s vs 270s en 5 págs) — solo gana en páginas con U-OCR masivo. "
            "Default: 1 (request por página, recomendado)."
)
_parser.add_argument(
    "--max-pages",
    type=int,
    default=0,
    help="Procesa solo las primeras N páginas (0 = todas). Útil para smoke tests "
            "cortos sin recorrer el capítulo completo."
)
_parser.add_argument(
    "--checkpoint-file",
    type=str,
    default=None,
    help="Archivo de checkpoint a usar. Por defecto se genera uno temporal "
            "(resultados_progreso_YYYYMMDD_HHMM.json): cada corrida usa su propio "
            "archivo y nunca pisa un run previo ni compite con otro proceso. "
            "Usar un archivo explícito permite RESUMIR una corrida anterior."
)
_parser.add_argument(
    "--force-uocr",
    action="store_true",
    default=False,
    help="Fuerza el refuerzo Unlimited-OCR en TODAS las páginas (skip del trigger "
            "v4.2). Útil para benchmarks: elimina el no-determinismo del trigger "
            "y mide la ventaja real de infer_multi compartiendo prefill."
)
_args, _ = _parser.parse_known_args()
OCR_MODE: str = _args.ocr_mode
MAX_WORKERS: int = _args.workers
PREFILTER: bool = _args.prefilter
BATCH_WINDOW: int = max(1, _args.batch_window)
MAX_PAGES: int = max(0, _args.max_pages)
CHECKPOINT_FILE = _args.checkpoint_file
# Sesión 133: el default genera un sufijo temporal (YYYYMMDD_HHMM) — cada
# corrida escribe su PROPIO archivo. Antes era un nombre fijo
# (resultados_progreso.json) y dos procesos o corridas solapaban el MISMO
# archivo, pisándose el checkpoint mutuamente (incidente de la sesión 128:
# dos process_all_pages compitiendo por el mismo checkpoint perdieron una
# corrida completa). Con --checkpoint-file explícito se usa el nombre tal
# cual → el resume sigue funcionando igual.
if CHECKPOINT_FILE is None:
    CHECKPOINT_FILE = f"resultados_progreso_{time.strftime('%Y%m%d_%H%M')}.json"
FORCE_UOCR: bool = _args.force_uocr

PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente.pdf"
API_URL = "http://127.0.0.1:5174"
# DOC_ID (sesión 126): scope por DOCUMENTO de los caches de decisión del
# servidor (trigger + §8.4.1). La sesión 124 midió 94% de colisión de firma
# de layout entre capítulos de la MISMA serie — si procesas el capítulo 47
# después del 43 en el mismo servidor, las páginas del 47 heredarían las
# decisiones del 43 (VLM suprimido en diálogo artístico). Prefijar la clave
# con un hash estable del nombre del PDF aísla cada capítulo. Se envía como
# "doc_id" en los payloads de /api/process-page y /api/process-page-batch.
DOC_ID = hashlib.md5(os.path.basename(PDF_PATH).encode("utf-8")).hexdigest()[:12]
ZOOM = 1.2                    # 1.5→1.2: -36% pixels, misma calidad visual
TARGET = "en"
SOURCE = "es"
TIMEOUT_PER_PAGE = 1800  # 1800s por página: en modo fusion el daemon U-OCR serializa
                         # (una inferencia a la vez, 130-500s c/u) y con workers=2 varias
                         # páginas disparadas se encolan: 2-3 inferencias ≈ 260-1446s.
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

# ── estado por defecto (main() lo sobrescribe desde el checkpoint) ──
# Definidos a nivel de módulo para que procesar_pagina/_registrar_resultado
# sean testeables en aislamiento sin ejecutar el flujo completo.
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
total_pages = 0
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
                        'ocr_mode': OCR_MODE, 'prefilter': PREFILTER,
                        'force_uocr': FORCE_UOCR, 'doc_id': DOC_ID},
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

    # Fix (2026-08-06): el refactor a _registrar_resultado perdió esta línea —
    # sin ella, data no está definida y el worker crasheaba con NameError en
    # silencio (0 páginas registradas aunque el servidor sí procesaba).
    data = resp.json()
    _registrar_resultado(page_num, render_t, elapsed, data.get("blocks", []), attempt)


def _registrar_resultado(page_num: int, render_t: float, elapsed: float,
                         blocks: list, attempt: int = 0) -> None:
    """Actualiza stats/resultados/checkpoint para una página completada.

    Compartido por procesar_pagina (single) y procesar_lote (batch Fase 1).
    """
    global _pages_processed_since_checkpoint
    n_blocks = len(blocks)
    n_translated = sum(1 for b in blocks if b.get("source", "") != b.get("translated", ""))

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

        # total_pages es 0 solo cuando se llama en aislamiento (tests);
        # main() siempre lo setea antes de procesar páginas reales.
        pct = (len(pages_done) + 1) / total_pages * 100 if total_pages else 0
        line = f"  Pág {page_num:3d} [{pct:3.0f}%] T{elapsed:5.1f}s R{render_t:4.1f}s | {n_blocks} bloq | {n_translated} trad | {status}"
        if n_blocks > 0 and n_translated > 0:
            first = blocks[0]
            src = first.get("source", "")[:35]
            trl = first.get("translated", "")[:35]
            if src != trl:
                line += f" | {src}→{trl}"
        print(line)

        results.append({"page": page_num, "status": status, "blocks": n_blocks,
                            "translated": n_translated, "time": elapsed,
                            "texts": [{"src": b.get("source", ""), "tgt": b.get("translated", "")} for b in blocks]})
        pages_done.add(page_num)
        page_times.append(elapsed)
        _pages_processed_since_checkpoint += 1

        if _pages_processed_since_checkpoint >= CHECKPOINT_EVERY:
            save_checkpoint(build_checkpoint())
            _pages_processed_since_checkpoint = 0


def procesar_lote(pages: list[tuple[int, str | None, float]]) -> None:
    """Procesa VARIAS páginas en un solo request (Fase 1 — batch U-OCR).

    Envía las páginas a /api/process-page-batch: el servidor corre el híbrido
    por página, agrupa las que disparan el trigger v4.2 y las envía juntas al
    daemon U-OCR con infer_multi (comparten el prefill del modelo).

    Retorna sin resultado si el batch completo falla (timeout/error HTTP);
    las páginas se registran con su resultado individual del batch.
    """
    global _pages_processed_since_checkpoint
    if not pages:
        return
    t0 = time.time()

    # Render previo fallido → registrar error y seguir con el resto del lote
    valid = []
    for page_num, b64, render_t in pages:
        if b64 is None:
            with _lock:
                print(f"  Pág {page_num:3d}: ERROR render previo (batch)")
                stats["pages_error"] += 1
                results.append({"page": page_num, "status": "render_error",
                                "blocks": 0, "translated": 0, "time": render_t})
                pages_done.add(page_num)
                _pages_processed_since_checkpoint += 1
        else:
            valid.append((page_num, b64, render_t))
    if not valid:
        return

    b64s = [b64 for _, b64, _ in valid]
    # El batch espera el daemon U-OCR (130-500s/inferencia): timeout cubre
    # hasta 4 inferencias encoladas + margen.
    timeout = TIMEOUT_PER_PAGE * min(len(valid), 4)
    resp = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = _http_session.post(
                f"{API_URL}/api/process-page-batch",
                json={'images': b64s, 'target': TARGET, 'source': SOURCE,
                      'ocr_mode': OCR_MODE, 'prefilter': PREFILTER,
                      'force_uocr': FORCE_UOCR, 'doc_id': DOC_ID},
                timeout=timeout,
            )
            break
        except requests.Timeout:
            elapsed = time.time() - t0
            # Elapsed compartido REPARTIDO por página (sesión 115): un lote
            # fallido de N páginas no debe heredar el elapsed completo N veces
            # en results/checkpoint — misma regla que el camino de éxito.
            per_page_elapsed = elapsed / max(len(valid), 1)
            with _lock:
                print(f"  LOTE {[p[0] for p in valid]}: TIMEOUT batch "
                      f"(intento {attempt+1}/{1+MAX_RETRIES}, {elapsed:.0f}s)")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                with _lock:
                    for page_num, _, render_t in valid:
                        stats["pages_error"] += 1
                        results.append({"page": page_num, "status": "timeout",
                                        "blocks": 0, "translated": 0, "time": per_page_elapsed})
                        pages_done.add(page_num)
                        _pages_processed_since_checkpoint += 1
                return
        except Exception as e:
            # Sesión 120: `elapsed` se define aquí TAMBIÉN — si la primera
            # excepción no era Timeout, el branch conn_error usaba `elapsed`
            # sin definir → NameError silencioso (el worker moría y el lote se
            # perdía sin registrar nada).
            elapsed = time.time() - t0
            per_page_elapsed = elapsed / max(len(valid), 1)
            with _lock:
                print(f"  LOTE {[p[0] for p in valid]}: ERROR conexión batch: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                with _lock:
                    for page_num, _, render_t in valid:
                        stats["pages_error"] += 1
                        results.append({"page": page_num, "status": "conn_error",
                                        "blocks": 0, "translated": 0, "time": per_page_elapsed})
                        pages_done.add(page_num)
                        _pages_processed_since_checkpoint += 1
                return

    elapsed = time.time() - t0
    per_page_elapsed = elapsed / max(len(valid), 1)
    if resp is None or resp.status_code != 200:
        code = resp.status_code if resp is not None else "?"
        with _lock:
            print(f"  LOTE {[p[0] for p in valid]}: ERROR HTTP {code} ({elapsed:.1f}s)")
            for page_num, _, render_t in valid:
                stats["pages_error"] += 1
                results.append({"page": page_num, "status": f"http_{code}",
                                "blocks": 0, "translated": 0, "time": per_page_elapsed})
                pages_done.add(page_num)
                _pages_processed_since_checkpoint += 1
        return

    data = resp.json()
    results_list = data.get("results", [])
    # El servidor devuelve una entrada por imagen en el MISMO orden de entrada.
    # El elapsed del lote se REPARTE entre sus páginas (per_page) en vez de
    # heredarlo completo: antes cada página sumaba el elapsed entero del lote a
    # page_times (un lote de 4 págs de 587s sumaba 4×587s), inflando ~4-8x el
    # "Tiempo total". Con el reparto, sum(page_times) == tiempo real del lote.
    per_page_elapsed = elapsed / max(len(valid), 1)
    with _lock:
        print(f"  LOTE {[p[0] for p in valid]}: {len(valid)} págs en {elapsed:.1f}s "
              f"({per_page_elapsed:.1f}s/pág)")
    for (page_num, _, render_t), page_res in zip(valid, results_list):
        if not isinstance(page_res, dict):
            with _lock:
                stats["pages_error"] += 1
                results.append({"page": page_num, "status": "bad_result",
                                "blocks": 0, "translated": 0, "time": per_page_elapsed})
                pages_done.add(page_num)
                _pages_processed_since_checkpoint += 1
            continue
        _registrar_resultado(page_num, render_t, per_page_elapsed, page_res.get("blocks", []))
    # Páginas del lote que el servidor no devolvió (p. ej. error interno)
    for i in range(len(results_list), len(valid)):
        page_num = valid[i][0]
        with _lock:
            stats["pages_error"] += 1
            results.append({"page": page_num, "status": "missing_result",
                            "blocks": 0, "translated": 0, "time": per_page_elapsed})
            pages_done.add(page_num)
            _pages_processed_since_checkpoint += 1

def main() -> None:
    """Ejecuta el flujo completo: health check, render, OCR, checkpoint y reporte."""
    global total_pages, pages_done, results, page_times, stats, _pages_processed_since_checkpoint

    # Encoding/entorno: solo relevante al ejecutar como script. Los tests
    # importan el módulo y no deben tocar el stdout/entorno del proceso pytest.
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    # ─── init ────────────────────────────────────────────────────
    # Verificar servidor (usando sesión HTTP compartida)
    try:
        r = _http_session.get(f"{API_URL}/api/health", timeout=5)
        if not r.json().get("ok"):
            print("ERROR: Servidor no saludable"); sys.exit(1)
        print(f"Servidor OK - {r.json().get('version','?')}")
        if FORCE_UOCR:
            uocr_state = r.json().get("unlimited_ocr", "?")
            if uocr_state != "ready":
                print(f"⚠️  --force-uocr activo pero el daemon U-OCR está '{uocr_state}'. "
                      f"TODAS las páginas fallarán con http_503 hasta que cargue. "
                      f"Espera a que /api/health reporte 'ready'.")
    except Exception as e:
        print(f"ERROR: No se puede conectar: {e}"); sys.exit(1)

    # Abrir PDF
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    if MAX_PAGES > 0:
        total_pages = min(total_pages, MAX_PAGES)
    print(f"\nPDF: {os.path.basename(PDF_PATH)} ({len(doc)} páginas, procesando {total_pages})")
    pf_label = "+prefilter" if PREFILTER else ""
    print(f"Idioma: {SOURCE} → {TARGET} | OCR: {OCR_MODE}{pf_label} | Workers: {MAX_WORKERS} | Inicio: {time.strftime('%H:%M:%S')}")
    print("=" * 80)

    # ── estado (cargar checkpoint) ──────────────────────────────
    checkpoint = load_checkpoint()
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
    _pages_processed_since_checkpoint = 0
    n_paginas_iniciales = len(pages_done)  # para medir solo las nuevas de esta corrida

    # ─── Lanzar render worker + N API workers ───────────────────
    # Tiempo de pared real de esta corrida (NO la suma de elapsed por página:
    # en modo batch cada página heredaba el elapsed COMPLETO de su lote, lo que
    # inflaba ~4-8x el total; y con workers paralelos la suma tampoco es el
    # tiempo real). Se mide desde el inicio del procesamiento hasta el final.
    t_wall_start = time.time()
    render_thread = threading.Thread(
        target=render_worker, args=(doc, total_pages, pages_done), daemon=True
    )
    render_thread.start()

    api_workers = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="page_api")
    futures = []
    pages_en_cola = 0

    # Consumir páginas renderizadas y lanzar API calls en paralelo.
    # Con BATCH_WINDOW > 1 (Fase 1): agrupar páginas contiguas en un solo request
    # /api/process-page-batch — el servidor junta los triggers U-OCR en un único
    # infer_multi al daemon (comparte el prefill del modelo).
    while True:
        if BATCH_WINDOW > 1:
            # Acumular hasta BATCH_WINDOW páginas renderizadas (la primera bloquea
            # hasta que haya render; el resto se toman sin esperar más)
            batch: list[tuple[int, str | None, float]] = []
            first = get_next_page(timeout=60)
            if first is None or first[0] is None:  # centinela de fin
                break
            batch.append(first)
            while len(batch) < BATCH_WINDOW:
                extra = get_next_page(timeout=0.2)
                if extra is None or extra[0] is None:
                    # Centinela de fin: ya se sacó de la cola — re-insertarlo para
                    # que la próxima iteración externa lo detecte y rompa (si lo
                    # dejamos fuera, la cola queda vacía con el render_worker
                    # terminado y la iteración siguiente espera 60s en vano).
                    if extra is not None and extra[0] is None:
                        with _lock:
                            _rendered_queue.insert(0, extra)
                    break
                batch.append(extra)
            pages_en_cola += len(batch)
            fut = api_workers.submit(procesar_lote, batch)
            futures.append(fut)
        else:
            page_num, b64, render_t = get_next_page(timeout=60)
            if page_num is None:  # centinela de fin
                break
            pages_en_cola += 1
            fut = api_workers.submit(procesar_pagina, page_num, b64, render_t)
            futures.append(fut)

    # Esperar a que todas las API calls terminen
    try:
        concurrent.futures.wait(futures)
    finally:
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
    # Tiempo de pared real medido en main() (correcto en batch y con workers
    # paralelos; la suma de elapsed por página ya no se usa como total).
    wall_time = time.time() - t_wall_start
    paginas_nuevas = max(len(pages_done) - n_paginas_iniciales, 1)
    print(f"Tiempo total (pared real): {wall_time:.0f}s ({wall_time/60:.1f} min)")
    print(f"Promedio por página (pared): {wall_time/paginas_nuevas:.1f}s")
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
        print(f" {s['pages_translated']} páginas traducidas correctamente")
    if s['pages_with_text'] - s['pages_translated'] > 0:
        print(f"  {s['pages_with_text'] - s['pages_translated']} páginas con texto sin traducir")
    if s['pages_empty'] > 0:
        print(f"ℹ  {s['pages_empty']} páginas sin texto (viñetas/arte)")
    if s['pages_error'] > 0:
        print(f" {s['pages_error']} páginas con error")
        failed = [r["page"] for r in results if r["status"] in ("timeout","render_error","conn_error") or str(r["status"]).startswith("http_")]
        if failed:
            print(f"   Páginas con error: {failed}")

    print(f"\nReporte guardado en: {CHECKPOINT_FILE}")
    print(f"Fin: {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
