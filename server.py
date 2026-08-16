from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flask import Flask, jsonify, send_from_directory, abort, request as _flask_request

# Force UTF-8 on Windows stdout
if sys.platform == "win32":
    try:
        # Algunos lanzadores/pytest reemplazan stdout por un wrapper con
        # codificaciÃ³n cp1252. Los logs de precarga contienen flechas y otros
        # sÃ­mbolos; `errors=replace` evita que un print aborte el hilo de
        # precarga aunque el proceso se haya iniciado fuera de UTF-8.
        # Los wrappers de pytest pueden no exponer reconfigure; el try/except
        # ya cubre ese caso en runtime.
        sys.stdout.reconfigure(  # type: ignore[union-attr]
            encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(  # type: ignore[union-attr]
            encoding="utf-8", errors="replace")
    except Exception:  # nosec
        pass

# Config import (paths, constants, CSP, etc.)
from config import (
    ROOT, DIST, IS_PRODUCTION,
    MAX_WORKERS, APP_VERSION, LANGUAGES, CSP_POLICY,
    MAX_IMAGE_BYTES,
    TIMEOUT_OPENCV_INIT_MS, TIMEOUT_PDFJS_CDN_MS,
    TIMEOUT_PDFJS_ES_MODULE_MS, TIMEOUT_PDF_RENDER_MS,
    TIMEOUT_TRANSLATE_MS, TIMEOUT_TRANSLATE_BATCH_MS,
    TIMEOUT_PROCESS_PAGE_MS, TIMEOUT_INPAINTED_IMAGE_MS,
    TIMEOUT_EXPORT_REVOKE_MS, TIMEOUT_CDN_LOAD_MS,
)

# ─── Caché de HuggingFace dentro del proyecto ──────────────────
# Regla "no tocar C:": toda la caché HF (tokenizers OPUS-MT, modelos de
# transformadores) vive en hf_cache/ del proyecto, igual que hace el daemon
# U-OCR (uocr_daemon.py:48-50). Sin esto, las descargas de tokenizers de CT2
# caerían en ~/.cache/huggingface del usuario y no viajarían con el proyecto.
# setdefault respeta un HF_HOME ya definido por el entorno si existiera.
import os as _os
_os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
_os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "hf_cache" / "hub"))
_os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "hf_cache" / "hub"))

app = Flask(__name__, static_folder=None)
app.config["ENV"] = "production" if IS_PRODUCTION else "development"
app.config["DEBUG"] = not IS_PRODUCTION

# ─── Security limits ────────────────────────────────────────────
# Max request body: 50MB (para base64 de imagenes grandes)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES
# Prohibir framebusting via Flask-Talisman no disponible, usamos CSP + headers
# Limitar tipos de contenido aceptados en POST
app.config["MAX_FORM_MEMORY_SIZE"] = 1024 * 100  # 100KB form data


# ─── Compresión gzip de respuestas (Fase 2.2) ────────────────────
# Sin dependencias extra (stdlib gzip). Solo comprime texto/JSON > 1 KB
# cuando el cliente lo anuncia (fetch envía Accept-Encoding: gzip por
# defecto). Las imágenes base64 ya comprimidas (JPEG/PNG) se saltan para
# no quemar CPU sin ganancia.
import gzip as _gzip


@app.after_request
def _compress_response(response: Any) -> Any:
    accept = _flask_request.headers.get("Accept-Encoding", "")
    if "gzip" not in accept.lower():
        return response
    ctype = (response.mimetype or "").lower()
    # text/* (html, css, js, plain, markdown...) + json/xml: comprimibles.
    # Las imágenes (base64 en JSON ya cuentan como json; binarios png/jpeg
    # NO entran por text/* ni json) se saltan — no ganan nada y queman CPU.
    if not (ctype.startswith("text/") or ctype in ("application/json", "application/javascript",
                                                   "application/xml")):
        return response
    # send_from_directory (estáticos) va en modo passthrough: no tocar,
    # ya son archivos en disco servidos por el WSGI directamente.
    if getattr(response, "direct_passthrough", False):
        return response
    data = response.get_data()
    if data is None or len(data) < 1024:
        return response
    try:
        gz = _gzip.compress(data, compresslevel=6)
    except Exception:  # nosec — nunca dejar caer la respuesta por compresión
        return response
    if len(gz) >= len(data):
        return response
    response.set_data(gz)
    response.headers["Content-Encoding"] = "gzip"
    response.headers["Vary"] = "Accept-Encoding"
    response.headers["Content-Length"] = str(len(gz))
    return response


# ─── Database init ──────────────────────────────────────────────
DB_AVAILABLE = False
try:
    from models import init_db, db
    init_db(app)
    DB_AVAILABLE = True
except Exception as e:
    print(f"[db] Base de datos no disponible: {e}")


# ─── Translation cache ──────────────────────────────────────────
TRANSLATION_CACHE_AVAILABLE = False
cache_get = None
cache_set = None

try:
    from cache import get as _cg, set as _cs
    cache_get = _cg
    cache_set = _cs
    TRANSLATION_CACHE_AVAILABLE = True
    print("[cache] Cache de traducciones activo")
except Exception as e:
    print(f"[cache] Cache no disponible: {e}")


# ─── Preload EasyOCR + CT2 (background) ─────────────────────────
# ORDEN CRÍTICO: EasyOCR PRIMERO (toma GPU, inicializa torch.cuda),
# luego CT2 (detecta CUDA disponible, carga en GPU).
# No se necesita force_cpu porque EasyOCR ya inicializó CUDA.
# Si se invierte el orden, CT2 carga sus DLLs CUDA/cuDNN primero y
# EasyOCR crashea con "Could not load symbol cudnnGetLibConfig".
#
# Ambos en GPU: EasyOCR ~0.5s/página, CT2 ~0.06s/traducción.
# Verificado: GTX 1050 Ti, 4GB VRAM, VRAM usada ~130MB (por modelo).
def _preload_background() -> None:
    try:
        # ── Paso 1: EasyOCR en GPU ──────────────────────────────
        print("[preload] Cargando EasyOCR (toma GPU, inicializa CUDA)...")
        from ocr_utils import _get_ocr_reader
        t0 = time.time()
        reader = _get_ocr_reader("es")
        if reader:
            print(f"[preload] EasyOCR cargado en {time.time()-t0:.1f}s")
        else:
            print("[preload] EasyOCR no disponible")
        
        # ── Paso 2: CT2 en GPU (detecta CUDA disponible, auto-selecciona GPU) ──
        from translator import _get_ct2_translator
        print("[preload] Precargando modelo CT2 es→en (auto-detecta GPU)...")
        t0 = time.time()
        translator, tokenizer = _get_ct2_translator("es", "en")
        if translator:
            print(f"[preload] CT2 es→en cargado en {time.time()-t0:.1f}s (device: {translator.device})")
        # También precargar en→es
        translator2, tokenizer2 = _get_ct2_translator("en", "es")
        if translator2:
            print(f"[preload] CT2 en→es cargado (device: {translator2.device})")
    except Exception as e:
        print(f"[preload] Error en precarga: {e}")

    # RapidOCR usa ONNX en CPU y no compite por la VRAM de EasyOCR/CT2.
    # Precargarlo aquí evita que la primera página pague la inicialización
    # del detector/reconocedor del tier híbrido.
    try:
        from ocr_utils import _get_rapid_engine
        t0 = time.time()
        engine_rapid = _get_rapid_engine()
        if engine_rapid:
            print(f"[preload] RapidOCR cargado en {time.time()-t0:.1f}s")
        else:
            print("[preload] RapidOCR no disponible")
    except Exception as e:
        print(f"[preload] RapidOCR no disponible: {e}")

    # ── Paso 3 (Fase 6): detector YOLO de regiones (CPU, ~1-2s one-time) ──
    # Se precarga para que la PRIMERA página no pague los ~20s de carga del
    # modelo. Sin ultralytics/modelo, _get_yolo_engine degrada a None y el
    # tier YOLO no aporta (el pipeline sigue igual). Se precarga DESPUÉS de
    # EasyOCR/CT2 (los modelos de ultralytics viven en CPU, sin VRAM).
    # El corrector ortografico post-OCR carga un diccionario grande en memoria.
    # Prepararlo aqui evita que la primera pagina pague esa inicializacion.
    try:
        from ocr_utils import _get_spellchecker, _get_foreign_spellchecker
        t0 = time.time()
        spellchecker = _get_spellchecker()
        if spellchecker:
            print(f"[preload] Corrector OCR cargado en {time.time()-t0:.1f}s")
        else:
            print("[preload] Corrector OCR no disponible")
        # 2026-08-15: precargar tambien los diccionarios extranjeros (en/pt)
        # que usa _contains_foreign_latin_tokens para detectar mezcla de
        # idiomas. Sin esto, la PRIMERA pagina con tokens no espanoles paga
        # ~350 ms de carga bajo demanda (en 83 ms + pt 266 ms) dentro del
        # tiempo medido de la pagina.
        for lang in ("en", "pt"):
            t1 = time.time()
            checker = _get_foreign_spellchecker(lang)
            if checker:
                print(f"[preload] Diccionario {lang} cargado en "
                      f"{time.time()-t1:.1f}s")
    except Exception as e:
        print(f"[preload] Corrector OCR no disponible: {e}")

    try:
        from ocr_utils import _get_yolo_engine
        t0 = time.time()
        engine = _get_yolo_engine()
        if engine:
            print(f"[preload] YOLO regions cargado en {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"[preload] YOLO no disponible: {e}")


# ─── Daemon Unlimited-OCR (GPU 4-bit): preload independiente ────
# El modelo (6.78 GB, 3B MoE) corre en env_uocr_gpu (torch cu126 +
# bitsandbytes), un venv separado del servidor (env/) para evitar el
# conflicto de DLLs CUDA con EasyOCR/CT2. Se lanza como subproceso
# persistente que carga el modelo UNA vez (~8 min) y queda escuchando
# en 127.0.0.1:5177 — así la primera página NO espera la carga.
#
# Se arranca en un HILO PROPIO e independiente del preload de
# EasyOCR/CT2: aunque la carga de EasyOCR tarde (descarga de modelos,
# primer arranque, etc.), el daemon U-OCR empieza a cargar al instante.
def _preload_unlimited_daemon() -> None:
    try:
        import uocr_client
        if uocr_client.spawn_daemon():
            print("[preload] Daemon Unlimited-OCR lanzado (modelo 4-bit cargando en background)")
        else:
            print("[preload] Daemon Unlimited-OCR no disponible (venv o script ausente)")
    except Exception as e:
        print(f"[preload] Daemon Unlimited-OCR no disponible: {e}")


_uocr_thread = threading.Thread(target=_preload_unlimited_daemon, daemon=True)
_uocr_thread.start()


_preload_thread = threading.Thread(target=_preload_background, daemon=True)
_preload_thread.start()


# ─── Rate limiting ──────────────────────────────────────────────
from ratelimit import init_limiter, limiter, RATE_LIMIT_AVAILABLE
init_limiter(app)


# ─── Shared ThreadPoolExecutor ─────────────────────────────────
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="translator")
_executor_lock = threading.Lock()
_executor_shutdown = False


def _get_executor() -> ThreadPoolExecutor:
    global _executor, _executor_shutdown
    with _executor_lock:
        if _executor_shutdown:
            _executor = ThreadPoolExecutor(
                max_workers=MAX_WORKERS, thread_name_prefix="translator"
            )
            _executor_shutdown = False
        return _executor


def shutdown_executor() -> None:
    global _executor, _executor_shutdown
    with _executor_lock:
        if not _executor_shutdown:
            _executor.shutdown(wait=True, cancel_futures=True)
            _executor_shutdown = True


# ─── Security headers (CSP, Brave Leo opt-out) ─────────────────
@app.after_request
def add_security_headers(response: Any) -> Any:
    response.headers["Content-Security-Policy"] = CSP_POLICY
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    return response


# ─── Translation wrapper (injects cache) ───────────────────────
from translator import _translate_one as _translate_one_impl


def _translate_one(
    text: str,
    source: str,
    target: str,
    *,
    block_type: str | None = None,
) -> str:
    return _translate_one_impl(
        text, source, target,
        cache_get=cache_get if TRANSLATION_CACHE_AVAILABLE else None,
        cache_set=cache_set if TRANSLATION_CACHE_AVAILABLE else None,
        translation_cache_available=TRANSLATION_CACHE_AVAILABLE,
        block_type=block_type,
    )


# ═══════════════════════════════════════════════════════════════
# REGISTER ALL SPECIFIC ROUTES FIRST (app-level + blueprints)
# THEN the catch-all at the end.
# ═══════════════════════════════════════════════════════════════

# ─── App-level specific routes ────────────────────────────────
@app.route("/api/config")
def serve_config() -> Any:
    # MODO_CPU se lee en runtime (no a nivel módulo) para que los tests y el
    # launcher puedan mutar config y verificar el preset sin reiniciar.
    from config import MODO_CPU, MODO_CPU_OCR_SCALE
    return jsonify({
        "version": APP_VERSION,
        "languages": LANGUAGES,
        "modo_cpu": MODO_CPU,
        # Escala de render que el frontend debe usar: 1.2 normal, la del preset
        # con MODO_CPU (menos píxeles a procesar — análogo a bajar resolución).
        "ocr_scale": MODO_CPU_OCR_SCALE if MODO_CPU else 1.2,
        "timeouts_ms": {
            "opencv_init": TIMEOUT_OPENCV_INIT_MS,
            "pdfjs_cdn": TIMEOUT_PDFJS_CDN_MS,
            "pdfjs_es_module": TIMEOUT_PDFJS_ES_MODULE_MS,
            "pdf_render": TIMEOUT_PDF_RENDER_MS,
            "translate": TIMEOUT_TRANSLATE_MS,
            "translate_batch": TIMEOUT_TRANSLATE_BATCH_MS,
            "process_page": TIMEOUT_PROCESS_PAGE_MS,
            "inpainted_image": TIMEOUT_INPAINTED_IMAGE_MS,
            "export_revoke": TIMEOUT_EXPORT_REVOKE_MS,
            "cdn_load": TIMEOUT_CDN_LOAD_MS,
        },
    })


@app.route("/app.min.js")
def serve_js() -> Any:
    return send_from_directory(str(ROOT), "app.js")


@app.route("/styles.min.css")
def serve_css() -> Any:
    return send_from_directory(str(ROOT), "styles.css")


# ─── App-level catch-all (BEFORE blueprints) ──────────────────
# En Flask 3.x, las rutas directas de la app se evalúan ANTES
# que las rutas de blueprints. El catch-all debe ir a nivel app.
@app.route("/<path:filename>")
def serve_root(filename: str) -> Any:
    # Path traversal protection: verificar que el archivo este dentro de ROOT
    target = (ROOT / filename).resolve()
    from routes.main import _is_within
    if not _is_within(ROOT, target):
        abort(404)
    if not target.exists():
        abort(404)
    return send_from_directory(str(ROOT), filename)


# ─── Blueprint routes (evaluadas después de las rutas directas) ─
from routes.main import main_bp
from routes.api import api_bp
app.register_blueprint(main_bp)
app.register_blueprint(api_bp)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    host = "127.0.0.1"  # Siempre localhost (evita CWE-605)
    port = 5174

    print(f"[server] Arrancando en http://{host}:{port}")
    print(f">>> SERVIDOR LISTO <<< http://{host}:{port}")
    sys.stdout.flush()
    # Servidor de producción: waitress (multi-threaded, estable).
    # El problema histórico de waitress con catch-all + blueprints en Flask 3.x
    # (404s) está resuelto: el catch-all /<path:filename> se registra a nivel
    # de app ANTES de los blueprints (ver serve_root arriba), y sirve solo
    # archivos existentes — los 404 reales los emite el framework. Validado
    # por el job server-test de run_ci.py (health + endpoints + estáticos).
    from waitress import serve
    serve(app, host=host, port=port, threads=8)
