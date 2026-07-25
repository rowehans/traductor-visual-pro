from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flask import Flask, jsonify, send_from_directory, abort

# Force UTF-8 on Windows stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Config import (paths, constants, CSP, etc.)
from config import (
    ROOT, DIST, IS_PRODUCTION,
    MAX_WORKERS, APP_VERSION, LANGUAGES, CSP_POLICY,
    TIMEOUT_OPENCV_INIT_MS, TIMEOUT_PDFJS_CDN_MS,
    TIMEOUT_PDFJS_ES_MODULE_MS, TIMEOUT_PDF_RENDER_MS,
    TIMEOUT_TRANSLATE_MS, TIMEOUT_TRANSLATE_BATCH_MS,
    TIMEOUT_PROCESS_PAGE_MS, TIMEOUT_INPAINTED_IMAGE_MS,
    TIMEOUT_EXPORT_REVOKE_MS, TIMEOUT_CDN_LOAD_MS,
)

app = Flask(__name__, static_folder=None)
app.config["ENV"] = "production" if IS_PRODUCTION else "development"
app.config["DEBUG"] = not IS_PRODUCTION

# ─── Security limits ────────────────────────────────────────────
# Max request body: 50MB (para base64 de imagenes grandes)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024
# Prohibir framebusting via Flask-Talisman no disponible, usamos CSP + headers
# Limitar tipos de contenido aceptados en POST
app.config["MAX_FORM_MEMORY_SIZE"] = 1024 * 100  # 100KB form data


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


# ─── CT2 preload (background) ──────────────────────────────────
# Pre-carga el modelo CT2 es→en al arrancar el servidor para evitar
# la penalidad de ~21.5s en la primera traducción (carga de modelo
# de 300MB a GPU, tokenizer, CUDA libraries, etc.).
# Se ejecuta en un hilo daemon para no bloquear el arranque.
def _preload_ct2_models() -> None:
    try:
        from translator import _get_ct2_translator
        print("[preload] Precargando modelo CT2 es→en (background)...")
        t0 = time.time()
        translator, tokenizer = _get_ct2_translator("es", "en")
        if translator:
            print(f"[preload] ✅ CT2 es→en cargado en {time.time()-t0:.1f}s")
        # También precargar en→es (común en traducción inversa)
        translator2, tokenizer2 = _get_ct2_translator("en", "es")
        if translator2:
            print(f"[preload] ✅ CT2 en→es cargado")
    except Exception as e:
        print(f"[preload] Error precargando CT2: {e}")


_preload_thread = threading.Thread(target=_preload_ct2_models, daemon=True)
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


def _translate_one(text: str, source: str, target: str) -> str:
    return _translate_one_impl(
        text, source, target,
        cache_get=cache_get if TRANSLATION_CACHE_AVAILABLE else None,
        cache_set=cache_set if TRANSLATION_CACHE_AVAILABLE else None,
        translation_cache_available=TRANSLATION_CACHE_AVAILABLE,
    )


# ═══════════════════════════════════════════════════════════════
# REGISTER ALL SPECIFIC ROUTES FIRST (app-level + blueprints)
# THEN the catch-all at the end.
# ═══════════════════════════════════════════════════════════════

# ─── App-level specific routes ────────────────────────────────
@app.route("/api/config")
def serve_config() -> Any:
    return jsonify({
        "version": APP_VERSION,
        "languages": LANGUAGES,
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
    return send_from_directory(".", "app.js")


@app.route("/styles.min.css")
def serve_css() -> Any:
    return send_from_directory(".", "styles.css")


# ─── App-level catch-all (BEFORE blueprints) ──────────────────
# En Flask 3.x, las rutas directas de la app se evalúan ANTES
# que las rutas de blueprints. El catch-all debe ir a nivel app.
@app.route("/<path:filename>")
def serve_root(filename: str) -> Any:
    # Path traversal protection: verificar que el archivo este dentro de ROOT
    target = (ROOT / filename).resolve()
    if not str(target).startswith(str(ROOT)):
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
    host = "127.0.0.1" if sys.platform == "win32" else "0.0.0.0"
    port = 5174
    
    print(f"[server] Arrancando en http://{host}:{port}")
    print(f">>> SERVIDOR LISTO <<< http://{host}:{port}")
    sys.stdout.flush()
    # Flask dev server (waitress tiene problemas con catch-all + blueprints en Flask 3.x)
    app.run(host=host, port=port, debug=False, use_reloader=False)
