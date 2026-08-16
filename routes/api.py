"""
routes/api.py — Blueprint para endpoints de API REST.
Mejorado con validación de requests, errores estandarizados
y rate limiting según mejores prácticas Flask.
"""
import cProfile
import functools
import gc
import io
import os
import pstats
import re
import time as _time
import traceback
from collections.abc import Callable
from concurrent.futures import as_completed
from typing import Any, ParamSpec, TypeVar, overload

import cv2
import numpy as np
import psutil
import threading
from flask import Blueprint, Response, jsonify, request
from numpy.typing import NDArray
_Img = np.ndarray

from config import (
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
    MAX_IMAGE_BYTES,
    MAX_BATCH_DECODE_PIXELS,
    UOCR_IMAGE_BLOCK_RATIO,
    GPU_VRAM_BUDGET_MB,
    GPU_MIN_FREE_VRAM_MB,
    LANGUAGES,
)
from ratelimit import limiter, RATE_LIMIT_AVAILABLE
from runtime_diagnostics import gpu_budget_allows, gpu_memory_snapshot
from translation_memory import get_document_memory

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ════════════════════════════════════════════════════════════════
# CACHE DE PÁGINA COMPLETA (optimización 2.7)
# ════════════════════════════════════════════════════════════════
# Re-correr el mismo capítulo hoy re-hace OCR + inpaint + traducción de
# todo. Este cache guarda la respuesta COMPLETA de /process-page por hash
# de la imagen de entrada + parámetros: la segunda vez que se procesa la
# misma página (mismo archivo, mismo doc_id), se devuelve al instante.
# La clave incluye doc_id para que capítulos distintos no colisionen y
# response_format porque cambia el encode de la imagen inpaintada.
_PAGE_CACHE_MAX: int = 24          # páginas en memoria (FIFO)
_page_cache: dict[str, dict[str, Any]] = {}


def _page_cache_key(
    image_bytes: bytes,
    *,
    target: str,
    source: str,
    ocr_mode: str,
    prefilter: bool,
    force_uocr: bool,
    disable_uocr: bool,
    pure_easyocr: bool,
    doc_id: str,
    response_format: str,
) -> str:
    """Clave estable del cache: bytes de la imagen + params que cambian
    el resultado. Los bytes de entrada (base64 o binario crudo) se usan
    tal cual, así que la misma página genera siempre la misma clave."""
    import hashlib

    h = hashlib.sha256(image_bytes)
    for part in (
        target, source, ocr_mode,
        str(prefilter), str(force_uocr), str(disable_uocr),
        str(pure_easyocr), doc_id, response_format,
    ):
        h.update(part.encode("utf-8"))
    return h.hexdigest()


def _page_cache_get(key: str) -> dict[str, Any] | None:
    entry = _page_cache.get(key)
    if entry is None:
        return None
    # Re-insertar para LRU-ish (los más recientes quedan al final).
    _page_cache.pop(key)
    _page_cache[key] = entry
    return entry


def _page_cache_set(key: str, entry: dict[str, Any]) -> None:
    _page_cache[key] = entry
    while len(_page_cache) > _PAGE_CACHE_MAX:
        _page_cache.pop(next(iter(_page_cache)))


# ════════════════════════════════════════════════════════════════
# CONSTANTES DE VALIDACIÓN DE SEGURIDAD
# ════════════════════════════════════════════════════════════════
# MAX_TEXT_LENGTH, MAX_BATCH_SIZE y MAX_IMAGE_BYTES se importan desde config.py
_MIN_TEXT_LENGTH: int = 1               # mínimo 1 char útil tras strip
_MAX_TEXT_WORDS: int = 5_000            # ~5000 palabras máx

# Idiomas válidos para el endpoint de traducción (excluye auto para target)
_LANG_CODES: frozenset[str] = frozenset(LANGUAGES)


# ════════════════════════════════════════════════════════════════
# WATCHDOG ANTI-ZOMBIE
# ════════════════════════════════════════════════════════════════
# Detecta cuando el servidor responde sospechosamente rápido con 0 bloques,
# lo que indica un proceso zombie con estado corrupto (como en el bug donde
# el servidor PID 7140 devolvía 0 bloques en 0.1s porque EasyOCR no se
# había cargado correctamente).
#
# Criterio: OCR en <2s con 0 bloques = sospechoso (una página real siempre
# tarda >2s aunque esté vacía, por la carga de EasyOCR + procesamiento).
# Tras N detecciones consecutivas, se marca como zombie.

_ZOMBIE_THRESHOLD: int = 3                 # N detecciones consecutivas = zombie
_ZOMBIE_COUNTER: int = 0                    # Contador actual
_ZOMBIE_FAST_TIME: float = 2.0              # Tiempo máximo considerado "sospechoso"
_ZOMBIE_LOCK: threading.Lock = threading.Lock()


def _is_zombie() -> bool:
    """Retorna True si el servidor está en estado zombie."""
    with _ZOMBIE_LOCK:
        return _ZOMBIE_COUNTER >= _ZOMBIE_THRESHOLD


def _reset_zombie_counter() -> None:
    """Resetea el contador (llamado tras una respuesta exitosa)."""
    global _ZOMBIE_COUNTER
    with _ZOMBIE_LOCK:
        if _ZOMBIE_COUNTER > 0:
            _ZOMBIE_COUNTER = 0
            print("[watchdog] Contador zombie reseteado (respuesta OK)")


def _increment_zombie_counter() -> bool:
    """
    Incrementa el contador zombie. Retorna True si se alcanzó el umbral.
    """
    global _ZOMBIE_COUNTER
    with _ZOMBIE_LOCK:
        _ZOMBIE_COUNTER += 1
        current = _ZOMBIE_COUNTER
    print(f"[watchdog] Posible zombie #{current}/{_ZOMBIE_THRESHOLD}: OCR rápido con 0 bloques")
    if current >= _ZOMBIE_THRESHOLD:
        print(f"[watchdog] ¡ZOMBIE DETECTADO! Umbral alcanzado ({current}/{_ZOMBIE_THRESHOLD}). "
              f"Reinicia el servidor para restaurar funcionamiento.")
        return True
    return False


# ════════════════════════════════════════════════════════════════
# HELPERS DE RESPUESTA Y VALIDACIÓN
# ════════════════════════════════════════════════════════════════

_P = ParamSpec("_P")
_R = TypeVar("_R")


def _error_response(
    message: str,
    status_code: int = 400,
    details: dict[str, Any] | None = None,
) -> tuple[Response, int]:
    """Retorna una respuesta de error estandarizada.

    Args:
        message: Mensaje descriptivo del error.
        status_code: Código HTTP (default 400).
        details: Campos adicionales opcionales (campos faltantes, etc.).

    Returns:
        Tuple (Flask Response, status code) para retornar del endpoint.
    """
    body: dict[str, Any] = {"error": message}
    if details:
        body.update(details)
    return jsonify(body), status_code


@overload
def _validate_json_content_type(
    f: Callable[_P, _R],
) -> Callable[_P, _R]: ...


@overload
def _validate_json_content_type(
    f: None = None,
    *,
    allow_binary: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...


def _validate_json_content_type(
    f: Callable[_P, _R] | None = None,
    *,
    allow_binary: bool = False,
) -> Callable[_P, _R] | Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorador: verifica que el Content-Type sea application/json
    en endpoints POST. Retorna 415 si no lo es.

    Soportado como @dec y como @dec(allow_binary=True).

    allow_binary=True (optimización 2.4): también acepta cuerpos binarios
    de imagen (image/* o application/octet-stream) — el cliente envía el
    canvas como blob y los flags van en el query string. El charset check
    solo aplica a JSON.
    """
    def _decorate(func: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not request.is_json:
                if allow_binary and (
                    (request.mimetype or "").startswith("image/")
                    or request.mimetype == "application/octet-stream"
                ):
                    return func(*args, **kwargs)
                return _error_response(
                    "Content-Type debe ser application/json",
                    status_code=415,
                )
            # Validar charset explícito: si se declara charset, debe ser UTF-8
            # para prevenir ataques de encoding (e.g. iso-2022-kr, utf-7).
            # RFC 8259 §8.1: JSON debe transmitirse en UTF-8.
            # Usar request.headers.get() en vez de request.content_type porque
            # el WSGI layer puede parsear/strippear parámetros del Content-Type.
            # Usar regex para manejar "charset=utf-8", "charset = utf-7",
            # "charset  =  \"utf-8\"", etc.
            raw_ct: str = request.headers.get("Content-Type", "").lower()
            charset_m = re.search(r'charset\s*=\s*["\']?([^;"\'\s]+)', raw_ct)
            if charset_m:
                charset_val: str = charset_m.group(1).strip().strip("'\"")
                if charset_val and charset_val != "utf-8":
                    return _error_response(
                        f"Charset '{charset_val}' no soportado. Use charset=utf-8 (o ninguno).",
                        status_code=415,
                    )
            return func(*args, **kwargs)
        return wrapper

    if f is not None:
        return _decorate(f)
    return _decorate


def _validate_payload_fields(
    *required: str,
    allow_binary: bool = False,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorador de fábrica: valida que el payload JSON contenga
    los campos requeridos.

    Args:
        required: Nombres de campos obligatorios.
        allow_binary: Si True, el endpoint también acepta cuerpos binarios
            de imagen (optimización 2.4) — en ese caso se salta la
            validación de campos requeridos porque los flags van en el
            query string.

    Uso:
        @api_bp.post("/translate")
        @_validate_payload_fields("text")
        def translate():
            data = request.get_json()
            ...
    """
    required_list = list(required)

    def decorator(f: Callable[_P, _R]) -> Callable[_P, _R]:
        @functools.wraps(f)
        @_validate_json_content_type(allow_binary=allow_binary)
        def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            data: dict[str, Any] = request.get_json(silent=True) or {}
            is_binary = (
                allow_binary and not request.is_json
                and request.data
            )
            missing = [field for field in required_list if field not in data]
            if missing and not is_binary:
                return _error_response(  # type: ignore[return-value]
                    "Campos requeridos faltantes",
                    status_code=400,
                    details={"missing_fields": missing, "required": required_list},
                )
            return f(*args, **kwargs)
        return wrapper
    return decorator


def _safe_str(val: Any, default: str = "") -> str:
    """Convierte un valor a string sanitizado (nunca None)."""
    if val is None:
        return default
    return str(val).strip()


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    """Normaliza flags JSON sin convertir cualquier cadena no vacía a True."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().casefold()
        if token in {"true", "1", "yes", "on"}:
            return True
        if token in {"false", "0", "no", "off", ""}:
            return False
    return bool(default)


def _validate_lang_code(code: str, allow_auto: bool = True) -> str | None:
    """Valida código de idioma. Retorna el código normalizado o None."""
    code = code.strip().lower()
    if code in _LANG_CODES:
        if code == "auto" and not allow_auto:
            return None
        return code
    return None


def _validate_lang_params(payload: dict[str, Any], allow_source_auto: bool = True) -> tuple[str | None, str | None, str | None]:
    """Valida source y target de un payload. Retorna (source, target, error_msg).
    Si error_msg no es None, la validación falló."""
    # Validar source
    source_raw = _safe_str(payload.get("source"), default="auto")
    source = _validate_lang_code(source_raw, allow_auto=allow_source_auto)
    if source is None:
        codes = ', '.join(sorted(_LANG_CODES)) if allow_source_auto else \
                ', '.join(sorted(c for c in _LANG_CODES if c != 'auto'))
        return None, None, f"Idioma origen no soportado: '{source_raw}'. Soportados: {codes}"

    # Validar target
    target_raw = _safe_str(payload.get("target"), default="es")
    target = _validate_lang_code(target_raw, allow_auto=False)
    if target is None:
        codes = ', '.join(sorted(c for c in _LANG_CODES if c != 'auto'))
        return None, None, f"Idioma destino no soportado: '{target_raw}'. Soportados: {codes}"

    # Validar que source != target (solo si source no es auto)
    if source != "auto" and source == target:
        return None, None, f"Los idiomas origen y destino son el mismo: '{source}'. No hay nada que traducir."

    return source, target, None


def _sanitize_doc_id(value: Any) -> str:
    """Normaliza el scope de memoria y de caches a un identificador seguro."""
    return re.sub(r"[^A-Za-z0-9_]", "", _safe_str(value))[:64]


def _memory_source_language(text: str, source: str) -> str:
    """Resuelve ``auto`` para que la memoria no mezcle idiomas distintos."""
    if source != "auto":
        return source
    try:
        from translator import _detect_language_robust
        detected = _detect_language_robust(text)
        return _safe_str(detected, default="auto").lower() or "auto"
    except Exception:
        return "auto"


def _learn_translation_if_valid(
    memory: Any,
    source_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str,
    quality: float,
) -> None:
    """Aprende solo traducciones válidas y con OCR suficientemente fiable."""
    try:
        from translator import (
            _es_traduccion_valida,
            _translation_is_likely_source_language,
        )
        if not _es_traduccion_valida(source_text, translated_text):
            return
        if _translation_is_likely_source_language(
                translated_text, source_lang, target_lang):
            return
        memory.learn(
            source_text,
            translated_text,
            source_lang,
            target_lang,
            quality=max(0.0, min(1.0, float(quality))),
        )
    except (TypeError, ValueError):
        return


def _document_memory_translation_is_safe(
    source_text: str,
    translated_text: str,
    source_lang: str,
    target_lang: str,
) -> bool:
    """Aplica los gates actuales tambiÃ©n a resultados ya memorizados."""
    try:
        from translator import (
            _es_traduccion_valida,
            _translation_is_likely_source_language,
        )
        candidate = str(translated_text or "").strip()
        if not candidate:
            return False
        if _translation_is_likely_source_language(
                candidate, source_lang, target_lang):
            return False
        return bool(_es_traduccion_valida(
            source_text,
            candidate,
            lenient=len(str(source_text).split()) <= 3,
        ))
    except Exception:
        # Un fallo del validador no debe convertir la memoria en un bypass de
        # seguridad/calidad: ante duda se fuerza una nueva traducciÃ³n.
        return False


def _allow_preserved_name_or_sfx(
    source_text: str,
    translated_text: str,
    block_type: str,
) -> bool:
    """Permite preservaciones semánticas estrechas en el gate final."""
    if block_type.strip().lower() == "sfx":
        return True
    source = str(source_text or "").strip()
    translated = str(translated_text or "").strip()
    if (
        source.casefold() != translated.casefold()
        or len(source.split()) != 1
        or not source.isupper()
    ):
        return False
    # Un cambio NARUTO -> Naruto es una preservación plausible de nombre;
    # no abrimos la excepción para frases completas ni para basura repetida.
    return bool(re.fullmatch(r"[A-ZÀ-Ý][a-zà-ÿ]{2,}", translated))


def _translate_with_document_memory(
    text: str,
    source: str,
    target: str,
    doc_id: str,
    save: bool = True,
) -> str:
    """Traduce usando la memoria automática del documento cuando existe."""
    from server import _translate_one

    memory = get_document_memory(doc_id)
    memory_source = _memory_source_language(text, source)
    if memory is not None:
        cached = memory.lookup(text, memory_source, target)
        if cached is not None and not _document_memory_translation_is_safe(
                text, cached, memory_source, target):
            if hasattr(memory, "discard"):
                memory.discard(text, memory_source, target)
            cached = None
        if cached is None:
            # Solo se permite la variante CJK conservadora de la memoria:
            # exige repetición, alta calidad y como máximo un error de OCR.
            cached = memory.lookup_variant(text, memory_source, target)
            if cached is not None and not _document_memory_translation_is_safe(
                    text, cached, memory_source, target):
                cached = None
        if cached is not None:
            return cached

    result = _translate_one(text, source, target)
    if memory is not None:
        _learn_translation_if_valid(
            memory, text, result, memory_source, target, quality=0.85)
        if save:
            memory.save()
    return result


def _translate_ocr_block(
    text: str,
    source: str,
    target: str,
    block_type: str,
) -> str:
    """Traduce un bloque OCR conservando su contexto semántico."""
    from server import _translate_one

    return _translate_one(text, source, target, block_type=block_type)


def _validate_text_length(text: str) -> str | None:
    """Valida largo de texto. Retorna mensaje de error o None si OK."""
    if len(text) > MAX_TEXT_LENGTH:
        return f"Texto demasiado largo ({len(text)} chars, max {MAX_TEXT_LENGTH})"
    words = text.split()
    if len(words) > _MAX_TEXT_WORDS:
        return f"Demasiadas palabras ({len(words)}, max {_MAX_TEXT_WORDS})"
    return None


# ── Profiling decorator (activar con ?profile=1) ───────────────
_PROFILE_DIR: str = ""


def _get_profile_dir() -> str:
    """Obtiene / crea el directorio de profiles."""
    global _PROFILE_DIR
    if not _PROFILE_DIR:
        _PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiles")
        os.makedirs(_PROFILE_DIR, exist_ok=True)
    return _PROFILE_DIR


def profile_endpoint(f: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorador: perfila el endpoint si ?profile=1 está presente.

    Uso:
        @api_bp.post("/translate")
        @profile_endpoint
        def translate():
            ...

    Resultados:
        - Archivo .prof en profiles/{endpoint}_{timestamp}.prof
        - Log en consola con top-15 por tiempo acumulado
        - Header HTTP X-Profile: endpoint y resumen rápido
    """
    @functools.wraps(f)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if request.args.get("profile") != "1":
            return f(*args, **kwargs)

        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = f(*args, **kwargs)
        except BaseException:
            profiler.disable()
            raise
        else:
            profiler.disable()

        # Post-procesamiento protegido: no debe romper la respuesta
        try:
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
            ps.print_stats(30)
            profile_text = s.getvalue()

            # Guardar archivo .prof
            ts = _time.strftime("%Y%m%d_%H%M%S")
            endpoint_name = request.endpoint or "unknown"
            safe_name = endpoint_name.replace(".", "_")
            prof_path = os.path.join(_get_profile_dir(), f"{safe_name}_{ts}.prof")
            ps.dump_stats(prof_path)

            # Log a consola
            print(f"\n{'='*60}", flush=True)
            print(f"[PROFILE] {endpoint_name} ({ts})", flush=True)
            print(f"[PROFILE] Archivo: {prof_path}", flush=True)

            # Extraer resumen
            total_time = 0.0
            top_funcs = []
            for key, (cc, nc, tt, ct, callers) in ps.stats.items():  # type: ignore[attr-defined]
                filename, lineno, funcname = key
                total_time = max(total_time, ct)
                top_funcs.append((ct, funcname, filename))
            top_funcs.sort(reverse=True)
            top5 = top_funcs[:5]

            print(f"[PROFILE] Tiempo total simulado: {total_time:.3f}s", flush=True)
            print(f"[PROFILE] Top-5 por tiempo acumulado:", flush=True)
            for ct, funcname, filename in top5:
                short_file = filename.split("\\")[-1] if "\\" in filename else filename.split("/")[-1]
                print(f"[PROFILE]   {ct*1000:.1f}ms  {funcname} ({short_file}:{ct:.3f}s)", flush=True)
            print(f"{'='*60}\n", flush=True)

            # Agregar header HTTP
            if isinstance(result, tuple):
                resp, status = result[0], result[1]
            else:
                resp, status = result, 200

            if hasattr(resp, "headers"):
                top_str = "; ".join([f"{fn}({ct*1000:.0f}ms)" for ct, fn, _ in top5])
                resp.headers["X-Profile"] = f"{endpoint_name} | {top_str}"[:400]
        except Exception as e:
            print(f"[PROFILE] Error en post-procesamiento: {e}", flush=True)
            traceback.print_exc()

        return result
    return wrapper


# ════════════════════════════════════════════════════════════════
# ERROR HANDLERS DEL BLUEPRINT
# ════════════════════════════════════════════════════════════════

@api_bp.errorhandler(400)
def _handle_bad_request(error: Any) -> tuple[Response, int]:
    """Maneja errores 400 (Bad Request) producidos por Flask."""
    return _error_response(str(error), status_code=400)


@api_bp.errorhandler(413)
def _handle_entity_too_large(error: Any) -> tuple[Response, int]:
    """Maneja errores 413 (Request Entity Too Large) — payload > 50MB."""
    return _error_response(
        "El cuerpo de la solicitud excede el tamaño máximo permitido (50MB)",
        status_code=413,
    )


@api_bp.errorhandler(415)
def _handle_unsupported_media(error: Any) -> tuple[Response, int]:
    """Maneja errores 415 (Unsupported Media Type)."""
    return _error_response(
        "Tipo de contenido no soportado. Use application/json.",
        status_code=415,
    )


@api_bp.errorhandler(429)
def _handle_rate_limit(error: Any) -> tuple[Response, int]:
    """Maneja errores 429 (Too Many Requests) del rate limiter."""
    return _error_response(
        "Demasiadas solicitudes. Intente de nuevo más tarde.",
        status_code=429,
    )


@api_bp.errorhandler(500)
def _handle_internal_error(error: Any) -> tuple[Response, int]:
    """Maneja errores 500 no capturados."""
    print(f"[api] ERROR 500 no capturado: {error}", flush=True)
    traceback.print_exc()
    return _error_response("Error interno del servidor", status_code=500)


# ════════════════════════════════════════════════════════════════
# HELPERS DE MEMORIA
# ════════════════════════════════════════════════════════════════

def _get_memory_mb() -> float:
    """Returns current RSS memory in MB, or 0 if unavailable."""
    try:
        return float(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return 0.0


def _ocr_with_unlimited(img_bgr: _Img) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """OCR con Unlimited-OCR vía daemon persistente (127.0.0.1:5177).

    El daemon (uocr_daemon.py, venv env_uocr_gpu) ya tiene el modelo 4-bit
    precargado en background al arrancar — aquí solo se envía la imagen.
    Retorna (blocks, image_panels, t_ocr_s):
      - blocks: bloques de texto útiles (ruido de página filtrado).
      - image_panels: rects de bloques type="image" grandes (≥15% de la
        página) — la materia prima de la Ruta C (re-OCR a nivel globo).
    Lanza RuntimeError si el modelo no está listo o el daemon falla.
    """
    import tempfile
    import uocr_client

    h = uocr_client.health()
    uocr_state = h.get("state")
    if uocr_state != "ready":
        if uocr_state == "error":
            # El daemon arrancó pero falló al cargar el modelo (VRAM insuficiente,
            # cuDNN, cuantización...): log distinto para que no pase desapercibido.
            print(f"[unlimited] DAEMON EN ESTADO ERROR: {h.get('error') or 'sin detalle'} "
                  f"— degradando a EasyOCR permanentemente esta sesión")
        else:
            print(f"[unlimited] Daemon no listo (estado: {uocr_state}) — "
                  f"el modelo se precarga en background (~8 min)")
        raise RuntimeError(
            f"Unlimited-OCR no listo (estado: {uocr_state}). "
            f"El modelo se precarga en background al arrancar (~8 min); reintenta en un momento."
        )

    gpu_snapshot = gpu_memory_snapshot()
    if not gpu_budget_allows(
        gpu_snapshot,
        required_free_mb=GPU_MIN_FREE_VRAM_MB,
        budget_mb=GPU_VRAM_BUDGET_MB,
    ):
        print(f"[unlimited] VRAM insuficiente: {gpu_snapshot} — se omite esta inferencia")
        raise RuntimeError(
            f"VRAM insuficiente para Unlimited-OCR ({gpu_snapshot.get('free_mb')}MB libres)"
        )

    fd, tmp = tempfile.mkstemp(suffix=".png", prefix="uocr_page_")
    os.close(fd)
    try:
        cv2.imwrite(tmp, img_bgr)
        # v4.2: serializar GPU — mientras el daemon U-OCR infiere (60-110s),
        # ningún EasyOCR del server debe correr (ambos comparten la GTX 1050 Ti
        # de 4GB). El daemon es un proceso separado; este lock del server hace
        # que los workers de EasyOCR esperen a que termine la inferencia U-OCR.
        # Benchmark 2026-08-03: sin esto, el daemon pasaba de 83s a 140-1439s
        # por contención de VRAM con EasyOCR del server.
        from ocr_utils import _gpu_lock, _uocr_inferring
        _uocr_inferring.set()  # v4.2: otros workers degradan a RapidOCR CPU
        try:
            with _gpu_lock:
                res = uocr_client.process_page(tmp)
        finally:
            _uocr_inferring.clear()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    if res.get("error"):
        raise RuntimeError(res["error"])

    blocks, image_panels = _parse_daemon_blocks(res.get("blocks", []), img_bgr)
    return blocks, image_panels, float(res.get("infer_s", 0.0))


def _ocr_with_unlimited_batch(
    img_bgrs: list[_Img],
) -> tuple[list[tuple[list[dict[str, Any]], list[dict[str, Any]]]], float]:
    """OCR de VARIAS páginas con Unlimited-OCR en UNA inferencia VLM (Fase 1).

    Usa uocr_client.process_batch() → POST /ocr-batch del daemon, que ejecuta
    _model.infer_multi() — las N imágenes comparten el prefill del modelo,
    amortizando el costo por página (~60-110s c/u en individual).

    Retorna (pages, infer_s) donde pages[i] = (blocks, image_panels) de la
    imagen i (mismo orden de entrada). Lanza RuntimeError si el modelo no
    está listo o el daemon falla.
    """
    import tempfile
    import uocr_client

    h = uocr_client.health()
    uocr_state = h.get("state")
    if uocr_state != "ready":
        if uocr_state == "error":
            print(f"[unlimited] DAEMON EN ESTADO ERROR: {h.get('error') or 'sin detalle'} "
                  f"— degradando a EasyOCR permanentemente esta sesión")
        else:
            print(f"[unlimited] Daemon no listo (estado: {uocr_state}) — "
                  f"el modelo se precarga en background (~8 min)")
        raise RuntimeError(
            f"Unlimited-OCR no listo (estado: {uocr_state}). "
            f"El modelo se precarga en background al arrancar (~8 min); reintenta en un momento."
        )

    gpu_snapshot = gpu_memory_snapshot()
    if not gpu_budget_allows(
        gpu_snapshot,
        required_free_mb=GPU_MIN_FREE_VRAM_MB,
        budget_mb=GPU_VRAM_BUDGET_MB,
    ):
        print(f"[unlimited] VRAM insuficiente para batch: {gpu_snapshot} — se omite")
        raise RuntimeError(
            f"VRAM insuficiente para Unlimited-OCR ({gpu_snapshot.get('free_mb')}MB libres)"
        )

    tmp_paths: list[str] = []
    try:
        for img_bgr in img_bgrs:
            fd, tmp = tempfile.mkstemp(suffix=".png", prefix="uocr_batch_")
            os.close(fd)
            cv2.imwrite(tmp, img_bgr)
            tmp_paths.append(tmp)
        # v4.2: serializar GPU — mientras el daemon U-OCR infiere (60-110s),
        # ningún EasyOCR del server debe correr (ambos comparten la GTX 1050 Ti
        # de 4GB). Mismo lock que _ocr_with_unlimited.
        from ocr_utils import _gpu_lock, _uocr_inferring
        _uocr_inferring.set()
        try:
            with _gpu_lock:
                res = uocr_client.process_batch(tmp_paths)
        finally:
            _uocr_inferring.clear()
    finally:
        for tmp in tmp_paths:
            try:
                os.remove(tmp)
            except OSError:
                pass

    if res.get("error"):
        raise RuntimeError(res["error"])

    pages_raw = res.get("pages", [])
    if len(pages_raw) != len(img_bgrs):
        raise RuntimeError(
            f"daemon devolvió {len(pages_raw)} páginas, esperaba {len(img_bgrs)}"
        )
    pages: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
    for page_raw, img_bgr in zip(pages_raw, img_bgrs):
        blocks, image_panels = _parse_daemon_blocks(page_raw.get("blocks", []), img_bgr)
        pages.append((blocks, image_panels))
    return pages, float(res.get("infer_s", 0.0))


def _finalize_page_blocks(
    img_bgr: _Img,
    blocks: list[dict[str, Any]],
    source: str,
    target: str,
    scale_x: float,
    scale_y: float,
    detected_lang: str,
    doc_id: str = "",
    response_format: str = "png",
) -> tuple[list[dict[str, Any]], str | None, str, float]:
    """Pipeline post-OCR compartido por /process-page y /process-page-batch:
    filtro de watermarks → re-detección de idioma → inpainting → traducción →
    armado de bloques de respuesta.

    Retorna (result_blocks, inpainted_b64, detected_lang_final, t_inpaint).
    Si no quedan bloques tras filtrar watermarks, devuelve ([], b64 del
    original, detected_lang, 0.0) — el endpoint decide el jsonify.

    response_format ("png"|"jpeg"): formato de la imagen inpaintada. La
    imagen solo se muestra en el cliente, así que JPEG (95) es 5-10x más
    chico que PNG; "png" queda como fallback compatible.
    """
    _img_fmt = ".jpg" if response_format == "jpeg" else ".png"
    from ocr_utils import (
        _build_inpaint_mask, _inpaint_image, _sample_bg_color,
        _filter_watermarks_from_blocks, _cv2_to_base64,
    )
    from translator import (
        _detect_language_robust,
        _detect_mixed_languages,
        _es_traduccion_valida,
        _translation_is_likely_source_language,
    )
    from server import _get_executor

    blocks = _filter_watermarks_from_blocks(blocks)
    if not blocks:
        return [], _cv2_to_base64(img_bgr, fmt=_img_fmt), detected_lang, 0.0

    # ── Re-detección de idioma post-OCR ────────────────────────
    block_languages: list[str] = [detected_lang] * len(blocks)
    if source == "auto" and blocks:
        combined_text = " ".join([str(b.get("text", "")) for b in blocks])
        page_fallback_lang = detected_lang
        try:
            page_fallback_lang = (
                _detect_language_robust(combined_text) or detected_lang)
            if page_fallback_lang == "auto":
                page_fallback_lang = detected_lang
        except Exception as exc:
            # El combinado es solo una pista; no debe contaminar los bloques
            # si falla por un texto raro o por un error puntual del detector.
            print(f"[process-page] Detección combinada omitida: {exc}")

        for index, block in enumerate(blocks):
            block_text = str(block.get("text", "")).strip()
            try:
                block_languages[index] = (
                    _detect_language_robust(block_text)
                    if block_text else page_fallback_lang
                ) or page_fallback_lang
            except Exception:
                block_languages[index] = page_fallback_lang
        counts: dict[str, int] = {}
        for block_lang in block_languages:
            counts[block_lang] = counts.get(block_lang, 0) + 1
        if counts:
            max_count = max(counts.values())
            tied = {lang for lang, count in counts.items() if count == max_count}
            if page_fallback_lang in tied:
                # El texto combinado es independiente del orden de los
                # globos y desempata de forma estable páginas mixtas.
                detected_lang = page_fallback_lang
            else:
                # Si el detector combinado no coincide con los candidatos,
                # conserva el primer idioma de la página como desempate
                # determinista, sin usar el orden del dict de conteos.
                detected_lang = next(
                    lang for lang in block_languages if lang in tied)
        else:
            detected_lang = page_fallback_lang
        print(f"[process-page] Idioma dominante={detected_lang}; "
              f"idiomas por bloque={block_languages}")

    # ── Inpainting ─────────────────────────────────────────────
    # OpenCV libera el GIL durante el trabajo pesado y la traduccion ya se
    # ejecuta en el executor compartido. Solapar ambos tramos reduce la
    # latencia sin cambiar el resultado: la imagen se espera antes de calcular
    # colores de fondo y construir la respuesta.
    inpainted: _Img | None = None
    inpainted_b64: str | None = None
    inpaint_result: list[tuple[_Img, str]] = []
    inpaint_error: list[Exception] = []
    inpaint_elapsed_result: list[float] = []
    t_inpaint_before = _time.time()

    def _run_inpaint_parallel() -> None:
        worker_started = _time.perf_counter()
        try:
            mask = _build_inpaint_mask(img_bgr, blocks)
            image_result = _inpaint_image(img_bgr, mask, blocks)
            encoded_result = _cv2_to_base64(image_result, fmt=_img_fmt)
            inpaint_result.append((image_result, encoded_result))
        except Exception as inpaint_err:
            inpaint_error.append(inpaint_err)
            inpaint_result.append((img_bgr, _cv2_to_base64(img_bgr, fmt=_img_fmt)))
        finally:
            inpaint_elapsed_result.append(
                _time.perf_counter() - worker_started)

    inpaint_thread = threading.Thread(
        target=_run_inpaint_parallel,
        name="inpaint-page",
        daemon=True,
    )
    inpaint_thread.start()

    # ── Traducción ─────────────────────────────────────────────
    source_texts = [str(b["text"]) for b in blocks]
    translated_texts = list(source_texts)
    # TraducciÃ³n memoizada por pÃ¡gina: dos globos que repiten exactamente el
    # mismo texto deben recibir el mismo resultado y no competir por Google.
    # AdemÃ¡s reduce llamadas CT2/HTTP en pÃ¡ginas con nombres o SFX repetidos.
    memory = get_document_memory(doc_id)
    unique_texts: dict[tuple[str, str, str, str], str] = {}
    translated_by_key: dict[tuple[str, str, str, str], str] = {}
    quality_by_key: dict[tuple[str, str, str, str], float] = {}
    for index, (block, text) in enumerate(zip(blocks, source_texts)):
        normalized = " ".join(text.split())
        if normalized:
            block_type = _safe_str(block.get("type"), default="text").strip().lower() or "text"
            block_lang = block_languages[index]
            key = (normalized, block_lang, target, block_type)
            try:
                quality_by_key[key] = max(
                    quality_by_key.get(key, 0.0),
                    max(0.0, min(1.0, float(block.get("confidence", 0.0)))),
                )
            except (TypeError, ValueError):
                quality_by_key.setdefault(key, 0.0)
            if memory is not None and block_type != "sfx":
                cached = memory.lookup(normalized, block_lang, target)
                if cached is not None and not _document_memory_translation_is_safe(
                        normalized, cached, block_lang, target):
                    if hasattr(memory, "discard"):
                        memory.discard(normalized, block_lang, target)
                    cached = None
                if cached is None and quality_by_key.get(key, 0.0) >= 0.75:
                    # La variante fuzzy nunca se usa con OCR dudoso. El umbral
                    # evita convertir un fragmento artístico en otro nombre.
                    cached = memory.lookup_variant(
                        normalized, block_lang, target)
                    if cached is not None and not _document_memory_translation_is_safe(
                            normalized, cached, block_lang, target):
                        cached = None
                if cached is not None:
                    translated_by_key[key] = cached
                    continue
            unique_texts.setdefault(key, normalized)

    # ── Optimización 2.6: pre-paso CT2 en lote ───────────────────
    # En vez de una translate_batch por bloque (cada uno dentro del
    # executor), se agrupan los textos únicos con par (block_lang, target)
    # cubierto por CT2 en UNA translate_batch por par — el prefill del
    # modelo se comparte y N textos cuestan mucho menos que N llamadas.
    # Solo se conservan los resultados que pasan la MISMA validación que
    # el ensamblado aplica después; los None/inválidos quedan en
    # unique_texts y el flujo individual los reintenta (Google fallback).
    # Se replica el fast path de _translate_one (limpieza, glosario,
    # title-case, post-process, honoríficos) para no divergir en calidad.
    if unique_texts:
        from translator import (
            _translate_ctranslate2_batch,
            _post_process_translation,
            _preservar_honorificos,
            _aplicar_glosario,
        )
        ct2_buckets: dict[tuple[str, str], list[tuple[Any, str, bool]]] = {}
        ct2_order: list[tuple[str, str]] = []
        for key, original in list(unique_texts.items()):
            block_lang, tgt = key[1], key[2]
            block_type = key[3]
            if block_type == "sfx" or block_lang == tgt:
                continue
            cleaned = re.sub(
                r'[@#$%^&*()+={}\[\]|:;<>/\\]', "", original).strip()
            if not cleaned or not any(c.isalpha() for c in cleaned):
                continue
            # Glosario y title-case idénticos al fast path de _translate_one.
            if source == "es" or source == "auto":
                cleaned = _aplicar_glosario(cleaned)
            is_caps = (
                cleaned.isupper()
                and any(c.isalpha() for c in cleaned)
                and len(cleaned) > 1
            )
            query = cleaned.title() if is_caps else cleaned
            pair = (block_lang, tgt)
            if pair not in ct2_buckets:
                ct2_buckets[pair] = []
                ct2_order.append(pair)
            ct2_buckets[pair].append((key, query, is_caps))

        for (block_lang, tgt), items in ct2_buckets.items():
            queries = [q for _, q, _ in items]
            batch_results = _translate_ctranslate2_batch(
                queries, block_lang, tgt)
            for (key, query, is_caps), result in zip(items, batch_results):
                if not result:
                    continue
                translated = result.upper() if is_caps else result
                try:
                    translated = _post_process_translation(
                        translated, block_lang, tgt)
                    translated = _preservar_honorificos(
                        query, translated, block_lang, tgt)
                except Exception as post_e:
                    print(f"[process-page] Post-proceso CT2 batch: {post_e}")
                # Mismas reglas del ensamblado: si la traducción parece
                # inválida o queda en idioma fuente, se descarta y el flujo
                # individual la reintenta (Google).
                preserved_semantic = _allow_preserved_name_or_sfx(
                    key[0], translated, key[3])
                invalid_translation = (
                    not _es_traduccion_valida(
                        key[0],
                        translated,
                        lenient=len(key[0].split()) <= 3,
                    )
                    and not preserved_semantic
                )
                source_language_output = (
                    not preserved_semantic
                    and _translation_is_likely_source_language(
                        translated, key[1], key[2])
                )
                if invalid_translation or source_language_output:
                    continue
                translated_by_key[key] = translated
                unique_texts.pop(key, None)

    if unique_texts:
        executor = _get_executor()
        fut = {
            executor.submit(_translate_ocr_block, original, key[1], key[2], key[3]): key
            for key, original in unique_texts.items()
        }
        for future in as_completed(fut):
            key = fut[future]
            try:
                translated = str(future.result() or "")
                preserved_semantic = _allow_preserved_name_or_sfx(
                    key[0], translated, key[3])
                invalid_translation = (
                    not _es_traduccion_valida(
                        key[0],
                        translated,
                        lenient=len(key[0].split()) <= 3,
                    )
                    and not preserved_semantic
                )
                source_language_output = (
                    not preserved_semantic
                    and _translation_is_likely_source_language(
                        translated, key[1], key[2])
                )
                if invalid_translation or source_language_output:
                    print(
                        f"[process-page] Traducción rechazada en ensamblado: "
                        f"{translated[:50]!r}"
                    )
                    translated = key[0]
                translated_by_key[key] = translated
                if memory is not None and key[3] != "sfx":
                    _learn_translation_if_valid(
                        memory,
                        key[0],
                        translated,
                        key[1],
                        key[2],
                        quality_by_key.get(key, 0.0),
                    )
            except Exception as e:
                print(f"[process-page] Error traduciendo texto {key[0][:40]!r}: {e}")
    if memory is not None:
        memory.save()

    inpaint_thread.join(timeout=120.0)
    if inpaint_thread.is_alive():
        print("[process-page] Inpainting paralelo excedio 120s; usando original")
        inpainted = img_bgr
        inpainted_b64 = _cv2_to_base64(img_bgr, fmt=_img_fmt)
    else:
        if inpaint_error:
            print(f"[process-page] Inpainting error: {inpaint_error[0]}, usando original")
        if inpaint_result:
            inpainted, inpainted_b64 = inpaint_result[0]
        else:
            inpainted = img_bgr
            inpainted_b64 = _cv2_to_base64(img_bgr, fmt=_img_fmt)
    t_inpaint = (
        inpaint_elapsed_result[0]
        if inpaint_elapsed_result
        else _time.time() - t_inpaint_before
    )

    for idx, (block, text) in enumerate(zip(blocks, source_texts)):
        block_type = _safe_str(block.get("type"), default="text").strip().lower() or "text"
        key = (" ".join(text.split()), block_languages[idx], target, block_type)
        translated_texts[idx] = translated_by_key.get(key, text)

    # ── Armar respuesta ────────────────────────────────────────
    result_blocks: list[dict[str, Any]] = []
    ref_img = inpainted if inpainted is not None else img_bgr
    for i, block in enumerate(blocks):
        try:
            bg_color = _sample_bg_color(ref_img, block) if ref_img is not None else "#ffffff"
        except Exception:
            bg_color = "#ffffff"
        bx = int(block["x"] * scale_x) if scale_x != 1.0 else block["x"]
        by = int(block["y"] * scale_y) if scale_y != 1.0 else block["y"]
        bw = int(block["w"] * scale_x) if scale_x != 1.0 else block["w"]
        bh = int(block["h"] * scale_y) if scale_y != 1.0 else block["h"]
        try:
            font_size = max(1, int(round(float(block["fontSize"]) * scale_y)))
        except (KeyError, TypeError, ValueError):
            font_size = max(1, int(round(max(1, bh) * 0.8)))
        source_langs = list(_detect_mixed_languages(
            source_texts[i], dominant=block_languages[i]))
        result_blocks.append({
            "x": bx, "y": by, "w": bw, "h": bh,
            "source": source_texts[i],
            "translated": translated_texts[i],
            "fontSize": font_size,
            "textColor": block["textColor"],
            "bgColor": bg_color,
            "confidence": block["confidence"],
            "source_lang": block_languages[i],
            # Un bloque puede contener code-switching aunque tenga un idioma
            # dominante. Exponerlo permite al frontend/diagnóstico distinguir
            # "es" de "es + en" sin cambiar el contrato principal.
            "source_langs": source_langs,
            "mixed_source": len(source_langs) > 1,
            # Fase 3: tipo semántico (text/title/header) cuando el motor
            # lo emite (U-OCR); el frontend puede usarlo para filtros.
            "type": block.get("type", "text"),
        })
    return result_blocks, inpainted_b64, detected_lang, t_inpaint


def _parse_daemon_blocks(
    res_blocks: list[dict[str, Any]],
    img_bgr: _Img,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convierte bloques crudos del daemon en bloques del servidor.

    Compartido por el camino single (_ocr_with_unlimited) y el batch
    (_ocr_with_unlimited_batch) — Fase 1. Filtra ruido de página
    (image/footer/page_number), estima confianza y fontSize, y propaga el
    type semántico del VLM (Fase 3). Retorna (blocks, image_panels).
    """
    # Import local: evita dependencia circular y mantiene el parseo
    # autocontenido (lo usan también los modos fusion/unlimited).
    from ocr_utils import _estimate_confidence_heuristic

    # Tipos de bloques que son ruido de página (pie de página, nº de página):
    # no traducirlos ni crear cajas con ellos. Los "header" se conservan porque
    # suelen ser títulos de capítulo legítimos; el frontend filtra el margen.
    _NOISE_TYPES = frozenset({"image", "footer", "page_number"})
    blocks: list[dict[str, Any]] = []
    image_panels: list[dict[str, Any]] = []
    page_h, page_w = img_bgr.shape[:2]
    page_area = float(page_h * page_w)
    for b in res_blocks:
        try:
            bx, by, bw, bh = int(b["x"]), int(b["y"]), int(b["w"]), int(b["h"])
        except (KeyError, TypeError, ValueError):
            continue
        raw_x2, raw_y2 = bx + bw, by + bh
        if (
            bw <= 0 or bh <= 0
            or raw_x2 <= 0 or raw_y2 <= 0
            or bx >= page_w or by >= page_h
        ):
            continue
        # El VLM es una frontera de confianza: sus coordenadas pueden quedar
        # fuera de la página o ser negativas. Normalizar aquí evita que
        # inpainting y el frontend reciban cajas inválidas.
        x0, y0 = max(0, bx), max(0, by)
        x1, y1 = min(page_w, raw_x2), min(page_h, raw_y2)
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        bx, by, bw, bh = x0, y0, x1 - x0, y1 - y0
        if b.get("type") == "image":
            # Conservar paneles artísticos grandes: son la materia prima de la
            # Ruta C (detección de globos + re-OCR a nivel globo en el server).
            if bw * bh >= page_area * UOCR_IMAGE_BLOCK_RATIO:
                image_panels.append({"x": bx, "y": by, "w": bw, "h": bh})
            continue
        if not b.get("text") or b.get("type") in _NOISE_TYPES:
            continue
        btype = b.get("type", "text")
        fontSize = max(10, int(bh * 0.8))  # estimado desde altura del bloque
        blocks.append({
            "x": bx, "y": by, "w": bw, "h": bh,
            "text": str(b["text"]).strip(),
            # Fase 3: propagar el tipo semántico del VLM (text/title/header)
            # a los bloques — la votación de _fusionar_blocks_multi lo pondera
            # (title/header pesan más). Sin esto, el type se perdía aquí.
            "type": btype,
            # El modelo NO emite confianza (validado empíricamente: logits
            # saturan ~0.997 sin discriminar). Se estima por heurística.
            "confidence": _estimate_confidence_heuristic(b, btype),
            "fontSize": fontSize,
            "textColor": "#000000",
            "engine": "unlimited",
            "from_art_recrop": bool(b.get("from_art_recrop")),
        })
    return blocks, image_panels


# ════════════════════════════════════════════════════════════════
# ENDPOINTS
# ════════════════════════════════════════════════════════════════

@api_bp.get("/health")
def health() -> Any:
    from config import APP_VERSION, IS_PRODUCTION
    from translator import _argo_ready
    from server import DB_AVAILABLE, TRANSLATION_CACHE_AVAILABLE
    ready = [f"{s}->{t}" for (s, t), v in _argo_ready.items() if v]
    mem_mb = _get_memory_mb()
    zombie_state = _is_zombie()
    # Estado del daemon Unlimited-OCR (modelo 4-bit precargado en background)
    uocr_state = "offline"
    uocr_load_s: float | None = None
    try:
        import uocr_client
        uh = uocr_client.health()
        uocr_state = uh.get("state", "offline")
        uocr_load_s = uh.get("load_s")
    except Exception:
        pass
    return jsonify({
        "ok": not zombie_state,
        "zombie": zombie_state,
        "zombie_count": _ZOMBIE_COUNTER,
        "zombie_threshold": _ZOMBIE_THRESHOLD,
        "version": APP_VERSION,
        "mode": "production" if IS_PRODUCTION else "development",
        "mit_available": False,
        "db_available": DB_AVAILABLE,
        "cache_available": TRANSLATION_CACHE_AVAILABLE,
        "rate_limiting": RATE_LIMIT_AVAILABLE,
        "offline_models": ready,
        "memory": f"{mem_mb:.0f}MB" if mem_mb else "N/A",
        "unlimited_ocr": uocr_state,
        "uocr_load_s": uocr_load_s,
    })


@api_bp.post("/translate")
@_validate_payload_fields("text")
@profile_endpoint
def translate() -> Any:
    payload: dict[str, Any] = request.get_json(silent=True) or {}

    # ── Validar y sanitizar texto ──────────────────────────────
    text = _safe_str(payload.get("text"))
    if not text or len(text) < _MIN_TEXT_LENGTH:
        return jsonify({"translatedText": "", "engine": "none"})

    text_err = _validate_text_length(text)
    if text_err:
        return _error_response(text_err, status_code=413)

    # ── Validar idiomas (función compartida) ───────────────────
    source, target, err = _validate_lang_params(payload)
    if err:
        return _error_response(err, status_code=400)
    assert source is not None and target is not None

    doc_id = _sanitize_doc_id(payload.get("doc_id"))
    result = _translate_with_document_memory(text, source, target, doc_id)
    return jsonify({"translatedText": result, "engine": "auto"})


@api_bp.post("/translate-batch")
@_validate_payload_fields("texts")
@profile_endpoint
def translate_batch() -> Any:
    from server import _get_executor

    payload = request.get_json(silent=True) or {}
    texts_raw: Any = payload.get("texts", [])

    # ── Validar que texts sea una lista ────────────────────────
    if not isinstance(texts_raw, list):
        return _error_response(
            "El campo 'texts' debe ser una lista de strings",
            status_code=400,
            details={"field": "texts", "expected_type": "list"},
        )

    # ── Validar tamaño del batch ───────────────────────────────
    if not texts_raw:
        return jsonify({"results": []})
    if len(texts_raw) > MAX_BATCH_SIZE:
        return _error_response(
            f"Demasiados textos ({len(texts_raw)}, max {MAX_BATCH_SIZE})",
            status_code=413,
        )

    # ── Validar que cada elemento sea string ───────────────────
    # NOTA: Si se agregan nuevos endpoints con validación de source/target,
    # crear una función _validate_lang_params(payload, allow_source_auto) compartida
    # para evitar duplicación entre translate() y translate-batch().

    invalid_items = [
        (i, type(t).__name__) for i, t in enumerate(texts_raw)
        if not isinstance(t, str)
    ]
    if invalid_items:
        examples = [f"índice {i} es {t}" for i, t in invalid_items[:5]]
        return _error_response(
            "Todos los elementos de 'texts' deben ser strings",
            status_code=400,
            details={"invalid_items": examples, "total_invalid": len(invalid_items)},
        )

    # ── Validar idiomas (función compartida) ───────────────────
    source, target, err = _validate_lang_params(payload)
    if err:
        return _error_response(err, status_code=400)
    assert source is not None and target is not None

    # ── Validar largo individual de cada texto ─────────────────
    oversized = []
    for i, t in enumerate(texts_raw):
        err = _validate_text_length(t)
        if err:
            oversized.append({"index": i, "error": err})
    if oversized:
        return _error_response(
            "Uno o más textos exceden los límites",
            status_code=413,
            details={"oversized": oversized[:10]},
        )

    doc_id = _sanitize_doc_id(payload.get("doc_id"))
    results = list(texts_raw)
    unique_texts: dict[str, str] = {}
    for text in texts_raw:
        normalized = " ".join(text.split())
        if normalized:
            unique_texts.setdefault(normalized, normalized)

    executor = _get_executor()
    memory = get_document_memory(doc_id)
    futures = {
        executor.submit(
            _translate_with_document_memory,
            original,
            source,
            target,
            doc_id,
            False,
        ): key
        for key, original in unique_texts.items()
    }
    translated_by_key: dict[str, str] = {}
    for future in as_completed(futures):
        key = futures[future]
        try:
            translated_by_key[key] = future.result()
        except Exception as e:
            print(f"[translate-batch] Error traduciendo texto {key[:40]!r}: {e}")
    for idx, text in enumerate(texts_raw):
        key = " ".join(text.split())
        if key:
            results[idx] = translated_by_key.get(key, text)
    if memory is not None:
        memory.save()
    return jsonify({"results": results})


@api_bp.post("/process-page")
@_validate_payload_fields("image", allow_binary=True)
@profile_endpoint
def process_page() -> Any:
    from ocr_utils import (
        _base64_to_cv2, _bytes_to_cv2, _cv2_to_base64,
        _build_inpaint_mask, _inpaint_image, _sample_bg_color,
        _filter_watermarks_from_blocks,
    )
    from server import _translate_one
    from translator import _detect_language_robust
    from server import _get_executor
    from config import MAX_IMAGE_DIMENSION, LANGUAGES

    # ── Transporte: JSON (base64, legacy) o binario ─────────────
    # Optimización 2.4: el cliente puede enviar el canvas como cuerpo
    # binario (canvas.toBlob) con los flags en el query string — elimina el
    # +33% del base64 y las dos conversiones CPU por página. Si el mimetype
    # es JSON, se conserva el flujo legacy (tests y clientes externos).
    is_binary = request.mimetype != "application/json" and request.data
    if is_binary:
        payload = {
            k: request.args.get(k)
            for k in ("target", "source", "ocr_mode", "prefilter",
                      "force_uocr", "disable_uocr", "pure_easyocr",
                      "doc_id", "response_format")
        }
        payload = {k: v for k, v in payload.items() if v is not None}
        raw_image: bytes = request.data
        b64_image = ""
    else:
        payload = request.get_json(silent=True) or {}
        b64_image = _safe_str(payload.get("image"))
        raw_image = b""

    # ── Validar que la imagen no esté vacía ────────────────────
    if not b64_image and not raw_image:
        return _error_response("No se proporcionó imagen", status_code=400)

    # ── Validar tamaño aproximado (evitar OOM) ─────────────────
    image_len = len(raw_image) if raw_image else len(b64_image)
    if image_len > MAX_IMAGE_BYTES:
        return _error_response(
            f"Imagen demasiado grande ({image_len} bytes, "
            f"max {MAX_IMAGE_BYTES})",
            status_code=413,
        )

    # ── Validar idioma destino ─────────────────────────────────
    target_raw = _safe_str(payload.get("target"), default="en")
    target = _validate_lang_code(target_raw, allow_auto=False)
    if target is None:
        return _error_response(
            f"Idioma destino no soportado: '{target_raw}'",
            status_code=400,
        )

    # ── Validar idioma origen ──────────────────────────────────
    source_raw = _safe_str(payload.get("source"), default="auto")
    source = _validate_lang_code(source_raw)
    if source is None:
        return _error_response(
            f"Idioma origen no soportado: '{source_raw}'",
            status_code=400,
        )

    # ── Validar modo OCR ───────────────────────────────────────
    # Fase 4: el default es "fusion" — híbrido EasyOCR+RapidOCR siempre,
    # Unlimited-OCR (daemon) solo si el trigger v4.2 lo decide.
    ocr_mode = _safe_str(payload.get("ocr_mode"), default="fusion").lower()
    if ocr_mode not in ("easyocr", "auto", "unlimited", "fusion"):
        return _error_response(
            f"Modo OCR no soportado: '{ocr_mode}'. "
            f"Use 'easyocr', 'auto', 'unlimited' o 'fusion'.",
            status_code=400,
        )

    # ── Validar prefilter (limpieza morfológica pre-OCR) ───────
    prefilter: bool = _coerce_bool(payload.get("prefilter"), default=True)

    # ── Validar force_uocr (debug/test: fuerza el refuerzo U-OCR + Ruta C
    #    en modo fusion aunque el trigger no dispare) ────────────────
    force_uocr: bool = _coerce_bool(payload.get("force_uocr"), default=False)

    # ── Validar disable_uocr (benchmark: desactiva el refuerzo U-OCR en
    #    modo fusion → solo EasyOCR+RapidOCR+merge, para medir el overhead
    #    puro de la fusión sin el costo de la inferencia VLM) ───────
    disable_uocr: bool = _coerce_bool(payload.get("disable_uocr"), default=False)

    # ── Validar pure_easyocr (benchmark: desactiva el tier híbrido RapidOCR
    #    en el modo easyocr → solo EasyOCR GPU puro. Sin esto, el "modo
    #    easyocr" de la app YA ejecuta EasyOCR+RapidOCR+_fusionar_blocks,
    #    por lo que comparar contra él no mide el overhead de la fusión) ──
    pure_easyocr: bool = _coerce_bool(payload.get("pure_easyocr"), default=False)

    # ── doc_id (sesión 126): identificador de documento/sesión que escopea
    #    la firma de los caches de decisión (trigger + §8.4.1). La sesión 124
    #    midió 94% de colisión de firma entre capítulos de la MISMA serie —
    #    sin scope, el capítulo 47 heredaría las decisiones del 43. El caller
    #    (process_all_pages.py, app.js) deriva el doc_id del PDF/archivo;
    #    vacío → scope legacy compartido (comportamiento previo). Se trunca a
    #    64 chars y se sanea a [A-Za-z0-9_] para mantener claves canónicas
    #    (un ":" en doc_id rompería el prefijo "doc_id:firma" de los caches).
    doc_id: str = re.sub(r"[^A-Za-z0-9_]", "", _safe_str(payload.get("doc_id")))[:64]

    # ── Validar response_format (optimización 2.3): la imagen inpaintada
    #    solo se muestra en el cliente, así que el caller puede pedirla en
    #    JPEG (5-10x más chica) sin perder nada; "png" es el fallback
    #    compatible con el comportamiento previo.
    response_format = _safe_str(payload.get("response_format"), default="png").lower()
    if response_format == "jpg":
        response_format = "jpeg"
    if response_format not in ("png", "jpeg"):
        return _error_response(
            f"response_format no soportado: '{response_format}'. "
            f"Use 'png' o 'jpeg'.",
            status_code=400,
        )
    _img_fmt = ".jpg" if response_format == "jpeg" else ".png"

    scale_x: float = 1.0
    scale_y: float = 1.0
    mem_before = _get_memory_mb()

    # ── Cache de página completa (optimización 2.7) ────────────
    # Si esta misma imagen (mismos bytes + params) ya se procesó, devolver
    # la respuesta guardada sin re-hacer OCR + inpaint + traducción. Re-correr
    # el mismo capítulo pasa de minutos a milisegundos. La consulta va
    # DESPUÉS de las validaciones (idioma, modo, tamaño, decode) para que
    # los errores 400/413 siempre se evalúen antes de servir cache.
    cache_bytes: bytes = raw_image if raw_image else b64_image.encode("utf-8")
    cache_key = _page_cache_key(
        cache_bytes,
        target=target,
        source=source,
        ocr_mode=ocr_mode,
        prefilter=prefilter,
        force_uocr=force_uocr,
        disable_uocr=disable_uocr,
        pure_easyocr=pure_easyocr,
        doc_id=doc_id,
        response_format=response_format,
    )

    try:
        t0 = _time.time()

        # ── Decodificar imagen ─────────────────────────────────
        if raw_image:
            img_bgr = _bytes_to_cv2(raw_image)
            if img_bgr is None:
                return _error_response(
                    "No se pudo decodificar la imagen (bytes inválidos)",
                    status_code=400)
        else:
            img_bgr = _base64_to_cv2(b64_image)
            if img_bgr is None:
                return _error_response(
                    "No se pudo decodificar la imagen (base64 inválido)",
                    status_code=400)
        orig_h, orig_w = img_bgr.shape[:2]

        # ── Validar dimensiones mínimas ────────────────────────
        if orig_w < 50 or orig_h < 50:
            return _error_response(
                f"Imagen demasiado pequeña ({orig_w}x{orig_h}). Mínimo 50x50 px.",
                status_code=400,
            )

        # ── Cache lookup (después de decode + validaciones) ────
        cached = _page_cache_get(cache_key)
        if cached is not None:
            print(f"[process-page] CACHE HIT (página completa) para {cache_key[:12]}")
            return jsonify(cached)

        # ── Safety resize para imágenes >4096px ────────────────
        if max(orig_w, orig_h) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = float(orig_w) / float(new_w)
            scale_y = float(orig_h) / float(new_h)
            print(f"[process-page] Imagen redimensionada: {orig_w}x{orig_h} → {new_w}x{new_h} (scale={scale:.3f})")

        print(f"[process-page] Imagen {img_bgr.shape[1]}x{img_bgr.shape[0]}, target={target}, source={source}")

        # ── Detección de idioma (optimizada) ───────────────────
        detected_lang: str = source if source in LANGUAGES and source != "auto" else "es"
        if source == "auto":
            print(f"[process-page] Source=auto, asumiendo es temporalmente (se corrige post-OCR)")

        # En auto no fijar el OCR al lector latino: OCRManager usa esta marca
        # para habilitar recuperación selectiva de kana/hangul/hanzi cuando
        # RapidOCR encuentra evidencia de una página multilingüe.
        ocr_lang = (
            "auto" if source == "auto"
            else "es" if detected_lang in ("es", "spa", "spanish", "espanol")
            else detected_lang
        )

        # ── OCR ────────────────────────────────────────────────
        # Delega en OCRManager (ocr_engine.py): orquesta EasyOCR + RapidOCR +
        # Unlimited-OCR con el trigger selectivo v4.2. Retorna
        # (blocks, ocr_engine_used, engines_used) — el mismo contrato que el
        # código inline anterior (modos: easyocr, auto, unlimited, fusion).
        from ocr_engine import OCRManager
        t_ocr_before = _time.time()
        ocr_manager = OCRManager()
        blocks, ocr_engine_used, engines_used = ocr_manager.run_ocr(
            img_bgr,
            ocr_lang,
            ocr_mode=ocr_mode,
            prefilter=prefilter,
            force_uocr=force_uocr,
            disable_uocr=disable_uocr,
            pure_easyocr=pure_easyocr,
            doc_id=doc_id,
        )
        ocr_diagnostics = ocr_manager.last_diagnostics
        t_ocr = _time.time() - t_ocr_before
        print(f"[process-page] OCR ({ocr_engine_used}): {len(blocks)} bloques en {t_ocr:.1f}s")

        # ── Watchdog: detección de zombie ──────────────────────
        # Si OCR devuelve 0 bloques en <2s, podría ser un proceso zombie
        # con estado corrupto (EasyOCR no cargado, modelo no funcional, etc.)
        if len(blocks) == 0 and t_ocr < _ZOMBIE_FAST_TIME:
            reached = _increment_zombie_counter()
            if reached:
                inpainted_b64: str | None = _cv2_to_base64(img_bgr, fmt=_img_fmt)
                img_bgr = None
                return jsonify({
                    "error": "ZOMBIE_SERVER",
                    "message": "El servidor parece estar en estado zombie (múltiples OCRs "
                              f"rápidos sin resultados). El último request ({len(blocks)} "
                              f"bloques en {t_ocr:.1f}s) sugiere que EasyOCR no está "
                              "funcionando correctamente. Reinicia el servidor para "
                              "restaurar el funcionamiento.",
                    "blocks": [],
                    "inpainted_image": inpainted_b64,
                    "diagnostics": ocr_diagnostics,
                }), 200
        elif len(blocks) > 0 or t_ocr >= _ZOMBIE_FAST_TIME:
            # Respuesta normal: resetear contador
            _reset_zombie_counter()

        # ── Pipeline post-OCR compartido (single y batch): filtro de
        #    watermarks → re-detección de idioma → inpainting → traducción →
        #    armado de bloques de respuesta ────────────────────────────
        result_blocks, inpainted_b64, detected_lang, t_inpaint = _finalize_page_blocks(
            img_bgr, blocks, source, target, scale_x, scale_y, detected_lang, doc_id,
            response_format)
        if not result_blocks:
            img_bgr = None
            return jsonify({"inpainted_image": inpainted_b64, "blocks": [],
                            "ocr_engine": ocr_engine_used,
                            "engines_used": engines_used,
                            "diagnostics": ocr_diagnostics})

        # ── Cleanup ────────────────────────────────────────────
        img_bgr = None

        mem_after = _get_memory_mb()
        mem_growth = mem_after - mem_before
        total_t = _time.time() - t0
        print(f"[process-page] Completado. {len(result_blocks)} bloques, {total_t:.1f}s "
              f"(OCR:{t_ocr:.1f}s + inpaint:{t_inpaint:.1f}s), "
              f"memoria: {mem_before:.0f}→{mem_after:.0f}MB ({mem_growth:+.0f}MB)")
        page_response = {
            "inpainted_image": inpainted_b64,
            "blocks": result_blocks,
            "ocr_engine": ocr_engine_used,
            "engines_used": engines_used,
            "diagnostics": ocr_diagnostics,
        }
        _page_cache_set(cache_key, page_response)
        return jsonify(page_response)

    except MemoryError:
        print(f"[process-page] MEMORY ERROR! Forzando limpieza total...")
        from ocr_utils import _ocr_readers
        _ocr_readers.clear()
        gc.collect()
        return _error_response("Memoria insuficiente. Reintente.", status_code=500)

    except Exception as e:
        print(f"[process-page] ERROR: {e}")
        traceback.print_exc()
        # NOTA: No hacemos gc.collect() aquí — el GC de Python corre automáticamente.
        # Solo se hace gc.collect() en el MemoryError handler (arriba).
        # No devolver excepciones internas al cliente: pueden revelar rutas,
        # nombres de modelos o detalles de la instalación local.
        return _error_response(
            "Error interno procesando la página", status_code=500)


@api_bp.post("/process-page-batch")
@_validate_payload_fields("images")
@profile_endpoint
def process_page_batch() -> Any:
    """Procesa VARIAS páginas en un solo request (Fase 1 — batch U-OCR).

    Body: {"images": [b64, ...] (1-4 páginas), "target", "source",
           "ocr_mode" (default fusion), "prefilter", "force_uocr",
           "disable_uocr", "pure_easyocr", "doc_id" (sesión 126)}.

    Delega en OCRManager.run_ocr_batch(): cada página corre el híbrido + el
    trigger v4.2 + Fase 2, y TODAS las páginas que necesitan el VLM se envían
    al daemon en UN solo /ocr-batch (infer_multi) — el prefill del modelo se
    comparte, amortizando ~60-110s por página.

    Respuesta: {"results": [{inpainted_image, blocks, ocr_engine,
    engines_used}, ...]} — una entrada por imagen en el MISMO orden.
    """
    from ocr_utils import _base64_to_cv2
    from config import MAX_IMAGE_DIMENSION, LANGUAGES

    payload = request.get_json(silent=True) or {}
    images_raw = payload.get("images") or []
    if not isinstance(images_raw, list) or not 1 <= len(images_raw) <= 4:
        return _error_response(
            "Se requieren entre 1 y 4 imágenes en 'images'",
            status_code=400,
        )

    # ── Validar tamaño de cada imagen (evitar OOM) ────────────
    for b64_image in images_raw:
        b64_image = _safe_str(b64_image)
        if not b64_image:
            return _error_response("Imagen vacía en 'images'", status_code=400)
        if len(b64_image) > MAX_IMAGE_BYTES:
            return _error_response(
                f"Imagen demasiado grande ({len(b64_image)} bytes base64, "
                f"max {MAX_IMAGE_BYTES})",
                status_code=413,
            )

    # ── Validar idioma destino ─────────────────────────────────
    target_raw = _safe_str(payload.get("target"), default="en")
    target = _validate_lang_code(target_raw, allow_auto=False)
    if target is None:
        return _error_response(
            f"Idioma destino no soportado: '{target_raw}'",
            status_code=400,
        )

    # ── Validar idioma origen ──────────────────────────────────
    source_raw = _safe_str(payload.get("source"), default="auto")
    source = _validate_lang_code(source_raw)
    if source is None:
        return _error_response(
            f"Idioma origen no soportado: '{source_raw}'",
            status_code=400,
        )

    # ── Validar modo OCR ───────────────────────────────────────
    ocr_mode = _safe_str(payload.get("ocr_mode"), default="fusion").lower()
    if ocr_mode not in ("easyocr", "auto", "unlimited", "fusion"):
        return _error_response(
            f"Modo OCR no soportado: '{ocr_mode}'. "
            f"Use 'easyocr', 'auto', 'unlimited' o 'fusion'.",
            status_code=400,
        )

    prefilter: bool = _coerce_bool(payload.get("prefilter"), default=True)
    force_uocr: bool = _coerce_bool(payload.get("force_uocr"), default=False)
    disable_uocr: bool = _coerce_bool(payload.get("disable_uocr"), default=False)
    pure_easyocr: bool = _coerce_bool(payload.get("pure_easyocr"), default=False)

    # ── doc_id (sesión 126): igual que en /process-page — escopea la firma
    #    de los caches de decisión por documento para que capítulos de la
    #    misma serie no hereden decisiones entre sí (94% colisión de layout,
    #    sesión 124). Vacío → scope legacy compartido. Sanizado igual que en
    #    el single (claves canónicas [A-Za-z0-9_]).
    doc_id: str = re.sub(r"[^A-Za-z0-9_]", "", _safe_str(payload.get("doc_id")))[:64]

    # ── response_format (optimización 2.3): igual que en /process-page.
    response_format = _safe_str(payload.get("response_format"), default="png").lower()
    if response_format == "jpg":
        response_format = "jpeg"
    if response_format not in ("png", "jpeg"):
        return _error_response(
            f"response_format no soportado: '{response_format}'. "
            f"Use 'png' o 'jpeg'.",
            status_code=400,
        )

    # ── Decodificar y normalizar cada imagen ───────────────────
    imgs: list[tuple[_Img, float, float]] = []  # (img, scale_x, scale_y)
    decoded_pixels = 0
    for b64_image in images_raw:
        remaining_pixels = MAX_BATCH_DECODE_PIXELS - decoded_pixels
        if remaining_pixels <= 0:
            return _error_response(
                "El lote supera el presupuesto máximo de píxeles decodificados",
                status_code=413,
            )
        img_bgr = _base64_to_cv2(
            _safe_str(b64_image), max_pixels=remaining_pixels)
        if img_bgr is None:
            return _error_response(
                "No se pudo decodificar una imagen (base64 inválido)",
                status_code=400,
            )
        orig_h, orig_w = img_bgr.shape[:2]
        if orig_w < 50 or orig_h < 50:
            return _error_response(
                f"Imagen demasiado pequeña ({orig_w}x{orig_h}). Mínimo 50x50 px.",
                status_code=400,
            )
        image_pixels = int(orig_w) * int(orig_h)
        if image_pixels > remaining_pixels:
            return _error_response(
                "El lote supera el presupuesto máximo de píxeles decodificados",
                status_code=413,
            )
        decoded_pixels += image_pixels
        scale_x = 1.0
        scale_y = 1.0
        if max(orig_w, orig_h) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = float(orig_w) / float(new_w)
            scale_y = float(orig_h) / float(new_h)
        imgs.append((img_bgr, scale_x, scale_y))

    detected_lang: str = source if source in LANGUAGES and source != "auto" else "es"
    ocr_lang = (
        "auto" if source == "auto"
        else "es" if detected_lang in ("es", "spa", "spanish", "espanol")
        else detected_lang
    )

    try:
        t0 = _time.time()
        from ocr_engine import OCRManager
        ocr_manager = OCRManager()
        per_page = ocr_manager.run_ocr_batch(
            [img for img, _, _ in imgs],
            ocr_lang,
            ocr_mode=ocr_mode,
            prefilter=prefilter,
            force_uocr=force_uocr,
            disable_uocr=disable_uocr,
            pure_easyocr=pure_easyocr,
            doc_id=doc_id,
        )
        t_ocr = _time.time() - t0

        results: list[dict[str, Any]] = []
        for idx, ((blocks, ocr_engine_used, engines_used),
                  (img_bgr, scale_x, scale_y)) in enumerate(zip(per_page, imgs)):
            result_blocks, inpainted_b64, _, t_inpaint = _finalize_page_blocks(
                img_bgr, blocks, source, target, scale_x, scale_y, detected_lang, doc_id,
                response_format)
            results.append({
                "inpainted_image": inpainted_b64,
                "blocks": result_blocks,
                "ocr_engine": ocr_engine_used,
                "engines_used": engines_used,
                "diagnostics": (
                    ocr_manager.last_batch_diagnostics[idx]
                    if idx < len(ocr_manager.last_batch_diagnostics) else None
                ),
            })

        total_t = _time.time() - t0
        n_total = sum(len(r["blocks"]) for r in results)
        print(f"[process-page-batch] {len(imgs)} páginas, {n_total} bloques, "
              f"{total_t:.1f}s (OCR:{t_ocr:.1f}s) | "
              f"engines: {[r['engines_used'] for r in results]}")
        return jsonify({"results": results, "t_total": round(total_t, 2)})

    except MemoryError:
        print(f"[process-page-batch] MEMORY ERROR! Forzando limpieza total...")
        from ocr_utils import _ocr_readers
        _ocr_readers.clear()
        gc.collect()
        return _error_response("Memoria insuficiente. Reintente.", status_code=500)

    except Exception as e:
        print(f"[process-page-batch] ERROR: {e}")
        traceback.print_exc()
        return _error_response(
            "Error interno procesando el lote", status_code=500)
