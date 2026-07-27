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
from concurrent.futures import as_completed
from typing import Any

import cv2
import numpy as np
import psutil
from flask import Blueprint, Response, jsonify, request
from numpy.typing import NDArray
_Img = np.ndarray  # type: ignore[type-arg]

from config import (
    MAX_TEXT_LENGTH,
    MAX_BATCH_SIZE,
    MAX_IMAGE_BYTES,
)
from ratelimit import limiter, RATE_LIMIT_AVAILABLE

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ════════════════════════════════════════════════════════════════
# CONSTANTES DE VALIDACIÓN DE SEGURIDAD
# ════════════════════════════════════════════════════════════════
# MAX_TEXT_LENGTH, MAX_BATCH_SIZE y MAX_IMAGE_BYTES se importan desde config.py
_MIN_TEXT_LENGTH: int = 1               # mínimo 1 char útil tras strip
_MAX_TEXT_WORDS: int = 5_000            # ~5000 palabras máx

# Idiomas válidos para el endpoint de traducción (excluye auto para target)
_LANG_CODES: frozenset[str] = frozenset({
    "es", "en", "pt", "fr", "de", "it",
    "ja", "ko", "zh", "zh-cn", "zh-tw",
    "auto",
})


# ════════════════════════════════════════════════════════════════
# HELPERS DE RESPUESTA Y VALIDACIÓN
# ════════════════════════════════════════════════════════════════

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


def _validate_json_content_type(f: Any) -> Any:
    """Decorador: verifica que el Content-Type sea application/json
    en endpoints POST. Retorna 415 si no lo es."""
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not request.is_json:
            return _error_response(
                "Content-Type debe ser application/json",
                status_code=415,
            )
        # Validar charset explícito: si se declara charset, debe ser UTF-8
        # para prevenir ataques de encoding (e.g. iso-2022-kr, utf-7).
        # RFC 8259 §8.1: JSON debe transmitirse en UTF-8.
        # Usar request.headers.get() en vez de request.content_type porque
        # el WSGI layer puede parsear/strippear parámetros del Content-Type.
        # Usar regex para manejar "charset=utf-8", "charset = utf-7", "charset  =  \"utf-8\"", etc.
        raw_ct: str = request.headers.get("Content-Type", "").lower()
        charset_m = re.search(r'charset\s*=\s*["\']?([^;"\'\s]+)', raw_ct)
        if charset_m:
            charset_val: str = charset_m.group(1).strip().strip("'\"")
            if charset_val and charset_val != "utf-8":
                return _error_response(
                    f"Charset '{charset_val}' no soportado. Use charset=utf-8 (o ninguno).",
                    status_code=415,
                )
        return f(*args, **kwargs)
    return wrapper


def _validate_payload_fields(*required: str) -> Any:
    """Decorador de fábrica: valida que el payload JSON contenga
    los campos requeridos.

    Args:
        required: Nombres de campos obligatorios.

    Uso:
        @api_bp.post("/translate")
        @_validate_payload_fields("text")
        def translate():
            data = request.get_json()
            ...
    """
    required_list = list(required)

    def decorator(f: Any) -> Any:
        @functools.wraps(f)
        @_validate_json_content_type
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            data: dict[str, Any] = request.get_json(silent=True) or {}
            missing = [field for field in required_list if field not in data]
            if missing:
                return _error_response(
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
    source = _validate_lang_code(source_raw)
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


def profile_endpoint(f: Any) -> Any:
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
    def wrapper(*args: Any, **kwargs: Any) -> Any:
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
            for key, (cc, nc, tt, ct, callers) in ps.stats.items():
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
    return jsonify({
        "ok": True,
        "version": APP_VERSION,
        "mode": "production" if IS_PRODUCTION else "development",
        "mit_available": False,
        "db_available": DB_AVAILABLE,
        "cache_available": TRANSLATION_CACHE_AVAILABLE,
        "rate_limiting": RATE_LIMIT_AVAILABLE,
        "offline_models": ready,
        "memory": f"{mem_mb:.0f}MB" if mem_mb else "N/A",
    })


@api_bp.post("/translate")
@_validate_payload_fields("text")
@profile_endpoint
def translate() -> Any:
    from server import _translate_one
    from config import LANGUAGES

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

    result = _translate_one(text, source, target)
    return jsonify({"translatedText": result, "engine": "auto"})


@api_bp.post("/translate-batch")
@_validate_payload_fields("texts")
@profile_endpoint
def translate_batch() -> Any:
    from server import _translate_one, _get_executor
    from config import LANGUAGES

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

    results = list(texts_raw)
    executor = _get_executor()
    futures = {
        executor.submit(_translate_one, t.strip(), source, target): i
        for i, t in enumerate(texts_raw) if t.strip()
    }
    for future in as_completed(futures):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            print(f"[translate-batch] Error traduciendo idx={idx}: {e}")
    return jsonify({"results": results})


@api_bp.post("/process-page")
@_validate_payload_fields("image")
@profile_endpoint
def process_page() -> Any:
    from ocr_utils import (
        _base64_to_cv2, _cv2_to_base64, _detect_and_ocr,
        _build_inpaint_mask, _inpaint_image, _sample_bg_color,
        _filter_watermarks_from_blocks,
    )
    from server import _translate_one
    from translator import _detect_language_robust
    from server import _get_executor
    from config import MAX_IMAGE_DIMENSION, LANGUAGES

    payload = request.get_json(silent=True) or {}
    b64_image = _safe_str(payload.get("image"))

    # ── Validar que la imagen no esté vacía ────────────────────
    if not b64_image:
        return _error_response("No se proporcionó imagen", status_code=400)

    # ── Validar tamaño aproximado (evitar OOM) ─────────────────
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
    ocr_mode = _safe_str(payload.get("ocr_mode"), default="easyocr").lower()
    if ocr_mode not in ("easyocr", "auto"):
        return _error_response(
            f"Modo OCR no soportado: '{ocr_mode}'. Use 'easyocr' o 'auto'.",
            status_code=400,
        )

    # ── Validar prefilter (limpieza morfológica pre-OCR) ───────
    prefilter: bool = bool(payload.get("prefilter", False))

    scale_x: float = 1.0
    scale_y: float = 1.0
    mem_before = _get_memory_mb()

    try:
        t0 = _time.time()

        # ── Decodificar imagen ─────────────────────────────────
        img_bgr = _base64_to_cv2(b64_image)
        if img_bgr is None:
            return _error_response("No se pudo decodificar la imagen (base64 inválido)", status_code=400)
        orig_h, orig_w = img_bgr.shape[:2]

        # ── Validar dimensiones mínimas ────────────────────────
        if orig_w < 50 or orig_h < 50:
            return _error_response(
                f"Imagen demasiado pequeña ({orig_w}x{orig_h}). Mínimo 50x50 px.",
                status_code=400,
            )

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

        ocr_lang = "es" if detected_lang in ("es", "spa", "spanish", "espanol") else detected_lang

        # ── OCR ────────────────────────────────────────────────
        t_ocr_before = _time.time()
        allow_fallback = ocr_mode != "easyocr"  # auto tiene fallback CLAHE
        blocks: list[dict[str, Any]] = _detect_and_ocr(
            img_bgr, ocr_lang,
            allow_fallback=allow_fallback,
            prefilter=prefilter,
        )
        t_ocr = _time.time() - t_ocr_before
        print(f"[process-page] OCR ({ocr_lang}): {len(blocks)} bloques en {t_ocr:.1f}s")

        if not blocks:
            inpainted_b64: str | None = _cv2_to_base64(img_bgr)
            img_bgr = None
            return jsonify({"inpainted_image": inpainted_b64, "blocks": []})

        # ── Re-detección de idioma post-OCR ────────────────────
        if source == "auto" and blocks:
            try:
                combined_text = " ".join([str(b.get("text", "")) for b in blocks])
                detected_lang = _detect_language_robust(combined_text)
                print(f"[process-page] Idioma detectado post-OCR: {detected_lang}")
            except Exception:
                detected_lang = "es"

        # ── Inpainting ─────────────────────────────────────────
        inpainted: _Img | None = None
        inpainted_b64 = None  # type: ignore[no-redef]
        mask: _Img | None = None
        t_inpaint_before = _time.time()
        try:
            mask = _build_inpaint_mask(img_bgr, blocks)
            inpainted = _inpaint_image(img_bgr, mask, blocks)
            inpainted_b64 = _cv2_to_base64(inpainted)
        except Exception as inpaint_err:
            print(f"[process-page] Inpainting error: {inpaint_err}, usando imagen original")
            inpainted_b64 = _cv2_to_base64(img_bgr)
        finally:
            mask = None
        t_inpaint = _time.time() - t_inpaint_before

        # ── Traducción ─────────────────────────────────────────
        source_texts = [str(b["text"]) for b in blocks]
        translated_texts = list(source_texts)
        executor = _get_executor()
        fut = {
            executor.submit(_translate_one, t, detected_lang, target): i
            for i, t in enumerate(source_texts) if t.strip()
        }
        for future in as_completed(fut):
            idx = fut[future]
            try:
                translated_texts[idx] = future.result()
            except Exception as e:
                print(f"[process-page] Error traduciendo bloque idx={idx}: {e}")

        # ── Armar respuesta ────────────────────────────────────
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
            result_blocks.append({
                "x": bx, "y": by, "w": bw, "h": bh,
                "source": source_texts[i],
                "translated": translated_texts[i],
                "fontSize": block["fontSize"],
                "textColor": block["textColor"],
                "bgColor": bg_color,
                "confidence": block["confidence"],
            })

        # ── Cleanup ────────────────────────────────────────────
        img_bgr = None
        inpainted = None
        mask = None

        mem_after = _get_memory_mb()
        mem_growth = mem_after - mem_before
        total_t = _time.time() - t0
        print(f"[process-page] Completado. {len(result_blocks)} bloques, {total_t:.1f}s "
              f"(OCR:{t_ocr:.1f}s + inpaint:{t_inpaint:.1f}s), "
              f"memoria: {mem_before:.0f}→{mem_after:.0f}MB ({mem_growth:+.0f}MB)")
        return jsonify({
            "inpainted_image": inpainted_b64,
            "blocks": result_blocks,
        })

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
        return _error_response(str(e), status_code=500)
