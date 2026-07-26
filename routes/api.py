"""
routes/api.py — Blueprint para endpoints de API REST.
"""
import cProfile
import functools
import gc
import io
import pstats
import time as _time
from concurrent.futures import as_completed
from typing import Any

import cv2
import numpy as np
import psutil
from flask import Blueprint, Response, jsonify, request
from numpy.typing import NDArray
_Img = np.ndarray  # type: ignore[type-arg]

from ratelimit import limiter, RATE_LIMIT_AVAILABLE

api_bp = Blueprint("api", __name__, url_prefix="/api")


# ── Profiling decorator (activar con ?profile=1) ───────────────
_PROFILE_DIR: str = ""  # Se inicializa en runtime


def _get_profile_dir() -> str:
    """Obtiene / crea el directorio de profiles."""
    global _PROFILE_DIR
    if not _PROFILE_DIR:
        import os
        _PROFILE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "profiles")
        os.makedirs(_PROFILE_DIR, exist_ok=True)
    return _PROFILE_DIR


def profile_endpoint(f):
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
    def wrapper(*args, **kwargs):
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
            import os
            import time
            ts = time.strftime("%Y%m%d_%H%M%S")
            endpoint_name = request.endpoint or "unknown"
            safe_name = endpoint_name.replace(".", "_")
            prof_path = os.path.join(_get_profile_dir(), f"{safe_name}_{ts}.prof")
            ps.dump_stats(prof_path)

            # Log a consola
            print(f"\n{'='*60}", flush=True)
            print(f"[PROFILE] {endpoint_name} ({ts})", flush=True)
            print(f"[PROFILE] Archivo: {prof_path}", flush=True)

            # Extraer resumen de ps.stats
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
            import traceback
            traceback.print_exc()

        return result
    return wrapper


def _get_memory_mb() -> float:
    """Returns current RSS memory in MB, or 0 if unavailable."""
    try:
        return float(psutil.Process().memory_info().rss / 1024 / 1024)
    except Exception:
        return 0.0


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


# Maximo texto aceptado para traduccion (previene DoS por texto gigante)
_MAX_TEXT_LENGTH: int = 20000  # ~20k chars
_MAX_BATCH_SIZE: int = 500      # max textos por batch
_MAX_IMAGE_BYTES: int = 50 * 1024 * 1024  # 50MB raw base64


@api_bp.post("/translate")
@profile_endpoint
def translate() -> Any:
    from server import _translate_one
    from config import LANGUAGES
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    target = str(payload.get("target", "es")).strip() or "es"
    source = str(payload.get("source", "auto")).strip() or "auto"
    if not text:
        return jsonify({"translatedText": "", "engine": "none"})
    if len(text) > _MAX_TEXT_LENGTH:
        return jsonify({"error": f"Texto demasiado largo ({len(text)} chars, max {_MAX_TEXT_LENGTH})"}), 413
    if target not in LANGUAGES:
        return jsonify({"error": f"Idioma no soportado: {target}"}), 400
    result = _translate_one(text, source, target)
    return jsonify({"translatedText": result, "engine": "auto"})


@api_bp.post("/translate-batch")
@profile_endpoint
def translate_batch() -> Any:
    from server import _translate_one, _get_executor
    from config import LANGUAGES
    payload = request.get_json(silent=True) or {}
    texts: list[str] = payload.get("texts", [])
    target = str(payload.get("target", "es")).strip() or "es"
    source = str(payload.get("source", "auto")).strip() or "auto"
    if not texts:
        return jsonify({"results": []})
    if len(texts) > _MAX_BATCH_SIZE:
        return jsonify({"error": f"Demasiados textos ({len(texts)}, max {_MAX_BATCH_SIZE})"}), 413
    if target not in LANGUAGES:
        return jsonify({"error": f"Idioma no soportado: {target}"}), 400
    results = list(texts)
    executor = _get_executor()
    futures = {
        executor.submit(_translate_one, t.strip(), source, target): i
        for i, t in enumerate(texts) if t.strip()
    }
    for future in as_completed(futures):
        idx = futures[future]
        try:
            results[idx] = future.result()
        except Exception as e:
            print(f"[translate-batch] Error traduciendo idx={idx}: {e}")
    return jsonify({"results": results})


@api_bp.post("/process-page")
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
    b64_image = str(payload.get("image", ""))
    target_lang = str(payload.get("target", "en")).strip() or "en"
    source_lang = str(payload.get("source", "auto")).strip() or "auto"
    ocr_mode = str(payload.get("ocr_mode", "ctd")).strip().lower()  # "auto", "easyocr", "ctd"
    if not b64_image:
        return jsonify({"error": "No se proporcion\u00f3 imagen"}), 400

    # Validar tamano aproximado de la imagen base64 (evitar OOM)
    if len(b64_image) > _MAX_IMAGE_BYTES:
        return jsonify({"error": "Imagen demasiado grande"}), 413

    scale_x: float = 1.0
    scale_y: float = 1.0
    mem_before = _get_memory_mb()

    try:
        t0 = _time.time()

        # Decode image
        img_bgr = _base64_to_cv2(b64_image)
        if img_bgr is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400
        orig_h, orig_w = img_bgr.shape[:2]

        # Safety resize for >4096px images
        if max(orig_w, orig_h) > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max(orig_w, orig_h)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            scale_x = float(orig_w) / float(new_w)
            scale_y = float(orig_h) / float(new_h)

        print(f"[process-page] Imagen {img_bgr.shape[1]}x{img_bgr.shape[0]}, target={target_lang}, source={source_lang}")

        # Language detection (optimizado: evitar pre-OCR que duplica el tiempo)
        # Si el usuario especifico idioma, usarlo. Si no, asumir "es" (manga
        # espanol) y corregir despues del OCR real si es necesario.
        # El pre-OCR al 25% previo hacia un OCR completo solo para detectar
        # idioma, duplicando el tiempo de procesamiento (~5-10s extra).
        detected_lang: str = source_lang if source_lang in LANGUAGES and source_lang != "auto" else "es"
        if source_lang == "auto":
            print(f"[process-page] Source=auto, asumiendo es temporalmente (se corrige post-OCR)")
        else:
            print(f"[process-page] Usando idioma especificado: {source_lang}")
        ocr_lang = "es" if detected_lang in ("es", "spa", "spanish", "espanol") else detected_lang

        # OCR
        ocr_lang = "es" if detected_lang in ("es", "spa", "spanish", "espanol") else detected_lang
        t_ocr_before = _time.time()
        # Modo OCR: "auto" (default, 3 niveles), "easyocr" (solo EasyOCR), "ctd" (solo CTD)
        use_ctd_only = ocr_mode == "ctd"
        allow_fallback = ocr_mode not in ("easyocr", "ctd")  # Solo auto tiene fallbacks
        blocks: list[dict[str, Any]] = _detect_and_ocr(
            img_bgr, ocr_lang,
            allow_fallback=allow_fallback,
            use_ctd_only=use_ctd_only,
        )
        t_ocr = _time.time() - t_ocr_before
        print(f"[process-page] OCR ({ocr_lang}): {len(blocks)} bloques en {t_ocr:.1f}s (detectado: {detected_lang})")

        if not blocks:
            inpainted_b64: str | None = _cv2_to_base64(img_bgr)
            img_bgr = None
            gc.collect()
            return jsonify({"inpainted_image": inpainted_b64, "blocks": []})

        # Re-detect language if source was auto (corregir el temporal "es")
        if source_lang == "auto" and blocks:
            try:
                combined_text = " ".join([str(b.get("text", "")) for b in blocks])
                detected_lang = _detect_language_robust(combined_text)
                print(f"[process-page] Idioma detectado post-OCR: {detected_lang}")
            except Exception:
                detected_lang = "es"

        # Si el idioma origen es un código multi-idioma (ej. "eng+spa+fra+deu"),
        # re-detectamos el idioma real después del OCR
        if source_lang not in LANGUAGES and source_lang != "auto" and blocks:
            try:
                combined_text = " ".join([str(b.get("text", "")) for b in blocks])
                detected_lang = _detect_language_robust(combined_text)
                print(f"[process-page] Código multi-idioma '{source_lang}', detectado: {detected_lang}")
            except Exception:
                detected_lang = "es"

        # Inpainting
        inpainted: _Img | None = None
        inpainted_b64: str | None = None  # type: ignore[no-redef]
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

        # Translation
        source_texts = [str(b["text"]) for b in blocks]
        translated_texts = list(source_texts)
        executor = _get_executor()
        fut = {
            executor.submit(_translate_one, t, detected_lang, target_lang): i
            for i, t in enumerate(source_texts) if t.strip()
        }
        for future in as_completed(fut):
            idx = fut[future]
            try:
                translated_texts[idx] = future.result()
            except Exception as e:
                print(f"[process-page] Error traduciendo bloque idx={idx}: {e}")

        # Build response blocks
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

        # Cleanup
        img_bgr = None
        inpainted = None
        mask = None
        gc.collect()

        mem_after = _get_memory_mb()
        mem_growth = mem_after - mem_before
        total_t = _time.time() - t0
        print(f"[process-page] Completado. {len(result_blocks)} bloques, {total_t:.1f}s (OCR:{t_ocr:.1f}s + inpaint:{t_inpaint:.1f}s), memoria: {mem_before:.0f}->{mem_after:.0f}MB ({mem_growth:+.0f}MB)")
        return jsonify({
            "inpainted_image": inpainted_b64,
            "blocks": result_blocks,
        })

    except MemoryError:
        print(f"[process-page] MEMORY ERROR! Forzando limpieza total...")
        from ocr_utils import _ocr_readers
        _ocr_readers.clear()
        gc.collect()
        return jsonify({"error": "Memoria insuficiente. Reintente."}), 500

    except Exception as e:
        import traceback
        print(f"[process-page] ERROR: {e}")
        traceback.print_exc()
        gc.collect()
        return jsonify({"error": str(e)}), 500
