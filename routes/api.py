"""
routes/api.py — Blueprint para endpoints de API REST.
"""
from concurrent.futures import as_completed

from flask import Blueprint, jsonify, request
from ratelimit import limiter, RATE_LIMIT_AVAILABLE

api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.get("/health")
def health():
    from server import _argo_ready, APP_VERSION, IS_PRODUCTION, MIT_AVAILABLE, DB_AVAILABLE, TRANSLATION_CACHE_AVAILABLE
    ready = [f"{s}->{t}" for (s, t), v in _argo_ready.items() if v]
    return jsonify({
        "ok": True,
        "version": APP_VERSION,
        "mode": "production" if IS_PRODUCTION else "development",
        "mit_available": MIT_AVAILABLE,
        "db_available": DB_AVAILABLE,
        "cache_available": TRANSLATION_CACHE_AVAILABLE,
        "rate_limiting": RATE_LIMIT_AVAILABLE,
        "offline_models": ready,
    })


@limiter.limit("30 per minute") if RATE_LIMIT_AVAILABLE else lambda f: f
@api_bp.post("/translate")
def translate():
    from server import _translate_one, LANGUAGES
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    target = str(payload.get("target", "es")).strip() or "es"
    source = str(payload.get("source", "auto")).strip() or "auto"
    if not text:
        return jsonify({"translatedText": "", "engine": "none"})
    if target not in LANGUAGES:
        return jsonify({"error": f"Idioma no soportado: {target}"}), 400
    return jsonify({"translatedText": _translate_one(text, source, target), "engine": "auto"})


@limiter.limit("20 per minute") if RATE_LIMIT_AVAILABLE else lambda f: f
@api_bp.post("/translate-batch")
def translate_batch():
    from server import _translate_one, _get_executor, LANGUAGES
    payload = request.get_json(silent=True) or {}
    texts: list[str] = payload.get("texts", [])
    target = str(payload.get("target", "es")).strip() or "es"
    source = str(payload.get("source", "auto")).strip() or "auto"
    if not texts:
        return jsonify({"results": []})
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
        except Exception:
            pass
    return jsonify({"results": results})


@limiter.limit("5 per minute") if RATE_LIMIT_AVAILABLE else lambda f: f
@api_bp.post("/process-page")
def process_page():
    from server import (
        _base64_to_cv2, _cv2_to_base64, _detect_and_ocr,
        _build_inpaint_mask, _inpaint_image, _sample_bg_color,
        _translate_one, _detect_language_robust, _get_executor,
        MIT_AVAILABLE, LANGUAGES,
    )
    payload = request.get_json(silent=True) or {}
    b64_image = payload.get("image", "")
    target_lang = str(payload.get("target", "en")).strip() or "en"
    source_lang = str(payload.get("source", "auto")).strip() or "auto"

    if not b64_image:
        return jsonify({"error": "No se proporcionó imagen"}), 400

    try:
        img_bgr = _base64_to_cv2(b64_image)
        if img_bgr is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400

        print(f"[process-page] Imagen {img_bgr.shape[1]}x{img_bgr.shape[0]}, target={target_lang}")

        mit_used = False
        if MIT_AVAILABLE:
            try:
                print(f"[process-page] Usando MIT pipeline (CTD + LaMa)...")
                from manga_pipeline import run_pipeline
                mit_result = run_pipeline(img_bgr)
                if mit_result.get("error"):
                    print(f"[process-page] MIT falló: {mit_result['error']}, usando legacy...")
                else:
                    inpainted_b64 = mit_result["inpainted_image"]
                    blocks = mit_result.get("blocks", [])
                    mit_used = True
                    inpainted = _base64_to_cv2(inpainted_b64)
                    print(f"[process-page] MIT: {len(blocks)} bloques detectados")
            except Exception as mit_err:
                print(f"[process-page] MIT excepción: {mit_err}, usando legacy...")

        if not mit_used:
            blocks = _detect_and_ocr(img_bgr, source_lang)
            print(f"[process-page] Legacy OCR: {len(blocks)} bloques")

        if not blocks:
            inpainted_b64 = _cv2_to_base64(img_bgr)
            return jsonify({"inpainted_image": inpainted_b64, "blocks": []})

        detected_lang = source_lang
        if (source_lang == "auto" or source_lang not in LANGUAGES) and blocks:
            combined_text = " ".join([b.get("text", "") for b in blocks])
            detected_lang = _detect_language_robust(combined_text)
            print(f"[process-page] Idioma detectado: {detected_lang}")

        if not mit_used:
            mask = _build_inpaint_mask(img_bgr, blocks)
            inpainted = _inpaint_image(img_bgr, mask, blocks)
            inpainted_b64 = _cv2_to_base64(inpainted)
        else:
            inpainted = _base64_to_cv2(inpainted_b64)

        source_texts = [b["text"] for b in blocks]
        translated_texts = list(source_texts)
        executor = _get_executor()
        futures = {
            executor.submit(_translate_one, t, detected_lang, target_lang): i
            for i, t in enumerate(source_texts) if t.strip()
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                translated_texts[idx] = future.result()
            except Exception:
                pass

        result_blocks = []
        for i, block in enumerate(blocks):
            bg_color = _sample_bg_color(inpainted, block)
            result_blocks.append({
                "x": block["x"],
                "y": block["y"],
                "w": block["w"],
                "h": block["h"],
                "source": source_texts[i],
                "translated": translated_texts[i],
                "fontSize": block["fontSize"],
                "textColor": block["textColor"],
                "bgColor": bg_color,
                "confidence": block["confidence"],
            })

        print(f"[process-page] Completado. {len(result_blocks)} bloques traducidos.")
        return jsonify({
            "inpainted_image": inpainted_b64,
            "blocks": result_blocks,
        })

    except Exception as e:
        import traceback
        print(f"[process-page] ERROR: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500