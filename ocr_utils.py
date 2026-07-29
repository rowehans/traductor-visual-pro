"""
ocr_utils.py — OCR (EasyOCR), inpainting (OpenCV), filtros de ruido y muestreo de color.

Extraído de server.py. Depende de config.py para patrones de ruido y constantes.
"""

import base64
import re
import threading
import unicodedata
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from config import ROOT, MARGIN_NOISE_PATTERNS, WATERMARK_PATTERNS

# Type alias for OpenCV images (suppresses overly-strict ndarray checks)
_Img = np.ndarray


# ─── EasyOCR (lazy load with multi-lang support + CPU fallback) ──
_ocr_readers: dict[str, Any] = {}
_ocr_lock: threading.Lock = threading.Lock()
# Semaforo para limitar concurrencia OCR: max 1 lectura simultanea
# porque EasyOCR no es thread-safe y cada reader consume ~1-2GB VRAM/RAM
_ocr_semaphore: threading.Semaphore = threading.Semaphore(1)
_rapid_semaphore: threading.Semaphore = threading.Semaphore(1)


def _get_ocr_reader(lang: str = "auto") -> Any:
    lang_key: str = "latin"
    if lang == "ja":
        lang_key = "ja"
    elif lang == "ko":
        lang_key = "ko"
    elif lang == "zh":
        lang_key = "zh"

    global _ocr_readers  # noqa: PLW0603
    if lang_key in _ocr_readers:
        return _ocr_readers[lang_key]

    with _ocr_lock:
        if lang_key in _ocr_readers:
            return _ocr_readers[lang_key]

        langs: list[str] = ["es", "en"]
        if lang_key == "ja":
            langs = ["ja", "en"]
        elif lang_key == "ko":
            langs = ["ko", "en"]
        elif lang_key == "zh":
            langs = ["ch_sim", "en"]
        else:
            langs = ["es", "en", "pt", "fr", "de"]

        # ═══ Carga EasyOCR: GPU si CUDA disponible, CPU si no ═══════
        # El orden de carga es CRÍTICO para evitar conflicto cuDNN:
        #   - Si CT2 carga PRIMERO CUDA/cuDNN, luego PyTorch no puede
        #     cargar sus propios símbolos cuDNN → crash.
        #   - Server.py ahora carga EasyOCR PRIMERO (PyTorch toma GPU,
        #     carga cuDNN), y luego CT2 con force_cpu=True.
        #   - Si el usuario llama EasyOCR sin preload (caso de esquina),
        #     también funciona: PyTorch inicializa CUDA, CT2 detecta
        #     que CUDA está disponible pero se pasa force_cpu como fallback.
        # ════════════════════════════════════════════════════════════
        print(f"[OCR] Cargando EasyOCR para {langs}...")

        try:
            import easyocr
            # Intentar GPU primero. Si falla (CUDA no disponible, memoria insuficiente),
            # EasyOCR automáticamente cae a CPU via el try/except que sigue.
            gpu_available = True
            try:
                import torch
                gpu_available = torch.cuda.is_available()
            except Exception:
                gpu_available = False

            reader = easyocr.Reader(
                langs,
                gpu=gpu_available,
                model_storage_directory=str(ROOT / "ocr_models"),
                download_enabled=True,
                verbose=False,
            )
            _ocr_readers[lang_key] = reader
            device_str = "GPU" if gpu_available else "CPU"
            print(f"[OCR] EasyOCR para {langs} listo en {device_str}.")
        except Exception as e:
            print(f"[OCR] Error cargando EasyOCR en GPU para {langs}: {e}")
            print(f"[OCR] Reintentando en CPU...")
            try:
                import easyocr
                reader = easyocr.Reader(
                    langs,
                    gpu=False,
                    model_storage_directory=str(ROOT / "ocr_models"),
                    download_enabled=True,
                    verbose=False,
                )
                _ocr_readers[lang_key] = reader
                print(f"[OCR] EasyOCR para {langs} listo en CPU (fallback).")
            except Exception as e2:
                print(f"[OCR] Error cargando EasyOCR incluso en CPU: {e2}")
                return None
    return _ocr_readers[lang_key]


# ─── RapidOCR (lazy load with thread safety) ─────────────────────
# Usa los mismos modelos PP-OCRv4 que PaddleOCR pero via ONNX Runtime,
# sin el conflicto de PaddlePaddle vs PyTorch.
_rapid_engine: Any = None
_rapid_lock: threading.Lock = threading.Lock()


def _get_rapid_engine() -> Any:
    global _rapid_engine
    if _rapid_engine is not None:
        return _rapid_engine
    with _rapid_lock:
        if _rapid_engine is not None:
            return _rapid_engine
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_engine = RapidOCR()
            print("[OCR] RapidOCR listo (CPU/ONNX)")
        except Exception as e:
            print(f"[OCR] Error cargando RapidOCR: {e}")
            return None
    return _rapid_engine


def _preprocess_rapid(img_bgr: _Img) -> _Img:
    """Preprocesamiento optimizado para RapidOCR (pre-filter + enhance)."""
    filtered = _pre_filter_image(img_bgr)
    enhanced = _preprocess_enhanced(filtered)
    return enhanced


def _run_rapidocr(img_bgr: _Img) -> list[dict[str, Any]]:
    """
    Ejecuta RapidOCR sobre una imagen y retorna bloques en el
    mismo formato que EasyOCR (x, y, w, h, text, confidence, ...).
    Adquiere _rapid_semaphore (ONNX Runtime no es thread-safe).
    """
    engine = _get_rapid_engine()
    if engine is None:
        return []
    acquired = _rapid_semaphore.acquire(blocking=True, timeout=120)
    if not acquired:
        print("[OCR] Timeout adquiriendo semaforo RapidOCR (120s)")
        return []
    try:
        result, _ = engine(img_bgr)
        blocks: list[dict[str, Any]] = []
        if result:
            for r in result:
                try:
                    bbox, text, conf = r
                    text = str(text).strip()
                    if not text or conf < 0.08:
                        continue
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    x, y = int(min(xs)), int(min(ys))
                    w, h = int(max(xs) - x), int(max(ys) - y)
                    if w < 3 or h < 3:
                        continue
                    blocks.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "text": text,
                        "confidence": float(conf),
                        "fontSize": max(8, int(h * 0.75)),
                        "textColor": "#000000",
                    })
                except (ValueError, IndexError, TypeError):
                    continue
        return _group_and_merge_blocks(blocks, img_bgr.shape[0])
    except Exception as e:
        print(f"[OCR] Error en RapidOCR: {e}")
        return []
    finally:
        if acquired:
            _rapid_semaphore.release()


def _normalize_text(t: str) -> str:
    """Normaliza texto para comparacion entre OCRs: lowercase + sin acentos."""
    t = t.lower().strip()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    return t


def _fusionar_blocks(
    easy_blocks: list[dict[str, Any]],
    rapid_blocks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Fusiona bloques de EasyOCR y RapidOCR eliminando duplicados.
    - Si ambos detectan el mismo texto, usa el de mayor confianza.
    - Si solo uno detecta un texto, lo conserva.
    - Compara textos normalizados (lowercase + sin acentos).
    """
    if not easy_blocks:
        return rapid_blocks
    if not rapid_blocks:
        return easy_blocks

    easy_by_text: dict[str, dict[str, Any]] = {_normalize_text(b["text"]): b for b in easy_blocks}
    rapid_by_text: dict[str, dict[str, Any]] = {_normalize_text(b["text"]): b for b in rapid_blocks}

    common = set(easy_by_text.keys()) & set(rapid_by_text.keys())
    only_easy = set(easy_by_text.keys()) - common
    only_rapid = set(rapid_by_text.keys()) - common

    result: list[dict[str, Any]] = []
    for t in common:
        if easy_by_text[t]["confidence"] >= rapid_by_text[t]["confidence"]:
            result.append(easy_by_text[t])
        else:
            result.append(rapid_by_text[t])
    for t in only_easy:
        result.append(easy_by_text[t])
    for t in only_rapid:
        result.append(rapid_by_text[t])

    return result


# ─── Image base64 conversion ─────────────────────────────────────
def _base64_to_cv2(b64: str) -> _Img | None:
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(b64)
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _cv2_to_base64(img: _Img, fmt: str = ".png") -> str:
    # Mapa de extension a MIME type
    _MIME_MAP: dict[str, str] = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
        ".tif": "image/tiff",
    }
    mime = _MIME_MAP.get(fmt, "image/png")
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if fmt in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    success, buf = cv2.imencode(fmt, img, params)
    if not success:
        raise ValueError(f"No se pudo codificar la imagen en formato {fmt}")
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


# ─── Preprocesamiento mejorado (fallback para texto artístico) ──
# Cuando EasyOCR no detecta texto con el pipeline normal,
# probamos técnicas más agresivas de mejora de contraste:
# 1. CLAHE en LAB: realza contraste local sin amplificar ruido
# 2. Unsharp mask: afila trazos finos (típico de texto artístico manga)
# 3. Gamma correction: ilumina sombras profundas sin quemar blancos
# 4. Bilateral filter: reduce ruido preservando bordes

def _preprocess_enhanced(img_bgr: _Img) -> _Img:
    """
    Preprocesamiento agresivo para texto artístico/decorativo que
    EasyOCR no detecta con el pipeline normal.

    Aplica CLAHE + Unsharp Mask + Gamma + Bilateral y fusiona
    para mejorar contraste local y nitidez de trazos finos.
    """
    h, w = img_bgr.shape[:2]

    # ── 1. CLAHE adaptativo en espacio LAB ───────────────────────
    # clip_limit adaptativo: imágenes más grandes necesitan más
    # contraste local. tile_grid_size proporcional al tamaño.
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    tile_size = max(4, min(16, min(w, h) // 100))
    clip_limit = min(4.0, max(2.0, 3.0 * max(w, h) / 2500.0))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    img_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    # ── 2. Gamma correction (ilumina sombras) ───────────────────
    # Gamma < 1 ilumina zonas oscuras (útil para texto en sombras
    # o escaneos subexpuestos). Aplicar solo si la imagen es oscura.
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(gray.mean())
    if mean_brightness < 100:
        gamma = 0.6 + 0.4 * (mean_brightness / 100.0)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255
                          for i in range(256)], dtype=np.uint8)
        img_gamma = cv2.LUT(img_bgr, table)
    else:
        img_gamma = img_bgr

    # ── 3. Bilateral filter (reduce ruido, preserva bordes) ─────
    # A diferencia de GaussianBlur, bilateral preserva bordes
    # de texto mientras suaviza ruido de fondo de escaneo.
    # d=5, sigmaColor=50, sigmaSpace=50 son valores suaves.
    img_denoised = cv2.bilateralFilter(img_gamma, d=5, sigmaColor=50, sigmaSpace=50)

    # ── 4. Unsharp mask sobre la imagen denoised ────────────────
    blurred = cv2.GaussianBlur(img_denoised, (0, 0), sigmaX=1.5)
    img_sharp = cv2.addWeighted(img_denoised, 1.8, blurred, -0.8, 0)

    # ── 5. Fusionar CLAHE + sharp ──────────────────────────────
    # CLAHE da contraste, sharp da nitidez, gamma ilumina sombras.
    enhanced = cv2.addWeighted(img_clahe, 0.5, img_sharp, 0.5, 0)

    return enhanced


# ─── Preprocesamiento morfológico (limpieza de ruido de escaneo) ─
# Elimina líneas horizontales finas, puntos de ruido aislados,
# y limpia bordes de página (común en escaneos manga).

def _pre_filter_image(img_bgr: _Img) -> _Img:
    """
    Limpieza morfológica pre-OCR.
    - Elimina líneas horizontales finas (artefactos de escaneo).
    - Remueve puntos de ruido aislados (speckle).
    - Limpia franjas 4% superior/inferior (sombras de borde).
    - Suaviza ruido de fondo de papel.
    """
    h, w = img_bgr.shape[:2]
    result = img_bgr.copy()

    # ── 1. Limpiar franjas 4% superior e inferior ───────────────
    # Los escaneos de manga suelen tener sombras o texto basura
    # en los bordes extremos de la página.
    margin_height = max(1, int(h * 0.04))
    # Superior: rellenar con el color promedio del borde
    top_strip = img_bgr[margin_height:margin_height * 2, :, :]
    if top_strip.size > 0:
        top_fill = np.median(top_strip.reshape(-1, 3), axis=0).astype(np.uint8)
        result[:margin_height, :, :] = top_fill
    # Inferior
    bot_strip = img_bgr[h - margin_height * 2:h - margin_height, :, :]
    if bot_strip.size > 0:
        bot_fill = np.median(bot_strip.reshape(-1, 3), axis=0).astype(np.uint8)
        result[h - margin_height:, :, :] = bot_fill

    # ── 2. Eliminar líneas horizontales finas (artefactos) ─────
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # Detectar bordes horizontales con kernel 1x15
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    detect_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
    # Umbral para identificar líneas
    _, thresh_lines = cv2.threshold(detect_lines, 50, 255, cv2.THRESH_BINARY)
    # Dilatar ligeramente para cubrir la línea completa
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    line_mask = cv2.dilate(thresh_lines, kernel_dilate, iterations=1)
    # Inpainting de las líneas detectadas
    if int(line_mask.max()) > 0:
        result = cv2.inpaint(result, line_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        # Re-calcular gray después del inpainting (modificó result)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # ── 3. Eliminar puntos de ruido aislados (speckle) ─────────
    # OTSU separa texto oscuro (0) de fondo claro (255)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # MORPH_OPEN remueve pequeños puntos blancos aislados (p. ej. speckle de
    # escaneo dentro de regiones oscuras). XOR entre binary y cleaned revela
    # EXACTAMENTE qué píxeles cambiaron de 255→0 por MORPH_OPEN — el speckle
    # real. En lugar de bitwise_and destructivo sobre el canal L completo,
    # sólo inpaintamos esos píxeles específicos, preservando texto y líneas
    # inpaintadas intactos.
    speckle_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, speckle_kernel, iterations=1)
    speckle_pixels = cv2.bitwise_xor(binary, cleaned)
    # Umbral mínimo de área: evitar inpainting por ruido sub-pixel
    # en páginas limpias (p.ej. bordes de imagen donde MORPH_OPEN
    # elimina 1-2 píxeles fronterizos sin beneficio real).
    if int(speckle_pixels.max()) > 0 and np.count_nonzero(speckle_pixels) > 50:
        # Dilatar 1px con kernel 3x3 para cubrir bordes de speckle
        speckle_mask = cv2.dilate(speckle_pixels, speckle_kernel, iterations=1)
        result = cv2.inpaint(result, speckle_mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

    # ── 4. Suavizado ligero de fondo (bilateral) ───────────────
    # Preserva bordes, reduce ruido de papel escaneado.
    result = cv2.bilateralFilter(result, d=3, sigmaColor=30, sigmaSpace=30)

    return result


# ─── Binarización adaptativa (fallback final para texto tenue) ───
# _binarize_image ELIMINADA — benchmark mostró 0 beneficios en páginas artísticas.
# El tier 3 (binarización) nunca agregó bloques que EasyOCR + CLAHE no hubieran
# detectado ya. Se ahorra ~0.8s por página.
    # blockSize impar, C constante restada de la media.
    # Valores típicos: blockSize=15-31, C=5-15
    block_size = max(11, (min(img_bgr.shape[:2]) // 20) | 1)  # impar
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, C=10,
    )

    # ── 3. Invertir si el fondo es más claro que el texto ──────
    white_pixels = float(np.sum(binary == 255))
    total_pixels = float(binary.size)
    if white_pixels / total_pixels < 0.3:
        binary = cv2.bitwise_not(binary)

    # ── 4. Convertir a 3 canales para EasyOCR ──────────────────
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ─── OCR principal ───────────────────────────────────────────────
# ─── OCR optimizado: equilibrio velocidad/detección ──
# Canvas size 2500px: un poco mas grande que antes (2000) para no perder
# texto pequeño en burbujas de manga, pero sin llegar al original (4096+)
# que saturaba EasyOCR.
_OCR_CANVAS_SIZE: int = 2500
# Thresholds ajustados para manga denso: más sensibles para capturar
# texto pequeño en burbujas, onomatopeyas y diálogos fragmentados.
_OCR_TEXT_THRESHOLD: float = 0.15
_OCR_LOW_TEXT: float = 0.10
_OCR_MIN_SIZE: int = 6
# Factor de upscaling interno de EasyOCR. 1.3 mejora detección de
# texto muy pequeño (manga) sin saturar con ruido de fondo.
_OCR_MAG_RATIO: float = 1.3


def _run_ocr_on_image(reader: Any, img_bgr: _Img) -> list[Any]:
    """
    Ejecuta EasyOCR sobre una imagen y retorna resultados crudos.
    Adquiere el semáforo OCR (solo una llamada a la vez).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    acquired = _ocr_semaphore.acquire(blocking=True, timeout=120)
    try:
        return reader.readtext(
            img_rgb,
            detail=1,
            paragraph=False,
            min_size=_OCR_MIN_SIZE,
            text_threshold=_OCR_TEXT_THRESHOLD,
            low_text=_OCR_LOW_TEXT,
            link_threshold=0.3,
            canvas_size=min(max(img_bgr.shape[:2]), _OCR_CANVAS_SIZE),
            mag_ratio=_OCR_MAG_RATIO,
        )
    except Exception as e:
        print(f"[OCR] Error en readtext: {e}")
        return []
    finally:
        if acquired:
            _ocr_semaphore.release()


def _ocr_results_to_blocks(results: list[Any], img_bgr: _Img) -> list[dict[str, Any]]:
    """Convierte resultados crudos de EasyOCR al formato interno de bloques."""
    blocks: list[dict[str, Any]] = []
    for (bbox, text, conf) in results:
        text = str(text).strip()
        if not text or conf < 0.08:
            continue
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        if w < 3 or h < 3:
            continue
        font_size = max(8, int(h * 0.75))
        cx, cy = x + w // 2, y + h // 2
        pad = max(2, h // 6)
        roi = img_bgr[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
        text_color = "#000000"
        if roi.size > 0:
            mean_b = int(np.mean(roi[:, :, 0]))
            mean_g = int(np.mean(roi[:, :, 1]))
            mean_r = int(np.mean(roi[:, :, 2]))
            brightness = mean_r * 0.299 + mean_g * 0.587 + mean_b * 0.114
            text_color = "#ffffff" if brightness > 128 else "#000000"

        blocks.append({
            "x": x, "y": y, "w": w, "h": h,
            "text": text,
            "confidence": float(conf),
            "fontSize": font_size,
            "textColor": text_color,
        })

    return _group_and_merge_blocks(blocks, img_bgr.shape[0])


def _detect_and_ocr(
    img_bgr: _Img,
    lang_hint: str = "auto",
    allow_fallback: bool = True,
    prefilter: bool = False,
    use_hybrid: bool = True,
    avg_conf_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """
    OCR con pipeline HÍBRIDO de 3 niveles EasyOCR + RapidOCR:

    1. **EasyOCR directo** (GPU, ~1.16s) — rápido para texto normal.
       Si prefilter=True, aplica limpieza morfológica antes.
    2. **CLAHE+sharpen -> EasyOCR** (~0.3s extra) — fallback para
       texto artístico/decorativo que EasyOCR no captura directamente.
    3. **RapidOCR** (CPU ONNX, ~2.4s) — fallback final usando los
       mismos modelos PP-OCRv4 que PaddleOCR pero sin conflictos
       de PaddlePaddle. Se activa solo cuando:
       - EasyOCR devuelve 0 bloques (todo falló), O
       - La confianza promedio de EasyOCR es < avg_conf_threshold
         (texto artístico detectado débilmente)

    Si ambos OCRs devuelven bloques y la confianza de EasyOCR es baja,
    los resultados se fusionan con _fusionar_blocks(): por cada texto
    detectado por ambos, se queda con el de mayor confianza.

    Args:
        allow_fallback: Si False, solo ejecuta tier 1 (EasyOCR directo).
        prefilter: Si True, aplica _pre_filter_image antes del tier 1.
        use_hybrid: Si True, activa tier 3 (RapidOCR) como fallback.
        avg_conf_threshold: Si la confianza promedio de EasyOCR es
            menor a este valor, se activa el tier hibrido (RapidOCR + fusión).
    """
    reader = _get_ocr_reader(lang_hint)
    if reader is None:
        return []

    # ── Pre-filter opcional (antes de tier 1) ────────────────────
    img_ocr = _pre_filter_image(img_bgr) if prefilter else img_bgr

    # ── Intento 1: EasyOCR directo ───────────────────────────────
    results = _run_ocr_on_image(reader, img_ocr)
    blocks_easy = _ocr_results_to_blocks(results, img_ocr)

    if blocks_easy and use_hybrid:
        avg_conf = float(np.mean([b.get("confidence", 0) for b in blocks_easy]))
        print(f"[OCR] EasyOCR: {len(blocks_easy)} bloques (conf={avg_conf:.2f})")

        # Si la confianza es suficiente, devolver EasyOCR directamente
        if avg_conf >= avg_conf_threshold:
            return blocks_easy

        # Confianza baja: activar pipeline híbrido RapidOCR
        print(f"[OCR] Confianza baja ({avg_conf:.2f} < {avg_conf_threshold}). "
              f"Ejecutando RapidOCR para complementar...")
        img_rapid = _preprocess_rapid(img_bgr)
        rapid_blocks = _run_rapidocr(img_rapid)

        if rapid_blocks:
            merged = _fusionar_blocks(blocks_easy, rapid_blocks)
            print(f"[OCR] Híbrido EasyOCR+ RapidOCR: {len(merged)} bloques"
                  f" (Easy={len(blocks_easy)}, Rapid={len(rapid_blocks)})")
            return merged
        return blocks_easy

    if blocks_easy:
        avg_conf = float(np.mean([b.get("confidence", 0) for b in blocks_easy]))
        print(f"[OCR] EasyOCR: {len(blocks_easy)} bloques (conf={avg_conf:.2f})")
        return blocks_easy

    # ── RapidOCR como fallback cuando use_hybrid=True ────────────
    # Se ejecuta incluso si allow_fallback=False, porque RapidOCR
    # es un motor diferente (ONNX), no "fallback de EasyOCR".
    if use_hybrid:
        print("[OCR] EasyOCR dio 0 bloques. Probando RapidOCR directamente...")
        img_rapid = _preprocess_rapid(img_bgr)
        rapid_blocks = _run_rapidocr(img_rapid)
        if rapid_blocks:
            print(f"[OCR] RapidOCR detecto {len(rapid_blocks)} bloques!")
            return rapid_blocks

    if not allow_fallback:
        return []

    # ── Intento 2: Pre-filter + CLAHE + sharpen ──────────────────
    if not prefilter:
        print("[OCR] 0 bloques con EasyOCR. Probando pre-filter + CLAHE+sharpen...")
        img_filtered = _pre_filter_image(img_bgr)
    else:
        print("[OCR] 0 bloques con EasyOCR (pre-filter ya aplicado). Probando CLAHE+sharpen...")
        img_filtered = img_ocr
    img_enhanced = _preprocess_enhanced(img_filtered)
    results2 = _run_ocr_on_image(reader, img_enhanced)
    blocks2 = _ocr_results_to_blocks(results2, img_enhanced)

    if blocks2:
        print(f"[OCR] Pre-filter+CLAHE detecto {len(blocks2)} bloques!")
        return blocks2

    print("[OCR] Todos los fallbacks agotados")
    return []


# ─── Filtro de marcas de agua ──────────────────────────────────
def _filter_watermarks_from_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not blocks:
        return []
    result: list[dict[str, Any]] = []
    for b in blocks:
        text = str(b.get("text", "")).strip()
        if any(p.search(text) for p in WATERMARK_PATTERNS):
            print(f"[filter-watermark] Bloque filtrado: '{text}'")
            continue
        result.append(b)
    return result


# ─── Agrupación y fusión de bloques ────────────────────────────
def _group_and_merge_blocks(blocks: list[dict[str, Any]], img_h: int | None = None) -> list[dict[str, Any]]:
    if not blocks:
        return []

    margin_top: float | None = img_h * 0.07 if img_h else None
    margin_bottom: float | None = img_h * 0.96 if img_h else None

    pre_filtered: list[dict[str, Any]] = []
    for b in blocks:
        text_raw = str(b["text"]).strip()
        h = b["h"]
        cy = b["y"] + h / 2

        # ── Filtros contra texto ORIGINAL (antes de limpiar simbolos) ──
        # IMPORTANTE: Los patrones de ruido (fecha, hora: "13/7/26", "4.58 p.m")
        # dependen de / y . que la limpieza de simbolos elimina. Verificar
        # contra text_raw preserva estos caracteres.
        if any(p.search(text_raw) for p in WATERMARK_PATTERNS):
            print(f"[OCR] Filtrando marca de agua pre-merge: '{text_raw[:50]}'")
            continue

        in_margin = margin_top is not None and (cy < margin_top or cy > margin_bottom)
        if in_margin:
            if any(p.search(text_raw) for p in MARGIN_NOISE_PATTERNS):
                print(f"[OCR] Filtrando metadato de margen pre-merge: '{text_raw[:50]}' en y={b['y']}")
                continue
            digit_count = sum(1 for c in text_raw if c.isdigit())
            if len(text_raw) > 0 and (digit_count / len(text_raw) >= 0.35) and len(text_raw.split()) <= 4:
                print(f"[OCR] Filtrando metadato numérico de margen pre-merge: '{text_raw[:50]}' en y={b['y']}")
                continue

        if re.search(r'https?://|www\.|\.(com|net|org|xyz|io)\b', text_raw, re.IGNORECASE):
            print(f"[OCR] Filtrando URL pre-merge: '{text_raw}'")
            continue

        # ── Ahora corregir errores de OCR con glosario y limpiar simbolos ──
        from translator import _aplicar_glosario
        text_corr = _aplicar_glosario(text_raw)
        text = re.sub(r'[@#$%^&*()+={}\[\]|:;<>/\\]', '', text_corr).strip()
        b["text"] = text if text else text_corr

        pre_filtered.append(b)

    if not pre_filtered:
        return []

    sorted_b = sorted(pre_filtered, key=lambda b: (b["y"] + b["h"] / 2, b["x"]))
    merged: list[dict[str, Any]] = []
    used = [False] * len(sorted_b)

    for i, b in enumerate(sorted_b):
        if used[i]:
            continue
        group = [b]
        used[i] = True
        for j, b2 in enumerate(sorted_b):
            if used[j] or i == j:
                continue
            group_x2 = max(g["x"] + g["w"] for g in group)
            cy1 = group[-1]["y"] + group[-1]["h"] / 2
            cy2 = b2["y"] + b2["h"] / 2
            max_h = max(group[-1]["h"], b2["h"])
            gap_x = b2["x"] - group_x2

            if abs(cy1 - cy2) < max_h * 0.45 and -b2["w"] < gap_x < max(35, group[-1]["w"] * 2.5):
                group.append(b2)
                used[j] = True

        if len(group) == 1:
            merged.append(b)
        else:
            group.sort(key=lambda g: g["x"])
            all_x = [g["x"] for g in group]
            all_y = [g["y"] for g in group]
            all_x2 = [g["x"] + g["w"] for g in group]
            all_y2 = [g["y"] + g["h"] for g in group]
            mx = min(all_x)
            my = min(all_y)
            mw = max(all_x2) - mx
            mh = max(all_y2) - my
            merged.append({
                "x": mx, "y": my, "w": mw, "h": mh,
                "text": " ".join(g["text"] for g in group),
                "confidence": float(np.mean([g["confidence"] for g in group])),
                "fontSize": max(g["fontSize"] for g in group),
                "textColor": group[0]["textColor"],
            })

    # ─── Segunda pasada: fusión vertical (líneas de mismo globo) ───
    # Combina bloques en la misma columna y cerca verticalmente
    if len(merged) > 1:
        sorted_v = sorted(merged, key=lambda b: (b["x"], b["y"]))
        v_merged: list[dict[str, Any]] = []
        v_used = [False] * len(sorted_v)
        for i, b in enumerate(sorted_v):
            if v_used[i]:
                continue
            v_group = [b]
            v_used[i] = True
            for j, b2 in enumerate(sorted_v):
                if v_used[j] or i == j:
                    continue
                # Mismo columna (x overlap significativo)
                x_overlap = min(b["x"] + b["w"], b2["x"] + b2["w"]) - max(b["x"], b2["x"])
                min_w = min(b["w"], b2["w"])
                if x_overlap < min_w * 0.5:
                    continue
                # Cerca verticalmente (gap < 1.5x altura)
                gap_y = abs(b2["y"] - (b["y"] + b["h"]))
                if gap_y < max(b["h"], b2["h"]) * 1.5:
                    v_group.append(b2)
                    v_used[j] = True
            if len(v_group) > 1:
                v_group.sort(key=lambda g: g["y"])
                all_x = [g["x"] for g in v_group]
                all_y = [g["y"] for g in v_group]
                all_x2 = [g["x"] + g["w"] for g in v_group]
                all_y2 = [g["y"] + g["h"] for g in v_group]
                mx = min(all_x)
                my = min(all_y)
                mw = max(all_x2) - mx
                mh = max(all_y2) - my
                v_merged.append({
                    "x": mx, "y": my, "w": mw, "h": mh,
                    "text": " ".join(g["text"] for g in v_group),
                    "confidence": float(np.mean([g["confidence"] for g in v_group])),
                    "fontSize": max(g["fontSize"] for g in v_group),
                    "textColor": v_group[0]["textColor"],
                })
            else:
                v_merged.append(b)
        merged = v_merged

    final_blocks: list[dict[str, Any]] = []
    for b in merged:
        w, h = b["w"], b["h"]
        text = str(b["text"]).strip()
        text_len = len(text)
        conf = float(b.get("confidence", 0))

        if text_len == 0:
            continue

        if re.match(r'^\d+$', text):
            print(f"[OCR] Filtrando número puro post-merge: '{text}' en [{b['x']},{b['y']}]")
            continue

        if re.match(r'^[\d\s.,;:!?%\'\"\-–—/\\]+$', text):
            print(f"[OCR] Filtrando patrón numérico post-merge: '{text}'")
            continue

        if re.match(r'^["\']+\d', text) and text_len <= 6:
            print(f"[OCR] Filtrando comilla+número post-merge: '{text}'")
            continue

        if re.match(r'^[~\'\"\-–—:;,.!¡¿?=]+$', text) and text_len <= 2:
            print(f"[OCR] Filtrando puntuación suelta post-merge: '{text}'")
            continue

        aspect = w / max(h, 1)
        if aspect < 0.4 and text_len <= 3:
            print(f"[OCR] Filtrando ruido estrecho post-merge: '{text}' aspect={aspect:.2f}")
            continue

        if text_len == 1 and conf < 0.25:
            print(f"[OCR] Filtrando carácter suelto post-merge: '{text}'")
            continue

        if conf < 0.15 and text_len <= 3:
            print(f"[OCR] Filtrando texto corto baja confianza post-merge: '{text}' conf={conf:.2f}")
            continue

        if text_len <= 3 and re.match(r'^\d+[a-zA-Z]?$', text):
            print(f"[OCR] Filtrando combo digito+letra post-merge: '{text}'")
            continue

        final_blocks.append(b)

    return final_blocks


# ─── Detección de globo de diálogo ─────────────────────────────
def _is_inside_speech_bubble(img_bgr: _Img, block: dict[str, Any]) -> bool:
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]
    edge = max(3, int(min(bw, bh) * 0.15))

    samples: list[NDArray[np.float64]] = []
    for region_coords in [
        (max(0, by), min(h, by + edge), max(0, bx), min(w, bx + bw)),
        (max(0, by + bh - edge), min(h, by + bh), max(0, bx), min(w, bx + bw)),
        (max(0, by), min(h, by + bh), max(0, bx), min(w, bx + edge)),
        (max(0, by), min(h, by + bh), max(0, bx + bw - edge), min(w, bx + bw)),
    ]:
        y1, y2, x1, x2 = region_coords
        if y2 > y1 and x2 > x1:
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                mean_bgr = roi.reshape(-1, 3).mean(axis=0)
                samples.append(mean_bgr)

    if len(samples) < 3:
        return False

    avg_bgr = np.mean(samples, axis=0)
    b, g, r = avg_bgr
    brightness = float(r * 0.299 + g * 0.587 + b * 0.114)

    if brightness > 80:
        return False

    samples_arr = np.array(samples)
    std_per_channel = samples_arr.std(axis=0)
    max_std = float(std_per_channel.max())
    if max_std > 35:
        return False

    return True


# ─── Máscara de glifos para globos de diálogo ──────────────────
def _build_glyph_mask_for_bubble(img_bgr: _Img, block: dict[str, Any]) -> _Img:
    h, w = img_bgr.shape[:2]
    mask: _Img = np.zeros((h, w), dtype=np.uint8)
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

    edge = max(3, int(min(bw, bh) * 0.15))
    bg_pixels: list[_Img] = []
    for region_coords in [
        (max(0, by), min(h, by + edge), max(0, bx), min(w, bx + bw)),
        (max(0, by + bh - edge), min(h, by + bh), max(0, bx), min(w, bx + bw)),
        (max(0, by), min(h, by + bh), max(0, bx), min(w, bx + edge)),
        (max(0, by), min(h, by + bh), max(0, bx + bw - edge), min(w, bx + bw)),
    ]:
        y1, y2, x1, x2 = region_coords
        if y2 > y1 and x2 > x1:
            roi = img_bgr[y1:y2, x1:x2]
            bg_pixels.append(roi.reshape(-1, 3))

    if not bg_pixels:
        pad_x = max(5, int(bw * 0.08))
        pad_y = max(5, int(bh * 0.12))
        x1 = max(0, bx - pad_x)
        y1 = max(0, by - pad_y)
        x2 = min(w, bx + bw + pad_x)
        y2 = min(h, by + bh + pad_y)
        mask[y1:y2, x1:x2] = 255
        return mask

    all_bg = np.concatenate(bg_pixels, axis=0)
    bg_mean = all_bg.mean(axis=0)

    pad_x = max(3, int(bw * 0.04))
    pad_y = max(3, int(bh * 0.06))
    rx1 = max(0, bx - pad_x)
    ry1 = max(0, by - pad_y)
    rx2 = min(w, bx + bw + pad_x)
    ry2 = min(h, by + bh + pad_y)

    roi = img_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return mask

    # ── Enfoque híbrido: diferencia de color + Canny ────────────
    # La diferencia de color funciona bien para texto sobre fondo
    # liso (globos de diálogo). Canny captura bordes finos que la
    # diferencia de color puede perder (glifos rotados o artísticos).

    # 1. Diferencia de color (método original)
    roi_float = roi.astype(np.float32)
    diff = roi_float - bg_mean.astype(np.float32)
    color_dist = np.sqrt((diff ** 2).sum(axis=2))
    glyph_threshold = 60
    color_mask = (color_dist > glyph_threshold).astype(np.uint8) * 255

    # 2. Canny edge detection en la región
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Umbral adaptativo: sigma de la mediana de la imagen
    median_val = float(np.median(roi_gray))
    sigma = 0.33
    lower = int(max(1, (1.0 - sigma) * median_val))  # Clamp min 1 (Canny con ambos thresholds=0 es indefinido)
    upper = int(min(255, (1.0 + sigma) * median_val))
    canny_edges = cv2.Canny(roi_gray, threshold1=lower, threshold2=upper)

    # 3. Fusionar: color OR canny
    combined = cv2.bitwise_or(color_mask, canny_edges)

    # 4. Cerrar pequeños gaps en los bordes
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    mask[ry1:ry2, rx1:rx2] = combined

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ─── Construcción de máscara de inpainting ─────────────────────
def _build_inpaint_mask(img_bgr: _Img, blocks: list[dict[str, Any]]) -> _Img:
    h, w = img_bgr.shape[:2]
    mask: _Img = np.zeros((h, w), dtype=np.uint8)

    for block in blocks:
        bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

        if _is_inside_speech_bubble(img_bgr, block):
            print(f"[inpaint] Bloque en globo de dialogo detectado: '{block['text'][:30]}' -> mascara solo-glifos")
            glyph_mask = _build_glyph_mask_for_bubble(img_bgr, block)
            mask = cv2.bitwise_or(mask, glyph_mask)
        else:
            pad_x = max(5, int(bw * 0.08))
            pad_y = max(5, int(bh * 0.12))
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)
            mask[y1:y2, x1:x2] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ─── Inpainting con OpenCV ──────────────────────────────────────
def _inpaint_image(img_bgr: _Img, mask: _Img, blocks: list[dict[str, Any]] | None = None) -> _Img:
    if int(mask.max()) == 0:
        return img_bgr.copy()

    if blocks:
        avg_height = float(np.mean([b["h"] for b in blocks])) if blocks else 20.0
        radius = int(np.clip(avg_height * 0.6, 5, 30))
    else:
        covered = float(np.sum(mask > 0))
        total = float(mask.size)
        coverage_ratio = covered / total if total > 0 else 0.0
        radius = int(np.clip(12 + coverage_ratio * 30, 10, 40))

    print(f"[inpaint] Radio: {radius}, bloques: {len(blocks) if blocks else 0}")

    flags = cv2.INPAINT_NS
    result = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=flags)
    return result


# ─── Muestreo de color de fondo ────────────────────────────────
def _sample_bg_color(img_bgr: _Img, block: dict[str, Any]) -> str:
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

    if _is_inside_speech_bubble(img_bgr, block):
        edge = max(4, int(min(bw, bh) * 0.12))
        samples: list[NDArray[np.float64]] = []
        for region_coords in [
            (max(0, by), min(h, by + edge), max(0, bx), min(w, bx + bw)),
            (max(0, by + bh - edge), min(h, by + bh), max(0, bx), min(w, bx + bw)),
            (max(0, by), min(h, by + bh), max(0, bx), min(w, bx + edge)),
            (max(0, by), min(h, by + bh), max(0, bx + bw - edge), min(w, bx + bw)),
        ]:
            y1, y2, x1, x2 = region_coords
            if y2 > y1 and x2 > x1:
                region = img_bgr[y1:y2, x1:x2]
                if region.size > 0:
                    samples.append(np.mean(region.reshape(-1, 3), axis=0))

        if samples:
            mean_bgr = np.mean(samples, axis=0)
            b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
            return f"#{r:02x}{g:02x}{b:02x}"

        return "#000000"

    pad = max(5, int(min(bw, bh) * 0.2))
    samples = []
    y1, y2 = max(0, by - pad), max(0, by)
    if y2 > y1:
        region = img_bgr[y1:y2, max(0, bx):min(w, bx + bw)]
        if region.size > 0:
            samples.append(np.mean(region.reshape(-1, 3), axis=0))
    y1, y2 = min(h, by + bh), min(h, by + bh + pad)
    if y2 > y1:
        region = img_bgr[y1:y2, max(0, bx):min(w, bx + bw)]
        if region.size > 0:
            samples.append(np.mean(region.reshape(-1, 3), axis=0))

    if not samples:
        return "#ffffff"
    mean_bgr = np.mean(samples, axis=0)
    b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"
