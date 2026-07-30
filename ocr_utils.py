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
                    font_size = max(8, int(h * 0.75))
                    cx, cy = x + w // 2, y + h // 2
                    pad = max(2, h // 6)
                    roi = img_bgr[max(0, cy - pad):cy + pad, max(0, cx - pad):cx + pad]
                    text_color = "#ffffff"
                    if roi.size > 0:
                        mean_b = int(np.mean(roi[:, :, 0]))
                        mean_g = int(np.mean(roi[:, :, 1]))
                        mean_r = int(np.mean(roi[:, :, 2]))
                        brightness = mean_r * 0.299 + mean_g * 0.587 + mean_b * 0.114
                        text_color = "#000000" if brightness > 128 else "#ffffff"

                    blocks.append({
                        "x": x, "y": y, "w": w, "h": h,
                        "text": text,
                        "confidence": float(conf),
                        "fontSize": font_size,
                        "textColor": text_color,
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

    def _block_score(b: dict[str, Any]) -> float:
        text_len = len(str(b.get("text", "")).strip())
        conf = float(b.get("confidence", 0.5))
        return conf * min(2.0, max(0.5, text_len / 5.0))

    def _overlap_ratio(b1: dict[str, Any], b2: dict[str, Any]) -> float:
        x1 = max(b1["x"], b2["x"])
        y1 = max(b1["y"], b2["y"])
        x2 = min(b1["x"] + b1["w"], b2["x"] + b2["w"])
        y2 = min(b1["y"] + b1["h"], b2["y"] + b2["h"])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        min_area = min(b1["w"] * b1["h"], b2["w"] * b2["h"])
        return inter / float(min_area) if min_area > 0 else 0.0

    # 1. Deduplicación por texto idéntico normalizado
    easy_by_text: dict[str, dict[str, Any]] = {_normalize_text(b["text"]): b for b in easy_blocks}
    rapid_by_text: dict[str, dict[str, Any]] = {_normalize_text(b["text"]): b for b in rapid_blocks}

    common = set(easy_by_text.keys()) & set(rapid_by_text.keys())
    only_easy = set(easy_by_text.keys()) - common
    only_rapid = set(rapid_by_text.keys()) - common

    candidates: list[dict[str, Any]] = []
    for t in common:
        if _block_score(easy_by_text[t]) >= _block_score(rapid_by_text[t]):
            candidates.append(easy_by_text[t])
        else:
            candidates.append(rapid_by_text[t])
    for t in only_easy:
        candidates.append(easy_by_text[t])
    for t in only_rapid:
        candidates.append(rapid_by_text[t])

    # 2. Deduplicación por solapamiento espacial (IoU / intersección > 40%)
    # Elimina cajas duplicadas que ambos motores hayan detectado en la misma posición
    candidates.sort(key=lambda b: _block_score(b), reverse=True)
    final_result: list[dict[str, Any]] = []
    for b in candidates:
        is_duplicate = False
        for existing in final_result:
            if _overlap_ratio(b, existing) > 0.40:
                is_duplicate = True
                break
        if not is_duplicate:
            final_result.append(b)

    return final_result


# ─── Corrector ortografico post-OCR (pyspellchecker) ────────────
# Usa pyspellchecker en lugar de un diccionario manual.
# pyspellchecker trae diccionarios completos de espanol, ingles,
# frances, aleman, portugues, etc. SIN mantenimiento manual.
# Correccion por distancia de Levenshtein (max 2).
# Fallback a _levenshtein() si pyspellchecker no esta instalado.
_OCR_SPELLCHECKER: Any = None
_OCR_SPELL_LOCK: threading.Lock = threading.Lock()


def _get_spellchecker(lang: str = "es") -> Any:
    """Retorna instancia de SpellChecker (lazy load, thread-safe)."""
    global _OCR_SPELLCHECKER
    if _OCR_SPELLCHECKER is not None:
        return _OCR_SPELLCHECKER
    with _OCR_SPELL_LOCK:
        if _OCR_SPELLCHECKER is not None:
            return _OCR_SPELLCHECKER
        try:
            from spellchecker import SpellChecker
            sp = SpellChecker(language=lang)
            wf = sp.word_frequency
            # Palabras del dominio manga que pyspellchecker podria no conocer
            # o tener baja frecuencia. Lista FIJA - no requiere mantenimiento.
            # Usamos .add(word, frequency) con alta frecuencia para que
            # pyspellchecker las prefiera sobre alternativas incorrectas.
            MANGA_WORDS = {
                'villano': 1000000, 'villanos': 1000000,
                'villana': 1000000, 'villanas': 1000000,
                'manga': 1000000, 'manhwa': 1000000,
                'webtoon': 1000000, 'scanlation': 1000000,
                'capitulo': 1000000, 'episodio': 1000000,
                'temporada': 1000000, 'personaje': 1000000,
                'protagonista': 1000000, 'antagonista': 1000000,
                'comic': 1000000, 'anime': 1000000,
            }
            for word, freq in MANGA_WORDS.items():
                wf.add(word, freq)
            _OCR_SPELLCHECKER = sp
            print(f"[OCR-spellcheck] pyspellchecker listo (lang={lang}, "
                  f"{len(wf.dictionary)} palabras + {len(MANGA_WORDS)} palabras manga)")
        except Exception as e:
            print(f"[OCR-spellcheck] No se pudo cargar pyspellchecker: {e}")
            return None
    return _OCR_SPELLCHECKER


def _levenshtein(s1: str, s2: str) -> int:
    """Distancia de Levenshtein entre dos strings (iterativa, O(n*m))."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if not s2:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(
                curr[j] + 1,          # delete
                prev[j + 1] + 1,      # insert
                prev[j] + cost         # substitute
            ))
        prev = curr
    return prev[-1]


def _ocr_spellcheck(text: str) -> str:
    """
    Corrector ortografico post-OCR.

    Para cada palabra no reconocida en el diccionario, busca la
    palabra mas cercana por distancia de Levenshtein (max 2).
    Si encuentra una coincidencia, la reemplaza.

    No corrige:
    - Palabras de 1-2 caracteres (muy ambiguas)
    - Palabras en mayusculas que parecen acronimos
    - Nombres propios (no estan en el diccionario, pero no los fuerza)

    Esto funciona para CUALQUIER PDF sin necesidad de mantener GLOSARIO_PRE.
    """
    if not text or len(text) <= 2:
        return text

    palabras = text.split()
    corregidas: list[str] = []
    # Obtener spellchecker UNA VEZ fuera del loop
    sp = _get_spellchecker()

    for p in palabras:
        stripped = p.strip("'\".,;:!?¡¿()[]{}")
        if not stripped or len(stripped) <= 2:
            corregidas.append(p)
            continue

        # Si parece un acronimo (todo mayusculas, >2 chars), no corregir
        if stripped.isupper() and len(stripped) > 2:
            corregidas.append(p)
            continue

        # Si contiene digitos, no corregir
        if any(c.isdigit() for c in stripped):
            corregidas.append(p)
            continue

        # Intentar correccion con pyspellchecker
        p_lower = stripped.lower()

        if sp is not None:
            # pyspellchecker.correction(word) retorna None si la palabra es correcta
            # o la palabra corregida si no lo es (distancia Levenshtein implicita)
            correccion = sp.correction(p_lower)
            if correccion is not None and correccion != p_lower:
                # Preservar capitalizacion original
                if p[0].isupper() and not p.isupper():
                    corregida = correccion.capitalize()
                elif p.isupper():
                    corregida = correccion.upper()
                else:
                    corregida = correccion
                corregidas.append(corregida)
                print(f"[OCR-spellcheck] '{p}' -> '{corregida}'")
                continue
        else:
            # Fallback: _levenshtein sobre palabras comunes clave
            # Solo un minimo necesario para que funcione sin pyspellchecker
            _FALLBACK_DICT: frozenset[str] = frozenset({
                "correctamente", "correcto", "villano", "villanos",
                "temporada", "capitulo", "como", "criar", "persona",
                "primero", "segundo", "tercero", "despues", "entonces",
                "the", "and", "for", "with", "from", "this", "that",
            })
            mejor_dist = 999
            mejor_palabra = None
            for dw in _FALLBACK_DICT:
                if abs(len(dw) - len(p_lower)) > 2:
                    continue
                dist = _levenshtein(p_lower, dw)
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_palabra = dw
                    if dist == 0:
                        break
            if mejor_palabra and 0 < mejor_dist <= 2:
                if p[0].isupper() and not p.isupper():
                    corregida = mejor_palabra.capitalize()
                elif p.isupper():
                    corregida = mejor_palabra.upper()
                else:
                    corregida = mejor_palabra
                corregidas.append(corregida)
                print(f"[OCR-spellcheck] '{p}' -> '{corregida}' (fallback, dist={mejor_dist})")
                continue

        corregidas.append(p)

    return " ".join(corregidas)


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


def _run_ocr_on_image(reader: Any, img_bgr: _Img, mag_ratio: float | None = None) -> list[Any]:
    """
    Ejecuta EasyOCR sobre una imagen y retorna resultados crudos.
    Adquiere el semáforo OCR (solo una llamada a la vez).

    Args:
        mag_ratio: Factor de upscaling. Si es None, usa _OCR_MAG_RATIO (1.3).
                   Valores mas altos (1.5-2.0) mejoran deteccion de texto
                   artistico pero anaden mas ruido.
    """
    mag = mag_ratio if mag_ratio is not None else _OCR_MAG_RATIO
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
            mag_ratio=mag,
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
            text_color = "#000000" if brightness > 128 else "#ffffff"

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
    prefilter: bool = True,
    use_hybrid: bool = True,
    avg_conf_threshold: float = 0.15,
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

    if blocks_easy:
        avg_conf = float(np.mean([b.get("confidence", 0) for b in blocks_easy]))
        print(f"[OCR] EasyOCR: {len(blocks_easy)} bloques (conf={avg_conf:.2f})")

        # ── mag_ratio adaptativo: si confianza baja, reintentar con mas resolucion ──
        # Esto corrige errores en tipografia artistica (portadas, titulos decorativos)
        # donde mag_ratio=1.3 no da suficiente detalle. Al aumentar a 1.8, EasyOCR
        # puede distinguir mejor caracteres como V vs Y, R vs Y, etc.
        if avg_conf < avg_conf_threshold:
            print(f"[OCR] Confianza baja ({avg_conf:.2f} < {avg_conf_threshold}). "
                  f"Reintentando con mag_ratio=1.8...")
            results_high = _run_ocr_on_image(reader, img_ocr, mag_ratio=1.8)
            blocks_high = _ocr_results_to_blocks(results_high, img_ocr)
            if blocks_high:
                avg_conf_high = float(np.mean([b.get("confidence", 0) for b in blocks_high]))
                print(f"[OCR] mag_ratio=1.8: {len(blocks_high)} bloques (conf={avg_conf_high:.2f})")
                # Usar el resultado de mayor mag_ratio si tiene mejor confianza O mas bloques
                if avg_conf_high > avg_conf or len(blocks_high) > len(blocks_easy):
                    blocks_easy = blocks_high
                    avg_conf = avg_conf_high

    # ── Híbrido: Ejecutar RapidOCR siempre para capturar texto estilizado/títulos ──
    if use_hybrid:
        print("[OCR] Ejecutando RapidOCR para complementar texto estilizado y títulos...")
        img_rapid = _preprocess_rapid(img_bgr)
        rapid_blocks = _run_rapidocr(img_rapid)

        if rapid_blocks:
            merged = _fusionar_blocks(blocks_easy, rapid_blocks)
            print(f"[OCR] Híbrido EasyOCR + RapidOCR: {len(merged)} bloques totales "
                  f"(Easy={len(blocks_easy)}, Rapid={len(rapid_blocks)})")
            return merged

    if blocks_easy:
        return blocks_easy

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

    margin_top: float | None = img_h * 0.085 if img_h else None
    margin_bottom: float | None = img_h * 0.955 if img_h else None

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

        # ── Ahora corregir errores de OCR con glosario, limpiar simbolos y spellcheck ──
        from translator import _aplicar_glosario
        text_corr = _aplicar_glosario(text_raw)
        text = re.sub(r'[@#$%^&*()+={}\[\]|:;<>/\\]', '', text_corr).strip()
        # Corrector ortografico post-OCR: corrige errores como V->Y, R->Y, etc.
        # SIN depender de GLOSARIO_PRE. Funciona para cualquier PDF.
        text = _ocr_spellcheck(text)
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
                # No fusionar verticalmente cabeceras del margen superior (y < 2.5% img_h) con contenido
                if img_h and (b["y"] < img_h * 0.025) != (b2["y"] < img_h * 0.025):
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
    """Detecta si un bloque está sobre un fondo uniforme (burbuja o panel).

    Un globo de diálogo se caracteriza por bordes de color UNIFORME
    (std < 35 por canal), sin importar si es blanco, negro o coloreado.
    Ya NO controla la estrategia de inpainting (siempre glifos primero).
    Se usa para logging y decisiones de estilo.
    """
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

    # Uniformidad del borde = burbuja/panel (blanco, negro, o cualquier color)
    samples_arr = np.array(samples)
    std_per_channel = samples_arr.std(axis=0)
    max_std = float(std_per_channel.max())
    return max_std <= 35


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

    pad_x = max(3, int(bw * 0.04))
    pad_y = max(3, int(bh * 0.06))
    rx1 = max(0, bx - pad_x)
    ry1 = max(0, by - pad_y)
    rx2 = min(w, bx + bw + pad_x)
    ry2 = min(h, by + bh + pad_y)

    roi = img_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return mask

    if bg_pixels:
        all_bg = np.concatenate(bg_pixels, axis=0)
        bg_mean = all_bg.mean(axis=0)
    else:
        bg_mean = roi.mean(axis=(0, 1))

    # ── Enfoque híbrido: diferencia de color + Canny ────────────
    # La diferencia de color funciona bien para texto sobre fondo
    # liso (globos de diálogo). Canny captura bordes finos que la
    # diferencia de color puede perder (glifos rotados o artísticos).

    # ── Detección de glifos basada en contraste local ───────────
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    local_median = int(np.median(roi_gray))
    diff_from_median = cv2.absdiff(roi_gray, local_median)
    
    # Píxeles de texto: aquellos que difieren de forma destacada de la mediana del ROI
    glyph_mask_roi = (diff_from_median > 45).astype(np.uint8) * 255

    # Refinar bordes con cierre ligero
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined = cv2.morphologyEx(glyph_mask_roi, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    mask[ry1:ry2, rx1:rx2] = combined
    return mask


# ─── Construcción de máscara de inpainting ─────────────────────
def _build_inpaint_mask(img_bgr: _Img, blocks: list[dict[str, Any]]) -> _Img:
    """Construye máscara de inpainting para borrar texto detectado.

    Estrategia: Usar SIEMPRE máscara de glifos basada en trazos de letra.
    No se utilizan máscaras rectangulares sólidas porque destruyen el arte
    del manga (burbujas, ilustraciones, gradientes).
    """
    h, w = img_bgr.shape[:2]
    mask: _Img = np.zeros((h, w), dtype=np.uint8)

    for block in blocks:
        glyph_mask = _build_glyph_mask_for_bubble(img_bgr, block)
        glyph_pixels = int(np.sum(glyph_mask > 0))
        if glyph_pixels > 0:
            mask = cv2.bitwise_or(mask, glyph_mask)

    # Dilatación mínima (3x3) para cubrir solo bordes inmediatos de los trazos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


# ─── Inpainting con OpenCV (TELEA + border-blend) ──────────────
def _inpaint_image(img_bgr: _Img, mask: _Img, blocks: list[dict[str, Any]] | None = None) -> _Img:
    """
    Inpaint 2 fases: TELEA + border-blend.

    Fase 1 — TELEA (Fast Marching Method) con radio adaptativo mejorado:
        - Texto pequeño (<15px): radio 3
        - Texto normal (15-30px): radio 4
        - Texto grande (25-40px): radio 6
        - Texto muy grande (>40px): radio 8

    Fase 2 — Border-blend post-inpainting:
        Para cada bloque, extrae el color MEDIO del borde exterior del
        bloque DESDE LA IMAGEN ORIGINAL y lo mezcla suavemente con el
        relleno de TELEA (alpha=0.35). Esto elimina los parches oscuros
        o irregulares que TELEA deja en fondos complejos (portadas,
        ilustraciones con gradientes, sombras).

    El blend usa máscara dilatada + GaussianBlur para bordes suaves,
    evitando artefactos visuales en la transición.
    """
    if int(mask.max()) == 0:
        return img_bgr.copy()

    h, w = img_bgr.shape[:2]

    # ---- Fase 1: TELEA con radio adaptativo mejorado ----
    if blocks:
        heights = [b["h"] for b in blocks if b.get("h", 0) > 0]
        if heights:
            max_h = max(heights)
            if max_h < 15:
                radius = 3
            elif max_h > 40:
                radius = 8
            elif max_h > 25:
                radius = 6
            elif max_h > 15:
                radius = 4
            else:
                radius = 3
        else:
            radius = 4
    else:
        radius = 4

    print(f"[inpaint] TELEA radio={radius}px + border-blend")
    img_telea = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    # Si no hay bloques, devolver solo TELEA
    if not blocks:
        return img_telea

    # ---- Fase 2: Post-procesamiento con color de borde ----
    # Para cada bloque, rellenar el area de la mascara con el color
    # promedio del borde circundante de la imagen ORIGINAL.
    # Esto evita parches oscuros en fondos complejos (portadas).
    result = img_telea.copy()

    for block in blocks:
        bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

        # Padding para el borde de muestreo (15% del tamano del bloque)
        pad_x = max(5, int(bw * 0.15))
        pad_y = max(5, int(bh * 0.15))

        # Area expandida para muestrear el borde
        y1_exp = max(0, by - pad_y)
        y2_exp = min(h, by + bh + pad_y)
        x1_exp = max(0, bx - pad_x)
        x2_exp = min(w, bx + bw + pad_x)

        # Extraer pixeles del borde (fuera del bloque) de la imagen ORIGINAL
        expanded_roi = img_bgr[y1_exp:y2_exp, x1_exp:x2_exp]
        if expanded_roi.size == 0:
            continue

        # Coordenadas del bloque dentro del ROI expandido
        inner_y1 = by - y1_exp
        inner_x1 = bx - x1_exp
        inner_y2 = inner_y1 + bh
        inner_x2 = inner_x1 + bw

        border_pixels = []
        # Borde superior
        if inner_y1 > 0:
            top_roi = expanded_roi[:inner_y1, :, :]
            if top_roi.size > 0:
                border_pixels.append(top_roi.reshape(-1, 3))
        # Borde inferior
        if inner_y2 < expanded_roi.shape[0]:
            bot_roi = expanded_roi[inner_y2:, :, :]
            if bot_roi.size > 0:
                border_pixels.append(bot_roi.reshape(-1, 3))
        # Borde izquierdo (solo en el rango vertical del bloque)
        if inner_x1 > 0:
            left_roi = expanded_roi[inner_y1:inner_y2, :inner_x1, :]
            if left_roi.size > 0:
                border_pixels.append(left_roi.reshape(-1, 3))
        # Borde derecho
        if inner_x2 < expanded_roi.shape[1]:
            right_roi = expanded_roi[inner_y1:inner_y2, inner_x2:, :]
            if right_roi.size > 0:
                border_pixels.append(right_roi.reshape(-1, 3))

        if not border_pixels:
            continue

        # Calcular el color MEDIO del borde (mediana es mas robusta que la media)
        all_border = np.concatenate(border_pixels, axis=0)
        border_color = np.median(all_border, axis=0).astype(np.uint8)

        # Extraer la region de la mascara para este bloque
        mask_roi = mask[by:min(h, by + bh), bx:min(w, bx + bw)]
        if mask_roi.size == 0 or int(mask_roi.max()) == 0:
            continue

        # Crear mascara de mezcla (dilatar + blur para bordes suaves)
        blend_mask = (mask_roi > 0).astype(np.float32)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blend_mask = cv2.dilate(blend_mask, k, iterations=2)
        blend_mask = cv2.GaussianBlur(blend_mask, (7, 7), sigmaX=2)

        # Obtener el ROI del resultado para este bloque
        result_roi = result[by:min(h, by + bh), bx:min(w, bx + bw)]
        if result_roi.shape[:2] != blend_mask.shape[:2]:
            continue

        # Crear imagen de color de borde
        border_img = np.full_like(result_roi, border_color)

        # Alpha ADAPTATIVO basado en varianza del borde:
        #   - Baja varianza (fondo uniforme, burbuja blanca): alpha ~0.25
        #     -> preserva mas textura TELEA (el fondo original ya es bueno)
        #   - Alta varianza (fondo complejo, portada, gradiente, sombra): alpha ~0.50
        #     -> mas color de borde para eliminar parches oscuros/irregulares
        # La desviacion estandar del borde es un proxy directo de la
        # complejidad del fondo circundante.
        border_std = float(np.std(all_border))
        # Normalizar: std=80 (fondo muy heterogeneo) -> factor=1.0
        # std=80 es el doble del umbral de _is_inside_speech_bubble (std<=35)
        border_factor = min(1.0, border_std / 80.0)
        adaptive_alpha = 0.25 + 0.25 * border_factor  # rango [0.25, 0.50]
        if adaptive_alpha > 0.35:
            print(f"[inpaint] border-blend alpha={adaptive_alpha:.2f} "
                  f"(std={border_std:.0f}, fondo complejo)")
        alpha = adaptive_alpha * blend_mask[..., np.newaxis]
        blended = (result_roi * (1 - alpha) + border_img * alpha).astype(np.uint8)

        result[by:min(h, by + bh), bx:min(w, bx + bw)] = blended

    return result


# ---- Muestreo de color de fondo ---------------------------------
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
