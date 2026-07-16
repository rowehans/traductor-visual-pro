from __future__ import annotations

import base64
import io
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

# ─── Manga-image-translator pipeline (MIT) ──────────────────────
MIT_AVAILABLE = False
try:
    from manga_pipeline import run_pipeline, ensure_ready as mit_ensure_ready
    MIT_AVAILABLE = True
    print("[MIT] Pipeline de manga-image-translator disponible")
except Exception as e:
    print(f"[MIT] No disponible (modo legacy EasyOCR): {e}")

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
IS_PRODUCTION = DIST.exists() and (DIST / "index.html").exists()
app = Flask(__name__, static_folder=None)
app.config["ENV"] = "production" if IS_PRODUCTION else "development"
app.config["DEBUG"] = not IS_PRODUCTION

# Inicializar base de datos (SQLite local o PostgreSQL/Supabase en prod)
try:
    from models import init_db, db
    init_db(app)
    DB_AVAILABLE = True
except Exception as e:
    DB_AVAILABLE = False
    print(f"[db] Base de datos no disponible: {e}")

# Config
MAX_WORKERS = min(8, (os.cpu_count() or 4))
REQUEST_TIMEOUT = 30  # seconds for external API calls
MAX_IMAGE_DIMENSION = 4096  # max width/height for OCR processing
APP_VERSION = "20260715"

LANGUAGES = {
    "es": "spanish", "en": "english", "pt": "portuguese",
    "fr": "french",  "de": "german",  "it": "italian",
    "ja": "japanese","ko": "korean",  "zh": "chinese (simplified)",
    "zh-cn": "chinese (simplified)", "zh-tw": "chinese (traditional)",
    "auto": "auto",
}

# ─── Shared ThreadPoolExecutor (single instance, proper shutdown) ────────────────
_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="translator")
_executor_lock = threading.Lock()
_executor_shutdown = False

def _get_executor() -> ThreadPoolExecutor:
    global _executor, _executor_shutdown
    with _executor_lock:
        if _executor_shutdown:
            _executor = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="translator")
            _executor_shutdown = False
        return _executor

def shutdown_executor():
    global _executor, _executor_shutdown
    with _executor_lock:
        if not _executor_shutdown:
            _executor.shutdown(wait=True)
            _executor_shutdown = True

# ─── Security headers (CSP, Brave Leo opt-out) ───────────────────────────────────
CSP_POLICY = (
    "default-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com "
    "https://fonts.googleapis.com https://fonts.gstatic.com https://docs.opencv.org "
    "data: blob:; "
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://docs.opencv.org; "
    "worker-src 'self' blob: https://cdnjs.cloudflare.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self' http://127.0.0.1:5174 https://cdnjs.cloudflare.com https://cdn.jsdelivr.net data:;"
)

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses."""
    # CSP header to prevent browser AI from injecting scripts
    response.headers['Content-Security-Policy'] = CSP_POLICY
    # Opt out of browser AI processing
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    return response

# ─── EasyOCR (lazy load with multi-lang support + CPU fallback) ────────────
_ocr_readers = {}
_ocr_lock = threading.Lock()

def _get_ocr_reader(lang: str = "auto"):
    # Map source language to EasyOCR language set
    lang_key = "latin"
    if lang == "ja":
        lang_key = "ja"
    elif lang == "ko":
        lang_key = "ko"
    elif lang == "zh":
        lang_key = "zh"

    global _ocr_readers
    if lang_key in _ocr_readers:
        return _ocr_readers[lang_key]

    with _ocr_lock:
        if lang_key in _ocr_readers:
            return _ocr_readers[lang_key]

        langs = ["es", "en"]
        if lang_key == "ja":
            langs = ["ja", "en"]
        elif lang_key == "ko":
            langs = ["ko", "en"]
        elif lang_key == "zh":
            langs = ["ch_sim", "en"]
        else:
            langs = ["es", "en", "pt", "fr", "de"]

        print(f"[OCR] Cargando EasyOCR para {langs}...")
        try:
            import easyocr
            import torch
            gpu_available = torch.cuda.is_available()
            
            # Try GPU first, fallback to CPU on any error
            for use_gpu in ([True, False] if gpu_available else [False]):
                try:
                    print(f"[OCR] Intentando {'GPU' if use_gpu else 'CPU'} para {langs}...")
                    reader = easyocr.Reader(
                        langs,
                        gpu=use_gpu,
                        model_storage_directory=str(ROOT / "ocr_models"),
                        download_enabled=True,
                        verbose=False,
                    )
                    _ocr_readers[lang_key] = reader
                    print(f"[OCR] EasyOCR para {langs} listo en {'GPU' if use_gpu else 'CPU'}.")
                    break
                except Exception as e:
                    if use_gpu:
                        print(f"[OCR] GPU falló ({e}), reintentando en CPU...")
                    else:
                        print(f"[OCR] Error cargando EasyOCR para {langs}: {e}")
                        return None
        except Exception as e:
            print(f"[OCR] Error importando easyocr/torch: {e}")
            return None
    return _ocr_readers[lang_key]


# ─── Translation (ArgosTranslate + Google fallback) ────────────────────────────
_argo_ready: dict[tuple[str, str], bool] = {}
_argo_lock = threading.Lock()

# Thread-local langdetect detector for thread safety
_thread_local = threading.local()

def _get_langdetect_detector():
    """Get thread-local langdetect detector."""
    if not hasattr(_thread_local, 'detector'):
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0  # deterministic
        from langdetect import detect
        _thread_local.detector = detect
    return _thread_local.detector


def _ensure_argo_package(src: str, tgt: str) -> bool:
    key = (src, tgt)
    with _argo_lock:
        if _argo_ready.get(key):
            return True
        try:
            from argostranslate import package, translate
            installed_codes = {l.code for l in translate.get_installed_languages()}
            if src in installed_codes and tgt in installed_codes:
                _argo_ready[key] = True
                return True
            print(f"[offline] Descargando modelo {src}->{tgt}...")
            package.update_package_index()
            available = package.get_available_packages()
            pkgs = [p for p in available if p.from_code == src and p.to_code == tgt]
            if pkgs:
                pkgs[0].install()
                _argo_ready[key] = True
                print(f"[offline] Modelo {src}->{tgt} instalado.")
                return True
        except Exception as e:
            print(f"[offline] Error cargando {src}->{tgt}: {e}")
    return False


def _translate_argos(text: str, src: str, tgt: str) -> str | None:
    try:
        from argostranslate import translate
        installed = translate.get_installed_languages()
        src_lang = next((l for l in installed if l.code == src), None)
        tgt_lang = next((l for l in installed if l.code == tgt), None)
        if src_lang and tgt_lang:
            tr = src_lang.get_translation(tgt_lang)
            if tr:
                return tr.translate(text)
    except Exception as e:
        print(f"[offline] Error traduciendo: {e}")
    return None


def _translate_google(text: str, source: str, target: str) -> str | None:
    try:
        from deep_translator import GoogleTranslator
        import requests
        
        # Create a session with timeout
        session = requests.Session()
        original_request = session.request
        
        def request_with_timeout(*args, **kwargs):
            kwargs['timeout'] = kwargs.get('timeout', REQUEST_TIMEOUT)
            return original_request(*args, **kwargs)
        
        session.request = request_with_timeout
        
        translator = GoogleTranslator(source=source, target=target)
        translator._session = session
        result = translator.translate(text)
        if result:
            return result
    except Exception as e:
        print(f"[google] Error: {e}")
    return None


def _detect_language_simple(text: str) -> str:
    if any(0xac00 <= ord(c) <= 0xd7a3 for c in text):
        return "ko"
    if any((0x3040 <= ord(c) <= 0x30ff) or (0x4e00 <= ord(c) <= 0x9faf) for c in text):
        return "ja"
    if any(c in "áéíóúñüÁÉÍÓÚÑÜ¿¡" for c in text):
        return "es"
    text_lower = text.lower()
    spa_words = {
        "el", "la", "los", "las", "que", "en", "un", "una", "de", "con", "es", "para", "por", "si", "no",
        "y", "pero", "como", "cómo", "mas", "más", "bien", "todo", "todos", "esta", "este", "tus", "sus", "mi",
        "me", "se", "lo", "le", "te", "al", "del", "tú", "yo", "criar", "villano", "villanos", "correcto",
        "correctamente", "ayudan", "administrar", "subimos", "oficial", "visitas", "hacer", "hola",
        "gracias", "capitulo", "capítulo", "temporada",
    }
    words = {w.strip(".,¡!¿?()[]{}*\"'") for w in text_lower.split()}
    if spa_words.intersection(words):
        return "es"
    # Detectar español por sufijos verbales: infinitivos (-ar, -er, -ir)
    # y enclíticos (-me, -te, -se, -le, -nos, -os pegados al verbo)
    for w in words:
        w_clean = w.strip()
        if any(w_clean.endswith(suf) for suf in ("arme", "erme", "irme", "arte", "erte", "irte",
                                                   "arse", "erse", "irse", "arle", "erle", "irle",
                                                   "arnos", "ernos", "irnos", "arlos", "erlos", "irlos",
                                                   "ar", "er", "ir")):
            return "es" if len(w_clean) > 2 else None
    return "en"


def _detect_language_robust(text: str) -> str:
    text = text.strip()
    if not text:
        return "en"
    if any(0xac00 <= ord(c) <= 0xd7a3 for c in text):
        return "ko"
    if any((0x3040 <= ord(c) <= 0x30ff) or (0x4e00 <= ord(c) <= 0x9faf) for c in text):
        return "ja"

    # Heurística simple primero (detecta español por sufijos verbales)
    simple = _detect_language_simple(text)

    try:
        detect = _get_langdetect_detector()
        lang = detect(text)
        if lang in ["es", "en", "pt", "fr", "de", "it", "ja", "ko", "zh-cn", "zh-tw"]:
            # langdetect es poco fiable para textos cortos (< 3 palabras)
            # Si dice "en" pero la heurística detecta español, confiar en heurística
            if len(text.split()) < 4 and lang == "en" and simple == "es":
                print(f"[langdetect] '{text}' -> {lang}, sobrescrito a {simple} (heurística)")
                return simple
            return "zh" if "zh" in lang else lang
    except Exception:
        pass

    return simple


def _translate_one(text: str, source: str, target: str) -> str:
    text = text.strip()
    if not text:
        return text
    src_lang = source if source != "auto" else _detect_language_robust(text)
    print(f"[translate] src_lang={src_lang}, target={target}, text='{text[:50]}'")
    if src_lang == target:
        print(f"[translate] SKIP: mismo idioma")
        return text
    if _ensure_argo_package(src_lang, target):
        result = _translate_argos(text, src_lang, target)
        if result and result != text:
            print(f"[translate] Argos OK: '{result[:50]}'")
            return result
        print(f"[translate] Argos devolvió mismo texto o None")
    if src_lang != "en" and target == "es":
        if _ensure_argo_package(src_lang, "en") and _ensure_argo_package("en", "es"):
            en = _translate_argos(text, src_lang, "en")
            if en:
                es = _translate_argos(en, "en", "es")
                if es:
                    print(f"[translate] Argos pivot OK: '{es[:50]}'")
                    return es
    result = _translate_google(text, src_lang, target)
    if result and result != text:
        print(f"[translate] Google OK: '{result[:50]}'")
        return result
    print(f"[translate] Todos los métodos fallaron, devolviendo original")
    return text


def _preload_models():
    print("[servidor] Precargando modelos offline...")
    for src, tgt in [("en", "es"), ("es", "en"), ("ja", "en"), ("ko", "en"), ("zh", "en")]:
        try:
            ok = _ensure_argo_package(src, tgt)
            if ok:
                print(f"[servidor] OK Modelo {src}->{tgt} listo.")
        except Exception as e:
            print(f"[servidor] FALLO {src}->{tgt}: {e}")
    print("[servidor] Modelos de traducción listos.")


# No pre-cargar al inicio para evitar conflictos de hilos con CUDA
# threading.Thread(target=_preload_models, daemon=True).start()


# ─── Image processing: OCR + Inpainting ───────────────────────────────

def _base64_to_cv2(b64: str) -> np.ndarray:
    """Convierte imagen base64 a array OpenCV BGR."""
    # Eliminar prefijo data:image/...;base64,
    if "," in b64:
        b64 = b64.split(",", 1)[1]
    img_bytes = base64.b64decode(b64)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return img


def _cv2_to_base64(img: np.ndarray, fmt: str = ".png") -> str:
    """Convierte array OpenCV BGR a base64 PNG."""
    success, buf = cv2.imencode(fmt, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise ValueError("No se pudo codificar la imagen")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")


def _detect_and_ocr(img_bgr: np.ndarray, lang_hint: str = "auto") -> list[dict]:
    """
    Detecta texto en la imagen usando EasyOCR.
    Retorna lista de {x, y, w, h, text, confidence}.
    """
    reader = _get_ocr_reader(lang_hint)
    if reader is None:
        return []

    # EasyOCR trabaja mejor con RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        # Sensibilidad máxima para texto estilizado y fuentes artísticas extremas
        # (terror/suspenso, rasgado, irregular, glifos parciales)
        results = reader.readtext(
            img_rgb,
            detail=1,
            paragraph=False,
            min_size=8,
            text_threshold=0.25,  # Umbral muy bajo para glifos irregulares/rasgados
            low_text=0.18,
            link_threshold=0.3,
            canvas_size=max(img_bgr.shape[:2]),
            mag_ratio=2.0,  # Ampliación 2x para mejorar detección de bordes finos
        )
    except Exception as e:
        print(f"[OCR] Error en readtext: {e}")
        return []

    blocks = []
    for (bbox, text, conf) in results:
        text = text.strip()
        # Permitir confianza muy baja (0.12) para capturar tipografías extremadamente
        # estilizadas (terror, rasgado, bordes irregulares) que EasyOCR apenas detecta
        if not text or conf < 0.12:
            continue
        # bbox es [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        if w < 5 or h < 5:
            continue
        # Estimar tamaño de fuente
        font_size = max(8, int(h * 0.75))
        # Muestrear color de texto (píxeles del centro de la caja)
        cx, cy = x + w // 2, y + h // 2
        pad = max(2, h // 6)
        roi = img_bgr[max(0, cy-pad):cy+pad, max(0, cx-pad):cx+pad]
        text_color = "#000000"
        if roi.size > 0:
            mean_b = int(np.mean(roi[:,:,0]))
            mean_g = int(np.mean(roi[:,:,1]))
            mean_r = int(np.mean(roi[:,:,2]))
            brightness = mean_r * 0.299 + mean_g * 0.587 + mean_b * 0.114
            text_color = "#ffffff" if brightness > 128 else "#000000"

        blocks.append({
            "x": x, "y": y, "w": w, "h": h,
            "text": text,
            "confidence": float(conf),
            "fontSize": font_size,
            "textColor": text_color,
        })

    # Filtrado de metadatos de margen + fusión horizontal de bloques de la misma línea
    blocks = _group_and_merge_blocks(blocks, img_bgr.shape[0])
    return blocks


# Ruido que SOLO se descarta si el bloque está en el margen superior o inferior
# (5% de la altura de página): fecha/hora impresa por el navegador, numeración de página.
_MARGIN_NOISE_PATTERNS = [
    # Fechas con formato estricto: 13/7/26, 13.07.2026, 13-7-26
    re.compile(r'\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4}'),
    # Fechas con error de OCR: segundo separador leído como '1' (13/7126)
    # o perdido por completo (13/726, 13.726)
    re.compile(r'\d{1,2}[/.\-]\d{1,2}1?\d{2}\b'),
    # Horas con o sin puntos: 4:58 P.M. / 4:58 p.m. / 4:58 p. m. / 4:58 pm / 4:58
    re.compile(r'\d{1,2}:\d{2}\s*([ap]\.?\s?m\.?)?', re.IGNORECASE),
    # Numeración de página: 3/128, 3 / 128, "3 de 128", "Pág. 3"
    re.compile(r'^\d{1,4}\s*/\s*\d{1,4}$'),
    re.compile(r'\b\d{1,4}\s+de\s+\d{1,4}\b', re.IGNORECASE),
    re.compile(r'\bp[aá]g(?:ina)?\.?\s?\d{1,4}\b', re.IGNORECASE),
    # Frases recurrentes de título/capítulo que aparecen en cabeceras/pies
    re.compile(r'cap[ií]tulo', re.IGNORECASE),
    re.compile(r'c[oó]mo\s?criar', re.IGNORECASE),
    re.compile(r'how\s?to\s?raise', re.IGNORECASE),
]

# Ruido que se descarta en CUALQUIER parte de la página: sellos de grupos de escaneo.
_WATERMARK_PATTERNS = [
    re.compile(r'\b(olympus|scanlation|zonaolympus|scan[\s-]?group)\b', re.IGNORECASE),
    re.compile(r'zonaolympus[\s-]?com', re.IGNORECASE),
    re.compile(r'\b1\s*[\s-]?c\s*[\s-]?2\s*[\s-]?e\b', re.IGNORECASE),  # sello "1 C 2 E"
]


def _group_and_merge_blocks(blocks: list[dict], img_h: int | None = None) -> list[dict]:
    """
    Filtra ruido, URLs, metadatos de margen (fecha/hora/numeración) y marcas de
    agua de grupos de escaneo, y fusiona horizontalmente bloques de la misma línea.
    NO fusiona verticalmente para evitar mega-bloques que destruyen la página.

    img_h: alto de la imagen en píxeles, usado para restringir los patrones de
    margen al 5% superior/inferior de la página. Si no se pasa, ese filtro se omite.
    """
    if not blocks:
        return []

    margin_top = img_h * 0.05 if img_h else None
    margin_bottom = img_h * 0.95 if img_h else None

    # 1. Filtrar ruido, URLs, metadatos de margen y marcas de agua
    cleaned = []
    for b in blocks:
        w, h = b["w"], b["h"]
        text = b["text"].strip()
        text_len = len(text)
        cy = b["y"] + h / 2

        # Marcas de agua de grupos de escaneo: se descartan en cualquier parte de la página
        if any(p.search(text) for p in _WATERMARK_PATTERNS):
            print(f"[OCR] Filtrando marca de agua: '{text}'")
            continue

        # Metadatos impresos (fecha/hora/numeración de página): solo si están en el margen
        in_margin = margin_top is not None and (cy < margin_top or cy > margin_bottom)
        if in_margin and any(p.search(text) for p in _MARGIN_NOISE_PATTERNS):
            print(f"[OCR] Filtrando metadato de margen: '{text}' en y={b['y']}")
            continue

        # Filtrar URLs y dominios web (en cualquier parte de la página)
        if re.search(r'https?://|www\.|\.(com|net|org|xyz|io)\b', text, re.IGNORECASE):
            print(f"[OCR] Filtrando URL: '{text}'")
            continue

        # Filtrar bloques que son solo números (arte/decoración como "8", "12", etc.)
        if re.match(r'^[\d\s.,]+$', text) and text_len <= 4:
            print(f"[OCR] Filtrando número suelto: '{text}' en [{b['x']},{b['y']}]")
            continue

        # Filtrar bloques extremadamente altos y estrechos (columnas de arte)
        aspect = w / max(h, 1)
        if aspect < 0.4 and text_len <= 3:
            print(f"[OCR] Filtrando ruido estrecho: '{text}' aspect={aspect:.2f}")
            continue

        # Filtrar texto de un solo carácter (ruido de OCR en ilustraciones)
        if text_len <= 1:
            print(f"[OCR] Filtrando carácter suelto: '{text}'")
            continue

        # Filtrar detecciones de muy baja confianza que son puntuación/números
        # sueltos (ruido de fuentes estilizadas que el OCR apenas detecta).
        # Estos se cuelan con min_conf bajo y luego se fusionan con texto real.
        conf = b.get("confidence", 0)
        if conf < 0.20 and re.match(r'^[\d\s.,;:!?\'\"\-–—]+$', text):
            print(f"[OCR] Filtrando ruido baja confianza: '{text}' conf={conf:.2f}")
            continue

        cleaned.append(b)

    if not cleaned:
        return []

    # 2. Fusión Horizontal SOLAMENTE: Mismas líneas de texto
    sorted_b = sorted(cleaned, key=lambda b: (b["y"] + b["h"]/2, b["x"]))
    merged = []
    used = [False] * len(sorted_b)

    for i, b in enumerate(sorted_b):
        if used[i]:
            continue
        group = [b]
        used[i] = True
        for j, b2 in enumerate(sorted_b):
            if used[j] or i == j:
                continue
            cy1 = b["y"] + b["h"] / 2
            cy2 = b2["y"] + b2["h"] / 2
            max_h = max(b["h"], b2["h"])
            gap_x = b2["x"] - (b["x"] + b["w"])

            if abs(cy1 - cy2) < max_h * 0.45 and -b["w"] < gap_x < b["w"] * 1.2:
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

    return merged


def _is_inside_speech_bubble(img_bgr: np.ndarray, block: dict) -> bool:
    """
    Detecta si un bloque de texto está dentro de un globo de diálogo.
    Un globo se caracteriza por tener un fondo oscuro y relativamente uniforme
    en el perímetro del bloque (borde exterior). Esto permite preservar la
    forma del globo durante el inpainting en vez de destruirla con un rectángulo.
    Retorna True si el bloque parece estar en un globo de diálogo.
    """
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]
    edge = max(3, int(min(bw, bh) * 0.15))

    # Muestrear el perímetro exterior del bloque (4 franjas de borde)
    samples = []
    for region_coords in [
        (max(0, by), min(h, by + edge), max(0, bx), min(w, bx + bw)),        # top edge
        (max(0, by + bh - edge), min(h, by + bh), max(0, bx), min(w, bx + bw)),  # bottom edge
        (max(0, by), min(h, by + bh), max(0, bx), min(w, bx + edge)),        # left edge
        (max(0, by), min(h, by + bh), max(0, bx + bw - edge), min(w, bx + bw)),  # right edge
    ]:
        y1, y2, x1, x2 = region_coords
        if y2 > y1 and x2 > x1:
            roi = img_bgr[y1:y2, x1:x2]
            if roi.size > 0:
                mean_bgr = roi.reshape(-1, 3).mean(axis=0)
                samples.append(mean_bgr)

    if len(samples) < 3:
        return False

    # Calcular brillo promedio del perímetro
    avg_bgr = np.mean(samples, axis=0)
    b, g, r = avg_bgr
    brightness = r * 0.299 + g * 0.587 + b * 0.114

    # Globo de diálogo: fondo oscuro (brillo < 80) y uniforme entre bordes
    if brightness > 80:
        return False

    # Verificar uniformidad: std entre las 4 muestras de borde debe ser baja
    samples_arr = np.array(samples)
    std_per_channel = samples_arr.std(axis=0)
    max_std = std_per_channel.max()
    if max_std > 35:
        return False  # Demasiada variación entre bordes (no es un globo uniforme)

    return True


def _build_glyph_mask_for_bubble(img_bgr: np.ndarray, block: dict) -> np.ndarray:
    """
    Genera una máscara de solo-glifos para un bloque dentro de un globo de diálogo.
    Solo marca los píxeles que son texto (contraste alto con el fondo oscuro),
    preservando el globo mismo. Retorna una máscara del tamaño de la imagen completa.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

    # Muestrear el color de fondo del globo desde el perímetro
    edge = max(3, int(min(bw, bh) * 0.15))
    bg_pixels = []
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
        # Fallback: no se pudo muestrear, marcar rectángulo completo
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

    # Margen pequeño para incluir bordes de glifos
    pad_x = max(3, int(bw * 0.04))
    pad_y = max(3, int(bh * 0.06))
    rx1 = max(0, bx - pad_x)
    ry1 = max(0, by - pad_y)
    rx2 = min(w, bx + bw + pad_x)
    ry2 = min(h, by + bh + pad_y)

    # Para cada pixel en el área del bloque, marcar si contrasta con el fondo del globo
    roi = img_bgr[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return mask

    # Diferencia de color euclidiana entre cada pixel y el fondo del globo
    diff = roi.astype(np.float32) - bg_mean.astype(np.float32)
    color_dist = np.sqrt((diff ** 2).sum(axis=2))

    # Umbral: texto blanco sobre globo negro tiene alto contraste (>60)
    glyph_threshold = 60
    glyph_pixels = (color_dist > glyph_threshold).astype(np.uint8) * 255

    mask[ry1:ry2, rx1:rx2] = glyph_pixels

    # Dilatar levemente para cubrir bordes de glifos estilizados
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _build_inpaint_mask(img_bgr: np.ndarray, blocks: list[dict]) -> np.ndarray:
    """
    Genera una máscara binaria para el inpainting.
    Para bloques dentro de globos de diálogo (fondo oscuro uniforme): usa máscara
    de solo-glifos para preservar la forma del globo.
    Para texto flotante sobre arte: usa rectángulo completo con margen.
    """
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for block in blocks:
        bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

        if _is_inside_speech_bubble(img_bgr, block):
            print(f"[inpaint] Bloque en globo de diálogo detectado: '{block['text'][:30]}' → máscara solo-glifos")
            glyph_mask = _build_glyph_mask_for_bubble(img_bgr, block)
            # Combinar (OR) con la máscara acumulada
            mask = cv2.bitwise_or(mask, glyph_mask)
        else:
            # Texto flotante sobre arte: máscara rectangular tradicional
            pad_x = max(5, int(bw * 0.08))
            pad_y = max(5, int(bh * 0.12))
            x1 = max(0, bx - pad_x)
            y1 = max(0, by - pad_y)
            x2 = min(w, bx + bw + pad_x)
            y2 = min(h, by + bh + pad_y)
            mask[y1:y2, x1:x2] = 255

    # Dilatar levemente para suavizar los bordes de inpainting
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def _inpaint_image(img_bgr: np.ndarray, mask: np.ndarray, blocks: list[dict] | None = None) -> np.ndarray:
    """
    Aplica inpainting con OpenCV.
    Radio adaptativo según el tamaño de los bloques de texto.
    """
    if mask.max() == 0:
        return img_bgr.copy()

    # Radio adaptativo: usar el tamaño promedio de los bloques de texto si está disponible
    if blocks:
        # Calcular altura promedio de los bloques de texto
        avg_height = np.mean([b["h"] for b in blocks]) if blocks else 20
        # Radio proporcional a la altura del texto (texto grande necesita radio mayor)
        radius = int(np.clip(avg_height * 0.6, 5, 30))
    else:
        # Fallback: usar ratio de cobertura
        covered = np.sum(mask > 0)
        total = mask.size
        coverage_ratio = covered / total
        radius = int(np.clip(12 + coverage_ratio * 30, 10, 40))
    
    print(f"[inpaint] Radio: {radius}, bloques: {len(blocks) if blocks else 0}")

    # Usar Navier-Stokes (NS) para fondos con textura, TELEA para fondos simples
    # NS produce mejores resultados en tramas/texturas pero es más lento
    flags = cv2.INPAINT_NS  # Mejor para manga/cómics con tramas
    result = cv2.inpaint(img_bgr, mask, inpaintRadius=radius, flags=flags)
    return result


def _sample_bg_color(img_bgr: np.ndarray, block: dict) -> str:
    """Muestrea el color de fondo ALREDEDOR del bloque (no dentro).
    Para bloques dentro de globos de diálogo, muestrea el borde interior del
    propio globo (perímetro del bloque), no franjas externas que pueden caer
    fuera del globo hacia el arte circundante.
    """
    h, w = img_bgr.shape[:2]
    bx, by, bw, bh = block["x"], block["y"], block["w"], block["h"]

    if _is_inside_speech_bubble(img_bgr, block):
        # Dentro de un globo: muestrear el perímetro interior del bloque
        # (los bordes del bloque están dentro del globo, no en el arte exterior)
        edge = max(4, int(min(bw, bh) * 0.12))
        samples = []
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

        # Fallback si no hay suficientes muestras
        return "#000000"

    # Texto flotante sobre arte: método tradicional con franjas superior/inferior
    pad = max(5, int(min(bw, bh) * 0.2))
    samples = []
    # Franja superior
    y1, y2 = max(0, by - pad), max(0, by)
    if y2 > y1:
        region = img_bgr[y1:y2, max(0, bx):min(w, bx+bw)]
        if region.size > 0:
            samples.append(np.mean(region.reshape(-1, 3), axis=0))
    # Franja inferior
    y1, y2 = min(h, by+bh), min(h, by+bh+pad)
    if y2 > y1:
        region = img_bgr[y1:y2, max(0, bx):min(w, bx+bw)]
        if region.size > 0:
            samples.append(np.mean(region.reshape(-1, 3), axis=0))

    if not samples:
        return "#ffffff"
    mean_bgr = np.mean(samples, axis=0)
    b, g, r = int(mean_bgr[0]), int(mean_bgr[1]), int(mean_bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"


# ─── Flask Routes ─────────────────────────────────────────────────────────────

@app.get("/")
def index():
    if IS_PRODUCTION:
        return send_from_directory(DIST, "index.html")
    return send_from_directory(ROOT, "index.html")


@app.get("/<path:path>")
def static_files(path: str):
    if IS_PRODUCTION:
        # En produccion, servir primero desde dist/
        prod_target = (DIST / path).resolve()
        if str(prod_target).startswith(str(DIST)) and prod_target.exists():
            return send_from_directory(DIST, path)
    target = (ROOT / path).resolve()
    if not str(target).startswith(str(ROOT)) or not target.exists():
        return "Not found", 404
    return send_from_directory(ROOT, path)


@app.get("/api/health")
def health():
    ready = [f"{s}->{t}" for (s, t), v in _argo_ready.items() if v]
    return jsonify({
        "ok": True,
        "version": APP_VERSION,
        "mode": "production" if IS_PRODUCTION else "development",
        "mit_available": MIT_AVAILABLE,
        "db_available": DB_AVAILABLE,
        "offline_models": ready,
    })


@app.post("/api/translate")
def translate():
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()
    target = str(payload.get("target", "es")).strip() or "es"
    source = str(payload.get("source", "auto")).strip() or "auto"
    if not text:
        return jsonify({"translatedText": "", "engine": "none"})
    if target not in LANGUAGES:
        return jsonify({"error": f"Idioma no soportado: {target}"}), 400
    return jsonify({"translatedText": _translate_one(text, source, target), "engine": "auto"})


@app.post("/api/translate-batch")
def translate_batch():
    """Traduce una lista completa de textos en paralelo con hilos compartidos."""
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


@app.post("/api/process-page")
def process_page():
    """
    Endpoint principal de procesamiento de página.
    Recibe: {image: base64, target: lang, source: lang}
    Devuelve: {inpainted_image: base64, blocks: [{x,y,w,h,source,translated,fontSize,textColor,bgColor}]}
    """
    payload = request.get_json(silent=True) or {}
    b64_image = payload.get("image", "")
    target_lang = str(payload.get("target", "en")).strip() or "en"
    source_lang = str(payload.get("source", "auto")).strip() or "auto"

    if not b64_image:
        return jsonify({"error": "No se proporcionó imagen"}), 400

    try:
        # 1. Decodificar imagen
        img_bgr = _base64_to_cv2(b64_image)
        if img_bgr is None:
            return jsonify({"error": "No se pudo decodificar la imagen"}), 400

        print(f"[process-page] Imagen {img_bgr.shape[1]}x{img_bgr.shape[0]}, target={target_lang}")

        # 2. OCR + Inpainting: MIT pipeline o legacy EasyOCR+OpenCV
        mit_used = False
        if MIT_AVAILABLE:
            try:
                print(f"[process-page] Usando MIT pipeline (CTD + LaMa)...")
                mit_result = run_pipeline(img_bgr)
                if mit_result.get("error"):
                    print(f"[process-page] MIT falló: {mit_result['error']}, usando legacy...")
                else:
                    inpainted_b64 = mit_result["inpainted_image"]
                    blocks = mit_result.get("blocks", [])
                    mit_used = True
                    # Decodificar imagen inpainted para muestreo de color
                    inpainted = _base64_to_cv2(inpainted_b64)
                    print(f"[process-page] MIT: {len(blocks)} bloques detectados")
            except Exception as mit_err:
                print(f"[process-page] MIT excepción: {mit_err}, usando legacy...")

        if not mit_used:
            # Legacy: EasyOCR + OpenCV
            blocks = _detect_and_ocr(img_bgr, source_lang)
            print(f"[process-page] Legacy OCR: {len(blocks)} bloques")

        if not blocks:
            inpainted_b64 = _cv2_to_base64(img_bgr)
            return jsonify({"inpainted_image": inpainted_b64, "blocks": []})

        # Detección de idioma (común a MIT y legacy)
        detected_lang = source_lang
        if (source_lang == "auto" or source_lang not in LANGUAGES) and blocks:
            combined_text = " ".join([b.get("text", "") for b in blocks])
            detected_lang = _detect_language_robust(combined_text)
            print(f"[process-page] Idioma detectado: {detected_lang}")

        if not mit_used:
            # Legacy: inpainting con OpenCV
            mask = _build_inpaint_mask(img_bgr, blocks)
            inpainted = _inpaint_image(img_bgr, mask, blocks)
            inpainted_b64 = _cv2_to_base64(inpainted)
        else:
            # MIT: decodificar inpainted para muestreo de color
            inpainted = _base64_to_cv2(inpainted_b64)

        # 5. Traducir textos en batch
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

        # 6. Armar la respuesta con coordenadas y traducciones
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


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=5174, threads=8)