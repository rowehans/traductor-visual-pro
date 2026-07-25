"""
translator.py — Detección de idioma y traducción (Argos, Google, CT2).

Extraído de server.py. Depende de config.py para constantes.
"""

import concurrent.futures
import os
import re
import shutil
import threading
import time
from collections.abc import Callable
from typing import Any

from config import GLOSARIO_REGEX, GLOSARIO_POST, REQUEST_TIMEOUT, LANGUAGES, ROOT


# ─── Thread-local langdetect detector ────────────────────────────
_thread_local: threading.local = threading.local()


def _get_langdetect_detector() -> Callable[[str], str]:
    if not hasattr(_thread_local, 'detector'):
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        from langdetect import detect
        _thread_local.detector = detect  # type: ignore[attr-defined]
    return _thread_local.detector


# ─── ArgosTranslate package management ───────────────────────────
_argo_ready: dict[tuple[str, str], bool] = {}
_argo_lock: threading.Lock = threading.Lock()


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


# ─── Google Translator HTTP session (connection pooling) ─────────
_google_session: Any = None
_google_session_lock: threading.Lock = threading.Lock()
_google_translators: dict[tuple[str, str], Any] = {}  # cache por (source, target)
_google_translators_lock: threading.Lock = threading.Lock()

# ─── Google Translate rate limit detection ───────────────────────
# Cuando Google devuelve N textos seguidos sin cambios (el mismo input),
# asumimos rate limiting y activamos backoff exponencial.
# El backoff solo suprime Google; CT2 y Argos siguen funcionando.
_google_rate_limit_state: dict[str, Any] = {
    "consecutive_unchanged": 0,
    "backoff_until": 0.0,
    "current_backoff": 60.0,
}
_google_rate_limit_lock: threading.Lock = threading.Lock()
_RATE_LIMIT_THRESHOLD: int = 3       # N textos sin cambios → gatillar backoff
_MAX_BACKOFF: float = 600.0          # 10 min máximo


def _get_google_session() -> Any:
    global _google_session
    if _google_session is None:
        with _google_session_lock:
            if _google_session is None:
                import requests
                s = requests.Session()
                original_request = s.request

                def request_with_timeout(*args: Any, **kwargs: Any) -> Any:
                    kwargs['timeout'] = kwargs.get('timeout', REQUEST_TIMEOUT)
                    return original_request(*args, **kwargs)

                s.request = request_with_timeout  # type: ignore[method-assign]
                _google_session = s
    return _google_session


def _translate_google(text: str, source: str, target: str) -> str | None:
    """
    Traduce con Google Translate.
    Detecta rate limiting (N textos seguidos sin cambios) y activa
    backoff exponencial. Durante el backoff retorna None inmediatamente
    para que otros motores (CT2, Argos) tomen el control.
    """
    # ── Verificar backoff por rate limiting ──────────────────────
    with _google_rate_limit_lock:
        now = time.time()
        if now < _google_rate_limit_state["backoff_until"]:
            remaining = _google_rate_limit_state["backoff_until"] - now
            print(f"[google] Rate limit backoff {remaining:.0f}s restantes, saltando")
            return None
        # Si ya pasó el backoff pero no se reinició explícitamente,
        # el próximo intento normal resetea todo.

    try:
        from deep_translator import GoogleTranslator
        session = _get_google_session()
        
        # Cachear instancias de GoogleTranslator por par de idioma
        key = (source, target)
        if key not in _google_translators:
            with _google_translators_lock:
                if key not in _google_translators:
                    t = GoogleTranslator(source=source, target=target)
                    t._session = session
                    _google_translators[key] = t
        translator = _google_translators[key]
        
        result = translator.translate(text)
        if result:
            result_str = str(result)  # type: ignore[no-any-return]
            # ── Detectar rate limiting: texto sin cambios ──────
            text_clean = text.strip().lower()
            result_clean = result_str.strip().lower()
            if text_clean == result_clean:
                with _google_rate_limit_lock:
                    _google_rate_limit_state["consecutive_unchanged"] += 1
                    count = _google_rate_limit_state["consecutive_unchanged"]
                    if count >= _RATE_LIMIT_THRESHOLD:
                        backoff = _google_rate_limit_state["current_backoff"]
                        _google_rate_limit_state["backoff_until"] = time.time() + backoff
                        # Backoff exponencial: 60 → 120 → 240 → ... → 600s max
                        _google_rate_limit_state["current_backoff"] = min(
                            backoff * 2, _MAX_BACKOFF
                        )
                        _google_rate_limit_state["consecutive_unchanged"] = 0
                        print(f"[google] ¡Rate limit detectado! {count} textos sin cambios. "
                              f"Backoff {backoff:.0f}s")
                    else:
                        print(f"[google] Texto sin cambios ({count}/{_RATE_LIMIT_THRESHOLD})")
                return result_str  # Devuelve igual, _resultado_valido lo rechazará
            else:
                # Traducción exitosa: reiniciar contador
                with _google_rate_limit_lock:
                    if _google_rate_limit_state["consecutive_unchanged"] > 0:
                        print(f"[google] Traducción exitosa, contador reiniciado "
                              f"(tenía {_google_rate_limit_state['consecutive_unchanged']} unchanged)")
                    _google_rate_limit_state["consecutive_unchanged"] = 0
                    _google_rate_limit_state["current_backoff"] = 60.0  # Reset backoff base
                return result_str
    except Exception as e:
        # Los errores HTTP también pueden ser rate limiting
        e_str = str(e).lower()
        if "429" in e_str or "too many" in e_str or "rate limit" in e_str:
            with _google_rate_limit_lock:
                _google_rate_limit_state["consecutive_unchanged"] += 1
                count = _google_rate_limit_state["consecutive_unchanged"]
                if count >= _RATE_LIMIT_THRESHOLD:
                    backoff = _google_rate_limit_state["current_backoff"]
                    _google_rate_limit_state["backoff_until"] = time.time() + backoff
                    _google_rate_limit_state["current_backoff"] = min(backoff * 2, _MAX_BACKOFF)
                    _google_rate_limit_state["consecutive_unchanged"] = 0
                    print(f"[google] ¡Rate limit detectado por error 429! Backoff {backoff:.0f}s")
        print(f"[google] Error: {e}")
    return None


# ─── Detección de idioma ─────────────────────────────────────────
# Set de palabras españolas (constante global, no se recrea por llamada)
_SPA_WORDS: set[str] = {
    "el", "la", "los", "las", "que", "en", "un", "una", "de", "con", "es", "para", "por", "si",
    "y", "pero", "como", "cómo", "mas", "más", "bien", "todo", "todos", "esta", "este", "tus", "sus", "mi",
    "me", "se", "lo", "le", "te", "al", "del", "tú", "yo", "criar", "villano", "villanos", "correcto",
    "correctamente", "ayudan", "administrar", "subimos", "oficial", "visitas", "hacer", "hola",
    "gracias", "capitulo", "capítulo", "temporada",
}

# Sufijos verbales enclíticos españoles
_SPA_VERB_SUFFIXES: tuple[str, ...] = (
    "arme", "erme", "irme", "arte", "erte", "irte",
    "arse", "erse", "irse", "arle", "erle", "irle",
    "arnos", "ernos", "irnos", "arlos", "erlos", "irlos",
    "ar", "er", "ir",
)


def _detect_language_simple(text: str) -> str:
    if any(0xac00 <= ord(c) <= 0xd7a3 for c in text):
        return "ko"
    # ── CJK: kana (hiragana/katakana) → ja, hanzi sin kana → zh ──
    has_kana = any(0x3040 <= ord(c) <= 0x30ff for c in text)
    has_hanzi = any(0x4e00 <= ord(c) <= 0x9faf for c in text)
    if has_kana:
        return "ja"
    if has_hanzi:
        return "zh"
    if any(c in "áéíóúñüÁÉÍÓÚÑÜ¿¡" for c in text):
        return "es"
    text_lower = text.lower()
    words = {w.strip(".,¡!¿?()[]{}*\"'") for w in text_lower.split()}
    if _SPA_WORDS.intersection(words):
        return "es"
    for w in words:
        w_clean = w.strip()
        if any(w_clean.endswith(suf) for suf in _SPA_VERB_SUFFIXES):
            return "es" if len(w_clean) > 2 else "en"
    return "en"


def _detect_language_robust(text: str) -> str:
    text = text.strip()
    if not text:
        return "en"
    if any(0xac00 <= ord(c) <= 0xd7a3 for c in text):
        return "ko"
    # ── CJK: kana → ja; hanzi sin kana → langdetect desambigua ──
    has_kana = any(0x3040 <= ord(c) <= 0x30ff for c in text)
    has_hanzi = any(0x4e00 <= ord(c) <= 0x9faf for c in text)
    if has_kana:
        return "ja"
    if has_hanzi:
        # Hanzi sin kana: puede ser chino (你好) o japonés kanji-puro (日本語)
        # Si langdetect retorna algo inesperado (ej: "ko" para texto sin hangul),
        # ignorarlo y asumir chino (más probable en manga scan).
        try:
            detect = _get_langdetect_detector()
            lang = detect(text)
            if "zh" in lang:
                return "zh"
            if lang == "ja":
                return lang
        except Exception:
            pass
        return "zh"
    simple = _detect_language_simple(text)
    has_spanish_accents = any(c in "áéíóúñüÁÉÍÓÚÑÜ¿¡" for c in text)
    try:
        detect = _get_langdetect_detector()
        lang = detect(text)
        if lang in ["es", "en", "pt", "fr", "de", "it", "ja", "ko", "zh-cn", "zh-tw"]:
            if lang == "en" and simple == "es" and has_spanish_accents:
                print(f"[langdetect] '{text[:50]}' -> {lang}, sobrescrito a {simple} (acentos españoles)")
                return simple
            if len(text.split()) < 4 and lang == "en" and simple == "es":
                print(f"[langdetect] '{text[:50]}' -> {lang}, sobrescrito a {simple} (heurística corta)")
                return simple
            return "zh" if "zh" in lang else lang
    except Exception:
        pass
    return simple


# ─── Glosario (correcciones pre/post traducción) ─────────────────
def _aplicar_glosario(text: str) -> str:
    """Aplica correcciones del glosario usando patrones pre-compilados (GLOSARIO_REGEX)."""
    t = text
    for pattern, replacement in GLOSARIO_REGEX:
        t = pattern.sub(replacement, t)
    return t


# ─── Detección de ruido OCR (pre-Argos) ──────────────────────────
# Si el texto parece OCR ruidoso, nos saltamos Argos (que tarda ~3s
# y produce basura como "mainstremainstre") y vamos directo a Google.
_OCR_NOISE_DIGIT_PAT = re.compile(r'\d{2,}')
_OCR_NOISE_SPECIAL_PAT = re.compile(r'[@#$%^&*+=<>{}|\\/]{2,}')
_OCR_NOISE_REPEAT_PAT = re.compile(r'(.)\1{4,}')


def _es_ocr_noise(text: str) -> bool:
    """
    Detecta rápidamente si un texto parece ruido OCR.
    Retorna True si es probablemente basura y debería saltarse Argos.
    """
    t = text.strip()
    if not t or len(t) < 2:
        return False

    # 1. Alta proporción de dígitos (>30%)
    digits = sum(1 for c in t if c.isdigit())
    if digits > 0 and digits / max(len(t), 1) > 0.30:
        return True

    # 2. Patrón de dígitos agrupados (ej: "T2n2", "GEAn374")
    if _OCR_NOISE_DIGIT_PAT.search(t):
        # Verificar que no sea numérico puro (ej: "CAPITULO 43" no es ruido)
        if not any(c.isalpha() for c in t):
            return True
        # Mayoría no letras + dígitos mezclados
        alpha = sum(1 for c in t if c.isalpha())
        if alpha / max(len(t), 1) < 0.4:
            return True

    # 3. Caracteres especiales extraños
    if _OCR_NOISE_SPECIAL_PAT.search(t):
        return True

    # 4. Caracteres repetidos: en manga las repeticiones son estilísticas
    # ("NOOOOOO!", "GRRRRRR", "WHAAAAT") - NO clasificar como ruido.
    # Solo si es UNA sola letra repetida ("AAAAA") y no es vocal ni H.
    upper = t.upper()
    if _OCR_NOISE_REPEAT_PAT.search(upper):
        unique_letters = set(c for c in upper if c.isalpha())
        # Excluir vocales y H (comunes en exclamaciones de manga)
        manga_chars = {'A', 'E', 'I', 'O', 'U', 'H'}
        non_manga = unique_letters - manga_chars
        if len(non_manga) == 0 and len(unique_letters) <= 1 and len(t) >= 5:
            return True

    # 5. Símbolos OCR como Œ (ligadura) que aparecen en OCR de mala calidad
    weird_ocr_chars = sum(1 for c in t if ord(c) > 127 and c not in 'áéíóúñüÁÉÍÓÚÑÜ¿¡')
    if weird_ocr_chars >= 2:
        return True

    return False


# ─── Validación de traducción ────────────────────────────────────
# Patrón para detectar fragmentos repetidos pegados (ej: "mainstremainstre",
# "power powerpower", "hellohellhello") que Argos produce con OCR ruidoso.
# Detecta cualquier chunk de 4-20 chars que aparezca >=2 veces en el string,
# sin importar posicion ni longitud total.
_REPEATED_CHUNK_PAT: re.Pattern[str] = re.compile(
    r'(.{4,20})\1+'  # cualquier fragmento de 4-20 chars repetido al menos una vez
)


def _es_traduccion_valida(orig: str, trad: str, lenient: bool = False) -> bool:
    if not trad or not trad.strip():
        return False
    if not lenient and trad == orig:
        return False
    # Detectar fragmentos repetidos pegados (reemplaza el viejo hardcode "mainstremainstre")
    if _REPEATED_CHUNK_PAT.search(trad):
        return False
    if len(trad) > 20:
        unicos = len(set(trad.lower()))
        if unicos / max(len(trad), 1) < 0.15:
            return False
    if len(trad) > 20:
        for window in [10, 15, 20]:
            if len(trad) >= window * 3:
                sub = trad[:window]
                if trad.count(sub) > len(trad) // window * 0.7:
                    return False
    if orig and len(trad) > len(orig) * 8:
        return False
    return True


# ─── Corrección post-CT2 para traducciones literales ─────────────
# CT2 produce traducciones literales. Aplicamos un glosario específico
# para corregir términos comunes de manga (TEMPORADA → SEASON, etc.)
_GLOSARIO_POST_REGEX: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in GLOSARIO_POST
]


def _corregir_ct2(text: str) -> str:
    """Aplica correcciones post-traducción para CT2."""
    t = text
    for pattern, replacement in _GLOSARIO_POST_REGEX:
        t = pattern.sub(replacement, t)
    return t


# ─── CT2 (CTranslate2 OPUS-MT multi-par, ~3x más rápido que HF) ─
# Cada par de idiomas tiene su propio modelo Helsinki-NLP OPUS-MT
# convertido a CTranslate2 con cuantización int8.
# El modelo se descarga y convierte automáticamente la primera vez.

_CT2_MODELS: dict[str, str] = {
    # Español ↔ Inglés
    "es|en": "Helsinki-NLP/opus-mt-es-en",
    "en|es": "Helsinki-NLP/opus-mt-en-es",
    # Inglés ↔ Francés
    "en|fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr|en": "Helsinki-NLP/opus-mt-fr-en",
    # Inglés ↔ Alemán
    "en|de": "Helsinki-NLP/opus-mt-en-de",
    "de|en": "Helsinki-NLP/opus-mt-de-en",
    # Inglés ↔ Portugués (modelo tc-big, es bidireccional)
    "en|pt": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    "pt|en": "Helsinki-NLP/opus-mt-tc-big-en-pt",
    # Inglés ↔ Italiano
    "en|it": "Helsinki-NLP/opus-mt-en-it",
    "it|en": "Helsinki-NLP/opus-mt-it-en",
    # Japonés ↔ Inglés
    "ja|en": "Helsinki-NLP/opus-mt-ja-en",
    "en|ja": "Helsinki-NLP/opus-mt-en-ja",
    # Coreano ↔ Inglés
    "ko|en": "Helsinki-NLP/opus-mt-ko-en",
    "en|ko": "Helsinki-NLP/opus-mt-en-ko",
    # Chino ↔ Inglés
    "zh|en": "Helsinki-NLP/opus-mt-zh-en",
    "en|zh": "Helsinki-NLP/opus-mt-en-zh",
}

_CT2_BASE_DIR: str = str(ROOT / "models" / "ct2")

_ct2_translators: dict[str, Any] = {}
_ct2_tokenizers: dict[str, Any] = {}
_ct2_lock: threading.Lock = threading.Lock()


def _get_ct2_translator(source: str, target: str) -> tuple[Any, Any] | tuple[None, None]:
    """
    Obtiene (translator, tokenizer) para el par source|target.
    Descarga y convierte el modelo HF→CT2 automáticamente en la primera
    llamada para cada par. Usa archivo centinela para evitar reconversión
    si se interrumpe el proceso.
    """
    pair_key = f"{source}|{target}"
    model_name = _CT2_MODELS.get(pair_key)
    if model_name is None:
        return None, None

    # Fast path: ya cargado
    if pair_key in _ct2_translators and pair_key in _ct2_tokenizers:
        return _ct2_translators[pair_key], _ct2_tokenizers[pair_key]

    with _ct2_lock:
        # Double-check dentro del lock
        if pair_key in _ct2_translators and pair_key in _ct2_tokenizers:
            return _ct2_translators[pair_key], _ct2_tokenizers[pair_key]

        try:
            model_dir = os.path.join(_CT2_BASE_DIR, pair_key.replace("|", "-"))
            sentinel = os.path.join(model_dir, ".ct2_conversion_ok")

            # Convertir HF → CT2 si no existe o está incompleto
            if not os.path.exists(sentinel):
                os.makedirs(os.path.dirname(model_dir), exist_ok=True)
                if os.path.isdir(model_dir):
                    shutil.rmtree(model_dir, ignore_errors=True)
                print(f"[CT2] Convirtiendo {model_name} a formato CT2 con int8...")
                from ctranslate2.converters import TransformersConverter
                converter = TransformersConverter(model_name)
                converter.convert(model_dir, quantization="int8", force=False)
                with open(sentinel, "w") as f:
                    f.write("ok")
                print(f"[CT2] Conversión completada → {model_dir}")

            # Cargar modelo CT2 (GPU si CUDA disponible, CPU si no)
            import ctranslate2
            import torch
            ct2_device = "cuda" if torch.cuda.is_available() else "cpu"
            _ct2_translators[pair_key] = ctranslate2.Translator(model_dir, device=ct2_device)
            if ct2_device == "cuda":
                print(f"[CT2] Modelo {pair_key} cargado en GPU (CUDA, int8)")
            else:
                print(f"[CT2] Modelo {pair_key} cargado (CPU, int8)")

            # Cargar tokenizer (HuggingFace, compartido entre HF y CT2)
            from transformers import AutoTokenizer
            _ct2_tokenizers[pair_key] = AutoTokenizer.from_pretrained(model_name)
            print(f"[CT2] Tokenizer {pair_key} cargado")

            return _ct2_translators[pair_key], _ct2_tokenizers[pair_key]

        except Exception as e:
            print(f"[CT2] Error cargando modelo {pair_key}: {e}")
            return None, None


def _translate_ctranslate2(text: str, source: str, target: str) -> str | None:
    """
    Traduce con CTranslate2 usando el modelo Helsinki-NLP OPUS-MT
    correspondiente al par source|target. Retorna None si no hay modelo
    para ese par o si ocurre un error (el pipeline cae a Argos/Google).
    """
    translator, tokenizer = _get_ct2_translator(source, target)
    if translator is None or tokenizer is None:
        return None
    try:
        # Tokenizar: texto → IDs → strings (formato CT2)
        input_ids = tokenizer.encode(text)
        source_tokens = tokenizer.convert_ids_to_tokens(input_ids)

        # Traducir con CTranslate2 (beam search nativo)
        results = translator.translate_batch(
            [source_tokens],
            beam_size=4,
            max_decoding_length=100,
        )
        output_tokens = results[0].hypotheses[0]

        # Decodificar: tokens → IDs → texto
        output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        translation = tokenizer.decode(output_ids)
        # Aplicar correcciones post-traducción (para términos literales)
        translation = _corregir_ct2(translation)
        return translation
    except Exception as e:
        print(f"[CT2] Error traduciendo {source}->{target}: {e}")
        return None


# ─── Shared executor para motores de traducción en paralelo ─────
# Cada llamada a _translate_one prueba CT2, Argos y Google en paralelo.
# Antes se creaba un ThreadPoolExecutor NUEVO por cada llamada (619 bloques
# × 3 threads = 1857 creaciones). Ahora usamos un executor compartido que
# mantiene 4 threads vivos, eliminando el overhead de crear/destruir threads.
_translate_engine_executor: concurrent.futures.ThreadPoolExecutor | None = None
_translate_engine_executor_lock: threading.Lock = threading.Lock()


def _get_translate_engine_executor() -> concurrent.futures.ThreadPoolExecutor:
    global _translate_engine_executor
    if _translate_engine_executor is None:
        with _translate_engine_executor_lock:
            if _translate_engine_executor is None:
                _translate_engine_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=4,  # suficiente para CT2 + Argos + Google
                    thread_name_prefix="translate_engines",
                )
    return _translate_engine_executor


# ─── ArgosTranslate direct (con lock global — NO es thread-safe) ──
_argos_translate_lock: threading.Lock = threading.Lock()


def _translate_argos(text: str, source: str, target: str) -> str | None:
    try:
        from argostranslate import translate  # type: ignore[import-untyped]
    except Exception:
        return None
    if not _ensure_argo_package(source, target):
        return None
    # ArgosTranslate no es thread-safe: acceso concurrente causa
    # errores de file locking en Windows. Serializamos con lock global.
    with _argos_translate_lock:
        try:
            installed = translate.get_installed_languages()
            src_lang = next((l for l in installed if l.code == source), None)
            tgt_lang = next((l for l in installed if l.code == target), None)
            if src_lang and tgt_lang:
                translation = src_lang.get_translation(tgt_lang)
                if translation:
                    return translation.translate(text)  # type: ignore[no-any-return]
        except Exception as e:
            print(f"[argos] Error traduciendo: {e}")
    return None


# ─── Pipeline principal de traducción ────────────────────────────
def _translate_one(
    text: str,
    source: str,
    target: str,
    cache_get: Callable[[str, str, str], str | None] | None = None,
    cache_set: Callable[[str, str, str, str], None] | None = None,
    translation_cache_available: bool = False,
) -> str:
    text = text.strip()
    if not text:
        return text

    # Limpieza universal de símbolos ruidosos del OCR (@, #, $, etc.)
    text = re.sub(r'[@#$%^&*()+={}\[\]|:;<>/\\]', '', text).strip()
    if not text:
        return ""

    text_processed = _aplicar_glosario(text) if source == "es" or source == "auto" else text

    src_lang = source if source != "auto" else _detect_language_robust(text_processed)
    if src_lang == target:
        return text_processed

    if translation_cache_available and cache_get is not None:
        cached = cache_get(text_processed, src_lang, target)
        if cached is not None:
            print(f"[translate] Cache HIT: '{cached[:50]}'")
            return cached  # type: ignore[no-any-return]

    print(f"[translate] src_lang={src_lang}, target={target}, text='{text_processed[:50]}'")

    is_lenient = source != "auto" and len(text_processed.split()) <= 3

    # Normalización de Casing para textos en MAYÚSCULAS COMPLETAS (ej: "AHORA YUTIA", "PRIMERO SEOLLANG")
    # Los motores como Google/Argos tienden a ignorar palabras en mayúsculas pegadas a nombres propios.
    is_all_caps = text_processed.isupper() and any(c.isalpha() for c in text_processed) and len(text_processed) > 1
    query_text = text_processed.title() if is_all_caps else text_processed

    # Detectar ruido OCR: si el texto parece basura, nos saltamos Argos
    # (que tarda ~3s y produce cadenas "mainstremainstre" inútiles)
    if _es_ocr_noise(text_processed):
        print(f"[translate] OCR noise detectado, saltando Argos")
        translation_fns: list[tuple[str, Callable[[str, str, str], str | None]]] = [
            ("ctranslate2", _translate_ctranslate2),
            ("google", lambda t, s, tg: _translate_google(t, s, tg)),
        ]
    else:
        translation_fns: list[tuple[str, Callable[[str, str, str], str | None]]] = [
            ("ctranslate2", _translate_ctranslate2),
            ("argos", _translate_argos),
            ("google", lambda t, s, tg: _translate_google(t, s, tg)),
        ]

    # ── Probar motores en PARALELO, aceptar el primer resultado valido ──
    # Usa executor compartido (no crea/destruye 3 threads por llamada)

    def _probar_motor(method_name: str, fn: Callable) -> tuple[str, str | None]:
        try:
            res = fn(query_text, src_lang, target)
            if res and is_all_caps:
                res = res.upper()
            return method_name, res
        except Exception as e:
            print(f"[translate] {method_name} error: {e}")
            return method_name, None

    def _resultado_valido(method_name: str, resultado: str | None, src_lang: str) -> bool:
        """Verifica si un resultado de traduccion es aceptable."""
        if not resultado:
            return False
        if resultado == text_processed:
            # Si el texto no cambio y el idioma origen NO es el destino,
            # es una NO-traduccion (garbage in, garbage out).
            # Esto evita que OCR ruidoso como 'momms@' pase validacion
            # solo porque _detect_language_robust('momms@') retorna 'en'
            # por defecto.
            if src_lang != target:
                print(f"[translate] {method_name} mismo texto "
                      f"(src_lang={src_lang}, target={target}) — descartado")
                return False
            if not is_lenient:
                return False
            result_lang = _detect_language_robust(resultado)
            if result_lang != target:
                print(f"[translate] {method_name} mismo texto "
                      f"(detectado={result_lang}, target={target}) — descartado")
                return False
        if not _es_traduccion_valida(text_processed, resultado, lenient=is_lenient):
            print(f"[translate] {method_name} inválido: '{resultado[:50]}'")
            return False
        return True

    # Lanzar todos los motores en paralelo usando executor compartido
    executor = _get_translate_engine_executor()
    fut_map = {
        executor.submit(_probar_motor, method_name, fn): method_name
        for method_name, fn in translation_fns
    }
    mejor_resultado: str | None = None
    mejor_nombre: str | None = None

    for future in concurrent.futures.as_completed(fut_map, timeout=30):
        method_name, resultado = future.result()
        if _resultado_valido(method_name, resultado, src_lang):
            # Primer resultado valido: aceptarlo inmediatamente
            print(f"[translate] {method_name} OK: '{resultado[:50]}'")
            if translation_cache_available and cache_set is not None:
                try:
                    cache_set(text_processed, src_lang, target, resultado)
                except Exception:
                    pass
            return resultado  # type: ignore[no-any-return]
        # Guardar el mejor para fallback si ninguno es valido
        if resultado and mejor_resultado is None:
            mejor_resultado = resultado
            mejor_nombre = method_name

    # ── Fallback: si ningun motor dio resultado valido ────────────
    # Intentamos Google con backoff progresivo (5s, 15s, 30s) porque
    # suele ser rate limiting temporal, no un error real del motor.
    backoff_delays = [5, 15, 30]
    for attempt, delay in enumerate(backoff_delays):
        print(f"[translate] Google retry {attempt + 1}/{len(backoff_delays)} "
              f"(esperando {delay}s)...")
        time.sleep(delay)
        # Resetear backoff de rate limiting para forzar el intento
        with _google_rate_limit_lock:
            _google_rate_limit_state["backoff_until"] = 0.0
        try:
            retry_result = _translate_google(query_text, src_lang, target)
            if retry_result and is_all_caps:
                retry_result = retry_result.upper()
            if retry_result and _resultado_valido("google-retry", retry_result, src_lang):
                print(f"[translate] Google retry {attempt + 1} OK: '{retry_result[:50]}'")
                if translation_cache_available and cache_set is not None:
                    try:
                        cache_set(text_processed, src_lang, target, retry_result)
                    except Exception:
                        pass
                return retry_result  # type: ignore[no-any-return]
        except Exception as e:
            print(f"[translate] Google retry {attempt + 1} error: {e}")

    # Fallback final: lo mejor que tengamos o el original
    if mejor_resultado is not None and mejor_nombre is not None:
        print(f"[translate] Fallback final ({mejor_nombre}): '{mejor_resultado[:50]}'")
        if translation_cache_available and cache_set is not None:
            try:
                cache_set(text_processed, src_lang, target, mejor_resultado)
            except Exception:
                pass
        return mejor_resultado

    print(f"[translate] Todos los métodos fallaron (incluyendo retry), devolviendo original")
    return text_processed
