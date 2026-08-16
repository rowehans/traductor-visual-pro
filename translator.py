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
from functools import lru_cache
from typing import Any, cast

from config import (
    GLOSARIO_REGEX, GLOSARIO_POST, REQUEST_TIMEOUT, LANGUAGES, ROOT,
    ACTIVE_LANGUAGE_CODES, DISABLED_LANGUAGES,
    GPU_MIN_FREE_VRAM_MB, GPU_VRAM_BUDGET_MB,
    CT2_NEW_MODEL_MIN_FREE_VRAM_MB,
)
from runtime_diagnostics import gpu_budget_allows, gpu_memory_snapshot

# ─── Caché de HuggingFace dentro del proyecto ──────────────────
# Regla "no tocar C:": la caché HF (tokenizers OPUS-MT, modelos de
# transformers) vive en hf_cache/ del proyecto (igual que en server.py y
# en el daemon uocr_daemon.py). Este módulo es quien descarga los modelos,
# así que también define las variables aquí para cubrir usos directos
# (tests, tools/) que no pasan por server.py. setdefault respeta un
# HF_HOME ya definido por el entorno si existiera.
import os as _os
_os.environ.setdefault("HF_HOME", str(ROOT / "hf_cache"))
_os.environ.setdefault("TRANSFORMERS_CACHE", str(ROOT / "hf_cache" / "hub"))
_os.environ.setdefault("HF_HUB_CACHE", str(ROOT / "hf_cache" / "hub"))


# ─── Thread-local langdetect detector ────────────────────────────
_thread_local: threading.local = threading.local()


def _get_langdetect_detector() -> Callable[[str], str]:
    if not hasattr(_thread_local, 'detector'):
        from langdetect import DetectorFactory
        DetectorFactory.seed = 0
        from langdetect import detect
        _thread_local.detector = detect
    return cast(Callable[[str], str], _thread_local.detector)


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
            # update_package_index con timeout para evitar cuelgues de red
            import concurrent.futures as _cf
            with _cf.ThreadPoolExecutor(max_workers=1) as _update_exec:
                _update_future = _update_exec.submit(package.update_package_index)
                try:
                    _update_future.result(timeout=30)
                except _cf.TimeoutError:
                    print(f"[offline] Timeout (30s) descargando indice de paquetes Argos")
                    return False
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
_translators: dict[tuple[str, str], Any] = {}  # cache por (source, target)
# Lock único para Google (session + translators + rate limit)
# Antes eran 3 locks separados; consolidado para reducir overhead
# de adquisición en batch (cada worker competía por 3 locks).
_google_lock: threading.Lock = threading.Lock()

# ─── Google Translate rate limit detection ───────────────────────
# Cuando Google devuelve N textos seguidos sin cambios (el mismo input),
# asumimos rate limiting y activamos backoff exponencial.
# El backoff solo suprime Google; CT2 y Argos siguen funcionando.
_google_rate_limit_state: dict[str, Any] = {
    "consecutive_unchanged": 0,
    "backoff_until": 0.0,
    "current_backoff": 10.0,
}
_RATE_LIMIT_THRESHOLD: int = 3       # N textos sin cambios → gatillar backoff
_MAX_BACKOFF: float = 120.0          # 2 min máximo (antes 600s)


def _get_google_session() -> Any:
    global _google_session
    if _google_session is None:
        with _google_lock:
            if _google_session is None:
                import requests
                s = requests.Session()
                original_request = s.request

                def request_with_timeout(*args: Any, **kwargs: Any) -> Any:
                    kwargs['timeout'] = kwargs.get('timeout', REQUEST_TIMEOUT)
                    return original_request(*args, **kwargs)

                s.request = request_with_timeout
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
    with _google_lock:
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
        if key not in _translators:
            with _google_lock:
                if key not in _translators:
                    t = GoogleTranslator(source=source, target=target)
                    t._session = session
                    _translators[key] = t
        translator = _translators[key]
        
        result = translator.translate(text)
        if result:
            result_str = str(result)
            # ── Detectar rate limiting: texto sin cambios ──────
            text_clean = text.strip().lower()
            result_clean = result_str.strip().lower()
            if text_clean == result_clean:
                with _google_lock:
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
                with _google_lock:
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
            with _google_lock:
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
    "realmente", "increible", "increíble", "increiblemente", "nunca", "siempre",
    "sino", "tambien", "también", "ahora", "entonces", "cada", "aunque",
    "algo", "alguien", "nada", "nadie", "quizas", "quizás", "talvez", "tal vez",
    "mismo", "misma", "propio", "propia", "gran", "grande", "mejor", "peor",
    "otro", "otra", "otros", "otras", "poco", "poca", "unos", "unas",
    "bueno", "buena", "malo", "mala", "primero", "primera", "ultimo", "último",
    "solo", "sólo", "aun", "aún", "sobre", "bajo", "contra", "entre",
    "ante", "segun", "según", "mediante", "durante", "despues", "después",
    "antes", "luego", "pronto", "tarde", "temprano", "siempre",
    "nuestro", "nuestra", "vuestro", "vuestra", "aquel", "aquella",
    "esto", "eso", "aquello", "ese", "esa", "esos", "esas",
    "donde", "dónde", "cuando", "cuándo", "como", "cual", "cuál",
    "quien", "quién", "cuanto", "cuánto", "cuan", "cuán",
}

# Sufijos verbales enclíticos españoles
_SPA_VERB_SUFFIXES: tuple[str, ...] = (
    "arme", "erme", "irme", "arte", "erte", "irte",
    "arse", "erse", "irse", "arle", "erle", "irle",
    "arnos", "ernos", "irnos", "arlos", "erlos", "irlos",
    "ar", "er", "ir",
)

# Fallback lexical conservador para el portugués, que sigue activo. Se exigen
# dos marcadores, salvo palabras de una sola pieza muy distintivas, para no
# convertir nombres o préstamos comunes en otro idioma.
_SIMPLE_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "pt": frozenset({
        "eu", "amo", "estou", "voce", "você", "nao", "não",
        "obrigado", "obrigada", "quero", "posso", "vamos",
    }),
}
_SIMPLE_LANGUAGE_DISTINCTIVE: dict[str, frozenset[str]] = {
    "pt": frozenset({"obrigado", "obrigada", "voce", "você", "nao", "não"}),
}

# Marcadores conservadores para detectar *code-switching* dentro de un
# bloque. No son un glosario del usuario ni agregan modelos: solo se usan
# para evitar que ``source=auto`` devuelva intacta una frase que tiene dos o
# más palabras de otro idioma. Se omiten palabras muy ambiguas (por ejemplo
# ``la``, ``que`` o ``no``) para no disparar Google por falsos positivos.
_MIXED_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "es": frozenset({
        "el", "la", "los", "las", "es", "un", "una", "hola", "quiero",
        "puedo", "puedes", "vamos", "tengo", "tienes",
        "mi", "amigo", "amiga", "dice", "digo",
        "casa", "mundo", "nuevo", "nueva", "grande", "mañana", "gracias",
        "adiós", "ahora", "estás", "estoy",
        "porque", "también", "dónde", "cómo", "esto", "eso", "hacer",
    }),
    "en": frozenset({
        "the", "and", "this", "that", "you", "your", "with", "from",
        "what", "where", "why", "please", "want", "need", "go", "home",
        "here", "there", "okay", "sorry", "not", "dont", "don't", "i",
        "my", "says", "said", "are",
        "am", "love", "hello", "world", "friend", "stop", "come", "look",
        "run", "wait", "really", "house", "big", "is", "was",
    }),
    "pt": frozenset({
        "eu", "amo", "sou", "estou",
        "você", "voce", "não", "nao", "vocês", "voces", "obrigado",
        "obrigada", "quero", "posso", "vamos", "casa", "amanhã", "amanha",
    }),
}

# Palabras inglesas muy distintivas que pueden aparecer como un único cambio
# dentro de una frase activa. Se usan
# solo cuando el idioma dominante ya tiene evidencia fuerte, para no marcar
# nombres o préstamos aislados como páginas mixtas.
_SINGLE_ENGLISH_SWITCH_MARKERS: frozenset[str] = frozenset({
    "you", "your", "the", "hello", "my", "i",
})

# Marcadores de un solo token que son suficientemente distintivos para
# reconocer el cambio inverso (p. ej. ``I love gracias``). Se mantienen
# separados de los marcadores generales: préstamos ambiguos como ``amigo``,
# ``casa`` o ``non`` no deben activar Google por sí solos.
_SINGLE_DISTINCTIVE_SWITCH_MARKERS: dict[str, frozenset[str]] = {
    "es": frozenset({
        "hola", "gracias", "adiós", "adios", "mañana", "manana",
        "quiero", "puedo", "dónde", "donde", "cómo", "como",
    }),
    "pt": frozenset({
        "você", "voce", "vocês", "voces", "obrigado", "obrigada",
        "não", "nao", "amanhã", "amanha",
    }),
}

# Abreviaturas comunes sin vocales (no confundir con OCR garbage)
_SHORT_TEXT_ALLOWED: set[str] = {"dr", "mr", "sr", "st", "jr", "vs", "tv", "cd", "pc", "ok", "km", "wc"}

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
        if w_clean.endswith("mente") and len(w_clean) > 6:
            return "es"

    for language, markers in _SIMPLE_LANGUAGE_MARKERS.items():
        if len(words.intersection(markers)) >= 2:
            return language
    for language, markers in _SIMPLE_LANGUAGE_DISTINCTIVE.items():
        if words.intersection(markers):
            return language
    return "en"


def _detect_mixed_languages(
    text: str,
    dominant: str | None = None,
) -> tuple[str, ...]:
    """Devuelve idiomas actuales con evidencia fuerte dentro de un bloque.

    La detección es deliberadamente conservadora: una palabra aislada o un
    nombre propio no convierte el bloque en mixto. Para idiomas latinos se
    exigen al menos dos marcadores distintivos; para CJK el alfabeto aporta la
    evidencia principal. El idioma dominante se coloca primero para que el
    llamador pueda usarlo como clave estable de memoria.
    """
    text = str(text or "").strip()
    if not text:
        return ()

    detected_dominant = str(dominant or "").strip().lower()
    if not detected_dominant:
        detected_dominant = _detect_language_robust(text)
    if detected_dominant in {"zh-cn", "zh-tw"}:
        detected_dominant = "zh"
    if detected_dominant in DISABLED_LANGUAGES:
        # Un caller antiguo puede pasar explícitamente uno de los idiomas
        # pausados; no permitimos que vuelva a entrar en la fusión.
        detected_dominant = "en"

    found: set[str] = set()
    has_hangul = any(0xAC00 <= ord(char) <= 0xD7A3 for char in text)
    has_kana = any(0x3040 <= ord(char) <= 0x30FF for char in text)
    has_hanzi = any(0x4E00 <= ord(char) <= 0x9FFF for char in text)
    if has_hangul:
        found.add("ko")
    if has_kana:
        found.add("ja")
    elif has_hanzi:
        # Kanji compartido + kana sigue siendo japonés; solo el texto sin
        # kana se considera chino en esta capa conservadora.
        found.add("zh")

    latin_tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-zÀ-ÿ]+(?:['’][A-Za-zÀ-ÿ]+)?", text)
    ]
    token_set = set(latin_tokens)
    if latin_tokens:
        for language, markers in _MIXED_LANGUAGE_MARKERS.items():
            if len(token_set.intersection(markers)) >= 2:
                found.add(language)
        dominant_markers = _MIXED_LANGUAGE_MARKERS.get(detected_dominant, frozenset())
        if (
            detected_dominant != "en"
            and len(token_set.intersection(dominant_markers)) >= 2
            and token_set.intersection(_SINGLE_ENGLISH_SWITCH_MARKERS)
        ):
            found.add("en")
        if len(token_set.intersection(dominant_markers)) >= 2:
            for language, markers in _SINGLE_DISTINCTIVE_SWITCH_MARKERS.items():
                if language != detected_dominant and token_set.intersection(markers):
                    found.add(language)

    if detected_dominant in ACTIVE_LANGUAGE_CODES:
        found.add(detected_dominant)

    if len(found) <= 1:
        return (detected_dominant,) if detected_dominant else tuple(found)

    ordered: list[str] = []
    if detected_dominant in found:
        ordered.append(detected_dominant)
    ordered.extend(sorted(found - set(ordered)))
    return tuple(ordered)


@lru_cache(maxsize=4096)
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
        except Exception as e:
            print(f"[langdetect] Error detectando idioma para hanzi: {e}")
        return "zh"
    simple = _detect_language_simple(text)
    if simple == "pt":
        # En bloques cortos el detector estadistico puede devolver es/en por
        # falta de contexto; la evidencia lexical conservadora es mas estable.
        return simple
    has_spanish_accents = any(c in "áéíóúñüÁÉÍÓÚÑÜ¿¡" for c in text)
    try:
        detect = _get_langdetect_detector()
        lang = detect(text)
        if lang in DISABLED_LANGUAGES:
            print(f"[langdetect] '{text[:50]}' -> {lang}, idioma temporalmente desactivado; fallback={simple}")
            return simple if simple in ACTIVE_LANGUAGE_CODES else "en"
        if lang in ["es", "en", "pt", "ja", "ko", "zh-cn", "zh-tw"]:
            if lang == "en" and simple == "es" and has_spanish_accents:
                print(f"[langdetect] '{text[:50]}' -> {lang}, sobrescrito a {simple} (acentos españoles)")
                return simple
            if len(text.split()) < 4 and lang == "en" and simple == "es":
                print(f"[langdetect] '{text[:50]}' -> {lang}, sobrescrito a {simple} (heurística corta)")
                return simple
            if lang == "pt" and simple == "es":
                print(f"[langdetect] '{text[:50]}' -> {lang}, sobrescrito a {simple} (SPA_WORDS match)")
                return simple
            if "zh" in lang and simple == "es":
                print(f"[langdetect] '{text[:50]}' -> {lang}, sobrescrito a {simple} (sin CJK real, SPA_WORDS match)")
                return simple
            return "zh" if "zh" in lang else lang
    except Exception as e:
        print(f"[langdetect] Error en detección robusta: {e}")
    return simple# ─── Detección de SFX/Onomatopeyas (preservar sin traducir) ───────
# Patrones comunes de onomatopeyas y efectos de sonido en manga/cómic
_SFX_PATTERNS: list[re.Pattern[str]] = [
    # Repetidos: "BANG BANG", "CRASH CRASH"
    re.compile(r'^([A-Z]{2,})\s+\1+$', re.IGNORECASE),
    # SFX con números: "BOOM 1", "CRASH 2"
    re.compile(r'^[A-Z]{3,}\s+\d+$', re.IGNORECASE),
    # SFX clásico manga: solo mayúsculas, 3-8 chars
    re.compile(r'^[A-Z]{3,8}$'),
    # SFX con puntuación: "KABOOM!", "CRASH...", "SFX:"
    re.compile(r'^[A-Z]{3,}[!?.…:]+$', re.IGNORECASE),
    # Onomatopeyas japonesas comunes romanizadas
    re.compile(r'^(DON|DOOON|BAKU|BOKU|GARA|GORO|KARA|PACHIN|PAN|PAKU|PON|ZUDON|ZUBAN|GAKU|GYAA|HAA|HYUU|KYUU|MOKU|NYUU|PIKU|PUN|PURU|PYON|SHIN|SHUU|TON|TSU|UZU|WAKU|ZU|ZUN|ZUZU|BAAM|BOOM|CRASH|SLAM|THUD|WHAM|ZAP|ZAP)\d*[!?.]*$', re.IGNORECASE),
    # Texto en burbuja de pensamiento: *pensamiento*
    re.compile(r'^\*[^*]+\*$'),
]

# Palabras españolas comunes que NO deben clasificarse como SFX aunque
# estén en mayúsculas. El patrón ^[A-Z]{3,8}$ detectaría "PERO", "ELLA",
# "ESTA", etc. como SFX, lo que impediría su traducción.
_SFX_EXCLUDE: frozenset[str] = frozenset({
    # Conjunciones y preposiciones
    "pero", "como", "cómo", "mas", "más", "sin", "con", "que", "para",
    "por", "desde", "hasta", "entre", "sobre", "segun", "según",
    # Pronombres y determinantes
    "ella", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
    "aquel", "aquella", "todo", "toda", "todos", "todas", "otro", "otra",
    "nadie", "algo", "nada", "cada", "tanto", "tanta", "varios", "varias",
    # Verbos comunes
    "eres", "tiene", "tienen", "hacer", "poder", "deber", "saber", "querer",
    "debe", "puede", "sabe", "quiere", "hace", "dice", "tener", "estar",
    # Adverbios
    "nunca", "siempre", "menos", "cerca", "lejos", "luego", "antes",
    "despues", "después", "mismo", "misma", "aún", "aun", "tarde",
    "temprano", "pronto", "todavía", "también", "tampoco",
    # Otras palabras españolas frecuentes en manga
    "bien", "mal", "gran", "casi", "solo", "sólo", "fue", "era", "son",
    "has", "han", "sea", "sean", "fuera", "fuese", "contra", "mediante",
    "durante", "excepto", "salvo", "incluso", "además", "acerca",
    "capitulo", "capítulo", "temporada",
})

# El patrón histórico de CAPS también veía como SFX palabras normales de
# inglés y portugués. Reutilizar los marcadores activos reduce ese falso
# positivo sin mantener un segundo diccionario manual.
_SFX_FOREIGN_LANGUAGE_EXCLUDE: frozenset[str] = frozenset().union(
    *(_MIXED_LANGUAGE_MARKERS.values())
)
_SFX_DIALOGUE_INTERJECTIONS: frozenset[str] = frozenset({
    "ah", "ahh", "ay", "hey", "help", "no", "oh", "ouch", "ow",
    "please", "sorry", "stop", "wait", "what", "why", "yes",
})


def _es_sfx(text: str) -> bool:
    """Detecta si un texto es onomatopeya/SFX y debe preservarse sin traducir."""
    t = text.strip()
    if not t or len(t) > 25:
        return False
    # Si contiene cualquier palabra común (capítulo, temporada, cómo, etc.), NO es SFX
    words_lower = [w.lower() for w in re.findall(r'\b\w+\b', t)]
    if any(
        w in _SFX_EXCLUDE or w in _SFX_FOREIGN_LANGUAGE_EXCLUDE
        for w in words_lower
    ):
        return False
    # Las vocales repetidas suelen ser una exclamación de diálogo
    # (``NOOOO``, ``AAAAH``), no una onomatopeya. Colapsar solo repeticiones
    # consecutivas permite excluir esas variantes sin tocar ``GRRRR`` o
    # ``HAA``, que siguen siendo SFX reconocibles.
    collapsed_words = [re.sub(r"(.)\1+", r"\1", word) for word in words_lower]
    if any(word in _SFX_DIALOGUE_INTERJECTIONS for word in collapsed_words):
        return False
    # SFX CJK inequívocos: repeticiones de un mismo glifo (ゴゴゴ, 哈哈哈,
    # ㅋㅋㅋ). No clasificar cualquier palabra CJK corta, porque puede ser
    # diálogo o un nombre propio válido.
    cjk_chars = [
        char for char in t
        if (
            0x3040 <= ord(char) <= 0x30FF      # hiragana / katakana
            or 0x3130 <= ord(char) <= 0x318F   # jamo
            or 0xAC00 <= ord(char) <= 0xD7A3   # hangul
            or 0x4E00 <= ord(char) <= 0x9FFF   # hanzi / kanji
        )
    ]
    if len(cjk_chars) >= 2 and len(set(cjk_chars)) == 1:
        return True
    for pat in _SFX_PATTERNS:
        if pat.match(t):
            return True
    return False


_HONORIFICOS_JA: dict[str, str] = {
    "さん": "-san",
    "ちゃん": "-chan",
    "くん": "-kun",
    "様": "-sama",
    "さま": "-sama",
    "先輩": "-senpai",
    "先生": "-sensei",
}
_JA_NAME_HONORIFIC_RE = re.compile(
    r"^[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff々]+\s*(さん|ちゃん|くん|様|さま|先輩|先生)$"
)


_HONORIFICOS_KO: dict[str, str] = {
    "\uc528": "-ssi",       # 씨
    "\ub2d8": "-nim",       # 님
    "\uad70": "-gun",       # 군
    "\uc591": "-yang",      # 양
    "\uc120\ubc30": "-sunbae",  # 선배
    "\ud6c4\ubc30": "-hubae",   # 후배
}
_KO_NAME_HONORIFIC_RE = re.compile(
    r"^[\uac00-\ud7a3]{2,5}\s*(씨|님|군|양|선배|후배)$"
)


def _preservar_honorificos(
    original: str,
    translated: str,
    source_lang: str,
    target_lang: str,
) -> str:
    """Conserva honorÃ­ficos japoneses solo en nombres aislados.

    Se limita deliberadamente a una caja cuyo contenido completo es un
    nombre japonÃ©s + honorÃ­fico. En frases completas el motor puede haber
    elegido una traducciÃ³n natural ("seÃ±or Tanaka") y forzar ``-san`` serÃ­a
    peor. La regla es determinista y no requiere una lista de personajes.
    """
    normalized_source = str(source_lang or "").strip().lower()
    if normalized_source in {"ja", "ja-jp"}:
        match = _JA_NAME_HONORIFIC_RE.fullmatch(original.strip())
        marker = _HONORIFICOS_JA[match.group(1)] if match else ""
    elif normalized_source == "ko":
        match = _KO_NAME_HONORIFIC_RE.fullmatch(original.strip())
        marker = _HONORIFICOS_KO[match.group(1)] if match else ""
    else:
        return translated
    if not match or not translated.strip():
        return translated
    if marker.casefold() in translated.casefold():
        return translated
    if match.group(1) in {"さん", "様", "さま"} and re.search(
        r"\b(?:sr\.?|sra\.?|señor(?:a)?|mr\.?|mrs\.?)\b",
        translated,
        re.IGNORECASE,
    ):
        return translated
    return translated.rstrip() + marker


def _post_process_translation(text: str, source_lang: str, target_lang: str) -> str:
    """
    Post-procesa la traducción para manga/cómic:
    - Capitalización correcta de primera letra
    - Normaliza espacios múltiples
    """
    if not text:
        return text
    t = text.strip()
    if not t:
        return t
    # Normalizar espacios múltiples
    t = re.sub(r'\s{2,}', ' ', t)
    # Si es SFX, devolver tal cual
    if _es_sfx(t):
        return t
    # Capitalizar primera letra si es minúscula y hay más texto
    if t[0].islower() and len(t) > 1 and t[1:].lstrip():
        t = t[0].upper() + t[1:]
    return t


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

# Tras el intento rapido de CT2, esta heuristica tambien protege el fallback
# de red: conservar OCR evidentemente roto es mas seguro que inventar una
# traduccion. Los alfabetos CJK se excluyen en el gate del pipeline.


def _es_ocr_noise(text: str) -> bool:
    """
    Detecta rápidamente si un texto parece ruido OCR.
    Retorna True si es probablemente basura y debería saltarse Argos.
    """
    t = text.strip()
    if not t or len(t) < 2:
        return False

    # 0. Excluir ordinales ingleses con dígito: "4th", "3rd", "1st", "2nd"
    # Estos son texto válido, no ruido OCR, pero check 1 los detectaría como
    # ruido por tener >30% de dígitos (1/3 = 33%).
    if re.match(r'^\d+(?:st|nd|rd|th)$', t, re.IGNORECASE):
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

    # 6. Textos de 1 carácter que no sean vocales ni Y
    #    ("Q", "N", "Z" son fragmentos de OCR, no palabras reales)
    if len(t) == 1:
        first = t[0].lower()
        if first not in ("a", "e", "i", "o", "u", "y"):
            return True

    # 7. Textos de 2-3 caracteres sin vocales ni digitos (OCR fragments como "ze", "kc")
    #    Excluye ordinales ("4th", "3rd") y etiquetas ("10%") con digitos.
    if len(t) <= 3:
        t_lower = t.lower()
        has_vowel = any(c in "aeiouáéíóúü" for c in t_lower)
        has_digit = any(c.isdigit() for c in t_lower)
        if not has_vowel and not has_digit and t_lower not in _SHORT_TEXT_ALLOWED:
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
    # Rechazar texto idÃ©ntico aunque el motor cambie mayÃºsculas o espacios.
    # De lo contrario una respuesta "hello" para "Hello" pasa la validaciÃ³n
    # y el pipeline la presenta como traducciÃ³n vÃ¡lida.
    orig_norm = re.sub(r"\s+", " ", orig.strip()).casefold()
    trad_norm = re.sub(r"\s+", " ", trad.strip()).casefold()
    if orig_norm and trad_norm == orig_norm:
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


def _translation_is_likely_source_language(
    text: str,
    source: str,
    target: str,
) -> bool:
    """Detecta una salida que conserva claramente el idioma de origen.

    Es una barrera de calidad, no un detector obligatorio de destino. Solo
    rechaza evidencia fuerte: dos marcadores léxicos del origen (en idiomas
    latinos) o detección/alfabeto inequívoco CJK. Esto evita descartar nombres,
    préstamos y traducciones cortas que no tienen suficientes palabras para
    identificar el idioma.
    """
    source_norm = {"zh-cn": "zh", "zh-tw": "zh"}.get(
        str(source or "").strip().lower(), str(source or "").strip().lower())
    target_norm = {"zh-cn": "zh", "zh-tw": "zh"}.get(
        str(target or "").strip().lower(), str(target or "").strip().lower())
    if not text or not source_norm or source_norm == target_norm:
        return False

    if source_norm in {"ja", "ko", "zh"}:
        # Para CJK exigimos evidencia del alfabeto de origen. langdetect
        # suele clasificar nombres romanizados (p. ej. "Tanaka") como el
        # idioma CJK indicado por el contexto, y rechazarlos romperÃ­a la
        # conservaciÃ³n de nombres propios/honorÃ­ficos desde cache.
        if source_norm == "ko":
            script_chars = [char for char in text
                            if 0xac00 <= ord(char) <= 0xd7a3]
        elif source_norm == "zh":
            script_chars = [char for char in text
                            if 0x4e00 <= ord(char) <= 0x9fff]
        else:
            script_chars = [char for char in text
                            if 0x3040 <= ord(char) <= 0x30ff]
        # Una o dos grafÃ­as pueden ser un nombre propio preservado. Una frase
        # realmente no traducida suele estar dominada por el script de origen;
        # el criterio combinado conserva nombres sin dejar pasar pÃ¡rrafos CJK.
        if not script_chars:
            return False
        letter_count = sum(char.isalpha() for char in text)
        script_ratio = len(script_chars) / max(letter_count, 1)
        return len(script_chars) >= 3 or script_ratio >= 0.5

    markers = _MIXED_LANGUAGE_MARKERS.get(source_norm)
    if not markers:
        return False
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-zÀ-ÿ]+(?:['’][A-Za-zÀ-ÿ]+)?", text)
    }
    source_score = len(tokens.intersection(markers))
    if source_score < 2:
        return False
    target_score = len(tokens.intersection(
        _MIXED_LANGUAGE_MARKERS.get(target_norm, frozenset())))
    return source_score >= target_score + 1


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

# ─── SHA256 checksums de modelos CT2 (mitigación B615) ────────
# Después de la descarga + conversión, se guarda un checksum SHA256
# de todos los archivos del modelo. Antes de cargar, se verifica
# que los archivos no hayan sido modificados.
_CT2_CHECKSUMS_FILE: str = str(ROOT / "models" / "ct2_checksums.json")
# Usar HuggingFace con local_files_only después del primer intento —
# el modelo se descarga durante la conversión (TransformersConverter)
# y queda cacheado; el tokenizer subsiguiente debe cargarlo desde caché.
_CT2_TOKENIZER_LOCAL_ONLY: bool = True


import hashlib


def _compute_file_sha256(filepath: str) -> str:
    """
    Computa el checksum SHA256 de un archivo en bloques de 64KB
    para no cargar todo en memoria (modelos de hasta ~300MB).
    """
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            block = f.read(65536)  # 64KB
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _load_ct2_checksums() -> dict[str, dict[str, str]]:
    """Carga el archivo de checksums. Retorna dict vacio si no existe."""
    import json
    if not os.path.exists(_CT2_CHECKSUMS_FILE):
        return {}
    try:
        with open(_CT2_CHECKSUMS_FILE, "r", encoding="utf-8") as f:
            return cast(dict[str, dict[str, str]], json.load(f))
    except (json.JSONDecodeError, OSError):
        print(f"[CT2] Error leyendo checksums, ignorando")
        return {}


def _save_ct2_checksums(pair_key: str, model_dir: str) -> None:
    """
    Computa SHA256 de todos los archivos en model_dir y los guarda
    en el archivo de checksums. Se llama después de la conversión HF→CT2.
    """
    import json
    if not os.path.isdir(model_dir):
        return
    checksums = _load_ct2_checksums()
    file_checksums: dict[str, str] = {}
    for root, _dirs, files in os.walk(model_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            try:
                file_checksums[fname] = _compute_file_sha256(fpath)
            except (OSError, PermissionError) as e:
                print(f"[CT2] Error calculando SHA256 de {fname}: {e}")
    checksums[pair_key] = file_checksums
    os.makedirs(os.path.dirname(_CT2_CHECKSUMS_FILE), exist_ok=True)
    with open(_CT2_CHECKSUMS_FILE, "w", encoding="utf-8") as f:
        json.dump(checksums, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(f"[CT2] Checksums SHA256 guardados para {pair_key} ({len(file_checksums)} archivos)")


def _verify_ct2_checksums(pair_key: str, model_dir: str) -> bool:
    """
    Verifica que los archivos del modelo coincidan con los checksums
    almacenados. Retorna True si todo coincide o si no hay checksums
    previos (primera vez).
    """
    checksums = _load_ct2_checksums()
    expected = checksums.get(pair_key)
    if expected is None:
        # Primera vez: no hay checksums guardados, no podemos verificar.
        # Esto pasa solo en la primera carga post-conversión.
        # Guardamos checksums ahora para futuras verificaciones.
        print(f"[CT2] No hay checksums previos para {pair_key}, generando...")
        _save_ct2_checksums(pair_key, model_dir)
        return True
    for fname, expected_sha in expected.items():
        fpath = os.path.join(model_dir, fname)
        if not os.path.exists(fpath):
            print(f"[CT2] ¡ARCHIVO FALTANTE! {fname} — se esperaba pero no existe")
            return False
        try:
            actual_sha = _compute_file_sha256(fpath)
        except (OSError, PermissionError) as e:
            print(f"[CT2] ¡ERROR leyendo {fname} para verificación SHA256: {e}")
            return False
        if actual_sha != expected_sha:
            print(f"[CT2] ¡CHECKSUM MISMATCH! {fname}: esperado {expected_sha[:16]}..., "
                  f"actual {actual_sha[:16]}... — modelo manipulado o corrupto!")
            return False
    print(f"[CT2] Checksums SHA256 verificados para {pair_key} ({len(expected)} archivos)")
    return True


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
    # Japonés ↔ Inglés (el repo reverso real es opus-mt-en-jap,
    # opus-mt-en-ja NO existe en HuggingFace)
    "ja|en": "Helsinki-NLP/opus-mt-ja-en",
    "en|ja": "Helsinki-NLP/opus-mt-en-jap",
    # Coreano ↔ Inglés (el repo reverso real es tc-big-en-ko,
    # opus-mt-en-ko NO existe en HuggingFace)
    "ko|en": "Helsinki-NLP/opus-mt-ko-en",
    "en|ko": "Helsinki-NLP/opus-mt-tc-big-en-ko",
    # Chino ↔ Inglés
    "zh|en": "Helsinki-NLP/opus-mt-zh-en",
    "en|zh": "Helsinki-NLP/opus-mt-en-zh",
}

# Revisiones pinneadas para cada modelo (satisface bandit B615).
# La seguridad real viene de SHA256 checksums + local_files_only=True.
# Se usa "main" como fallback; en producción reemplazar con commit SHA.
_CT2_REVISIONS: dict[str, str] = {
    k: "main" for k in _CT2_MODELS
}

_CT2_BASE_DIR: str = str(ROOT / "models" / "ct2")

_ct2_translators: dict[str, Any] = {}
_ct2_tokenizers: dict[str, Any] = {}
_ct2_lock: threading.Lock = threading.Lock()


def _ct2_gpu_allowed(snapshot: dict[str, Any] | None = None) -> bool:
    """Decide si un nuevo modelo CT2 puede cargarse en la GPU.

    Los traductores CT2 ya cargados se conservan para no penalizar la
    latencia, pero los pares nuevos deben respetar el presupuesto observable
    de la GPU. En una GTX 1050 Ti compartida con EasyOCR/U-OCR, degradar solo
    ese par a CPU es preferible a provocar un OOM que tumbe todo el proceso.
    Si CUDA no está disponible o el diagnóstico no puede leerse, se mantiene
    el comportamiento anterior y CT2 decide su propio fallback.
    """
    try:
        state = snapshot if snapshot is not None else gpu_memory_snapshot()
        if not state.get("available"):
            return True
        allowed = gpu_budget_allows(
            state,
            required_free_mb=max(
                GPU_MIN_FREE_VRAM_MB,
                CT2_NEW_MODEL_MIN_FREE_VRAM_MB,
            ),
            budget_mb=GPU_VRAM_BUDGET_MB,
        )
        if not allowed:
            print(
                "[CT2] VRAM bajo presupuesto; el nuevo par se cargará en CPU "
                f"(snapshot={state})"
            )
        return allowed
    except Exception as exc:
        print(f"[CT2] Diagnóstico VRAM no disponible, se permite GPU: {exc}")
        return True


def _get_ct2_translator(source: str, target: str, force_cpu: bool = False) -> tuple[Any, Any] | tuple[None, None]:
    """
    Obtiene (translator, tokenizer) para el par source|target.
    Descarga y convierte el modelo HF→CT2 automáticamente en la primera
    llamada para cada par. Usa archivo centinela para evitar reconversión
    si se interrumpe el proceso.

    Args:
        force_cpu: Si True, fuerza CT2 a usar CPU aunque CUDA esté disponible.
                   Útil cuando EasyOCR ya tomó la GPU para evitar conflicto cuDNN.
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
                # Después de la conversión, guardar checksums SHA256 de los archivos
                _save_ct2_checksums(pair_key, model_dir)

            # Verificar integridad SHA256 del modelo antes de cargar
            if not _verify_ct2_checksums(pair_key, model_dir):
                print(f"[CT2] ¡CHECKSUM FAIL! El modelo {pair_key} no pasó la verificación "
                      f"de integridad. Rechazando carga.")
                return None, None

            # Cargar modelo CT2 (GPU si CUDA disponible y force_cpu=False, CPU si no)
            import ctranslate2
            import torch
            use_gpu = (
                not force_cpu
                and bool(torch.cuda.is_available())
                and _ct2_gpu_allowed()
            )
            ct2_device = "cuda" if use_gpu else "cpu"
            _ct2_translators[pair_key] = ctranslate2.Translator(model_dir, device=ct2_device)
            if ct2_device == "cuda":
                print(f"[CT2] Modelo {pair_key} cargado en GPU (CUDA, int8)")
            else:
                print(f"[CT2] Modelo {pair_key} cargado (CPU, int8)")

            # Cargar tokenizer (HuggingFace, compartido entre HF y CT2)
            # Usar local_files_only=True para NO descargar nada de internet.
            # El tokenizer se cachea durante la conversión (TransformersConverter).
            from transformers import AutoTokenizer
            local_files_only = _CT2_TOKENIZER_LOCAL_ONLY
            ct2_revision = _CT2_REVISIONS.get(pair_key, "main")
            _ct2_tokenizers[pair_key] = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=local_files_only,
                revision=ct2_revision,  # B615 requiere revision explícita (no via **kwargs)
            )
            print(f"[CT2] Tokenizer {pair_key} cargado "
                  f"(local_files_only={local_files_only}, revision={ct2_revision})")

            return _ct2_translators[pair_key], _ct2_tokenizers[pair_key]

        except Exception as e:
            print(f"[CT2] Error cargando modelo {pair_key}: {e}")
            return None, None


def _translate_ctranslate2_batch(
    texts: list[str],
    source: str,
    target: str,
) -> list[str | None]:
    """
    Traduce una lista de textos con CTranslate2 en UNA sola translate_batch
    (optimización 2.6). El prefill del modelo se comparte entre los ítems del
    batch, así que N textos cuestan mucho menos que N llamadas individuales.

    Usa greedy (beam_size=1): en bloques cortos de manga el beam search de
    4 rinde poco y cuesta 2-4x más. Retorna una lista del mismo largo, con
    None por cada ítem que falló (el pipeline cae a Google/Argos para ese
    ítem individualmente). Retorna lista vacía si no hay modelo para el par.
    """
    clean_texts = [t for t in texts if t.strip()]
    if not clean_texts:
        return [None] * len(texts)
    translator, tokenizer = _get_ct2_translator(source, target)
    if translator is None or tokenizer is None:
        return [None] * len(texts)
    try:
        # Tokenizar todos los textos ANTES de llamar al batch (una sola
        # llamada a translate_batch para toda la página).
        batch_input: list[list[str]] = []
        for text in clean_texts:
            input_ids = tokenizer.encode(text)
            batch_input.append(
                tokenizer.convert_ids_to_tokens(input_ids))

        results = translator.translate_batch(
            batch_input,
            beam_size=1,
            max_decoding_length=100,
        )
        out_by_index: dict[int, str | None] = {}
        for i, result in enumerate(results):
            try:
                output_tokens = result.hypotheses[0]
                output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
                translation = tokenizer.decode(output_ids)
                # Aplicar correcciones post-traducción (términos literales)
                translation = _corregir_ct2(translation)
                out_by_index[i] = translation
            except Exception as inner_e:
                print(f"[CT2] Error decodificando ítem {i}: {inner_e}")
                out_by_index[i] = None
        # Re-mapear al largo original (los textos vacíos quedan None)
        mapped: list[str | None] = []
        iter_idx = 0
        for text in texts:
            if text.strip():
                mapped.append(out_by_index.get(iter_idx))
                iter_idx += 1
            else:
                mapped.append(None)
        return mapped
    except Exception as e:
        print(f"[CT2] Error traduciendo batch {source}->{target}: {e}")
        return [None] * len(texts)


def _translate_ctranslate2(text: str, source: str, target: str) -> str | None:
    """
    Traduce con CTranslate2 usando el modelo Helsinki-NLP OPUS-MT
    correspondiente al par source|target. Retorna None si no hay modelo
    para ese par o si ocurre un error (el pipeline cae a Argos/Google).
    """
    results = _translate_ctranslate2_batch([text], source, target)
    return results[0] if results else None


# ─── ArgosTranslate direct (con lock global — NO es thread-safe) ──
# NOTA: Ya no se usa en el pipeline principal de _translate_one().
# Se mantiene la funcion por si se necesita como fallback adicional.
_argos_translate_lock: threading.Lock = threading.Lock()


def _translate_argos(text: str, source: str, target: str) -> str | None:
    try:
        from argostranslate import translate
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
    block_type: str | None = None,
) -> str:
    text = text.strip()
    if not text:
        return text

    source_norm = str(source or "").strip().lower()
    target_norm = str(target or "").strip().lower()
    if source_norm in DISABLED_LANGUAGES or target_norm in DISABLED_LANGUAGES:
        print(
            f"[translate] idioma temporalmente desactivado; se conserva el texto "
            f"({source_norm}->{target_norm})"
        )
        return text

    semantic_type = str(block_type or "").strip().lower()
    non_sfx_text_types = {
        "dialogue", "speech", "thought", "title", "header",
        "caption", "narration", "narrative", "cartel",
    }
    # Clasificar antes de limpiar símbolos conserva marcadores de SFX como
    # *sigh*/*thinking* y su casing/puntuación original. Los bloques con tipo
    # semántico de texto siguen entrando al traductor aunque estén en CAPS.
    if _es_sfx(text) and semantic_type not in non_sfx_text_types:
        print(f"[translate] SFX detectado, preservando: '{text}'")
        return text

    # Limpieza universal de símbolos ruidosos del OCR (@, #, $, etc.)
    text = re.sub(r'[@#$%^&*()+={}\[\]|:;<>/\\]', '', text).strip()
    if not text:
        return ""

    # ── SFX/Onomatopeyas: detectar y preservar sin traducir ──
    # Sin tipo semántico mantenemos la heurística histórica. Si el detector
    # identifica diálogo/narración/título, las mayúsculas por sí solas no
    # deben bloquear la traducción (p. ej. «NARUTO, ¡ESPERA!»).
    if _es_sfx(text) and semantic_type not in non_sfx_text_types:
        print(f"[translate] SFX detectado, preservando: '{text}'")
        return text

    text_processed = _aplicar_glosario(text) if source == "es" or source == "auto" else text

    src_lang = source if source != "auto" else _detect_language_robust(text_processed)
    # También se calcula con source explícito: el usuario puede indicar
    # ``es`` y aun así tener una frase inglesa incrustada en el globo.
    mixed_languages = _detect_mixed_languages(
        text_processed, dominant=src_lang)
    # Los bloques mixtos también deben consultar cache antes de activar el
    # fallback de red. Se mantiene el mismo gate de idioma origen y basura
    # que usa el cache normal; si la entrada es inválida, el flujo continúa.
    if (
        src_lang != target
        and translation_cache_available
        and cache_get is not None
    ):
        cached = cache_get(text_processed, src_lang, target)
        if cached is not None:
            cached_text = str(cached)
            if (
                not _translation_is_likely_source_language(
                    cached_text, src_lang, target)
                and _es_traduccion_valida(
                    text_processed,
                    cached_text,
                    lenient=len(text_processed.split()) <= 3,
                )
            ):
                print(f"[translate] Cache HIT: '{cached_text[:50]}'")
                return _preservar_honorificos(
                    text_processed, cached_text, src_lang, target)
            print("[translate] Cache rechazado por validación; se continúa")
    if src_lang != target and len(mixed_languages) > 1:
        # Un bloque con cambio de idioma no debe entrar directamente en un
        # modelo monolingüe (p. ej. es->en) porque puede deformar la parte ya
        # escrita en el idioma destino. Google con source=auto es el único
        # motor existente capaz de resolver el bloque completo sin dividirlo,
        # incluso cuando el usuario fijó source explícitamente.
        mixed_result = _translate_google(text_processed, "auto", target)
        if (
            mixed_result
            and mixed_result.strip().casefold() != text_processed.casefold()
            and not _translation_is_likely_source_language(
                mixed_result, src_lang, target)
            and _es_traduccion_valida(text_processed, mixed_result, lenient=True)
        ):
            final_result = _post_process_translation(
                mixed_result, src_lang, target)
            final_result = _preservar_honorificos(
                text_processed, final_result, src_lang, target)
            print(
                f"[translate] mixed auto->{target} OK: "
                f"'{final_result[:50]}'"
            )
            if translation_cache_available and cache_set is not None:
                try:
                    cache_set(text_processed, src_lang, target, final_result)
                except Exception as e:
                    print(f"[translate] Error guardando en cache: {e}")
            return final_result
        print(
            f"[translate] mezcla auto sin traducción segura; "
            f"se continúa con fallback dominante: idiomas={mixed_languages}"
        )
    if src_lang == target:
        if len(mixed_languages) > 1:
            # Si el idioma dominante ya es el destino, el retorno temprano
            # histórico dejaba intacta una frase incrustada (p. ej.
            # "No puedo go home"). Google con source=auto es el único motor
            # existente que puede traducir el bloque mixto completo sin
            # dividirlo artificialmente y romper su gramática. Si no está
            # disponible, conservar el original es más seguro que forzar un
            # modelo con el idioma equivocado.
            mixed_result = _translate_google(text_processed, "auto", target)
            if (
                mixed_result
                and mixed_result.strip().casefold() != text_processed.casefold()
                and _es_traduccion_valida(text_processed, mixed_result, lenient=True)
            ):
                final_result = _post_process_translation(
                    mixed_result, src_lang, target)
                final_result = _preservar_honorificos(
                    text_processed, final_result, src_lang, target)
                print(
                    f"[translate] mixed auto->{target} OK: "
                    f"'{final_result[:50]}'"
                )
                if translation_cache_available and cache_set is not None:
                    try:
                        cache_set(text_processed, src_lang, target, final_result)
                    except Exception as e:
                        print(f"[translate] Error guardando en cache: {e}")
                return final_result
            print(
                f"[translate] mezcla detectada sin traducción segura; "
                f"se conserva el texto: idiomas={mixed_languages}"
            )
        return text_processed

    print(f"[translate] src_lang={src_lang}, target={target}, text='{text_processed[:50]}'")

    is_lenient = len(text_processed.split()) <= 3

    # Normalización de Casing para textos en MAYÚSCULAS COMPLETAS (ej: "AHORA YUTIA", "PRIMERO SEOLLANG")
    # Los motores como Google/Argos tienden a ignorar palabras en mayúsculas pegadas a nombres propios.
    is_all_caps = text_processed.isupper() and any(c.isalpha() for c in text_processed) and len(text_processed) > 1
    query_text = text_processed.title() if is_all_caps else text_processed

    # ── ESTRATEGIA OPTIMIZADA: CT2 primero (síncrono), Google fallback ──
    # Ya no se usa el pipeline paralelo (CT2+Argos+Google con timeout 30s)
    # que causaba contienda de workers y colgaba por 30-80s por texto.
    # Ahora es secuencial: CT2 (~0.12s GPU) -> Google (~2s) -> SIN_TRAD.

    def _resultado_valido(method_name: str, resultado: str | None, src_lang: str) -> bool:
        """Verifica si un resultado de traduccion es aceptable."""
        if not resultado:
            return False
        if _translation_is_likely_source_language(
            resultado, src_lang, target):
            print(
                f"[translate] {method_name} conserva idioma origen "
                f"({src_lang}->{target}); descartado"
            )
            return False
        if resultado == text_processed:
            if src_lang != target:
                words_lower = [w.lower() for w in re.findall(r'\b\w+\b', text_processed)]
                has_src_words = any(w in _SPA_WORDS for w in words_lower)
                if has_src_words:
                    print(f"[translate] {method_name} texto sin traducir (src={src_lang}, target={target}) — descartado")
                    return False
                print(f"[translate] {method_name} mismo texto (sin traducir) "
                      f"src_lang={src_lang}, target={target} — descartado")
                return False
            result_lang = _detect_language_robust(resultado)
            if result_lang != target:
                print(f"[translate] {method_name} mismo texto "
                      f"(detectado={result_lang}, target={target}) — descartado")
                return False
        if not _es_traduccion_valida(text_processed, resultado, lenient=is_lenient):
            print(f"[translate] {method_name} invalido: '{resultado[:50]}'")
            return False
        return True

    # ── ESTRATEGIA OPTIMIZADA: CT2 primero (síncrono, rápido) ────
    # El cache comparte el mismo contrato de calidad que los motores. Una
    # entrada antigua puede haber sido guardada antes de los gates actuales o
    # por una respuesta parcial de red; no debe saltarse la validación de
    # idioma origen ni la comprobación anti-basura.
    if translation_cache_available and cache_get is not None:
        cached = cache_get(text_processed, src_lang, target)
        if cached is not None:
            cached_text = str(cached)
            if _resultado_valido("cache", cached_text, src_lang):
                print(f"[translate] Cache HIT: '{cached_text[:50]}'")
                return _preservar_honorificos(
                    text_processed, cached_text, src_lang, target)
            print("[translate] Cache rechazado por validación; se reintenta")

    # CT2 es el motor mas rapido (~0.12s en GPU) y no necesita executor.
    # Probarlo primero evita la contienda de workers con Google/Argos
    # y reduce la latencia de ~30s a <1s en el caso comun.
    ct2_result = _translate_ctranslate2(query_text, src_lang, target)
    if ct2_result is not None and is_all_caps:
        ct2_result = ct2_result.upper()
    if ct2_result and _resultado_valido("ctranslate2", ct2_result, src_lang):
        final_result = _post_process_translation(ct2_result, src_lang, target)
        final_result = _preservar_honorificos(
            text_processed, final_result, src_lang, target)
        print(f"[translate] ctranslate2 OK (fast path): '{final_result[:50]}'")
        if translation_cache_available and cache_set is not None:
            try:
                cache_set(text_processed, src_lang, target, final_result)
            except Exception as e:
                print(f"[translate] Error guardando en cache: {e}")
        return final_result

    # ── Fallback: Google (con timeout rapido) ────────────────────
    # Google es el unico fallback realista. Argos se omite porque:
    #   1. Produce basura con texto OCR ruidoso ("mainstremainstre")
    #   2. Tarda ~3s en cargar el modelo la primera vez
    #   3. Requiere descarga de internet
    # Google retorna None si esta en backoff de rate limiting.
    # CT2 ya tuvo la oportunidad de recuperar el texto. Si tampoco entrega
    # una salida valida, saltar Google para ruido evidente evita falsos
    # positivos y una peticion de red innecesaria por cada fragmento OCR.
    # No aplicar este gate a CJK: la heuristica historica considera sus
    # glifos no latinos "weird" y podria impedir el pivote ja/ko/zh -> en -> es.
    is_cjk_source = src_lang in {"ja", "ko", "zh", "zh-cn", "zh-tw"}
    if not is_cjk_source and _es_ocr_noise(text_processed):
        print(
            f"[translate] Ruido OCR sin traduccion segura; "
            f"se conserva: '{text_processed[:50]}'"
        )
        return text_processed

    google_result = _translate_google(query_text, src_lang, target)
    if google_result and is_all_caps:
        google_result = google_result.upper()
    if google_result and _resultado_valido("google", google_result, src_lang):
        final_result = _post_process_translation(google_result, src_lang, target)
        final_result = _preservar_honorificos(
            text_processed, final_result, src_lang, target)
        print(f"[translate] google OK (fallback): '{final_result[:50]}'")
        if translation_cache_available and cache_set is not None:
            try:
                cache_set(text_processed, src_lang, target, final_result)
            except Exception as e:
                print(f"[translate] Error guardando en cache: {e}")
        return final_result

    # ── SIN_TRAD: devolver original sin cambios ──────────────────
    # Esto pasa cuando CT2 y Google fallan (raro). En vez de esperar
    # 30s+ con backoff exponencial, devolvemos el texto original
    # inmediatamente. El usuario ve el texto sin traducir pero al menos
    # no se cuelga todo el pipeline por un solo bloque.
    # Fallback offline para CJK->es: los modelos disponibles son CJK->en y
    # en->es. Se usa SOLO despuÃ©s de Google directo para no degradar la
    # calidad cuando el traductor online estÃ¡ disponible.
    pivot_source = {"zh-cn": "zh", "zh-tw": "zh"}.get(src_lang, src_lang)
    if target == "es" and pivot_source in {"ja", "ko", "zh"}:
        pivot_en = _translate_ctranslate2(text_processed, pivot_source, "en")
        if pivot_en and _resultado_valido("ctranslate2-pivot-cjk-en", pivot_en, pivot_source):
            pivot_es = _translate_ctranslate2(pivot_en, "en", "es")
            if pivot_es and _resultado_valido("ctranslate2-pivot-en-es", pivot_es, "en"):
                final_result = _post_process_translation(pivot_es, "en", "es")
                final_result = _preservar_honorificos(
                    text_processed, final_result, src_lang, target)
                print(f"[translate] ctranslate2 pivot {pivot_source}->en->es OK: "
                      f"'{final_result[:50]}'")
                if translation_cache_available and cache_set is not None:
                    try:
                        cache_set(text_processed, src_lang, target, final_result)
                    except Exception as e:
                        print(f"[translate] Error guardando en cache: {e}")
                return final_result

    print(f"[translate] SIN_TRAD (fallback final): '{text_processed[:50]}'")
    return text_processed
