# Contexto: bug de traducciones que quedan sin traducir — Traductor Visual Pro

## 1. El problema

App local de traducción de manga/cómics (Flask + EasyOCR + pipeline de
traducción en cascada CT2 → Argos → Google). Al traducir lotes grandes
de páginas, **algunos bloques de texto quedan sin traducir** (el texto
"traducido" es idéntico al original) de forma aparentemente aleatoria,
sin ningún error visible en logs ni en el status de la página.

No hay capacidad de ver imágenes en esta conversación — pero **no hace
falta**: toda la información relevante ya existe en texto plano
(el JSON de checkpoint guarda `src`/`tgt` de cada bloque de texto).

## 2. Causa raíz encontrada

Archivo: `translator.py`, función `_translate_one` (pipeline principal).

```python
is_lenient = source != "auto" and len(text_processed.split()) <= 3
...
for method_name, fn in translation_fns:
    resultado = fn(text_processed, src_lang, target)
    if resultado and (is_lenient or resultado != text_processed) and \
       _es_traduccion_valida(text_processed, resultado, lenient=is_lenient):
        ...
        return resultado
```

Y dentro de `_es_traduccion_valida`:

```python
def _es_traduccion_valida(orig: str, trad: str, lenient: bool = False) -> bool:
    if not trad or not trad.strip():
        return False
    if not lenient and trad == orig:
        return False
    ...
```

**El problema**: para frases de **3 palabras o menos** (`is_lenient = True`),
el pipeline acepta como "traducción válida" un resultado idéntico al
original. Si CT2/Argos/Google devuelven el mismo texto sin traducir
(diálogo corto, interjecciones, nombres, fragmentos de OCR ruidoso),
el sistema:
1. Lo marca como **éxito**, no como fallo.
2. Lo **cachea** en `cache/translations/` (TTL 7 días) — así que el
   mismo resultado fallido se repite indefinidamente si el mismo texto
   vuelve a aparecer.
3. Nunca se reintenta, porque `reprocess_failed.py` (el script que
   reintenta páginas fallidas) solo detecta fallos de **página completa**
   (`timeout`, `render_error`, `conn_error`, `http_*`) — no fallos de
   **bloque individual** dentro de una página marcada como exitosa.

## 3. Mitigación ya aplicada (no toca la lógica de producción)

Se extendió `reprocess_failed.py` para que, además de reprocesar
páginas rotas, detecte bloques con `src == tgt` (excluyendo
onomatopeyas legítimas con la misma heurística de `analisis_calidad.py`),
borre la entrada de caché correspondiente, y pida la traducción de
nuevo directamente vía `/api/translate` (sin reabrir el PDF). Ver
`reprocess_failed.py` adjunto — función `es_onomatopeya`, `borrar_cache`,
`pedir_traduccion`, y la sección "FASE 2: bloques silenciosos" en `main()`.

Esto es un parche de recuperación posterior, **no arregla la causa
raíz** en `translator.py`.

## 4. Preguntas para la IA

1. ¿Existe una forma más segura de corregir `is_lenient` /
   `_es_traduccion_valida` en `translator.py` que **no** rompa los casos
   legítimos donde la traducción correcta de una frase corta coincide
   con el original (ej. nombres propios, "OK", "NO", "Ah")?
   - Idea a evaluar: en vez de permitir `trad == orig` como válido para
     cualquier frase corta, restringirlo a un diccionario cerrado de
     palabras/nombres conocidos donde eso es esperable, y tratar todo
     lo demás como fallo real (reintentar).
   - Alternativa: distinguir "el motor devolvió literalmente el mismo
     string" (sospechoso) de "el motor no encontró nada que traducir
     porque ya está en el idioma destino" (llamando a
     `_detect_language_robust` sobre el resultado también).
2. ¿Vale la pena registrar explícitamente en el JSON de resultado un
   campo tipo `"engine": "ninguno_valido"` cuando el pipeline cae al
   `return text_processed` final, en vez de depender de comparar
   `src == tgt` después? Eso haría explícito el fallo sin heurística.
3. ¿El umbral de `is_lenient` (`<=3 palabras`) es razonable para
   diálogo de manga en español, o convendría bajarlo a `<=2` para
   reducir falsos negativos, dado que la mayoría de onomatopeyas real
   caen en 1 palabra?

## 5. Archivos relevantes adjuntos

- `translator.py` (pipeline de traducción completo, causa raíz aquí)
- `reprocess_failed.py` (ya parchado con la mitigación de bloques silenciosos)

---

## translator.py (completo)

```python
"""
translator.py — Detección de idioma y traducción (Argos, Google, CT2).

Extraído de server.py. Depende de config.py para constantes.
"""

import os
import re
import shutil
import threading
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
            return str(result)  # type: ignore[no-any-return]
    except Exception as e:
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
    if any((0x3040 <= ord(c) <= 0x30ff) or (0x4e00 <= ord(c) <= 0x9faf) for c in text):
        return "ja"
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
    if any((0x3040 <= ord(c) <= 0x30ff) or (0x4e00 <= ord(c) <= 0x9faf) for c in text):
        return "ja"
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

            # Cargar modelo CT2
            import ctranslate2
            _ct2_translators[pair_key] = ctranslate2.Translator(model_dir, device="cpu")
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

    for method_name, fn in translation_fns:
        try:
            resultado = fn(text_processed, src_lang, target)
            if resultado and (is_lenient or resultado != text_processed) and _es_traduccion_valida(text_processed, resultado, lenient=is_lenient):
                print(f"[translate] {method_name} OK: '{resultado[:50]}'")
                if translation_cache_available and cache_set is not None:
                    try:
                        cache_set(text_processed, src_lang, target, resultado)
                    except Exception:
                        pass
                return resultado
            print(f"[translate] {method_name} inválido: '{resultado[:50] if resultado else 'None'}'")
        except Exception as e:
            print(f"[translate] {method_name} error: {e}")

    print(f"[translate] Todos los métodos fallaron, devolviendo original")
    return text_processed

```

---

## reprocess_failed.py (ya parchado)

```python
"""
reprocess_failed.py — Reintenta lo que falló en el batch original.

Cubre DOS tipos de fallo distintos:

  1. FALLOS DE PÁGINA (visibles): status timeout / render_error /
     conn_error / http_*. Se reprocesa la página completa contra
     /api/process-page (igual que antes).

  2. FALLOS DE BLOQUE (silenciosos): bloques donde source == translated
     dentro de una página que SÍ se marcó como procesada con éxito.
     Causa raíz: en translator.py, is_lenient permite que frases de
     <=3 palabras pasen la validación aunque la "traducción" sea
     idéntica al original — esos bloques nunca se marcan como error
     y por eso nunca se reintentaban antes de este parche.
     Se reparan SIN reabrir el PDF: se llama directo a /api/translate
     con el texto ya extraído, después de purgar la entrada de caché
     correspondiente (si no se purga, /api/translate devolvería el
     mismo resultado fallido cacheado).

Actualiza resultados_progreso.json al final con ambos tipos de fix.
"""
import os, sys, time, json, base64, gc, hashlib, re, argparse
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8')

import requests

CHECKPOINT_FILE = "resultados_progreso.json"
PDF_PATH = "Capítulo 43 de Cómo criar villanos correctamente _ Olympus Scanlation_compressed.pdf"
API_URL = "http://127.0.0.1:5174"
ZOOM = 1.5
TARGET = "en"
SOURCE = "es"
TIMEOUT = 90
MAX_RETRIES = 3
CACHE_DIR = os.path.join("cache", "translations")

# ── heurística de onomatopeya (misma lógica que analisis_calidad.py) ──
SHORT_SPANISH = {
    'EL','LA','LOS','LAS','UN','UNA','UNOS','UNAS','DEL','CON','POR','QUE','QUÉ',
    'SER','ESTA','ESTE','ES','NO','SI','YA','PERO','MAS','MÁS','SUS','ERA','HAN',
    'LES','LE','TUS','NOS','SON','AL','MI','TU','SE','TE','ME','LO','DE','EN','A',
    'Y','O','NI','VA','VE','FUE','IR','HAY','HE','HA','HAS','SOY','ERES','SEA',
    'TAN','TAL','AH','OH','EH','AY','OK','BIEN','MAL','TODO','SOLO','SÓLO','MUY',
    'VALE','LISTO','CLARO','CIERTO','BUENO','MALO','COMO','CÓMO','AHORA','HOY',
}
KNOWN_SFX = {
    'BOOM','PUM','ZAS','CRASH','CLICK','PLOP','TOC','RING','FLASH','BOING','POW',
    'BANG','SMASH','SPLASH','BUMP','THUD','WHAM','GRRR','GRR','CLANG','SNIFF',
    'GROAN','SLAM','BEEP','WOOSH','KABOOM','RUMBLE','SQUEAK','WHIR','ZOOM',
    'VROOM','SCREECH','GROWL','HOWL','SNAP','CRACKLE','POP','FIZZ','HISS','BUZZ',
    'DING','DONG','SPLAT','SQUISH','WHOOSH','HUH','HEH','HAH','HMPH','PSST','SHH',
    'GASP','PANT','PHEW','WHEW','SIGH','UFF','OW','OUCH','AY','OY','BAH','MEH',
    'ACK','EEK','TAP','KNOCK','RAP','PIT','PAT',
}

def es_onomatopeya(t: str) -> bool:
    s = t.strip(' \'"\u00a1\u00bf!?.,;:~-_()').upper()
    if not s or not s.isalpha():
        return False
    if len(s) < 3 or len(s) > 8:
        return False
    if s in SHORT_SPANISH:
        return False
    if s in KNOWN_SFX:
        return True
    if re.search(r'(.)\1{2,}', s):
        return True
    return False

def cache_key(text: str, src: str, tgt: str) -> str:
    raw = f"{text}||{src}||{tgt}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

def borrar_cache(text: str, src: str, tgt: str) -> bool:
    path = os.path.join(CACHE_DIR, f"{cache_key(text, src, tgt)}.json")
    if os.path.exists(path):
        os.remove(path)
        return True
    return False

def pedir_traduccion(text: str) -> str | None:
    try:
        r = requests.post(f"{API_URL}/api/translate",
                           json={"text": text, "source": SOURCE, "target": TARGET},
                           timeout=30)
        if r.status_code == 200:
            return r.json().get("translatedText")
    except Exception as e:
        print(f"    [ERROR] {e}")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-bloques", action="store_true",
                     help="Omite el reprocesamiento de páginas completas (no requiere el PDF ni fitz/cv2)")
    args = ap.parse_args()

    # ── Load checkpoint ──────────────────────────────────────────
    if not os.path.exists(CHECKPOINT_FILE):
        print(f"ERROR: No se encuentra {CHECKPOINT_FILE}")
        sys.exit(1)

    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        cp = json.load(f)

    total_pages = cp.get("total_pages", 128)
    results = cp.get("results", [])
    pages_done = set(cp.get("pages_done", []))

    # ── Identify page-level failures ────────────────────────────
    failed_pages = []
    for r in results:
        status = r.get("status", "")
        if status in ("timeout", "render_error", "conn_error") or str(status).startswith("http"):
            failed_pages.append(r["page"])
    failed_pages = sorted(set(failed_pages))

    # ── Identify block-level silent failures (src == tgt) ──────
    # Se excluyen las páginas que ya van a reprocesarse completas.
    bloques_fallidos = []  # (result_idx, texto_idx, page, src)
    for ri, r in enumerate(results):
        if r.get("page") in failed_pages:
            continue
        for ti, t in enumerate(r.get("texts", [])):
            src = (t.get("src") or "").strip()
            tgt = (t.get("tgt") or "").strip()
            if src and src == tgt and not es_onomatopeya(src):
                bloques_fallidos.append((ri, ti, r.get("page"), src))

    print(f"Páginas fallidas (error de página): {len(failed_pages)}")
    print(f"  {failed_pages}")
    print(f"Bloques fallidos (traducción silenciosamente idéntica): {len(bloques_fallidos)}")
    print()

    if not failed_pages and not bloques_fallidos:
        print("No hay nada que reprocesar. ✅")
        sys.exit(0)

    # ── Verify server ────────────────────────────────────────────
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        health = r.json()
        print(f"Servidor OK - memoria: {health.get('memory','?')}")
    except Exception as e:
        print(f"ERROR: No se puede conectar al servidor: {e}")
        sys.exit(1)

    new_results = []
    recovered = 0

    # ═══════════════════ FASE 1: páginas completas ═══════════════
    if failed_pages and not args.solo_bloques:
        import fitz, cv2, numpy as np
        from PIL import Image
        from io import BytesIO

        doc = fitz.open(PDF_PATH)

        for retry_count in range(1, MAX_RETRIES + 1):
            recovered_set = {r["page"] for r in new_results}
            remaining = [p for p in failed_pages if p not in recovered_set]
            if not remaining:
                break

            print(f"\n{'='*60}")
            print(f"  INTENTO {retry_count}/{MAX_RETRIES} — {len(remaining)} páginas pendientes")
            print(f"{'='*60}")

            for i, page_num in enumerate(remaining):
                pg_idx = page_num - 1
                print(f"  [{i+1}/{len(remaining)}] Pág {page_num} (intento {retry_count})...", end=" ", flush=True)

                try:
                    page = doc[pg_idx]
                    pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM))
                    img = Image.open(BytesIO(pix.tobytes('png')))
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode('.png', img_cv, [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    b64 = 'data:image/png;base64,' + base64.b64encode(buf.tobytes()).decode()
                    del img, img_cv, buf, pix, page

                    t0 = time.time()
                    resp = requests.post(f"{API_URL}/api/process-page",
                        json={'image': b64, 'target': TARGET, 'source': SOURCE},
                        timeout=TIMEOUT)
                    elapsed = time.time() - t0
                    del b64

                    if resp.status_code == 200:
                        data = resp.json()
                        blocks = data.get("blocks", [])
                        n_blocks = len(blocks)
                        n_translated = sum(1 for b in blocks if b.get("source","") != b.get("translated",""))
                        status = ("OK" if n_translated == n_blocks > 0 else
                                  "PARCIAL" if n_translated > 0 else
                                  "SIN_TRAD" if n_blocks > 0 else "VACIO")

                        print(f"✅ {status} ({elapsed:.1f}s, {n_blocks} bloques, {n_translated} trad)")

                        new_entry = {
                            "page": page_num,
                            "status": status,
                            "blocks": n_blocks,
                            "translated": n_translated,
                            "time": elapsed,
                            "texts": [{"src": b.get("source",""), "tgt": b.get("translated","")} for b in blocks]
                        }
                        new_results.append(new_entry)
                        pages_done.add(page_num)
                        recovered += 1
                    else:
                        err_msg = ""
                        try:
                            err_data = resp.json()
                            err_msg = err_data.get("error", str(resp.status_code))
                        except Exception:
                            err_msg = f"HTTP {resp.status_code}"
                        print(f"❌ {err_msg} ({elapsed:.1f}s)")

                except requests.Timeout:
                    print(f"⏰ Timeout (>{TIMEOUT}s)")
                except Exception as e:
                    print(f"💥 Error: {str(e)[:60]}")

                gc.collect()
                if i < len(remaining) - 1:
                    time.sleep(0.5)

            if retry_count < MAX_RETRIES:
                recovered_set = {r["page"] for r in new_results}
                still_failed_count = len([p for p in failed_pages if p not in recovered_set])
                if still_failed_count == 0:
                    break
                print(f"\n  {still_failed_count} páginas aún fallan. Reintentando...")

        doc.close()

    recovered_pages = {r["page"] for r in new_results}
    still_failed_pages = sorted([p for p in failed_pages if p not in recovered_pages])
    total_recovered = len(recovered_pages)

    for entry in new_results:
        pg = entry["page"]
        found = False
        for i, r in enumerate(results):
            if r["page"] == pg:
                results[i] = entry
                found = True
                break
        if not found:
            results.append(entry)

    # ═══════════════════ FASE 2: bloques silenciosos ══════════════
    bloques_mejorados = 0
    bloques_reporte = []  # (page, src, nueva, estado)
    if bloques_fallidos:
        print(f"\n{'='*60}")
        print(f"  REPARANDO {len(bloques_fallidos)} BLOQUES SILENCIOSOS")
        print(f"{'='*60}")
        for ri, ti, page, src in bloques_fallidos:
            borrado = borrar_cache(src, SOURCE, TARGET)
            nueva = pedir_traduccion(src)
            estado = "SIN CAMBIO"
            if nueva and nueva.strip() and nueva.strip() != src:
                results[ri]["texts"][ti]["tgt"] = nueva
                # Actualizar contador "translated" de esa página si existe
                if 0 <= ri < len(results) and "translated" in results[ri]:
                    results[ri]["translated"] = sum(
                        1 for tt in results[ri]["texts"]
                        if (tt.get("src") or "") != (tt.get("tgt") or "")
                    )
                estado = "MEJORADO"
                bloques_mejorados += 1
            print(f"  Pág {page:>3} [{'cache borrada' if borrado else 'sin cache':>13}] "
                  f"{estado:>10}: {src[:35]!r} -> {(nueva or src)[:35]!r}")
            bloques_reporte.append((page, src, nueva or src, estado))

    # ── Re-count stats from scratch for accuracy ────────────────
    all_pages_with_text = 0
    all_pages_translated = 0
    all_pages_empty = 0
    all_pages_error = 0
    all_blocks_found = 0
    all_blocks_translated = 0

    for r in results:
        status = r.get("status", "")
        if status in ("timeout", "render_error", "conn_error") or str(status).startswith("http"):
            all_pages_error += 1
            continue
        if status == "VACIO":
            all_pages_empty += 1
            continue
        n_blocks = r.get("blocks", 0)
        n_translated = sum(
            1 for tt in r.get("texts", [])
            if (tt.get("src") or "") != (tt.get("tgt") or "")
        ) if r.get("texts") else r.get("translated", 0)
        all_blocks_found += n_blocks
        all_blocks_translated += n_translated
        if n_blocks > 0:
            all_pages_with_text += 1
            if n_translated > 0:
                all_pages_translated += 1

    updated_cp = {
        "total_pages": total_pages,
        "pages_done": sorted(pages_done),
        "results": results,
        "stats": {
            "total_blocks_found": all_blocks_found,
            "total_blocks_translated": all_blocks_translated,
            "pages_with_text": all_pages_with_text,
            "pages_translated": all_pages_translated,
            "pages_empty": all_pages_empty,
            "pages_error": all_pages_error,
        },
        "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    tmp_file = CHECKPOINT_FILE + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(updated_cp, f, ensure_ascii=False, indent=2)
    os.replace(tmp_file, CHECKPOINT_FILE)

    # ── Final report ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  REPORTE DE RECUPERACION")
    print("=" * 60)
    if failed_pages:
        print(f"\n  Páginas fallidas originales: {len(failed_pages)}")
        print(f"  ✅ Recuperadas: {total_recovered}")
        print(f"  ❌ Siguen fallando: {len(still_failed_pages)}")
        print(f"     Páginas: {still_failed_pages if still_failed_pages else 'ninguna'}")
    if bloques_fallidos:
        print(f"\n  Bloques silenciosos detectados: {len(bloques_fallidos)}")
        print(f"  ✅ Mejorados: {bloques_mejorados}")
        print(f"  ⚠️  Sin cambio: {len(bloques_fallidos) - bloques_mejorados}")
    print()
    print(f"  Nuevas estadísticas globales:")
    s = updated_cp["stats"]
    print(f"  ✅ Traducidas correctamente: {s['pages_translated']}")
    print(f"  ⚠️  Con texto sin traducir:  {s['pages_with_text'] - s['pages_translated']}")
    print(f"  ℹ️  Vacías (arte):            {s['pages_empty']}")
    print(f"  ❌ Con error:                 {s['pages_error']}")
    print(f"  📊 Total bloques:             {s['total_blocks_found']}")
    print(f"  📊 Traducidos:                {s['total_blocks_translated']}")
    if s['total_blocks_found'] > 0:
        print(f"  📊 Tasa traducción:           {s['total_blocks_translated']/s['total_blocks_found']*100:.1f}%")
    print(f"\n  Checkpoint actualizado: {CHECKPOINT_FILE}")

    # ── Reporte en texto plano para pegar a una IA de solo texto ──
    pendientes = [x for x in bloques_reporte if x[3] == "SIN CAMBIO"]
    if pendientes:
        print()
        print("=" * 60)
        print("  BLOQUES QUE SIGUEN SIN TRADUCIR (para IA de texto)")
        print("=" * 60)
        for page, src, _, _ in pendientes:
            print(f"  Página {page} | texto original (es): {src}")

if __name__ == "__main__":
    main()

```
