"""
translator_offline.py — Traducción offline con ctranslate2 (OPUS-MT) + fallbacks.
Prioridad: CTranslate2 (int8) -> ArgosTranslate -> Google (online).
"""
import os
import threading
from pathlib import Path

CT2_AVAILABLE = False
try:
    import ctranslate2
    from transformers import AutoTokenizer
    CT2_AVAILABLE = True
except Exception:
    pass

MODEL_DIR = Path(__file__).parent / "models" / "opus-mt-es-en-ct2"
HF_MODEL = "Helsinki-NLP/opus-mt-es-en"

_ct2_translator = None
_ct2_tokenizer = None
_ct2_lock = threading.Lock()
_ct2_init_error = None


def _ensure_model_downloaded():
    """Descarga y convierte OPUS-MT ES-EN a ctranslate2 int8 si no existe."""
    global _ct2_init_error
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        return True
    
    try:
        print("[CT2] Descargando y convirtiendo modelo OPUS-MT ES-EN (primera vez)...")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Usar API Python y guardar tokenizer explícitamente
        from ctranslate2.converters import TransformersConverter
        from transformers import AutoTokenizer
        
        converter = TransformersConverter(HF_MODEL)
        converter.convert(str(MODEL_DIR), quantization="int8", force=True)
        
        # Guardar tokenizer compatible
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        tokenizer.save_pretrained(str(MODEL_DIR))
        
        print(f"[CT2] Modelo listo en {MODEL_DIR}")
        return True
    except Exception as e:
        print(f"[CT2] Error preparando modelo: {e}")
        _ct2_init_error = str(e)
        return False


def _get_ct2_translator():
    """Inicializa translator y tokenizer (lazy, thread-safe)."""
    global _ct2_translator, _ct2_tokenizer, _ct2_init_error
    
    if _ct2_translator is not None and _ct2_tokenizer is not None:
        return _ct2_translator, _ct2_tokenizer
    
    if not CT2_AVAILABLE:
        _ct2_init_error = "ctranslate2 no instalado"
        return None, None
    
    with _ct2_lock:
        if _ct2_translator is not None and _ct2_tokenizer is not None:
            return _ct2_translator, _ct2_tokenizer
        
        if not _ensure_model_downloaded():
            return None, None
        
        try:
            _ct2_translator = ctranslate2.Translator(str(MODEL_DIR), device="cpu", compute_type="int8")
            _ct2_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
            print("[CT2] Translator cargado (int8, CPU)")
            return _ct2_translator, _ct2_tokenizer
        except Exception as e:
            print(f"[CT2] Error cargando modelo: {e}")
            _ct2_init_error = str(e)
            return None, None


def translate_ct2(text: str, source: str = "es", target: str = "en") -> str | None:
    """Traduce con CTranslate2 (OPUS-MT). Solo soporta ES->EN por ahora."""
    if source != "es" or target != "en":
        return None
    
    translator, tokenizer = _get_ct2_translator()
    if translator is None or tokenizer is None:
        return None
    
    try:
        # Tokenizar
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        # Traducir
        results = translator.translate_batch([input_ids])
        output_ids = results[0].hypotheses[0]
        # Detokenizar
        translated = tokenizer.decode(output_ids, skip_special_tokens=True)
        return translated.strip()
    except Exception as e:
        print(f"[CT2] Error traduciendo: {e}")
        return None


def _translate_argos(text: str, source: str, target: str) -> str | None:
    """Fallback 1: ArgosTranslate offline."""
    try:
        import argostranslate.translate
        installed = argostranslate.translate.get_installed_languages()
        src_lang = next((l for l in installed if l.code == source), None)
        tgt_lang = next((l for l in installed if l.code == target), None)
        if src_lang and tgt_lang:
            tr = src_lang.get_translation(tgt_lang)
            if tr:
                return tr.translate(text)
    except Exception as e:
        print(f"[Argos] Error: {e}")
    return None


def _translate_google(text: str, source: str, target: str) -> str | None:
    """Fallback 2: Google Translate (online)."""
    try:
        from deep_translator import GoogleTranslator
        import requests
        session = requests.Session()
        original_request = session.request
        def request_with_timeout(*args, **kwargs):
            kwargs['timeout'] = kwargs.get('timeout', 15)
            return original_request(*args, **kwargs)
        session.request = request_with_timeout
        translator = GoogleTranslator(source=source, target=target)
        translator._session = session
        result = translator.translate(text)
        return result
    except Exception as e:
        print(f"[Google] Error: {e}")
    return None


def translate_offline(text: str, source: str = "auto", target: str = "en") -> str:
    """
    Pipeline de traducción con fallbacks priorizados:
    1. CTranslate2 (OPUS-MT ES->EN, int8, offline, rápido)
    2. ArgosTranslate (offline, multi-idioma)
    3. Google Translate (online, último recurso)
    
    Retorna la mejor traducción disponible o el texto original si todo falla.
    """
    text = text.strip()
    if not text:
        return text
    
    # Detectar idioma si auto
    if source == "auto":
        source = _detect_lang_simple(text)
    
    if source == target:
        return text
    
    # Intento 1: CTranslate2 (solo ES->EN)
    if source == "es" and target == "en":
        result = translate_ct2(text, source, target)
        if result and _es_valida(text, result):
            return result
        print("[Pipeline] CT2 falló o traducción inválida -> Argos")
    
    # Intento 2: Argos
    result = _translate_argos(text, source, target)
    if result and _es_valida(text, result):
        return result
    print("[Pipeline] Argos falló o traducción inválida -> Google")
    
    # Intento 3: Google (online)
    result = _translate_google(text, source, target)
    if result and _es_valida(text, result):
        return result
    
    print("[Pipeline] Todos los fallbacks fallaron -> devolviendo original")
    return text


def _detect_lang_simple(text: str) -> str:
    """Detección simple ES/EN/JA/KO/ZH."""
    if any(0xac00 <= ord(c) <= 0xd7a3 for c in text):
        return "ko"
    if any((0x3040 <= ord(c) <= 0x30ff) or (0x4e00 <= ord(c) <= 0x9faf) for c in text):
        return "ja"
    if any(c in "áéíóúñüÁÉÍÓÚÑÜ¿¡" for c in text):
        return "es"
    spa = {"el","la","los","las","que","en","un","una","de","con","es","para","por","si","no","y","pero","como","mas","mas","bien","todo","esta","este"}
    if spa.intersection({w.strip(".,¡!¿?()[]{}\"'") for w in text.lower().split()}):
        return "es"
    return "en"


def _es_valida(orig: str, trad: str) -> bool:
    """Valida que la traducción no sea basura."""
    if not trad or not trad.strip():
        return False
    if trad.strip() == orig.strip():
        return False
    # Demasiado repetitivo
    if len(trad) > 20:
        unicos = len(set(trad.lower()))
        if unicos / max(len(trad), 1) < 0.15:
            return False
    # Excesivamente larga
    if orig and len(trad) > len(orig) * 10:
        return False
    # Substring repetido muchas veces
    if len(trad) > 10:
        for l in range(2, min(30, len(trad) // 5)):
            sub = trad[:l]
            if trad.count(sub) >= 5 and len(sub) * 5 <= len(trad):
                return False
    return True


# Para compatibilidad con server.py
def translate_one(text: str, source: str, target: str) -> str:
    """Wrapper compatible con firma de server.py:_translate_one."""
    return translate_offline(text, source, target)


def get_ct2_status() -> dict:
    """Info de estado para health check."""
    return {
        "available": CT2_AVAILABLE,
        "model_loaded": _ct2_translator is not None,
        "model_path": str(MODEL_DIR),
        "init_error": _ct2_init_error,
    }