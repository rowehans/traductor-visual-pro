"""
test_translator.py — Tests unitarios para translator.py.

Cubre las funciones clave del pipeline de traducción:
- Detección de idioma (_detect_language_simple, _detect_language_robust)
- Detección SFX (_es_sfx)
- Post-procesamiento (_post_process_translation)
- Glosario (_aplicar_glosario)
- Detección de ruido OCR (_es_ocr_noise)
- Validación de traducción (_es_traduccion_valida)
- Corrección CT2 (_corregir_ct2)
- Pipeline principal (_translate_one) con mocks
"""

import sys
import os
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

# Asegurar que el proyecto esté en sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from translator import (
    _detect_language_simple,
    _detect_language_robust,
    _detect_mixed_languages,
    _es_sfx,
    _translation_is_likely_source_language,
    _post_process_translation,
    _preservar_honorificos,
    _aplicar_glosario,
    _es_ocr_noise,
    _es_traduccion_valida,
    _corregir_ct2,
    _translate_one,
    _ct2_gpu_allowed,
    _GLOSARIO_POST_REGEX,
    _SFX_PATTERNS,
    # CT2 / Google internals
    _compute_file_sha256,
    _load_ct2_checksums,
    _save_ct2_checksums,
    _verify_ct2_checksums,
    _get_ct2_translator,
    _translate_ctranslate2,
    _translate_argos,
    _get_google_session,
    _translate_google,
    _ensure_argo_package,
    # Constantes y estado global (para aislar entre tests)
    _CT2_CHECKSUMS_FILE,
    _ct2_translators,
    _ct2_tokenizers,
    _CT2_BASE_DIR,
    _google_session,
    _google_rate_limit_state,
    _translators,
    _RATE_LIMIT_THRESHOLD,
    _MAX_BACKOFF,
    _argo_lock,
    _argo_ready,
)


# ═══════════════════════════════════════════════════════════════
# _detect_language_simple
# ═══════════════════════════════════════════════════════════════

class TestDetectLanguageSimple:
    def test_detects_spanish_by_accents(self):
        assert _detect_language_simple("áéíóú") == "es"
        assert _detect_language_simple("canción") == "es"
        assert _detect_language_simple("época") == "es"
        assert _detect_language_simple("¿Qué tal?") == "es"
        assert _detect_language_simple("¡Increíble!") == "es"

    def test_detects_spanish_by_common_words(self):
        assert _detect_language_simple("el perro es grande") == "es"
        assert _detect_language_simple("la casa está lejos") == "es"
        assert _detect_language_simple("quiero hacer algo") == "es"
        assert _detect_language_simple("los villanos correctamente") == "es"
        assert _detect_language_simple("temporada") == "es"

    def test_detects_spanish_by_verb_suffixes(self):
        assert _detect_language_simple("correr") == "es"
        assert _detect_language_simple("hablarme") == "es"
        assert _detect_language_simple("diciéndole") == "es"

    def test_detects_english_by_default(self):
        assert _detect_language_simple("hello world") == "en"
        # Evitar palabras que terminan en sufijos verbales españoles como 'er' (power→ends with er→falso positivo)
        assert _detect_language_simple("my house is big") == "en"
        assert _detect_language_simple("season seven") == "en"
        assert _detect_language_simple("I am inevitable") == "en"
        # 'the' no esta en SPA_WORDS pero 'de' si — evitar usar 'de' en textos ingles
        assert _detect_language_simple("the world is mine") == "en"

    def test_detects_short_portuguese_text(self):
        """Portugués sigue activo aunque se pausen otros idiomas latinos."""
        text, expected = "Eu amo voce", "pt"
        assert _detect_language_simple(text) == expected

    def test_detects_korean_by_hangul(self):
        assert _detect_language_simple("안녕하세요") == "ko"
        assert _detect_language_simple("감사합니다") == "ko"

    def test_detects_japanese_by_kana(self):
        assert _detect_language_simple("こんにちは") == "ja"
        assert _detect_language_simple("さようなら") == "ja"
        # Katakana
        assert _detect_language_simple("コンニチハ") == "ja"

    def test_detects_chinese_by_hanzi(self):
        assert _detect_language_simple("你好") == "zh"
        assert _detect_language_simple("谢谢") == "zh"

    def test_handles_mixed_caps_spanish(self):
        assert _detect_language_simple("INCREÍBLE") == "es"
        assert _detect_language_simple("TEMPORADA") == "es"

    def test_handles_empty_text(self):
        assert _detect_language_simple("") == "en"

    def test_handles_short_numbers_text(self):
        assert _detect_language_simple("123") == "en"
        assert _detect_language_simple("3/128") == "en"

    # ── Casos heredados de test_ci.py (migrados como parametrizados) ──

    @pytest.mark.parametrize("text,expected", [
        pytest.param("RESPONDERME?", "es", id="resp-verbo-enclitico-con-signo"),
        pytest.param("RESPONDERME", "es", id="resp-verbo-enclitico"),
        pytest.param("RESPÓNDEME", "es", id="resp-verbo-enclitico-acentuado"),
        pytest.param("Hello world", "en", id="hello-world"),
        pytest.param("HELLO WORLD", "en", id="hello-world-mayusculas"),
        pytest.param("¿Puedes ayudarme?", "es", id="puedes-ayudarme"),
        pytest.param("Gracias por todo", "es", id="gracias-por-todo"),
        pytest.param("¿Cómo estás?", "es", id="como-estas"),
        pytest.param("Hola, ¿qué tal?", "es", id="hola-que-tal"),
        pytest.param("This is a test", "en", id="this-is-a-test"),
    ])
    def test_ci_legacy_cases(self, text, expected):
        """
        Casos heredados de test_ci.py — cubren detección de español
        con verbos enclíticos, signos ¿¡, mayúsculas sostenidas, etc.
        """
        assert _detect_language_simple(text) == expected


# ═══════════════════════════════════════════════════════════════
# _detect_language_robust
# ═══════════════════════════════════════════════════════════════

class TestDetectLanguageRobust:
    def test_korean_always_korean(self):
        assert _detect_language_robust("안녕하세요") == "ko"
        assert _detect_language_robust("감사합니다 여러분") == "ko"

    def test_japanese_kana(self):
        assert _detect_language_robust("こんにちは") == "ja"

    def test_chinese_hanzi(self):
        result = _detect_language_robust("你好世界")
        assert result == "zh"

    def test_spanish_text(self):
        assert _detect_language_robust("Hola, ¿cómo estás?") == "es"
        assert _detect_language_robust("Esto es una prueba") == "es"

    def test_english_text(self):
        assert _detect_language_robust("Hello world") == "en"
        assert _detect_language_robust("This is a test") == "en"

    def test_short_spanish_text(self):
        # Textos cortos (<4 palabras) con acentos/heurística española
        assert _detect_language_robust("Soy") == "en"  # No tiene acentos

    def test_empty_text(self):
        assert _detect_language_robust("") == "en"

    def test_reutiliza_deteccion_de_texto_repetido(self):
        """Las frases repetidas no deben volver a invocar langdetect."""
        _detect_language_robust.cache_clear()

        assert _detect_language_robust("Hello world") == "en"
        before = _detect_language_robust.cache_info().hits
        assert _detect_language_robust("Hello world") == "en"

        assert _detect_language_robust.cache_info().hits == before + 1

    @pytest.mark.parametrize("text", ["Bonjour merci", "Ich liebe dich", "Ciao grazie"])
    def test_disabled_languages_never_selected_automatically(self, text):
        assert _detect_language_robust(text) not in {"fr", "de", "it"}

    @pytest.mark.parametrize("source", ["fr", "de", "it"])
    def test_disabled_language_translation_is_preserved(self, source):
        assert _translate_one("texto de prueba", source, "es") == "texto de prueba"


class TestMixedLanguageDetection:
    """Detecta cambio de idioma sin confundir nombres o palabras aisladas."""

    def test_detects_confident_mixed_spanish_english_phrase(self):
        languages = _detect_mixed_languages("No puedo go home", dominant="es")
        assert languages[0] == "es"
        assert set(languages) == {"es", "en"}

    def test_detects_short_english_phrase_inside_cjk_or_latin_block(self):
        languages = _detect_mixed_languages("君が好き I love you", dominant="ja")
        assert languages[0] == "ja"
        assert "en" in languages

    def test_does_not_flag_single_loanword_as_mixed(self):
        assert _detect_mixed_languages("Hola amigo", dominant="es") == ("es",)

    def test_detects_natural_code_switch_with_common_words(self):
        languages = _detect_mixed_languages("Mi amigo says hello", dominant="es")
        assert languages[0] == "es"
        assert set(languages) == {"es", "en"}

    @pytest.mark.parametrize(
        ("text", "dominant"),
        [("Eu amo you", "pt")],
    )
    def test_detects_romance_language_english_code_switch(self, text, dominant):
        languages = _detect_mixed_languages(text, dominant=dominant)
        assert languages[0] == dominant
        assert "en" in languages

    @pytest.mark.parametrize(
        ("text", "dominant", "foreign"),
        [
            ("I love gracias", "en", "es"),
        ],
    )
    def test_no_confunde_marcadores_de_un_mismo_idioma(self, text, dominant, foreign):
        """Una palabra extranjera solo cuenta si aporta otro idioma real."""
        languages = _detect_mixed_languages(text, dominant=dominant)
        assert languages[0] == dominant
        assert foreign in languages

    def test_detects_palabra_distintiva_extranjera_con_dominante_fuerte(self):
        languages = _detect_mixed_languages("I love gracias", dominant="en")
        assert languages == ("en", "es")

    def test_no_activa_mixed_por_prestamo_ambiguo(self):
        assert _detect_mixed_languages("I love amigo", dominant="en") == ("en",)


class TestCt2GpuBudget:
    def test_rejects_new_gpu_model_when_vram_headroom_is_low(self):
        snapshot = {
            "available": True,
            "free_mb": 500.0,
            "total_mb": 4096.0,
        }
        assert _ct2_gpu_allowed(snapshot) is False


# ═══════════════════════════════════════════════════════════════
# _es_sfx
# ═══════════════════════════════════════════════════════════════

class TestEsSfx:
    def test_common_sfx(self):
        assert _es_sfx("BANG") is True
        assert _es_sfx("CRASH") is True
        assert _es_sfx("BOOM") is True
        assert _es_sfx("SLAM") is True
        assert _es_sfx("WHAM") is True
        assert _es_sfx("ZAP") is True

    def test_repeated_sfx(self):
        assert _es_sfx("BANG BANG") is True
        assert _es_sfx("CRASH CRASH") is True

    def test_sfx_with_numbers(self):
        assert _es_sfx("BOOM 1") is True
        assert _es_sfx("CRASH 2") is True

    def test_sfx_with_punctuation(self):
        assert _es_sfx("KABOOM!") is True
        assert _es_sfx("CRASH...") is True

    def test_japanese_onomatopoeia(self):
        assert _es_sfx("DON") is True
        assert _es_sfx("GYAA") is True
        assert _es_sfx("PACHIN") is True
        assert _es_sfx("ZUDON") is True
        assert _es_sfx("ZUBAN") is True

    def test_repeated_cjk_sfx(self):
        """Repeticiones inequívocas CJK son SFX, no diálogo."""
        assert _es_sfx("ゴゴゴ") is True
        assert _es_sfx("哈哈哈") is True
        assert _es_sfx("ㅋㅋㅋ") is True
        assert _es_sfx("こんにちは") is False

    def test_thought_bubbles(self):
        assert _es_sfx("*thinking*") is True
        assert _es_sfx("*sigh*") is True

    def test_not_sfx(self):
        assert _es_sfx("Hola mundo") is False
        assert _es_sfx("¿Cómo estás?") is False
        assert _es_sfx("Esto es una prueba larga") is False
        assert _es_sfx("") is False

    def test_all_caps_names_are_sfx(self):
        # Nombres de personajes en mayúsculas (controversial)
        assert _es_sfx("NARUTO") is True
        assert _es_sfx("SAKURA") is True
        assert _es_sfx("GOKU") is True

    def test_spanish_words_in_uppercase_not_sfx(self):
        # Palabras españolas en mayúsculas — excluidas vía _SFX_EXCLUDE
        assert _es_sfx("PERO") is False
        assert _es_sfx("ELLA") is False
        assert _es_sfx("ESTA") is False
        assert _es_sfx("BIEN") is False
        assert _es_sfx("NUNCA") is False

    def test_common_words_in_supported_languages_not_sfx(self):
        """Una palabra de diálogo en CAPS no debe perderse como SFX."""
        for word in ("STOP", "WHAT", "OBRIGADO"):
            assert _es_sfx(word) is False

    def test_interjecciones_alargadas_no_son_sfx(self):
        for text in ("NOOOOO!", "HELP!", "WAIT!", "AAAAH!"):
            assert _es_sfx(text) is False

    def test_sfx_repetitivo_inequivoco_se_conserva(self):
        assert _es_sfx("GRRRR") is True

    def test_short_texts_not_sfx(self):
        assert _es_sfx("Hola") is False
        assert _es_sfx("Adiós") is False
        assert _es_sfx("Bien") is False

    def test_long_text_not_sfx(self):
        assert _es_sfx("A" * 30) is False
        # Texto de más de 20 chars no puede ser SFX
        assert _es_sfx("This is a very long text that should not be SFX") is False


# ═══════════════════════════════════════════════════════════════
# _post_process_translation
# ═══════════════════════════════════════════════════════════════

class TestPostProcessTranslation:
    def test_capitalizes_first_letter(self):
        result = _post_process_translation("hello world", "es", "en")
        assert result.startswith("Hello world")

    def test_normalizes_spaces(self):
        result = _post_process_translation("hello    world", "es", "en")
        assert "  " not in result

    def test_capitalizes_first_letter(self):
        result = _post_process_translation("hello world", "es", "en")
        assert result == "Hello world"

    def test_handles_empty_text(self):
        assert _post_process_translation("", "es", "en") == ""
        assert _post_process_translation(None, "es", "en") is None

    def test_preserves_sfx(self):
        result = _post_process_translation("BANG!", "es", "en")
        assert result == "BANG!"

    def test_preserves_existing_caps(self):
        # Nombres en mayúsculas sostenidas no se modifican
        result = _post_process_translation("SEOLLANG", "es", "en")
        assert result == "SEOLLANG"

    def test_preserves_existing_punctuation(self):
        result = _post_process_translation("Hello!", "es", "en")
        assert result == "Hello!"


class TestHonorificos:
    def test_preserva_honorifico_japones_en_nombre_aislado(self):
        assert _preservar_honorificos("田中さん", "Tanaka", "ja", "es") == "Tanaka-san"

    def test_preserva_honorifico_japones_en_otro_destino_soportado(self):
        assert _preservar_honorificos("田中さん", "Tanaka", "ja", "en") == "Tanaka-san"

    def test_preserva_honorifico_coreano_sin_glosario_manual(self):
        assert _preservar_honorificos("김민수씨", "Kim Min-su", "ko", "es") == "Kim Min-su-ssi"

    def test_no_inventa_honorifico_coreano_en_frase_completa(self):
        text = "Kim Min-su viene mañana."
        assert _preservar_honorificos("김민수씨가 왔어요.", text, "ko", "es") == text

    def test_no_inventa_honorifico_en_frase_completa(self):
        text = "Tanaka viene mañana."
        assert _preservar_honorificos("田中さん、来て。", text, "ja", "es") == text

    def test_no_aplica_a_otro_idioma(self):
        assert _preservar_honorificos("田中さん", "Tanaka", "en", "es") == "Tanaka"


# ═══════════════════════════════════════════════════════════════
# _aplicar_glosario
# ═══════════════════════════════════════════════════════════════

class TestAplicarGlosario:
    def test_passes_through_traido_unchanged(self):
        # TRAIDO no tiene correccion directa en GLOSARIO_PRE (no hay patron "TRAIDO").
        # Verificar que pasa sin cambios.
        result = _aplicar_glosario("TRAIDO")
        assert result == "TRAIDO"

    def test_corrects_at_symbol_cinco(self):
        result = _aplicar_glosario("@NCO")
        assert result == "CINCO"

    def test_corrects_nco(self):
        result = _aplicar_glosario("NCO")
        assert result == "CINCO"

    def test_no_change_for_clean_text(self):
        result = _aplicar_glosario("Hola mundo")
        assert result == "Hola mundo"

    def test_empty_text(self):
        assert _aplicar_glosario("") == ""


# ═══════════════════════════════════════════════════════════════
# _es_ocr_noise
# ═══════════════════════════════════════════════════════════════

class TestEsOcrNoise:
    def test_high_digit_ratio(self):
        assert _es_ocr_noise("T2n2") is True
        assert _es_ocr_noise("GEAn374") is True

    def test_special_chars(self):
        assert _es_ocr_noise("@#$%^&") is True
        # '@' solo (1 char) no activa el patron {2,}, y no es >127 — no es ruido
        assert _es_ocr_noise("momms@") is False

    def test_weird_unicode(self):
        assert _es_ocr_noise("œŒ") is True

    def test_single_consonant(self):
        # Early return: len(t) < 2 sale antes de check 6, asi que 'Q' no es ruido
        assert _es_ocr_noise("Q") is False
        assert _es_ocr_noise("N") is False
        assert _es_ocr_noise("Z") is False

    def test_single_vowel_not_noise(self):
        assert _es_ocr_noise("A") is False
        assert _es_ocr_noise("I") is False
        assert _es_ocr_noise("Y") is False
        assert _es_ocr_noise("O") is False

    def test_short_no_vowel_no_digit_is_noise(self):
        # Textos de 2-3 chars sin vocal NI digito — SÍ son OCR noise
        # (caen en check 7: len<=3, no vowel, no digit, no allowed)
        assert _es_ocr_noise("kc") is True
        assert _es_ocr_noise("mn") is True
        assert _es_ocr_noise("zx") is True
        assert _es_ocr_noise("xyz") is True

    def test_short_abbreviation_not_noise(self):
        assert _es_ocr_noise("dr") is False
        assert _es_ocr_noise("mr") is False
        assert _es_ocr_noise("sr") is False
        assert _es_ocr_noise("st") is False

    def test_ordinal_with_digit_not_noise(self):
        # Ordinales con digito (4th, 3rd) — excluidos via regex check 0
        assert _es_ocr_noise("4th") is False
        assert _es_ocr_noise("3rd") is False
        assert _es_ocr_noise("1st") is False
        assert _es_ocr_noise("2nd") is False
        assert _es_ocr_noise("1024th") is False

    def test_clean_text_not_noise(self):
        assert _es_ocr_noise("Hola") is False
        assert _es_ocr_noise("Hello") is False
        assert _es_ocr_noise("INCREÍBLE") is False
        assert _es_ocr_noise("TEMPORADA 7") is False

    def test_manga_repetitions_not_noise(self):
        # Repeticiones estilísticas de manga como "NOOOOOO" no son ruido
        assert _es_ocr_noise("NOOOOOO") is False
        assert _es_ocr_noise("GRRRRRR") is False
        assert _es_ocr_noise("WHAAAAT") is False

    def test_empty_or_too_short(self):
        assert _es_ocr_noise("") is False
        assert _es_ocr_noise("X") is False  # Menos de 2 chars sale antes

    def test_pure_numeric(self):
        assert _es_ocr_noise("12345") is True
        assert _es_ocr_noise("3/128") is True
        assert _es_ocr_noise("42") is True

    def test_spanish_words_not_noise(self):
        assert _es_ocr_noise("el") is False
        assert _es_ocr_noise("la") is False
        assert _es_ocr_noise("y") is False

    # ── Casos límite: fechas ───────────────────────────────────
    def test_date_formats_are_noise(self):
        """Fechas como '13/7/26' tienen alta proporcion de digitos."""
        assert _es_ocr_noise("13/7/26") is True
        assert _es_ocr_noise("2026/07/13") is True
        assert _es_ocr_noise("07/13/2026") is True
        assert _es_ocr_noise("13-07-2026") is True

    # ── Casos límite: horas ───────────────────────────────────
    def test_time_formats_are_noise(self):
        """Horas como '4.58 p.m.' tienen alta proporcion de digitos."""
        assert _es_ocr_noise("4.58 p.m.") is True
        assert _es_ocr_noise("10:30 AM") is True
        assert _es_ocr_noise("11:45pm") is True
        assert _es_ocr_noise("16:30") is True

    # ── Casos límite: metadatos de escaneo ─────────────────────
    def test_scan_metadata_are_noise(self):
        """Metadatos de pagina como '3/128' (alta proporcion de digitos) son ruido."""
        assert _es_ocr_noise("3/128") is True
        assert _es_ocr_noise("12/128") is True
        assert _es_ocr_noise("128/128") is True

    def test_datetime_stamp_is_noise(self):
        """Sello combinado fecha+hora tipico de escaneos."""
        assert _es_ocr_noise("13/7/26, 4.58 p.m.") is True

    # ── Casos límite: URLs (no deberian llegar aqui, pero por si acaso) ─
    def test_urls_not_noise(self):
        """URLs sin doble barra ni digitos no son ruido (se filtran en _group_and_merge_blocks).
        Nota: 'https://' tiene // que activa check 3 (caracteres especiales).
        """
        assert _es_ocr_noise("www.olympusxyz.com") is False
        assert _es_ocr_noise("olympusxyz.com") is False

    def test_urls_https_is_noise_by_special_chars(self):
        """'https://' contiene // (2 barras) que activa el check de caracteres especiales.
        Esto es aceptable porque solo salta Argos; CT2 y Google siguen disponibles."""
        assert _es_ocr_noise("https://olympusxyz.com") is True

    def test_urls_with_digits_low_ratio_not_noise(self):
        """URL con digitos pero baja proporcion (<30%) y bastantes letras no es ruido."""
        assert _es_ocr_noise("olympusxyz.com/capitulo/130468") is False

    # ── Casos límite: numeros romanos (NO son ruido) ────────────
    def test_roman_numerals_not_noise(self):
        """Numeros romanos tienen vocales (I, V, X) y son texto valido."""
        assert _es_ocr_noise("IV") is False
        assert _es_ocr_noise("VII") is False
        assert _es_ocr_noise("XIII") is False
        assert _es_ocr_noise("MCMXCVIII") is False

    # ── Caso borde: porcentajes ────────────────────────────────
    def test_percentage_is_noise_by_digit_ratio(self):
        """'100%' tiene 75% digitos, se clasifica como ruido (aunque sea texto valido).
        Esto es aceptable porque solo hace que se salte Argos; CT2 y Google siguen."""
        assert _es_ocr_noise("100%") is True

    # ── Casos limite: codigos de scanlation ─────────────────────
    def test_scanlation_codes_are_noise(self):
        """Codigos como 'A-01' o 'B-12' mezclan letras y digitos."""
        assert _es_ocr_noise("A-01") is True
        assert _es_ocr_noise("B-12") is True
        assert _es_ocr_noise("SEC-03") is True


# ═══════════════════════════════════════════════════════════════
# _es_traduccion_valida
# ═══════════════════════════════════════════════════════════════

class TestEsTraduccionValida:
    def test_empty_translation_invalid(self):
        assert _es_traduccion_valida("hello", "") is False
        assert _es_traduccion_valida("hello", "   ") is False

    def test_same_text_invalid_when_not_lenient(self):
        assert _es_traduccion_valida("hello", "hello") is False

    def test_same_text_with_only_case_or_spaces_is_invalid(self):
        assert _es_traduccion_valida("Hello", " hello ") is False
        assert _es_traduccion_valida("HOLA", "hola") is False

    def test_lenient_accepts_similar_short_text(self):
        # El modo lenient acepta texto corto que difiere del original
        # (no relaja la validación de texto IDÉNTICO — ese siempre es rechazado)
        assert _es_traduccion_valida("hello", "Hi there", lenient=True) is True

    def test_repeated_chunk_invalid(self):
        assert _es_traduccion_valida("test", "mainstremainstre") is False
        assert _es_traduccion_valida("test", "power powerpower") is False

    def test_valid_translation(self):
        assert _es_traduccion_valida("hola", "hello") is True
        assert _es_traduccion_valida("adiós", "goodbye") is True
        assert _es_traduccion_valida("casa", "house") is True

    def test_too_many_characters_expansion(self):
        # Traducción > 8x el original
        orig = "hola"
        trad = "h" + "o" * 50
        assert _es_traduccion_valida(orig, trad) is False


class TestTranslationLanguageValidation:
    def test_rejects_confident_english_output_for_spanish_source(self):
        assert _translation_is_likely_source_language(
            "the house is big", "en", "es") is True

    def test_does_not_reject_short_name_or_loanword(self):
        assert _translation_is_likely_source_language(
            "Tanaka", "ja", "es") is False
        assert _translation_is_likely_source_language(
            "hello", "en", "es") is False

    def test_rejects_cjk_output_when_source_script_is_preserved(self):
        assert _translation_is_likely_source_language(
            "こんにちは世界", "ja", "es") is True

    def test_preserved_single_cjk_name_does_not_invalidate_translation(self):
        assert _translation_is_likely_source_language(
            "Hola 李", "zh", "es") is False


# ═══════════════════════════════════════════════════════════════
# _corregir_ct2
# ═══════════════════════════════════════════════════════════════

class TestCorregirCt2:
    def test_corrects_temporary_to_season(self):
        # CT2 produce "TEMPORARY 7" en vez de "SEASON 7"
        assert _corregir_ct2("TEMPORARY 7") == "SEASON 7"
        assert _corregir_ct2("TEMPORARILY 7") == "SEASON 7"
        assert _corregir_ct2("temporary 7") == "SEASON 7"

    def test_normalizes_scan_terms(self):
        assert _corregir_ct2("SCAN") == "scan"
        assert _corregir_ct2("SCANLATION") == "scanlation"

    def test_no_change_for_normal_text(self):
        result = _corregir_ct2("Hello world")
        assert result == "Hello world"

    def test_empty_text(self):
        assert _corregir_ct2("") == ""


# ═══════════════════════════════════════════════════════════════
# _translate_one (con mocks para dependencias externas)
# ═══════════════════════════════════════════════════════════════

# Fixture compartido: mockea el 80% del boilerplate común.
# Cada test sobreescribe solo lo que necesita diferente.


@pytest.fixture
def translate_mocks(mocker):
    """
    Configura los mocks comunes del pipeline _translate_one.
    Retorna un dict con los mocks clave para que cada test
    pueda sobreescribir valores específicos.

    Por defecto (pipeline secuencial CT2->Google->SIN_TRAD):
    - No es SFX, no es OCR noise
    - Idioma: español
    - Glosario: passthrough
    - CT2 devuelve "translated" (éxito rápido)
    - Google no se usa
    - sleep mockeado
    - Traducción válida
    """
    mocks = {}

    # ── Mocks de funciones sin side effects ──────────────────
    mocks["sfx"] = mocker.patch("translator._es_sfx", return_value=False)
    mocks["glosario"] = mocker.patch(
        "translator._aplicar_glosario", side_effect=lambda x: x
    )
    mocks["detect_lang"] = mocker.patch(
        "translator._detect_language_robust", return_value="es"
    )
    mocks["sleep"] = mocker.patch(
        "translator.time.sleep", return_value=None
    )

    # ── Mock del pipeline secuencial: CT2 → Google → SIN_TRAD ─
    # Por defecto: CT2 tiene éxito rápido
    mocks["ct2"] = mocker.patch(
        "translator._translate_ctranslate2", return_value="translated"
    )
    mocks["google"] = mocker.patch(
        "translator._translate_google", return_value=None
    )

    # ── Post-procesamiento: passthrough ──────────────────────
    mocks["post_process"] = mocker.patch(
        "translator._post_process_translation", side_effect=lambda x, s, t: x
    )
    mocks["valida"] = mocker.patch(
        "translator._es_traduccion_valida", return_value=True
    )

    return mocks


class TestTranslateOne:
    def test_empty_text_returns_empty(self):
        assert _translate_one("", "es", "en") == ""
        # Texto de solo espacios se limpia (strip) y queda vacio
        text = _translate_one("   ", "es", "en")
        assert text == ""

    def test_sfx_preserved(self):
        """SFX debe preservarse sin traducir."""
        result = _translate_one("BOOM", "es", "en")
        assert result == "BOOM"

    def test_sfx_preserva_marcadores_antes_de_limpiar_ocr(self):
        """Los marcadores *...* no deben desaparecer al preservar SFX."""
        result = _translate_one("*sigh*", "en", "es")

        assert result == "*sigh*"

    def test_dialogue_context_does_not_preserve_all_caps_as_sfx(self, translate_mocks):
        """Un diálogo en mayúsculas debe llegar al traductor aunque parezca SFX."""
        translate_mocks["sfx"].return_value = True

        result = _translate_one("NARUTO", "es", "en", block_type="dialogue")

        assert result != "NARUTO"
        translate_mocks["ct2"].assert_called_once()

    def test_short_spanish_text(self, translate_mocks):
        """Texto corto español debe traducirse."""
        # El fixture por defecto: CT2 devuelve "translated", es válido
        result = _translate_one("hola", "es", "en")
        assert result is not None
        assert result != ""

    def test_discards_translation_that_stays_in_source_language(
        self, translate_mocks
    ):
        translate_mocks["ct2"].return_value = "hola mundo"
        translate_mocks["google"].return_value = "hello world"

        result = _translate_one("hola mundo", "es", "en")

        assert result == "hello world"
        translate_mocks["ct2"].assert_called_once()
        translate_mocks["google"].assert_called_once()

    def test_auto_mixed_phrase_is_not_skipped_when_target_is_dominant(
        self, translate_mocks
    ):
        """Una frase inglesa incrustada debe poder traducirse a español."""
        translate_mocks["detect_lang"].return_value = "es"
        translate_mocks["google"].return_value = "No puedo ir a casa"

        result = _translate_one("No puedo go home", "auto", "es")

        assert result == "No puedo ir a casa"
        translate_mocks["google"].assert_called_once_with(
            "No puedo go home", "auto", "es"
        )
        translate_mocks["ct2"].assert_not_called()

    def test_auto_mixed_phrase_uses_auto_google_when_target_differs(self, translate_mocks):
        translate_mocks["detect_lang"].return_value = "es"
        translate_mocks["google"].return_value = "I cannot go home"

        result = _translate_one("No puedo go home", "auto", "en")

        assert result == "I cannot go home"
        translate_mocks["google"].assert_called_once_with(
            "No puedo go home", "auto", "en"
        )
        translate_mocks["ct2"].assert_not_called()

    def test_explicit_source_mixed_phrase_uses_auto_google_when_target_differs(
        self, translate_mocks
    ):
        """Un source fijado no debe enviar code-switching a CT2 monolingüe."""
        translate_mocks["google"].return_value = "I cannot go home"

        result = _translate_one("No puedo go home", "es", "en")

        assert result == "I cannot go home"
        translate_mocks["google"].assert_called_once_with(
            "No puedo go home", "auto", "en"
        )
        translate_mocks["ct2"].assert_not_called()

    def test_auto_mixed_google_source_language_is_rejected(self, translate_mocks):
        translate_mocks["detect_lang"].return_value = "es"
        translate_mocks["google"].return_value = "No puedo ir a casa"
        translate_mocks["ct2"].return_value = "I cannot go home"

        result = _translate_one("No puedo go home", "auto", "en")

        assert result == "I cannot go home"
        translate_mocks["google"].assert_called_once_with(
            "No puedo go home", "auto", "en"
        )
        translate_mocks["ct2"].assert_called_once()

    def test_explicit_dominant_language_still_handles_mixed_phrase(
        self, translate_mocks
    ):
        translate_mocks["google"].return_value = "No puedo ir a casa"

        result = _translate_one("No puedo go home", "es", "es")

        assert result == "No puedo ir a casa"
        translate_mocks["google"].assert_called_once_with(
            "No puedo go home", "auto", "es"
        )

    def test_ocr_noise_still_translated_by_ct2(self, translate_mocks):
        """Texto con ruido OCR debe traducirse via CT2 (no se salta traduccion).
        En el pipeline secuencial, CT2 procesa todo texto que no sea SFX."""
        translate_mocks["ct2"].return_value = "noise translated"

        result = _translate_one("Q7%zn2", "es", "en")
        assert result is not None

    def test_ocr_noise_sin_ct2_no_activa_google(self, translate_mocks):
        """La basura OCR no debe convertirse en una falsa traduccion de red."""
        translate_mocks["ct2"].return_value = None
        translate_mocks["google"].return_value = "noise translated"

        result = _translate_one("Q7%zn2", "es", "en")

        assert result == "Q7zn2"
        translate_mocks["google"].assert_not_called()

    def test_sin_trad_fallback_returns_original(self, translate_mocks):
        """Cuando CT2 y Google fallan, debe devolver el texto original (SIN_TRAD)."""
        # CT2 devuelve None (falla)
        translate_mocks["ct2"].return_value = None
        # Google devuelve None (falla)
        translate_mocks["google"].return_value = None

        result = _translate_one("Texto original", "es", "en")
        assert result == "Texto original"

    def test_japones_espanol_usa_pivote_ct2_si_google_falla(self, translate_mocks):
        """ja->es conserva un fallback offline sin degradar Google directo."""
        translate_mocks["ct2"].side_effect = lambda text, source, target: {
            ("ja", "es"): None,
            ("ja", "en"): "hello",
            ("en", "es"): "hola",
        }.get((source, target))
        translate_mocks["google"].return_value = None

        result = _translate_one("こんにちは", "ja", "es")

        assert result == "hola"
        assert translate_mocks["ct2"].call_args_list[0].args[1:] == ("ja", "es")
        assert translate_mocks["ct2"].call_args_list[-2].args[1:] == ("ja", "en")
        assert translate_mocks["ct2"].call_args_list[-1].args[1:] == ("en", "es")

    def test_cache_hit_tambien_preserva_honorifico(self, translate_mocks):
        translate_mocks["detect_lang"].return_value = "ja"
        translate_mocks["ct2"].return_value = None

        result = _translate_one(
            "田中さん", "ja", "es",
            cache_get=lambda *_args: "Tanaka",
            translation_cache_available=True,
        )

        assert result == "Tanaka-san"

    def test_cache_no_salta_gate_de_idioma_origen(self, translate_mocks):
        """Una entrada vieja sin traducir no debe contaminar páginas futuras."""
        translate_mocks["ct2"].return_value = "hello world"

        result = _translate_one(
            "hola mundo", "es", "en",
            cache_get=lambda *_args: "hola mundo",
            translation_cache_available=True,
        )

        assert result == "hello world"
        translate_mocks["ct2"].assert_called_once()

    def test_cache_hit_mixed_auto_avoids_network_fallback(self, translate_mocks):
        translate_mocks["detect_lang"].return_value = "es"

        result = _translate_one(
            "No puedo go home", "auto", "en",
            cache_get=lambda *_args: "I cannot go home",
            translation_cache_available=True,
        )

        assert result == "I cannot go home"
        translate_mocks["google"].assert_not_called()
        translate_mocks["ct2"].assert_not_called()


# ═══════════════════════════════════════════════════════════════
# Checksums SHA256 CT2 (_compute/_load/_save/_verify)
# ═══════════════════════════════════════════════════════════════

class TestCt2Checksums:
    def test_compute_sha256_estable(self, tmp_path: Path) -> None:
        f = tmp_path / "model.bin"
        f.write_bytes(b"datos" * 1000)
        h1 = _compute_file_sha256(str(f))
        h2 = _compute_file_sha256(str(f))
        assert h1 == h2
        assert len(h1) == 64  # hex SHA256
        # Cambiar contenido cambia el hash
        f.write_bytes(b"otros")
        assert _compute_file_sha256(str(f)) != h1

    def test_load_sin_archivo_devuelve_vacio(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(
            "translator._CT2_CHECKSUMS_FILE",
            str(tmp_path / "no_existe.json"),
        )
        assert _load_ct2_checksums() == {}

    def test_load_json_corrupto_devuelve_vacio(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        f = tmp_path / "checksums.json"
        f.write_text("{no-json", encoding="utf-8")
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(f))
        assert _load_ct2_checksums() == {}

    def test_save_escribe_checksums_y_load_los_recupera(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        csum_file = tmp_path / "ct2" / "checksums.json"
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"AAA")
        (model_dir / "config.json").write_bytes(b"{}")
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))

        _save_ct2_checksums("es|en", str(model_dir))

        loaded = _load_ct2_checksums()
        assert "es|en" in loaded
        assert loaded["es|en"]["model.bin"] == _compute_file_sha256(
            str(model_dir / "model.bin"))
        assert loaded["es|en"]["config.json"] == _compute_file_sha256(
            str(model_dir / "config.json"))

    def test_save_sin_directorio_no_escribe(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        csum_file = tmp_path / "checksums.json"
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))
        _save_ct2_checksums("es|en", str(tmp_path / "no_existe"))
        assert not csum_file.exists()

    def test_verify_primera_vez_genera_checksums(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        csum_file = tmp_path / "checksums.json"
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"AAA")
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))

        assert _verify_ct2_checksums("es|en", str(model_dir)) is True
        # Generó checksums para futuras verificaciones
        assert "es|en" in _load_ct2_checksums()

    def test_verify_checksum_match_pasa(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        csum_file = tmp_path / "checksums.json"
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"AAA")
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))

        _save_ct2_checksums("es|en", str(model_dir))
        assert _verify_ct2_checksums("es|en", str(model_dir)) is True

    def test_verify_archivo_modificado_falla(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        csum_file = tmp_path / "checksums.json"
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        archivo = model_dir / "model.bin"
        archivo.write_bytes(b"AAA")
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))

        _save_ct2_checksums("es|en", str(model_dir))
        archivo.write_bytes(b"BBB")  # manipulado
        assert _verify_ct2_checksums("es|en", str(model_dir)) is False

    def test_verify_archivo_faltante_falla(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        csum_file = tmp_path / "checksums.json"
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.bin").write_bytes(b"AAA")
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))

        _save_ct2_checksums("es|en", str(model_dir))
        (model_dir / "model.bin").unlink()
        assert _verify_ct2_checksums("es|en", str(model_dir)) is False


# ═══════════════════════════════════════════════════════════════
# _get_ct2_translator / _translate_ctranslate2
# ═══════════════════════════════════════════════════════════════

class _FakeCt2Translator:
    def __init__(self, model_dir: str, device: str) -> None:
        self.model_dir = model_dir
        self.device = device

    def translate_batch(self, batch: object, **kw: object) -> list[object]:
        class _Hyp:
            hypotheses = [["hola"] * 5]

        return [_Hyp()]


class _FakeTokenizer:
    def __init__(self, text: str = "hola mundo") -> None:
        self._text = text

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]

    def convert_ids_to_tokens(self, ids: object) -> list[str]:
        return ["hola", "mundo"]

    def convert_tokens_to_ids(self, tokens: object) -> list[int]:
        return [1, 2]

    def decode(self, ids: object) -> str:
        return self._text


@pytest.fixture(autouse=True)
def _ct2_state_clean() -> Iterator[None]:
    """Aísla el estado global de CT2 entre tests (translators/tokenizers)."""
    _ct2_translators.clear()
    _ct2_tokenizers.clear()
    yield
    _ct2_translators.clear()
    _ct2_tokenizers.clear()


class TestGetCt2Translator:
    def test_par_no_soportado_devuelve_none(self) -> None:
        assert _get_ct2_translator("xx", "yy") == (None, None)

    def test_fast_path_devuelve_cargado(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = _FakeCt2Translator("x", "cpu")
        tok = _FakeTokenizer()
        _ct2_translators["es|en"] = t
        _ct2_tokenizers["es|en"] = tok
        assert _get_ct2_translator("es", "en") == (t, tok)

    def test_carga_completa_con_conversion(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        base = tmp_path / "ct2"
        csum_file = tmp_path / "checksums.json"
        monkeypatch.setattr("translator._CT2_BASE_DIR", str(base))
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))

        class _FakeConverter:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name

            def convert(self, out_dir: str, **kw: object) -> None:
                Path(out_dir).mkdir(parents=True)
                (Path(out_dir) / "model.bin").write_bytes(b"MODEL")
                (Path(out_dir) / ".ct2_conversion_ok").write_text("ok",
                                                                    encoding="utf-8")

        class _FakeCt2Module:
            Translator = _FakeCt2Translator

            class converters:
                TransformersConverter = _FakeConverter

        class _FakeCuda:
            @staticmethod
            def is_available() -> bool:
                return False

        class _FakeTorch:
            cuda = _FakeCuda()

        class _FakeAutoTokenizer:
            @classmethod
            def from_pretrained(cls, *a: object, **k: object) -> _FakeTokenizer:
                return _FakeTokenizer()

        class _FakeTransformers:
            AutoTokenizer = _FakeAutoTokenizer

        monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCt2Module())
        monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
        monkeypatch.setitem(sys.modules, "transformers", _FakeTransformers())
        monkeypatch.setitem(sys.modules, "ctranslate2.converters",
                            _FakeCt2Module.converters)

        translator_obj, tokenizer = _get_ct2_translator("es", "en")

        assert translator_obj is not None
        assert tokenizer is not None
        assert translator_obj.device == "cpu"
        assert isinstance(tokenizer, _FakeTokenizer)
        # Checksums generados y verificados (primera vez OK)
        assert "es|en" in _load_ct2_checksums()

    def test_checksum_fail_rechaza_carga(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        base = tmp_path / "ct2"
        csum_file = tmp_path / "checksums.json"
        model_dir = base / "es-en"
        model_dir.mkdir(parents=True)
        (model_dir / "model.bin").write_bytes(b"AAA")
        # Checksums previos con hash distinto -> mismatch
        (csum_file.parent).mkdir(parents=True, exist_ok=True)
        csum_file.write_text(
            '{"es|en": {"model.bin": "' + "0" * 64 + '"}}',
            encoding="utf-8",
        )
        monkeypatch.setattr("translator._CT2_BASE_DIR", str(base))
        monkeypatch.setattr("translator._CT2_CHECKSUMS_FILE", str(csum_file))
        # Sentinel presente -> no reconvierte, va directo a verificación
        (model_dir / ".ct2_conversion_ok").write_text("ok", encoding="utf-8")

        assert _get_ct2_translator("es", "en") == (None, None)

    def test_error_de_carga_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        base = tmp_path / "ct2"
        monkeypatch.setattr("translator._CT2_BASE_DIR", str(base))

        class _BoomConverter:
            def __init__(self, model_name: str) -> None:
                raise RuntimeError("sin red")

        class _FakeCt2Module:
            Translator = _FakeCt2Translator

            class converters:
                TransformersConverter = _BoomConverter

        monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCt2Module())
        monkeypatch.setitem(sys.modules, "ctranslate2.converters",
                            _FakeCt2Module.converters)

        assert _get_ct2_translator("es", "en") == (None, None)


class TestTranslateCtranslate2:
    def test_sin_modelo_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("translator._get_ct2_translator",
                            lambda s, t: (None, None))
        assert _translate_ctranslate2("hola", "es", "en") is None

    def test_traduccion_ok_y_correccion_ct2(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        t = _FakeCt2Translator("x", "cpu")
        tok = _FakeTokenizer()
        monkeypatch.setattr("translator._get_ct2_translator",
                            lambda s, tgt: (t, tok))
        result = _translate_ctranslate2("hola", "es", "en")
        assert result is not None
        assert "hola" in result  # decode del fake + _corregir_ct2

    def test_batch_traduce_todos_en_una_llamada(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Optimización 2.6: translate_batch recibe TODOS los textos de la
        página en una sola llamada, con greedy (beam_size=1)."""
        from translator import _translate_ctranslate2_batch

        captured: dict[str, object] = {}

        class _BatchCt2Translator:
            def translate_batch(
                    self, batch: list[list[str]], **kw: object) -> list[object]:
                captured["batch"] = batch
                captured["kw"] = kw

                class _Hyp:
                    hypotheses = [["ok"] * 3]

                return [_Hyp() for _ in batch]

        t = _BatchCt2Translator()
        tok = _FakeTokenizer()
        monkeypatch.setattr("translator._get_ct2_translator",
                            lambda s, tgt: (t, tok))

        results = _translate_ctranslate2_batch(
            ["hola", "mundo", "  "], "es", "en")

        # Una sola llamada con los 2 textos no vacíos; beam_size=1 (greedy).
        assert len(captured["batch"]) == 2  # type: ignore[arg-type]
        assert captured["kw"].get("beam_size") == 1
        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is None  # texto vacío → None, sin llamar al batch

    def test_batch_sin_modelo_devuelve_nones(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from translator import _translate_ctranslate2_batch
        monkeypatch.setattr("translator._get_ct2_translator",
                            lambda s, tgt: (None, None))
        results = _translate_ctranslate2_batch(["hola", "mundo"], "es", "en")
        assert results == [None, None]

    def test_batch_vacio_devuelve_nones_sin_llamar(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        from translator import _translate_ctranslate2_batch
        monkeypatch.setattr("translator._get_ct2_translator",
                            lambda s, tgt: (_FakeCt2Translator("x", "cpu"),
                                            _FakeTokenizer()))
        assert _translate_ctranslate2_batch([], "es", "en") == []

    def test_error_traduccion_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _BoomTranslator:
            def translate_batch(self, batch: object, **kw: object) -> object:
                raise RuntimeError("fallo")

        monkeypatch.setattr("translator._get_ct2_translator",
                            lambda s, t: (_BoomTranslator(), _FakeTokenizer()))
        from translator import _translate_ctranslate2_batch
        assert _translate_ctranslate2("hola", "es", "en") is None
        assert _translate_ctranslate2_batch(["hola"], "es", "en") == [None]


# ═══════════════════════════════════════════════════════════════
# _translate_argos
# ═══════════════════════════════════════════════════════════════

class TestTranslateArgos:
    def test_sin_argostranslate_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "argostranslate", None)
        assert _translate_argos("hola", "es", "en") is None

    def test_paquete_no_disponible_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("translator._ensure_argo_package",
                            lambda s, t: False)
        assert _translate_argos("hola", "es", "en") is None

    def test_traduccion_ok(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Lang:
            def __init__(self, code: str) -> None:
                self.code = code

            def get_translation(self, other: object) -> object:
                class _T:
                    def translate(self, text: str) -> str:
                        return "hello"

                return _T()

        class _Translate:
            @staticmethod
            def get_installed_languages() -> list[object]:
                return [_Lang("es"), _Lang("en")]

        class _FakeArgo:
            translate = _Translate()

        monkeypatch.setattr("translator._ensure_argo_package",
                            lambda s, t: True)
        monkeypatch.setitem(sys.modules, "argostranslate", _FakeArgo())
        monkeypatch.setitem(sys.modules, "argostranslate.translate",
                            _Translate())

        assert _translate_argos("hola", "es", "en") == "hello"

    def test_error_traduccion_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Translate:
            @staticmethod
            def get_installed_languages() -> object:
                raise RuntimeError("file lock")

        class _FakeArgo:
            translate = _Translate()

        monkeypatch.setattr("translator._ensure_argo_package",
                            lambda s, t: True)
        monkeypatch.setitem(sys.modules, "argostranslate", _FakeArgo())

        assert _translate_argos("hola", "es", "en") is None


# ═══════════════════════════════════════════════════════════════
# Google session + rate limiting (_get_google_session/_translate_google)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _google_state_clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Resetea el estado global de Google entre tests (sesión, cache, rate limit)."""
    monkeypatch.setattr("translator._google_session", None)
    _translators.clear()
    _google_rate_limit_state.update({
        "consecutive_unchanged": 0,
        "backoff_until": 0.0,
        "current_backoff": 10.0,
    })
    yield
    _translators.clear()
    monkeypatch.setattr("translator._google_session", None)


class TestGetGoogleSession:
    def test_crea_sesion_con_timeout(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class _FakeSession:
            def __init__(self) -> None:
                captured["timeout"] = None
                self.request = self._request

            def _request(self, *args: object, **kwargs: object) -> object:
                captured["timeout"] = kwargs.get("timeout")
                return object()

        class _FakeRequests:
            @staticmethod
            def Session() -> _FakeSession:
                return _FakeSession()

        monkeypatch.setitem(sys.modules, "requests", _FakeRequests())
        session = _get_google_session()
        assert session is not None
        # La sesión se reutiliza (cacheada global)
        assert _get_google_session() is session

    def test_sesion_cacheada_sin_recrear(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[int] = []

        class _FakeSession:
            def __init__(self) -> None:
                self.request = self._request

            def _request(self, *args: object, **kwargs: object) -> object:
                return object()

        class _FakeRequests:
            @staticmethod
            def Session() -> _FakeSession:
                calls.append(1)
                return _FakeSession()

        monkeypatch.setitem(sys.modules, "requests", _FakeRequests())
        s1 = _get_google_session()
        s2 = _get_google_session()
        assert s1 is s2
        assert len(calls) == 1


class TestTranslateGoogle:
    def test_backoff_activo_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        _google_rate_limit_state["backoff_until"] = time.time() + 60
        assert _translate_google("hola", "es", "en") is None

    def test_exito_resetea_contador(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        _google_rate_limit_state["consecutive_unchanged"] = 1

        class _FakeTranslator:
            def __init__(self, source: str, target: str) -> None:
                pass

            def translate(self, text: str) -> str:
                return "hello"

        class _FakeDeepTranslator:
            GoogleTranslator = _FakeTranslator

        monkeypatch.setitem(sys.modules, "deep_translator", _FakeDeepTranslator())

        result = _translate_google("hola", "es", "en")
        assert result == "hello"
        assert _google_rate_limit_state["consecutive_unchanged"] == 0
        assert _google_rate_limit_state["current_backoff"] == 60.0

    def test_texto_sin_cambios_acumula_y_devuelve(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeTranslator:
            def __init__(self, source: str, target: str) -> None:
                pass

            def translate(self, text: str) -> str:
                return text  # sin cambios (rate limit)

        class _FakeDeepTranslator:
            GoogleTranslator = _FakeTranslator

        monkeypatch.setitem(sys.modules, "deep_translator", _FakeDeepTranslator())

        result = _translate_google("hola", "es", "en")
        assert result == "hola"  # devuelve igual; _resultado_valido lo rechaza
        assert _google_rate_limit_state["consecutive_unchanged"] == 1

    def test_rate_limit_detectado_activa_backoff(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeTranslator:
            def __init__(self, source: str, target: str) -> None:
                pass

            def translate(self, text: str) -> str:
                return text

        class _FakeDeepTranslator:
            GoogleTranslator = _FakeTranslator

        monkeypatch.setitem(sys.modules, "deep_translator", _FakeDeepTranslator())

        # Llamar _RATE_LIMIT_THRESHOLD veces seguidas sin cambios
        for _ in range(_RATE_LIMIT_THRESHOLD):
            _translate_google("hola", "es", "en")

        assert _google_rate_limit_state["backoff_until"] > time.time()
        assert _google_rate_limit_state["current_backoff"] == 20.0  # 10*2

    def test_error_429_activa_backoff(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeTranslator:
            def __init__(self, source: str, target: str) -> None:
                pass

            def translate(self, text: str) -> str:
                raise Exception("429 Too Many Requests")

        class _FakeDeepTranslator:
            GoogleTranslator = _FakeTranslator

        monkeypatch.setitem(sys.modules, "deep_translator", _FakeDeepTranslator())

        for _ in range(_RATE_LIMIT_THRESHOLD):
            _translate_google("hola", "es", "en")

        assert _google_rate_limit_state["backoff_until"] > time.time()

    def test_error_generico_devuelve_none(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeTranslator:
            def __init__(self, source: str, target: str) -> None:
                pass

            def translate(self, text: str) -> str:
                raise Exception("connection reset")

        class _FakeDeepTranslator:
            GoogleTranslator = _FakeTranslator

        monkeypatch.setitem(sys.modules, "deep_translator", _FakeDeepTranslator())

        assert _translate_google("hola", "es", "en") is None

    def test_backoff_exponencial_se_duplica(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeTranslator:
            def __init__(self, source: str, target: str) -> None:
                pass

            def translate(self, text: str) -> str:
                return text

        class _FakeDeepTranslator:
            GoogleTranslator = _FakeTranslator

        monkeypatch.setitem(sys.modules, "deep_translator", _FakeDeepTranslator())

        # Primera ronda: 10 -> 20
        for _ in range(_RATE_LIMIT_THRESHOLD):
            _translate_google("hola", "es", "en")
        assert _google_rate_limit_state["current_backoff"] == 20.0

        # Segunda ronda (backoff expirado manualmente): 20 -> 40
        _google_rate_limit_state["backoff_until"] = 0.0
        _google_rate_limit_state["consecutive_unchanged"] = 0
        for _ in range(_RATE_LIMIT_THRESHOLD):
            _translate_google("hola", "es", "en")
        assert _google_rate_limit_state["current_backoff"] == 40.0


# ═══════════════════════════════════════════════════════════════
# _ensure_argo_package (ruta con paquetes ya instalados)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(autouse=True)
def _argo_state_clean() -> Iterator[None]:
    """La caché _argo_ready persiste entre tests; se limpia para aislarlos."""
    _argo_ready.clear()
    yield
    _argo_ready.clear()


class TestEnsureArgoPackage:
    def test_pares_ya_instalados_devuelve_true(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Lang:
            def __init__(self, code: str) -> None:
                self.code = code

        class _Translate:
            @staticmethod
            def get_installed_languages() -> list[object]:
                return [_Lang("es"), _Lang("en")]

        class _Package:
            pass

        class _FakeArgo:
            translate = _Translate()
            package = _Package()

        monkeypatch.setitem(sys.modules, "argostranslate", _FakeArgo())
        monkeypatch.setitem(sys.modules, "argostranslate.translate", _Translate())
        monkeypatch.setitem(sys.modules, "argostranslate.package", _Package())

        assert _ensure_argo_package("es", "en") is True
        # Caché: segunda llamada no vuelve a consultar
        assert _ensure_argo_package("es", "en") is True

    def test_paquete_faltante_devuelve_false(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Lang:
            def __init__(self, code: str) -> None:
                self.code = code

        class _Translate:
            @staticmethod
            def get_installed_languages() -> list[object]:
                return [_Lang("fr")]

        class _Package:
            @staticmethod
            def update_package_index() -> None:
                return None

            @staticmethod
            def get_available_packages() -> list[object]:
                return []

        class _FakeArgo:
            translate = _Translate()
            package = _Package()

        monkeypatch.setitem(sys.modules, "argostranslate", _FakeArgo())
        monkeypatch.setitem(sys.modules, "argostranslate.translate", _Translate())
        monkeypatch.setitem(sys.modules, "argostranslate.package", _Package())

        # Sin red ni paquetes disponibles, no debe instalar
        assert _ensure_argo_package("es", "en") is False
