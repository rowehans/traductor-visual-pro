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
import pytest

# Asegurar que el proyecto esté en sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from translator import (
    _detect_language_simple,
    _detect_language_robust,
    _es_sfx,
    _post_process_translation,
    _aplicar_glosario,
    _es_ocr_noise,
    _es_traduccion_valida,
    _corregir_ct2,
    _translate_one,
    _GLOSARIO_POST_REGEX,
    _SFX_PATTERNS,
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

    def test_preserves_sfx(self):
        result = _post_process_translation("BANG!", "es", "en")
        assert result == "BANG!"

    def test_adds_ellipsis_to_short_dialogue(self):
        result = _post_process_translation("Hello there", "es", "en")
        assert result == "Hello there..."

    def test_adds_period_to_long_sentences(self):
        text = "This is a very long sentence that has more than eight words in total here"
        result = _post_process_translation(text, "es", "en")
        assert result.endswith(".")

    def test_normalizes_spaces(self):
        result = _post_process_translation("hello    world", "es", "en")
        assert "  " not in result

    def test_handles_empty_text(self):
        assert _post_process_translation("", "es", "en") == ""
        assert _post_process_translation(None, "es", "en") is None

    def test_doesnt_add_punctuation_to_single_caps_word(self):
        # Ej: nombre propio "SEOLLANG" en mayúsculas
        result = _post_process_translation("SEOLLANG", "es", "en")
        assert result == "SEOLLANG"

    def test_target_language_spanish(self):
        result = _post_process_translation("hello", "en", "es")
        assert "..." in result or result == "hello..."

    def test_preserves_existing_punctuation(self):
        result = _post_process_translation("Hello!", "es", "en")
        assert result.rstrip(".") == "Hello!"


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

    def test_same_text_valid_when_lenient(self):
        assert _es_traduccion_valida("hello", "hello", lenient=True) is True

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

    Por defecto:
    - No es SFX, no es OCR noise
    - Idioma: español
    - Glosario: passthrough
    - Executor mockeado con un solo future que retorna ("ctranslate2", "translated")
    - sleep mockeado para evitar backoff real de 50s
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
    mocks["ocr_noise"] = mocker.patch(
        "translator._es_ocr_noise", return_value=False
    )
    mocks["sleep"] = mocker.patch(
        "translator.time.sleep", return_value=None
    )

    # ── Mock del executor + future ────────────────────────────
    mock_future = mocker.MagicMock()
    mock_future.result.return_value = ("ctranslate2", "translated")
    mocks["future"] = mock_future

    mock_executor = mocker.MagicMock()
    mock_executor.submit.return_value = mock_future
    mocks["executor"] = mock_executor

    def _as_completed(fut_map, timeout=None):
        yield mock_future

    mocks["as_completed"] = mocker.patch(
        "translator.concurrent.futures.as_completed", _as_completed
    )
    mocks["get_executor"] = mocker.patch(
        "translator._get_translate_engine_executor", return_value=mock_executor
    )

    # ── Post-procesamiento: passthrough ──────────────────────
    mocks["post_process"] = mocker.patch(
        "translator._post_process_translation", side_effect=lambda x, s, t: x
    )
    mocks["valida"] = mocker.patch(
        "translator._es_traduccion_valida", return_value=True
    )

    # Google Translate mockeado por defecto (se sobreescribe si se necesita)
    mocks["google"] = mocker.patch(
        "translator._translate_google", return_value=None
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

    def test_short_spanish_text(self, translate_mocks):
        """Texto corto español debe traducirse."""
        # El fixture por defecto: CT2 devuelve "translated", es válido
        result = _translate_one("hola", "es", "en")
        assert result is not None
        assert result != ""

    def test_ocr_noise_skips_argos(self, translate_mocks):
        """Texto con ruido OCR debe saltarse Argos y usar solo CT2+Google.
        El fixture mockea _es_ocr_noise → False por defecto; lo sobreescribimos."""
        translate_mocks["ocr_noise"].return_value = True  # activar OCR noise
        translate_mocks["future"].result.return_value = ("ctranslate2", "noise result")

        result = _translate_one("Q7%zn2", "es", "en")
        assert result is not None

    def test_sin_trad_fallback_returns_original(self, translate_mocks):
        """Cuando todos los motores fallan, debe devolver el texto original (SIN_TRAD)."""
        # Todos los motores devuelven None
        translate_mocks["future"].result.return_value = ("ctranslate2", None)
        translate_mocks["valida"].return_value = False  # nada pasa validación

        result = _translate_one("Texto original", "es", "en")
        assert result == "Texto original"
