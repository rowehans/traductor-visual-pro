"""Pruebas de memoria automÃ¡tica de traducciÃ³n por documento."""

from translation_memory import DocumentTranslationMemory


def test_memoria_reutiliza_traduccion_estable_del_mismo_documento():
    memory = DocumentTranslationMemory(doc_id="capitulo-1", persist=False)
    memory.learn("田中さん", "Tanaka-san", "ja", "es", quality=0.92)

    assert memory.lookup(" 田中さん ", "ja", "es") == "Tanaka-san"


def test_memoria_no_aprende_resultado_de_baja_calidad():
    memory = DocumentTranslationMemory(doc_id="capitulo-1", persist=False)
    memory.learn("Yuki", "Yuki raro", "ja", "es", quality=0.40)

    assert memory.lookup("Yuki", "ja", "es") is None


def test_conflicto_de_traduccion_no_sobrescribe_sin_mejora_clara():
    memory = DocumentTranslationMemory(doc_id="capitulo-1", persist=False)
    memory.learn("先生", "sensei", "ja", "es", quality=0.82)
    memory.learn("先生", "profesor", "ja", "es", quality=0.84)

    assert memory.lookup("先生", "ja", "es") == "sensei"


def test_mejor_calidad_puede_reemplazar_una_entrada_debil():
    memory = DocumentTranslationMemory(doc_id="capitulo-1", persist=False)
    memory.learn("先生", "profesor", "ja", "es", quality=0.70)
    memory.learn("先生", "sensei", "ja", "es", quality=0.94)

    assert memory.lookup("先生", "ja", "es") == "sensei"


def test_memoria_persistida_se_recarga_sin_glosario_manual(tmp_path):
    first = DocumentTranslationMemory(
        doc_id="capitulo-1", storage_dir=tmp_path)
    first.learn("黒崎", "Kurosaki", "ja", "es", quality=0.90)
    first.save()

    second = DocumentTranslationMemory(
        doc_id="capitulo-1", storage_dir=tmp_path)

    assert second.lookup("黒崎", "ja", "es") == "Kurosaki"


def test_memoria_corrige_variante_cjk_de_un_caracter_si_el_term_se_repitio():
    """Un error OCR de un glifo no rompe un nombre ya estabilizado."""
    memory = DocumentTranslationMemory(doc_id="capitulo-variantes", persist=False)
    for _ in range(2):
        memory.learn("田中さん", "Tanaka-san", "ja", "es", quality=0.96)

    assert memory.lookup_variant("田中さ人", "ja", "es") == "Tanaka-san"


def test_memoria_no_elige_variante_ambigua():
    """Si dos nombres son igual de cercanos, no inventa una identidad."""
    memory = DocumentTranslationMemory(doc_id="capitulo-ambiguo", persist=False)
    for _ in range(2):
        memory.learn("田中", "Tanaka", "ja", "es", quality=0.96)
        memory.learn("田口", "Taguchi", "ja", "es", quality=0.96)

    assert memory.lookup_variant("田子", "ja", "es") is None
