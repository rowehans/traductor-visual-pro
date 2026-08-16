"""Pruebas del evaluador de calidad multilingüe."""

from quality_analysis import analyze_checkpoint, classify_pair
from analisis_calidad import main as quality_cli


def test_classifies_preserved_sfx_as_accepted_without_translation():
    result = classify_pair(
        "ドン", "ドン", source_lang="ja", target_lang="es", block_type="sfx"
    )

    assert result.category == "SFX_PRESERVED"
    assert result.accepted is True


def test_same_cjk_text_without_semantic_type_is_untranslated():
    result = classify_pair(
        "こんにちは", "こんにちは", source_lang="ja", target_lang="es"
    )

    assert result.category == "UNTRANSLATED"
    assert result.accepted is False


def test_explicit_name_can_be_preserved_without_glossary():
    result = classify_pair(
        "田中", "田中", source_lang="ja", target_lang="es", block_type="name"
    )

    assert result.category == "NAME_PRESERVED"
    assert result.accepted is True


def test_same_text_in_target_language_is_not_counted_as_failure():
    result = classify_pair(
        "Hola mundo", "Hola mundo", source_lang="es", target_lang="es"
    )

    assert result.category == "ALREADY_TARGET"
    assert result.accepted is True


def test_report_groups_by_language_pair_and_exposes_metadata_coverage():
    report = analyze_checkpoint({
        "source_lang": "ja",
        "target_lang": "es",
        "results": [{
            "page": 1,
            "texts": [
                {"src": "こんにちは", "tgt": "Hola", "type": "dialogue"},
                {"src": "ドン", "tgt": "ドン", "type": "sfx"},
            ],
        }],
    })

    assert report.total == 2
    assert report.metadata_coverage == 1.0
    assert report.by_pair["ja→es"].total == 2
    assert report.by_pair["ja→es"].accepted == 2


def test_cli_accepts_explicit_language_pair(tmp_path, capsys):
    path = tmp_path / "checkpoint.json"
    path.write_text(
        '{"results": [{"texts": [{"src": "こんにちは", "tgt": "Hola"}]}]}',
        encoding="utf-8",
    )

    assert quality_cli(["--input", str(path), "--source", "ja", "--target", "es"]) == 0
    assert "ja→es" in capsys.readouterr().out


def test_recognizes_non_latin_target_scripts_without_language_specific_words():
    result = classify_pair(
        "Hello", "Привет", source_lang="en", target_lang="ru"
    )

    assert result.category == "GOOD_TRANSLATION"
    assert result.accepted is True


def test_report_detects_inconsistent_repeated_translation_across_pages():
    report = analyze_checkpoint({
        "source_lang": "ja",
        "target_lang": "es",
        "results": [
            {"page": 1, "texts": [{"src": "田中さん", "tgt": "Tanaka-san", "type": "dialogue"}]},
            {"page": 2, "texts": [{"src": "田中さん", "tgt": "Tanaka", "type": "dialogue"}]},
        ],
    })

    assert report.consistency_conflict_count == 1
