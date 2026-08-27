"""tests/test_baseline_calidad.py — Tests del baseline de calidad estable (P2).

Verifica que `baseline_calidad.compute_baseline` separe correctamente los
pares traducibles de los no-traducibles (SFX/OCR garbage/nombres no deben
castigar la tasa efectiva) y que `--compare` reporte el delta.
"""

import json

import baseline_calidad as bc


def _checkpoint(*items):
    return {
        "source_lang": "es",
        "target_lang": "en",
        "total_pages": 1,
        "results": [{"page": 1, "texts": list(items)}],
    }


def test_tasa_efectiva_excluye_no_traducibles():
    """Los SFX preservados y el OCR basura NO deben castigar la tasa
    efectiva — son comportamiento correcto/ruido de fuente, no fallos de
    traducción."""
    cp = _checkpoint(
        {"src": "¡Hola!", "tgt": "Hello!", "type": "dialogue"},
        {"src": "GOLPE", "tgt": "GOLPE", "type": "sfx"},
        {"src": "KJ7##!!", "tgt": "", "type": "text"},  # OCR garbage
    )
    base = bc.compute_baseline(cp, source_lang="es", target_lang="en")
    # 1 traducible (el diálogo) → tasa efectiva = 100 %
    assert base["corpus"]["traducibles"] == 1
    assert base["corpus"]["no_traducibles"] == 2
    assert base["tasa_efectiva"] == 100.0
    # La tasa global (métrica legada) sí incluye todo:
    assert base["tasa_global"] < 100.0


def test_fallos_reales_se_desglosan():
    """UNTRANSLATED / REVIEW_LANGUAGE / BAD_TRANSLATION se cuentan como
    fallos reales y aparecen en el desglose."""
    cp = _checkpoint(
        {"src": "Hola", "tgt": "Hola", "type": "dialogue"},       # UNTRANSLATED
        {"src": "Perro", "tgt": "xyzzy", "type": "dialogue"},     # REVIEW_LANGUAGE
    )
    base = bc.compute_baseline(cp, source_lang="es", target_lang="en")
    assert base["fallos"] == 2
    assert base["fallos_detalle"]["UNTRANSLATED"] == 1
    assert base["fallos_detalle"]["REVIEW_LANGUAGE"] == 1
    assert base["tasa_efectiva"] == 0.0


def test_guardar_y_comparar_delta(tmp_path, monkeypatch, capsys):
    """--save escribe el JSON y --compare muestra el delta contra el previo."""
    cp = _checkpoint({"src": "Hola", "tgt": "Hello", "type": "dialogue"})
    path = tmp_path / "baseline.json"

    rc = bc.main(["--input", str(tmp_path / "cp.json"), "--save", str(path)])
    assert rc == 2  # corpus no existe → error controlado

    cp_path = tmp_path / "cp.json"
    cp_path.write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")
    rc = bc.main(["--input", str(cp_path), "--save", str(path)])
    assert rc == 0
    assert path.is_file()
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["tasa_efectiva"] == 100.0

    # Mismo corpus → delta 0
    rc = bc.main(["--input", str(cp_path), "--save", str(path), "--compare"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "+0.00 pp" in out

    # Corpus degradado → delta negativo visible
    cp2_path = tmp_path / "cp2.json"
    cp2 = _checkpoint(
        {"src": "Hola", "tgt": "Hello", "type": "dialogue"},
        {"src": "Adiós", "tgt": "Adiós", "type": "dialogue"},  # UNTRANSLATED
    )
    cp2_path.write_text(json.dumps(cp2, ensure_ascii=False), encoding="utf-8")
    rc = bc.main(["--input", str(cp2_path), "--save", str(path), "--compare"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "-50.00 pp" in out or "50.0" in out
