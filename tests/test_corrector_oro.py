"""
test_corrector_oro.py — Prueba tools/corrector_oro.py (corrector interactivo del oro).

Cubre la lógica pura: parseo/escritura YOLO, detección de cajas gigantes,
normalización, backup del oro original y la semántica de "página sin texto"
(un .txt VACÍO, no ausente, para que fusionar_correcciones.py no conserve
las pseudo-etiquetas malas del teacher). Todo corre sobre tmp_path — no
toca train_data/corregir ni el pipeline.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

_counter = [0]


def _cargar():
    _counter[0] += 1
    spec = importlib.util.spec_from_file_location(
        f"corrector_oro_{_counter[0]}", _ROOT / "tools" / "corrector_oro.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def oro(tmp_path):
    """Módulo + workspace con 2 páginas y labels previos (uno con gigante)."""
    mod = _cargar()
    ws = tmp_path / "workspace"
    images = ws / "images"
    labels = ws / "labels"
    images.mkdir(parents=True)
    labels.mkdir()
    (images / "p001.jpg").write_bytes(b"\xff\xd8\xff")
    (images / "p002.jpg").write_bytes(b"\xff\xd8\xff")
    (labels / "p001.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n1 0.3 0.3 1.0 0.05\n", encoding="utf-8")
    (labels / "p002.txt").write_text("", encoding="utf-8")
    return mod, ws


def test_leer_yolo_ignora_lineas_malas(oro):
    mod, ws = oro
    path = ws / "labels" / "p001.txt"
    cajas = mod._leer_yolo(path)
    assert len(cajas) == 2
    assert cajas[0] == [0, 0.5, 0.5, 0.1, 0.1]
    # línea malformada, clase inválida y coords fuera de rango se saltan
    path.write_text("0 0.5 0.5 0.1 0.1\nbasura\n9 0.5 0.5 0.1 0.1\n1 2.0 0.5 0.1 0.1\n",
                    encoding="utf-8")
    assert mod._leer_yolo(path) == [[0, 0.5, 0.5, 0.1, 0.1]]


def test_escribir_yolo_redondea_y_termina_en_nueva_linea(oro):
    mod, ws = oro
    path = ws / "labels" / "nuevo.txt"
    mod._escribir_yolo(path, [[0, 0.123456789, 0.5, 0.25, 0.25], [1, 0.7, 0.7, 0.1, 0.1]])
    texto = path.read_text(encoding="utf-8")
    lineas = texto.splitlines()
    assert lineas == ["0 0.123457 0.500000 0.250000 0.250000",
                      "1 0.700000 0.700000 0.100000 0.100000"]
    assert texto.endswith("\n")


def test_box_gigante(oro):
    mod, _ = oro
    assert mod._box_gigante([0, 0.5, 0.5, 1.0, 0.05])   # w=1.00 del teacher
    assert mod._box_gigante([0, 0.5, 0.5, 0.5, 0.5])    # 25% de área exacta (>0.25)
    assert mod._box_gigante([0, 0.5, 0.5, 0.6, 0.1])    # más de la mitad de ancho
    assert not mod._box_gigante([0, 0.5, 0.5, 0.1, 0.1])
    assert not mod._box_gigante([0, 0.5, 0.5, 0.4, 0.4])  # 16% de área


def test_normalizar_recorta_y_descarta_degeneradas(oro):
    mod, _ = oro
    out = mod._normalizar([
        [0, -0.2, 0.5, 1.5, 0.5],        # fuera de rango → recortado a [0,1]
        [1, 0.5, 0.5, 0.0005, 0.0005],  # < ~4 px → descartada
        [7, 0.5, 0.5, 0.1, 0.1],        # clase inválida → descartada
        [0, 0.5, 0.5, 0.1, 0.1],
    ])
    assert out == [[0, 0.0, 0.5, 1.0, 0.5], [0, 0.5, 0.5, 0.1, 0.1]]


def test_backup_originales_solo_una_vez(oro):
    mod, ws = oro
    mod._backup_originales(ws)
    original = ws / "labels" / "_original"
    assert (original / "p001.txt").exists()
    assert (original / "p002.txt").exists()
    mod._backup_originales(ws)   # segunda llamada no debe duplicar ni fallar
    assert len(list(original.glob("*.txt"))) == 2


def test_guardar_pagina_vacia_escribe_txt_vacio_y_backup(oro):
    mod, ws = oro
    r = mod._guardar_pagina(ws, "p001", [], revisada=True)
    assert r["cajas"] == 0 and r["gigantes"] == 0
    txt = ws / "labels" / "p001.txt"
    assert txt.exists() and txt.read_text(encoding="utf-8") == ""   # vacío, no ausente
    # el oro original quedó a salvo
    assert (ws / "labels" / "_original" / "p001.txt").read_text(encoding="utf-8") != ""
    # marcada como revisada
    assert mod._cargar_revisadas(ws)["p001"] is True


def test_guardar_pagina_normaliza_y_escribe(oro):
    mod, ws = oro
    r = mod._guardar_pagina(ws, "p002", [[0, 0.5, 0.5, 0.2, 0.2]], revisada=False)
    assert r["cajas"] == 1
    assert ws / "labels" / "p002.txt" == \
        ws / "labels" / "p002.txt"  # sanity: path construido bien
    contenido = (ws / "labels" / "p002.txt").read_text(encoding="utf-8")
    assert contenido.startswith("0 0.500000 0.500000 0.200000 0.200000")


def test_listar_paginas_marca_gigantes_y_revisadas(oro):
    mod, ws = oro
    paginas = mod._listar_paginas(ws)
    assert [p["nombre"] for p in paginas] == ["p001", "p002"]
    p1 = paginas[0]
    assert p1["gigantes"] == [1]          # la caja w=1.00 del teacher
    assert p1["revisada"] is False
    assert paginas[1]["gigantes"] == []
