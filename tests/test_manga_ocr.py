"""
test_manga_ocr.py — Tests unitarios para manga_ocr.py (Paso 3, PLAN_MANGA_OCR).

Cubre:
- _bloque_a_schema: mapeo de bloques internos al schema de salida (motores/detector)
- _texto_plano: orden de lectura natural (y, luego x)
- _escaneo_archivos / _parse_rango: entrada y rango de páginas
- _escribir_salida: JSON + TXT incrementales
- main(): integración con OCRManager mockeado (salida real, gate UOCR_ENABLED,
  omisión de archivos ya procesados)
"""

import sys
import os
import json

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

import config  # noqa: E402
import manga_ocr  # noqa: E402


@pytest.fixture(autouse=True)
def _limpiar_estado():
    """Aisla el estado global que muta main(): UOCR_ENABLED y el set de stems
    ya usados (evita la colisión de nombres entre tests)."""
    import manga_ocr as mo
    original = config.UOCR_ENABLED
    mo._STEM_EN_USO.clear()
    yield
    mo._STEM_EN_USO.clear()
    config.UOCR_ENABLED = original


class TestBloqueASchema:
    def test_bloque_hibrido(self):
        b = {"x": 10, "y": 20, "w": 80, "h": 30, "text": "HOLA",
             "confidence": 0.91}
        s = manga_ocr._bloque_a_schema(b)
        assert s["texto"] == "HOLA"
        assert s["bbox"] == [10, 20, 90, 50]
        assert abs(s["conf"] - 0.91) < 1e-6
        assert s["motores"] == ["easyocr", "rapid"]
        assert s["detector"] == "hibrido"

    def test_bloque_origenes_yolo_ctd_vlm(self):
        yolo = manga_ocr._bloque_a_schema(
            {"x": 0, "y": 0, "w": 10, "h": 10, "text": "A",
             "confidence": 0.5, "source": "yolo"})
        assert yolo["detector"] == "yolo"
        assert yolo["motores"] == ["easyocr", "rapid", "yolo"]
        ctd = manga_ocr._bloque_a_schema(
            {"x": 0, "y": 0, "w": 10, "h": 10, "text": "B",
             "confidence": 0.5, "source": "ctd_line"})
        assert ctd["detector"] == "ctd"
        vlm = manga_ocr._bloque_a_schema(
            {"x": 0, "y": 0, "w": 10, "h": 10, "text": "C",
             "confidence": 0.5, "source": "unlimited"})
        assert vlm["detector"] == "vlm"

    def test_bloque_sin_texto_se_filtra_en_main_pero_no_aqui(self):
        # El filtrado de texto vacío ocurre en main(); el mapper es fiel.
        s = manga_ocr._bloque_a_schema({"x": 1, "y": 2, "w": 3, "h": 4})
        assert s["texto"] == ""
        assert s["bbox"] == [1, 2, 4, 6]


class TestTextoPlano:
    def test_orden_lectura_y_luego_x(self):
        bloques = [
            {"texto": "ABAJO", "bbox": [10, 100, 50, 130]},
            {"texto": "DERECHA", "bbox": [200, 10, 300, 40]},
            {"texto": "ARRIBA", "bbox": [10, 10, 50, 40]},
        ]
        assert manga_ocr._texto_plano(bloques) == "ARRIBA\nDERECHA\nABAJO"

    def test_ignora_texto_vacio(self):
        bloques = [{"texto": "", "bbox": [0, 0, 10, 10]},
                   {"texto": "  ", "bbox": [0, 20, 10, 30]},
                   {"texto": "OK", "bbox": [0, 40, 10, 50]}]
        assert manga_ocr._texto_plano(bloques) == "OK"


class TestEscaneoYRango:
    def test_escaneo_filtra_y_ordena(self, tmp_path):
        for nombre in ("b.png", "a.pdf", "z.txt", "README.md", "c.JPG"):
            (tmp_path / nombre).write_bytes(b"x")
        docs = manga_ocr._escaneo_documentos(tmp_path)
        assert [(d.tipo, d.nombre) for d in docs] == [
            ("pdf", "a"), ("imagen", "b"), ("imagen", "c")]

    def test_escaneo_agrupa_carpetas(self, tmp_path):
        # Carpeta anidada con webp → UN documento, páginas en orden natural
        cap = tmp_path / "BookDownloads" / "serie" / "cap1"
        cap.mkdir(parents=True)
        for nombre in ("1.webp", "10.webp", "2.webp", "0.webp"):
            (cap / nombre).write_bytes(b"x")
        docs = manga_ocr._escaneo_documentos(tmp_path)
        assert len(docs) == 1
        d = docs[0]
        assert d.tipo == "carpeta" and d.nombre == "cap1"
        assert [p.name for p in d.paginas] == ["0.webp", "1.webp", "2.webp",
                                               "10.webp"]

    def test_escaneo_ignora_carpetas_sin_imagenes(self, tmp_path):
        (tmp_path / "vacia").mkdir()
        (tmp_path / "nota.txt").write_bytes(b"x")
        assert manga_ocr._escaneo_documentos(tmp_path) == []

    def test_orden_natural(self):
        clave = manga_ocr._orden_natural
        assert clave("2.webp") < clave("10.webp")
        assert clave("0.webp") < clave("1.webp")
        assert clave("a2.webp") < clave("a10.webp")

    def test_parse_rango(self):
        assert manga_ocr._parse_rango("") is None
        assert manga_ocr._parse_rango("3-5") == (3, 5)
        assert manga_ocr._parse_rango("7") == (7, 7)
        assert manga_ocr._parse_rango("0-5") == (1, 5)  # 1-indexado
        with pytest.raises(SystemExit):
            manga_ocr._parse_rango("abc")


class TestEscribirSalida:
    def test_escribe_json_y_txt(self, tmp_path):
        paginas = [{"n": 1, "bloques": [{"texto": "HOLA", "bbox": [0, 0, 10, 10],
                                         "conf": 0.9, "motores": ["easyocr"],
                                         "detector": "hibrido"}],
                    "texto_plano": "HOLA", "t_s": 1.2, "engines": ["easyocr+rapid"],
                    "n_bloques": 1}]
        meta = {"archivo": "cap.pdf", "generado": "2026-08-11T00:00:00+00:00",
                "ocr_mode": "fusion", "detectores": ["easyocr", "rapid"],
                "zoom": 2.0}
        manga_ocr._escribir_salida(tmp_path, "cap", meta, paginas)
        doc = json.loads((tmp_path / "cap.json").read_text(encoding="utf-8"))
        assert doc["archivo"] == "cap.pdf"
        assert len(doc["paginas"]) == 1
        assert doc["paginas"][0]["texto_plano"] == "HOLA"
        txt = (tmp_path / "cap.txt").read_text(encoding="utf-8")
        assert "=== Página 1 ===" in txt
        assert "HOLA" in txt


class TestMain:
    def _imagen(self, path):
        img = np.ones((200, 150, 3), dtype=np.uint8) * 200
        cv2.imwrite(str(path), img)
        return path

    def _fake_manager(self, mocker):
        fake = MagicMock()
        fake.run_ocr.return_value = (
            [{"x": 10, "y": 20, "w": 80, "h": 30, "text": "HOLA",
              "confidence": 0.91}],
            "fusion", ["easyocr+rapid"])
        mocker.patch("manga_ocr.OCRManager", return_value=fake)
        return fake

    def test_main_procesa_imagen_y_escribe_salida(self, mocker, tmp_path):
        entrada = tmp_path / "in"
        salida = tmp_path / "out"
        entrada.mkdir()
        self._imagen(entrada / "cap.png")
        fake = self._fake_manager(mocker)

        rc = manga_ocr.main(["--input", str(entrada), "--output", str(salida)])

        assert rc == 0
        # Extracción pura: sin --vlm → el gate del refuerzo VLM queda OFF
        assert config.UOCR_ENABLED is False
        jp = salida / "cap.json"
        tp = salida / "cap.txt"
        assert jp.exists() and tp.exists()
        doc = json.loads(jp.read_text(encoding="utf-8"))
        assert doc["archivo"] == "cap.png"
        assert doc["ocr_mode"] == "fusion"
        assert len(doc["paginas"]) == 1
        p = doc["paginas"][0]
        assert p["n"] == 1
        assert p["bloques"][0]["texto"] == "HOLA"
        assert p["bloques"][0]["bbox"] == [10, 20, 90, 50]
        assert p["texto_plano"] == "HOLA"
        assert p["engines"] == ["easyocr+rapid"]
        txt = tp.read_text(encoding="utf-8")
        assert "=== Página 1 ===" in txt and "HOLA" in txt
        # doc_id escopeado (md5 del nombre, sesión 126) se pasa a run_ocr
        doc_id = fake.run_ocr.call_args.kwargs["doc_id"]
        assert len(doc_id) == 12

    def test_main_vlm_no_apaga_el_gate(self, mocker, tmp_path):
        entrada = tmp_path / "in"
        salida = tmp_path / "out"
        entrada.mkdir()
        self._imagen(entrada / "cap.png")
        self._fake_manager(mocker)

        rc = manga_ocr.main(["--input", str(entrada), "--output", str(salida),
                             "--vlm"])

        assert rc == 0
        assert config.UOCR_ENABLED is True  # --vlm no desactiva el refuerzo

    def test_main_omite_existente_sin_force(self, mocker, tmp_path):
        entrada = tmp_path / "in"
        salida = tmp_path / "out"
        entrada.mkdir()
        salida.mkdir()
        self._imagen(entrada / "cap.png")
        (salida / "cap.json").write_text("{}", encoding="utf-8")
        fake = self._fake_manager(mocker)

        rc = manga_ocr.main(["--input", str(entrada), "--output", str(salida)])

        assert rc == 0
        fake.run_ocr.assert_not_called()
        # --force sí reprocesa
        rc = manga_ocr.main(["--input", str(entrada), "--output", str(salida),
                             "--force"])
        assert rc == 0
        assert fake.run_ocr.called

    def test_main_entrada_inexistente_retorna_1(self, tmp_path):
        assert manga_ocr.main(["--input", str(tmp_path / "no_existe")]) == 1
