"""
test_uocr_daemon.py — Tests del endpoint /ocr-batch (Fase 1: infer_multi).

Prueba el parseo multi-imagen del daemon Unlimited-OCR:
- _parse_blocks_multi: divide el stream de stdout por <PAGE> y parsea cada página.
- _map_multi_blocks_to_page: mapea bloques del espacio 640x640 al de la página.
- _run_ocr_batch: con un modelo FAKE que imprime el stream (sin GPU), verifica
  que las N páginas se parsean, mapean y devuelven en el mismo orden.

NO carga el modelo real (4-bit, ~8 min) — se usa un stub de _model.infer_multi.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uocr_daemon as ud


# ─── Helpers ─────────────────────────────────────────────────────

def _fake_png(path: str, w: int = 1280, h: int = 960) -> None:
    """Crea una PNG blanca con texto simulado (para PIL en el daemon)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((100, 200, 400, 260), outline=(0, 0, 0), width=3)
    d.text((120, 210), "HOLA", fill=(0, 0, 0))
    img.save(path, "PNG")


class _FakeModel:
    """Stub de _model.infer_multi: imprime el stream por stdout (como el real)."""

    def __init__(self, pages_stream: list[str]):
        # pages_stream: un stream RAW por página (con tags <|det|>), SIN <PAGE>
        self.pages_stream = pages_stream

    def infer_multi(self, tokenizer, prompt="", image_files=None,
                    output_path="", image_size=640, max_length=32768,
                    tps_interval=0, no_repeat_ngram_size=0,
                    ngram_window=0, temperature=0.0, save_results=False):
        assert image_files, "infer_multi requiere imágenes"
        # El modelo real emite <PAGE> ANTES de cada página (ver
        # modeling_unlimitedocr.py: outputs.split('<PAGE>')[1:]).
        for page in self.pages_stream:
            print("<PAGE>", flush=True)
            print(page, flush=True)
        # El modelo real retorna (outputs, output_tokens)
        return "<PAGE>\n" + "\n<PAGE>\n".join(self.pages_stream), 0


# ─── _parse_blocks_multi ─────────────────────────────────────────

class TestParseBlocksMulti:
    def test_dos_paginas_separadas_por_page(self):
        # El modelo real emite <PAGE> ANTES de cada página (split [1:] del
        # save_results oficial), no entre páginas.
        stream = (
            "<PAGE>\n"
            "<|det|>text [10, 20, 100, 30]<|/det|>Hola\n"
            "<PAGE>\n"
            "<|det|>title [5, 5, 200, 40]<|/det|>CAPITULO 43\n"
        )
        pages = ud._parse_blocks_multi(stream, n_images=2)
        assert len(pages) == 2
        assert pages[0] == [{"type": "text", "x": 10, "y": 20, "w": 100,
                             "h": 30, "text": "Hola"}]
        assert pages[1][0]["type"] == "title"
        assert pages[1][0]["text"] == "CAPITULO 43"

    def test_ruido_pre_primer_page_se_descarta(self):
        """El split [1:] descarta ruido (sin tags <|det|>) antes del 1er
        <PAGE> — con skip_prompt=True el streamer no repite el prompt, así
        que el ruido real es texto suelto sin detecciones."""
        stream = (
            "ruido sin tags que no es contenido\n"
            "<PAGE>\n"
            "<|det|>text [10, 10, 50, 20]<|/det|>Bien\n"
        )
        pages = ud._parse_blocks_multi(stream, n_images=1)
        assert len(pages) == 1
        assert pages[0][0]["text"] == "Bien"

    def test_menos_paginas_de_las_esperadas_se_rellena(self):
        stream = (
            "<PAGE>\n"
            "<|det|>text [1, 1, 10, 10]<|/det|>Solo una\n"
        )
        pages = ud._parse_blocks_multi(stream, n_images=3)
        assert len(pages) == 3
        assert len(pages[0]) == 1
        assert pages[1] == []
        assert pages[2] == []

    def test_sin_bloques_devuelve_listas_vacias(self):
        pages = ud._parse_blocks_multi("texto sin tags", n_images=2)
        assert pages == [[], []]

    def test_pagina_vacia_en_medio_no_desalinea(self):
        """Bug corregido: una página sin texto emite <PAGE>\n<PAGE>
        consecutivos; la sección vacía NO debe filtrarse o las páginas
        posteriores se desplazan un índice."""
        stream = (
            "<PAGE>\n"
            "<|det|>text [10, 10, 50, 20]<|/det|>Primera\n"
            "<PAGE>\n"
            "<PAGE>\n"  # página 1 vacía (sin texto)
            "<|det|>title [5, 5, 100, 20]<|/det|>Tercera\n"
        )
        pages = ud._parse_blocks_multi(stream, n_images=3)
        assert len(pages) == 3
        assert pages[0][0]["text"] == "Primera"
        assert pages[1] == []  # página vacía en medio
        assert pages[2][0]["text"] == "Tercera"  # NO desplazada

    def test_sin_marcador_inicial_usa_primera_seccion(self):
        """Defensa N=1 / modelo sin <PAGE> inicial: si la sección
        pre-primer-marcador contiene tags <|det|>, es contenido real."""
        stream = (
            "<|det|>text [1, 1, 10, 10]<|/det|>Unica\n"
            "<PAGE>\n"
            "<|det|>title [2, 2, 20, 10]<|/det|>Extra\n"
        )
        pages = ud._parse_blocks_multi(stream, n_images=2)
        assert len(pages) == 2
        assert pages[0][0]["text"] == "Unica"
        assert pages[1][0]["text"] == "Extra"


# ─── _map_multi_blocks_to_page ───────────────────────────────────

class TestMapMultiBlocksToPage:
    def test_mapeo_640_a_pagina(self):
        blocks = [{"type": "text", "x": 320, "y": 160, "w": 128, "h": 64,
                   "text": "Hola"}]
        mapped = ud._map_multi_blocks_to_page(blocks, page_w=1280, page_h=960)
        b = mapped[0]
        # x: 320/640*1280 = 640; y: 160/640*960 = 240; w: 128/640*1280 = 256
        assert b["x"] == 640
        assert b["y"] == 240
        assert b["w"] == 256
        assert b["h"] == 96
        assert b["text"] == "Hola"

    def test_no_modifica_el_original(self):
        blocks = [{"type": "text", "x": 100, "y": 100, "w": 50, "h": 25,
                   "text": "x"}]
        ud._map_multi_blocks_to_page(blocks, page_w=640, page_h=640)
        assert blocks[0]["x"] == 100  # intocada (lista original)

    def test_clampa_tamano_minimo(self):
        blocks = [{"type": "image", "x": 1, "y": 1, "w": 1, "h": 1, "text": ""}]
        mapped = ud._map_multi_blocks_to_page(blocks, page_w=6400, page_h=6400)
        assert mapped[0]["w"] >= 1
        assert mapped[0]["h"] >= 1


# ─── _run_ocr_batch (modelo fake, sin GPU) ───────────────────────

class TestRunOcrBatch:
    def test_batch_dos_paginas_ordena_y_mapea(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "p1.png")
            p2 = os.path.join(tmp, "p2.png")
            _fake_png(p1, 1280, 960)
            _fake_png(p2, 640, 640)

            fake = _FakeModel([
                "<|det|>text [320, 160, 128, 64]<|/det|>Hola",
                "<|det|>title [10, 10, 200, 40]<|/det|>CAPITULO",
            ])
            monkeypatch.setattr(ud, "_model", fake)

            result = ud._run_ocr_batch([p1, p2], max_length=4096)

            assert "pages" in result
            assert result["n_images"] == 2
            assert len(result["pages"]) == 2
            # Página 1 (1280x960): x=320/640*1280=640, y=160/640*960=240
            b1 = result["pages"][0]["blocks"][0]
            assert b1["text"] == "Hola"
            assert b1["x"] == 640
            assert b1["y"] == 240
            # Página 2 (640x640): sin escala
            b2 = result["pages"][1]["blocks"][0]
            assert b2["text"] == "CAPITULO"
            assert b2["x"] == 10
            assert result["pages"][0]["recovered_from_art"] == 0

    def test_batch_pagina_fallida_no_tumba_las_demas(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "p1.png")
            _fake_png(p1, 640, 640)
            fake = _FakeModel(["<|det|>text [10, 10, 50, 20]<|/det|>OK"])
            monkeypatch.setattr(ud, "_model", fake)

            # Página 2 inexistente → el batch debe devolver error solo para ella
            result = ud._run_ocr_batch([p1, os.path.join(tmp, "no_existe.png")],
                                       max_length=4096)
            assert len(result["pages"]) == 2
            assert result["pages"][0]["blocks"][0]["text"] == "OK"
            assert "error" in result["pages"][1]
