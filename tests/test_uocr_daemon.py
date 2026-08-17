"""
test_uocr_daemon.py — Tests del endpoint /ocr-batch (Fase 1: infer_multi).

Prueba el parseo multi-imagen del daemon Unlimited-OCR:
- _parse_blocks_multi: divide el stream de stdout por <PAGE> y parsea cada página.
- _map_multi_blocks_to_page: mapea bloques del espacio 640x640 al de la página.
- _run_ocr_batch: con un modelo FAKE que imprime el stream (sin GPU), verifica
  que las N páginas se parsean, mapean y devuelven en el mismo orden.

NO carga el modelo real (4-bit, ~8 min) — se usa un stub de _model.infer_multi.
"""
import json
import os
import sys
import tempfile
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import uocr_daemon as ud


# ─── Fixtures ────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _torch_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """El daemon importa torch dentro de _run_ocr/_run_ocr_batch/_load_model.

    En el CI no está instalado (pip install no lo incluye): se inyecta un
    fake mínimo para que las rutas de inferencia se ejecuten sin GPU. El
    test de _load_model lo reemplaza con su propio fake más completo.
    """
    class _Cuda:
        @staticmethod
        def reset_peak_memory_stats() -> None:
            return None

        @staticmethod
        def max_memory_allocated() -> float:
            return 2e9

    class _Torch:
        cuda = _Cuda()

    monkeypatch.setitem(sys.modules, "torch", _Torch())


@pytest.fixture(autouse=True)
def _status_restore() -> Iterator[None]:
    """Restaura _status (lo mutan _load_model y los tests del handler)."""
    snapshot = dict(ud._status)
    yield
    ud._status.clear()
    ud._status.update(snapshot)


# ─── Helpers ─────────────────────────────────────────────────────


def _make_handler(path: str, body: bytes | None = None,
                  content_length: int | None = None) -> Any:
    """Handler HTTP con las partes de red reemplazadas por registros.

    Devuelve ``Any`` a propósito: muta atributos y métodos de instancia de
    ``ud._Handler`` (headers, send_response, _sent_code) que el stub de
    mypy no conoce; el handler con red mockeada no necesita tipado estático.
    """
    h = cast(Any, object.__new__(ud._Handler))
    h.path = path
    raw = body if body is not None else b"{}"
    length = len(raw) if content_length is None else content_length
    h.headers = {"Content-Length": str(length)}
    h.rfile = BytesIO(raw)
    h.wfile = BytesIO()
    h._sent_code = None
    h._sent_headers = {}

    def _send_response(code: int, message: str | None = None) -> None:
        h._sent_code = code

    def _send_header(k: str, v: str) -> None:
        h._sent_headers[k] = v

    def _end_headers() -> None:
        pass

    h.send_response = _send_response
    h.send_header = _send_header
    h.end_headers = _end_headers
    return h

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

    def infer_multi(self, tokenizer: object, prompt: str = "",
                    image_files: object = None,
                    output_path: str = "", image_size: int = 640,
                    max_length: int = 32768, tps_interval: int = 0,
                    no_repeat_ngram_size: int = 0, ngram_window: int = 0,
                    temperature: float = 0.0,
                    save_results: bool = False) -> tuple[str, int]:
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
    def test_dos_paginas_separadas_por_page(self) -> None:
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

    def test_ruido_pre_primer_page_se_descarta(self) -> None:
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

    def test_menos_paginas_de_las_esperadas_se_rellena(self) -> None:
        stream = (
            "<PAGE>\n"
            "<|det|>text [1, 1, 10, 10]<|/det|>Solo una\n"
        )
        pages = ud._parse_blocks_multi(stream, n_images=3)
        assert len(pages) == 3
        assert len(pages[0]) == 1
        assert pages[1] == []
        assert pages[2] == []

    def test_sin_bloques_devuelve_listas_vacias(self) -> None:
        pages = ud._parse_blocks_multi("texto sin tags", n_images=2)
        assert pages == [[], []]

    def test_pagina_vacia_en_medio_no_desalinea(self) -> None:
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

    def test_sin_marcador_inicial_usa_primera_seccion(self) -> None:
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
    def test_mapeo_640_a_pagina(self) -> None:
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

    def test_no_modifica_el_original(self) -> None:
        blocks = [{"type": "text", "x": 100, "y": 100, "w": 50, "h": 25,
                   "text": "x"}]
        ud._map_multi_blocks_to_page(blocks, page_w=640, page_h=640)
        assert blocks[0]["x"] == 100  # intocada (lista original)

    def test_clampa_tamano_minimo(self) -> None:
        blocks = [{"type": "image", "x": 1, "y": 1, "w": 1, "h": 1, "text": ""}]
        mapped = ud._map_multi_blocks_to_page(blocks, page_w=6400, page_h=6400)
        assert mapped[0]["w"] >= 1
        assert mapped[0]["h"] >= 1


# ─── _run_ocr_batch (modelo fake, sin GPU) ───────────────────────

class TestRunOcrBatch:
    def test_batch_dos_paginas_ordena_y_mapea(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def test_batch_pagina_fallida_no_tumba_las_demas(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
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


class TestDaemonRequestGuards:
    def test_input_path_only_permite_proyecto_y_temporal(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        import tempfile as _tempfile

        project = tmp_path / "project"
        temp_root = tmp_path / "temp"
        outside = tmp_path / "outside"
        project.mkdir()
        temp_root.mkdir()
        outside.mkdir()
        project_file = project / "page.png"
        temp_file = temp_root / "page.png"
        outside_file = outside / "secret.png"
        for path in (project_file, temp_file, outside_file):
            path.write_bytes(b"x")

        monkeypatch.setattr(ud, "ROOT", str(project))
        monkeypatch.setattr(_tempfile, "gettempdir", lambda: str(temp_root))

        assert ud._is_allowed_input_path(str(project_file))
        assert ud._is_allowed_input_path(str(temp_file))
        assert not ud._is_allowed_input_path(str(outside_file))
        assert not ud._is_allowed_input_path(str(project / "missing.png"))

    def test_input_path_temporal_en_otra_unidad_no_se_rechaza(
        self, monkeypatch: pytest.MonkeyPatch) -> None:
        """El proyecto D: y TEMP C: son válidos simultáneamente en Windows."""
        import tempfile as _tempfile

        monkeypatch.setattr(ud, "ROOT", r"D:\crear traductor")
        monkeypatch.setattr(_tempfile, "gettempdir", lambda: r"C:\Temp")
        monkeypatch.setattr(os.path, "isfile", lambda _path: True)

        assert ud._is_allowed_input_path(r"C:\Temp\uocr_page.png")

    def test_cleanup_old_out_dirs_incluye_reocr_de_arte(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        base = tmp_path / "uocr_daemon_out"
        base.mkdir()
        nombres = ["req_old", "art_old", "req_new", "art_new"]
        for i, nombre in enumerate(nombres):
            carpeta = base / nombre
            carpeta.mkdir()
            os.utime(carpeta, (100 + i, 100 + i))

        monkeypatch.setattr(ud, "ROOT", str(tmp_path))
        monkeypatch.setattr(ud, "_MAX_REQ_DIRS", 2)

        ud._cleanup_old_out_dirs()

        assert sorted(p.name for p in base.iterdir()) == ["art_new", "req_new"]

    def test_read_json_body_rechaza_cuerpo_excesivo(self) -> None:
        handler = cast(Any, object.__new__(ud._Handler))
        handler.headers = {"Content-Length": str(ud._MAX_JSON_BODY_BYTES + 1)}
        handler.rfile = BytesIO(b"{}")

        assert ud._Handler._read_json_body(handler) is None

    def test_read_json_body_exige_objeto_json(self) -> None:
        handler = cast(Any, object.__new__(ud._Handler))
        payload = b"[]"
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = BytesIO(payload)

        assert ud._Handler._read_json_body(handler) is None


# ─── _parse_blocks ───────────────────────────────────────────────

class TestParseBlocks:
    def test_parsea_un_bloque(self) -> None:
        blocks = ud._parse_blocks(
            "<|det|>text [10, 20, 100, 30]<|/det|>Hola mundo")
        assert blocks == [{"type": "text", "x": 10, "y": 20, "w": 100,
                           "h": 30, "text": "Hola mundo"}]

    def test_parsea_varios_bloques(self) -> None:
        text = (
            "<|det|>title [1, 2, 3, 4]<|/det|>TITULO\n"
            "<|det|>image [5, 6, 7, 8]<|/det|>\n"
        )
        blocks = ud._parse_blocks(text)
        assert len(blocks) == 2
        assert blocks[0]["type"] == "title"
        assert blocks[1]["type"] == "image"
        assert blocks[1]["text"] == ""

    def test_sin_tags_devuelve_vacio(self) -> None:
        assert ud._parse_blocks("texto sin detecciones") == []


# ─── _load_model ─────────────────────────────────────────────────

class TestLoadModel:
    def test_carga_exitosa(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeCuda:
            @staticmethod
            def reset_peak_memory_stats() -> None:
                return None

            @staticmethod
            def max_memory_allocated() -> float:
                return 1e9

        class _FakeTorch:
            float16 = "float16"
            cuda = _FakeCuda()

        class _FakeConfig:
            def __init__(self, **kw: object) -> None:
                pass

        class _FakeAutoModel:
            device: str = "cpu"

            def eval(self) -> "_FakeAutoModel":
                return self

            @classmethod
            def from_pretrained(cls, *a: object, **k: object) -> "_FakeAutoModel":
                m = object.__new__(cls)
                m.device = "cuda"
                return m

        class _FakeTransformers:
            BitsAndBytesConfig = _FakeConfig
            AutoTokenizer = type(
                "AutoTokenizer", (), {
                    "from_pretrained": classmethod(
                        lambda cls, *a, **k: "tok"),
                })
            AutoModel = _FakeAutoModel

        monkeypatch.setitem(sys.modules, "torch", _FakeTorch())
        monkeypatch.setitem(sys.modules, "transformers", _FakeTransformers())

        ud._load_model()

        assert ud._status["state"] == "ready"
        assert ud._tokenizer == "tok"
        assert ud._status["vram_gb"] == 1.0

    def test_error_de_carga_marca_error(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)  # import falla

        ud._load_model()

        assert ud._status["state"] == "error"
        assert ud._status["error"]


# ─── _infer_once ─────────────────────────────────────────────────

class TestInferOnce:
    def test_lee_result_md_y_parsea_stream(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeModel:
            def infer(self, tokenizer: object, **kw: object) -> None:
                print("<|det|>text [1, 2, 30, 10]<|/det|>Hola")

        monkeypatch.setattr(ud, "_model", _FakeModel())
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        (out_dir / "result.md").write_text("Hola", encoding="utf-8")

        text, blocks, infer_s = ud._infer_once("img.png", str(out_dir), 4096)

        assert text == "Hola"
        assert blocks[0]["text"] == "Hola"
        assert infer_s >= 0

    def test_sin_result_md_texto_vacio(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeModel:
            def infer(self, tokenizer: object, **kw: object) -> None:
                print("<|det|>title [1, 1, 9, 9]<|/det|>X")

        monkeypatch.setattr(ud, "_model", _FakeModel())
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        text, blocks, _ = ud._infer_once("img.png", str(out_dir), 4096)

        assert text == ""
        assert blocks[0]["type"] == "title"


# ─── _recover_art_dialogue ───────────────────────────────────────

class TestRecoverArtDialogue:
    def test_sin_bloque_grande_no_toca_nada(
            self, tmp_path: Path) -> None:
        img = tmp_path / "p.png"
        _fake_png(str(img), 1280, 960)
        blocks = [{"type": "text", "x": 10, "y": 10, "w": 100, "h": 20,
                   "text": "Hola"}]

        out, n = ud._recover_art_dialogue(str(img), blocks, 4096)

        assert n == 0
        assert out == blocks

    def test_recupera_dialogo_de_arte(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        img = tmp_path / "p.png"
        _fake_png(str(img), 1280, 960)
        big = {"type": "image", "x": 100, "y": 100, "w": 600, "h": 800,
               "text": ""}

        def fake_infer_once(path: str, out_dir: str, max_length: int,
                            crop_mode: bool = True, image_size: int = 640,
                            prompt: str | None = None,
                            ngram_size: int | None = None,
                            ngram_window: int | None = None) -> tuple[str, list[dict[str, object]], float]:
            return ("", [{"type": "text", "x": 320, "y": 320, "w": 100,
                           "h": 40, "text": "ARTE"}], 0.1)

        monkeypatch.setattr(ud, "_infer_once", fake_infer_once)

        out, n = ud._recover_art_dialogue(str(img), [big], 4096)

        assert n == 1
        rec = [b for b in out if b.get("from_art_recrop")]
        assert len(rec) == 1
        assert rec[0]["text"] == "ARTE"
        assert rec[0]["x"] >= 100  # mapeado de vuelta al espacio de página

    def test_bloque_estrecho_se_omite(
            self, tmp_path: Path) -> None:
        img = tmp_path / "p.png"
        _fake_png(str(img), 100, 100)  # página pequeña
        # área 20*200 = 4000 > 0.3*10000 → entra, pero cw=20 < 32 → se omite
        estrecho = {"type": "image", "x": 0, "y": 0, "w": 20, "h": 200,
                    "text": ""}
        out, n = ud._recover_art_dialogue(str(img), [estrecho], 4096)
        assert n == 0
        assert out == [estrecho]

    def test_imagen_ilegible_devuelve_original(
            self, tmp_path: Path) -> None:
        img = tmp_path / "no_es_imagen.png"
        img.write_bytes(b"no es png")
        blocks = [{"type": "image", "x": 0, "y": 0, "w": 500, "h": 500,
                   "text": ""}]
        out, n = ud._recover_art_dialogue(str(img), blocks, 4096)
        assert n == 0
        assert out == blocks


# ─── _run_ocr ────────────────────────────────────────────────────

class TestRunOcr:
    def test_ocr_con_modelo_fake(
            self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # ROOT apuntado al tmp: _run_ocr escribe el dir de salida req_* ahí.
        monkeypatch.setattr(ud, "ROOT", str(tmp_path))

        def fake_infer_once(image_path: str, out_dir: str, max_length: int,
                            crop_mode: bool = True, image_size: int = 640,
                            prompt: str | None = None,
                            ngram_size: int | None = None,
                            ngram_window: int | None = None) -> tuple[str, list[dict[str, object]], float]:
            return ("Hola", [{"type": "text", "x": 10, "y": 20, "w": 100,
                               "h": 30, "text": "Hola"}], 0.1)

        monkeypatch.setattr(ud, "_infer_once", fake_infer_once)
        monkeypatch.setattr(ud, "_recover_art_dialogue",
                            lambda img, blocks, ml, prompt=None,
                            ngram_size=None, ngram_window=None: (blocks, 0))
        img = tmp_path / "p.png"
        _fake_png(str(img), 640, 640)

        r = ud._run_ocr(str(img), 4096)

        assert r["text"] == "Hola"
        assert r["blocks"][0]["text"] == "Hola"
        assert r["recovered_from_art"] == 0
        assert r["infer_s"] >= 0


# ─── Handler HTTP ────────────────────────────────────────────────

class TestHandler:
    def test_do_get_health(self) -> None:
        ud._status.update(state="ready")
        h = _make_handler("/health")
        h.do_GET()
        assert h._sent_code == 200
        assert json.loads(h.wfile.getvalue())["state"] == "ready"

    def test_do_get_ruta_desconocida_404(self) -> None:
        h = _make_handler("/otra")
        h.do_GET()
        assert h._sent_code == 404

    def test_do_post_ruta_desconocida_404(self) -> None:
        h = _make_handler("/nope", body=b"{}")
        h.do_POST()
        assert h._sent_code == 404

    def test_do_post_modelo_no_listo_503(self) -> None:
        ud._status.update(state="loading")
        h = _make_handler("/ocr", body=b"{}")
        h.do_POST()
        assert h._sent_code == 503

    def test_do_post_json_invalido_400(self) -> None:
        ud._status.update(state="ready")
        h = _make_handler("/ocr", body=b"no-json")
        h.do_POST()
        assert h._sent_code == 400

    def test_do_post_max_length_no_entero_400(self) -> None:
        ud._status.update(state="ready")
        h = _make_handler("/ocr",
                          body=b'{"image_path": "a.png", "max_length": "x"}')
        h.do_POST()
        assert h._sent_code == 400

    def test_do_post_max_length_fuera_rango_400(self) -> None:
        ud._status.update(state="ready")
        h = _make_handler("/ocr",
                          body=b'{"image_path": "a.png", "max_length": 10}')
        h.do_POST()
        assert h._sent_code == 400

    def test_do_post_prompt_no_string_400(self) -> None:
        """Plan §10.2 item 2: prompt no-string → 400 (no silenciar el error
        de configuración del A/B)."""
        ud._status.update(state="ready")
        h = _make_handler("/ocr", body=b'{"image_path": "a.png", '
                                         b'"max_length": 1000, "prompt": 5}')
        h.do_POST()
        assert h._sent_code == 400

    def test_do_post_ngram_invalido_400(self) -> None:
        """Plan §10.2 item 2: ngram no-entero o fuera de 1-64 → 400."""
        ud._status.update(state="ready")
        for body in (b'{"image_path": "a.png", "max_length": 1000, '
                     b'"ngram": "x"}',
                     b'{"image_path": "a.png", "max_length": 1000, '
                     b'"ngram": 0}',
                     b'{"image_path": "a.png", "max_length": 1000, '
                     b'"ngram": 99}'):
            h = _make_handler("/ocr", body=body)
            h.do_POST()
            assert h._sent_code == 400

    def test_do_post_prompt_y_ngram_se_reenvian(self,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
        """Plan §10.2 item 2: prompt/ngram válidos llegan a _run_ocr (el A/B
        los varía por request sin reiniciar el daemon)."""
        ud._status.update(state="ready")
        monkeypatch.setattr(ud, "_is_allowed_input_path", lambda v: True)
        capturado: dict[str, object] = {}

        def _spy(p: str, ml: int, prompt: str | None = None,
                 ngram_size: int | None = None,
                 image_size: int | None = None) -> dict[str, object]:
            capturado["prompt"] = prompt
            capturado["ngram"] = ngram_size
            capturado["image_size"] = image_size
            return {"text": "Hola", "blocks": []}

        monkeypatch.setattr(ud, "_run_ocr", _spy)
        h = _make_handler("/ocr", body=b'{"image_path": "x.png", '
                                         b'"max_length": 1000, "prompt": '
                                         b'"<image>extrae", "ngram": 15}')
        h.do_POST()
        assert h._sent_code == 200
        assert capturado["prompt"] == "<image>extrae"
        assert capturado["ngram"] == 15
        assert capturado["image_size"] is None  # no enviado → default

    def test_do_post_image_size_invalido_400(self) -> None:
        """Plan §10.2 item 5: image_size no-entero o fuera de 256-1024 → 400."""
        ud._status.update(state="ready")
        for body in (b'{"image_path": "a.png", "max_length": 1000, '
                     b'"image_size": "x"}',
                     b'{"image_path": "a.png", "max_length": 1000, '
                     b'"image_size": 100}',
                     b'{"image_path": "a.png", "max_length": 1000, '
                     b'"image_size": 2048}'):
            h = _make_handler("/ocr", body=body)
            h.do_POST()
            assert h._sent_code == 400

    def test_do_post_image_size_se_reenvia(self,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
        """Plan §10.2 item 5: image_size válido llega a _run_ocr."""
        ud._status.update(state="ready")
        monkeypatch.setattr(ud, "_is_allowed_input_path", lambda v: True)
        capturado: dict[str, object] = {}

        def _spy2(p: str, ml: int, prompt: str | None = None,
                  ngram_size: int | None = None,
                  image_size: int | None = None) -> dict[str, object]:
            capturado["image_size"] = image_size
            return {"text": "Hola", "blocks": []}

        monkeypatch.setattr(ud, "_run_ocr", _spy2)
        h = _make_handler("/ocr", body=b'{"image_path": "x.png", '
                                         b'"max_length": 1000, '
                                         b'"image_size": 512}')
        h.do_POST()
        assert h._sent_code == 200
        assert capturado["image_size"] == 512

    def test_do_post_image_path_no_permitido_400(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        ud._status.update(state="ready")
        monkeypatch.setattr(ud, "_is_allowed_input_path", lambda v: False)
        h = _make_handler("/ocr",
                          body=b'{"image_path": "x.png", "max_length": 1000}')
        h.do_POST()
        assert h._sent_code == 400

    def test_do_post_batch_images_invalido_400(self) -> None:
        ud._status.update(state="ready")
        h = _make_handler("/ocr-batch",
                          body=b'{"images": [], "max_length": 1000}')
        h.do_POST()
        assert h._sent_code == 400

    def test_do_post_ocr_ok(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        ud._status.update(state="ready")
        monkeypatch.setattr(ud, "_is_allowed_input_path", lambda v: True)
        monkeypatch.setattr(ud, "_run_ocr",
                            lambda p, ml, prompt=None, ngram_size=None,
                            image_size=None: {"text": "Hola", "blocks": []})
        h = _make_handler("/ocr",
                          body=b'{"image_path": "x.png", "max_length": 1000}')
        h.do_POST()
        assert h._sent_code == 200
        assert json.loads(h.wfile.getvalue())["text"] == "Hola"

    def test_do_post_batch_ok(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        ud._status.update(state="ready")
        monkeypatch.setattr(ud, "_is_allowed_input_path", lambda v: True)
        monkeypatch.setattr(ud, "_run_ocr_batch",
                            lambda imgs, ml, prompt=None, ngram_size=None,
                            image_size=None: {"pages": [], "n_images": 1})
        h = _make_handler("/ocr-batch",
                          body=b'{"images": ["a.png"], "max_length": 1000}')
        h.do_POST()
        assert h._sent_code == 200

    def test_do_post_500_en_error_interno(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        ud._status.update(state="ready")
        monkeypatch.setattr(ud, "_is_allowed_input_path", lambda v: True)

        def _boom(p: str, ml: int) -> dict[str, object]:
            raise RuntimeError("fallo interno")

        monkeypatch.setattr(ud, "_run_ocr", _boom)
        h = _make_handler("/ocr",
                          body=b'{"image_path": "x.png", "max_length": 1000}')
        h.do_POST()
        assert h._sent_code == 500
