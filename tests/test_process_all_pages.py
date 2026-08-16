"""
test_process_all_pages.py — Tests para process_all_pages.py.

Cubre dos cosas (sesión 2026-08-06):

1. **Fix NameError en `procesar_pagina`**: el refactor a `_registrar_resultado`
   perdió la línea `data = resp.json()`, así que el camino single
   (--batch-window 1) crasheaba en silencio (0 páginas registradas aunque el
   servidor sí procesaba). El test llama a `procesar_pagina` directamente con
   el requests mockeado y verifica que registra resultados sin excepción.

2. **`--max-pages N` limita el render**: se ejecuta `main()` completo con
   fitz/requests mockeados sobre un PDF fake de 53 páginas y se verifica que
   solo se renderizan las N primeras (y que el checkpoint resultante refleja
   ese límite).

El módulo se carga en fresco por test (importlib con nombre único) porque
process_all_pages.py parsea sys.argv y define su estado a nivel de módulo;
el flujo completo solo se ejecuta bajo `if __name__ == "__main__"`.
"""

import json
import hashlib
import os
import re
import sys
import time

import pytest
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_PROCESS_ALL_PAGES = os.path.join(os.path.dirname(__file__), "..", "process_all_pages.py")


# ─── Helpers ──────────────────────────────────────────────────────

class FakeResponse:
    """Respuesta requests minimalista (status_code + json())."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


_load_counter = [0]


def _load_module(argv=None):
    """Carga process_all_pages.py en fresco con sys.argv controlado."""
    import importlib.util

    _load_counter[0] += 1
    name = f"pap_test_mod_{_load_counter[0]}"
    spec = importlib.util.spec_from_file_location(name, _PROCESS_ALL_PAGES)
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    sys.argv = argv if argv is not None else ["process_all_pages.py"]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = old_argv
    return mod


def _argv_isolado(tmp_path, extra=None):
    """argv base que aísla el checkpoint del real (resultados_progreso.json)."""
    argv = ["process_all_pages.py", "--checkpoint-file", str(tmp_path / "cp.json")]
    if extra:
        argv += extra
    return argv


def _mock_batch_post(monkeypatch, tmp_path, results, argv_extra=None,
                     status_code=200):
    """Helper compartido del camino batch (Fase 1): carga process_all_pages.py
    en fresco y mockea el POST de /api/process-page-batch con captura de
    payload — elimina el boilerplate repetido en los tests de procesar_lote.

    Devuelve (mod, captured): mod es el módulo recién cargado (estado limpio,
    argv controlado); captured es un dict que recibe {'url': ..., 'json': ...}
    del POST del lote (para verificar endpoint, imágenes y flags).

    results es la lista de resultados que se envuelve como
    {"results": results} en la respuesta; status_code permite simular un
    error HTTP (p. ej. 500) sin romper la captura del payload.
    """
    mod = _load_module(_argv_isolado(tmp_path, argv_extra))
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["json"] = kwargs.get("json", {})
        return FakeResponse({"results": results}, status_code=status_code)

    monkeypatch.setattr(mod._http_session, "post", fake_post)
    return mod, captured


# ─── procesar_pagina (fix NameError 2026-08-06) ──────────────────

class TestProcesarPagina:
    """procesar_pagina registra resultados sin NameError y actualiza stats."""

    def test_registra_resultados_sin_nameerror(self, monkeypatch, tmp_path):
        """El camino single (sin batch) debe registrar la página y sus stats.

        Antes del fix esto lanzaba `NameError: name 'data' is not defined`
        (la línea `data = resp.json()` se perdió en el refactor), dejando
        results/pages_done vacíos aunque el servidor respondiera 200.
        """
        mod = _load_module(_argv_isolado(tmp_path))
        resp = FakeResponse({"blocks": [
            {"source": "HOLA", "translated": "HELLO"},
            {"source": "N", "translated": "N"},  # SIN_TRAD: orig == trad
        ]})
        monkeypatch.setattr(mod._http_session, "post", lambda *a, **k: resp)

        # No debe lanzar ninguna excepción
        mod.procesar_pagina(1, "data:image/png;base64,AAAA", 0.3)

        assert mod.pages_done == {1}
        assert mod.stats["total_blocks_found"] == 2
        assert mod.stats["total_blocks_translated"] == 1
        assert mod.stats["pages_with_text"] == 1
        assert mod.stats["pages_empty"] == 0
        assert len(mod.results) == 1
        r = mod.results[0]
        assert r["page"] == 1
        assert r["blocks"] == 2
        assert r["translated"] == 1
        assert r["status"] == "PARCIAL"  # 1 de 2 bloques traducidos

    def test_vacio_registra_vacio(self, monkeypatch, tmp_path):
        """Sin bloques → status VACIO y stats pages_empty += 1."""
        mod = _load_module(_argv_isolado(tmp_path))
        monkeypatch.setattr(mod._http_session, "post",
                            lambda *a, **k: FakeResponse({"blocks": []}))
        mod.procesar_pagina(1, "b64", 0.3)
        assert mod.results[0]["status"] == "VACIO"
        assert mod.stats["pages_empty"] == 1
        assert mod.stats["pages_error"] == 0

    def test_sin_traduccion_registra_sin_trad(self, monkeypatch, tmp_path):
        """Bloques sin traducir (orig == trad) → status SIN_TRAD."""
        mod = _load_module(_argv_isolado(tmp_path))
        resp = FakeResponse({"blocks": [
            {"source": "SFX", "translated": "SFX"},
            {"source": "Non-Text", "translated": "Non-Text"},
        ]})
        monkeypatch.setattr(mod._http_session, "post", lambda *a, **k: resp)
        mod.procesar_pagina(1, "b64", 0.3)
        assert mod.results[0]["status"] == "SIN_TRAD"
        assert mod.stats["total_blocks_found"] == 2
        assert mod.stats["total_blocks_translated"] == 0

    def test_render_previo_fallido_registra_render_error(self, tmp_path):
        """b64=None (error de render) → status render_error sin tocar HTTP."""
        mod = _load_module(_argv_isolado(tmp_path))
        mod.procesar_pagina(1, None, 0.0)
        assert mod.results[0]["status"] == "render_error"
        assert mod.results[0]["blocks"] == 0
        assert mod.stats["pages_error"] == 1

    def test_timeout_definitivo_registra_timeout(self, monkeypatch, tmp_path):
        """Timeout agotando reintentos → status timeout, sin NameError."""
        mod = _load_module(_argv_isolado(tmp_path))
        monkeypatch.setattr(mod, "MAX_RETRIES", 0)  # sin reintentos: rápido

        def boom(*a, **k):
            raise requests.Timeout()

        monkeypatch.setattr(mod._http_session, "post", boom)
        mod.procesar_pagina(1, "b64", 0.3)
        assert mod.results[0]["status"] == "timeout"
        assert mod.stats["pages_error"] == 1
        assert mod.pages_done == {1}

    def test_conn_error_definitivo_registra_sin_nameerror(self, monkeypatch, tmp_path):
        """Una excepción de conexión en modo single no pierde la página."""
        mod = _load_module(_argv_isolado(tmp_path))
        monkeypatch.setattr(mod, "MAX_RETRIES", 0)

        def boom(*_args, **_kwargs):
            raise requests.ConnectionError("boom")

        monkeypatch.setattr(mod._http_session, "post", boom)
        seq = [100.0, 110.0]
        monkeypatch.setattr(mod.time, "time", lambda: seq.pop(0))

        mod.procesar_pagina(1, "b64", 0.3)

        assert [(r["page"], r["status"], r["time"]) for r in mod.results] == [
            (1, "conn_error", 10.0)
        ]
        assert mod.stats["pages_error"] == 1
        assert mod.pages_done == {1}

    def test_http_error_registra_status(self, monkeypatch, tmp_path):
        """HTTP != 200 → status http_<code>."""
        mod = _load_module(_argv_isolado(tmp_path))
        monkeypatch.setattr(mod._http_session, "post",
                            lambda *a, **k: FakeResponse({}, status_code=500))
        mod.procesar_pagina(1, "b64", 0.3)
        assert mod.results[0]["status"] == "http_500"
        assert mod.stats["pages_error"] == 1

    # -- --force-uocr (sesión 113) -- --

    def test_force_uocr_true_incluido_en_payload_single(self, monkeypatch, tmp_path):
        """--force-uocr → el payload de procesar_pagina lleva force_uocr=True."""
        mod = _load_module(_argv_isolado(tmp_path, ["--force-uocr"]))
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return FakeResponse({"blocks": []})

        monkeypatch.setattr(mod._http_session, "post", fake_post)
        mod.procesar_pagina(1, "b64", 0.3)
        assert captured["json"].get("force_uocr") is True
        # El resto del payload sigue intacto
        assert captured["json"].get("ocr_mode") == mod.OCR_MODE

    def test_force_uocr_false_por_defecto_single(self, monkeypatch, tmp_path):
        """Sin --force-uocr → force_uocr=False en el payload single."""
        mod = _load_module(_argv_isolado(tmp_path))
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return FakeResponse({"blocks": []})

        monkeypatch.setattr(mod._http_session, "post", fake_post)
        mod.procesar_pagina(1, "b64", 0.3)
        assert captured["json"].get("force_uocr") is False

    def test_force_uocr_true_en_payload_batch(self, monkeypatch, tmp_path):
        """--force-uocr → procesar_lote incluye force_uocr=True en el batch."""
        mod, captured = _mock_batch_post(
            monkeypatch, tmp_path, [{"blocks": []}, {"blocks": []}],
            argv_extra=["--force-uocr"])
        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])
        assert captured["json"].get("force_uocr") is True
        assert captured["json"].get("images") is not None

    def test_force_uocr_false_por_defecto_batch(self, monkeypatch, tmp_path):
        """Sin --force-uocr → force_uocr=False en el payload batch."""
        mod, captured = _mock_batch_post(monkeypatch, tmp_path, [{"blocks": []}])
        mod.procesar_lote([(1, "b64", 0.1)])
        assert captured["json"].get("force_uocr") is False

    # -- -- doc_id (sesión 126: scope por documento) -- --

    def test_payload_single_incluye_doc_id(self, monkeypatch, tmp_path):
        """procesar_pagina envía doc_id (hash del PDF) — el servidor escopea
        los caches de decisión por documento (cap 47 no hereda del 43)."""
        mod = _load_module(_argv_isolado(tmp_path))
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json", {})
            return FakeResponse({"blocks": []})

        monkeypatch.setattr(mod._http_session, "post", fake_post)
        mod.procesar_pagina(1, "b64", 0.3)

        assert captured["json"].get("doc_id") == mod.DOC_ID
        assert isinstance(mod.DOC_ID, str) and len(mod.DOC_ID) == 12

    def test_doc_id_usa_hash_resistente_a_colisiones(self, tmp_path):
        """El namespace del documento no debe depender de MD5."""
        mod = _load_module(_argv_isolado(tmp_path))
        esperado = hashlib.sha256(
            os.path.basename(mod.PDF_PATH).encode("utf-8")
        ).hexdigest()[:12]

        assert mod.DOC_ID == esperado

    def test_checkpoint_persiste_idiomas_y_metadatos_semanticos(self, tmp_path):
        """La auditoría necesita saber el par y el tipo sin usar glosario."""
        mod = _load_module(_argv_isolado(tmp_path, [
            "--source", "ja", "--target", "es",
        ]))
        mod.total_pages = 1
        mod._registrar_resultado(1, 0.1, 0.2, [{
            "source": "田中",
            "translated": "Tanaka",
            "type": "name",
            "confidence": 0.91,
            "source_lang": "ja",
            "target_lang": "es",
        }])

        checkpoint = mod.build_checkpoint()
        text = checkpoint["results"][0]["texts"][0]
        assert checkpoint["source_lang"] == "ja"
        assert checkpoint["target_lang"] == "es"
        assert text["type"] == "name"
        assert text["confidence"] == 0.91
        assert text["source_lang"] == "ja"

    def test_payload_batch_incluye_doc_id(self, monkeypatch, tmp_path):
        """procesar_lote envía doc_id en el payload del batch."""
        mod, captured = _mock_batch_post(monkeypatch, tmp_path, [{"blocks": []}])
        mod.procesar_lote([(1, "b64", 0.1)])
        assert captured["json"].get("doc_id") == mod.DOC_ID

    def test_force_uocr_warning_si_daemon_no_ready(self, monkeypatch, tmp_path, capsys):
        """main() con --force-uocr y daemon no-ready imprime warning (no aborta)."""
        mod = _load_module(_argv_isolado(tmp_path, ["--force-uocr", "--max-pages", "1"]))
        doc = _FakeDoc(53)
        monkeypatch.setattr(mod.fitz, "open", lambda path: doc)
        monkeypatch.setattr(mod._http_session, "get",
                            lambda url, timeout: FakeResponse(
                                {"ok": True, "version": "test",
                                 "unlimited_ocr": "loading"}))
        monkeypatch.setattr(mod._http_session, "post",
                            lambda *a, **k: FakeResponse(
                                {"blocks": [{"source": "HOLA", "translated": "HELLO"}]}))
        mod.main()
        out = capsys.readouterr().out
        assert "force-uocr" in out and "loading" in out

    # -- -- Fix métrica de tiempos (sesión 115) -- --

    def test_lote_reparte_elapsed_entre_paginas(self, monkeypatch, tmp_path):
        """El elapsed del lote se reparte (per_page) — no hereda el completo.

        Un lote de 2 páginas que tarda 10s debe sumar 10s en page_times
        ([5.0, 5.0]), no 20s ([10.0, 10.0]) como antes del fix.
        """
        mod, _ = _mock_batch_post(
            monkeypatch, tmp_path, [{"blocks": []}, {"blocks": []}])
        # time.time() → 100.0 (t0 del lote), luego 110.0 → elapsed = 10.0s
        seq = [100.0, 110.0]
        monkeypatch.setattr(mod.time, "time", lambda: seq.pop(0))

        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])

        assert mod.page_times == [5.0, 5.0]
        assert sum(mod.page_times) == 10.0  # == elapsed del lote (1 vez, no 2)

    def test_lote_single_suma_elapsed(self, monkeypatch, tmp_path):
        """Lote de 1 página → page_times[0] == elapsed completo (sin dividir)."""
        mod, _ = _mock_batch_post(monkeypatch, tmp_path, [{"blocks": []}])
        seq = [100.0, 107.0]
        monkeypatch.setattr(mod.time, "time", lambda: seq.pop(0))

        mod.procesar_lote([(1, "b64", 0.1)])

        assert mod.page_times == [7.0]

    def test_reporte_final_usa_pared_real(self, monkeypatch, tmp_path, capsys):
        """El REPORTE FINAL imprime 'pared real' (no 'suma') y el promedio (pared)."""
        mod = _load_module(_argv_isolado(tmp_path, ["--max-pages", "2"]))
        doc = _FakeDoc(53)
        monkeypatch.setattr(mod.fitz, "open", lambda path: doc)
        monkeypatch.setattr(mod._http_session, "get",
                            lambda url, timeout: FakeResponse({"ok": True, "version": "test"}))
        monkeypatch.setattr(mod._http_session, "post",
                            lambda *a, **k: FakeResponse(
                                {"blocks": [{"source": "HOLA", "translated": "HELLO"}]}))
        mod.main()
        out = capsys.readouterr().out
        assert "Tiempo total (pared real)" in out
        assert "Tiempo total (suma)" not in out
        assert "Promedio por página (pared)" in out


# ─── procesar_lote (camino batch Fase 1) ─────────────────────────

class TestProcesarLote:
    """procesar_lote registra las N páginas del lote en el MISMO orden y con
    los stats correctos, y cubre los bordes: b64=None (render previo fallido),
    respuesta con menos resultados que páginas (missing_result) y resultados
    no-dict (bad_result). Incluye la regresión del fix NameError: el camino
    batch usa `data = resp.json()` (que el refactor de la sesión 98 sí
    conservó) y debe registrar resultados sin excepción."""

    def test_registra_todas_las_paginas_del_lote(self, monkeypatch, tmp_path):
        """POST /api/process-page-batch mockeado: un lote de 2 páginas registra
        2 resultados en el MISMO orden, con URL y payload correctos, y stats
        agregados coherentes (1 bloque traducido + 1 página vacía)."""
        mod, captured = _mock_batch_post(monkeypatch, tmp_path, [
            {"blocks": [{"source": "HOLA", "translated": "HELLO"}]},
            {"blocks": []},
        ])
        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])

        # Payload: endpoint batch + imágenes en el MISMO orden de entrada
        assert captured["url"] == f"{mod.API_URL}/api/process-page-batch"
        assert captured["json"]["images"] == ["b64", "b64"]
        assert captured["json"]["ocr_mode"] == mod.OCR_MODE
        # Registro: 2 páginas, orden 1→2, stats agregados
        assert mod.pages_done == {1, 2}
        assert len(mod.results) == 2
        r1, r2 = mod.results[0], mod.results[1]
        assert r1["page"] == 1 and r1["blocks"] == 1 and r1["translated"] == 1
        assert r1["status"] == "OK"
        assert r2["page"] == 2 and r2["blocks"] == 0 and r2["translated"] == 0
        assert r2["status"] == "VACIO"
        assert mod.stats["total_blocks_found"] == 1
        assert mod.stats["total_blocks_translated"] == 1
        assert mod.stats["pages_with_text"] == 1
        assert mod.stats["pages_empty"] == 1

    def test_lote_con_b64_none_no_envia_esa_pagina(self, monkeypatch, tmp_path):
        """b64=None dentro del lote (render previo fallido): la página se
        registra como render_error y NO viaja en el payload del batch — el
        resto del lote se procesa normal."""
        mod, captured = _mock_batch_post(monkeypatch, tmp_path, [{"blocks": []}])
        mod.procesar_lote([(1, None, 0.5), (2, "b64", 0.2)])

        # La página 1 (b64 None) no se envía al servidor
        assert captured["json"]["images"] == ["b64"]
        assert mod.pages_done == {1, 2}
        r1 = next(r for r in mod.results if r["page"] == 1)
        assert r1["status"] == "render_error"
        assert r1["time"] == 0.5  # guarda el render_t, no el elapsed del lote
        r2 = next(r for r in mod.results if r["page"] == 2)
        assert r2["status"] == "VACIO"
        assert mod.stats["pages_error"] == 1

    def test_lote_respuesta_con_menos_resultados_que_paginas(self, monkeypatch, tmp_path):
        """El servidor devuelve menos resultados que páginas enviadas (p. ej.
        error interno en una): las páginas faltantes se registran como
        missing_result y cuentan como pages_error — sin perder páginas."""
        mod, _ = _mock_batch_post(monkeypatch, tmp_path, [{"blocks": []}])
        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2), (3, "b64", 0.3)])

        assert mod.pages_done == {1, 2, 3}
        assert len(mod.results) == 3
        assert [(r["page"], r["status"]) for r in mod.results] == [
            (1, "VACIO"), (2, "missing_result"), (3, "missing_result"),
        ]
        assert mod.stats["pages_error"] == 2

    def test_lote_resultado_no_dict_registra_bad_result(self, monkeypatch, tmp_path):
        """Un resultado del lote que no es dict (payload corrupto) → bad_result
        sin romper el resto del lote."""
        mod, _ = _mock_batch_post(monkeypatch, tmp_path, [
            None, {"blocks": [{"source": "X", "translated": "Y"}]},
        ])
        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])

        assert mod.pages_done == {1, 2}
        assert [(r["page"], r["status"]) for r in mod.results] == [
            (1, "bad_result"), (2, "OK"),
        ]
        assert mod.stats["pages_error"] == 1
        assert mod.stats["total_blocks_found"] == 1

    def test_batch_registra_sin_nameerror(self, monkeypatch, tmp_path):
        """Regresión del fix NameError (sesión 109): el camino BATCH debe
        registrar resultados sin excepción — `data = resp.json()` sí existe en
        procesar_lote (el refactor solo lo perdió en procesar_pagina). Con
        bloques mezclados (traducido + SIN_TRAD) los stats son correctos."""
        mod, _ = _mock_batch_post(monkeypatch, tmp_path, [
            {"blocks": [{"source": "HOLA", "translated": "HELLO"}]},
            {"blocks": [{"source": "N", "translated": "N"}]},  # SIN_TRAD
        ])

        # No debe lanzar ninguna excepción
        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])

        assert mod.pages_done == {1, 2}
        assert mod.stats["total_blocks_found"] == 2
        assert mod.stats["total_blocks_translated"] == 1
        assert mod.stats["pages_with_text"] == 2
        assert len(mod.results) == 2
        assert mod.results[0]["status"] == "OK"
        assert mod.results[1]["status"] == "SIN_TRAD"  # 0 de 1 traducido (orig == trad)

    # -- -- Branches de error del lote (sesión 120) -- --

    def test_lote_timeout_definitivo_registra_timeout(self, monkeypatch, tmp_path):
        """Timeout agotando reintentos → TODAS las páginas del lote se registran
        como 'timeout', con el elapsed COMPARTIDO del lote REPARTIDO por página
        (per_page, sesión 115) y pages_done completo."""
        mod = _load_module(_argv_isolado(tmp_path))
        monkeypatch.setattr(mod, "MAX_RETRIES", 0)  # sin reintentos: rápido

        def boom(*a, **k):
            raise requests.Timeout()

        monkeypatch.setattr(mod._http_session, "post", boom)
        # time.time() → 100.0 (t0 del lote), luego 110.0 → elapsed = 10.0s
        seq = [100.0, 110.0]
        monkeypatch.setattr(mod.time, "time", lambda: seq.pop(0))

        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])

        assert mod.pages_done == {1, 2}  # ninguna página se pierde
        assert [(r["page"], r["status"], r["time"]) for r in mod.results] == [
            (1, "timeout", 5.0),   # elapsed 10s / 2 páginas = 5s c/u
            (2, "timeout", 5.0),
        ]
        assert mod.stats["pages_error"] == 2

    def test_lote_conn_error_registra_conn_error_sin_nameerror(self, monkeypatch, tmp_path):
        """Error de conexión (excepción genérica, NO timeout) en todos los
        intentos → todas las páginas como 'conn_error'.

        Regresión del NameError latente (sesión 120): el branch conn_error
        usaba `elapsed` sin definirlo cuando la PRIMERA excepción no era
        Timeout → NameError y el lote se perdía sin registrar nada.
        """
        mod = _load_module(_argv_isolado(tmp_path))
        monkeypatch.setattr(mod, "MAX_RETRIES", 0)

        def boom(*a, **k):
            raise requests.ConnectionError("boom")

        monkeypatch.setattr(mod._http_session, "post", boom)
        seq = [100.0, 110.0]
        monkeypatch.setattr(mod.time, "time", lambda: seq.pop(0))

        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])

        assert mod.pages_done == {1, 2}
        assert [(r["page"], r["status"], r["time"]) for r in mod.results] == [
            (1, "conn_error", 5.0),
            (2, "conn_error", 5.0),
        ]
        assert mod.stats["pages_error"] == 2

    def test_lote_http_error_registra_http_status(self, monkeypatch, tmp_path):
        """HTTP != 200 → TODAS las páginas del lote como 'http_<code>', con el
        elapsed compartido repartido por página y pages_done completo."""
        mod, _ = _mock_batch_post(monkeypatch, tmp_path,
                                  [{"blocks": []}, {"blocks": []}],
                                  status_code=500)
        seq = [100.0, 110.0]
        monkeypatch.setattr(mod.time, "time", lambda: seq.pop(0))

        mod.procesar_lote([(1, "b64", 0.1), (2, "b64", 0.2)])

        assert mod.pages_done == {1, 2}
        assert [(r["page"], r["status"], r["time"]) for r in mod.results] == [
            (1, "http_500", 5.0),
            (2, "http_500", 5.0),
        ]
        assert mod.stats["pages_error"] == 2


# ─── main() + --max-pages ─────────────────────────────────────────

class _FakePix:
    def tobytes(self, fmt="png"):
        return b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


class _FakePage:
    def get_pixmap(self, matrix=None):
        return _FakePix()


class _FakeDoc:
    """PDF fake: len() reporta N páginas; cada __getitem__ cuenta un render."""

    def __init__(self, n_pages):
        self._n = n_pages
        self.renders = 0

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        self.renders += 1
        return _FakePage()

    def close(self):
        pass


class TestMainMaxPages:
    """main() con fitz/requests mockeados: --max-pages limita el render."""

    def _run_main(self, monkeypatch, tmp_path, extra_argv, n_pages=53):
        mod = _load_module(_argv_isolado(tmp_path, extra_argv))
        doc = _FakeDoc(n_pages)
        monkeypatch.setattr(mod.fitz, "open", lambda path: doc)
        monkeypatch.setattr(mod._http_session, "get",
                            lambda url, timeout: FakeResponse({"ok": True, "version": "test"}))
        monkeypatch.setattr(mod._http_session, "post",
                            lambda *a, **k: FakeResponse(
                                {"blocks": [{"source": "HOLA", "translated": "HELLO"}]}))
        mod.main()
        return mod, doc

    def test_max_pages_5_limita_render_y_checkpoint(self, monkeypatch, tmp_path):
        """--max-pages 5 sobre un PDF de 53 → 5 renders y checkpoint de 5."""
        cp = tmp_path / "cp.json"
        mod, doc = self._run_main(monkeypatch, tmp_path, ["--max-pages", "5"])

        assert mod.total_pages == 5
        assert doc.renders == 5, f"Se renderizaron {doc.renders} páginas, se esperaban 5"
        assert mod.stats["pages_translated"] == 5
        assert mod.stats["pages_error"] == 0

        saved = json.loads(cp.read_text(encoding="utf-8"))
        assert saved["total_pages"] == 5
        assert saved["pages_done"] == [1, 2, 3, 4, 5]
        assert saved["stats"]["total_blocks_found"] == 5

    def test_sin_max_pages_procesa_todas(self, monkeypatch, tmp_path):
        """Sin --max-pages (default 0) → se procesan las 53 páginas."""
        mod, doc = self._run_main(monkeypatch, tmp_path, [])

        assert mod.total_pages == 53
        assert doc.renders == 53
        assert mod.stats["pages_translated"] == 53

    def test_max_pages_mayor_que_pdf_se_limita_al_pdf(self, monkeypatch, tmp_path):
        """--max-pages 999 → min(999, len(doc)) = 53 (no crashea)."""
        mod, doc = self._run_main(monkeypatch, tmp_path, ["--max-pages", "999"])
        assert mod.total_pages == 53
        assert doc.renders == 53

    def test_max_pages_cero_procesa_todas(self, monkeypatch, tmp_path):
        """--max-pages 0 (default explícito) → todas las páginas."""
        mod, doc = self._run_main(monkeypatch, tmp_path, ["--max-pages", "0"])
        assert mod.total_pages == 53
        assert doc.renders == 53

    def test_checkpoint_resume_salta_paginas_ya_hechas(self, monkeypatch, tmp_path):
        """Checkpoint previo (total_pages coincidente) → solo se re-renderizan las pendientes."""
        cp = tmp_path / "cp.json"
        cp.write_text(json.dumps({
            "total_pages": 3,
            "pages_done": [1, 2],
            "results": [],
            "page_times": [],
            "stats": {"total_blocks_found": 0, "total_blocks_translated": 0,
                      "pages_with_text": 0, "pages_translated": 0,
                      "pages_empty": 0, "pages_error": 0},
        }, ensure_ascii=False), encoding="utf-8")
        mod, doc = self._run_main(monkeypatch, tmp_path, ["--max-pages", "3"], n_pages=3)

        # Las páginas 1 y 2 ya estaban en el checkpoint: solo se renderiza la 3
        assert doc.renders == 1
        assert mod.pages_done == {1, 2, 3}
        saved = json.loads(cp.read_text(encoding="utf-8"))
        assert saved["pages_done"] == [1, 2, 3]
        assert saved["stats"]["total_blocks_found"] == 1


# ─── main() + --batch-window (integración del camino batch) ──────

class TestMainBatchWindow:
    """main() completo con --batch-window N: verifica el bucle de acumulación
    de lotes (agrupa páginas contiguas en un solo POST /api/process-page-batch),
    que NUNCA se usa el endpoint single, el re-insertado del centinela de fin en
    lotes parciales (sin stall de 60s) y el checkpoint final con páginas en
    orden. Mockea SOLO POST batch (y get health); el single no debe llamarse.
    """

    def _run_batch_main(self, monkeypatch, tmp_path, n_pages, extra_argv):
        """Ejecuta main() en modo batch. Devuelve (mod, doc, captured_payloads)."""
        mod = _load_module(_argv_isolado(tmp_path, extra_argv))
        doc = _FakeDoc(n_pages)
        monkeypatch.setattr(mod.fitz, "open", lambda path: doc)
        monkeypatch.setattr(mod._http_session, "get",
                            lambda url, timeout: FakeResponse({"ok": True, "version": "test"}))
        captured = []

        def fake_post(url, **kwargs):
            # Guarda: el modo batch NUNCA debe llamar al endpoint single
            assert url.endswith("/api/process-page-batch"), \
                f"Se llamó a un endpoint no-batch: {url}"
            assert not url.endswith("/api/process-page"), \
                f"Se llamó al endpoint single: {url}"
            payload = kwargs.get("json", {})
            captured.append(payload)
            # Una entrada de resultado por imagen enviada (páginas del lote)
            n = len(payload.get("images", []))
            return FakeResponse({"results": [
                {"blocks": [{"source": "HOLA", "translated": "HELLO"}]}
                for _ in range(n)
            ]})

        monkeypatch.setattr(mod._http_session, "post", fake_post)
        mod.main()
        return mod, doc, captured

    def test_batch_window_2_agrupa_4_paginas_en_2_lotes(self, monkeypatch, tmp_path):
        """PDF de 4 páginas con --batch-window 2 → 2 POSTs batch de 2 imágenes
        cada uno ([1,2] y [3,4]), nunca el single, y checkpoint final con las
        4 páginas en orden."""
        mod, doc, captured = self._run_batch_main(
            monkeypatch, tmp_path, 4, ["--batch-window", "2"])

        # Bucle de acumulación: 2 lotes de 2 páginas (4/2), solo endpoint batch
        assert len(captured) == 2
        assert [len(p["images"]) for p in captured] == [2, 2]
        assert doc.renders == 4
        # Checkpoint final en orden (páginas 1..4, no en desorden)
        saved = json.loads((tmp_path / "cp.json").read_text(encoding="utf-8"))
        assert saved["pages_done"] == [1, 2, 3, 4]
        assert saved["total_pages"] == 4
        assert saved["stats"]["total_blocks_found"] == 4
        assert saved["stats"]["pages_translated"] == 4
        assert saved["stats"]["pages_error"] == 0

    def test_batch_window_2_lote_parcial_reinserta_centinela(self, monkeypatch, tmp_path):
        """PDF de 3 páginas con --batch-window 2 → lote [1,2] + lote parcial [3]
        (el centinela de fin se encuentra como `extra` y se RE-INSERTA en la
        cola para que la iteración externa lo detecte). Sin el re-insertado,
        main() esperaría 60s de stall con la cola vacía; el test falla si tarda
        más de 10s (el render fake es instantáneo)."""
        t0 = time.time()
        mod, doc, captured = self._run_batch_main(
            monkeypatch, tmp_path, 3, ["--batch-window", "2"])
        wall = time.time() - t0

        # Lote parcial: 2 POSTs, uno con 2 imágenes y otro con 1. Los lotes van
        # a un pool de MAX_WORKERS threads → el ORDEN de ejecución de los POST
        # no está garantizado (el lote [3] podría postear antes que [1,2]), así
        # que el assert es order-independent.
        assert len(captured) == 2
        assert sorted(len(p["images"]) for p in captured) == [1, 2]
        assert doc.renders == 3
        # Checkpoint con las 3 páginas en orden
        saved = json.loads((tmp_path / "cp.json").read_text(encoding="utf-8"))
        assert saved["pages_done"] == [1, 2, 3]
        # No-stall: el re-insertado del centinela evita el timeout de 60s
        assert wall < 10, f"main() tardó {wall:.1f}s — el centinela no se re-insertó"


# ─── Checkpoint default temporal (sesión 133) ──────────────────────

class TestCheckpointDefaultTemporal:
    """El checkpoint por defecto lleva sufijo temporal (YYYYMMDD_HHMM) para
    que dos procesos/corridas no solapen el mismo archivo; --checkpoint-file
    explícito usa el nombre EXACTO (imprescindible para resume)."""

    def test_default_genera_sufijo_temporal(self, monkeypatch):
        """Sin --checkpoint-file → resultados_progreso_YYYYMMDD_HHMM.json."""
        mod = _load_module(["process_all_pages.py"])
        m = re.fullmatch(
            r"resultados_progreso_\d{8}_\d{4}\.json", mod.CHECKPOINT_FILE)
        assert m is not None, \
            f"Checkpoint default inesperado: {mod.CHECKPOINT_FILE}"
        # No es el nombre fijo antiguo (que permitía pisarse entre procesos)
        assert mod.CHECKPOINT_FILE != "resultados_progreso.json"

    def test_checkpoint_file_explicito_usa_nombre_exacto(self, tmp_path):
        """Con --checkpoint-file explícito → se usa el nombre tal cual (resume
        de una corrida anterior, independiente del reloj)."""
        nombre = str(tmp_path / "mi_corrida.json")
        mod = _load_module(["process_all_pages.py", "--checkpoint-file", nombre])
        assert mod.CHECKPOINT_FILE == nombre

    def test_dos_corridas_default_generan_archivos_distintos(self, monkeypatch):
        """Dos corridas sin --checkpoint-file → nombres DIFERENTES siempre
        (el sufijo es la protección contra solapamiento de archivo).

        Determinista: se fuerza `time.strftime` GLOBAL (el módulo process_all_pages
        importa el mismo objeto `time`) a devolver un valor distinto en cada
        carga, independiente del reloj real."""
        secuencia = iter(["20260807_1200", "20260807_1215"])

        def fake_strftime(fmt):
            return next(secuencia)

        monkeypatch.setattr(time, "strftime", fake_strftime)
        mod_a = _load_module(["process_all_pages.py"])
        mod_b = _load_module(["process_all_pages.py"])

        assert mod_a.CHECKPOINT_FILE == "resultados_progreso_20260807_1200.json"
        assert mod_b.CHECKPOINT_FILE == "resultados_progreso_20260807_1215.json"
        assert mod_a.CHECKPOINT_FILE != mod_b.CHECKPOINT_FILE
