"""test_server.py — Cobertura del servidor Flask (server.py).

Importa ``server`` con los hilos de precarga BLOQUEADOS: en tests no debe
cargarse EasyOCR/CT2/YOLO (ni lanzarse el daemon U-OCR). Ejercita los
endpoints de app, los security headers, el wrapper de traducción con caché,
el executor compartido y la degradación de los preloads sin dependencias.
"""
import threading as _threading
from typing import Any

import pytest
from werkzeug.exceptions import NotFound

# Bloquear los hilos de precarga ANTES de importar server: los threads de
# preload lanzarían EasyOCR/CT2 (y el daemon U-OCR) durante los tests.
_RealThread = _threading.Thread


class _NoStartThread(_RealThread):
    def start(self) -> None:
        pass


_threading.Thread = _NoStartThread  # type: ignore[misc]
try:
    import server
finally:
    _threading.Thread = _RealThread  # type: ignore[misc]

from config import APP_VERSION, CSP_POLICY, TIMEOUT_TRANSLATE_MS  # noqa: E402


# Ruta de prueba para verificar la compresión (Fase 2.2): se registra a nivel
# de módulo ANTES de que ningún test dispare el primer request (Flask cierra
# el setup del app en el primer request y ya no admite rutas nuevas).
if not server.app.view_functions.get("_big_json_test"):
    @server.app.get("/_big_json_test")
    def _big_json_test() -> Any:  # firma de route de Flask
        # > 1 KB de JSON repetitivo (bien comprimible).
        from flask import jsonify
        return jsonify({"data": ["frase de prueba repetida " * 40] * 20})


class TestAppEndpoints:
    def test_serve_config_devuelve_estructura(self) -> None:
        client = server.app.test_client()
        rv = client.get("/api/config")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["version"] == APP_VERSION
        assert "languages" in data
        assert data["timeouts_ms"]["translate"] == TIMEOUT_TRANSLATE_MS

    def test_serve_config_expone_modo_cpu_y_ocr_scale(self, monkeypatch) -> None:
        """Preset modo_cpu: /api/config expone MODO_CPU y la escala de render
        efectiva que el frontend debe usar — 1.2 normal, MODO_CPU_OCR_SCALE
        (0.8) con el preset activo. Se lee en runtime para poder verificar
        ambas variantes sin reiniciar el servidor."""
        import config
        client = server.app.test_client()

        # Default (sin GPU dedicada apagado) → ocr_scale 1.2 (comportamiento
        # histórico: el frontend sigue renderizando a la escala de producción).
        monkeypatch.setattr(config, "MODO_CPU", False)
        rv = client.get("/api/config")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["modo_cpu"] is False
        assert data["ocr_scale"] == 1.2

        # Preset activo → modo_cpu True + escala reducida (menos píxeles).
        monkeypatch.setattr(config, "MODO_CPU", True)
        rv = client.get("/api/config")
        data = rv.get_json()
        assert data["modo_cpu"] is True
        assert data["ocr_scale"] == config.MODO_CPU_OCR_SCALE
        assert data["ocr_scale"] < 1.2

    def test_serve_js_y_css(self) -> None:
        client = server.app.test_client()
        assert client.get("/app.min.js").status_code == 200
        assert client.get("/styles.min.css").status_code == 200

    def test_serve_root_archivo_existente(self) -> None:
        client = server.app.test_client()
        rv = client.get("/README.md")
        assert rv.status_code == 200
        assert len(rv.data) > 0

    def test_serve_root_archivo_inexistente_404(self) -> None:
        client = server.app.test_client()
        assert client.get("/no_existe_xyz_abc.py").status_code == 404

    def test_serve_root_bloquea_path_traversal(self) -> None:
        with server.app.test_request_context("/x"):
            with pytest.raises(NotFound):
                server.serve_root("../../../../../etc/passwd")

    def test_respuesta_grande_se_comprime_con_gzip(self) -> None:
        """Fase 2.2: JSON > 1 KB se comprime con gzip cuando el cliente lo anuncia."""
        import gzip
        client = server.app.test_client()
        rv = client.get("/_big_json_test",
                        headers={"Accept-Encoding": "gzip"})
        assert rv.status_code == 200
        assert rv.headers.get("Content-Encoding") == "gzip"
        assert "Vary" in rv.headers
        raw = gzip.decompress(rv.data)
        assert b"frase de prueba" in raw

    def test_respuesta_pequena_no_se_comprime(self) -> None:
        """Fase 2.2: respuestas < 1 KB o sin Accept-Encoding van sin comprimir."""
        client = server.app.test_client()
        # Sin Accept-Encoding → sin gzip (el cliente no lo anuncia).
        rv = client.get("/api/config")
        assert rv.status_code == 200
        assert rv.headers.get("Content-Encoding") is None
        rv.get_json()  # el cuerpo sigue siendo JSON válido sin comprimir


class TestSecurityHeaders:
    def test_headers_seguridad_en_respuesta(self) -> None:
        client = server.app.test_client()
        rv = client.get("/api/config")
        assert rv.headers["Content-Security-Policy"] == CSP_POLICY
        assert rv.headers["X-Frame-Options"] == "DENY"
        assert rv.headers["X-Content-Type-Options"] == "nosniff"
        assert rv.headers["Referrer-Policy"] == "no-referrer"

    def test_add_security_headers_devuelve_la_misma_respuesta(self) -> None:
        class _FakeResponse:
            def __init__(self) -> None:
                self.headers: dict[str, str] = {}

        resp = _FakeResponse()
        out = server.add_security_headers(resp)
        assert out is resp
        assert resp.headers["X-Frame-Options"] == "DENY"


class TestTranslationWrapper:
    def test_translate_one_delega_con_cache(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, dict[str, object]] = {}

        def _fake_impl(text: str, source: str, target: str,
                       **kw: object) -> str:
            seen["kw"] = dict(kw)
            return "HOLA"

        monkeypatch.setattr(server, "_translate_one_impl", _fake_impl)
        out = server._translate_one("hola", "es", "en", block_type="text")
        assert out == "HOLA"
        kw = seen["kw"]
        assert kw["block_type"] == "text"
        # En este entorno la caché está disponible: se pasa cache_get/cache_set.
        assert kw["translation_cache_available"] is True


class TestExecutor:
    def test_executor_se_recrea_tras_shutdown(self) -> None:
        ex1 = server._get_executor()
        assert ex1 is not None
        server.shutdown_executor()
        assert server._executor_shutdown is True
        ex2 = server._get_executor()
        assert ex2 is not ex1
        assert server._executor_shutdown is False

    def test_shutdown_idempotente(self) -> None:
        server.shutdown_executor()
        server.shutdown_executor()  # segunda llamada: no-op
        assert server._executor_shutdown is True


class TestPreloads:
    def test_preload_background_degrada_sin_dependencias(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*a: object, **k: object) -> None:
            raise RuntimeError("no disponible")

        monkeypatch.setattr("ocr_utils._get_ocr_reader", _boom)
        monkeypatch.setattr("ocr_utils._get_rapid_engine", _boom)
        monkeypatch.setattr("ocr_utils._get_spellchecker", _boom)
        monkeypatch.setattr("ocr_utils._get_foreign_spellchecker", _boom)
        monkeypatch.setattr("ocr_utils._get_yolo_engine", _boom)
        monkeypatch.setattr("translator._get_ct2_translator", _boom)
        # No debe crashear: cada bloque degrada con un mensaje.
        server._preload_background()

    def test_preload_background_camino_feliz(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakeReader:
            pass

        class _FakeTranslator:
            device = "cpu"

        monkeypatch.setattr("ocr_utils._get_ocr_reader",
                            lambda lang: _FakeReader())
        monkeypatch.setattr("ocr_utils._get_rapid_engine",
                            lambda: _FakeReader())
        monkeypatch.setattr("ocr_utils._get_spellchecker",
                            lambda: _FakeReader())
        monkeypatch.setattr("ocr_utils._get_foreign_spellchecker",
                            lambda lang: _FakeReader())
        monkeypatch.setattr("ocr_utils._get_yolo_engine",
                            lambda: _FakeReader())
        monkeypatch.setattr(
            "translator._get_ct2_translator",
            lambda a, b: (_FakeTranslator(), None))
        server._preload_background()  # sin excepciones

    def test_preload_carga_diccionarios_extranjeros(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        # 2026-08-15: la precarga debe incluir en/pt (_get_foreign_spellchecker)
        # para que la PRIMERA página con mezcla latina no pague ~350 ms de
        # carga bajo demanda dentro del tiempo medido.
        import server as server_mod

        calls: list[str] = []
        monkeypatch.setattr("ocr_utils._get_ocr_reader",
                            lambda lang: object())
        monkeypatch.setattr("ocr_utils._get_rapid_engine", lambda: object())
        monkeypatch.setattr("ocr_utils._get_spellchecker", lambda: object())

        def fake_foreign(lang: str) -> object:
            calls.append(lang)
            return object()

        monkeypatch.setattr("ocr_utils._get_foreign_spellchecker",
                            fake_foreign)
        monkeypatch.setattr("ocr_utils._get_yolo_engine", lambda: object())
        monkeypatch.setattr(
            "translator._get_ct2_translator",
            lambda a, b: (object(), None))
        server_mod._preload_background()
        assert "en" in calls and "pt" in calls

    def test_preload_unlimited_daemon_degrada(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("uocr_client.spawn_daemon", lambda: False)
        server._preload_unlimited_daemon()

    def test_preload_unlimited_daemon_lanza(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("uocr_client.spawn_daemon", lambda: True)
        server._preload_unlimited_daemon()
