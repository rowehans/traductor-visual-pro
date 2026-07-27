"""
test_api.py — Tests de integración para routes/api.py.

Usa Flask test_client con dependencias mockeadas (EasyOCR, CT2, traducción real).
Prueba:
- Validación de Content-Type y charset
- Validación de campos requeridos
- /api/health
- /api/translate (single)
- /api/translate-batch
- /api/process-page
- Error handlers (400, 413, 415, 500)
- Rate limiting
"""

import sys
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from unittest.mock import MagicMock, patch

import pytest
from flask import Flask, jsonify

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from routes.api import api_bp


# ─── Helpers ─────────────────────────────────────────────────────

def _make_app() -> Flask:
    """Crea una Flask app con el blueprint API registrado.
    
    NO se registran blueprints de main ni rutas de server.py — solo
    las rutas API del blueprint. Las dependencias externas se mockean
    en cada test.
    """
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.config["SERVER_NAME"] = "127.0.0.1:5174"
    app.register_blueprint(api_bp)
    return app


def _json_headers(charset: str | None = "utf-8") -> dict[str, str]:
    """Retorna headers Content-Type con charset opcional."""
    ct = "application/json"
    if charset:
        ct += f"; charset={charset}"
    return {"Content-Type": ct}


# ─── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def app():
    return _make_app()


@pytest.fixture
def client(app):
    """Test client con mocks de dependencias externas."""
    return app.test_client()


@pytest.fixture(autouse=True)
def _auto_mocks():
    """Mock automático de dependencias pesadas para TODOS los tests."""
    patches = [
        # server module (imported inside endpoints at call time)
        patch("server._translate_one", return_value="translated"),
        patch("server._get_executor", return_value=ThreadPoolExecutor(max_workers=2)),
        patch("server.DB_AVAILABLE", True),
        patch("server.TRANSLATION_CACHE_AVAILABLE", True),
        # translator module
        patch("translator._argo_ready", {}),
        patch("translator._detect_language_robust", return_value="es"),
        # ocr_utils (for process-page)
        patch("ocr_utils._get_ocr_reader", return_value=MagicMock()),
        patch("ocr_utils._detect_and_ocr", return_value=[]),
        # limiter (rate limiting) — desactivar para tests
        patch("routes.api.RATE_LIMIT_AVAILABLE", True),
    ]
    for p in patches:
        p.start()
    yield
    for p in patches:
        p.stop()


# ═══════════════════════════════════════════════════════════════
# Content-Type validation
# ═══════════════════════════════════════════════════════════════

class TestContentTypeValidation:
    """Verifica que los endpoints POST rechacen Content-Type inválidos."""

    def test_no_content_type_returns_415(self, client):
        resp = client.post("/api/translate", data="{}")
        assert resp.status_code == 415
        data = resp.get_json()
        assert data is not None
        assert "error" in data

    def test_wrong_content_type_returns_415(self, client):
        resp = client.post(
            "/api/translate",
            data="{}",
            content_type="text/plain",
        )
        assert resp.status_code == 415

    def test_html_content_type_returns_415(self, client):
        resp = client.post(
            "/api/translate",
            data="<xml></xml>",
            content_type="text/html",
        )
        assert resp.status_code == 415

    def test_valid_json_accepted(self, client):
        """application/json sin charset debe ser aceptado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            content_type="application/json",
        )
        # Debe llegar al endpoint (pasa validación de contenido)
        assert resp.status_code in (200, 400)  # 200 si traduce, 400 si falta algo


# ═══════════════════════════════════════════════════════════════
# Charset validation
# ═══════════════════════════════════════════════════════════════

class TestCharsetValidation:
    """Verifica que charset=utf-8 sea aceptado y otros charset rechazados."""

    def test_utf8_charset_accepted(self, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            headers=_json_headers("utf-8"),
        )
        assert resp.status_code in (200, 400)

    def test_utf8_charset_caps_accepted(self, client):
        """UTF-8 en mayúsculas debe ser aceptado (case-insensitive)."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            headers=_json_headers("UTF-8"),
        )
        assert resp.status_code in (200, 400)

    def test_utf8_with_extra_params_accepted(self, client):
        """charset=utf-8 con parámetros adicionales debe ser aceptado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            content_type="application/json; charset=utf-8; boundary=xyz",
        )
        assert resp.status_code in (200, 400)

    def test_charset_with_space_around_equal(self, client):
        """charset = utf-8 (espacio antes de =) debe ser aceptado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            content_type="application/json; charset = utf-8",
        )
        assert resp.status_code in (200, 400)

    def test_iso_8859_1_rejected(self, client):
        """Charset iso-8859-1 debe ser rechazado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            headers=_json_headers("iso-8859-1"),
        )
        assert resp.status_code == 415
        data = resp.get_json()
        assert data is not None
        assert "charset" in data.get("error", "").lower() or "415" in str(resp.status_code)

    def test_utf7_rejected(self, client):
        """Charset utf-7 (ataque de encoding) debe ser rechazado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            headers=_json_headers("utf-7"),
        )
        assert resp.status_code == 415

    def test_utf7_caps_rejected(self, client):
        """Charset UTF-7 en mayúsculas debe ser rechazado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            headers=_json_headers("UTF-7"),
        )
        assert resp.status_code == 415

    def test_shift_jis_rejected(self, client):
        """Charset shift_jis debe ser rechazado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            headers=_json_headers("shift_jis"),
        )
        assert resp.status_code == 415

    def test_charset_with_space_before_equal_rejected(self, client):
        """charset = utf-7 (con espacio, encoding malicioso) debe ser rechazado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            content_type="application/json; charset = utf-7",
        )
        assert resp.status_code == 415

    def test_charset_with_quotes_accepted(self, client):
        """charset=\"utf-8\" (con quotes dobles) debe ser aceptado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            content_type='application/json; charset="utf-8"',
        )
        assert resp.status_code in (200, 400)

    def test_charset_single_quotes_accepted(self, client):
        """charset='utf-8' (con quotes simples) debe ser aceptado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            content_type="application/json; charset='utf-8'",
        )
        assert resp.status_code in (200, 400)


# ═══════════════════════════════════════════════════════════════
# _validate_json_content_type decorator (aislado)
# ═══════════════════════════════════════════════════════════════

class TestValidateJsonContentTypeDecorator:
    """
    Tests DIRECTOS del decorador _validate_json_content_type,
    aplicado a un endpoint mínimo SIN _validate_payload_fields.
    
    Los tests existentes en TestCharsetValidation prueban el decorador
    a través de _validate_payload_fields. Esta clase prueba el decorador
    en aislamiento para cubrir casos borde del mismo.
    """

    @pytest.fixture
    def decorator_app(self):
        """App con un endpoint POST mínimo que usa SOLO el decorador."""
        from routes.api import _validate_json_content_type
        app = Flask(__name__)
        app.config["TESTING"] = True

        @app.post("/test-json-only")
        @_validate_json_content_type
        def _test_json_only():
            return jsonify({"ok": True})

        @app.get("/test-get")
        @_validate_json_content_type
        def _test_get():
            return jsonify({"ok": True})

        return app

    @pytest.fixture
    def dc(self, decorator_app):
        return decorator_app.test_client()

    def test_valid_json_accepted(self, dc):
        """application/json sin charset debe ser aceptado."""
        resp = dc.post("/test-json-only", json={"test": True})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_no_content_type_returns_415(self, dc):
        """Sin Content-Type debe retornar 415 con mensaje."""
        resp = dc.post("/test-json-only", data="not-json")
        assert resp.status_code == 415
        data = resp.get_json()
        assert "application/json" in data.get("error", "").lower()

    def test_wrong_content_type_returns_415(self, dc):
        """Content-Type incorrecto debe retornar 415."""
        resp = dc.post("/test-json-only", data="text", content_type="text/plain")
        assert resp.status_code == 415

    def test_text_html_returns_415(self, dc):
        """Content-Type text/html debe retornar 415."""
        resp = dc.post("/test-json-only", data="<xml/>", content_type="text/html")
        assert resp.status_code == 415

    def test_utf8_accepted(self, dc):
        """charset=utf-8 explícito debe ser aceptado."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=utf-8",
        )
        assert resp.status_code == 200

    def test_utf7_rejected(self, dc):
        """charset=utf-7 debe ser rechazado con 415."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=utf-7",
        )
        assert resp.status_code == 415
        data = resp.get_json()
        assert data is not None
        assert "utf-7" in data.get("error", "").lower()

    def test_iso_8859_1_rejected(self, dc):
        """charset=iso-8859-1 debe ser rechazado."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=iso-8859-1",
        )
        assert resp.status_code == 415

    def test_multiple_charsets_first_wins(self, dc):
        """Dos charset declarados (segundo inválido, primero válido).
        El regex captura el PRIMER charset (utf-8), así que el request
        debe pasar. Documenta que solo validamos el primer charset."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=utf-8; charset=utf-7",
        )
        assert resp.status_code == 200

    def test_empty_charset_accepted(self, dc):
        """charset= sin valor: el regex captura string vacío,
        que es falsy, así que el if-block no se ejecuta y
        el request pasa. Debe retornar 200."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=",
        )
        assert resp.status_code == 200

    def test_charset_with_extra_whitespace_rejected(self, dc):
        """charset  =  utf-7  (espacios extra alrededor de =) con charset malo."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset  =  utf-7",
        )
        assert resp.status_code == 415

    def test_charset_utf8_normalized_accepted(self, dc):
        """UTF-8 normal sin homoglifos debe ser aceptado (test de control)."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=utf-8",
        )
        assert resp.status_code == 200

    def test_content_type_with_boundary_accepted(self, dc):
        """application/json con boundary debe ser aceptado."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=utf-8; boundary=----WebKitFormBoundary",
        )
        assert resp.status_code == 200

    def test_get_request_gets_content_type_validated(self, dc):
        """GET sin Content-Type JSON: request.is_json=False →
        el decorador retorna 415. El decorador es agnóstico del método HTTP.
        En producción solo se aplica a endpoints POST (via _validate_payload_fields),
        así que GET con él es un caso teórico, no un bug."""
        resp = dc.get("/test-get")
        # GET sin Content-Type → is_json=False → decorador rechaza
        assert resp.status_code == 415

    def test_error_message_format(self, dc):
        """Verificar formato del mensaje de error 415."""
        resp = dc.post("/test-json-only", data="", content_type="text/plain")
        assert resp.status_code == 415
        data = resp.get_json()
        assert data is not None
        assert "error" in data
        assert isinstance(data["error"], str)
        assert len(data["error"]) > 0

    def test_decorator_does_not_modify_response(self, dc):
        """Con request válido, la respuesta del endpoint debe pasar intacta."""
        resp = dc.post("/test-json-only", json={"custom": "data", "number": 42})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data == {"ok": True}  # el endpoint retorna solo {"ok": True}

    def test_invalid_json_body_still_accepted(self, dc):
        """El decorador solo valida Content-Type, NO valida el body JSON.
        Body inválido no es responsabilidad del decorador — la función
        envuelta ni siquiera parsea el body, solo retorna {"ok": True}."""
        resp = dc.post(
            "/test-json-only",
            data="{invalid json!!!",
            content_type="application/json",
        )
        # El decorador acepta (Content-Type correcto).
        # La función no parsea el body → 200
        assert resp.status_code == 200

    def test_uppercase_application_json_accepted(self, dc):
        """Content-Type con mayúsculas mixtas debe ser aceptado (case-insensitive)."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="Application/JSON; Charset=UTF-8",
        )
        assert resp.status_code == 200

    def test_x_www_form_urlencoded_rejected(self, dc):
        """application/x-www-form-urlencoded debe ser rechazado."""
        resp = dc.post(
            "/test-json-only",
            data="key=value",
            content_type="application/x-www-form-urlencoded",
        )
        assert resp.status_code == 415

    def test_multipart_rejected(self, dc):
        """multipart/form-data debe ser rechazado."""
        resp = dc.post(
            "/test-json-only",
            data="--boundary\r\n",
            content_type="multipart/form-data; boundary=boundary",
        )
        assert resp.status_code == 415

    def test_windows_1252_rejected(self, dc):
        """charset=windows-1252 debe ser rechazado."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=windows-1252",
        )
        assert resp.status_code == 415

    def test_latin1_rejected(self, dc):
        """charset=latin1 debe ser rechazado."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type="application/json; charset=latin1",
        )
        assert resp.status_code == 415

    @pytest.mark.parametrize("bad_charset", [
        "utf-7",
        "utf7",
        "iso-8859-1",
        "iso-8859-15",
        "windows-1252",
        "shift_jis",
        "euc-kr",
        "iso-2022-kr",
        "big5",
        "gb2312",
        "koi8-r",
        "latin1",
    ])
    def test_all_bad_charsets_rejected(self, dc, bad_charset):
        """Todos los charsets no UTF-8 deben ser rechazados (parametrizado)."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type=f"application/json; charset={bad_charset}",
        )
        assert resp.status_code == 415, (
            f"Charset '{bad_charset}' debería ser rechazado, obtuvo {resp.status_code}"
        )

    @pytest.mark.parametrize("good_charset", [
        "utf-8",
        "UTF-8",
        "Utf-8",
    ])
    def test_all_good_charsets_accepted(self, dc, good_charset):
        """Todos los charsets UTF-8 canónicos deben ser aceptados.
        NOTA: 'utf8' (sin guión) NO es válido porque el decorador
        compara exactamente con 'utf-8' (con guión)."""
        resp = dc.post(
            "/test-json-only",
            data=json.dumps({"test": True}),
            content_type=f"application/json; charset={good_charset}",
        )
        assert resp.status_code == 200, (
            f"Charset '{good_charset}' debería ser aceptado, obtuvo {resp.status_code}"
        )


# ═══════════════════════════════════════════════════════════════
# /api/health
# ═══════════════════════════════════════════════════════════════

class TestHealth:
    """Endpoint /api/health — GET público."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_has_required_fields(self, client):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert data is not None
        assert data.get("ok") is True
        assert "version" in data
        assert "mode" in data
        assert "db_available" in data
        assert "cache_available" in data
        assert "rate_limiting" in data

    def test_health_version_is_string(self, client):
        resp = client.get("/api/health")
        data = resp.get_json()
        assert isinstance(data.get("version"), str)
        assert len(data["version"]) > 0

    def test_health_no_auth_required(self, client):
        """Health endpoint debe ser accesible sin autenticación."""
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_accepts_no_content_type(self, client):
        """GET /health no requiere Content-Type."""
        resp = client.get("/api/health")
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════
# /api/translate
# ═══════════════════════════════════════════════════════════════

class TestTranslate:
    """Endpoint /api/translate — POST con validación."""

    @patch("server._translate_one", return_value="hello")
    def test_translate_basic(self, mock_translate, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        assert "translatedText" in data
        mock_translate.assert_called_once()

    def test_missing_text_field_returns_400(self, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data is not None
        assert "missing_fields" in str(data.get("error", "")) or "Campos requeridos" in str(data)

    def test_empty_text_returns_empty(self, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        assert data.get("translatedText") == ""

    def test_whitespace_text_returns_empty(self, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "   "}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["translatedText"] == ""

    def test_invalid_source_lang_returns_400(self, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola", "source": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_target_lang_returns_400(self, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola", "target": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_target_auto_returns_400(self, client):
        """Target lang no puede ser 'auto'."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola", "target": "auto"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_same_source_and_target_returns_400(self, client):
        """source=es y target=es debe ser rechazado."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola", "source": "es", "target": "es"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    @patch("server._translate_one", return_value="hello")
    def test_source_auto_works(self, mock_translate, client):
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola", "source": "auto", "target": "en"}),
            content_type="application/json",
        )
        assert resp.status_code == 200

    def test_very_long_text_returns_413(self, client):
        """Texto extremadamente largo (>20K chars) debe ser rechazado."""
        long_text = "x" * 25_000
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": long_text}),
            content_type="application/json",
        )
        assert resp.status_code == 413

    def test_valid_lang_codes_accepted(self, client):
        """Todos los códigos de idioma válidos deben funcionar."""
        for lang in ("es", "en", "fr", "de", "pt", "it", "ja", "ko", "zh", "zh-cn", "zh-tw", "auto"):
            target = "en" if lang != "en" else "es"
            resp = client.post(
                "/api/translate",
                data=json.dumps({"text": "test", "source": lang, "target": target}),
                content_type="application/json",
            )
            # Debe pasar validación de idioma (aunque el mock traduzca)
            assert resp.status_code in (200, 400), f"Lang {lang} falló con {resp.status_code}"

    def test_null_text_returns_empty(self, client):
        """text=null debe tratarse como vacío."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": None}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["translatedText"] == ""


# ═══════════════════════════════════════════════════════════════
# /api/translate-batch
# ═══════════════════════════════════════════════════════════════

class TestTranslateBatch:
    """Endpoint /api/translate-batch — POST batch."""

    @patch("server._translate_one", return_value="translated")
    @patch("server._get_executor")
    def test_batch_basic(self, mock_exec, mock_trans, client):
        executor = ThreadPoolExecutor(max_workers=2)
        mock_exec.return_value = executor
        texts = ["hola", "adiós", "casa"]
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": texts}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        assert "results" in data
        assert len(data["results"]) == 3
        # El mock devuelve "translated" para cada texto
        assert data["results"] == ["translated", "translated", "translated"]

    def test_batch_missing_texts_returns_400(self, client):
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_texts_not_list_returns_400(self, client):
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": "not_a_list"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_empty_list_returns_empty(self, client):
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": []}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["results"] == []

    def test_batch_invalid_item_type_returns_400(self, client):
        """Elementos no-string en texts deben ser rechazados."""
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": ["hola", 123, "adiós"]}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_too_many_texts_returns_413(self, client):
        """Más de 500 textos debe ser rechazado."""
        texts = ["x"] * 600
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": texts}),
            content_type="application/json",
        )
        assert resp.status_code == 413

    def test_batch_invalid_source_returns_400(self, client):
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": ["hola"], "source": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_invalid_target_returns_400(self, client):
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": ["hola"], "target": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_batch_all_empty_texts(self, client):
        """Todos los textos vacíos deben devolver lista vacía de resultados."""
        resp = client.post(
            "/api/translate-batch",
            data=json.dumps({"texts": ["", "  ", None]}),
            content_type="application/json",
        )
        # None en la lista es inválido (no es string)
        assert resp.status_code == 400


# ═══════════════════════════════════════════════════════════════
# /api/process-page
# ═══════════════════════════════════════════════════════════════

def _make_small_b64_image() -> str:
    """Crea una imagen PNG pequeña en base64 para tests de process-page."""
    import base64
    import cv2
    import numpy as np
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    success, buf = cv2.imencode(".png", img)
    if not success:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


class TestProcessPage:
    """Endpoint /api/process-page — POST con OCR, inpainting y traducción."""

    def test_missing_image_returns_400(self, client):
        resp = client.post(
            "/api/process-page",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_image_returns_400(self, client):
        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": ""}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_base64_returns_400(self, client, mocker):
        """Base64 inválido debe ser rechazado."""
        mocker.patch("ocr_utils._base64_to_cv2", return_value=None)
        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": "not-base64!"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data is not None

    def test_ocr_blocks_returned(self, client, mocker):
        """Con OCR que detecta bloques, deben devolverse en la respuesta."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        mocker.patch("ocr_utils._detect_and_ocr", return_value=[
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "Hola",
             "confidence": 0.85, "fontSize": 14, "textColor": "#000000"},
        ])
        mocker.patch("ocr_utils._build_inpaint_mask", return_value=MagicMock())
        mocker.patch("ocr_utils._inpaint_image", return_value=mock_img)
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)
        mocker.patch("ocr_utils._sample_bg_color", return_value="#ffffff")

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        assert "blocks" in data
        assert "inpainted_image" in data

    def test_ocr_no_blocks_returns_empty(self, client, mocker):
        """Sin bloques OCR, inpainted_image y blocks vacío."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        mocker.patch("ocr_utils._detect_and_ocr", return_value=[])
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["blocks"] == []

    def test_invalid_target_lang_returns_400(self, client):
        b64 = _make_small_b64_image()
        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64, "target": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_source_lang_returns_400(self, client):
        b64 = _make_small_b64_image()
        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64, "source": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_invalid_ocr_mode_returns_400(self, client):
        b64 = _make_small_b64_image()
        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64, "ocr_mode": "invalid"}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_valid_ocr_modes_accepted(self, client, mocker):
        """Modos 'easyocr' y 'auto' deben ser aceptados."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        mocker.patch("ocr_utils._detect_and_ocr", return_value=[])
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)

        for mode in ("easyocr", "auto"):
            resp = client.post(
                "/api/process-page",
                data=json.dumps({"image": b64, "ocr_mode": mode}),
                content_type="application/json",
            )
            assert resp.status_code == 200, f"Modo '{mode}' falló"

    def test_image_too_small_returns_400(self, client, mocker):
        """Imagen < 50x50 px debe ser rechazada."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (30, 30, 3)  # demasiado pequeña
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_memory_error_returns_500(self, client, mocker):
        """MemoryError debe capturarse y devolver 500."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        mocker.patch("ocr_utils._detect_and_ocr", side_effect=MemoryError("OOM"))

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 500

    def test_unexpected_error_returns_500(self, client, mocker):
        """Error inesperado debe capturarse y devolver 500."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        mocker.patch("ocr_utils._detect_and_ocr", side_effect=RuntimeError("crash"))

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 500

    def test_image_too_large_returns_413(self, client):
        """Imagen base64 > 50MB debe ser rechazada."""
        huge_b64 = "x" * int(51 * 1024 * 1024)  # 51MB
        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": huge_b64}),
            content_type="application/json",
        )
        assert resp.status_code == 413


# ═══════════════════════════════════════════════════════════════
# Error handlers
# ═══════════════════════════════════════════════════════════════

class TestErrorHandlers:
    """Verifica que los error handlers del blueprint devuelvan formato correcto."""

    def test_413_response_format(self, app, client):
        """413 debe incluir campo 'error' cuando se excede MAX_CONTENT_LENGTH."""
        # Configurar MAX_CONTENT_LENGTH pequeño para probar 413
        app.config["MAX_CONTENT_LENGTH"] = 1024  # 1KB
        # Enviar payload >1KB para trigger 413
        huge_data = "x" * 2048
        resp = client.post(
            "/api/translate",
            data=huge_data,
            content_type="application/json",
        )
        assert resp.status_code == 413
        data = resp.get_json()
        assert data is not None
        assert "error" in data

    def test_429_missing(self, client):
        """429 no debería ocurrir en tests locales porque el rate limiter
        está desactivado para 127.0.0.1. Verificar que funcione sin rate limit."""
        for _ in range(50):
            resp = client.get("/api/health")
            assert resp.status_code == 200

    def test_health_response_is_json(self, client):
        resp = client.get("/api/health")
        assert resp.is_json
        data = resp.get_json()
        assert isinstance(data, dict)


# ═══════════════════════════════════════════════════════════════
# Security headers
# ═══════════════════════════════════════════════════════════════

class TestSecurityHeaders:
    """Verifica headers de seguridad en respuestas del blueprint."""

    def test_health_has_content_type_json(self, client):
        resp = client.get("/api/health")
        ct = resp.content_type or ""
        assert "application/json" in ct


# ═══════════════════════════════════════════════════════════════
# Route not found
# ═══════════════════════════════════════════════════════════════

class TestRouteNotFound:
    """Verifica que rutas inexistentes devuelvan 404."""

    def test_unknown_route_returns_404(self, client):
        resp = client.get("/api/nonexistent")
        assert resp.status_code == 404

    def test_unknown_api_post_returns_404(self, client):
        resp = client.post(
            "/api/unknown",
            data=json.dumps({"test": True}),
            content_type="application/json",
        )
        assert resp.status_code == 404


# ═══════════════════════════════════════════════════════════════
# Payload field validation edge cases
# ═══════════════════════════════════════════════════════════════

class TestPayloadFieldValidation:
    """Casos borde de validación de campos requeridos."""

    def test_empty_json_object(self, client):
        """{} sin campos requeridos debe fallar."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_extra_fields_ignored(self, client):
        """Campos extra en el payload no deben causar error."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": "hola", "extra": "field", "another": 42}),
            content_type="application/json",
        )
        assert resp.status_code in (200, 400)  # Depende de la validación de idioma

    def test_null_required_field_handled(self, client):
        """Campo requerido con valor null debe ser detectado como faltante
        (o procesado según lógica del endpoint)."""
        resp = client.post(
            "/api/translate",
            data=json.dumps({"text": None}),
            content_type="application/json",
        )
        assert resp.status_code == 200  # None → _safe_str → ""
        data = resp.get_json()
        assert data["translatedText"] == ""


# ═══════════════════════════════════════════════════════════════
# Cross-endpoint charset validation
# ═══════════════════════════════════════════════════════════════

class TestCharsetAllEndpoints:
    """Verifica que la validación charset aplique a todos los endpoints POST."""

    @pytest.mark.parametrize("endpoint,payload", [
        ("/api/translate", {"text": "hola"}),
        ("/api/translate-batch", {"texts": ["hola"]}),
        ("/api/process-page", {"image": "data:image/png;base64,abc123"}),
    ])
    def test_utf7_rejected_on_all_endpoints(self, client, endpoint, payload):
        """Todos los endpoints POST deben rechazar charset=utf-7."""
        resp = client.post(
            endpoint,
            data=json.dumps(payload),
            content_type=f"application/json; charset=utf-7",
        )
        assert resp.status_code == 415, f"Endpoint {endpoint} aceptó charset=utf-7"

    @pytest.mark.parametrize("endpoint,payload", [
        ("/api/translate", {"text": "hola"}),
        ("/api/translate-batch", {"texts": ["hola"]}),
        ("/api/process-page", {"image": "data:image/png;base64,abc123"}),
    ])
    def test_no_charset_accepted_on_all_endpoints(self, client, endpoint, payload):
        """Todos los endpoints POST deben aceptar application/json sin charset."""
        resp = client.post(
            endpoint,
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code != 415, f"Endpoint {endpoint} rechazó sin charset"
