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

# Función real capturada a nivel de módulo: el fixture autouse parchea
# ocr_utils._detect_and_ocr a [] para TODOS los tests; para probar el camino
# de degradación v4.2 necesitamos la referencia original (capturada aquí,
# antes de que corra el fixture).
import ocr_utils as _ocr_utils_mod
_REAL_DETECT_AND_OCR = _ocr_utils_mod._detect_and_ocr


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
        # Fase 4 (default=fusion): _has_big_panel corre la heurística real sobre
        # el MagicMock — mockarla explícitamente hace el test determinista
        # (1 bloque conf 0.85 → sin trigger, sin daemon).
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
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
        """Sin bloques OCR, inpainted_image y blocks vacío.

        Fase 4 (default=fusion): 0 bloques dispara el trigger v4.2 — mockear
        el camino de refuerzo completo (Fase 2 reintento + daemon caído) para
        que degrade silenciosamente al híbrido vacío.
        """
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        mocker.patch("ocr_utils._detect_and_ocr", return_value=[])
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="")
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        mocker.patch("routes.api._ocr_with_unlimited",
                     side_effect=RuntimeError("daemon no listo (mock)"))
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["blocks"] == []

    def test_default_ocr_mode_es_fusion(self, client, mocker):
        """Fase 4: POST sin ocr_mode usa fusion (página fácil → sin daemon,
        ocr_engine='fusion')."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        # 3 bloques con conf alta → el trigger v4.2 NO dispara el daemon
        mocker.patch("ocr_utils._detect_and_ocr", return_value=[
            {"x": 10, "y": 10, "w": 50, "h": 15, "text": f"t{i}",
             "confidence": 0.5, "fontSize": 12, "textColor": "#000"}
            for i in range(3)
        ])
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._filter_watermarks_from_blocks",
                     side_effect=lambda b: b)
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited")
        mocker.patch("ocr_utils._build_inpaint_mask", return_value=mock_img)
        mocker.patch("ocr_utils._inpaint_image", return_value=mock_img)
        mocker.patch("ocr_utils._sample_bg_color", return_value="#ffffff")
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)
        mocker.patch("server._translate_one", return_value="TRANSLATED")

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ocr_engine"] == "fusion"
        assert data.get("engines_used") == ["easyocr+rapid"]
        uocr_mock.assert_not_called()

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

    def test_doc_id_se_pasa_a_ocrmanager(self, client, mocker):
        """Sesión 126: el campo opcional doc_id del payload se propaga a
        OCRManager.run_ocr (escopea los caches de decisión por documento)."""
        import ocr_engine
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        run_ocr_mock = mocker.patch(
            "ocr_engine.OCRManager.run_ocr",
            return_value=([], "fusion", ["easyocr+rapid"]),
        )
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64, "doc_id": "cap47"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert run_ocr_mock.call_args.kwargs.get("doc_id") == "cap47"

    def test_doc_id_ausente_default_vacio(self, client, mocker):
        """Sin doc_id en el payload → run_ocr recibe doc_id="" (scope legacy
        compartido, comportamiento previo intacto)."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        run_ocr_mock = mocker.patch(
            "ocr_engine.OCRManager.run_ocr",
            return_value=([], "fusion", ["easyocr+rapid"]),
        )
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert run_ocr_mock.call_args.kwargs.get("doc_id") == ""

    def test_valid_ocr_modes_accepted(self, client, mocker):
        """Modos 'easyocr', 'auto' y 'fusion' deben ser aceptados."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        mocker.patch("ocr_utils._detect_and_ocr", return_value=[])
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)
        # El modo fusion dispara el refuerzo U-OCR (0 bloques) — mockear el
        # daemon para que degrade silenciosamente al híbrido (comportamiento real
        # cuando el daemon no está disponible).
        mocker.patch("routes.api._ocr_with_unlimited",
                     side_effect=RuntimeError("daemon no listo (mock)"))

        for mode in ("easyocr", "auto", "fusion"):
            resp = client.post(
                "/api/process-page",
                data=json.dumps({"image": b64, "ocr_mode": mode}),
                content_type="application/json",
            )
            assert resp.status_code == 200, f"Modo '{mode}' falló"

    def test_fusion_uses_uocr_when_trigger_and_merges(self, client, mocker):
        """Modo fusion: si la página es difícil (<3 bloques), llama a U-OCR y
        fusiona bloques híbridos + U-OCR."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        # Híbrido devuelve 2 bloques con confianza baja (<0.2, v4.2) → dispara refuerzo
        hybrid_blocks = [
            {"x": 10, "y": 10, "w": 50, "h": 15, "text": "hola",
             "confidence": 0.15, "fontSize": 12, "textColor": "#000"},
            {"x": 70, "y": 10, "w": 50, "h": 15, "text": "mundo",
             "confidence": 0.15, "fontSize": 12, "textColor": "#000"},
        ]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid_blocks)
        # U-OCR devuelve un bloque adicional de alta confianza heurística
        uocr_blocks = [
            {"x": 200, "y": 50, "w": 80, "h": 20, "text": "título dorado",
             "confidence": 0.93, "fontSize": 16, "textColor": "#000",
             "engine": "unlimited"},
        ]
        # Firma actual: (blocks, image_panels, t_ocr_s) — los paneles image
        # son la materia prima de la Ruta C (bubble re-OCR). Sin paneles en
        # este test, no se dispara la recuperación de globos.
        mocker.patch("routes.api._ocr_with_unlimited",
                     return_value=(uocr_blocks, [], 5.0))
        mocker.patch("ocr_utils._filter_watermarks_from_blocks",
                     side_effect=lambda b: b)
        # Mocks para inpainting/traducción (lo que sigue tras OCR)
        mocker.patch("ocr_utils._build_inpaint_mask", return_value=mock_img)
        mocker.patch("ocr_utils._inpaint_image", return_value=mock_img)
        mocker.patch("ocr_utils._sample_bg_color", return_value="#ffffff")
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)
        mocker.patch("server._translate_one", return_value="TRANSLATED")

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64, "ocr_mode": "fusion"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        # 2 híbridos + 1 U-OCR = 3 bloques fusionados
        assert len(data["blocks"]) == 3
        assert data["ocr_engine"] == "fusion"
        assert "unlimited" in data.get("engines_used", [])
        # El bloque de U-OCR no se descarta (confianza heurística alta)
        sources = [b["source"] for b in data["blocks"]]
        assert "título dorado" in sources

    def test_fusion_does_not_trigger_when_conf_high(self, client, mocker):
        """v4.2: con conf >= 0.2 y >= 3 bloques, el trigger NO dispara U-OCR.

        Frontera del trigger selectivo: antes (0.25) esta página disparaba
        refuerzo; ahora (0.20 estricto) no debe llamar al daemon.
        """
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        # 3 bloques con conf 0.5 (bien por encima del umbral 0.2)
        hybrid_blocks = [
            {"x": 10, "y": 10, "w": 50, "h": 15, "text": f"t{i}",
             "confidence": 0.5, "fontSize": 12, "textColor": "#000"}
            for i in range(3)
        ]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=hybrid_blocks)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._filter_watermarks_from_blocks",
                     side_effect=lambda b: b)
        uocr_mock = mocker.patch("routes.api._ocr_with_unlimited")
        mocker.patch("ocr_utils._build_inpaint_mask", return_value=mock_img)
        mocker.patch("ocr_utils._inpaint_image", return_value=mock_img)
        mocker.patch("ocr_utils._sample_bg_color", return_value="#ffffff")
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)
        mocker.patch("server._translate_one", return_value="TRANSLATED")

        resp = client.post(
            "/api/process-page",
            data=json.dumps({"image": b64, "ocr_mode": "fusion"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        uocr_mock.assert_not_called()
        data = resp.get_json()
        assert "unlimited" not in data.get("engines_used", [])

    def test_detect_and_ocr_degrades_to_rapid_when_daemon_inferring(self, mocker):
        """v4.2: con _uocr_inferring activo, _detect_and_ocr degrada a RapidOCR
        CPU SIN cargar el reader de EasyOCR (GPU)."""
        from ocr_utils import _uocr_inferring
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        fake_rapid_blocks = [{"x": 0, "y": 0, "w": 10, "h": 10, "text": "cpu",
                              "confidence": 0.7}]
        # El fixture autouse parchea _detect_and_ocr a [] — usar la referencia
        # real capturada a nivel de módulo para probar el código de degradación.
        reader_mock = mocker.patch("ocr_utils._get_ocr_reader")
        mocker.patch("ocr_utils._preprocess_rapid", return_value=mock_img)
        mocker.patch("ocr_utils._run_rapidocr", return_value=fake_rapid_blocks)
        try:
            _uocr_inferring.set()
            blocks = _REAL_DETECT_AND_OCR(mock_img, "es")
        finally:
            _uocr_inferring.clear()
        assert blocks == fake_rapid_blocks
        # El reader de EasyOCR (GPU) NO debe cargarse durante la degradación
        reader_mock.assert_not_called()

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

    def test_ocr_with_unlimited_propaga_tipo_semantico(self, mocker):
        """Fase 3: los bloques U-OCR conservan el type semántico
        (text/title/header) hasta la fusión; image/footer se filtran."""
        import numpy as np
        import routes.api
        import uocr_client

        mocker.patch.object(uocr_client, "health",
                            return_value={"state": "ready"})
        mocker.patch.object(uocr_client, "process_page", return_value={
            "blocks": [
                {"x": 10, "y": 10, "w": 100, "h": 30, "type": "title",
                 "text": "CAPITULO 43"},
                {"x": 50, "y": 100, "w": 80, "h": 20, "type": "text",
                 "text": "Hola"},
                {"x": 0, "y": 0, "w": 640, "h": 15, "type": "header",
                 "text": "4.58 p.m"},
                {"x": 0, "y": 0, "w": 500, "h": 400, "type": "image",
                 "text": ""},
                {"x": 0, "y": 0, "w": 100, "h": 15, "type": "footer",
                 "text": "p. 12"},
            ],
            "infer_s": 5.0,
        })
        img = np.zeros((800, 600, 3), dtype=np.uint8)  # 800x600: panel image 500x400 = 41% > 15%
        blocks, panels, t = routes.api._ocr_with_unlimited(img)

        by_text = {b["text"]: b for b in blocks}
        assert by_text["CAPITULO 43"]["type"] == "title"
        assert by_text["Hola"]["type"] == "text"
        assert by_text["4.58 p.m"]["type"] == "header"
        # image (panel grande) va a image_panels, no a blocks; footer es ruido
        assert all(b.get("type") != "image" for b in blocks)
        assert "p. 12" not in by_text
        assert len(panels) == 1
        assert panels[0]["w"] == 500

    def test_ocr_with_unlimited_batch_parsea_multi_imagen(self, mocker):
        """Fase 1: _ocr_with_unlimited_batch mapea los bloques del batch
        (infer_multi) por página y propaga type semántico a cada una."""
        import numpy as np
        import routes.api
        import uocr_client

        mocker.patch.object(uocr_client, "health",
                            return_value={"state": "ready"})
        mocker.patch.object(uocr_client, "process_batch", return_value={
            "pages": [
                {"blocks": [
                    {"x": 10, "y": 10, "w": 100, "h": 30, "type": "title",
                     "text": "CAPITULO 43"},
                    {"x": 0, "y": 0, "w": 500, "h": 400, "type": "image",
                     "text": ""},
                ], "recovered_from_art": 0},
                {"blocks": [
                    {"x": 50, "y": 100, "w": 80, "h": 20, "type": "text",
                     "text": "Hola"},
                ], "recovered_from_art": 0},
            ],
            "infer_s": 7.0,
        })
        img_a = np.zeros((800, 600, 3), dtype=np.uint8)
        img_b = np.zeros((800, 600, 3), dtype=np.uint8)

        pages, infer_s = routes.api._ocr_with_unlimited_batch([img_a, img_b])

        assert infer_s == 7.0
        assert len(pages) == 2
        blocks_a, panels_a = pages[0]
        blocks_b, panels_b = pages[1]
        assert {b["text"]: b["type"] for b in blocks_a} == {"CAPITULO 43": "title"}
        assert len(panels_a) == 1  # image grande → panel de Ruta C
        assert blocks_b[0]["text"] == "Hola"
        assert blocks_b[0]["type"] == "text"
        assert blocks_b[0]["engine"] == "unlimited"

    def test_process_page_batch_rechaza_fuera_de_rango(self, client):
        """Menos de 1 o más de 4 imágenes → 400."""
        resp = client.post(
            "/api/process-page-batch",
            data=json.dumps({"images": []}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        resp = client.post(
            "/api/process-page-batch",
            data=json.dumps({"images": ["a", "b", "c", "d", "e"]}),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_process_page_batch_devuelve_por_pagina(self, client, mocker):
        """2 imágenes → 2 resultados en el mismo orden, sin daemon si el
        híbrido resuelve (conf alta, >=3 bloques)."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        ok_blocks = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": f"T{i}",
             "confidence": 0.85, "fontSize": 14, "textColor": "#000000"}
            for i in range(3)
        ]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=ok_blocks)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        batch_mock = mocker.patch("routes.api._ocr_with_unlimited_batch")
        mocker.patch("ocr_utils._build_inpaint_mask", return_value=MagicMock())
        mocker.patch("ocr_utils._inpaint_image", return_value=mock_img)
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)
        mocker.patch("ocr_utils._sample_bg_color", return_value="#ffffff")

        resp = client.post(
            "/api/process-page-batch",
            data=json.dumps({"images": [b64, b64]}),
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        batch_mock.assert_not_called()
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert r["ocr_engine"] == "fusion"
            assert r["engines_used"] == ["easyocr+rapid"]
            assert len(r["blocks"]) == 3
            assert "inpainted_image" in r

    def test_process_page_batch_doc_id_se_pasa_a_ocrmanager(self, client, mocker):
        """Sesión 126: doc_id del payload batch llega a run_ocr_batch."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        run_batch_mock = mocker.patch(
            "ocr_engine.OCRManager.run_ocr_batch",
            return_value=[([], "fusion", ["easyocr+rapid"]),
                          ([], "fusion", ["easyocr+rapid"])],
        )
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)

        resp = client.post(
            "/api/process-page-batch",
            data=json.dumps({"images": [b64, b64], "doc_id": "cap47"}),
            content_type="application/json",
        )
        assert resp.status_code == 200
        assert run_batch_mock.call_args.kwargs.get("doc_id") == "cap47"

    def test_process_page_batch_llama_daemon_una_vez(self, client, mocker):
        """2 páginas difíciles → el daemon batch se llama UNA vez con 2 imgs."""
        b64 = _make_small_b64_image()
        mock_img = MagicMock()
        mock_img.shape = (100, 100, 3)
        mocker.patch("ocr_utils._base64_to_cv2", return_value=mock_img)
        bad_blocks = [
            {"x": 10, "y": 20, "w": 80, "h": 15, "text": "hola",
             "confidence": 0.15, "fontSize": 14, "textColor": "#000000"}
        ]
        mocker.patch("ocr_utils._detect_and_ocr", return_value=bad_blocks)
        mocker.patch("ocr_utils._page_has_large_image_panel", return_value=False)
        mocker.patch("ocr_utils._page_signature", return_value="")
        mocker.patch("ocr_utils._preprocess_rapid", side_effect=lambda x: x)
        mocker.patch("ocr_utils._run_rapidocr", return_value=[])
        ublocks = [
            {"x": 50, "y": 50, "w": 80, "h": 20, "text": "título",
             "confidence": 0.93, "fontSize": 16, "textColor": "#000000"}
        ]
        batch_mock = mocker.patch(
            "routes.api._ocr_with_unlimited_batch",
            return_value=([(ublocks, []), (ublocks, [])], 5.0),
        )
        mocker.patch("ocr_utils._detect_bubble_regions_in_panel", return_value=[])
        mocker.patch("ocr_utils._fusionar_blocks_multi",
                     side_effect=lambda sources, weights:
                     list(sources[0]) + list(sources[1]))
        mocker.patch("ocr_utils._build_inpaint_mask", return_value=MagicMock())
        mocker.patch("ocr_utils._inpaint_image", return_value=mock_img)
        mocker.patch("ocr_utils._cv2_to_base64", return_value=b64)
        mocker.patch("ocr_utils._sample_bg_color", return_value="#ffffff")

        resp = client.post(
            "/api/process-page-batch",
            data=json.dumps({"images": [b64, b64]}),
            content_type="application/json",
        )

        assert resp.status_code == 200
        data = resp.get_json()
        assert data is not None
        assert batch_mock.call_count == 1
        assert len(batch_mock.call_args.args[0]) == 2
        engines = data["results"][0]["engines_used"]
        assert "unlimited-batch" in engines


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
