"""test_uocr_client.py — Cobertura del cliente del daemon U-OCR.

Prueba la resolución de la raíz (dev/frozen), el ciclo de vida del daemon
(spawn/adoptar/relanzar/degradar), health, esperas y envío de páginas/batch
— todo con subprocess/Popen/urlopen mockeados (sin lanzar procesos reales).
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

import uocr_client as uc


class _FakeProc:
    def __init__(self, pid: int = 1234, alive: bool = True,
                 returncode: int | None = None) -> None:
        self.pid = pid
        self._alive = alive
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self._alive:
            return None
        return self.returncode if self.returncode is not None else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


class TestResolveRoot:
    def test_dev_raiz_del_proyecto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delattr(sys, "frozen", raising=False)
        root = uc._resolve_root()
        assert (root / "uocr_daemon.py").exists()
        assert (root / "uocr_client.py").exists()

    def test_frozen_sube_buscando_env_uocr(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        exe = tmp_path / "dist" / "main" / "app.exe"
        exe.parent.mkdir(parents=True)
        env_py = tmp_path / "env_uocr_gpu" / "Scripts" / "python.exe"
        env_py.parent.mkdir(parents=True)
        env_py.touch()

        monkeypatch.setattr(sys, "frozen", True, raising=False)
        monkeypatch.setattr(sys, "executable", str(exe))

        assert uc._resolve_root() == tmp_path


class TestAvailableAndRunning:
    def test_available_true_cuando_existe_entorno(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        py = tmp_path / "env_uocr_gpu" / "Scripts" / "python.exe"
        script = tmp_path / "uocr_daemon.py"
        py.parent.mkdir(parents=True)
        py.touch()
        script.touch()
        monkeypatch.setattr(uc, "UOCR_PYTHON", py)
        monkeypatch.setattr(uc, "DAEMON_SCRIPT", script)
        assert uc.available() is True

    def test_available_false_sin_entorno(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr(uc, "UOCR_PYTHON", tmp_path / "no_existe.exe")
        monkeypatch.setattr(uc, "DAEMON_SCRIPT", tmp_path / "no_existe.py")
        assert uc.available() is False

    def test_is_daemon_running(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "_proc", _FakeProc(alive=True))
        assert uc.is_daemon_running() is True
        monkeypatch.setattr(uc, "_proc", _FakeProc(alive=False))
        assert uc.is_daemon_running() is False
        monkeypatch.setattr(uc, "_proc", None)
        assert uc.is_daemon_running() is False


class TestSpawnDaemon:
    def test_ya_corriendo_devuelve_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "_proc", _FakeProc(alive=True))
        assert uc.spawn_daemon() is True

    def test_adopta_daemon_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "_proc", None)
        monkeypatch.setattr(uc, "health", lambda: {"state": "ready"})
        assert uc.spawn_daemon() is True

    def test_adopta_daemon_loading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "_proc", None)
        monkeypatch.setattr(uc, "health", lambda: {"state": "loading"})
        assert uc.spawn_daemon() is True

    def test_estado_error_sin_hijo_propio_no_mata_nada(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "_proc", None)
        monkeypatch.setattr(uc, "health", lambda: {
            "state": "error", "error": "proceso no identificado"})
        monkeypatch.setattr(uc, "available", lambda: True)
        assert uc.spawn_daemon() is False

    def test_no_disponible_devuelve_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "_proc", None)
        monkeypatch.setattr(uc, "health", lambda: {"state": "offline"})
        monkeypatch.setattr(uc, "available", lambda: False)
        assert uc.spawn_daemon() is False

    def test_daemon_muere_al_arrancar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        muerto = _FakeProc(pid=1, alive=False, returncode=2)
        monkeypatch.setattr(uc, "_proc", None)
        monkeypatch.setattr(uc, "health", lambda: {"state": "offline"})
        monkeypatch.setattr(uc, "available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: muerto)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        assert uc.spawn_daemon() is False

    def test_lanzado_sin_health_aun_responde(self, monkeypatch: pytest.MonkeyPatch) -> None:
        vivo = _FakeProc(pid=7, alive=True)
        monkeypatch.setattr(uc, "_proc", None)
        monkeypatch.setattr(uc, "health", lambda: {"state": "offline"})
        monkeypatch.setattr(uc, "available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: vivo)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        # Tras 10 intentos sin /health, se reporta lanzado igualmente.
        assert uc.spawn_daemon() is True

    def test_popen_recibe_flags_de_entorno(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}
        vivo = _FakeProc(pid=7, alive=True)

        def fake_popen(args: list[str], **kw: object) -> _FakeProc:
            captured["args"] = args
            captured["env"] = kw.get("env")
            return vivo

        monkeypatch.setattr(uc, "_proc", None)
        states = iter(("offline", "loading"))
        monkeypatch.setattr(uc, "health",
                            lambda: {"state": next(states)})
        monkeypatch.setattr(uc, "available", lambda: True)
        monkeypatch.setattr(subprocess, "Popen", fake_popen)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        assert uc.spawn_daemon() is True
        args = captured["args"]
        assert isinstance(args, list)
        assert args[0] == str(uc.UOCR_PYTHON)
        assert str(uc.DAEMON_SCRIPT) in args
        env = captured["env"]
        assert isinstance(env, dict)
        assert "HF_HOME" in env


class TestHealth:
    def test_health_ok(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "_request",
                            lambda *a, **k: {"state": "ready"})
        assert uc.health()["state"] == "ready"

    def test_health_offline_sin_excepcion(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*a: object, **k: object) -> dict[str, object]:
            raise ConnectionError("no responde")

        monkeypatch.setattr(uc, "_request", _boom)
        h = uc.health()
        assert h["state"] == "offline"


class TestWaitReady:
    def test_listo_devuelve_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "health",
                            lambda: {"state": "ready"})
        assert uc.wait_ready(timeout_s=5.0) is True

    def test_error_devuelve_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "health",
                            lambda: {"state": "error", "error": "x"})
        assert uc.wait_ready(timeout_s=5.0) is False

    def test_timeout_devuelve_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _Clock:
            def __init__(self) -> None:
                self.now = 0.0

            def __call__(self) -> float:
                return self.now

        clock = _Clock()
        monkeypatch.setattr(time, "time", clock)
        monkeypatch.setattr(time, "sleep", lambda s: setattr(clock, "now",
                                                              clock.now + s))
        monkeypatch.setattr(uc, "health", lambda: {"state": "loading"})

        assert uc.wait_ready(timeout_s=10.0) is False


class TestRequest:
    def test_request_envia_json_y_parsea(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        class _FakeResp:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

            def __enter__(self) -> "_FakeResp":
                return self

            def __exit__(self, *a: object) -> None:
                return None

        def fake_urlopen(req: urllib.request.Request,
                         timeout: float = 5.0) -> _FakeResp:
            captured["req"] = req
            captured["timeout"] = timeout
            return _FakeResp(b'{"ok": true}')

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        out = uc._request("POST", "/ocr", {"image_path": "x.png"}, timeout=3.0)

        assert out == {"ok": True}
        assert captured["timeout"] == 3.0
        req = captured["req"]
        assert isinstance(req, urllib.request.Request)
        assert req.method == "POST"
        assert json.loads(req.data) == {"image_path": "x.png"}  # type: ignore[arg-type]


class TestProcessPage:
    def test_no_listo_devuelve_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "wait_ready", lambda t: False)
        r = uc.process_page("img.png", wait_timeout_s=0.5)
        assert r["error"] == "modelo no listo"

    def test_ok_envia_imagen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake_request(method: str, path: str,
                         payload: dict[str, object] | None = None,
                         timeout: float = 5.0) -> dict[str, object]:
            seen["path"] = path
            seen["payload"] = payload
            return {"text": "hola", "blocks": []}

        monkeypatch.setattr(uc, "wait_ready", lambda t: True)
        monkeypatch.setattr(uc, "_request", fake_request)

        r = uc.process_page("img.png", max_length=2048, wait_timeout_s=1.0)

        assert r["text"] == "hola"
        assert seen["path"] == "/ocr"
        payload = seen["payload"]
        assert isinstance(payload, dict)
        assert payload["image_path"] == "img.png"
        assert payload["max_length"] == 2048

    def test_error_comunicacion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*a: object, **k: object) -> dict[str, object]:
            raise TimeoutError("cayo")

        monkeypatch.setattr(uc, "wait_ready", lambda t: True)
        monkeypatch.setattr(uc, "_request", _boom)
        r = uc.process_page("img.png", wait_timeout_s=1.0)
        assert "error de comunicación" in r["error"]

    def test_default_max_length_usa_constante_de_config(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Fase 2.3: el default del cliente es UOCR_MAX_LENGTH (no 4096 hardcodeado)."""
        seen: dict[str, object] = {}

        def fake_request(method: str, path: str,
                         payload: dict[str, object] | None = None,
                         timeout: float = 5.0) -> dict[str, object]:
            seen["payload"] = payload
            return {"text": "hola", "blocks": []}

        monkeypatch.setattr(uc, "wait_ready", lambda t: True)
        monkeypatch.setattr(uc, "_request", fake_request)

        # Sin max_length explícito → el default debe ser la constante.
        uc.process_page("img.png", wait_timeout_s=1.0)
        payload = seen["payload"]
        assert isinstance(payload, dict)
        from config import UOCR_MAX_LENGTH
        assert payload["max_length"] == UOCR_MAX_LENGTH
        assert payload["max_length"] < 4096


class TestProcessBatch:
    def test_vacio_devuelve_error(self) -> None:
        r = uc.process_batch([])
        assert r["error"] == "lista de imágenes vacía"
        assert r["pages"] == []

    def test_no_listo_devuelve_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(uc, "wait_ready", lambda t: False)
        r = uc.process_batch(["a.png"], wait_timeout_s=0.5)
        assert r["error"] == "modelo no listo"

    def test_ok_envia_batch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, object] = {}

        def fake_request(method: str, path: str,
                         payload: dict[str, object] | None = None,
                         timeout: float = 5.0) -> dict[str, object]:
            seen["path"] = path
            seen["payload"] = payload
            return {"pages": [{"blocks": []}], "n_images": 2}

        monkeypatch.setattr(uc, "wait_ready", lambda t: True)
        monkeypatch.setattr(uc, "_request", fake_request)

        r = uc.process_batch(["a.png", "b.png"], max_length=512,
                             wait_timeout_s=1.0)

        assert r["n_images"] == 2
        assert seen["path"] == "/ocr-batch"
        payload = seen["payload"]
        assert isinstance(payload, dict)
        assert payload["images"] == ["a.png", "b.png"]
        assert payload["max_length"] == 512

    def test_error_comunicacion(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*a: object, **k: object) -> dict[str, object]:
            raise ConnectionError("cayo")

        monkeypatch.setattr(uc, "wait_ready", lambda t: True)
        monkeypatch.setattr(uc, "_request", _boom)
        r = uc.process_batch(["a.png"], wait_timeout_s=1.0)
        assert "error de comunicación" in r["error"]
