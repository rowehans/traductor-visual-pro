"""Tests de main.py (entry point del .exe): acceso directo de escritorio,
cwd en modo frozen, ocultar consola, navegador y arranque del servidor —
con sys.frozen, ctypes y subprocess mockeados (sin procesos ni navegadores
reales)."""

import os
import socket
import subprocess
import sys
import threading
import time
import types
import webbrowser
from pathlib import Path
from typing import Any

import pytest

import main


@pytest.fixture(autouse=True)
def _main_aislado(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin frozen, sin chdir real: cada test decide si activa el modo .exe."""
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(os, "chdir", lambda p: None)


def _patched_ctypes_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reemplaza ctypes por un módulo sin sub-módulos: `import ctypes.wintypes`
    falla y el código cae al fallback del escritorio (~/Desktop)."""
    monkeypatch.setitem(sys.modules, "ctypes", types.ModuleType("ctypes"))


def test_create_shortcut_no_frozen_no_op(monkeypatch: pytest.MonkeyPatch,
                                         capsys: pytest.CaptureFixture[str]) -> None:
    def fail_run(*args: object, **kw: object) -> object:
        raise AssertionError("no debe ejecutarse subprocess sin frozen")

    monkeypatch.setattr(subprocess, "run", fail_run)
    main._create_desktop_shortcut()
    assert capsys.readouterr().out == ""


def test_create_shortcut_platform_no_win32(monkeypatch: pytest.MonkeyPatch,
                                           capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "frozen", True)
    monkeypatch.setattr(sys, "platform", "linux")

    def fail_run(*args: object, **kw: object) -> object:
        raise AssertionError("no debe ejecutarse subprocess fuera de win32")

    monkeypatch.setattr(subprocess, "run", fail_run)
    main._create_desktop_shortcut()
    assert capsys.readouterr().out == ""


def test_create_shortcut_crea_lnk(monkeypatch: pytest.MonkeyPatch,
                                  tmp_path: Path,
                                  capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "frozen", True)
    monkeypatch.setattr(sys, "platform", "win32")
    exe = tmp_path / "app.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(exe))
    _patched_ctypes_fake(monkeypatch)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(tmp_path / "Desktop"))
    monkeypatch.setattr(os.path, "exists", lambda p: p == str(exe))

    seen: list[list[str]] = []

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        seen.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, "OK", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    main._create_desktop_shortcut()

    assert "Acceso directo creado" in capsys.readouterr().out
    assert len(seen) == 1
    assert seen[0][0].endswith("powershell.exe")


def test_create_shortcut_ya_existe_no_sobrescribe(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "frozen", True)
    monkeypatch.setattr(sys, "platform", "win32")
    exe = tmp_path / "app.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(exe))
    _patched_ctypes_fake(monkeypatch)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(tmp_path / "Desktop"))
    # El .lnk ya existe → se corta antes de tocar PowerShell.
    monkeypatch.setattr(os.path, "exists", lambda p: True)

    def fail_run(*args: object, **kw: object) -> object:
        raise AssertionError("no debe recrearse un acceso directo existente")

    monkeypatch.setattr(subprocess, "run", fail_run)

    main._create_desktop_shortcut()

    assert capsys.readouterr().out == ""


def test_create_shortcut_error_de_powershell(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "frozen", True)
    monkeypatch.setattr(sys, "platform", "win32")
    exe = tmp_path / "app.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(exe))
    _patched_ctypes_fake(monkeypatch)
    monkeypatch.setattr(os.path, "expanduser",
                        lambda p: str(tmp_path / "Desktop"))
    monkeypatch.setattr(os.path, "exists", lambda p: p == str(exe))

    def boom(*args: object, **kw: object) -> object:
        raise RuntimeError("ps falló")

    monkeypatch.setattr(subprocess, "run", boom)

    main._create_desktop_shortcut()

    assert "No se pudo crear acceso directo" in capsys.readouterr().out


def test_fix_cwd_no_frozen_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    path_before = list(sys.path)
    main._fix_cwd()
    assert list(sys.path) == path_before


def test_fix_cwd_frozen_encuentra_env(monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Path) -> None:
    monkeypatch.setattr(sys, "frozen", True)
    exe = tmp_path / "app.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "meipass"),
                        raising=False)
    site_packages = tmp_path / "env" / "Lib" / "site-packages"
    site_packages.mkdir(parents=True)

    path_before = list(sys.path)
    try:
        main._fix_cwd()
        assert str(site_packages) in sys.path
        assert sys.path[0] == str(tmp_path / "meipass")
    finally:
        sys.path[:] = path_before


def test_fix_cwd_frozen_sin_env_avisa(monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Path,
                                      capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "frozen", True)
    exe = tmp_path / "app.exe"
    exe.write_text("x", encoding="utf-8")
    monkeypatch.setattr(sys, "executable", str(exe))
    # Nada de site-packages existe (ni en el árbol ni en ubicaciones fijas).
    monkeypatch.setattr(os.path, "exists", lambda p: p == str(exe))

    main._fix_cwd()

    assert "env no encontrado" in capsys.readouterr().out


def test_hide_console_no_frozen_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", False)
    main._hide_console()  # no-op: no debe explotar ni hacer nada


def test_hide_console_frozen_win32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "frozen", True)
    monkeypatch.setattr(sys, "platform", "win32")

    shown: list[tuple[int, int]] = []

    class _Kernel:
        @staticmethod
        def GetConsoleWindow() -> int:
            return 7

    class _User:
        @staticmethod
        def ShowWindow(hwnd: int, cmd: int) -> int:
            shown.append((hwnd, cmd))
            return 0

    class _Windll:
        kernel32 = _Kernel()
        user32 = _User()

    fake_ctypes = types.ModuleType("ctypes")
    setattr(fake_ctypes, "windll", _Windll())
    monkeypatch.setitem(sys.modules, "ctypes", fake_ctypes)

    main._hide_console()

    assert shown == [(7, 0)]  # SW_HIDE


def test_wait_for_port_abre(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Conn:
        def __enter__(self) -> "_Conn":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(socket, "create_connection",
                        lambda addr, timeout: _Conn())
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert main.wait_for_port(timeout=5) is True


def test_wait_for_port_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(addr: tuple[str, int], timeout: float) -> object:
        raise OSError("sin servidor")

    monkeypatch.setattr(socket, "create_connection", refused)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    # Reloj ficticio: avanza en cada consulta para que el deadline expire.
    now = {"t": 1000.0}

    def fake_time() -> float:
        now["t"] += 0.5
        return now["t"]

    monkeypatch.setattr(time, "time", fake_time)
    assert main.wait_for_port(timeout=1) is False


def test_open_browser_chrome(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_chrome = "C:/fake/chrome.exe"
    monkeypatch.setattr(os.path, "expandvars",
                        lambda s: fake_chrome if "Google" in s else s)
    monkeypatch.setattr(os.path, "exists", lambda p: p == fake_chrome)

    seen: list[list[str]] = []

    def fake_popen(args: list[str], **kw: object) -> object:
        seen.append(list(args))
        return object()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    main.open_browser()

    assert len(seen) == 1
    assert seen[0][0] == fake_chrome
    assert seen[0][1] == "--app=http://127.0.0.1:5174"


def test_open_browser_fallback_webbrowser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os.path, "expandvars", lambda s: s)
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    def no_chrome(*a: object, **k: object) -> object:
        raise RuntimeError("sin chrome")

    monkeypatch.setattr(subprocess, "Popen", no_chrome)

    opened: list[str] = []

    def _open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", _open)

    main.open_browser()

    assert opened == ["http://127.0.0.1:5174"]


def test_run_server_arranca_waitress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fase 2.1: run_server delega en waitress.serve (no en app.run)."""
    monkeypatch.setattr(main, "_fix_cwd", lambda: None)
    calls: list[dict[str, object]] = []

    class _App:
        pass

    fake_server = types.ModuleType("server")
    setattr(fake_server, "app", _App())
    monkeypatch.setitem(sys.modules, "server", fake_server)

    def fake_serve(app: object, **kw: object) -> None:
        calls.append({"app": app, **kw})

    fake_waitress = types.ModuleType("waitress")
    setattr(fake_waitress, "serve", fake_serve)
    monkeypatch.setitem(sys.modules, "waitress", fake_waitress)

    main.run_server()

    assert len(calls) == 1
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 5174
    assert calls[0]["threads"] == 8


def test_run_launcher_reutiliza_sesion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_fix_cwd", lambda: None)
    monkeypatch.setattr(main, "_hide_console", lambda: None)
    monkeypatch.setattr(main, "_create_desktop_shortcut", lambda: None)
    monkeypatch.setattr(main, "wait_for_port", lambda timeout=60: True)
    opened: list[str] = []

    def _open_browser() -> None:
        opened.append("browser")

    monkeypatch.setattr(main, "open_browser", _open_browser)
    ran_server: list[bool] = []

    def _run_server() -> None:
        ran_server.append(True)

    monkeypatch.setattr(main, "run_server", _run_server)

    main.run_launcher()

    assert opened == ["browser"]
    assert ran_server == []


def test_run_launcher_arranca_servidor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "_fix_cwd", lambda: None)
    monkeypatch.setattr(main, "_hide_console", lambda: None)
    monkeypatch.setattr(main, "_create_desktop_shortcut", lambda: None)
    # Primera llamada (run_launcher): no hay servidor; la del thread: sí.
    estado = {"n": 0}

    def wait_port(timeout: int = 60) -> bool:
        estado["n"] += 1
        return estado["n"] > 1

    monkeypatch.setattr(main, "wait_for_port", wait_port)
    opened: list[str] = []

    def _open_browser() -> None:
        opened.append("browser")

    monkeypatch.setattr(main, "open_browser", _open_browser)
    ran_server: list[bool] = []

    def _run_server() -> None:
        ran_server.append(True)

    monkeypatch.setattr(main, "run_server", _run_server)

    # El thread del navegador se ejecuta de forma síncrona en el test.
    class _FakeThread:
        def __init__(self, target: Any = None, daemon: object = None) -> None:
            self._target: Any = target

        def start(self) -> None:
            if self._target is not None:
                self._target()

    monkeypatch.setattr(threading, "Thread", _FakeThread)

    main.run_launcher()

    assert ran_server == [True]
    assert opened == ["browser"]
