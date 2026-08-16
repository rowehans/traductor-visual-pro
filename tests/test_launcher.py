"""Tests del lanzador (launcher.py): arranque completo con argv/Popen
mockeados — sin procesos reales, sin navegador y sin sleeps."""

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import pytest

import launcher


class _FakeProc:
    """Popen simulado del servidor: registra wait/terminate."""

    def __init__(self, wait_raises: type[BaseException] | None = None) -> None:
        self.waits = 0
        self.terminated = False
        self._wait_raises = wait_raises

    def wait(self, timeout: float | None = None) -> int:
        self.waits += 1
        if self._wait_raises is not None:
            raise self._wait_raises
        return 0

    def terminate(self) -> None:
        self.terminated = True


class _Conn:
    """Context manager de socket simulado (with ... as s: return True)."""

    def __enter__(self) -> "_Conn":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture(autouse=True)
def _launcher_aislado(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Sin procesos reales: chdir, sleeps y navegador mockeados.

    Devuelve la lista de URLs que se intentaron abrir en el navegador, para
    que los tests verifiquen cuántas veces se abrió.
    """
    opened: list[str] = []
    monkeypatch.setattr(os, "chdir", lambda p: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def _open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", _open)
    return opened


def test_port_open_true_cuando_hay_conexion(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_connect(addr: tuple[str, int], timeout: float) -> _Conn:
        return _Conn()

    monkeypatch.setattr(socket, "create_connection", fake_connect)
    assert launcher.port_open("127.0.0.1", 5174) is True


def test_port_open_false_sin_conexion(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(addr: tuple[str, int], timeout: float) -> _Conn:
        raise ConnectionRefusedError()

    monkeypatch.setattr(socket, "create_connection", refused)
    assert launcher.port_open("127.0.0.1", 5174) is False


def test_port_open_false_con_error_os(monkeypatch: pytest.MonkeyPatch) -> None:
    def os_error(addr: tuple[str, int], timeout: float) -> _Conn:
        raise OSError("boom")

    monkeypatch.setattr(socket, "create_connection", os_error)
    assert launcher.port_open("127.0.0.1", 5174) is False


def test_main_sin_python_devuelve_1(monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(launcher, "PYTHON",
                        Path("ruta/inexistente/python.exe"))
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    rc = launcher.main([])
    assert rc == 1
    assert "No se encuentra Python" in capsys.readouterr().out


def test_main_reutiliza_servidor_activo(monkeypatch: pytest.MonkeyPatch,
                                        _launcher_aislado: list[str],
                                        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(launcher, "port_open", lambda host, port: True)
    rc = launcher.main([])
    assert rc == 0
    assert "reutilizando sesión" in capsys.readouterr().out
    assert len(_launcher_aislado) == 1  # navegador abierto una vez


def test_main_reutiliza_activo_con_cpu_avisa_que_no_aplica(
        monkeypatch: pytest.MonkeyPatch, _launcher_aislado: list[str],
        capsys: pytest.CaptureFixture[str]) -> None:
    # --cpu con un servidor ya activo: la sesión en curso no cambia de modo;
    # se avisa y se reutiliza igual.
    monkeypatch.setattr(launcher, "port_open", lambda host, port: True)
    rc = launcher.main(["--cpu"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "reutilizando sesión" in out
    assert "no se aplica a la sesión en curso" in out
    assert len(_launcher_aislado) == 1


def test_main_arranca_servidor_y_abre_navegador(
        monkeypatch: pytest.MonkeyPatch, _launcher_aislado: list[str],
        capsys: pytest.CaptureFixture[str]) -> None:
    llamadas = {"n": 0}

    def port_open(host: str, port: int) -> bool:
        llamadas["n"] += 1
        return llamadas["n"] > 1  # primero no hay servidor; luego sí

    proc = _FakeProc()
    popen_calls: list[list[str]] = []

    def fake_popen(cmd: list[str], **kw: object) -> _FakeProc:
        popen_calls.append(cmd)
        return proc

    monkeypatch.setattr(launcher, "port_open", port_open)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    rc = launcher.main([])

    assert rc == 0
    assert len(popen_calls) == 1
    cmd = popen_calls[0]
    assert cmd[0].endswith("python.exe")
    assert cmd[2].endswith("server.py")
    assert "Servidor listo" in capsys.readouterr().out
    assert len(_launcher_aislado) == 1


def test_main_servidor_no_responde_abre_igual(
        monkeypatch: pytest.MonkeyPatch, _launcher_aislado: list[str],
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(launcher, "port_open", lambda host, port: False)
    proc = _FakeProc()
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: proc)

    rc = launcher.main([])

    assert rc == 0
    assert "no respondió a tiempo" in capsys.readouterr().out
    assert len(_launcher_aislado) == 1
    assert proc.waits == 1


def test_main_ctrl_c_detiene_el_servidor(
        monkeypatch: pytest.MonkeyPatch, _launcher_aislado: list[str],
        capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(launcher, "port_open", lambda host, port: False)
    proc = _FakeProc(wait_raises=KeyboardInterrupt)
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **kw: proc)

    rc = launcher.main([])

    assert rc == 0
    assert proc.terminated is True
    assert "Servidor detenido" in capsys.readouterr().out


def test_main_cpu_inyecta_env_al_servidor(
        monkeypatch: pytest.MonkeyPatch, _launcher_aislado: list[str],
        capsys: pytest.CaptureFixture[str]) -> None:
    # --cpu: el subproceso del servidor recibe UOCR_MODO_CPU=1 en su entorno
    # (sin tocar config.py) — config.MODO_CPU se activa en el hijo por la env.
    llamadas = {"n": 0}

    def port_open(host: str, port: int) -> bool:
        llamadas["n"] += 1
        return llamadas["n"] > 1

    envs: list[dict[str, str]] = []
    monkeypatch.setattr(launcher, "port_open", port_open)
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda cmd, **kw: envs.append(kw["env"]) or _FakeProc(),
    )

    rc = launcher.main(["--cpu"])

    assert rc == 0
    assert len(envs) == 1
    assert envs[0].get(launcher.MODO_CPU_ENV) == "1"
    assert "MODO_CPU" in capsys.readouterr().out


def test_main_sin_cpu_no_inyecta_env(
        monkeypatch: pytest.MonkeyPatch, _launcher_aislado: list[str]) -> None:
    llamadas = {"n": 0}

    def port_open(host: str, port: int) -> bool:
        llamadas["n"] += 1
        return llamadas["n"] > 1

    envs: list[dict[str, str]] = []
    monkeypatch.setattr(launcher, "port_open", port_open)
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda cmd, **kw: envs.append(kw["env"]) or _FakeProc(),
    )

    rc = launcher.main([])

    assert rc == 0
    assert len(envs) == 1
    assert launcher.MODO_CPU_ENV not in envs[0]


def test_config_modo_cpu_se_activa_por_env() -> None:
    # La cadena completa: la env que inyecta el launcher activa
    # config.MODO_CPU en un proceso NUEVO (subproceso limpio, sin el módulo
    # ya importado por la suite).
    env = dict(os.environ)
    env["UOCR_MODO_CPU"] = "1"
    r = subprocess.run(  # nosec
        [sys.executable, "-c",
         "import config; print(config.MODO_CPU)"],
        capture_output=True, text=True, env=env, cwd=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))),
        timeout=60,
    )
    assert r.returncode == 0, r.stderr
    assert "True" in r.stdout

    # Sin la env: False (default del preset).
    env.pop("UOCR_MODO_CPU", None)
    r2 = subprocess.run(  # nosec
        [sys.executable, "-c",
         "import config; print(config.MODO_CPU)"],
        capture_output=True, text=True, env=env, cwd=os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))),
        timeout=60,
    )
    assert r2.returncode == 0, r2.stderr
    assert "False" in r2.stdout
