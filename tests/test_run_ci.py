"""Regresiones del runner CI local."""

import json
import os
import signal
import subprocess
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any

import pytest

import run_ci


def _fake_completed(args: list[str]) -> subprocess.CompletedProcess[str]:
    """CompletedProcess simulado: returncode 0 sin stdout (salta JS/bandit)."""
    return subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")


def test_pytest_files_are_unique_and_include_processing_regressions() -> None:
    files = run_ci._pytest_test_files()

    assert len(files) == len(set(files))
    assert "tests/test_process_all_pages.py" in files
    assert "tests/test_correccion_detector.py" in files


def test_pytest_command_uses_usable_basetemp(monkeypatch: pytest.MonkeyPatch,
                                             tmp_path: Path) -> None:
    base = tmp_path / ".tmp_pytest_ci"
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP", base)
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP_ALT", tmp_path / ".tmp_pytest_ci_alt")

    command = run_ci._build_pytest_command(with_coverage=False)

    assert "--basetemp" in command
    assert Path(command[command.index("--basetemp") + 1]) == base


def test_pytest_command_falls_back_when_basetemp_restricted(monkeypatch: pytest.MonkeyPatch,
                                                           tmp_path: Path) -> None:
    base = tmp_path / ".tmp_pytest_ci"
    alt = tmp_path / ".tmp_pytest_ci_alt"
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP", base)
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP_ALT", alt)

    # Simular ACL inaccesible en el basetemp primario pero usable el alternativo
    monkeypatch.setattr(run_ci, "_basetemp_usable", lambda p: p == alt)

    command = run_ci._build_pytest_command(with_coverage=False)

    assert "--basetemp" in command
    assert Path(command[command.index("--basetemp") + 1]) == alt


def test_pytest_command_falls_back_to_system_temp(monkeypatch: pytest.MonkeyPatch,
                                                 tmp_path: Path) -> None:
    base = tmp_path / ".tmp_pytest_ci"
    alt = tmp_path / ".tmp_pytest_ci_alt"
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP", base)
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP_ALT", alt)

    # Ambos inaccesibles → tempfile del sistema (mockeado para no dejar basura)
    monkeypatch.setattr(run_ci, "_basetemp_usable", lambda p: False)
    monkeypatch.setattr(
        "tempfile.mkdtemp", lambda **kw: str(tmp_path / "system_tmp")
    )

    command = run_ci._build_pytest_command(with_coverage=False)

    assert "--basetemp" in command
    assert Path(command[command.index("--basetemp") + 1]) == tmp_path / "system_tmp"


def test_pytest_command_solo_reporte_json(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """El comando pytest emite el JSON (fuente del gate y del HTML); el HTML
    lo genera _write_coverage_html desde el JSON, no pytest-cov (que arrastraría
    los scripts de dev al reporte)."""
    base = tmp_path / ".tmp_pytest_ci"
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP", base)
    monkeypatch.setattr(run_ci, "PYTEST_BASETEMP_ALT", tmp_path / ".tmp_pytest_ci_alt")

    cmd = run_ci._build_pytest_command(with_coverage=True)
    assert any(a.startswith("--cov-report=json:") for a in cmd)
    assert not any(a.startswith("--cov-report=html:") for a in cmd)

    cmd_off = run_ci._build_pytest_command(with_coverage=False)
    assert not any(a.startswith("--cov-report=json:") for a in cmd_off)


def test_syntax_check_incluye_modulos_es6_del_frontend() -> None:
    files = run_ci._js_syntax_files()

    assert files[0] == "app.js"
    assert {
        "js/config.js",
        "js/filters.js",
        "js/theme.js",
        "js/toast.js",
        "js/utils.js",
    }.issubset(files)


def test_prod_files_unique_and_complete() -> None:
    """_PROD_PY_FILES no debe tener duplicados y debe cubrir los módulos clave."""
    files = run_ci._PROD_PY_FILES

    assert len(files) == len(set(files))
    assert "server.py" in files
    assert "config.py" in files
    assert "translator.py" in files
    assert "ocr_engine.py" in files
    assert "uocr_daemon.py" in files
    assert "routes/api.py" in files
    assert "routes/main.py" in files


def test_prod_py_files_discovery_contract() -> None:
    """El descubrimiento automático produce EXACTAMENTE los módulos de producción.

    Fija el contrato del walk (raíz + routes/, sin scripts de dev): si un
    archivo nuevo aparece o desaparece de la lista, este test lo señala para
    que el cambio sea intencional (módulo nuevo → actualizar el baseline;
    script de dev nuevo → agregarlo a los filtros de _discover_prod_py_files).
    """
    files = set(run_ci._PROD_PY_FILES)
    known = set(run_ci._KNOWN_PROD_PY_FILES)

    # El walk y el baseline deben coincidir: no debe haber archivos sin
    # clasificar (si los hay, el CI emite un [WARN] al correr).
    assert files == known, (
        "Archivos sin clasificar detectados: "
        f"{sorted(files - known)}. Clasifícalos en _KNOWN_PROD_PY_FILES "
        "(producción) o en los filtros de dev."
    )

    # Los scripts de dev conocidos NO deben colarse en la auditoría.
    dev_scripts = {
        "benchmark_ocr.py", "test_all_pages.py", "generate_report.py",
        "build.py", "run_unlimited_ocr.py", "translator_offline.py",
        "stress_test_memory.py", "manga_ocr.py", "buscar.py", "gestor.py",
    }
    assert not files & dev_scripts


def test_unclassified_py_files_detects_new_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Un .py nuevo (no dev, no en el baseline) se reporta como sin clasificar.

    Simula la aparición de un módulo nuevo: el walk lo incluye en
    _PROD_PY_FILES pero _KNOWN_PROD_PY_FILES no lo conoce → el CI debe
    avisarlo (y este test lo detecta) en vez de incluirlo silenciosamente.
    """
    prod = list(run_ci._PROD_PY_FILES) + ["nuevo_modulo.py"]
    monkeypatch.setattr(run_ci, "_PROD_PY_FILES", tuple(prod))

    unclassified = run_ci._unclassified_py_files()

    assert "nuevo_modulo.py" in unclassified
    # Los módulos conocidos no se marcan como sin clasificar.
    assert "server.py" not in unclassified


def test_unclassified_py_files_empty_when_baseline_synced() -> None:
    """Con el baseline sincronizado no hay archivos sin clasificar."""
    assert run_ci._unclassified_py_files() == ()


def test_strict_classification_enabled_por_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--strict-classification exige totalidad aunque no haya CI env."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert run_ci._strict_classification_enabled(flag=True) is True


def test_strict_classification_enabled_en_ci_sin_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """En GitHub Actions el gate está activo por defecto (sin flag)."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert run_ci._strict_classification_enabled(flag=False) is True


def test_strict_classification_apagado_local_sin_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local, sin flag: solo avisa (no falla)."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert run_ci._strict_classification_enabled(flag=False) is False


def test_main_acepta_flag_strict_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    """El flag --strict-classification llega a _check_unclassified_files
    (devuelve False con un archivo sin clasificar → exit 1)."""
    import sys

    seen: list[bool] = []
    monkeypatch.setattr(sys, "argv", ["run_ci.py", "--strict-classification"])
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    def _record_ci(ci: bool) -> bool:
        seen.append(ci)
        return False

    monkeypatch.setattr(run_ci, "_check_unclassified_files", _record_ci)
    # Que el resto de pasos no corra: fallamos antes con all_passed=False
    # pero los pasos seguirían — saltar los pasos pesados con skips y
    # cortar tras la clasificación no es posible, así que verificamos que
    # _check_unclassified_files recibe True (strict) con argv correcto.
    # Para evitar correr la suite, forzamos excepción de salida temprana.

    class _Exit(Exception):
        pass

    def _stop(*a: object, **k: object) -> None:
        raise _Exit()

    monkeypatch.setattr(run_ci, "step_syntax", _stop)

    try:
        run_ci.main()
    except _Exit:
        pass
    assert seen == [True]


def test_is_github_actions_true_con_env_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """GitHub Actions define CI=true: el interruptor debe detectarlo."""
    monkeypatch.setenv("CI", "true")
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert run_ci._is_github_actions() is True


def test_is_github_actions_false_en_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin CI ni GITHUB_ACTIONS (local) el interruptor está apagado."""
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    assert run_ci._is_github_actions() is False


def test_unclassified_falla_ci_pero_solo_avisa_local(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Un archivo sin clasificar: en CI (ci=True) devuelve False y registra
    FAIL; en local (ci=False) devuelve True y solo imprime WARN."""
    monkeypatch.setattr(run_ci, "_unclassified_py_files",
                        lambda: ("nuevo_modulo.py",))

    # CI: falla el job
    run_ci.RESULTS.clear()
    ok_ci = run_ci._check_unclassified_files(ci=True)
    assert ok_ci is False
    out_ci = capsys.readouterr().out
    assert "[FAIL] Archivo .py sin clasificar" in out_ci
    assert any(r["name"] == "clasificación de módulos"
               and r["status"] == "FAIL" for r in run_ci.RESULTS)

    # Local: solo avisa, no falla
    run_ci.RESULTS.clear()
    ok_local = run_ci._check_unclassified_files(ci=False)
    assert ok_local is True
    out_local = capsys.readouterr().out
    assert "[WARN] Archivo .py sin clasificar" in out_local
    assert not any(r["name"] == "clasificación de módulos"
                   for r in run_ci.RESULTS)


def test_syntax_bandit_mypy_use_exactly_prod_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Syntax, bandit y mypy deben pasar EXACTAMENTE la misma lista de archivos.

    Captura las llamadas a subprocess.run de cada paso y verifica que las
    listas de archivos Python sean idénticas a _PROD_PY_FILES (mismo orden,
    sin listas hardcodeadas duplicadas en los pasos).
    """
    captured: list[list[str]] = []

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        captured.append(list(args))
        return _fake_completed(list(args))

    monkeypatch.setattr(subprocess, "run", fake_run)

    run_ci.step_syntax()
    run_ci.step_bandit()
    run_ci.step_mypy()

    def rel(path: str) -> str:
        """Path absoluto del subprocess → relativo al proyecto, con '/': """
        p = Path(path).resolve()
        return str(p.relative_to(run_ci.PROJECT_ROOT.resolve())).replace("\\", "/")

    syntax_files: list[str] = []
    bandit_files: list[str] = []
    mypy_files: list[str] = []
    for args in captured:
        if "-m" in args and "py_compile" in args:
            syntax_files.append(rel(args[args.index("py_compile") + 1]))
        elif "-m" in args and "bandit" in args:
            bandit_files = args[args.index("json") + 1 :]
        elif "-m" in args and "mypy" in args:
            mypy_files = args[args.index("mypy") + 1 :]

    expected = list(run_ci._PROD_PY_FILES)
    assert syntax_files == expected, "step_syntax no usa _PROD_PY_FILES"
    assert bandit_files == expected, "step_bandit no usa _PROD_PY_FILES"
    assert mypy_files == expected, "step_mypy no usa _PROD_PY_FILES"


def test_coverage_thresholds_cubren_todos_los_modulos_prod() -> None:
    """Todo módulo de producción tiene umbral de cobertura — si se agrega un
    módulo nuevo a _PROD_PY_FILES y no se le da umbral, este test falla."""
    missing = [f for f in run_ci._PROD_PY_FILES
               if f not in run_ci._COVERAGE_THRESHOLDS]
    assert missing == [], (
        f"Módulos sin umbral en _COVERAGE_THRESHOLDS: {missing}")
    # Sin umbrales huérfanos (módulos que ya no existen en producción)
    extra = [m for m in run_ci._COVERAGE_THRESHOLDS
             if m not in run_ci._PROD_PY_FILES]
    assert extra == [], f"Umbrales de módulos fuera de producción: {extra}"


def _coverage_json(tmp_path: Path, overrides: dict[str, float]) -> Path:
    """Escribe un reporte JSON de coverage sintético con la cobertura de cada
    módulo de producción en su umbral, salvo los que ``overrides`` cambie."""
    files: dict[str, dict[str, object]] = {}
    for mod, umbral in run_ci._COVERAGE_THRESHOLDS.items():
        pct = overrides.get(mod, umbral)
        files[mod] = {"summary": {"percent_covered": pct}}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")
    return path


def test_check_module_coverage_pasa_con_umbrales_cumplidos(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Con todos los módulos en su umbral, la verificación pasa."""
    ok = run_ci._check_module_coverage(_coverage_json(tmp_path, {}))
    assert ok is True
    out = capsys.readouterr().out
    assert "todos cumplen umbral" in out


def test_check_module_coverage_falla_cuando_un_modulo_baja(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Un módulo por debajo de su umbral hace FALLAR la verificación."""
    ok = run_ci._check_module_coverage(
        _coverage_json(tmp_path, {"translator.py": 20.0}))
    assert ok is False
    out = capsys.readouterr().out
    assert "translator.py: 20.0% < umbral 70.0%" in out


def test_check_module_coverage_reporta_sin_umbral(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Si falta el umbral de un módulo, la verificación falla (baseline
    desactualizado) en vez de pasar en silencio."""
    path = _coverage_json(tmp_path, {})
    sin_umbral = dict(run_ci._COVERAGE_THRESHOLDS)
    sin_umbral.pop("cache.py")

    orig = run_ci._COVERAGE_THRESHOLDS
    try:
        run_ci._COVERAGE_THRESHOLDS = sin_umbral
        ok = run_ci._check_module_coverage(path)
    finally:
        run_ci._COVERAGE_THRESHOLDS = orig
    assert ok is False
    out = capsys.readouterr().out
    assert "cache.py: sin umbral" in out


# ─── Gate de cobertura acotado al diff del PR ───────────────────────────


def test_git_base_commit_usa_github_base_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    """En un PR de GitHub Actions, GITHUB_BASE_REF es el primer candidato."""
    monkeypatch.setenv("GITHUB_BASE_REF", "main")
    seen: list[list[str]] = []

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        seen.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, "abc123\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_ci._git_base_commit() == "main"
    assert seen[0] == ["git", "rev-parse", "--verify", "--quiet", "main"]


def test_git_base_commit_local_usa_origin_main(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local (sin env), cae a origin/main si existe."""
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        rc = 0 if "origin/main" in args else 1
        return subprocess.CompletedProcess(
            list(args), rc, "abc123\n" if rc == 0 else "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_ci._git_base_commit() == "origin/main"


def test_git_base_commit_none_sin_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin base resoluble (repo sin remoto/historia) → None (gate completo)."""
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(args), 1, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_ci._git_base_commit() is None


def test_touched_prod_modules_solo_produccion(monkeypatch: pytest.MonkeyPatch) -> None:
    """El diff puede listar README/tests/dev scripts — solo entran los módulos
    de producción, con la ruta relativa del repo."""
    monkeypatch.setenv("GITHUB_BASE_REF", "main")

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in args:
            return subprocess.CompletedProcess(list(args), 0, "abc123\n", "")
        if "diff" in args:
            # <base>...HEAD = commits de la rama; HEAD = working tree + staged
            spec = args[args.index("--name-only") + 1]
            if spec == "main...HEAD":
                out = "translator.py\nREADME.md\ntests/test_translator.py\n"
            else:
                out = "routes/api.py\nbenchmark_ocr.py\n"
            return subprocess.CompletedProcess(list(args), 0, out, "")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_ci._touched_prod_modules() == {"translator.py", "routes/api.py"}


def test_touched_prod_modules_incluye_cambios_sin_commitear(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """El runner local con trabajo en curso (sin commitear) debe acotar el gate
    a los módulos que se están tocando, aunque el diff de commits sea vacío."""
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        if "rev-parse" in args:
            rc = 0 if "origin/main" in args else 1
            return subprocess.CompletedProcess(
                list(args), rc, "abc123\n" if rc == 0 else "", "")
        if "diff" in args:
            spec = args[args.index("--name-only") + 1]
            if spec == "origin/main...HEAD":
                out = ""  # nada commiteado aún
            else:
                out = "cache.py\nREADME.md\n"
            return subprocess.CompletedProcess(list(args), 0, out, "")
        return subprocess.CompletedProcess(list(args), 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_ci._touched_prod_modules() == {"cache.py"}


def test_touched_prod_modules_none_sin_base(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin base resoluble, el gate cae al modo completo (None)."""
    monkeypatch.delenv("GITHUB_BASE_REF", raising=False)

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(args), 1, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert run_ci._touched_prod_modules() is None


def test_check_module_coverage_ignora_modulos_no_tocados(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Un módulo bajo umbral que el diff NO toca es estado global: NO falla el
    gate (solo [WARN]), siempre que los módulos tocados cumplan."""
    path = _coverage_json(tmp_path, {"translator.py": 20.0})

    ok = run_ci._check_module_coverage(path, touched={"cache.py"})

    assert ok is True
    out = capsys.readouterr().out
    assert "todos cumplen umbral" in out
    assert "[WARN] translator.py: 20.0% bajo umbral 70.0%" in out


def test_check_module_coverage_falla_modulo_tocado_bajo_umbral(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Un módulo que el PR SÍ toca y baja de umbral → FAIL del gate."""
    path = _coverage_json(tmp_path, {"translator.py": 20.0})

    ok = run_ci._check_module_coverage(path, touched={"translator.py"})

    assert ok is False
    out = capsys.readouterr().out
    assert "translator.py: 20.0% < umbral 70.0%" in out


def test_check_module_coverage_touched_vacio_pasa(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Si el PR no toca ningún módulo de producción, el gate pasa aunque haya
    deuda global (estado informado como [WARN])."""
    path = _coverage_json(tmp_path, {"translator.py": 20.0})

    ok = run_ci._check_module_coverage(path, touched=set())

    assert ok is True
    out = capsys.readouterr().out
    assert "todos cumplen umbral" in out
    assert "[WARN] translator.py" in out


def test_missing_ranges_compacta_lineas() -> None:
    """Las líneas sin cubrir se compactan en rangos (12-14, 45, 90)."""
    assert run_ci._missing_ranges([]) == ""
    assert run_ci._missing_ranges([7]) == "7"
    assert run_ci._missing_ranges([12, 13, 14, 45, 90]) == "12-14, 45, 90"
    assert run_ci._missing_ranges([1, 3, 5]) == "1, 3, 5"


def test_write_coverage_html_genera_reporte_por_modulo(
        tmp_path: Path) -> None:
    """El reporte HTML lista SOLO los módulos de producción con %, umbral,
    estado y líneas sin cubrir; marca los módulos que toca el diff; en modo
    completo lo dice explícitamente."""
    files: dict[str, dict[str, object]] = {}
    for mod, umbral in run_ci._COVERAGE_THRESHOLDS.items():
        files[mod] = {"summary": {"percent_covered": umbral,
                                   "num_statements": 100}}
    files["translator.py"]["summary"] = {
        "percent_covered": 20.0, "num_statements": 100}
    files["translator.py"]["missing_lines"] = [12, 13, 14, 45, 90]
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")

    out = tmp_path / "cov_html"
    run_ci._write_coverage_html(path, out, touched={"cache.py"})

    html = (out / "index.html").read_text(encoding="utf-8")
    assert "translator.py" in html
    assert "20.0%" in html
    # translator no está tocado por el diff → BAJO (no tocado), no FAIL
    assert ">BAJO (no tocado)<" in html
    assert "▲ diff" in html  # cache.py marcado como tocado por el diff
    assert "12-14, 45, 90" in html
    assert "benchmark_ocr" not in html  # sin scripts de dev
    # Fila resaltada: translator (bajo umbral, no tocado) con tinte ámbar;
    # cache.py cumple su umbral → fila sin resaltar
    assert '<tr class="row-warn"><td>translator.py' in html
    assert "<tr><td>cache.py" in html
    # Ninguna fila resaltada en rojo (FAIL) en este modo — el CSS siempre
    # define la clase, pero no debe aplicarse a ninguna fila
    assert '<tr class="row-fail">' not in html
    # Diseño: fondo blanco explícito (legible en cualquier visor, incluso con
    # tema oscuro), la tabla envuelta en un contenedor con scroll horizontal
    # (no desborda la página), y header sticky con scroll vertical acotado
    assert "background: #ffffff" in html
    assert '<div class="table-wrap"><table>' in html
    assert "overflow-x: auto" in html
    assert "position: sticky" in html
    assert "max-height: 70vh" in html
    # Leyenda de colores: explica rojo/ámbar/verde y el marcador ▲ diff para
    # quien abre el artifact sin contexto, con los tres dots y sus textos
    assert '<div class="legend">' in html
    assert 'class="dot ok"' in html
    assert 'class="dot warn"' in html
    assert 'class="dot fail"' in html
    assert "cumple su umbral" in html
    assert "el PR lo hace bajar" in html
    assert "deuda del repo" in html

    # Modo completo (sin diff): translator bajo umbral → FAIL, y el encabezado
    # lo indica. La fila se resalta en rojo.
    run_ci._write_coverage_html(path, tmp_path / "cov_html_full")
    full = (tmp_path / "cov_html_full" / "index.html").read_text(
        encoding="utf-8")
    assert ">FAIL<" in full
    assert '<tr class="row-fail"><td>translator.py' in full
    assert "modo completo" in full


def test_write_coverage_html_columna_delta_contra_base(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con snapshot de la base, el reporte agrega la columna Δ (verde si subió,
    rojo si bajó, '—' si el módulo no está en el snapshot); sin snapshot la
    columna no aparece."""
    files: dict[str, dict[str, object]] = {}
    for mod, umbral in run_ci._COVERAGE_THRESHOLDS.items():
        files[mod] = {"summary": {"percent_covered": umbral,
                                   "num_statements": 50}}
    files["translator.py"]["summary"] = {"percent_covered": 75.0,
                                           "num_statements": 50}
    files["cache.py"]["summary"] = {"percent_covered": 60.0,
                                       "num_statements": 50}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")

    base = dict(run_ci._COVERAGE_THRESHOLDS)
    base["translator.py"] = 70.0  # base 70.0 → actual 75.0: +5.0
    base["cache.py"] = 68.0       # base 68.0 → actual 60.0: -8.0
    del base["routes/main.py"]    # ausente del snapshot → columna '—'

    out = tmp_path / "cov_html_delta"
    run_ci._write_coverage_html(path, out, touched=set(), base_pct=base)
    html = (out / "index.html").read_text(encoding="utf-8")

    assert ">Δ base<" in html
    # translator subió vs base → verde (+5.0); cache bajó → rojo (-8.0)
    assert "delta-up\">+5.0<" in html
    assert "delta-down\">-8.0<" in html
    # un módulo que no está en el snapshot → '—'
    assert "delta-same\">\u2014<" in html

    # Sin snapshot: la columna no se renderiza
    run_ci._write_coverage_html(path, tmp_path / "cov_html_nobase",
                                touched=set())
    nobase = (tmp_path / "cov_html_nobase" / "index.html").read_text(
        encoding="utf-8")
    assert "Δ base" not in nobase


def test_write_base_coverage_escribe_snapshot_produccion(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El snapshot solo incluye módulos de producción, con el % medido; un
    JSON ausente o inválido no rompe (WARN y nada escrito)."""
    files: dict[str, dict[str, object]] = {}
    for mod, umbral in run_ci._COVERAGE_THRESHOLDS.items():
        files[mod] = {"summary": {"percent_covered": umbral,
                                   "num_statements": 50}}
    files["benchmark_ocr.py"] = {"summary": {"percent_covered": 99.0,
                                               "num_statements": 10}}
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps({"files": files}), encoding="utf-8")

    base_file = tmp_path / "coverage_base.json"
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", base_file)
    run_ci._write_base_coverage(path)

    data = json.loads(base_file.read_text(encoding="utf-8"))
    assert set(data) == set(run_ci._PROD_PY_FILES)
    assert data["translator.py"] == run_ci._COVERAGE_THRESHOLDS["translator.py"]
    assert "benchmark_ocr.py" not in data

    # JSON ausente: WARN y el snapshot previo queda intacto
    run_ci._write_base_coverage(tmp_path / "no_existe.json")
    assert base_file.exists()
    assert set(json.loads(base_file.read_text(encoding="utf-8"))) == set(
        run_ci._PROD_PY_FILES)


def test_load_base_coverage_tolerante(monkeypatch: pytest.MonkeyPatch,
                                      tmp_path: Path) -> None:
    """Lee el snapshot normalizado; tolera archivo ausente, vacío, no-JSON y
    valores basura (los descarta)."""
    base_file = tmp_path / "coverage_base.json"
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", base_file)
    assert run_ci._load_base_coverage() is None  # archivo inexistente

    base_file.write_text("{not json", encoding="utf-8")
    assert run_ci._load_base_coverage() is None

    base_file.write_text(json.dumps({"cache.py": 68.5, "basura": "x"}),
                         encoding="utf-8")
    loaded = run_ci._load_base_coverage()
    assert loaded == {"cache.py": 68.5}

    base_file.write_text(json.dumps({}), encoding="utf-8")
    assert run_ci._load_base_coverage() is None


# ─── Gate de cobertura con el arranque real del servidor (server-test) ────


class _FakeProc:
    """Popen simulado: registra señales/waits/kills sin procesos reales."""

    pid = 1234

    def __init__(self) -> None:
        self.sent: list[int] = []
        self.waited: float | None = None
        self.killed = False

    def send_signal(self, sig: int) -> None:
        self.sent.append(sig)

    def wait(self, timeout: float | None = None) -> None:
        self.waited = timeout

    def kill(self) -> None:
        self.killed = True


def _patch_server_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks de step_server: sin procesos, sin red, endpoints OK."""
    monkeypatch.setattr(run_ci, "_kill_process_on_port", lambda port: False)
    monkeypatch.setattr(run_ci, "server_health",
                        lambda **kw: {"db_available": True, "memory": "1MB"})
    monkeypatch.setattr(run_ci, "http_post",
                        lambda *a, **k: (200, {"translatedText": "Hello",
                                               "engine": "x",
                                               "results": ["a", "b"]}))
    monkeypatch.setattr(run_ci, "http_get", lambda *a, **k: (200, b"body"))


def test_step_server_tracea_con_cobertura_cuando_hay_baseline(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con COVERAGE_JSON (pytest ya midió), step_server lanza el servidor bajo
    `coverage run --append` y re-verifica el gate al terminar."""
    json_path = tmp_path / "coverage.json"
    json_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_ci, "COVERAGE_JSON", json_path)
    _patch_server_helpers(monkeypatch)

    captured_cmd: list[list[str]] = []
    proc = _FakeProc()

    def fake_popen(cmd: list[str], **kw: object) -> _FakeProc:
        captured_cmd.append(cmd)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    refreshed: list[bool] = []

    def _refresh() -> bool:
        refreshed.append(True)
        return True

    monkeypatch.setattr(run_ci, "_refresh_coverage_after_server", _refresh)

    ok = run_ci.step_server(with_coverage=True)

    assert ok is True
    assert captured_cmd == [[run_ci.PYTHON, "-u", "-m", "coverage", "run",
                             "--append", "--rcfile", str(run_ci.COVERAGE_RC),
                             "server.py"]]
    assert refreshed == [True]


def test_step_server_sin_baseline_corre_plano(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin baseline de pytest (COVERAGE_JSON inexistente), el servidor corre
    normal y NO se re-verifica cobertura (no hay data que combinar)."""
    monkeypatch.setattr(run_ci, "COVERAGE_JSON", tmp_path / "no_existe.json")
    _patch_server_helpers(monkeypatch)

    captured_cmd: list[list[str]] = []
    proc = _FakeProc()

    def fake_popen(cmd: list[str], **kw: object) -> _FakeProc:
        captured_cmd.append(cmd)
        return proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    refreshed: list[bool] = []

    def _refresh() -> bool:
        refreshed.append(True)
        return True

    monkeypatch.setattr(run_ci, "_refresh_coverage_after_server", _refresh)

    ok = run_ci.step_server(with_coverage=True)

    assert ok is True
    assert captured_cmd == [[run_ci.PYTHON, "-u", "server.py"]]
    assert refreshed == []


def test_stop_server_gracefully_posix_envia_sigint(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """En POSIX el servidor se detiene con SIGINT (atexit → coverage
    flushea) y _server_proc queda en None."""
    monkeypatch.setattr(os, "name", "posix")
    proc = _FakeProc()
    monkeypatch.setattr(run_ci, "_server_proc", proc)

    ok = run_ci._stop_server_gracefully()

    assert ok is True
    assert proc.sent == [signal.SIGINT]
    assert proc.waited == 10.0
    assert proc.killed is False
    assert run_ci._server_proc is None


def test_stop_server_gracefully_windows_cae_a_kill_duro(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """En Windows no hay SIGINT portable: se degrada al kill duro (pierde la
    data del servidor, conserva la de pytest) y devuelve False."""
    monkeypatch.setattr(os, "name", "nt")
    proc = _FakeProc()
    monkeypatch.setattr(run_ci, "_server_proc", proc)
    monkeypatch.setattr(run_ci, "_kill_process_on_port", lambda port: False)

    ok = run_ci._stop_server_gracefully()

    assert ok is False
    assert proc.sent == []
    assert proc.killed is True
    assert run_ci._server_proc is None


def test_refresh_coverage_after_server_reverifica_gate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tras detener el servidor, se regenera el JSON combinado y se re-corre
    el gate con nombre distinto; el HTML también se actualiza."""
    monkeypatch.setattr(run_ci, "_stop_server_gracefully", lambda: True)
    json_path = tmp_path / "coverage.json"
    monkeypatch.setattr(run_ci, "COVERAGE_JSON", json_path)

    seen_run: list[list[str]] = []

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        seen_run.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(run_ci, "_touched_prod_modules", lambda: None)
    gate_calls: list[tuple[object, object, str]] = []

    def _fake_gate(p: object, t: object, result_name: str = "x") -> bool:
        gate_calls.append((p, t, result_name))
        return True

    monkeypatch.setattr(run_ci, "_check_module_coverage", _fake_gate)
    html_calls: list[bool] = []

    def _fake_html(p: object, o: object, t: object,
                   b: object | None = None) -> None:
        html_calls.append(True)

    monkeypatch.setattr(run_ci, "_write_coverage_html", _fake_html)

    ok = run_ci._refresh_coverage_after_server()

    assert ok is True
    assert seen_run == [[run_ci.PYTHON, "-m", "coverage", "json",
                         "--rcfile", str(run_ci.COVERAGE_RC), "-o",
                         str(json_path)]]
    assert gate_calls == [
        (json_path, None, "cobertura por módulo (con servidor)")]
    assert html_calls == [True]


def test_refresh_coverage_propaga_fail_del_gate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si con el arranque real del servidor un módulo baja de umbral, el
    paso devuelve False (el job server-test falla)."""
    monkeypatch.setattr(run_ci, "_stop_server_gracefully", lambda: True)
    json_path = tmp_path / "coverage.json"
    monkeypatch.setattr(run_ci, "COVERAGE_JSON", json_path)

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(args), 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(run_ci, "_touched_prod_modules", lambda: None)
    monkeypatch.setattr(run_ci, "_check_module_coverage",
                        lambda p, t, result_name="x": False)
    monkeypatch.setattr(run_ci, "_write_coverage_html",
                        lambda p, o, t, b=None: None)

    ok = run_ci._refresh_coverage_after_server()

    assert ok is False


def test_refresh_coverage_no_bloquea_si_coverage_json_falla(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Un fallo de tooling (coverage json con rc != 0) no bloquea el paso:
    se avisa y se conserva la verificación de pytest."""
    monkeypatch.setattr(run_ci, "_stop_server_gracefully", lambda: True)
    monkeypatch.setattr(run_ci, "COVERAGE_JSON", tmp_path / "coverage.json")

    def fake_run(args: list[str], *a: object, **kw: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(args), 1, "", "boom")

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok = run_ci._refresh_coverage_after_server()

    assert ok is True
    assert "coverage json falló" in capsys.readouterr().out


# ─── Resumen de cobertura en GITHUB_STEP_SUMMARY (lint-test) ──────────────


def test_coverage_summary_markdown_tabla_completa(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con todos los módulos en su umbral, la tabla lista los 20 módulos de
    producción con estado OK y sin módulos bajo umbral."""
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", tmp_path / "no_base.json")
    md = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {}), touched=None)

    # El callout al tope confirma el estado global antes de la tabla
    assert md.startswith("> ✅ **Todos los módulos cumplen su umbral**")
    assert "### Cobertura por módulo" in md
    assert "| Módulo | Cubierto | Umbral | Estado |" in md
    for mod in run_ci._PROD_PY_FILES:
        assert f"| {mod} " in md
    assert "BAJO UMBRAL" not in md
    assert "📈" not in md  # sin snapshot de base, no hay línea de mejoras


def test_coverage_summary_marca_modulos_bajo_umbral(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Un módulo tocado por el diff que baja de umbral se marca como BAJO
    UMBRAL y aparece en la lista final."""
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", tmp_path / "no_base.json")
    md = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {"translator.py": 20.0}),
        touched={"translator.py"})

    assert "❌ **BAJO UMBRAL**" in md
    # Callout al tope con el módulo y su delta, y negrita en la columna módulo
    assert "> ⚠️ **1 módulo(s) bajo umbral:**" in md
    assert "`translator.py` (20.0% < 70.0%)" in md
    assert "| **translator.py** " in md


def test_coverage_summary_mejoras_contra_base(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con snapshot de base, el callout lista los módulos que más mejoraron
    (top 3, sin incluir los que están bajo umbral); sin base, no hay línea."""
    base_file = tmp_path / "coverage_base.json"
    base_file.write_text(json.dumps({"translator.py": 70.0,
                                     "routes/main.py": 95.0,
                                     "cache.py": 40.0}),
                         encoding="utf-8")
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", base_file)

    # translator +10.0 (mejora), routes/main +5.0 (mejora), cache -10.0 pero
    # está bajo umbral (no entra a mejoras)
    md = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {"translator.py": 80.0,
                                  "routes/main.py": 100.0,
                                  "cache.py": 30.0}),
        touched={"translator.py", "routes/main.py", "cache.py"})

    assert "📈 **Mejoran:**" in md
    assert "`translator.py` (+10.0)" in md
    assert "`routes/main.py` (+5.0)" in md
    # cache.py está en la línea de ⚠️ (bajo umbral), no en la de mejoras
    assert "`cache.py` (+10.0)" not in md
    assert "`cache.py` (30.0% < 68.0%)" in md
    # El orden respeta el mayor delta primero
    assert md.index("`translator.py` (+10.0)") < md.index(
        "`routes/main.py` (+5.0)")

    # Todo OK con mejoras: la línea ✅ y la de 📈 conviven (cache bajo umbral
    # pero no tocado → no entra a bajos; su delta vs base es negativo)
    md_ok = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {"translator.py": 80.0,
                                  "cache.py": 30.0}),
        touched={"translator.py"})
    assert md_ok.startswith("> ✅ **Todos los módulos cumplen su umbral**")
    assert "📈 **Mejoran:** `translator.py` (+10.0)" in md_ok


def test_coverage_summary_no_tocado_bajo_umbral_no_falla(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Un módulo bajo umbral que el diff NO toca se muestra como deuda del
    repo ("bajo (no tocado)") y no entra a la lista de módulos bajo umbral."""
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", tmp_path / "no_base.json")
    md = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {"translator.py": 20.0}),
        touched={"cache.py"})

    assert "⬇️ bajo (no tocado)" in md
    assert "BAJO UMBRAL" not in md
    assert "> ✅ **Todos los módulos cumplen su umbral**" in md


def test_coverage_summary_json_ausente_no_explota(tmp_path: Path) -> None:
    """Sin reporte JSON (pytest no corrió), el resumen avisa en vez de
    fallar el paso."""
    md = run_ci._coverage_summary_markdown(
        tmp_path / "no_existe.json", touched=None)

    assert "No se pudo leer el reporte de cobertura" in md


def test_artifact_url_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    """El enlace al artifact del run se arma con GITHUB_REPOSITORY y
    GITHUB_RUN_ID (y server personalizable); sin ellos, None."""
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    assert run_ci._artifact_url() is None

    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    assert run_ci._artifact_url() is None  # falta GITHUB_RUN_ID

    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    assert run_ci._artifact_url() == (
        "https://github.com/owner/repo/actions/runs/12345#artifacts")

    monkeypatch.setenv("GITHUB_SERVER_URL", "https://ghes.example.com")
    assert run_ci._artifact_url() == (
        "https://ghes.example.com/owner/repo/actions/runs/12345#artifacts")


def test_coverage_summary_markdown_enlace_artifact(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con artifact_url, el callout agrega el enlace al reporte completo en
    ambos escenarios (bajo umbral y todo OK); sin URL, no hay enlace."""
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", tmp_path / "no_base.json")
    url = "https://github.com/o/r/actions/runs/1#artifacts"
    md = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {"translator.py": 20.0}),
        touched={"translator.py"}, artifact_url=url)
    assert f"[ver reporte completo]({url})" in md
    assert md.startswith("> ⚠️")

    md_ok = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {}), touched=None, artifact_url=url)
    assert f"[ver reporte completo]({url})" in md_ok
    assert md_ok.startswith("> ✅")

    md_sin = run_ci._coverage_summary_markdown(
        _coverage_json(tmp_path, {"translator.py": 20.0}),
        touched={"translator.py"})
    assert "ver reporte completo" not in md_sin


def test_write_coverage_summary_escribe_al_env(tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    """Con GITHUB_STEP_SUMMARY definido, la tabla se appendea al archivo del
    step summary del run."""
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(run_ci, "COVERAGE_JSON", _coverage_json(tmp_path, {}))
    monkeypatch.setattr(run_ci, "COVERAGE_BASE", tmp_path / "no_base.json")
    monkeypatch.setattr(run_ci, "_touched_prod_modules", lambda: None)

    run_ci.write_coverage_summary()

    content = summary.read_text(encoding="utf-8")
    assert "> ✅ **Todos los módulos cumplen su umbral**" in content
    assert "### Cobertura por módulo" in content


def test_write_coverage_summary_sin_env_solo_avisa(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Sin GITHUB_STEP_SUMMARY (local), el paso avisa y no escribe nada."""
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    run_ci.write_coverage_summary()

    assert "GITHUB_STEP_SUMMARY no definido" in capsys.readouterr().out


# ─── Comentario del bot en el PR (GITHUB_TOKEN) ───────────────────────────


class _FakeHttpResp:
    """Respuesta HTTP simulada para urllib."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_FakeHttpResp":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_github_api_autentica_y_parsea(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con GITHUB_TOKEN, la llamada va a la API con Bearer y headers de
    versión, y devuelve el JSON parseado."""
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    seen: list[tuple[object, float]] = []

    def fake_urlopen(req: object, timeout: float) -> _FakeHttpResp:
        seen.append((req, timeout))
        return _FakeHttpResp(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = run_ci._github_api("/repos/x/y/issues/1/comments")

    assert result == {"ok": True}
    assert seen[0][1] == 15
    req = seen[0][0]
    assert isinstance(req, urllib.request.Request)
    assert req.full_url == "https://api.github.com/repos/x/y/issues/1/comments"
    assert req.get_header("Authorization") == "Bearer tok"
    # urllib normaliza los nombres con capitalize() (solo primera mayúscula).
    assert req.get_header("X-github-api-version") == "2022-11-28"
    assert req.get_method() == "GET"


def test_github_api_error_http_devuelve_none(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Un 4xx/5xx de la API no explota: devuelve None y avisa."""
    monkeypatch.setenv("GITHUB_TOKEN", "tok")

    def fake_urlopen(req: object, timeout: float) -> object:
        raise urllib.error.HTTPError(
            "https://api.github.com/x", 403, "Forbidden", Message(), None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    assert run_ci._github_api("/x") is None


def test_github_api_sin_token_devuelve_none(
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Sin GITHUB_TOKEN no se intenta siquiera la llamada."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fail_urlopen(*a: object, **k: object) -> object:
        raise AssertionError("no debe llamarse urlopen sin token")

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    assert run_ci._github_api("/x") is None


def _fake_pr_env(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        coverage: dict[str, float] | None = None,
        touched: set[str] | None = None,
        base: dict[str, float] | None = None) -> None:
    """Entorno de un PR de GitHub Actions para write_coverage_comment.

    ``coverage``: % por módulo (default: todos en su umbral, vía
    ``_coverage_json``). ``touched``: módulos que toca el diff (default
    ``None``: el diff no se computó y cualquier módulo bajo umbral cuenta
    como BAJO UMBRAL). ``base``: snapshot de la base para la línea de
    mejoras (default: sin snapshot, ``COVERAGE_BASE`` apunta a un archivo
    inexistente).
    """
    monkeypatch.setenv("GITHUB_TOKEN", "tok")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"pull_request": {"number": 7}}),
                     encoding="utf-8")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setattr(run_ci, "COVERAGE_JSON",
                        _coverage_json(tmp_path, coverage or {}))
    if base is None:
        monkeypatch.setattr(run_ci, "COVERAGE_BASE",
                            tmp_path / "no_base.json")
    else:
        base_file = tmp_path / "coverage_base.json"
        base_file.write_text(json.dumps(base), encoding="utf-8")
        monkeypatch.setattr(run_ci, "COVERAGE_BASE", base_file)
    monkeypatch.setattr(run_ci, "_touched_prod_modules", lambda: touched)


def test_write_coverage_comment_crea_comentario(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin comentario previo del bot, se crea uno nuevo con la tabla y el
    marcador."""
    _fake_pr_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str, dict[str, str] | None]] = []

    def fake_api(endpoint: str, method: str = "GET",
                 payload: dict[str, str] | None = None) -> Any:
        calls.append((endpoint, method, payload))
        return [] if method == "GET" else {"id": 99}

    monkeypatch.setattr(run_ci, "_github_api", fake_api)

    run_ci.write_coverage_comment()

    assert calls[0] == ("/repos/owner/repo/issues/7/comments", "GET", None)
    assert calls[1][0] == "/repos/owner/repo/issues/7/comments"
    assert calls[1][1] == "POST"
    payload = calls[1][2]
    assert payload is not None
    assert "### Cobertura por módulo" in payload["body"]
    assert run_ci._COVERAGE_COMMENT_MARKER in payload["body"]


def test_write_coverage_comment_actualiza_existente(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Si el bot ya comentó (marcador presente), se actualiza ese comentario
    con PATCH en vez de crear otro (sin spam por commit)."""
    _fake_pr_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str, dict[str, str] | None]] = []

    def fake_api(endpoint: str, method: str = "GET",
                 payload: dict[str, str] | None = None) -> Any:
        calls.append((endpoint, method, payload))
        if method == "GET":
            return [{"id": 42, "body": "viejo "
                     + run_ci._COVERAGE_COMMENT_MARKER}]
        return {"id": 42}

    monkeypatch.setattr(run_ci, "_github_api", fake_api)

    run_ci.write_coverage_comment()

    assert calls[1] == ("/repos/owner/repo/issues/comments/42", "PATCH",
                        {"body": calls[1][2]["body"]} if calls[1][2] else None)
    assert calls[1][1] == "PATCH"
    payload = calls[1][2]
    assert payload is not None
    assert run_ci._COVERAGE_COMMENT_MARKER in payload["body"]


def test_write_coverage_comment_incluye_callout_bajo_umbral(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con un módulo tocado bajo umbral y otro que mejora contra la base, el
    body del comentario incluye el callout ⚠️ (bajos) y el 📈 (mejoras), tanto
    en el POST (crear) como en el PATCH (actualizar comentario previo)."""
    # Base con launcher en 90.0: con el umbral 95.0 como pct, su delta es
    # +5.0 → entra a la línea de mejoras (translator está bajo umbral y
    # queda excluido de mejoras)
    _fake_pr_env(tmp_path, monkeypatch,
                 coverage={"translator.py": 20.0},
                 touched={"translator.py"},
                 base={"launcher.py": 90.0})

    calls: list[tuple[str, str, dict[str, str] | None]] = []
    state: dict[str, list[dict[str, object]]] = {"existing": []}

    def fake_api(endpoint: str, method: str = "GET",
                 payload: dict[str, str] | None = None) -> Any:
        calls.append((endpoint, method, payload))
        if method == "GET":
            return state["existing"]
        return {"id": 99}

    monkeypatch.setattr(run_ci, "_github_api", fake_api)

    # POST: sin comentario previo → se crea con ambos callouts
    run_ci.write_coverage_comment()
    assert calls[1][1] == "POST"
    created = calls[1][2]
    assert created is not None
    assert "> ⚠️ **1 módulo(s) bajo umbral:**" in created["body"]
    assert "`translator.py` (20.0% < 70.0%)" in created["body"]
    assert "❌ **BAJO UMBRAL**" in created["body"]
    assert "📈 **Mejoran:** `launcher.py` (+5.0)" in created["body"]

    # PATCH: comentario previo con el marcador → se actualiza con ambos
    state["existing"] = [{"id": 42, "body": "viejo "
                           + run_ci._COVERAGE_COMMENT_MARKER}]
    calls.clear()
    run_ci.write_coverage_comment()
    assert calls[1][1] == "PATCH"
    updated = calls[1][2]
    assert updated is not None
    assert "> ⚠️ **1 módulo(s) bajo umbral:**" in updated["body"]
    assert "`translator.py` (20.0% < 70.0%)" in updated["body"]
    assert "📈 **Mejoran:** `launcher.py` (+5.0)" in updated["body"]


def test_write_coverage_comment_incluye_enlace_artifact(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Con GITHUB_RUN_ID definido, el body del comentario incluye el enlace
    al artifact del run (POST y PATCH); sin run_id, el callout no lo
    agrega."""
    _fake_pr_env(tmp_path, monkeypatch)
    calls: list[tuple[str, str, dict[str, str] | None]] = []

    def fake_api(endpoint: str, method: str = "GET",
                 payload: dict[str, str] | None = None) -> Any:
        calls.append((endpoint, method, payload))
        return {"id": 99}

    monkeypatch.setattr(run_ci, "_github_api", fake_api)
    expected_url = ("https://github.com/owner/repo/actions/runs/"
                    "12345#artifacts")

    # Sin GITHUB_RUN_ID: el body NO incluye el enlace
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    run_ci.write_coverage_comment()
    assert calls[1][1] == "POST"
    created = calls[1][2]
    assert created is not None
    assert expected_url not in created["body"]
    assert "[ver reporte completo]" not in created["body"]

    # Con GITHUB_RUN_ID: el enlace aparece en el callout (POST)
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    calls.clear()
    run_ci.write_coverage_comment()
    assert calls[1][1] == "POST"
    created = calls[1][2]
    assert created is not None
    assert f"[ver reporte completo]({expected_url})" in created["body"]

    # Y en el PATCH al actualizar el comentario previo
    state: dict[str, list[dict[str, object]]] = {"existing": []}

    def fake_api2(endpoint: str, method: str = "GET",
                  payload: dict[str, str] | None = None) -> Any:
        calls.append((endpoint, method, payload))
        if method == "GET":
            return state["existing"]
        return {"id": 42}

    monkeypatch.setattr(run_ci, "_github_api", fake_api2)
    state["existing"] = [{"id": 42, "body": "viejo "
                           + run_ci._COVERAGE_COMMENT_MARKER}]
    calls.clear()
    run_ci.write_coverage_comment()
    assert calls[1][1] == "PATCH"
    updated = calls[1][2]
    assert updated is not None
    assert f"[ver reporte completo]({expected_url})" in updated["body"]


def test_write_coverage_comment_sin_token_solo_avisa(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """Sin GITHUB_TOKEN (local / fork sin permisos), avisa y no toca la API."""
    _fake_pr_env(tmp_path, monkeypatch)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def fail_api(*a: object, **k: object) -> object:
        raise AssertionError("no debe llamarse a la API sin token")

    monkeypatch.setattr(run_ci, "_github_api", fail_api)

    run_ci.write_coverage_comment()

    assert "GITHUB_TOKEN no definido" in capsys.readouterr().out


def test_write_coverage_comment_solo_en_prs(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str]) -> None:
    """En push a main (sin PR) no se comenta nada."""
    _fake_pr_env(tmp_path, monkeypatch)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "push")

    def fail_api(*a: object, **k: object) -> object:
        raise AssertionError("no debe llamarse a la API fuera de un PR")

    monkeypatch.setattr(run_ci, "_github_api", fail_api)

    run_ci.write_coverage_comment()

    assert "solo se comenta en PRs" in capsys.readouterr().out
