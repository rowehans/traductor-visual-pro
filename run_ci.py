#!/usr/bin/env python
"""
run_ci.py - CI local para Traductor Visual Pro.

Ejecuta bateria completa de tests:
  1. Syntax check (Python + JS)
  2. mypy strict (type check)
  3. bandit (security audit)
  4. pytest + cobertura (suite unitaria: traducción, OCR, API y procesamiento)
  5. Servidor Flask + health + endpoints API
  6. analisis_calidad.py (calidad de traducciones)
  7. stress_test_memory.py (opcional, ~10 min con --full)

No depende de PowerShell. Usa env/ o Python del sistema.

Uso:
    python run_ci.py                    # Tests rapidos (~2 min)
    python run_ci.py --full             # Incluye stress test (~12 min)
    python run_ci.py --server           # Solo inicia servidor + health check
    python run_ci.py --skip-syntax      # Omitir syntax check
    python run_ci.py --skip-mypy        # Omitir type check mypy
    python run_ci.py --skip-bandit      # Omitir audit de seguridad
    python run_ci.py --skip-pytest      # Omitir tests unitarios
    python run_ci.py --skip-cov         # Omitir cobertura (mas rapido)
    python run_ci.py --strict-classification   # Sin clasificar = error (exit 1)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any, cast

# Windows console UTF-8 fix
if sys.version_info >= (3, 7):
    try:
        sys.stdout.reconfigure(  # type: ignore[union-attr]
            encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent

# Detectar entorno: si env/ existe (local), usar ese Python; si no (CI/GitHub Actions), usar sys.executable
# USAR SIEMPRE ruta ABSOLUTA para evitar problemas con subprocess en bash/shells mixtos
_env_python = PROJECT_ROOT / "env" / "Scripts" / "python.exe"
if _env_python.is_file():
    PYTHON = str(_env_python.resolve())
else:
    _env_python = PROJECT_ROOT / "env" / "bin" / "python"  # Linux/macOS venv
    if _env_python.is_file():
        PYTHON = str(_env_python.resolve())
    else:
        PYTHON = sys.executable  # Fallback: usar el Python del sistema

SERVER_PORT = 5174
SERVER_HOST = "127.0.0.1"
RESULTS: list[dict[str, str]] = []
_STRESS_RESULT: dict[str, Any] | None = None
PYTEST_BASETEMP = PROJECT_ROOT / ".tmp_pytest_ci"
PYTEST_BASETEMP_ALT = PROJECT_ROOT / ".tmp_pytest_ci_alt"

# Prefijos de scripts de desarrollo/herramientas ad-hoc que NO son
# producción: benchmarks, tests sueltos, generadores de reportes, runners
# manuales y utilidades de build/reproceso.
_DEV_SCRIPT_PREFIXES = (
    "benchmark_", "test_", "check_", "generate_", "analizar_",
    "extraer_", "rebuild_", "reprocess_", "run_", "codegraph", "diag_",
)

# Scripts de dev con nombres que no siguen un prefijo reconocible.
_DEV_SCRIPT_NAMES = frozenset({
    "build.py", "buscar.py", "diag_pdf.py", "gestor.py", "manga_ocr.py",
    "run_ci.py", "stress_test_memory.py", "translator_offline.py",
})


def _discover_prod_py_files() -> tuple[str, ...]:
    """Descubre los módulos Python de producción recorriendo el repo.

    Regla: todo ``*.py`` de la raíz del proyecto y de ``routes/`` (el único
    paquete de la app), EXCEPTO los scripts de desarrollo/herramientas
    (benchmarks, tests sueltos, generadores, diagnóstico, runners). Así un
    módulo nuevo se audita automáticamente en syntax check, bandit y mypy
    sin tener que editar esta función.
    """
    found: list[str] = []
    for base in (PROJECT_ROOT, PROJECT_ROOT / "routes"):
        if not base.is_dir():
            continue
        for path in sorted(base.iterdir()):
            if not path.is_file() or path.suffix != ".py":
                continue
            name = path.name
            if name in _DEV_SCRIPT_NAMES or name.startswith(_DEV_SCRIPT_PREFIXES):
                continue
            found.append(path.relative_to(PROJECT_ROOT).as_posix())
    return tuple(found)


# Archivos Python de producción auditados por syntax check, bandit y mypy.
# Se generan recorriendo el repo (raíz + routes/) filtrando scripts de dev;
# los tres pasos comparten esta única fuente de verdad.
_PROD_PY_FILES: tuple[str, ...] = _discover_prod_py_files()

# Baseline explícito de los módulos de producción intencionales. El walk
# auto-descubre los ``.py`` de raíz + routes/, pero cualquier archivo que
# incluya y NO esté en este set es "sin clasificar": se audita igualmente
# (default seguro) y se reporta como [WARN] para que nadie agregue
# producción al CI por accidente. Clasificar = decidir deliberadamente.
_KNOWN_PROD_PY_FILES: frozenset[str] = frozenset({
    "analisis_calidad.py", "cache.py", "config.py", "launcher.py",
    "main.py", "models.py", "ocr_engine.py", "ocr_utils.py",
    "process_all_pages.py", "quality_analysis.py", "ratelimit.py",
    "runtime_diagnostics.py", "server.py", "translation_memory.py",
    "translator.py", "uocr_client.py", "uocr_daemon.py",
    "routes/__init__.py", "routes/api.py", "routes/main.py",
})


def _unclassified_py_files() -> tuple[str, ...]:
    """Archivos ``.py`` que el walk incluye pero nadie clasificó.

    Ni dev (prefijos/lista de exclusión) ni producción conocida
    (``_KNOWN_PROD_PY_FILES``): un módulo nuevo que el walk auto-incluye.
    Se audita igualmente en los tres pasos, pero se reporta para que la
    clasificación sea una decisión explícita.
    """
    return tuple(sorted(set(_PROD_PY_FILES) - _KNOWN_PROD_PY_FILES))


def _is_github_actions() -> bool:
    """True si el CI corre en GitHub Actions (env ``CI=true``).

    GitHub Actions define ``CI=true`` (y ``GITHUB_ACTIONS=true``) en todos
    los jobs; en local ninguna de las dos existe. Es el interruptor que
    convierte el aviso de archivos sin clasificar en un FAIL.
    """
    return (os.environ.get("CI") == "true"
            or os.environ.get("GITHUB_ACTIONS") == "true")


def _strict_classification_enabled(flag: bool) -> bool:
    """True si los archivos sin clasificar deben FALLAR el CI.

    El flag ``--strict-classification`` exige totalidad en cualquier entorno;
    en GitHub Actions (env ``CI=true``) el gate está activo por defecto.
    """
    return flag or _is_github_actions()


def _check_unclassified_files(ci: bool) -> bool:
    """Reporta los archivos ``.py`` sin clasificar; False = el CI debe fallar.

    En GitHub Actions (``ci=True``) cada archivo sin clasificar se imprime
    como ``[FAIL]`` y se registra un resultado FAIL — un PR que introduzca
    un módulo sin clasificar rompe el job. En local (``ci=False``) solo se
    avisa con ``[WARN]``, sin afectar el resultado.
    """
    unclassified = _unclassified_py_files()
    for name in unclassified:
        tag = "[FAIL]" if ci else "[WARN]"
        print(f"  {tag} Archivo .py sin clasificar detectado: {name}")
        print("         Se auditará como producción. Si es un script de dev,")
        print("         renómbralo (prefijo dev) o agrégalo a _DEV_SCRIPT_NAMES;")
        print("         si es producción, agrégalo a _KNOWN_PROD_PY_FILES.")
    if ci and unclassified:
        result("clasificación de módulos", "FAIL",
               f"{len(unclassified)} archivo(s) sin clasificar")
        return False
    return True


# --- Helpers -----------------------------------------------------------

def _basetemp_usable(path: Path) -> bool:
    """True si pytest puede usar ``path`` como basetemp.

    Prueba real (crear + escribir + borrar) en lugar de ``os.access``, que
    con ACLs restrictivas de Windows puede reportar permisos que luego
    fallan en runtime. pytest no puede limpiar un basetemp que no puede
    borrar (PermissionError WinError 5), así que la detección temprana
    evita que el CI muera en la fase de setup.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".ci_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _resolve_basetemp() -> Path:
    """Devuelve el basetemp de pytest a usar.

    Si ``.tmp_pytest_ci`` quedó con permisos inaccesibles (p. ej. creado por
    un proceso elevado/sandbox con ACL que excluye al usuario actual), cae a
    un alternativo dentro del proyecto. Último recurso: directorio temporal
    del sistema. El resultado es estable dentro de una corrida para que el
    paso pytest y los tests usen el mismo directorio.
    """
    if _basetemp_usable(PYTEST_BASETEMP):
        return PYTEST_BASETEMP
    print(f"  [WARN] {PYTEST_BASETEMP.name} inaccesible; usando basetemp alternativo")
    if _basetemp_usable(PYTEST_BASETEMP_ALT):
        return PYTEST_BASETEMP_ALT
    print(f"  [WARN] {PYTEST_BASETEMP_ALT.name} también inaccesible; usando temp del sistema")
    import tempfile
    return Path(tempfile.mkdtemp(prefix="tvp_pytest_"))

def _check_node_available() -> bool:
    """Verifica si node.js esta disponible en PATH."""
    try:
        r = subprocess.run(
            ["node", "--version"],
            capture_output=True, text=True, timeout=5
        )
        return r.returncode == 0 and r.stdout.startswith("v")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def step(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def result(name: str, status: str, detail: str = "") -> None:
    RESULTS.append({"name": name, "status": status, "detail": detail})
    icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "WARN": "[WARN]"}.get(status, "[?]")
    print(f"  {icon} {name} - {detail}")


def run_python(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str] | None:
    """Ejecuta un script Python con env/ y retorna el resultado."""
    try:
        return subprocess.run(
            [PYTHON, *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout, cwd=str(PROJECT_ROOT)
        )
    except subprocess.TimeoutExpired:
        print(f"  [FAIL] Timeout ({timeout}s) ejecutando {' '.join(args)}")
        return None


def server_health(timeout: int = 15) -> dict[str, Any] | None:
    """Espera hasta que el servidor responda en /api/health.

    Reintenta mientras el servidor arranca: en el primer arranque la carga
    de modelos (EasyOCR/CT2/YOLO) puede retrasar la respuesta mas alla del
    timeout por request, asi que un TimeoutError aqui no debe crashear el CI.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(
                f"http://{SERVER_HOST}:{SERVER_PORT}/api/health", timeout=2
            )
            if r.status == 200:
                return cast(dict[str, Any], json.loads(r.read().decode("utf-8")))
        except OSError:
            # Cubre URLError, ConnectionRefusedError y TimeoutError
            # (socket.timeout). El servidor aun no responde: se reintenta.
            pass
        time.sleep(0.5)
    return None


def http_get(path: str, timeout: int = 10) -> tuple[int, bytes]:
    """GET request al servidor."""
    r = urllib.request.urlopen(
        f"http://{SERVER_HOST}:{SERVER_PORT}{path}", timeout=timeout
    )
    return r.status, r.read()


def http_post(path: str, data: dict[str, Any], timeout: int = 30) -> tuple[int, dict[str, Any]]:
    """POST request al servidor."""
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"http://{SERVER_HOST}:{SERVER_PORT}{path}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    r = urllib.request.urlopen(req, timeout=timeout)
    return r.status, json.loads(r.read().decode("utf-8"))


# --- Steps -------------------------------------------------------------

def step_syntax() -> bool:
    """Paso 1: Syntax check en todos los archivos Python."""
    step("PASO 1/7 - Syntax check")
    all_ok = True
    files = list(_PROD_PY_FILES)
    for f in files:
        path = os.path.join(PROJECT_ROOT, f)
        if not os.path.exists(path):
            print(f"  [?] {f} no encontrado, saltando")
            continue
        r = subprocess.run(
            [PYTHON, "-m", "py_compile", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        if r.returncode == 0:
            print(f"  [OK] {f}")
        else:
            print(f"  [FAIL] {f} - ERROR DE SINTAXIS")
            print(f"     {r.stderr[:200]}")
            all_ok = False

    # ─── Verificar JavaScript ────────────────────────────
    js_files = _js_syntax_files()
    js_ok = True
    if not _check_node_available():
        print("  [SKIP] node no encontrado en PATH — saltando verificación JS")
    else:
        for f in js_files:
            path = os.path.join(PROJECT_ROOT, f)
            if not os.path.exists(path):
                print(f"  [?] {f} no encontrado, saltando")
                continue
            r = subprocess.run(
                ["node", "--check", path],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
            )
            if r.returncode == 0:
                print(f"  [OK] {f}")
            else:
                print(f"  [FAIL] {f} - ERROR DE SINTAXIS")
                print(f"     {r.stderr[:200]}")
                js_ok = False

    all_ok = all_ok and js_ok
    status = "PASS" if all_ok else "FAIL"
    detail_parts = []
    if all_ok:
        detail_parts.append("Python OK")
        if _check_node_available():
            detail_parts.append("JS OK")
        else:
            detail_parts.append("JS saltado (node no disponible)")
    else:
        detail_parts.append("Hay errores")
    result("Syntax check", status, " — ".join(detail_parts))
    return all_ok


def _pytest_test_files() -> list[str]:
    """Devuelve la suite CI sin duplicados y en un orden estable."""
    files = [
        "tests/test_translator.py",
        "tests/test_ocr_utils.py",
        "tests/test_ocr_functions.py",
        "tests/test_api.py",
        "tests/test_ocr_engine.py",
        "tests/test_uocr_daemon.py",
        "tests/test_uocr_client.py",
        "tests/test_ratelimit.py",
        "tests/test_runtime_diagnostics.py",
        "tests/test_translation_memory.py",
        "tests/test_models.py",
        "tests/test_packaging.py",
        "tests/test_path_security.py",
        "tests/test_cache.py",
        "tests/test_process_all_pages.py",
        "tests/test_correccion_detector.py",
        "tests/test_corrector_oro.py",
        "tests/test_manga_ocr.py",
        "tests/test_run_ci.py",
        "tests/test_quality_analysis.py",
        "tests/test_server.py",
        "tests/test_main_routes.py",
        "tests/test_launcher.py",
        "tests/test_main.py",
    ]
    return list(dict.fromkeys(files))


def _js_syntax_files() -> list[str]:
    """Archivos JS que deben pasar ``node --check`` en cada CI."""
    return [
        "app.js",
        "js/config.js",
        "js/filters.js",
        "js/theme.js",
        "js/toast.js",
        "js/utils.js",
    ]


def step_mypy() -> bool:
    """Paso 2: mypy strict sobre los módulos de producción."""
    step("PASO 2/7 - mypy strict (type check)")

    files = list(_PROD_PY_FILES)
    try:
        r = subprocess.run(
            [PYTHON, "-m", "mypy", *files],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=180, cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        print("  [FAIL] mypy excedió el timeout de 180s")
        if output:
            print(f"  Output parcial:\n{output[:2000]}")
        result("mypy strict", "FAIL", "Timeout después de 180s")
        return False
    except FileNotFoundError:
        print("  [SKIP] mypy no instalado. Ejecuta: pip install mypy")
        result("mypy strict", "SKIP", "mypy no instalado")
        return True

    stdout = r.stdout or ""

    if r.returncode == 0:
        print("  [OK] Sin errores de tipos")
        result("mypy strict", "PASS", "0 errores")
        return True

    # mypy encontró errores — mostrar las líneas relevantes
    error_lines = [
        line for line in stdout.split("\n")
        if "error:" in line or "note: unused" in line
    ]
    for line in error_lines:
        print(f"  {line}")

    n_errors = sum(1 for line in stdout.split("\n") if "error:" in line)
    if n_errors == 0:
        n_errors = len(error_lines)
    print(f"  [FAIL] mypy reportó {n_errors} errores de tipos")
    result("mypy strict", "FAIL", f"{n_errors} errores de tipos")
    return False


# Cobertura mínima por módulo de producción (baseline medido con la suite
# completa del CI el 2026-08-14; redondeado a 1 decimal). Un módulo que baje
# de su umbral hace FALLAR el paso pytest + cobertura. `launcher.py` (97.96)
# y `main.py` (88.0) se miden con tests de arranque que mockean argv/Popen.
_COVERAGE_THRESHOLDS: dict[str, float] = {
    "analisis_calidad.py": 75.0,
    "cache.py": 68.0,
    "config.py": 81.8,
    "launcher.py": 95.0,
    "main.py": 85.0,
    "models.py": 76.3,
    "ocr_engine.py": 94.6,
    "ocr_utils.py": 80.4,
    "process_all_pages.py": 91.2,
    "quality_analysis.py": 87.1,
    "ratelimit.py": 80.0,
    "runtime_diagnostics.py": 88.9,
    "server.py": 60.0,
    "translation_memory.py": 80.1,
    "translator.py": 70.0,
    "uocr_client.py": 60.0,
    "uocr_daemon.py": 60.0,
    "routes/__init__.py": 100.0,
    "routes/api.py": 80.6,
    "routes/main.py": 70.0,
}


COVERAGE_RC = PROJECT_ROOT / ".coveragerc"
COVERAGE_JSON = _resolve_basetemp() / "coverage.json"
# Reporte HTML de cobertura por módulo de producción (autocontenido, generado
# por _write_coverage_html desde el JSON). Se escribe en la raíz del repo para
# que GitHub Actions lo suba como artifact del PR. Está en .gitignore.
COVERAGE_HTML = PROJECT_ROOT / "coverage_html"
# Snapshot verificado del % cubierto por módulo medido en la base (main).
# El workflow lo refresca en cada push a main; el reporte HTML lo usa como
# "valor en la base del PR" para la columna de delta. A diferencia del JSON
# de coverage (gitignored), este archivo SÍ se commitea.
COVERAGE_BASE = PROJECT_ROOT / "coverage_base.json"


def _build_pytest_command(with_coverage: bool = True) -> list[str]:
    """Construye el comando pytest con aislamiento de temporales."""
    cmd = [
        PYTHON,
        "-m",
        "pytest",
        *_pytest_test_files(),
        "-v",
        "--tb=short",
        "--basetemp",
        str(_resolve_basetemp()),
    ]
    if with_coverage:
        # --cov=. mide por RUTA (--cov=cache resolvía al directorio cache/ en
        # vez de cache.py). .coveragerc omite site-packages/dist/env y tolera
        # archivos sin source (cv2/config-3.py). El JSON en el basetemp lo lee
        # la verificación por módulo (_check_module_coverage).
        cmd += [
            "--cov=.",
            "--cov-config",
            str(COVERAGE_RC),
            "--cov-report=term-missing",
            f"--cov-report=json:{COVERAGE_JSON}",
        ]
    return cmd


def _normalize_coverage_path(path: str) -> str:
    """Normaliza una ruta del reporte JSON de coverage a la forma relativa
    del proyecto (``routes/api.py`` en cualquier plataforma)."""
    p = path.replace("\\", "/")
    # Quitar el prefijo absoluto del proyecto si coverage lo incluyó.
    root_prefix = str(PROJECT_ROOT.resolve()).replace("\\", "/") + "/"
    if p.startswith(root_prefix):
        p = p[len(root_prefix):]
    return p


def _git_base_commit() -> str | None:
    """Resuelve la rama base contra la cual calcular el diff de la rama/PR.

    Orden de candidatos: ``GITHUB_BASE_REF`` (que GitHub Actions define en los
    PRs, con y sin prefijo ``origin/``) y luego remotos/ramas comunes en local
    (``origin/main``, ``master``, ``develop``...). Devuelve el primer ref que
    git pueda resolver, o None si no hay base disponible (repo sin remoto o
    historia insuficiente) — en ese caso el gate de cobertura cae al modo
    completo, verificando todos los módulos.
    """
    candidates: list[str] = []
    base_ref = os.environ.get("GITHUB_BASE_REF")
    if base_ref:
        candidates += [base_ref, f"origin/{base_ref}"]
    candidates += [
        "origin/main", "origin/master", "origin/develop",
        "main", "master", "develop",
    ]
    for cand in candidates:
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", cand],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                cwd=str(PROJECT_ROOT), timeout=30,
            )
        except OSError:
            # git no disponible — el gate cae al modo completo.
            return None
        if r.returncode == 0 and r.stdout.strip():
            return cand
    return None


def _touched_prod_modules() -> set[str] | None:
    """Módulos de producción modificados por el diff de la rama/PR.

    Combina dos fuentes:

    1. ``<base>...HEAD`` (diff de tres puntos): solo los cambios que la rama
       introduce sobre el merge-base — exactamente el alcance de un PR.
    2. ``HEAD`` (working tree + staged): los cambios sin commitear que el
       desarrollador está haciendo ahora mismo (el caso típico de correr el
       runner local con trabajo en curso). En GitHub Actions el tree está
       limpio, así que esta fuente no aporta nada.

    Devuelve None cuando no hay información de diff disponible (sin git o sin
    base resoluble) — el gate de cobertura entonces verifica todos los
    módulos, como antes.
    """
    base = _git_base_commit()
    if base is None:
        return None
    changed: set[str] = set()
    for spec in (f"{base}...HEAD", "HEAD"):
        try:
            r = subprocess.run(
                ["git", "diff", "--name-only", spec],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", cwd=str(PROJECT_ROOT), timeout=30,
            )
        except OSError:
            return None
        if r.returncode != 0:
            return None
        for ln in r.stdout.splitlines():
            ln = ln.strip()
            if ln:
                changed.add(ln.replace("\\", "/"))
    return {p for p in changed if p in _PROD_PY_FILES}


def _check_module_coverage(json_path: Path,
                           touched: set[str] | None = None,
                           result_name: str = "cobertura por módulo") -> bool:
    """Verifica la cobertura de los módulos de producción contra sus umbrales.

    Lee el reporte JSON de coverage generado por pytest-cov y compara el
    porcentaje de cada archivo de ``_PROD_PY_FILES`` contra
    ``_COVERAGE_THRESHOLDS``. Devuelve True si todos cumplen.

    ``result_name`` permite distinguir en el resumen la verificación con el
    arranque real del servidor (job server-test) de la de pytest.

    Con ``touched`` (módulos que el diff del PR toca) solo se verifican esos
    módulos: un módulo que el PR no toca y está bajo su umbral es estado
    global del repo, no responsabilidad del PR, así que se reporta como
    [WARN] y NO hace fallar el gate. Sin ``touched`` (sin diff disponible) se
    verifican todos, comportamiento previo.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  [FAIL] No se pudo leer el reporte de cobertura: {exc}")
        result("cobertura por módulo", "FAIL", "reporte JSON ilegible")
        return False

    files = data.get("files", {})
    by_module: dict[str, float] = {}
    for path, info in files.items():
        by_module[_normalize_coverage_path(path)] = float(
            info.get("summary", {}).get("percent_covered", 0.0))

    if touched is None:
        scope = list(_PROD_PY_FILES)
    else:
        scope = sorted(touched & set(_PROD_PY_FILES))

    ok = True
    for mod in scope:
        umbral = _COVERAGE_THRESHOLDS.get(mod)
        if umbral is None:
            # Módulo sin umbral definido: no se puede verificar — mejor fallar
            # para que el baseline se actualice deliberadamente.
            print(f"  [WARN] {mod}: sin umbral en _COVERAGE_THRESHOLDS")
            ok = False
            continue
        pct = by_module.get(mod, 0.0)
        if pct < umbral - 0.05:
            print(f"  [FAIL] {mod}: {pct:.1f}% < umbral {umbral:.1f}%")
            ok = False
        else:
            print(f"  [OK]   {mod}: {pct:.1f}% (umbral {umbral:.1f}%)")

    if touched is not None:
        # Estado global: módulos que el diff NO toca pero están bajo umbral.
        # Se reportan como advertencia — son deuda del repo, no del PR.
        for mod in sorted(set(_PROD_PY_FILES) - touched):
            umbral = _COVERAGE_THRESHOLDS.get(mod)
            if umbral is None:
                continue
            pct = by_module.get(mod, 0.0)
            if pct < umbral - 0.05:
                print(f"  [WARN] {mod}: {pct:.1f}% bajo umbral {umbral:.1f}% "
                      "(no tocado por el diff — estado global)")

    status = "PASS" if ok else "FAIL"
    detalle = "todos cumplen umbral" if ok else "algún módulo bajo su umbral"
    result(result_name, status, detalle)
    return ok


_COV_HTML_CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif;
       margin: 2rem auto; max-width: 920px; color: #1f2937;
       background: #ffffff; padding: 0 1rem; }
h1 { font-size: 1.35rem; margin-bottom: 0.25rem; }
.meta { color: #6b7280; font-size: 0.85rem; margin-bottom: 1.25rem; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; }
th, td { text-align: left; padding: 0.45rem 0.6rem;
         border-bottom: 1px solid #e5e7eb; vertical-align: top; }
th { background: #f9fafb; position: sticky; top: 0; z-index: 1;
     box-shadow: 0 1px 0 #e5e7eb; }
td.num { font-variant-numeric: tabular-nums; text-align: right; }
.badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px;
         font-size: 0.72rem; font-weight: 700; white-space: nowrap; }
.ok { background: #dcfce7; color: #166534; }
.fail { background: #fee2e2; color: #991b1b; }
.warn { background: #fef3c7; color: #92400e; }
tr.row-fail { background: #fee2e2; }
tr.row-warn { background: #fef3c7; }
.delta-up { color: #166534; font-weight: 700; }
.delta-down { color: #991b1b; font-weight: 700; }
.delta-same { color: #6b7280; }
.tocado { color: #2563eb; font-weight: 700; }
.lines { color: #6b7280; font-size: 0.8rem; }
.resumen { display: flex; gap: 1rem; margin-bottom: 1rem; font-size: 0.85rem; }
.resumen b { font-variant-numeric: tabular-nums; }
.legend { display: flex; flex-wrap: wrap; gap: 0.6rem 1.25rem;
          margin-bottom: 1rem; font-size: 0.78rem; color: #4b5563; }
.legend-item { display: inline-flex; align-items: center; }
.dot { display: inline-block; width: 0.7rem; height: 0.7rem;
       border-radius: 999px; margin-right: 0.4rem; flex: none; }
.dot.ok { background: #dcfce7; border: 1px solid #86efac; }
.dot.warn { background: #fef3c7; border: 1px solid #fde68a; }
.dot.fail { background: #fee2e2; border: 1px solid #fecaca; }
.table-wrap { overflow-x: auto; overflow-y: auto; max-height: 70vh;
              border: 1px solid #e5e7eb; border-radius: 8px; }
"""


def _load_base_coverage() -> dict[str, float] | None:
    """Lee el snapshot de cobertura de la base (``coverage_base.json``).

    Devuelve ``{módulo: % cubierto}`` normalizado con las rutas de
    ``_PROD_PY_FILES``, o ``None`` si el archivo no existe o está vacío
    (local sin snapshot, o main todavía sin baseline). Tolerante a basura:
    los valores no numéricos se descartan.
    """
    try:
        data = json.loads(COVERAGE_BASE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    out: dict[str, float] = {}
    for mod, val in data.items():
        if not isinstance(mod, str):
            continue
        try:
            out[mod] = float(val)
        except (TypeError, ValueError):
            continue
    return out or None


def _write_base_coverage(json_path: Path) -> None:
    """Refresca ``coverage_base.json`` con el % medido en el JSON dado.

    El workflow lo ejecuta tras cada push a main: el snapshot queda como la
    "base del PR" contra la que el reporte HTML muestra el delta por módulo.
    Solo escribe módulos de producción (``_PROD_PY_FILES``), coherente con el
    resto del pipeline. Idempotente: si el contenido no cambia, el paso de
    commit del workflow no genera commit.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  [WARN] No se pudo refrescar el baseline de cobertura: {exc}")
        return
    files = data.get("files", {})
    by_module: dict[str, dict[str, object]] = {}
    for path, info in files.items():
        by_module[_normalize_coverage_path(path)] = info

    out: dict[str, float] = {}
    for mod in _PROD_PY_FILES:
        info = by_module.get(mod, {})
        summary_raw = info.get("summary", {})
        summary = summary_raw if isinstance(summary_raw, dict) else {}
        out[mod] = round(float(summary.get("percent_covered", 0.0)), 2)
    COVERAGE_BASE.write_text(
        json.dumps(out, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    print(f"  [OK] Baseline de cobertura actualizado: {len(out)} módulos")


def _missing_ranges(missing: list[int]) -> str:
    """Compacta una lista de líneas sin cubrir en rangos '12, 45-48, 90'."""
    if not missing:
        return ""
    nums = sorted(set(missing))
    ranges: list[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def _write_coverage_html(json_path: Path, out_dir: Path,
                         touched: set[str] | None = None,
                         base_pct: dict[str, float] | None = None) -> None:
    """Genera el reporte HTML autocontenido de cobertura por módulo.

    Lista cada módulo de ``_PROD_PY_FILES`` con su % cubierto, umbral,
    estado y líneas sin cubrir — la vista "por módulo" que el gate verifica.
    ``touched`` (módulos que toca el diff del PR) marca qué módulos entran al
    gate; los que el PR no toca se muestran como "bajo umbral (no tocado)".
    ``base_pct`` (snapshot de la base, ``coverage_base.json``) agrega una
    columna Δ con el cambio contra la base del PR, coloreada por signo.
    Funciona igual en local y en CI; GitHub Actions lo sube como artifact.
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"  [WARN] No se pudo generar el HTML de cobertura: {exc}")
        return
    files = data.get("files", {})
    by_module: dict[str, dict[str, object]] = {}
    for path, info in files.items():
        by_module[_normalize_coverage_path(path)] = info

    rows: list[str] = []
    n_ok = n_fail = n_low_untouched = 0
    for mod in _PROD_PY_FILES:
        info = by_module.get(mod, {})
        summary_raw = info.get("summary", {})
        summary = summary_raw if isinstance(summary_raw, dict) else {}
        pct = float(summary.get("percent_covered", 0.0))
        statements = int(summary.get("num_statements", 0))
        missing_raw = info.get("missing_lines")
        missing_list = (list(missing_raw) if isinstance(missing_raw, list)
                        else [])
        missing: list[int] = []
        for n in missing_list:
            if isinstance(n, int):
                missing.append(n)
        umbral = _COVERAGE_THRESHOLDS.get(mod)
        if umbral is None:
            badge, cls = "SIN UMBRAL", "warn"
        elif pct < umbral - 0.05:
            if touched is None or mod in touched:
                badge, cls = "FAIL", "fail"
                n_fail += 1
            else:
                badge, cls = "BAJO (no tocado)", "warn"
                n_low_untouched += 1
        else:
            badge, cls = "OK", "ok"
            n_ok += 1
        tocado = (" <span class=\"tocado\">\u25b2 diff</span>"
                  if touched is not None and mod in touched else "")
        # La fila hereda el tinte del badge: rojo para FAIL (módulo del diff
        # bajo umbral), ámbar para BAJO (no tocado) / SIN UMBRAL. Así el
        # artifact muestra de un vistazo qué módulos están bajo umbral.
        row_cls = f" class=\"row-{cls}\"" if cls in ("fail", "warn") else ""
        umbral_str = f"{umbral:.1f}%" if umbral is not None else "\u2014"
        lines_str = _missing_ranges(missing) if missing else "\u2014"
        # Delta contra la base del PR: solo si hay snapshot y el módulo está
        # en él; color por signo (verde si subió, rojo si bajó).
        if base_pct is not None and mod in base_pct:
            delta = pct - base_pct[mod]
            if delta > 0.05:
                delta_cell = (f"<td class=\"num delta-up\">"
                              f"+{delta:.1f}</td>")
            elif delta < -0.05:
                delta_cell = (f"<td class=\"num delta-down\">"
                              f"{delta:.1f}</td>")
            else:
                # Sin cambio (dentro de ±0.05): '0.0' neutro, sin signo
                delta_cell = "<td class=\"num delta-same\">0.0</td>"
        else:
            delta_cell = "<td class=\"num delta-same\">\u2014</td>"
        rows.append(
            f"<tr{row_cls}><td>{mod}{tocado}</td>"
            f"<td class=\"num\">{pct:.1f}%</td>"
            f"<td class=\"num\">{umbral_str}</td>"
            f"<td>{statements}</td>"
            + delta_cell
            + f"<td><span class=\"badge {cls}\">{badge}</span></td>"
            + f"<td class=\"lines\">{lines_str}</td></tr>")

    if touched is None:
        alcance = "modo completo (sin diff disponible)"
    elif touched:
        alcance = (f"{len(touched)} módulo(s) tocados por el diff: "
                   + ", ".join(sorted(touched)))
    else:
        alcance = "el diff no toca módulos de producción"

    html = (
        "<!DOCTYPE html><html lang=\"es\"><head><meta charset=\"utf-8\">"
        "<title>Cobertura por módulo</title><style>" + _COV_HTML_CSS
        + "</style></head><body>"
        + f"<h1>Cobertura por módulo de producción</h1>"
        + f"<div class=\"meta\">Generado por run_ci.py \u2014 {alcance}</div>"
        + f"<div class=\"resumen\"><span><b>{n_ok}</b> OK</span>"
        + f"<span><b>{n_fail}</b> FAIL</span>"
        + f"<span><b>{n_low_untouched}</b> bajo umbral (no tocado)</span></div>"
        + "<div class=\"legend\">"
        + "<span class=\"legend-item\"><span class=\"dot ok\"></span>"
        + "<b>OK</b> — cumple su umbral</span>"
        + "<span class=\"legend-item\"><span class=\"dot warn\"></span>"
        + "<b>Ámbar</b> — bajo umbral, pero el PR no lo toca (deuda del "
        + "repo) o sin umbral configurado</span>"
        + "<span class=\"legend-item\"><span class=\"dot fail\"></span>"
        + "<b>Rojo</b> — módulo del diff bajo su umbral: el PR lo hace "
        + "bajar</span>"
        + "<span class=\"legend-item\"><span class=\"tocado\">▲</span> "
        + "diff — módulo que toca el diff del PR</span>"
        + "</div>"
        + "<div class=\"table-wrap\"><table><thead><tr><th>Módulo</th>"
        + "<th class=\"num\">Cubierto</th>"
        + "<th class=\"num\">Umbral</th><th class=\"num\">Sentencias</th>"
        + ("<th class=\"num\">Δ base</th>" if base_pct is not None else "")
        + "<th>Estado</th><th>Líneas sin cubrir</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div></body></html>")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _artifact_url() -> str | None:
    """URL del run de Actions donde viven los artifacts de cobertura.

    ``https://<server>/<repo>/actions/runs/<run_id>#artifacts`` — la sección
    de artifacts del run, donde está ``coverage-report`` (y
    ``coverage-report-server``). Solo disponible en Actions (depende de
    ``GITHUB_REPOSITORY`` y ``GITHUB_RUN_ID``); fuera de CI devuelve
    ``None`` y el callout no agrega enlace.
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not repo or not run_id:
        return None
    server = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    return f"{server}/{repo}/actions/runs/{run_id}#artifacts"


def _coverage_summary_markdown(json_path: Path,
                               touched: set[str] | None = None,
                               artifact_url: str | None = None) -> str:
    """Markdown de la cobertura por módulo para ``GITHUB_STEP_SUMMARY``.

    Misma lógica de estado que el gate (``_check_module_coverage``): tabla
    por módulo de ``_PROD_PY_FILES`` con % cubierto, umbral y estado. Un
    callout al tope destaca los módulos que bajaron de umbral (en negrita y
    con el delta) y una línea con los que más mejoraron contra la base
    (``coverage_base.json``), para balancear la lectura; los módulos que el
    diff no toca y están bajo umbral se muestran como "bajo (no tocado)" —
    deuda del repo, no del PR. ``artifact_url`` (si se da)
    agrega un enlace directo al reporte HTML del artifact en el callout.
    Devuelve el texto listo para appendear a ``$GITHUB_STEP_SUMMARY``
    (tolerante a un JSON ausente: pytest no corrió o falló temprano).
    """
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return (f"### Cobertura por módulo\n\n"
                f"No se pudo leer el reporte de cobertura ({exc}).\n\n")

    files = data.get("files", {})
    by_module: dict[str, float] = {}
    for path, info in files.items():
        by_module[_normalize_coverage_path(path)] = float(
            info.get("summary", {}).get("percent_covered", 0.0))

    rows: list[str] = []
    bajos: list[str] = []
    bajos_mods: set[str] = set()
    for mod in _PROD_PY_FILES:
        umbral = _COVERAGE_THRESHOLDS.get(mod)
        pct = by_module.get(mod, 0.0)
        tocado = " (diff)" if touched is not None and mod in touched else ""
        if umbral is None:
            estado = "⚠️ sin umbral"
            bajos.append(f"`{mod}` (sin umbral)")
            bajos_mods.add(mod)
            mod_cell = mod
        elif pct < umbral - 0.05:
            if touched is None or mod in touched:
                estado = "❌ **BAJO UMBRAL**"
                bajos.append(f"`{mod}` ({pct:.1f}% < {umbral:.1f}%)")
                bajos_mods.add(mod)
                # Negrita en la columna módulo para que la fila salte a la
                # vista en la conversación del PR
                mod_cell = f"**{mod}**"
            else:
                estado = "⬇️ bajo (no tocado)"
                mod_cell = mod
        else:
            estado = "✅ OK"
            mod_cell = mod
        umbral_str = f"{umbral:.1f}%" if umbral is not None else "—"
        rows.append(
            f"| {mod_cell}{tocado} | {pct:.1f}% | {umbral_str} | {estado} |")

    # Mejoras contra la base (coverage_base.json): el mismo delta que la
    # columna Δ del reporte HTML. Solo módulos que NO están bajo umbral (no
    # mezclar un FAIL con su mejora relativa), con delta > 0.05 para evitar
    # ruido de redondeo, y top 3 para mantener el callout compacto.
    base = _load_base_coverage()
    mejoras: list[tuple[str, float]] = []
    if base:
        for mod in _PROD_PY_FILES:
            if mod in bajos_mods:
                continue
            base_val = base.get(mod)
            if base_val is None:
                continue
            delta = by_module.get(mod, 0.0) - base_val
            if delta > 0.05:
                mejoras.append((mod, delta))
        mejoras.sort(key=lambda item: item[1], reverse=True)
        mejoras = mejoras[:3]

    if touched is None:
        alcance = "modo completo (sin diff)"
    elif touched:
        alcance = f"{len(touched)} módulo(s) tocados por el diff"
    else:
        alcance = "el diff no toca módulos de producción"

    # Callout al tope: lo primero que se lee en la conversación del PR. Con
    # módulos bajo umbral, lista cuáles y el delta; si todo cumple, el OK.
    # Además, una línea con los módulos que MÁS mejoraron contra la base,
    # para balancear la lectura. Con artifact_url se agrega un enlace
    # directo al reporte completo al final del callout.
    enlace = (f" — [ver reporte completo]({artifact_url})"
              if artifact_url else "")
    lineas: list[str] = []
    if bajos:
        lineas.append(f"> ⚠️ **{len(bajos)} módulo(s) bajo umbral:** "
                      + ", ".join(sorted(set(bajos))))
    else:
        lineas.append("> ✅ **Todos los módulos cumplen su umbral**")
    if mejoras:
        mejo = ", ".join(f"`{mod}` (+{delta:.1f})" for mod, delta in mejoras)
        lineas.append(f"> 📈 **Mejoran:** {mejo}")
    destacado = "\n".join(lineas) + enlace + "\n\n"

    return (destacado
            + f"### Cobertura por módulo ({alcance})\n\n"
            + "| Módulo | Cubierto | Umbral | Estado |\n"
            + "|---|---|---|---|\n"
            + "\n".join(rows) + "\n")


def write_coverage_summary() -> None:
    """Append del resumen de cobertura a ``$GITHUB_STEP_SUMMARY``.

    Paso del workflow (job ``lint-test``) que se ejecuta tras correr
    ``run_ci.py``: escribe la tabla de cobertura por módulo en el step
    summary del run, destacando los módulos que bajaron de umbral. Si el
    entorno no define ``GITHUB_STEP_SUMMARY`` (p. ej. en local) solo avisa.
    """
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        print("[WARN] GITHUB_STEP_SUMMARY no definido — resumen omitido")
        return
    md = _coverage_summary_markdown(COVERAGE_JSON, _touched_prod_modules(),
                                    _artifact_url())
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(md)
    except OSError as exc:
        print(f"[WARN] No se pudo escribir el resumen de cobertura: {exc}")


# Marcador que identifica el comentario del bot de cobertura en el PR: al
# buscar/reemplazar el comentario previo se evita spamear uno por commit.
_COVERAGE_COMMENT_MARKER = "<!-- tvp-coverage-report -->"


def _github_api(endpoint: str, method: str = "GET",
                payload: dict[str, Any] | None = None) -> Any:
    """Llamada a la API de GitHub autenticada con ``GITHUB_TOKEN``.

    Devuelve el JSON decodificado, o None ante cualquier error (token
    ausente, HTTP != 2xx, red) — el comentario es informativo y nunca debe
    hacer fallar el job.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return None
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(
        f"https://api.github.com{endpoint}", data=data,
        headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return json.loads(body) if body else None
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:  # nosec: fp puede faltar en stubs de test
            err_body = ""
        print(f"  [WARN] GitHub API {method} {endpoint} -> "
              f"{exc.code}: {err_body}")
        return None
    except OSError as exc:
        print(f"  [WARN] GitHub API {method} {endpoint} -> {exc}")
        return None


def write_coverage_comment() -> None:
    """Comenta la cobertura por módulo en el PR con el bot (GITHUB_TOKEN).

    Publica la misma tabla de ``_coverage_summary_markdown`` (por módulo:
    % cubierto, umbral y estado) como comentario del PR, marcada con
    ``_COVERAGE_COMMENT_MARKER``: busca el comentario previo del bot y lo
    actualiza (PATCH) en vez de crear uno por commit. Solo aplica en
    ``pull_request`` (en push a main no hay PR); sin token o con error de
    API se avisa y no se falla el job.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[WARN] GITHUB_TOKEN no definido — comentario de cobertura "
              "omitido")
        return
    event_name = os.environ.get("GITHUB_EVENT_NAME", "")
    if event_name != "pull_request":
        print(f"  [INFO] GITHUB_EVENT_NAME={event_name or '(vacío)'} — solo "
              "se comenta en PRs")
        return
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if not event_path or not Path(event_path).exists():
        print("[WARN] GITHUB_EVENT_PATH no disponible — comentario omitido")
        return
    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"[WARN] No se pudo leer GITHUB_EVENT_PATH: {exc}")
        return
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    pr = (event.get("pull_request") or {}).get("number")
    if not repo or not pr:
        print("[WARN] Sin GITHUB_REPOSITORY o número de PR — comentario "
              "omitido")
        return

    md = _coverage_summary_markdown(COVERAGE_JSON, _touched_prod_modules(),
                                    _artifact_url())
    body = (md + "\n\n<sub>Comentario automático del CI de cobertura por "
            f"módulo.</sub>\n{_COVERAGE_COMMENT_MARKER}")
    comments_url = f"/repos/{repo}/issues/{pr}/comments"

    comments = _github_api(comments_url)
    if isinstance(comments, list):
        for comment in comments:
            if _COVERAGE_COMMENT_MARKER in str(comment.get("body", "")):
                cid = comment.get("id")
                if cid is not None:
                    _github_api(
                        f"/repos/{repo}/issues/comments/{cid}",
                        method="PATCH", payload={"body": body})
                    print(f"  [OK] Comentario de cobertura actualizado "
                          f"(id {cid})")
                    return

    created = _github_api(comments_url, method="POST",
                          payload={"body": body})
    if created is not None:
        print(f"  [OK] Comentario de cobertura creado en el PR #{pr}")
    else:
        print("  [WARN] No se pudo crear el comentario de cobertura")


def step_pytest(with_coverage: bool = True) -> bool:
    """Paso 3: pytest con todos los tests unitarios."""
    label = "pytest + cobertura" if with_coverage else "pytest"
    step(f"PASO 4/7 - {label} (suite: translator + OCR + API + procesamiento)")

    test_files = [
        "tests/test_translator.py",
        "tests/test_ocr_utils.py",
        "tests/test_ocr_functions.py",
        "tests/test_api.py",
        "tests/test_ocr_engine.py",
        "tests/test_uocr_daemon.py",
        "tests/test_uocr_client.py",
        "tests/test_ratelimit.py",
        "tests/test_runtime_diagnostics.py",
        "tests/test_translation_memory.py",
        "tests/test_models.py",
        "tests/test_packaging.py",
        "tests/test_path_security.py",
        "tests/test_cache.py",
        # Procesamiento completo y herramientas de calibración del detector.
        # Se incluyen para que los fallos silenciosos de batch/single y los
        # tests de YOLO no queden fuera del CI principal.
        "tests/test_process_all_pages.py",
        "tests/test_correccion_detector.py",
        "tests/test_corrector_oro.py",
        "tests/test_manga_ocr.py",
    ]

    cmd = _build_pytest_command(with_coverage=with_coverage)
    try:
        r = subprocess.run(
            cmd,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            # 2026-08-15: la suite con --cov mide ~189 s (972 tests) — el
            # timeout de 180 s dejaba el CI rojo por margen. 300 s sigue
            # detectando cuelgues reales (el daemon VLM cargando ralentiza
            # la suite ~2×) con margen para el crecimiento de tests.
            timeout=300, cwd=str(PROJECT_ROOT),
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        print("  [FAIL] pytest excedió el timeout de 300s")
        if output:
            print(f"  Output parcial:\n{output[:2000]}")
        result("pytest (suite completa)", "FAIL", "Timeout después de 300s")
        return False

    # Parsear resumen de pytest
    stdout = r.stdout or ""
    stderr = r.stderr or ""
    summary_line = ""
    for line in stdout.split("\n"):
        line_stripped = line.strip()
        # Buscar linea como "247 passed in 53.2s" o "245 passed, 2 failed in 55.1s"
        if "passed" in line_stripped or "failed" in line_stripped:
            summary_line = line_stripped
        # Mostrar failures
        if "FAILED" in line_stripped:
            print(f"  {line_stripped}")

    # Extraer conteos
    passed_count = 0
    failed_count = 0
    m_passed = re.search(r"(\d+) passed", stdout)
    m_failed = re.search(r"(\d+) failed", stdout)
    if m_passed:
        passed_count = int(m_passed.group(1))
    if m_failed:
        failed_count = int(m_failed.group(1))

    # Extraer tiempo
    time_str = ""
    m_time = re.search(r"in (\d+\.?\d*)s", stdout)
    if m_time:
        time_str = f"en {m_time.group(1)}s"

    ok = r.returncode == 0
    total = passed_count + failed_count

    # Mostrar resumen detallado
    print(f"  {'[OK]' if ok else '[FAIL]'} {total} tests: {passed_count} passed",
          end="")
    if failed_count > 0:
        print(f", {failed_count} FAILED", end="")
    if time_str:
        print(f" {time_str}", end="")
    print()

    if stderr and "Error" in stderr:
        # Mostrar solo errores relevantes (no warnings de deprecation)
        for line in stderr.split("\n"):
            if "Error" in line or "Traceback" in line:
                print(f"  STDERR: {line.strip()}")

    if total == 0:
        # Fallback: mostrar output completo si no se pudo parsear
        print(f"  Output completo:\n{stdout[:1000]}")
        if stderr:
            print(f"  STDERR:\n{stderr[:2000]}")
        result("pytest (suite completa)", "FAIL", "Pytest terminó sin resumen interpretable")
        return False

    if ok:
        status = "PASS"
        detail = f"{passed_count}/{total} passed {time_str}"
    else:
        status = "FAIL"
        detail = f"{failed_count} FAILED en {total} tests {time_str}"

    result("pytest (suite completa)", status, detail)

    # Verificación de cobertura por módulo: ningún archivo de producción puede
    # bajar de su umbral. Solo aplica cuando se midió cobertura y los tests
    # pasaron (si los tests fallan, el motivo ya está reportado).
    #
    # El gate se acota al diff de la rama/PR: solo falla si un módulo QUE EL
    # PR TOCA baja de su umbral. Los módulos que el PR no toca y están bajo
    # umbral son estado global del repo (deuda acumulada) y solo se reportan
    # como [WARN]. Si no hay diff disponible (sin git/base), se verifican
    # todos, comportamiento previo.
    cov_ok = True
    if with_coverage:
        touched = _touched_prod_modules()
        if touched is not None:
            if touched:
                mods = ", ".join(sorted(touched))
                print(f"  Diff vs base: {len(touched)} módulo(s) de producción "
                      f"tocado(s): {mods}")
            else:
                print("  Diff vs base: el PR no toca módulos de producción "
                      "(cobertura no aplica, solo se informa el estado global)")
        # Reporte HTML por módulo: se genera siempre que se midió cobertura
        # (incluso si los tests fallaron — el reporte ayuda a diagnosticar).
        _write_coverage_html(COVERAGE_JSON, COVERAGE_HTML, touched,
                             _load_base_coverage())
        if ok:
            cov_ok = _check_module_coverage(COVERAGE_JSON, touched)

    return ok and cov_ok


def step_server(with_coverage: bool = True) -> bool:
    """Paso 4: Iniciar servidor y verificar health + endpoints.

    Con ``with_coverage`` el servidor se lanza bajo ``coverage run --append``
    (misma base de datos .coverage que pytest-cov) y, tras verificar los
    endpoints, se regenera el JSON de cobertura y se re-corre el gate por
    módulo: el arranque real del servidor cuenta para la cobertura. Es lo
    que usa el job ``server-test`` del workflow. Solo aplica si pytest ya
    midió (COVERAGE_JSON existe); sin baseline el servidor corre normal.
    """
    global _server_proc
    trace_server = with_coverage and COVERAGE_JSON.exists()
    label = "Servidor Flask + tests" + (" (con cobertura)" if trace_server else "")
    step(f"PASO 5/7 - {label}")

    # Matar servidor previo si existe (con fallback a netstat+taskkill)
    killed = _kill_process_on_port(SERVER_PORT)
    if killed:
        print(f"  Servidor anterior detenido")
        time.sleep(2)

    # Iniciar servidor. Con cobertura, se lanza bajo `coverage run --append`
    # sobre el mismo .coverage que escribió pytest-cov, para que el arranque
    # real del servidor (imports, rutas, blueprints) sume a la medición.
    log_path = os.path.join(PROJECT_ROOT, "ci_server.log")
    if trace_server:
        server_cmd = [PYTHON, "-u", "-m", "coverage", "run", "--append",
                      "--rcfile", str(COVERAGE_RC), "server.py"]
    else:
        server_cmd = [PYTHON, "-u", "server.py"]
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            server_cmd,
            stdout=log_file, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    print(f"  Servidor iniciado (PID {proc.pid})")
    # Registrar el proc de inmediato: cualquier fallo posterior (health,
    # endpoints) debe poder detenerlo y no dejar procesos huerfanos.
    _server_proc = proc

    # Esperar health. Timeout generoso: en el primer arranque la carga de
    # modelos puede tardar >20s, y server_health reintenta los TimeoutError.
    # Con tracing de coverage el arranque es más lento — se da más margen.
    health = server_health(timeout=90 if trace_server else 60)
    if health is None:
        with open(log_path, encoding="utf-8") as f:
            tail = "".join(f.readlines()[-10:])
        print(f"  STDERR LOG:\n{tail}")
        result("Server startup", "FAIL", "No respondio en /api/health")
        _stop_server()
        return False

    mem = health.get("memory", "?")
    print(f"  [OK] Servidor OK - db={health.get('db_available')}, mem={mem}MB")
    result("Server startup", "PASS", f"PID {proc.pid}, mem={mem}MB")

    try:
        # Test translate
        status, data = http_post("/api/translate", {
            "text": "Hola mundo", "source": "es", "target": "en"
        }, timeout=30)
        translated = data.get("translatedText", "N/A")
        engine = data.get("engine", "?")
        ok = status == 200 and translated and translated != "N/A"
        print(f"  {'[OK]' if ok else '[FAIL]'} GET /api/translate - '{translated}' (engine: {engine})")
        result("API translate", "PASS" if ok else "FAIL",
               f"'Hola mundo' -> '{translated}'")

        # Test translate-batch
        status, data = http_post("/api/translate-batch", {
            "texts": ["Buenos dias", "Gracias"], "source": "es", "target": "en"
        }, timeout=30)
        results_data = data.get("results", [])
        ok = status == 200 and len(results_data) == 2
        print(f"  {'[OK]' if ok else '[FAIL]'} POST /api/translate-batch - {len(results_data)} resultados")
        result("API translate-batch", "PASS" if ok else "FAIL",
               f"{len(results_data)}/2 traducidos")

        # Test config
        code, _ = http_get("/api/config", timeout=5)
        ok = code == 200
        print(f"  {'[OK]' if ok else '[FAIL]'} GET /api/config")
        result("API config", "PASS" if ok else "FAIL", "")

        # Test static files
        for path in ["/app.js", "/styles.css", "/"]:
            code, body = http_get(path, timeout=5)
            ok = code == 200
            print(f"  {'[OK]' if ok else '[FAIL]'} GET {path} ({len(body)}b)")
            result(f"Static {path}", "PASS" if ok else "FAIL", f"{len(body)}b")
    except Exception:
        # Limpiar el servidor ante cualquier error inesperado (p.ej. un
        # TimeoutError en un endpoint) para no dejar procesos huerfanos,
        # y relanzar para que el CI falle de forma visible.
        _stop_server()
        raise

    # El arranque real ya se verificó (health + endpoints). Con cobertura
    # activa, se detiene el servidor limpiamente (SIGINT → atexit → coverage
    # flushea), se regenera el JSON combinado y se re-corre el gate por
    # módulo. En el modo --full el servidor sigue vivo para el stress test.
    if trace_server:
        return _refresh_coverage_after_server()

    # El proc queda registrado para que main() lo detenga al final del CI
    # (o el stress test pueda usarlo si corre con --full).
    return True


_server_proc: subprocess.Popen[bytes] | None = None


def _kill_process_on_port(port: int) -> bool:
    """Mata procesos escuchando en un puerto. Fallback netstat+taskkill si psutil no esta."""
    try:
        import psutil
        for proc in psutil.process_iter(["pid", "name"]):
            try:
                for conn in proc.net_connections(kind="inet"):
                    if conn.laddr.port == port and conn.status == "LISTEN":
                        proc.kill()
                        return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return False
    except ImportError:
        pass

    # Fallback: netstat + taskkill
    try:
        ns_result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=10
        )
        for line in ns_result.stdout.split("\n"):
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1] if parts else ""
                if pid and pid != "0":
                    subprocess.run(
                        ["taskkill", "-f", "-pid", pid],
                        capture_output=True, timeout=5
                    )
                    return True
    except Exception:
        pass
    return False


def _stop_server() -> None:
    """Detiene el servidor CI si esta corriendo."""
    global _server_proc
    if _server_proc is not None:
        try:
            _server_proc.kill()
            _server_proc.wait(timeout=5)
            _kill_process_on_port(SERVER_PORT)
            print("  Servidor detenido")
        except Exception:
            pass
        _server_proc = None


def _stop_server_gracefully(timeout: float = 10.0) -> bool:
    """Detiene el servidor de forma que coverage flushee su data.

    En POSIX un SIGINT deja que Python corra atexit (coverage escribe el
    .coverage con el arranque real del servidor). En Windows el equivalente
    requiere process group (CREATE_NEW_PROCESS_GROUP), así que se degrada al
    kill duro — pierde la data del servidor pero conserva la de pytest, ya
    escrita. Devuelve True si se flusheó limpiamente.
    """
    global _server_proc
    proc = _server_proc
    if proc is None:
        return False
    if os.name != "nt":
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=timeout)
            _server_proc = None
            print("  Servidor detenido (SIGINT — coverage flusheado)")
            return True
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
    _stop_server()
    return False


def _refresh_coverage_after_server() -> bool:
    """Re-verifica el gate de cobertura incluyendo el arranque real.

    El servidor corrió bajo ``coverage run --append`` sobre el mismo
    ``.coverage`` que pytest-cov. Se detiene limpiamente (SIGINT en POSIX —
    atexit flushea la data del servidor), se regenera el JSON combinado y se
    re-corre ``_check_module_coverage``: así el arranque real del servidor
    cuenta para la cobertura por módulo. En Windows el kill duro pierde la
    data del servidor y la re-verificación repite la de pytest (idéntica).
    Si la regeneración falla (tooling), no se bloquea el paso.
    """
    flushed = _stop_server_gracefully()
    if not flushed:
        print("  [WARN] Kill duro del servidor — la cobertura del arranque "
              "real no quedó disponible (se re-verifica la data de pytest)")
    try:
        r = subprocess.run(
            [PYTHON, "-m", "coverage", "json", "--rcfile",
             str(COVERAGE_RC), "-o", str(COVERAGE_JSON)],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, cwd=str(PROJECT_ROOT),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"  [WARN] No se pudo regenerar el JSON de cobertura: {exc}")
        return True
    if r.returncode != 0:
        print(f"  [WARN] coverage json falló: {r.stderr.strip()[:300]}")
        return True

    touched = _touched_prod_modules()
    if touched is not None and not touched:
        print("  Diff vs base: el PR no toca módulos de producción "
              "(cobertura no aplica)")
    ok = _check_module_coverage(COVERAGE_JSON, touched,
                                result_name="cobertura por módulo (con servidor)")
    # El reporte HTML del artifact también refleja la data combinada.
    _write_coverage_html(COVERAGE_JSON, COVERAGE_HTML, touched,
                         _load_base_coverage())
    if ok:
        print("  [OK] cobertura re-verificada con el arranque real del servidor")
    return ok


def _export_results_json(start_time: float) -> None:
    """Exporta los resultados del CI a un archivo JSON para reportes."""
    duration_sec = time.time() - start_time
    duration_str = f"{int(duration_sec // 60)}m {int(duration_sec % 60)}s"

    data = {
        "steps": RESULTS,
        "stress": _STRESS_RESULT,
        "duration": duration_str,
        "duration_sec": round(duration_sec, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(RESULTS),
            "passed": sum(1 for r in RESULTS if r["status"] == "PASS"),
            "failed": sum(1 for r in RESULTS if r["status"] == "FAIL"),
            "warned": sum(1 for r in RESULTS if r["status"] == "WARN"),
            "skipped": sum(1 for r in RESULTS if r["status"] == "SKIP"),
        }
    }

    json_path = os.path.join(PROJECT_ROOT, "ci_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  Resultados exportados: {json_path}")


def _generate_report() -> None:
    """Genera reporte HTML del CI usando generate_report.py."""
    report_script = os.path.join(PROJECT_ROOT, "generate_report.py")
    if not os.path.exists(report_script):
        print("  [WARN] generate_report.py no encontrado, no se generara reporte HTML")
        return

    json_path = os.path.join(PROJECT_ROOT, "ci_results.json")
    html_path = os.path.join(PROJECT_ROOT, "ci_report.html")

    r = subprocess.run(
        [PYTHON, report_script, json_path, "-o", html_path],
        capture_output=True, text=True, timeout=30,
    )
    if r.returncode == 0:
        print(f"  Reporte HTML generado: {html_path}")
    else:
        print(f"  [WARN] Error generando reporte: {r.stderr[:200]}")


def step_analisis_calidad() -> bool:
    """Paso 5: analisis_calidad.py (calidad de traducciones)."""
    step("PASO 6/7 - analisis_calidad.py")

    corpus_path = os.path.join(PROJECT_ROOT, "resultados_progreso.json")
    if not os.path.exists(corpus_path):
        print("  [SKIP] resultados_progreso.json no encontrado")
        result("analisis_calidad.py", "SKIP", "Corpus no disponible")
        return True

    r = run_python(["analisis_calidad.py"], timeout=60)
    if r is None:
        result("analisis_calidad.py", "FAIL", "Timeout")
        return False

    ok = r.returncode == 0

    # Extraer tasa de aceptacion y cobertura de metadatos
    accept_rate = None
    metadata_coverage = None
    for line in (r.stdout or "").split("\n"):
        if "Tasa de aceptacion global" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                accept_rate = float(m.group(1))
        if "Cobertura de metadatos" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                metadata_coverage = float(m.group(1))

    if ok and accept_rate is not None:
        status = "PASS" if accept_rate >= 75 else "WARN"
        detail = f"Tasa de aceptacion: {accept_rate}%"
        if metadata_coverage is not None and metadata_coverage < 80:
            detail += f"; metadatos: {metadata_coverage}% (corpus antiguo/parcial)"
        result("analisis_calidad.py", status, detail)
    elif ok:
        result("analisis_calidad.py", "PASS", "Completado")
    else:
        print(f"  Output: {(r.stdout or '')[:300]}")
        result("analisis_calidad.py", "FAIL", "Error de ejecucion")
        return False

    # Mostrar resumen de calidad
    for line in (r.stdout or "").split("\n"):
        if any(x in line for x in [
            "BUENA", "LITERAL", "BASURA", "ONOMATOPEYA", "SIN TRADUCIR",
            "GOOD_TRANSLATION", "LITERAL_TRANSLATION", "OCR_GARBAGE",
            "SFX_PRESERVED", "UNTRANSLATED", "REVIEW_LANGUAGE",
        ]):
            print(f"  {line}")
    return True


def step_bandit() -> bool:
    """Paso 3: bandit security audit en archivos Python fuente."""
    step("PASO 3/7 - bandit (security audit)")

    files = list(_PROD_PY_FILES)

    try:
        r = subprocess.run(
            [PYTHON, "-m", "bandit", "-f", "json", *files],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30, cwd=str(PROJECT_ROOT),
        )
    except FileNotFoundError:
        print("  [SKIP] bandit no instalado. Ejecuta: pip install bandit")
        result("bandit", "SKIP", "bandit no instalado")
        return True

    stdout = r.stdout or ""

    # bandit retorna 0 si no hay issues, 1 si hay issues
    if r.returncode == 0:
        print(f"  [OK] Sin vulnerabilidades detectadas")
        result("bandit", "PASS", "0 issues")
        return True

    # bandit encontro issues — parsear JSON
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        print(f"  [WARN] No se pudo parsear JSON de bandit: {e}")
        print(f"  Output: {stdout[:500]}")
        result("bandit", "WARN", "No se pudo parsear, revisar output")
        return True

    results_list = data.get("results", [])
    high_count = 0
    medium_count = 0
    low_count = 0

    for issue in results_list:
        severity = (issue.get("issue_severity") or "").upper()
        fname = issue.get("filename", "")
        line_num = issue.get("line_number", "")
        test_id = issue.get("test_id", "")
        test_name = issue.get("test_name", "")
        print(f"  [{severity}] {fname}:{line_num} - {test_id}:{test_name}")
        if severity == "HIGH":
            high_count += 1
        elif severity == "MEDIUM":
            medium_count += 1
        elif severity == "LOW":
            low_count += 1

    total_issues = high_count + medium_count + low_count
    detail = f"{high_count} HIGH, {medium_count} MEDIUM, {low_count} LOW"

    if high_count > 0:
        result("bandit", "FAIL", detail)
        return False
    elif medium_count > 0:
        result("bandit", "WARN", detail)
        return True
    elif total_issues == 0:
        result("bandit", "WARN", "Issues sin severidad detectable")
        return True
    else:
        result("bandit", "PASS", detail)
        return True


def step_stress_test() -> bool:
    """Paso 7 (opcional): stress_test_memory.py (50 paginas)."""
    step("PASO 7/7 - stress_test_memory.py (50 paginas)")

    pdf_path = os.path.join(
        PROJECT_ROOT,
        "Capítulo 43 de Cómo criar villanos correctamente.pdf"
    )
    if not os.path.exists(pdf_path):
        print("  [SKIP] PDF de prueba no encontrado")
        result("stress_test_memory.py", "SKIP", "PDF no disponible")
        return True

    r = run_python(["stress_test_memory.py"], timeout=600)
    if r is None:
        result("stress_test_memory.py", "FAIL", "Timeout")
        return False

    ok = r.returncode == 0

    # Intentar parsear JSON estructurado (__stress_result__)
    json_result = None
    for line in (r.stdout or "").split("\n"):
        line = line.strip()
        if line.startswith('{"__stress_result__'):
            try:
                json_result = json.loads(line)
            except json.JSONDecodeError:
                pass
            break

    if json_result is not None and "success" in json_result and "errors" in json_result:
        global _STRESS_RESULT
        _STRESS_RESULT = json_result
        # Parseo JSON estructurado
        success = json_result.get("success", 0)
        errors = json_result.get("errors", 0)
        total = json_result.get("total", 0)
        avg_time = json_result.get("avg_time_s", 0.0)
        mem_growth = json_result.get("mem_growth_mb", 0.0)
        leak = json_result.get("leak_detected", False)

        print(f"  {success}/{total} exitosas, {errors} errores")
        print(f"  Tiempo promedio: {avg_time:.1f}s | Crecimiento memoria: {mem_growth:+.1f}MB")

        if errors > 0:
            result("stress_test_memory.py", "FAIL", f"{errors} errores en {total} paginas")
        elif leak:
            result("stress_test_memory.py", "WARN", f"Memoria crecio {mem_growth:+.1f}MB (posible leak)")
        elif success == total:
            result("stress_test_memory.py", "PASS", f"{success}/{total} paginas, {avg_time:.1f}s promedio")
        else:
            result("stress_test_memory.py", "PASS", f"{success}/{total} paginas")
    else:
        # Fallback: parseo por cadenas de texto (legacy)
        found_status = False
        errors_found = 0
        for line in (r.stdout or "").split("\n"):
            if "Sin errores" in line:
                found_status = True
                result("stress_test_memory.py", "PASS", "50/50 paginas OK")
                print(f"  {line}")
                return True
            if "errores detectados" in line:
                found_status = True
                m = re.search(r"(\d+) errores", line)
                if m:
                    errors_found = int(m.group(1))
                break
            if "FAIL" in line or "Traceback" in line or "Error" in line:
                found_status = True
                errors_found = max(errors_found, 1)

        if errors_found:
            result("stress_test_memory.py", "WARN", f"{errors_found} errores")
        elif not found_status:
            result("stress_test_memory.py", "WARN", "No se pudo determinar el resultado")
        elif ok:
            result("stress_test_memory.py", "PASS", "Completado")
        else:
            result("stress_test_memory.py", "FAIL", "Error de ejecucion")

    for line in (r.stdout or "").split("\n")[-5:]:
        print(f"  {line}")
    return True


# --- Main --------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CI local para Traductor Visual Pro",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  %(prog)s                    Tests rapidos (~2 min)
  %(prog)s --full             Tests completos (~12 min, incluye stress)
  %(prog)s --server           Solo inicia servidor + health check
  %(prog)s --skip-syntax      Omitir syntax check
  %(prog)s --skip-mypy        Omitir type check mypy
  %(prog)s --skip-pytest      Omitir tests unitarios (rapido)
  %(prog)s --skip-server      Omitir paso del servidor
  %(prog)s --strict-classification  Sin clasificar = error (exit 1)
  %(prog)s --report           Generar reporte HTML con graficos
  %(prog)s --report --full    CI completo + reporte HTML"""
    )
    parser.add_argument("--full", action="store_true",
                        help="Incluye stress test (50 paginas, ~10 min)")
    parser.add_argument("--server", action="store_true",
                        help="Solo ejecuta paso del servidor")
    parser.add_argument("--skip-syntax", action="store_true",
                        help="Omitir syntax check")
    parser.add_argument("--skip-mypy", action="store_true",
                        help="Omitir type check mypy")
    parser.add_argument("--skip-pytest", action="store_true",
                        help="Omitir tests unitarios pytest")
    parser.add_argument("--skip-server", action="store_true",
                        help="Omitir paso del servidor (util en CI sin deps pesadas)")
    parser.add_argument("--skip-bandit", action="store_true",
                        help="Omitir audit de seguridad bandit")
    parser.add_argument("--skip-cov", action="store_true",
                        help="Omitir reporte de cobertura (mas rapido)")
    parser.add_argument("--strict-classification", action="store_true",
                        help="Archivos sin clasificar = ERROR (exit 1), aunque"
                             " no se esté en GitHub Actions")
    parser.add_argument("--report", action="store_true",
                        help="Generar reporte HTML del CI")
    args = parser.parse_args()

    start_time = time.time()
    all_passed = True

    # Archivos .py nuevos que el walk auto-incluye pero nadie clasificó
    # (ni dev ni producción conocida): se auditan, pero se avisa para que
    # la clasificación sea explícita. En GitHub Actions (env CI=true) un
    # archivo sin clasificar FALLA el job; en local solo es un [WARN] salvo
    # que se pase --strict-classification (exigencia total en cualquier
    # entorno).
    if not _check_unclassified_files(
            _strict_classification_enabled(args.strict_classification)):
        all_passed = False

    if args.server:
        # Modo smoke: solo health + endpoints, sin gate de cobertura (no hay
        # baseline de pytest para combinar).
        step_server(with_coverage=False)
        print_summary(start_time)
        _export_results_json(start_time)
        if args.report:
            _generate_report()
        return 0

    if not args.skip_syntax:
        if not step_syntax():
            all_passed = False

    if not args.skip_mypy:
        if not step_mypy():
            all_passed = False
    else:
        step("PASO 2/7 - mypy strict [OMITIDO]")
        result("mypy strict", "SKIP", "Omitido por --skip-mypy")

    if not args.skip_bandit:
        if not step_bandit():
            all_passed = False
    else:
        step("PASO 3/7 - bandit [OMITIDO]")
        result("bandit", "SKIP", "Omitido por --skip-bandit")

    if not args.skip_pytest:
        if not step_pytest(with_coverage=not args.skip_cov):
            all_passed = False
    else:
        step("PASO 4/7 - pytest [OMITIDO]")
        print("  Usa --skip-pytest para omitir tests unitarios")
        result("pytest (suite completa)", "SKIP", "Omitido por --skip-pytest")

    if not args.skip_server:
        # Con cobertura salvo que se pida omitirla (--skip-cov) o que se corra
        # --full (el stress test necesita el servidor vivo; además, bajo
        # tracing el stress sería lentísimo).
        if not step_server(
                with_coverage=not args.skip_cov and not args.full):
            all_passed = False
    else:
        step("PASO 5/7 - Servidor Flask + tests [OMITIDO]")
        print("  Usa --skip-server para CI sin dependencias pesadas")
        result("Servidor Flask", "SKIP", "Omitido por --skip-server")

    if not step_analisis_calidad():
        all_passed = False

    if args.full:
        # Stress test (el server ya esta corriendo de step_server)
        if not step_stress_test():
            all_passed = False
    else:
        step("PASO 7/7 - stress_test_memory.py [OMITIDO]")
        print("  Usa --full para incluir stress test (50 pags, ~10 min)")
        result("stress_test_memory.py", "SKIP", "Usa --full para ejecutar")

    # Detener servidor
    _stop_server()

    print_summary(start_time)

    # Exportar resultados a JSON para reportes
    _export_results_json(start_time)

    # Generar reporte HTML si se solicito
    if args.report:
        _generate_report()

    return 0 if all_passed else 1


def print_summary(start_time: float) -> None:
    duration = time.time() - start_time
    mins = int(duration // 60)
    secs = int(duration % 60)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = sum(1 for r in RESULTS if r["status"] == "FAIL")
    total = len(RESULTS)

    print()
    print("=" * 70)
    print("  RESUMEN CI")
    print("=" * 70)
    print(f"  Duracion: {mins}m {secs}s")
    print(f"  Resultados: {passed}/{total} pasaron")
    print()
    for r in RESULTS:
        icon = {"PASS": "[OK]", "FAIL": "[FAIL]", "SKIP": "[SKIP]", "WARN": "[WARN]"}.get(
            r["status"], "[?]"
        )
        print(f"  {icon} {r['name']} - {r['detail']}")
    print()

    if failed:
        print("  [FAIL] CI FAILED - Revisa los errores arriba")
    elif any(r["status"] == "WARN" for r in RESULTS):
        print("  [WARN] CI PASSED WITH WARNINGS")
    else:
        print("  [OK] CI PASSED - Todos los tests OK")


if __name__ == "__main__":
    sys.exit(main())
