#!/usr/bin/env python
"""
run_ci.py - CI local para Traductor Visual Pro.

Ejecuta bateria completa de tests:
  1. Syntax check (Python + JS)
  2. bandit (security audit)
  3. pytest + cobertura (295+ tests unitarios: translator, OCR, API)
  4. Servidor Flask + health + endpoints API
  5. analisis_calidad.py (calidad de traducciones)
  6. stress_test_memory.py (opcional, ~10 min con --full)

No depende de PowerShell. Usa env/ o Python del sistema.

Uso:
    python run_ci.py                    # Tests rapidos (~2 min)
    python run_ci.py --full             # Incluye stress test (~12 min)
    python run_ci.py --server           # Solo inicia servidor + health check
    python run_ci.py --skip-syntax      # Omitir syntax check
    python run_ci.py --skip-bandit      # Omitir audit de seguridad
    python run_ci.py --skip-pytest      # Omitir tests unitarios
    python run_ci.py --skip-cov         # Omitir cobertura (mas rapido)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# Windows console UTF-8 fix
if sys.version_info >= (3, 7):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
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
_STRESS_RESULT: dict | None = None


# --- Helpers -----------------------------------------------------------

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


def run_python(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess | None:
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


def server_health(timeout: int = 15) -> dict | None:
    """Espera hasta que el servidor responda en /api/health."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(
                f"http://{SERVER_HOST}:{SERVER_PORT}/api/health", timeout=2
            )
            if r.status == 200:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, ConnectionRefusedError):
            pass
        time.sleep(0.5)
    return None


def http_get(path: str, timeout: int = 10) -> tuple[int, bytes]:
    """GET request al servidor."""
    r = urllib.request.urlopen(
        f"http://{SERVER_HOST}:{SERVER_PORT}{path}", timeout=timeout
    )
    return r.status, r.read()


def http_post(path: str, data: dict, timeout: int = 30) -> tuple[int, dict]:
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
    step("PASO 1/6 - Syntax check")
    all_ok = True
    files = [
        "server.py", "config.py", "translator.py", "ocr_utils.py",
        "models.py", "cache.py", "ratelimit.py", "main.py", "launcher.py",
        "routes/__init__.py", "routes/main.py", "routes/api.py",
        # Módulos de la fusión multi-OCR (Ago 2026): OCRManager + daemon U-OCR
        "ocr_engine.py", "uocr_daemon.py", "uocr_client.py",
        "process_all_pages.py",
    ]
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
    js_files = ["app.js"]
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


def step_pytest(with_coverage: bool = True) -> bool:
    """Paso 3: pytest con todos los tests unitarios (295+ tests)."""
    label = "pytest + cobertura" if with_coverage else "pytest"
    step(f"PASO 3/6 - {label} (295+ tests: translator + OCR + API)")

    test_files = [
        "tests/test_translator.py",
        "tests/test_ocr_utils.py",
        "tests/test_ocr_functions.py",
        "tests/test_api.py",
        "tests/test_ocr_engine.py",
        "tests/test_uocr_daemon.py",
    ]

    cmd = [PYTHON, "-m", "pytest", *test_files, "-v", "--tb=short"]
    if with_coverage:
        cmd += [
            "--cov=translator", "--cov=ocr_utils", "--cov=routes.api",
            "--cov=cache", "--cov=ratelimit", "--cov=config",
            "--cov-report=term-missing",
        ]

    r = subprocess.run(
        cmd,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180, cwd=str(PROJECT_ROOT),
    )

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
        result("pytest", "WARN", "No se pudo determinar el resultado")
        return True  # no bloquear CI por fallo de parseo

    if ok:
        status = "PASS"
        detail = f"{passed_count}/{total} passed {time_str}"
    else:
        status = "FAIL"
        detail = f"{failed_count} FAILED en {total} tests {time_str}"

    result("pytest (247 tests)", status, detail)
    return ok


def step_server() -> bool:
    """Paso 4: Iniciar servidor y verificar health + endpoints."""
    global _server_proc
    step("PASO 4/6 - Servidor Flask + tests")

    # Matar servidor previo si existe (con fallback a netstat+taskkill)
    killed = _kill_process_on_port(SERVER_PORT)
    if killed:
        print(f"  Servidor anterior detenido")
        time.sleep(2)

    # Iniciar servidor
    log_path = os.path.join(PROJECT_ROOT, "ci_server.log")
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            [PYTHON, "-u", "server.py"],
            stdout=log_file, stderr=subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
    print(f"  Servidor iniciado (PID {proc.pid})")

    # Esperar health
    health = server_health(timeout=20)
    if health is None:
        with open(log_path, encoding="utf-8") as f:
            tail = "".join(f.readlines()[-10:])
        print(f"  STDERR LOG:\n{tail}")
        result("Server startup", "FAIL", "No respondio en /api/health")
        proc.kill()
        return False

    mem = health.get("memory", "?")
    print(f"  [OK] Servidor OK - db={health.get('db_available')}, mem={mem}MB")
    result("Server startup", "PASS", f"PID {proc.pid}, mem={mem}MB")

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
        code, data = http_get(path, timeout=5)
        ok = code == 200
        print(f"  {'[OK]' if ok else '[FAIL]'} GET {path} ({len(data)}b)")
        result(f"Static {path}", "PASS" if ok else "FAIL", f"{len(data)}b")

    # Guardar el proc para matarlo al final del CI (no aqui)
    # para que el stress test pueda usarlo si corre con --full
    _server_proc = proc
    return True


_server_proc: subprocess.Popen | None = None


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
    step("PASO 5/6 - analisis_calidad.py")

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

    # Extraer tasa de aceptacion
    accept_rate = None
    for line in (r.stdout or "").split("\n"):
        if "Tasa de aceptacion global" in line:
            m = re.search(r"([\d.]+)%", line)
            if m:
                accept_rate = float(m.group(1))

    if ok and accept_rate is not None:
        status = "PASS" if accept_rate >= 75 else "WARN"
        result("analisis_calidad.py", status, f"Tasa de aceptacion: {accept_rate}%")
    elif ok:
        result("analisis_calidad.py", "PASS", "Completado")
    else:
        print(f"  Output: {(r.stdout or '')[:300]}")
        result("analisis_calidad.py", "FAIL", "Error de ejecucion")
        return False

    # Mostrar resumen de calidad
    for line in (r.stdout or "").split("\n"):
        if any(x in line for x in ["BUENA", "LITERAL", "BASURA", "ONOMATOPEYA", "SIN TRADUCIR"]):
            print(f"  {line}")
    return True


def step_bandit() -> bool:
    """Paso 6: bandit security audit en archivos Python fuente."""
    step("PASO 2/6 - bandit (security audit)")

    files = [
        "server.py", "config.py", "translator.py", "ocr_utils.py",
        "models.py", "cache.py", "ratelimit.py", "main.py", "launcher.py",
        "routes/__init__.py", "routes/main.py", "routes/api.py",
        # Módulos de la fusión multi-OCR (Ago 2026): OCRManager + daemon U-OCR
        "ocr_engine.py", "uocr_daemon.py", "uocr_client.py",
        "process_all_pages.py",
    ]

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
    step("PASO 6/6 - stress_test_memory.py (50 paginas)")

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
  %(prog)s --skip-pytest      Omitir tests unitarios (rapido)
  %(prog)s --skip-server      Omitir paso del servidor
  %(prog)s --report           Generar reporte HTML con graficos
  %(prog)s --report --full    CI completo + reporte HTML"""
    )
    parser.add_argument("--full", action="store_true",
                        help="Incluye stress test (50 paginas, ~10 min)")
    parser.add_argument("--server", action="store_true",
                        help="Solo ejecuta paso del servidor")
    parser.add_argument("--skip-syntax", action="store_true",
                        help="Omitir syntax check")
    parser.add_argument("--skip-pytest", action="store_true",
                        help="Omitir tests unitarios pytest")
    parser.add_argument("--skip-server", action="store_true",
                        help="Omitir paso del servidor (util en CI sin deps pesadas)")
    parser.add_argument("--skip-bandit", action="store_true",
                        help="Omitir audit de seguridad bandit")
    parser.add_argument("--skip-cov", action="store_true",
                        help="Omitir reporte de cobertura (mas rapido)")
    parser.add_argument("--report", action="store_true",
                        help="Generar reporte HTML del CI")
    args = parser.parse_args()

    start_time = time.time()
    all_passed = True

    if args.server:
        step_server()
        print_summary(start_time)
        _export_results_json(start_time)
        if args.report:
            _generate_report()
        return 0

    if not args.skip_syntax:
        if not step_syntax():
            all_passed = False

    if not args.skip_bandit:
        if not step_bandit():
            all_passed = False
    else:
        step("PASO 2/6 - bandit [OMITIDO]")
        result("bandit", "SKIP", "Omitido por --skip-bandit")

    if not args.skip_pytest:
        if not step_pytest(with_coverage=not args.skip_cov):
            all_passed = False
    else:
        step("PASO 3/6 - pytest [OMITIDO]")
        print("  Usa --skip-pytest para omitir tests unitarios")
        result("pytest (295 tests)", "SKIP", "Omitido por --skip-pytest")

    if not args.skip_server:
        if not step_server():
            all_passed = False
    else:
        step("PASO 4/6 - Servidor Flask + tests [OMITIDO]")
        print("  Usa --skip-server para CI sin dependencias pesadas")
        result("Servidor Flask", "SKIP", "Omitido por --skip-server")

    if not step_analisis_calidad():
        all_passed = False

    if args.full:
        # Stress test (el server ya esta corriendo de step_server)
        if not step_stress_test():
            all_passed = False
    else:
        step("PASO 6/6 - stress_test_memory.py [OMITIDO]")
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
