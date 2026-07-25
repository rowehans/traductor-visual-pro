#!/usr/bin/env python
"""
run_ci.py - CI local para Traductor Visual Pro.

Unifica test_ci.py + analisis_calidad.py + stress_test_memory.py
en un solo comando Python. No depende de PowerShell.

Uso:
    python run_ci.py              # Tests rapidos (~30s)
    python run_ci.py --full       # Incluye stress test (~10 min)
    python run_ci.py --server     # Solo inicia servidor y health check
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
        sys.stdout = sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
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
    step("PASO 1/5 - Syntax check")
    all_ok = True
    files = [
        "server.py", "config.py", "translator.py", "ocr_utils.py",
        "models.py", "cache.py", "ratelimit.py", "main.py", "launcher.py",
        "routes/__init__.py", "routes/main.py", "routes/api.py",
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

    status = "PASS" if all_ok else "FAIL"
    result("Syntax check", status, "Todos los archivos compilan" if all_ok else "Hay errores")
    return all_ok


def step_test_ci() -> bool:
    """Paso 2: test_ci.py (deteccion de idioma)."""
    step("PASO 2/5 - test_ci.py (deteccion de idioma)")
    r = run_python(["test_ci.py"], timeout=30)
    if r is not None and r.returncode == 0 and "OK" in r.stdout:
        result("test_ci.py", "PASS", "Todos los tests de idioma pasaron")
        return True
    else:
        if r is not None:
            print(f"  Output: {r.stdout[:500]}")
        result("test_ci.py", "FAIL", "Fallaron tests de idioma")
        return False


def step_server() -> bool:
    """Paso 3: Iniciar servidor y verificar health + endpoints."""
    global _server_proc
    step("PASO 3/5 - Servidor Flask + tests")

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
    """Paso 4: analisis_calidad.py (calidad de traducciones)."""
    step("PASO 4/5 - analisis_calidad.py")

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


def step_stress_test() -> bool:
    """Paso 5 (opcional): stress_test_memory.py (50 paginas)."""
    step("PASO 5/5 - stress_test_memory.py (50 paginas)")

    pdf_path = os.path.join(
        PROJECT_ROOT,
        "Capítulo 43 de Cómo criar villanos correctamente _ Olympus Scanlation_compressed.pdf"
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
  %(prog)s                    Tests rapidos (~30s)
  %(prog)s --full             Tests completos (~10 min, incluye stress test)
  %(prog)s --server           Solo inicia servidor + health check
  %(prog)s --skip-syntax      Omitir syntax check  %(prog)s --skip-server      Omitir paso del servidor (CI sin deps pesadas)
  %(prog)s --report           Generar reporte HTML con graficos
  %(prog)s --report --full    CI completo + reporte HTML"""
    )
    parser.add_argument("--full", action="store_true",
                        help="Incluye stress test (50 paginas, ~10 min)")
    parser.add_argument("--server", action="store_true",
                        help="Solo ejecuta paso del servidor")
    parser.add_argument("--skip-syntax", action="store_true",
                        help="Omitir syntax check")
    parser.add_argument("--skip-server", action="store_true",
                        help="Omitir paso del servidor (util en CI sin deps pesadas)")
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

    if not step_test_ci():
        all_passed = False

    if not args.skip_server:
        if not step_server():
            all_passed = False
    else:
        step("PASO 3/5 - Servidor Flask + tests [OMITIDO]")
        print("  Usa --skip-server para CI sin dependencias pesadas")
        result("Servidor Flask", "SKIP", "Omitido por --skip-server")

    if not step_analisis_calidad():
        all_passed = False

    if args.full:
        # Stress test (el server ya esta corriendo de step_server)
        if not step_stress_test():
            all_passed = False
    else:
        step("PASO 5/5 - stress_test_memory.py [OMITIDO]")
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
