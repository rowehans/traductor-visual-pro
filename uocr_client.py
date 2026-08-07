"""
uocr_client.py — Cliente del daemon Unlimited-OCR (para el servidor Flask env/).

El daemon (uocr_daemon.py) corre en env_uocr_gpu (torch cu126 + bitsandbytes),
que NO puede importarse dentro del proceso del servidor (conflicto de DLLs
CUDA con EasyOCR/CT2 de env/). Este módulo lo lanza como subproceso en
background al arrancar y le habla por HTTP en 127.0.0.1:5177.

Funciones:
  spawn_daemon()        -> lanza el daemon si no está corriendo (no bloquea)
  is_daemon_running()   -> True si el subproceso sigue vivo
  health()              -> dict con estado del daemon (sin excepción)
  wait_ready(timeout_s) -> True si el modelo quedó listo antes del timeout
  process_page(image_path, max_length) -> dict {text, infer_s, blocks}

Dependencias: solo stdlib (urllib/subprocess) — seguro para importar en env/.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

def _resolve_root() -> Path:
    """Resuelve la raíz del proyecto (donde viven env_uocr_gpu/ y hf_cache/).

    En modo desarrollo, __file__ apunta a la raíz del proyecto.
    En modo frozen (.exe), __file__ apunta a _MEIPASS (dist/main/_internal),
    donde NO existe env_uocr_gpu — la raíz real es el directorio del exe
    (dist/main/) o un ancestro que contenga env_uocr_gpu (mismo patrón que
    _fix_cwd() en main.py).
    """
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        # Estrategia 1: subir desde el directorio del exe buscando env_uocr_gpu
        current = exe_dir
        for _ in range(8):
            if (current / "env_uocr_gpu" / "Scripts" / "python.exe").exists():
                return current
            parent = current.parent
            if parent == current:
                break
            current = parent
        # Estrategia 2: ubicaciones fijas conocidas del proyecto
        known = [Path(r"D:\crear traductor"), exe_dir.parent, exe_dir.parent.parent]
        for cand in known:
            if (cand / "env_uocr_gpu" / "Scripts" / "python.exe").exists():
                return cand
        return exe_dir
    return Path(__file__).resolve().parent


ROOT = _resolve_root()
DAEMON_PORT = 5177
DAEMON_URL = f"http://127.0.0.1:{DAEMON_PORT}"
UOCR_PYTHON = ROOT / "env_uocr_gpu" / "Scripts" / "python.exe"
DAEMON_SCRIPT = ROOT / "uocr_daemon.py"
LOG_FILE = ROOT / "uocr_daemon.log"

_proc: subprocess.Popen | None = None


def available() -> bool:
    """True si el entorno del daemon (venv GPU + script) existe en D:."""
    return UOCR_PYTHON.exists() and DAEMON_SCRIPT.exists()


def spawn_daemon() -> bool:
    """Lanza el daemon en background si no está ya corriendo.

    No bloquea: el daemon carga el modelo (~8 min) en su propio proceso
    mientras el servidor sigue arrancando. Retorna True si quedó lanzado
    (o ya estaba vivo).
    """
    global _proc
    if _proc is not None and _proc.poll() is None:
        return True
    # Adoptar un daemon ya vivo en el puerto (quedó de una sesión previa,
    # arrancado a mano, etc.) — evita lanzar un proceso duplicado.
    h = health()
    if h.get("state") in ("loading", "ready"):
        print("[uocr] Daemon ya activo en el puerto — adoptado")
        return True
    if h.get("state") == "error":
        # Modelo en estado error: intentar relanzar una vez (el daemon viejo
        # se quedó con la carga fallida; matarlo y arrancar uno limpio).
        print(f"[uocr] Daemon en estado error ({h.get('error')}) — relanzando...")
        _kill_daemon_on_port()
        time.sleep(1)
    if not available():
        print("[uocr] Daemon no disponible (env_uocr_gpu o uocr_daemon.py ausentes)")
        return False
    try:
        log = open(LOG_FILE, "a", encoding="utf-8")
        env = dict(os.environ)
        env["HF_HOME"] = str(ROOT / "hf_cache")
        env["TRANSFORMERS_CACHE"] = str(ROOT / "hf_cache" / "hub")
        env["HF_HUB_CACHE"] = str(ROOT / "hf_cache" / "hub")
        env["PYTHONUNBUFFERED"] = "1"
        _proc = subprocess.Popen(
            [str(UOCR_PYTHON), str(DAEMON_SCRIPT), "--port", str(DAEMON_PORT)],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        print(f"[uocr] Daemon Unlimited-OCR lanzado (PID {_proc.pid}), modelo cargando en background...")
        # Verificación post-lanzamiento: dar 10s para que el daemon arranque
        # el servidor HTTP (el modelo tarda ~8 min, pero /health responde al
        # instante). Si el puerto estaba ocupado por otro proceso, el daemon
        # muere y lo detectamos aquí.
        for _ in range(10):
            time.sleep(1)
            st = health().get("state")
            if st in ("loading", "ready"):
                return True
            if _proc.poll() is not None:
                print(f"[uocr] El daemon murió al arrancar (exit {_proc.returncode})")
                return False
        print("[uocr] Daemon lanzado pero /health no responde aún")
        return True
    except Exception as e:
        print(f"[uocr] Error lanzando daemon: {e}")
        return False


def _kill_daemon_on_port() -> None:
    """Mata cualquier proceso escuchando en el puerto del daemon (Windows).

    Usa netstat + taskkill — evita que un daemon zombi en estado error
    siga ocupando el puerto y la VRAM.
    """
    global _proc
    try:
        import subprocess as sp
        out = sp.run(["netstat", "-ano"], capture_output=True, text=True, timeout=10)
        pids = set()
        for line in out.stdout.splitlines():
            if f":{DAEMON_PORT}" in line and "LISTENING" in line:
                parts = line.split()
                if parts and parts[-1].isdigit():
                    pids.add(parts[-1])
        for pid in pids:
            sp.run(["taskkill", "/F", "/PID", pid], capture_output=True, timeout=10)
            print(f"[uocr] Daemon zombi PID {pid} terminado")
    except Exception as e:
        print(f"[uocr] No se pudo limpiar el puerto {DAEMON_PORT}: {e}")
    finally:
        _proc = None


def is_daemon_running() -> bool:
    return _proc is not None and _proc.poll() is None


def _request(method: str, path: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        DAEMON_URL + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health() -> dict:
    """Estado del daemon: {"state": "offline"|"loading"|"ready"|"error", ...}.

    Nunca lanza excepción (el servidor no debe caerse por el daemon).
    """
    try:
        return _request("GET", "/health", timeout=2.0)
    except Exception:
        return {"state": "offline", "error": "daemon no responde"}


def wait_ready(timeout_s: float = 900.0) -> bool:
    """Espera hasta que el modelo esté listo. True si listo antes del timeout."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        h = health()
        if h.get("state") == "ready":
            return True
        if h.get("state") == "error":
            return False
        time.sleep(5)
    return False


def process_page(image_path: str, max_length: int = 4096,
                 wait_timeout_s: float = 900.0) -> dict:
    """Envía una imagen al daemon y devuelve {text, infer_s, blocks}.

    Si el modelo aún carga, espera hasta wait_timeout_s antes de fallar.
    """
    if not wait_ready(wait_timeout_s):
        return {"error": "modelo no listo", "status": health()}
    try:
        return _request(
            "POST", "/ocr",
            {"image_path": str(image_path), "max_length": max_length},
            timeout=1800.0,  # la inferencia 4-bit puede tardar ~1-5 min
        )
    except Exception as e:
        return {"error": f"error de comunicación con el daemon: {e}"}


def process_batch(image_paths: list[str], max_length: int = 4096,
                  wait_timeout_s: float = 900.0) -> dict:
    """Envía VARIAS páginas al daemon en una sola inferencia VLM (Fase 1).

    Usa POST /ocr-batch del daemon, que ejecuta _model.infer_multi() — las
    N imágenes comparten el prefill del modelo, amortizando el costo por
    página. Devuelve {pages: [{blocks, recovered_from_art}, ...], infer_s}.

    Args:
        image_paths: lista de rutas a PNG (máx 4 por request, límite VRAM).
        max_length: longitud máxima de generación (compartida por el batch).
        wait_timeout_s: espera máxima si el modelo aún carga.

    Returns:
        dict con "pages" (una entrada por imagen en el mismo orden) o
        {"error": ...} si el daemon no está listo o falla la comunicación.
    """
    if not image_paths:
        return {"error": "lista de imágenes vacía", "pages": []}
    if not wait_ready(wait_timeout_s):
        return {"error": "modelo no listo", "status": health()}
    try:
        return _request(
            "POST", "/ocr-batch",
            {"images": [str(p) for p in image_paths],
             "max_length": max_length},
            # N páginas 4-bit en una sola generación: hasta ~5 min cada una
            timeout=1800.0 * max(1, min(len(image_paths), 4)),
        )
    except Exception as e:
        return {"error": f"error de comunicación con el daemon: {e}", "pages": []}


if __name__ == "__main__":
    # CLI de diagnóstico: env\\Scripts\\python.exe uocr_client.py [image.png]
    print("Daemon disponible:", available())
    print("Health:", health())
    if len(sys.argv) > 1:
        import os
        r = process_page(os.path.abspath(sys.argv[1]))
        print("infer_s:", r.get("infer_s"), "| bloques:", len(r.get("blocks", [])))
        print("texto:", (r.get("text") or "")[:200])
