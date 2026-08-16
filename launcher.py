"""
launcher.py — Lanzador de Traductor Visual Pro.
Inicia el servidor, espera a que esté listo y abre el navegador.
Sin necesidad de .bat, doble click y funciona.

Uso:
  python launcher.py            # modo normal (GPU si CUDA disponible)
  python launcher.py --cpu      # MODO_CPU: sin GPU dedicada — VLM apagado,
                                # YOLO en CPU, escala de render reducida
                                # (config.MODO_CPU se activa vía la env
                                # UOCR_MODO_CPU=1 que se inyecta al servidor).
"""
import argparse
import os
import socket
import subprocess  # nosec B404
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "env" / "Scripts" / "python.exe"
SERVER = ROOT / "server.py"
PORT = 5174
HOST = "127.0.0.1"
# Variable de entorno que activa config.MODO_CPU en el proceso del servidor.
# El launcher la inyecta con --cpu; cualquiera puede ponerla manualmente.
MODO_CPU_ENV = "UOCR_MODO_CPU"


def port_open(host: str, port: int) -> bool:
    """Verifica si el puerto está escuchando."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="launcher.py",
        description="Lanzador de Traductor Visual Pro — inicia el servidor "
                    "y abre el navegador.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Arrancar en MODO_CPU (sin GPU dedicada): VLM apagado, YOLO en "
             "CPU, escala de render reducida (config.MODO_CPU).",
    )
    args = parser.parse_args(argv)

    os.environ["SKIP_MIT_INIT"] = "1"
    os.chdir(str(ROOT))

    if not PYTHON.exists():
        print(f"[ERROR] No se encuentra Python en {PYTHON}")
        input("Presiona Enter para salir...")
        return 1

    if port_open(HOST, PORT):
        print(f"Servidor ya activo en http://{HOST}:{PORT}; reutilizando sesión.")
        if args.cpu:
            print("[MODO_CPU] El servidor ya estaba activo; --cpu no se aplica "
                  "a la sesión en curso (reinícialo con --cpu para el preset).")
        webbrowser.open(f"http://{HOST}:{PORT}")
        return 0

    print("=== Traductor Visual Pro ===")
    if args.cpu:
        print("[MODO_CPU] Arrancando sin GPU dedicada: VLM apagado, YOLO en "
              "CPU, escala de render 0.8.")
    print(f"Iniciando servidor en http://{HOST}:{PORT}")
    print("Espera unos segundos...")

    # B603: seguro — sin shell=True, usa lista de argumentos, path absoluto
    env = dict(os.environ)
    if args.cpu:
        env[MODO_CPU_ENV] = "1"
    proc = subprocess.Popen(  # nosec
        [str(PYTHON), "-u", str(SERVER)],
        env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    # Esperar hasta 30s a que el servidor responda
    for i in range(60):
        time.sleep(0.5)
        if port_open(HOST, PORT):
            print(f"\nServidor listo en http://{HOST}:{PORT}")
            webbrowser.open(f"http://{HOST}:{PORT}")
            break
    else:
        print("\n[WARN] El servidor no respondió a tiempo, abriendo de todas formas...")
        webbrowser.open(f"http://{HOST}:{PORT}")

    # Mantener vivo hasta Ctrl+C
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nServidor detenido.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
