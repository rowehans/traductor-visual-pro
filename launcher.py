"""
launcher.py — Lanzador de Traductor Visual Pro.
Inicia el servidor, espera a que esté listo y abre el navegador.
Sin necesidad de .bat, doble click y funciona.
"""
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PYTHON = ROOT / "env" / "Scripts" / "python.exe"
SERVER = ROOT / "server.py"
PORT = 5174
HOST = "127.0.0.1"


def port_open(host: str, port: int) -> bool:
    """Verifica si el puerto está escuchando."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


def main() -> int:
    os.environ["SKIP_MIT_INIT"] = "1"
    os.chdir(str(ROOT))

    if not PYTHON.exists():
        print(f"[ERROR] No se encuentra Python en {PYTHON}")
        input("Presiona Enter para salir...")
        return 1

    print("=== Traductor Visual Pro ===")
    print(f"Iniciando servidor en http://{HOST}:{PORT}")
    print("Espera unos segundos...")

    proc = subprocess.Popen(
        [str(PYTHON), "-u", str(SERVER)],
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
