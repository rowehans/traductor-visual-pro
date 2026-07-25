"""
main.py — Punto de entrada para el ejecutable (.exe).

Modo launcher: arranca instantáneamente, lanza el servidor Flask en el
mismo proceso (ocultando consola) y abre Chrome cuando el puerto esté listo.
Modo servidor (--server): solo inicia el servidor, útil para depuración.

NO usa waitress — waitress tiene problemas con catch-all + blueprints en Flask 3.x.
"""
import os
import socket
import subprocess
import sys
import threading
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

PORT = 5174
HOST = "127.0.0.1"


def _fix_cwd() -> None:
    """
    En modo frozen (PyInstaller), el CWD debe ser el directorio del exe,
    no _MEIPASS. También busca 'env/Lib/site-packages' subiendo directorios
    desde el exe hasta la raíz del proyecto.
    """
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        os.chdir(exe_dir)
        if hasattr(sys, "_MEIPASS"):
            sys.path.insert(0, sys._MEIPASS)

        # Buscar env/ subiendo directorios desde exe_dir
        current = exe_dir
        for _ in range(10):  # Máximo 10 niveles hacia arriba
            candidate = os.path.join(current, "env", "Lib", "site-packages")
            if os.path.exists(candidate):
                if candidate not in sys.path:
                    sys.path.append(candidate)
                    print(f"[cwd] Enlazado env desde: {candidate}")
                return
            parent = os.path.dirname(current)
            if parent == current:  # Llegamos a la raíz del disco
                break
            current = parent
        print(f"[cwd] env no encontrado. exe_dir={exe_dir}")


def _hide_console() -> None:
    """Oculta la ventana de consola en Windows (modo frozen)."""
    if getattr(sys, "frozen", False) and sys.platform == "win32":
        try:
            import ctypes
            hwnd = ctypes.windll.kernel32.GetConsoleWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 0)  # SW_HIDE
        except Exception:
            pass


def wait_for_port(timeout: int = 60) -> bool:
    """Espera hasta que el servidor esté escuchando en HOST:PORT."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((HOST, PORT), timeout=1):
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.4)
    return False


def open_browser() -> None:
    """Abre Chrome en modo app; si no hay Chrome, usa el navegador por defecto."""
    url = f"http://{HOST}:{PORT}"
    chrome_paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles%\Microsoft\Edge\Application\msedge.exe"),
    ]
    browser = next((p for p in chrome_paths if os.path.exists(p)), None)
    if browser:
        try:
            subprocess.Popen([browser, f"--app={url}", "--window-size=1400,900"])
            return
        except Exception:
            pass
    import webbrowser
    webbrowser.open(url)


def run_server() -> None:
    """
    Importa y ejecuta server.py con Flask dev server.
    NO usa waitress (causa 404 con catch-all + blueprints en Flask 3.x).
    """
    _fix_cwd()

    from server import app

    # Flask dev server — confiable para desarrollo local
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)


def run_launcher() -> None:
    """
    Modo launcher (ejecución normal del usuario).
    Inicia el servidor Flask en el mismo proceso (ocultando consola).
    """
    _fix_cwd()
    _hide_console()

    # Abrir navegador en background mientras el servidor arranca
    def open_browser_delayed() -> None:
        if wait_for_port(timeout=90):
            open_browser()
        else:
            open_browser()

    threading.Thread(target=open_browser_delayed, daemon=True).start()

    # Ejecutar servidor (bloquea aquí)
    run_server()


if __name__ == "__main__":
    if "--server" in sys.argv:
        run_server()
    else:
        run_launcher()
