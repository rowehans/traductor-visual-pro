"""
main.py — Punto de entrada para el ejecutable (.exe).
Levanta server.py en background y abre Chrome en modo app.
Sin ventana de consola cuando se compila con --noconsole.
"""
import os
import socket
import subprocess
import sys
import threading
import time

os.environ["SKIP_MIT_INIT"] = "1"


def wait_for_port(port, host="127.0.0.1"):
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                break
        time.sleep(0.5)


def launch_chrome():
    wait_for_port(5174)
    paths = [
        os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
    ]
    chrome_path = next((p for p in paths if os.path.exists(p)), None)
    if chrome_path:
        subprocess.Popen([chrome_path, '--app=http://127.0.0.1:5174', '--window-size=1400,900'])
    else:
        import webbrowser
        webbrowser.open('http://127.0.0.1:5174')


if __name__ == '__main__':
    # Ajustar path para PyInstaller frozen mode
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        sys.path.insert(0, sys._MEIPASS)
        os.chdir(sys._MEIPASS)

    threading.Thread(target=launch_chrome, daemon=True).start()
    from server import app
    app.run(host="127.0.0.1", port=5174, debug=False, use_reloader=False)