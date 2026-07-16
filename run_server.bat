@echo off
cd /d "D:\crear traductor"
set PYTHONIOENCODING=utf-8
set SKIP_MIT_INIT=1
echo.
echo === Traductor Visual ===
echo Servidor iniciando en http://127.0.0.1:5174
echo Espera hasta ver "=== SERVIDOR LISTO ==="
echo.
"D:\crear traductor\env\Scripts\python.exe" "D:\crear traductor\server.py"