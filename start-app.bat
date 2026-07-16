@echo off
setlocal
set "ROOT=%~dp0"
set "PY=%ROOT%env\Scripts\python.exe"
set SKIP_MIT_INIT=1

if not exist "%PY%" (
  echo No se encontro el entorno virtual en "%ROOT%env".
  echo Ejecuta la instalacion de dependencias primero ^(o revisa el nombre de la carpeta del entorno^).
  pause
  exit /b 1
)

echo Iniciando Traductor Visual Pro...
start "" "%PY%" "%ROOT%server.py"

:waitloop
timeout /t 1 /nobreak >nul
netstat -ano | findstr "LISTENING.*5174" >nul 2>&1
if errorlevel 1 goto waitloop

rem Servidor listo, abrir navegador
set "CHROME=%ProgramFiles%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"
if not exist "%CHROME%" set "CHROME=%LocalAppData%\Google\Chrome\Application\chrome.exe"

if exist "%CHROME%" (
  start "" "%CHROME%" --app="http://127.0.0.1:5174" --window-size=1400,900
) else (
  start "" "http://127.0.0.1:5174"
)