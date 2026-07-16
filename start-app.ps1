$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "env\Scripts\python.exe"
$server = Join-Path $root "server.py"

if (-not (Test-Path $python)) {
  Write-Host "No se encontro el entorno Python. Ejecuta la instalacion de dependencias primero."
  exit 1
}

Set-Location $root
Write-Host "Iniciando Traductor visual en http://127.0.0.1:5174/"
Start-Process "http://127.0.0.1:5174/"
& $python $server