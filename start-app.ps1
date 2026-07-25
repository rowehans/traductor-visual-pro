$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root "env\Scripts\python.exe"
$server = Join-Path $root "server.py"
$port = 5174

if (-not (Test-Path $python)) {
  Write-Host "No se encontro el entorno Python. Ejecuta la instalacion de dependencias primero."
  exit 1
}

Set-Location $root
$env:SKIP_MIT_INIT = "1"

# ─── Limpiar procesos zombie en el puerto ──────────────────────
Write-Host "Verificando puerto $port..."
$connections = netstat -ano | Select-String ":$port\s" | Select-String "LISTENING"
foreach ($conn in $connections) {
  $parts = $conn -split '\s+'
  $pid = $parts[-1]
  if ($pid -and $pid -ne "0") {
    Write-Host "Matando proceso zombie PID=$pid en puerto $port..."
    taskkill -f -pid $pid 2>$null
    # Pequena pausa para que el SO libere el puerto
    Start-Sleep -Milliseconds 300
  }
}

# ─── Iniciar servidor ──────────────────────────────────────────
Write-Host "Iniciando servidor..."
$proc = Start-Process -FilePath $python -ArgumentList $server -PassThru -NoNewWindow

# ─── Esperar a que el servidor esté listo (hasta 15s) ─────────
Write-Host "Esperando al servidor en http://127.0.0.1:$port ..."
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Milliseconds 500
  try {
    $req = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if ($req.StatusCode -eq 200) {
      Write-Host "Servidor listo!"
      Start-Process "http://127.0.0.1:$port/"
      exit 0
    }
  } catch {
    # Todavia no responde
  }
}

Write-Host "El servidor no respondio a tiempo. Abriendo de todas formas..."
Start-Process "http://127.0.0.1:$port/"