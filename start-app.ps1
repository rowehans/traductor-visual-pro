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

# ─── Limpiar daemon Unlimited-OCR zombi (puerto 5177) ──────────
# El daemon (uocr_daemon.py) es un subproceso persistente que mantiene el
# modelo 4-bit en VRAM. Si quedó de una sesión anterior (2.25 GB VRAM),
# matarlo aquí para liberar la GPU antes de arrancar el servidor.
$daemonPort = 5177
Write-Host "Verificando puerto del daemon U-OCR $daemonPort..."
$dconns = netstat -ano | Select-String ":$daemonPort\s" | Select-String "LISTENING"
foreach ($conn in $dconns) {
  $parts = $conn -split '\s+'
  $dpid = $parts[-1]
  if ($dpid -and $dpid -ne "0") {
    Write-Host "Matando daemon U-OCR zombie PID=$dpid en puerto $daemonPort..."
    taskkill -f -pid $dpid 2>$null
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