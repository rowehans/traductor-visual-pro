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

# ─── Reutilizar una sesión sana sin matar PIDs por puerto ───────
# El servidor y el daemon escuchan en loopback. Un puerto ocupado no prueba
# que el proceso sea nuestro: nunca terminamos procesos globalmente. Si ya
# hay un servidor Traductor Visual Pro sano, simplemente abrimos la UI; el
# propio uocr_client adopta un daemon U-OCR compatible que ya esté vivo.
function Test-ServerReady {
  try {
    $response = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" `
      -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    return $response.StatusCode -eq 200
  } catch {
    return $false
  }
}

if (Test-ServerReady) {
  Write-Host "Servidor ya activo en http://127.0.0.1:$port; reutilizando sesión."
  Start-Process "http://127.0.0.1:$port/"
  exit 0
}

# ─── Iniciar servidor ──────────────────────────────────────────
Write-Host "Iniciando servidor..."
$proc = Start-Process -FilePath $python -ArgumentList $server -PassThru -NoNewWindow

# ─── Esperar a que el servidor esté listo (hasta 15s) ─────────
Write-Host "Esperando al servidor en http://127.0.0.1:$port ..."
for ($i = 0; $i -lt 30; $i++) {
  Start-Sleep -Milliseconds 500
  if ($proc.HasExited) {
    Write-Host "El servidor terminó durante el arranque (código $($proc.ExitCode))."
    Write-Host "No se modificó el proceso que ocupaba el puerto $port."
    exit 1
  }
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
