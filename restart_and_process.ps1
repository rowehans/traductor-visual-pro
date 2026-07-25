# restart_and_process.ps1 — Script todo-en-uno
# Ejecutar con: .\restart_and_process.ps1
# Opcional: .\restart_and_process.ps1 -SkipProcess (solo reiniciar servidor)

param(
    [switch]$SkipProcess,
    [switch]$SkipQuality
)

$ProjectRoot = "D:\crear traductor"
$Python = "$ProjectRoot\env\Scripts\python.exe"
$ServerScript = "$ProjectRoot\server.py"
$ProcessScript = "$ProjectRoot\process_all_pages.py"
$QualityScript = "$ProjectRoot\analisis_calidad.py"
$Checkpoint = "$ProjectRoot\resultados_progreso.json"

# Verificar que el proyecto existe
if (-not (Test-Path $Python)) {
    Write-Host "ERROR: No se encontro Python en $Python" -ForegroundColor Red
    exit 1
}

# Cambiar al directorio del proyecto para que las rutas relativas funcionen
Set-Location $ProjectRoot

Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  REINICIO + PROCESAMIENTO + CALIDAD" -ForegroundColor Cyan
Write-Host "============================================`n" -ForegroundColor Cyan

# ── PASO 1: Matar servidor viejo ──────────────────────────────────────────
Write-Host "[1/5] Deteniendo servidor anterior..." -ForegroundColor Yellow
$connections = Get-NetTCPConnection -LocalPort 5174 -ErrorAction SilentlyContinue
if ($connections) {
    $connections | ForEach-Object { 
        try { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue } catch {
            Write-Host "  Advertencia: no se pudo detener proceso $($_.OwningProcess): $($_.Exception.Message)" -ForegroundColor Yellow
        }
    }
    Start-Sleep -Seconds 2
    Write-Host "  Servidor detenido" -ForegroundColor Green
} else {
    Write-Host "  No habia servidor corriendo en puerto 5174" -ForegroundColor Gray
}

# ── PASO 2: Iniciar servidor nuevo ────────────────────────────────────────
Write-Host "`n[2/5] Iniciando servidor nuevo..." -ForegroundColor Yellow
$proc = Start-Process -FilePath $Python -ArgumentList $ServerScript -PassThru -WorkingDirectory $ProjectRoot
Write-Host "  PID: $($proc.Id)" -ForegroundColor Gray

Write-Host "  Esperando 15s para que cargue EasyOCR..." -ForegroundColor Yellow
$ready = $false
for ($i = 0; $i -lt 6; $i++) {
    Start-Sleep -Seconds 3
    try {
        $health = Invoke-WebRequest -Uri "http://127.0.0.1:5174/api/health" -TimeoutSec 5 -ErrorAction Stop
        $ready = $true
        break
    } catch {}
}

# Verificar que el servidor no crasheo durante startup
if ($proc.HasExited) {
    Write-Host "  ERROR: El servidor se detuvo inesperadamente (exit code: $($proc.ExitCode))" -ForegroundColor Red
    exit 1
}

if ($ready) {
    Write-Host "  Servidor listo!" -ForegroundColor Green
} else {
    Write-Host "  El servidor tarda en responder. Continuando de todos modos..." -ForegroundColor Yellow
}

# ── PASO 3: Respaldar checkpoint anterior ─────────────────────────────────
Write-Host "`n[3/5] Respaldando checkpoint..." -ForegroundColor Yellow
if (Test-Path $Checkpoint) {
    $backup = "$ProjectRoot\resultados_progreso_backup_$(Get-Date -Format 'yyyyMMdd_HHmmss').json"
    Copy-Item $Checkpoint $backup
    Write-Host "  Respaldado como: $(Split-Path $backup -Leaf)" -ForegroundColor Gray
    Remove-Item $Checkpoint
    Write-Host "  Checkpoint eliminado para empezar de cero" -ForegroundColor Gray
} else {
    Write-Host "  No habia checkpoint previo" -ForegroundColor Gray
}

# ── PASO 4: Procesar 128 paginas ─────────────────────────────────────────
if (-not $SkipProcess) {
    Write-Host "`n[4/5] Procesando 128 paginas (~27 min)..." -ForegroundColor Yellow
    & $Python $ProcessScript
    $processExit = $LASTEXITCODE
    if ($processExit -eq 0) {
        Write-Host "  Procesamiento completado!" -ForegroundColor Green
    } else {
        Write-Host "  Algunas paginas pudieron fallar (revisar resultados)" -ForegroundColor Yellow
    }
} else {
    Write-Host "`n[4/5] Saltando procesamiento (-SkipProcess)" -ForegroundColor Gray
}

# ── PASO 5: Analizar calidad ──────────────────────────────────────────────
if (-not $SkipQuality -and (Test-Path $QualityScript)) {
    Write-Host "`n[5/5] Analizando calidad..." -ForegroundColor Yellow
    & $Python $QualityScript
} else {
    Write-Host "`n[5/5] Saltando analisis de calidad" -ForegroundColor Gray
}

# ── RESUMEN ───────────────────────────────────────────────────────────────
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "  COMPLETADO" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  Servidor corriendo en: http://127.0.0.1:5174" -ForegroundColor White
Write-Host "  Abrir en navegador para traducir manualmente`n" -ForegroundColor White
