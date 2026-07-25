<#
.SYNOPSIS
    run_ci.ps1 - Integracion continua local para Traductor Visual Pro.
    Ejecuta: syntax check + test_ci.py + stress_test + analisis_calidad.py

.DESCRIPTION
    Bateria completa de tests para verificar que server.py y routes/api.py
    funcionan correctamente despues de cambios.

    Por defecto ejecuta: syntax check + test_ci.py + analisis_calidad.py.
    Con -Full: tambien ejecuta stress_test_memory.py (50 pags, ~10 min).

    Requiere: Python 3.12+ con dependencias en env/Scripts/python.exe.

.PARAMETER Full
    Incluye stress_test_memory.py (50 paginas, ~10 minutos).

.EXAMPLE
    .\run_ci.ps1          # Tests rapidos (~30s)
    .\run_ci.ps1 -Full    # Tests completos (~11 min)
#>

param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$PythonExe = Join-Path $ProjectRoot "env\Scripts\python.exe"
$ServerPort = 5174
$StartTime = Get-Date
$Results = @()
$RunFullCI = $Full

function Write-Step {
    param([string]$Title)
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
}

function Write-Result {
    param([string]$Test, [string]$Status, [string]$Detail)
    $Results += [PSCustomObject]@{ Test = $Test; Status = $Status; Detail = $Detail }
    $icon = if ($Status -eq "PASS") { "[OK]" } else { "[FAIL]" }
    Write-Host "  $icon $Test - $Detail" -ForegroundColor $(if ($Status -eq "PASS") { "Green" } else { "Red" })
}

try {
    Write-Step "PASO 1/5 - Verificacion de sintaxis Python"
    $syntaxOk = $true
    $files = @("server.py", "routes/api.py", "routes/main.py", "config.py", "translator.py", "ocr_utils.py", "models.py", "cache.py", "ratelimit.py", "main.py", "launcher.py")
    foreach ($file in $files) {
        $path = Join-Path $ProjectRoot $file
        if (-not (Test-Path $path)) { 
            Write-Host "  [?] $file no encontrado, saltando" -ForegroundColor Yellow
            continue 
        }
        $result = & $PythonExe -m py_compile $path 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [OK] $file - sintaxis correcta" -ForegroundColor Green
        } else {
            Write-Host "  [FAIL] $file - ERROR DE SINTAXIS:" -ForegroundColor Red
            Write-Host "     $result" -ForegroundColor Red
            $syntaxOk = $false
        }
    }
    if (-not $syntaxOk) { throw "Errores de sintaxis encontrados" }
    Write-Result "Syntax check" "PASS" "Todos los archivos compilan"

    Write-Step "PASO 2/5 - test_ci.py (deteccion de idioma)"
    $ciOut = & $PythonExe (Join-Path $ProjectRoot "test_ci.py") 2>&1
    if ($LASTEXITCODE -eq 0 -and $ciOut -match "OK") {
        Write-Result "test_ci.py" "PASS" "Todos los tests de idioma pasaron"
    } else {
        Write-Host "  Output: $ciOut" -ForegroundColor Red
        throw "test_ci.py fallo (exit code: $LASTEXITCODE)"
    }
    
    Write-Step "PASO 3/5 - Iniciar servidor Flask"
    $existing = Get-NetTCPConnection -LocalPort $ServerPort -ErrorAction SilentlyContinue
    if ($existing) {
        Stop-Process -Id $existing.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  Servidor anterior detenido (PID $($existing.OwningProcess))"
        Start-Sleep -Seconds 2
    }
    
    $serverLog = Join-Path $ProjectRoot "ci_server.log"
    $serverProc = Start-Process -NoNewWindow -FilePath $PythonExe -ArgumentList (Join-Path $ProjectRoot "server.py") -RedirectStandardOutput $serverLog -RedirectStandardError $serverLog -PassThru
    Write-Host "  Servidor iniciado (PID $($serverProc.Id)), esperando 12s..."
    Start-Sleep -Seconds 12
    
    try {
        $health = Invoke-WebRequest -Uri "http://127.0.0.1:$ServerPort/api/health" -TimeoutSec 10
        $healthData = $health.Content | ConvertFrom-Json
        Write-Host "  Servidor OK - memoria: $($healthData.memory)" -ForegroundColor Green
        Write-Result "Server startup" "PASS" "PID $($serverProc.Id), memoria $($healthData.memory)"
    } catch {
        Write-Host "  STDERR LOG:" -ForegroundColor Yellow
        Get-Content $serverLog -Tail 10 | ForEach-Object { Write-Host "     $_" }
        throw "El servidor no respondio en /api/health"
    }
    
    if ($RunFullCI) {
        Write-Step "PASO 4/5 - stress_test_memory.py (50 paginas)"
        $stressOut = & $PythonExe (Join-Path $ProjectRoot "stress_test_memory.py") 2>&1
        if ($LASTEXITCODE -eq 0) {
            $summaryLine = ($stressOut -split "`n") | Select-String "Sin errores|errores detectados"
            if ($summaryLine -match "Sin errores") {
                Write-Result "stress_test_memory.py" "PASS" "50/50 paginas OK"
            } elseif ($summaryLine -match "(\d+) errores") {
                $errCount = $matches[1]
                Write-Result "stress_test_memory.py" "WARN" "$errCount errores en 50 paginas"
            } else {
                Write-Result "stress_test_memory.py" "PASS" "Completado"
            }
            $stressOut -split "`n" | Select-Object -Last 3 | ForEach-Object { Write-Host "  $_" }
        } else {
            Write-Host "  Output: $stressOut" -ForegroundColor Red
            throw "stress_test_memory.py fallo (exit code: $LASTEXITCODE)"
        }
    } else {
        Write-Step "PASO 4/5 - stress_test_memory.py [OMITIDO]"
        Write-Host "  Usa -Full para incluir stress test (50 pags, ~10 min)" -ForegroundColor Yellow
        Write-Result "stress_test_memory.py" "SKIP" "Usa -Full para ejecutar"
    }
    
    Write-Step "PASO 5/5 - analisis_calidad.py"
    $qualityOut = & $PythonExe (Join-Path $ProjectRoot "analisis_calidad.py") 2>&1
    if ($LASTEXITCODE -eq 0) {
        $acceptLine = ($qualityOut -split "`n") | Select-String "aceptacion global"
        if ($acceptLine -match "([\d.]+)%") {
            $rate = [float]$matches[1]
            $status = if ($rate -ge 75) { "PASS" } else { "WARN" }
            Write-Result "analisis_calidad.py" $status "Tasa de aceptacion: $rate%"
        } else {
            Write-Result "analisis_calidad.py" "PASS" "Completado"
        }
        $qualityOut -split "`n" | Select-String "BUENA|LITERAL|BASURA|ONOMATOPEYA|SIN TRADUCIR" | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "  Output: $qualityOut" -ForegroundColor Red
        Write-Result "analisis_calidad.py" "FAIL" "Error de ejecucion"
    }

    Write-Step "RESUMEN CI"
    $duration = (Get-Date) - $StartTime
    $allPass = ($Results | Where-Object { $_.Status -eq "PASS" }).Count
    $total = $Results.Count
    Write-Host ""
    Write-Host "  Duracion: $($duration.Minutes)m $($duration.Seconds)s"
    Write-Host "  Resultados: $allPass/$total pasaron"
    Write-Host ""
    $Results | Format-Table -AutoSize | Out-String | ForEach-Object { Write-Host $_ }
    
    if ($Results | Where-Object { $_.Status -eq "FAIL" }) {
        Write-Host "  [FAIL] CI FAILED - Revisa los errores arriba" -ForegroundColor Red
        exit 1
    } elseif ($Results | Where-Object { $_.Status -eq "WARN" }) {
        Write-Host "  [?] CI PASSED WITH WARNINGS" -ForegroundColor Yellow
        exit 0
    } else {
        Write-Host "  [OK] CI PASSED - Todos los tests OK" -ForegroundColor Green
        exit 0
    }

} catch {
    Write-Host ""
    Write-Host "=" * 70 -ForegroundColor Red
    Write-Host "  CI FALLO" -ForegroundColor Red
    Write-Host "=" * 70 -ForegroundColor Red
    Write-Host "  Error: $_" -ForegroundColor Red
    Write-Host ""
    exit 1
} finally {
    $srv = Get-NetTCPConnection -LocalPort $ServerPort -ErrorAction SilentlyContinue
    if ($srv) {
        Stop-Process -Id $srv.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  Servidor detenido (PID $($srv.OwningProcess))" -ForegroundColor Gray
    }
}
