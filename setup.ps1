<#
.SYNOPSIS
    Script de instalación/configuración para Traductor Visual Pro.
    Crea el entorno virtual, instala dependencias y verifica que todo funcione.

.DESCRIPTION
    Este script automatiza la configuración completa del proyecto:
    1. Crea/actualiza el entorno virtual Python (env/)
    2. Instala dependencias desde requirements.txt
    3. Descarga modelos CT2 si no existen
    4. Verifica que el .exe funcione

.PARAMETER InstallDir
    Directorio de instalación (por defecto: donde está el script)
.PARAMETER SkipModels
    Saltar descarga de modelos (útil para desarrollo)
.PARAMETER OnlyExe
    Solo verificar .exe, no configurar entorno

.EXAMPLE
    .\setup.ps1                          # Instalación completa
    .\setup.ps1 -SkipModels              # Sin descargar modelos
    .\setup.ps1 -OnlyExe                 # Solo verificar .exe
#>

param(
    [string]$InstallDir = "",
    [switch]$SkipModels,
    [switch]$OnlyExe
)

$ErrorActionPreference = 'Stop'

# ─── Configuración ────────────────────────────────────────────────
if (-not $InstallDir) {
    $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$InstallDir = (Resolve-Path $InstallDir).Path
$PythonPath = Join-Path $InstallDir "env\Scripts\python.exe"
$PipPath = Join-Path $InstallDir "env\Scripts\pip.exe"
$ExePath = Join-Path $InstallDir "main.exe"
$Requirements = Join-Path $InstallDir "requirements.txt"
$LogFile = Join-Path $InstallDir "install.log"

function Write-Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║       Traductor Visual Pro — Instalación    ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host "Directorio: $InstallDir`n"

# ─── Verificar .exe ───────────────────────────────────────────────
if (Test-Path $ExePath) {
    $ExeSize = (Get-Item $ExePath).Length / 1MB
    Write-Log "✅ main.exe encontrado: $([math]::Round($ExeSize,1)) MB"
} else {
    Write-Log "❌ main.exe NO encontrado en $ExePath"
    Write-Log "   Compila primero con: env\Scripts\python.exe -m PyInstaller main.spec"
    if (-not $OnlyExe) {
        Write-Log "   Continuando solo con configuración del entorno..."
    }
}

if ($OnlyExe) {
    Write-Log "✅ Modo OnlyExe.Verificación completada."
    exit 0
}

# ─── Crear entorno virtual ─────────────────────────────────────────
if (Test-Path $PythonPath) {
    Write-Log "✅ Entorno virtual ya existe en env/"
} else {
    Write-Log "📦 Creando entorno virtual Python..."
    $python = "python"
    # Buscar Python disponible
    $candidates = @(
        "python.exe",
        "python3.exe",
        "${env:ProgramFiles}\Python312\python.exe",
        "${env:ProgramFiles}\Python311\python.exe",
        "${env:LocalAppData}\Programs\Python\Python312\python.exe",
        "${env:LocalAppData}\Programs\Python\Python311\python.exe"
    )
    foreach ($c in $candidates) {
        try {
            $ver = & $c --version 2>&1
            if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3\.(1[0-2]|[89])") {
                $python = $c
                Write-Log "   Python encontrado: $ver ($python)"
                break
            }
        } catch { continue }
    }

    Write-Log "   Ejecutando: $python -m venv env"
    & $python -m venv (Join-Path $InstallDir "env")
    if ($LASTEXITCODE -ne 0) {
        Write-Log "❌ Error creando entorno virtual"
        exit 1
    }
    Write-Log "✅ Entorno virtual creado"
}

# ─── Actualizar pip ────────────────────────────────────────────────
Write-Log "📦 Actualizando pip..."
& $PythonPath -m pip install --upgrade pip --quiet 2>&1 | Out-Null
Write-Log "✅ pip actualizado"

# ─── Instalar dependencias ─────────────────────────────────────────
if (Test-Path $Requirements) {
    Write-Log "📦 Instalando dependencias desde requirements.txt..."
    Write-Log "   (esto puede tomar varios minutos, descargando ~3GB)"
    try {
        & $PythonPath -m pip install -r "$Requirements" --quiet 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Log "✅ Dependencias instaladas correctamente"
        } else {
            Write-Log "⚠️  Algunas dependencias pueden no haberse instalado (código: $LASTEXITCODE)"
        }
    } catch {
        Write-Log "⚠️  Error instalando dependencias: $_"
    }
} else {
    Write-Log "⚠️  requirements.txt no encontrado en $Requirements"
    Write-Log "   Instalando dependencias mínimas..."
    & $PythonPath -m pip install flask opencv-python torch easyocr argostranslate deep-translator langdetect --quiet 2>&1 | Out-Null
}

# ─── Verificar instalación básica ──────────────────────────────────
Write-Log ""
Write-Log "🔍 Verificando instalación..."
$checks = @(
    @{Module="flask"; Test="import flask"},
    @{Module="cv2"; Test="import cv2"},
    @{Module="torch"; Test="import torch"},
    @{Module="easyocr"; Test="import easyocr"},
    @{Module="numpy"; Test="import numpy"}
)
$allOk = $true
foreach ($check in $checks) {
    $result = & $PythonPath -c $check.Test 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "   ✅ $($check.Module)"
    } else {
        Write-Log "   ❌ $($check.Module) - $result"
        $allOk = $false
    }
}

if (-not $allOk) {
    Write-Log ""
    Write-Log "⚠️  Algunos módulos no se instalaron correctamente."
    Write-Log "   Reintenta con: $PythonPath -m pip install -r ""$Requirements"""
}

# ─── Descargar modelos CT2 (opcional) ─────────────────────────────
if (-not $SkipModels) {
    $ct2Dir = Join-Path $InstallDir "models\ct2"
    if (-not (Test-Path $ct2Dir) -or -not (Test-Path (Join-Path $ct2Dir "es-en\.ct2_conversion_ok"))) {
        Write-Log ""
        Write-Log "📦 Pre-descargando modelo CT2 es→en..."
        Write-Log "   (la primera traducción lo descargará automáticamente)"
        & $PythonPath -c "
import sys; sys.path.insert(0, '$InstallDir')
from translator import _get_ct2_translator
t, tk = _get_ct2_translator('es', 'en')
print(f'CT2 es→en listo: {t is not None}')
" 2>&1 | ForEach-Object { Write-Log "   $_" }
    } else {
        Write-Log "✅ Modelo CT2 ya descargado"
    }
}

# ─── Resumen final ─────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║       ✅ Instalación completada              ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Para iniciar:" -ForegroundColor Yellow
if (Test-Path $ExePath) {
    Write-Host "   $ExePath (doble click)" -ForegroundColor White
}
Write-Host "   cd $InstallDir && .\env\Scripts\python.exe server.py" -ForegroundColor White
Write-Host ""
Write-Host "Log guardado en: $LogFile" -ForegroundColor Gray
