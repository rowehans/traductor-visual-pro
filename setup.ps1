<#
.SYNOPSIS
    Script de instalacion/configuracion para Traductor Visual Pro.
#>

param(
    [string]$InstallDir = "",
    [switch]$SkipModels,
    [switch]$OnlyExe
)

$ErrorActionPreference = "Stop"

if (-not $InstallDir) {
    $InstallDir = Split-Path -Parent $MyInvocation.MyCommand.Path
}
$InstallDir = (Resolve-Path $InstallDir).Path
$PythonPath = Join-Path $InstallDir "env\Scripts\python.exe"
$ExePath = Join-Path $InstallDir "main.exe"
$Requirements = Join-Path $InstallDir "requirements.txt"
$LogFile = Join-Path $InstallDir "install.log"

function Write-Log {
    param([string]$Message, [string]$ForegroundColor = "White")
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $Message"
    Write-Host $line -ForegroundColor $ForegroundColor
    Add-Content -Path $LogFile -Value $line
}

Write-Host "Traductor Visual Pro - Instalacion" -ForegroundColor Cyan
Write-Host "Directorio: $InstallDir"
Write-Host ""

# ─── Verificar .exe ─────────────────────────────────────────
if (Test-Path $ExePath) {
    $size = (Get-Item $ExePath).Length / 1MB
    $sizeRound = [math]::Round($size, 1)
    Write-Log "main.exe encontrado: ${sizeRound} MB"
} else {
    Write-Log "main.exe NO encontrado"
}

if ($OnlyExe) {
    Write-Log "Modo OnlyExe. Listo."
    exit 0
}

# ─── Crear entorno virtual ──────────────────────────────────
if (Test-Path $PythonPath) {
    Write-Log "Entorno virtual ya existe"
} else {
    Write-Log "Buscando Python 3..."
    $py = $null
    $candidates = @(
        "${env:ProgramFiles}\Python313\python.exe",
        "${env:ProgramFiles}\Python312\python.exe",
        "${env:ProgramFiles}\Python311\python.exe",
        "${env:LocalAppData}\Programs\Python\Python313\python.exe",
        "${env:LocalAppData}\Programs\Python\Python312\python.exe",
        "${env:LocalAppData}\Programs\Python\Python311\python.exe",
        "python.exe",
        "python3.exe"
    )
    foreach ($c in $candidates) {
        try {
            if (-not (Get-Command $c -ErrorAction SilentlyContinue)) { continue }
            $v = & $c --version 2>&1
            if ($v -match "Python 3") {
                $py = $c
                Write-Log "  Python encontrado: $v"
                break
            }
        } catch {
            Write-Log "  Intentando $c ... no disponible"
        }
    }
    
    if (-not $py) {
        Write-Log ""
        Write-Log "ERROR: No se encontro Python 3 instalado." -ForegroundColor Red
        Write-Log "Este programa requiere Python 3.11 o superior."
        Write-Log "Descargalo desde: https://www.python.org/downloads/"
        Write-Log ""
        Write-Log "Rutas buscadas (excluyendo PATH):"
        foreach ($c in $candidates) {
            if ($c -match "^[a-zA-Z]:") {
                # Path absoluto: verificar con Test-Path
                if (Test-Path $c) { Write-Log "  [EXISTE] $c" }
                else { Write-Log "  [NO]    $c" }
            } else {
                # Comando en PATH: verificar con Get-Command
                if (Get-Command $c -ErrorAction SilentlyContinue) {
                    $resolved = (Get-Command $c).Source
                    Write-Log "  [EXISTE] $c -> $resolved"
                } else { Write-Log "  [NO]    $c" }
            }
        }
        Write-Log ""
        Write-Log "Despues de instalar Python, ejecuta este script nuevamente."
        exit 1
    }
    
    & $py -m venv (Join-Path $InstallDir "env")
    if ($LASTEXITCODE -ne 0) {
        Write-Log "Error creando venv con: $py"
        Write-Log "Verifica que Python 3 este correctamente instalado."
        exit 1
    }
    Write-Log "Entorno virtual creado"
}

# ─── Instalar dependencias ──────────────────────────────────
Write-Log "Instalando dependencias..."
try {
    & $PythonPath -m pip install --upgrade pip --quiet
} catch { Write-Log "pip upgrade: $_" }

if (Test-Path $Requirements) {
    & $PythonPath -m pip install -r $Requirements --quiet
} else {
    & $PythonPath -m pip install flask opencv-python torch easyocr --quiet
}
Write-Log "Dependencias instaladas"

# ─── Verificar modulos ──────────────────────────────────────
Write-Log "Verificando modulos..."
$mods = @("flask", "cv2", "torch", "easyocr", "numpy")
foreach ($m in $mods) {
    $r = & $PythonPath -c "import $m" 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Log "  $m OK"
    } else {
        Write-Log "  $m ERROR"
    }
}

# ─── Descargar CT2 ──────────────────────────────────────────
if (-not $SkipModels) {
    $sentinel = Join-Path $InstallDir "models\ct2\es-en\.ct2_conversion_ok"
    if (-not (Test-Path $sentinel)) {
        Write-Log "Descargando modelo CT2 es-en..."
        $guid = [guid]::NewGuid().ToString("N").Substring(0, 8)
        $pyFile = Join-Path $env:TEMP "ct2_dl_${guid}.py"
        $dirPy = $InstallDir -replace "\\", "/"

        $pyCode = @"
import sys
sys.path.insert(0, '$dirPy')
from translator import _get_ct2_translator
t, tk = _get_ct2_translator('es', 'en')
print('CT2 listo: ' + str(t is not None))
"@
        $pyCode | Out-File -FilePath $pyFile -Encoding ASCII
        $r = & $PythonPath $pyFile 2>&1
        Remove-Item $pyFile -Force
        foreach ($line in $r) { Write-Log "  $line" }
    } else {
        Write-Log "Modelo CT2 ya descargado"
    }
}

# ─── Resumen ────────────────────────────────────────────────
Write-Host ""
Write-Host "Instalacion completada" -ForegroundColor Green
Write-Host ""
Write-Host "Para iniciar:" -ForegroundColor Yellow
if (Test-Path $ExePath) {
    Write-Host "  $ExePath"
}
Write-Host "  cd $InstallDir"
Write-Host "  .\env\Scripts\python.exe server.py"
Write-Host ""
Write-Host "Log: $LogFile" -ForegroundColor Gray
