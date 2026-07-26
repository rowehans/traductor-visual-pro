<#
.SYNOPSIS
    Crea un release en GitHub usando el .exe ya compilado LOCALMENTE (con GPU/CUDA).
    Alternativa al CI build que no tiene GPU.

.DESCRIPTION
    Los runners de GitHub Actions no tienen CUDA, por lo que compilar el .exe en CI
    produce un binario CPU-only (~5s/pág vs 0.88s/pág con GPU). Este script usa los
    assets ya compilados localmente con soporte GPU completo.

    Requisitos:
    - gh CLI instalado y autenticado (gh auth login)
    - dist/main/main.exe existente (PyInstaller)
    - dist/installer/TraductorVisual_Setup_*.exe (opcional, Inno Setup)

.PARAMETER TagVersion
    Etiqueta del release (ej: v0.1.40). Opcional: si no se provee, genera
    una automáticamente basada en el último tag + patch.

.PARAMETER Force
    Si se especifica, sobrescribe el tag si ya existe y permite crear release
    aunque haya cambios sin commitear.

.PARAMETER Push
    Si se especifica, hace git push del tag automáticamente.

.EXAMPLE
    .\create_release.ps1 -TagVersion v0.1.40 -Push
    Crea release v0.1.40 con los assets actuales y pushea el tag.

.EXAMPLE
    .\create_release.ps1 -Force
    Genera tag automático, sobrescribe si existe, ignora dirty repo.
#>

param(
    [string]$TagVersion = "",
    [switch]$Force = $false,
    [switch]$Push = $false
)

$ErrorActionPreference = "Stop"

# ─── Configuración ─────────────────────────────────────────────
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExePath = Join-Path $ProjectRoot "dist\main\main.exe"
$InstallerDir = Join-Path $ProjectRoot "dist\installer"
$InstallerPattern = "TraductorVisual_Setup_*.exe"

# ─── Guard: repo limpio ────────────────────────────────────────
$status = git -C $ProjectRoot status --porcelain
if ($status) {
    Write-Host "`n⚠️  Hay cambios sin commitear:" -ForegroundColor Yellow
    $status | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    if (-not $Force) {
        Write-Host "  Usa -Force para ignorar, o commitea primero. Abortando." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "  -Force activo: continuando..." -ForegroundColor DarkYellow
}

# ─── Paso 1: Determinar versión ────────────────────────────────
if (-not $TagVersion) {
    Write-Host "Generando versión automática..." -ForegroundColor Cyan
    $lastTag = git -C $ProjectRoot tag --sort=-version:refname | Select-Object -First 1
    if (-not $lastTag -or $lastTag -notmatch '^v\d+\.\d+\.\d+$') {
        $lastTag = "v0.0.0"
    }
    $parts = $lastTag.Substring(1) -split '\.'
    $patch = [int]$parts[2] + 1
    $TagVersion = "v$($parts[0]).$($parts[1]).$patch"
    Write-Host "  Último tag: $lastTag → Nuevo: $TagVersion" -ForegroundColor Yellow
}

Write-Host "`n=== CREANDO RELEASE $TagVersion ===" -ForegroundColor Green
Write-Host "Proyecto: $ProjectRoot"

# ─── Paso 2: Verificar assets ──────────────────────────────────
if (-not (Test-Path $ExePath)) {
    Write-Host "ERROR: .exe no encontrado en $ExePath" -ForegroundColor Red
    Write-Host "Ejecuta: python -m PyInstaller main.spec --clean --noconfirm" -ForegroundColor Yellow
    exit 1
}
$exeSize = (Get-Item $ExePath).Length / 1MB
Write-Host ".exe: $ExePath ($([math]::Round($exeSize, 1)) MB) ✅" -ForegroundColor Green

$installer = Get-ChildItem -Path $InstallerDir -Filter $InstallerPattern | Select-Object -First 1
if (-not $installer) {
    Write-Host "AVISO: Instalador no encontrado — solo se incluirá el .exe" -ForegroundColor Yellow
} else {
    $instSize = $installer.Length / 1MB
    Write-Host "Instalador: $($installer.FullName) ($([math]::Round($instSize, 1)) MB) ✅" -ForegroundColor Green
}

# ─── Paso 3: Verificar gh CLI ──────────────────────────────────
try {
    $null = gh --version
} catch {
    Write-Host "ERROR: gh CLI no instalado. Descarga: https://cli.github.com/" -ForegroundColor Red
    exit 1
}

$ghStatus = gh auth status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️  gh CLI no autenticado. Ejecuta:" -ForegroundColor Yellow
    Write-Host "    gh auth login" -ForegroundColor Cyan
    Write-Host "  Luego reintenta." -ForegroundColor Yellow
    exit 1
}
Write-Host "gh CLI: autenticado ✅" -ForegroundColor Green

# ─── Paso 4: Crear tag local ───────────────────────────────────
$existingTag = git -C $ProjectRoot tag -l $TagVersion
if ($existingTag) {
    if (-not $Force) {
        Write-Host "ERROR: Tag $TagVersion ya existe. Usa -Force para sobrescribir." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "Tag $TagVersion ya existe. Sobrescribiendo (-Force)..."
    git -C $ProjectRoot tag -d $TagVersion
}
git -C $ProjectRoot tag $TagVersion
Write-Host "Tag local creado: $TagVersion ✅" -ForegroundColor Green

# ─── Paso 5: Push tag ──────────────────────────────────────────
if ($Push) {
    Write-Host "Pusheando tag $TagVersion..." -ForegroundColor Cyan
    $pushArgs = @("push", "origin", $TagVersion)
    if ($Force) { $pushArgs += "--force" }
    & git -C $ProjectRoot $pushArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: No se pudo pushear el tag." -ForegroundColor Red
        exit 1
    }
    Write-Host "Tag pusheado: origin/$TagVersion ✅" -ForegroundColor Green
}

# ─── Paso 6: Generar release notes ─────────────────────────────
$releaseNotes = @"
# Traductor Visual Pro $TagVersion

Release generado localmente con GPU/CUDA.

## Assets
- **main.exe**: $([math]::Round($exeSize,1)) MB (compilado con GPU support)
$(if ($installer) { "- **Instalador**: $([math]::Round($instSize,1)) MB (Inno Setup)" })

## Commits recientes
$(git -C $ProjectRoot log --oneline --no-decorate -20 | ForEach-Object { "    $_" })
"@

$notesFile = Join-Path $env:TEMP "release_notes_$TagVersion.md"
$releaseNotes | Out-File -FilePath $notesFile -Encoding utf8
Write-Host "Release notes generadas ✅" -ForegroundColor Green

# ─── Paso 7: Crear release ────────────────────────────────────
Write-Host "`nCreando release $TagVersion en GitHub..." -ForegroundColor Cyan

$assets = @("""$ExePath""")
if ($installer) {
    $assets += """$($installer.FullName)"""
}
$assetStr = $assets -join " "

$cmd = "gh release create ""$TagVersion"" $assetStr --title ""Traductor Visual Pro $TagVersion"" --notes-file ""$notesFile"""
Write-Host "Ejecutando: gh release create..."
Invoke-Expression $cmd

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✅ Release $TagVersion creado exitosamente!" -ForegroundColor Green
    Write-Host "   https://github.com/rowehans/traductor-visual-pro/releases/tag/$TagVersion" -ForegroundColor Cyan
} else {
    Write-Host "`n❌ Error creando release. Revisa el mensaje arriba." -ForegroundColor Red
    exit 1
}

# Limpiar
Remove-Item $notesFile -Force -ErrorAction SilentlyContinue
